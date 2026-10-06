"""Precise global beat-grid estimation from the beat head's activations.

Why this exists: the beat head predicts beat/downbeat activations on the
encoder's 4-frame (~46.4 ms) grid. Raw peak picking + median(diff) is far too
imprecise for barline anchoring — each peak carries +-23 ms bin-quantization
noise, a single-peak anchor offsets the whole chart by 1-3 TJA slots, and a
0.1% period error drifts ~200 ms over a three-minute song. That is exactly the
"notes are not on the barline" failure mode.

Fix: a TJA needs one rigid grid anyway (constant BPM + one OFFSET), so we
estimate (period, phase) GLOBALLY from every peak in the song:

  1. activation curves over OVERLAPPING windows (hop = WINDOW/2), using BOTH
     channels (ch0 = beat, ch1 = downbeat) — ~5x more evidence than
     downbeats-only;
  2. sub-bin peak times via parabolic interpolation (46 ms bins -> ~5-10 ms);
  3. robust periodic regression t_k ~= phase + k*period with integer-index
     assignment and outlier rejection, initialized at bar level and refined at
     beat level (precision grows ~1/(N*span): 100+ bars pins BPM to <0.01);
  4. integer / half-integer BPM snap, accepted only when the residual does not
     degrade (community charts are almost always integer BPM);
  5. bar phase chosen by downbeat-activation voting over the meter offsets.

Residual statistics are returned so callers can fall back (no barline
anchoring) when the song does not have a constant tempo.
"""

import numpy as np
import torch

from .vocab import FPS, WINDOW

BIN = 4.0 / FPS  # activation bin length in seconds (~46.4 ms)


@torch.no_grad()
def beat_activations(model, mel, device="cuda", hop=WINDOW // 2):
    """Averaged beat/downbeat activation curves over overlapping windows.

    mel: (n_mels, T). Returns (acc, bin_s): acc is (2, ceil(T/factor)) with
    row 0 = beat, row 1 = downbeat; bin_s is the bin length in seconds.
    The head resolution is auto-detected: the classic linear head runs at
    encoder rate (4-frame bins, ~46 ms), the hi-res head at frame rate
    (~11.6 ms bins).
    """
    from .generate import _autocast

    if isinstance(mel, torch.Tensor):
        mel = mel.numpy()
    T = mel.shape[1]
    L = WINDOW // 4
    acc = cnt = None
    factor = 4
    for st in range(0, max(T - hop, 1), hop):
        w = torch.from_numpy(mel[:, st : st + WINDOW].astype(np.float32))
        if w.shape[1] < WINDOW:
            w = torch.nn.functional.pad(w, (0, WINDOW - w.shape[1]),
                                        value=float(np.log(1e-5)))
        if w.shape[0] < model.in_ch:  # dual/slot models: blank grid channels
            w = torch.cat([w, torch.full((model.in_ch - w.shape[0], WINDOW), -1.0)])
        with _autocast(device):
            mem = model.encode(w[None].to(device))
            pr = torch.sigmoid(model.beat(mem[:, -L:]).float())[0].cpu().numpy().T
        if acc is None:
            factor = WINDOW // pr.shape[1]
            nbin = (T + factor - 1) // factor
            acc = np.zeros((2, nbin))
            cnt = np.zeros(nbin)
        b0 = st // factor
        n = min(pr.shape[1], nbin - b0)  # skip bins past the real song end
        if n > 0:
            acc[:, b0 : b0 + n] += pr[:, :n]
            cnt[b0 : b0 + n] += 1
    if acc is None:
        return np.zeros((2, 1)), factor / FPS
    return acc / np.maximum(cnt, 1), factor / FPS


def _refined_peaks(act, height, min_dist_s, bin_s=BIN):
    """Peak times (seconds) with sub-bin parabolic interpolation."""
    from scipy.signal import find_peaks

    idx, _ = find_peaks(act, height=height, distance=max(2, int(min_dist_s / bin_s)))
    out = []
    for i in idx:
        dx = 0.0
        if 0 < i < len(act) - 1:
            d = act[i - 1] - 2 * act[i] + act[i + 1]
            if d < -1e-9:
                dx = float(np.clip(0.5 * (act[i - 1] - act[i + 1]) / d, -0.5, 0.5))
        # +0.5: training targets bin FLOOR(t*FPS/div), so a peak at bin i
        # means the event sits at the bin CENTER — without this everything
        # reads one half-bin early
        out.append((i + dx + 0.5) * bin_s)
    return np.asarray(out)


def _fit_periodic(t, period, phase, reject=0.16, iters=6):
    """Robust regression t_i ~= phase + k_i*period with unknown integer k_i.
    Returns (period, phase, inlier_mask, rms_seconds)."""
    t = np.asarray(t, float)
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        k = np.round((t - phase) / period)
        r = t - (phase + k * period)
        keep = np.abs(r) < reject * period
        if keep.sum() < 4:
            return period, phase, keep, float("inf")
        kk, tt = k[keep], t[keep]
        km = kk.mean()
        denom = float(((kk - km) ** 2).sum())
        if denom < 1e-9:
            break
        period = float(((kk - km) * (tt - tt.mean())).sum() / denom)
        phase = float(tt.mean() - period * km)
    r = t - (phase + np.round((t - phase) / period) * period)
    rms = float(np.sqrt(np.mean(r[keep] ** 2)))
    return period, phase, keep, rms


def _refit_phase(t, period, phase, reject=0.22, iters=3):
    """Phase-only refit for a FIXED period (used by the BPM-snap step)."""
    t = np.asarray(t, float)
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        k = np.round((t - phase) / period)
        r = t - (phase + k * period)
        keep = np.abs(r) < reject * period
        if keep.sum() < 4:
            return phase, keep, float("inf")
        phase += float(np.median(r[keep]))
    r = t - (phase + np.round((t - phase) / period) * period)
    rms = float(np.sqrt(np.mean(r[keep] ** 2)))
    return phase, keep, rms


@torch.no_grad()
def fit_grid(model, mel, device="cuda", meter=None):
    """Estimate a rigid (BPM, barline phase) grid for the whole song.

    Returns a dict with bpm / beat / bar / phase (a barline time), synthesized
    downbeats+beats covering the song, and fit diagnostics (inlier_frac,
    rms_ms, ok). Returns None when there are not enough peaks to fit.
    """
    if isinstance(mel, torch.Tensor):
        mel_np = mel.numpy()
    else:
        mel_np = mel
    act, bin_s = beat_activations(model, mel_np, device)
    dur = mel_np.shape[1] / FPS
    t_d = _refined_peaks(act[1], height=0.35, min_dist_s=0.9, bin_s=bin_s)
    t_b = _refined_peaks(act[0], height=0.35, min_dist_s=0.22, bin_s=bin_s)
    if len(t_d) < 4:
        return None

    # 1) bar-level robust fit (several phase inits; first peak may be spurious)
    p0 = float(np.median(np.diff(t_d)))
    fits = [_fit_periodic(t_d, p0, float(ph)) for ph in t_d[:3]]
    p_bar0, phi_d, keep_d, rms_d = min(fits, key=lambda f: f[3])

    # 2) meter from the bar/beat period ratio; taiko is almost always 4/4
    if meter is None:
        meter = 4
        if len(t_b) >= 8:
            m = p_bar0 / max(float(np.median(np.diff(t_b))), 1e-6)
            if abs(m - 3) < 0.25:
                meter = 3

    # 3) beat-level refinement with ALL peaks (downbeats are beats too)
    t_all = np.concatenate([t_b, t_d]) if len(t_b) else t_d
    p_beat, phi_beat, keep_a, rms_a = _fit_periodic(
        t_all, p_bar0 / meter, phi_d, reject=0.22)
    if not np.isfinite(rms_a):
        p_beat, phi_beat, rms_a = p_bar0 / meter, phi_d, rms_d

    # 4) integer / half-integer BPM snap + octave normalization: prefer the
    #    common-range notation (a 120 song fit at 60 plays identically but
    #    notates ugly), accept only if the residual holds up
    bpm = 60.0 / p_beat
    cands = [round(bpm), round(bpm * 2) / 2]
    if bpm < 80:   # octave normalization: prefer common-range notation
        cands = [round(bpm * 2), round(bpm * 2) * 1.0] + cands
    elif bpm > 210:
        cands = [round(bpm / 2)] + cands
    # rank: in [80, 210] first, integers before halves, keep insertion order
    ranked = sorted(dict.fromkeys(cands),
                    key=lambda c: (not (80 <= c <= 210), abs(c - round(c)) > 0,
                                   cands.index(c)))
    best = (rms_a, bpm, p_beat, phi_beat)
    for cand in ranked:
        if not 40 <= cand <= 320 or not (0.4 <= cand / max(bpm, 1e-6) <= 2.5):
            continue
        pb = 60.0 / cand
        ph, _, rms = _refit_phase(t_all, pb, phi_beat)
        if rms <= best[0] + 0.0015:  # snap unless it costs >1.5 ms rms
            best = (rms, float(cand), pb, ph)
            break
    rms_a, bpm, p_beat, phi_beat = best
    p_bar = meter * p_beat

    # 5) bar phase: among the `meter` beat offsets, pick the one whose barline
    #    comb collects the most downbeat activation
    nb = act.shape[1]

    def _comb_score(ph):
        ts = np.arange(ph if ph >= 0 else ph + p_bar, dur, p_bar)
        ix = np.clip(np.round(ts / bin_s).astype(int), 0, nb - 1)
        return float(act[1][ix].sum())

    base = phi_beat + np.round((phi_d - phi_beat) / p_beat) * p_beat
    offs = [(base + j * p_beat - p_bar * np.floor((base + j * p_beat) / p_bar))
            for j in range(meter)]
    phase = max(offs, key=_comb_score)

    k_max = int(np.floor((dur - phase) / p_bar)) + 1
    downbeats = phase + np.arange(0, max(k_max, 1)) * p_bar
    downbeats = downbeats[(downbeats >= 0) & (downbeats < dur)]
    beats = phase + np.arange(0, int((dur - phase) / p_beat) + 1) * p_beat
    beats = beats[(beats >= 0) & (beats < dur)]

    inlier = float(keep_a.mean()) if len(keep_a) else 0.0
    rms_ms = rms_a * 1000.0
    return {
        "bpm": float(bpm), "beat": float(p_beat), "bar": float(p_bar),
        "meter": int(meter), "phase": float(phase),
        "downbeats": downbeats, "beats": beats,
        "n_db_peaks": int(len(t_d)), "n_beat_peaks": int(len(t_b)),
        "inlier_frac": inlier, "rms_ms": float(rms_ms),
        "db_peaks": t_d,  # raw refined peaks, for diagnostics/eval
        "ok": bool(inlier >= 0.6 and rms_ms <= 30.0 and len(t_d) >= 8),
    }


@torch.no_grad()
def fit_grid_fixed_bpm(model, mel, bpm, device="cuda", meter=4):
    """Grid fit with a USER-SUPPLIED BPM: the period is trusted, only the
    phase is estimated (a far easier problem than the free fit — this often
    unlocks the slot-exact path on songs whose free fit fails on octave or
    period confusion). Returns a grid dict or None."""
    if isinstance(mel, torch.Tensor):
        mel = mel.numpy()
    act, bin_s = beat_activations(model, mel, device)
    dur = mel.shape[1] / FPS
    p_beat = 60.0 / float(bpm)
    p_bar = meter * p_beat
    t_d = _refined_peaks(act[1], height=0.30, min_dist_s=0.5, bin_s=bin_s)
    t_b = _refined_peaks(act[0], height=0.30, min_dist_s=0.22, bin_s=bin_s)
    t_all = np.concatenate([t_b, t_d]) if len(t_b) else t_d
    if len(t_all) < 8:
        return None
    # beat-level phase (dense evidence), then bar phase by activation voting
    ph, keep, rms = _refit_phase(t_all, p_beat, float(np.median(t_all[:5])), iters=5)
    if not np.isfinite(rms):
        return None
    nb = act.shape[1]

    def _score(o):
        ts = np.arange(o - np.floor(o / p_bar) * p_bar, dur, p_bar)
        ix = np.clip(np.round(ts / bin_s).astype(int), 0, nb - 1)
        return float(act[1][ix].sum())

    phase = max((ph + j * p_beat for j in range(meter)), key=_score)
    phase -= np.floor(phase / p_bar) * p_bar
    downbeats = phase + np.arange(0, int((dur - phase) / p_bar) + 1) * p_bar
    downbeats = downbeats[(downbeats >= 0) & (downbeats < dur)]
    beats = phase + np.arange(0, int((dur - phase) / p_beat) + 1) * p_beat
    beats = beats[(beats >= 0) & (beats < dur)]
    inlier = float(keep.mean()) if len(keep) else 0.0
    return {
        "bpm": float(bpm), "beat": p_beat, "bar": p_bar, "meter": meter,
        "phase": float(phase), "downbeats": downbeats, "beats": beats,
        "n_db_peaks": int(len(t_d)), "n_beat_peaks": int(len(t_b)),
        "inlier_frac": inlier, "rms_ms": float(rms * 1000),
        "db_peaks": t_d, "fixed_bpm": True,
        # period is user-trusted, so the gate only needs the PHASE to be sane
        "ok": bool(inlier >= 0.5 and rms * 1000 <= 40.0),
    }


# --- adaptive piecewise-constant tempo fit ------------------------------------
# The BPM census over all 1155 songs: 34% are multi-BPM; among them the main
# tempo covers a median 91% of measures, contiguous-segment count is <=4 for
# 82% of songs but p99 = 29 (chained ritardando). So segmentation must be
# ADAPTIVE (change-point DP with a per-segment penalty), not a fixed K.


def _seg_fit(t, reject=0.16):
    """Fit t_k ~= phase + k*period over ONE segment of consecutive downbeat
    peaks (missing peaks allowed: k advances by round(gap/period)).
    Returns (period, phase, sse_seconds^2, n_inlier)."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return None
    d = np.diff(t)
    p0 = float(np.median(d))
    if p0 <= 0:
        return None
    k = np.concatenate([[0], np.cumsum(np.maximum(1, np.round(d / p0)))])
    for _ in range(3):
        km, tm = k.mean(), t.mean()
        denom = float(((k - km) ** 2).sum())
        if denom < 1e-9:
            return None
        p = float(((k - km) * (t - tm)).sum() / denom)
        if p <= 0:
            return None
        phi = float(tm - p * km)
        r = t - (phi + k * p)
        k = k + np.round(-r / p)  # re-assign indices after refinement
    r = t - (phi + k * p)
    keep = np.abs(r) < reject * p
    sse = float((r[keep] ** 2).sum()) + float((~keep).sum()) * (reject * p) ** 2
    return p, phi, sse, int(keep.sum())


def fit_grid_piecewise(model, mel, device="cuda", dev_thresh=0.18,
                       min_run=3, seg_rms_ms=25.0):
    """Main-grid + spliced exceptions (v2 after the DP version over-segmented).

    The BPM census says multi-BPM songs are ~91% main-tempo, so: fit the rigid
    grid first (full-song precision, outliers rejected), then find CONTIGUOUS
    runs of downbeat peaks that consistently deviate from it, refit each run
    locally, and splice. Segment count stays small by construction and the
    main grid keeps its precision. Returns None if the main fit fails.
    """
    if isinstance(mel, torch.Tensor):
        mel = mel.numpy()
    g = fit_grid(model, mel, device=device)
    if g is None:
        return None
    t_d = g["db_peaks"]
    if len(t_d) < 8:
        return None
    if not g["ok"]:
        # the single rigid grid failed: true multi-tempo song. Try every
        # coarse split point, fit both halves rigidly, keep the best pair
        # (recursion depth 1 -> up to 2 segments; census: covers most cases).
        best = None
        for cut in range(6, len(t_d) - 6, 3):
            fa = _seg_fit(t_d[:cut])
            fb = _seg_fit(t_d[cut:])
            if fa is None or fb is None:
                continue
            rms = np.sqrt((fa[2] + fb[2]) / len(t_d))
            if best is None or rms < best[0]:
                best = (rms, cut, fa, fb)
        if best is not None and best[0] * 1000 <= 30.0:
            _, cut, (pa, fia, _, na), (pb, fib, _, nb2) = best
            dur = mel.shape[1] / FPS
            t_mid = 0.5 * (t_d[cut - 1] + t_d[cut])
            da = fia + np.arange(0, int(np.ceil((t_mid - fia) / pa)) + 1) * pa
            da = da[(da >= 0) & (da < t_mid)]
            db2 = fib + np.arange(0, int(np.ceil((dur - fib) / pb)) + 1) * pb
            db2 = db2[(db2 >= t_mid) & (db2 < dur)]
            downbeats = np.sort(np.concatenate([da, db2]))
            inl = (na + nb2) / len(t_d)
            return {**g, "downbeats": downbeats,
                    "bpm": 240.0 / pa, "beat": 60.0 / (240.0 / pa) if pa else g["beat"],
                    "bar": pa, "phase": float(downbeats[0]) if len(downbeats) else g["phase"],
                    "segments": [(float(fia), 240.0 / pa), (float(t_mid), 240.0 / pb)],
                    "piecewise": True, "n_segments": 2,
                    "inlier_frac": float(inl), "rms_ms": float(best[0] * 1000),
                    "ok": bool(inl >= 0.7)}
        return {**g, "piecewise": False, "n_segments": 1,
                "segments": [(float(g["phase"]), float(g["bpm"]))]}
    bar = g["bar"]
    phi = g["phase"]
    dur = mel.shape[1] / FPS
    r = t_d - (phi + np.round((t_d - phi) / bar) * bar)
    bad = np.abs(r) > dev_thresh * bar
    # contiguous deviant runs of >= min_run peaks
    runs = []
    i = 0
    while i < len(t_d):
        if bad[i]:
            j = i
            while j + 1 < len(t_d) and bad[j + 1]:
                j += 1
            if j - i + 1 >= min_run:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    segments = [(float(phi), float(g["bpm"]))]
    if not runs:  # effectively constant: return the rigid fit unchanged
        return {**g, "piecewise": False, "n_segments": 1,
                "segments": segments}

    spliced = []
    n_bad_fixed = 0
    for i, j in runs:
        f = _seg_fit(t_d[i : j + 1])
        if f is None:
            continue
        p_loc, phi_loc, sse, ninl = f
        rms = np.sqrt(sse / max(j - i + 1, 1)) * 1000
        if rms > seg_rms_ms or not (0.8 <= p_loc <= 8.0):
            continue  # local section too messy: leave it to the main grid
        bpm_loc = 240.0 / p_loc
        if abs(bpm_loc - round(bpm_loc)) < 0.35:
            p2 = 240.0 / round(bpm_loc)
            rr = t_d[i:j+1] - (phi_loc + np.round((t_d[i:j+1] - phi_loc) / p2) * p2)
            if float(np.sqrt(np.mean(rr ** 2))) * 1000 <= rms * 1.3 + 4:
                p_loc = p2
                phi_loc = phi_loc + float(np.median(rr))
                bpm_loc = float(round(bpm_loc))
        t_lo = t_d[i] - 0.25 * p_loc
        t_hi = (t_d[j] + p_loc) if j + 1 >= len(t_d) else t_d[j + 1] - 0.25 * bar
        spliced.append((t_lo, t_hi, p_loc, phi_loc, bpm_loc))
        n_bad_fixed += j - i + 1

    if not spliced:  # nothing splice-worthy: keep the rigid fit verbatim
        return {**g, "piecewise": False, "n_segments": 1, "segments": segments}

    # main-grid barlines outside spliced spans + local barlines inside
    main_db = phi + np.arange(0, int(np.ceil((dur - phi) / bar)) + 1) * bar
    main_db = main_db[(main_db >= 0) & (main_db < dur)]
    keep = np.ones(len(main_db), bool)
    downbeats = []
    for t_lo, t_hi, p_loc, phi_loc, bpm_loc in spliced:
        keep &= ~((main_db >= t_lo) & (main_db < t_hi))
        loc = phi_loc + np.arange(0, int(np.ceil((t_hi - phi_loc) / p_loc)) + 1) * p_loc
        loc = loc[(loc >= max(t_lo, 0)) & (loc < min(t_hi, dur))]
        downbeats.append(loc)
        segments.append((float(max(t_lo, 0.0)), float(bpm_loc)))
    downbeats.append(main_db[keep])
    downbeats = np.concatenate(downbeats)
    downbeats = np.sort(downbeats)
    dk = np.concatenate([[True], np.diff(downbeats) > 0.3])
    downbeats = downbeats[dk]

    # residual after splicing (all peaks vs nearest final barline)
    err = np.array([np.min(np.abs(downbeats - t)) for t in t_d])
    inlier = float(np.mean(err < dev_thresh * bar))
    rms_ms = float(np.sqrt(np.mean(np.minimum(err, dev_thresh * bar) ** 2)) * 1000)
    return {
        **g,
        "downbeats": downbeats, "segments": sorted(segments),
        "piecewise": bool(spliced), "n_segments": 1 + len(spliced),
        "inlier_frac": inlier, "rms_ms": rms_ms,
        # splices only ADD local corrections to an already-vetted main grid —
        # never downgrade the rigid fit's verdict
        "ok": bool(g["ok"] or (inlier >= 0.75 and rms_ms <= 35.0)),
    }


# --- onset evidence: metrically weighted slot-comb refinement -----------------
# Audio onsets (spectral flux) are ~4x sharper in time than the beat head's
# 46 ms bins and 10-20x denser than downbeats. They cannot determine the
# metrical LEVEL (which line is beat 1) — but given the beat head's fit as an
# anchor, they lock (period, phase) far more precisely, and they can rescue
# songs whose beat-head activations are confused but whose percussion is clean.
# The comb is the TJA slot lattice (beat/24) with metrical-hierarchy weights:
# occupancy concentrates on strong positions, which is what makes a 16 ms
# lattice identifiable at all (a flat comb that fine would fit noise).

_W24 = np.full(24, 0.05)
_W24[0] = 1.0                    # beat
_W24[12] = 0.55                  # 8th
_W24[[6, 18]] = 0.30             # 16ths
_W24[[8, 16]] = 0.22             # 8th triplets
_W24[[3, 9, 15, 21]] = 0.10      # 32nds
_W24[[4, 20]] = 0.10             # 16th triplets


def onset_peaks(mel, max_n=3000):
    """Spectral-flux onset times (seconds) with sub-frame refinement."""
    from scipy.signal import find_peaks

    flux = np.maximum(0, np.diff(mel.astype(np.float32), axis=1)).sum(0)
    flux = np.concatenate([[0.0], flux])
    med = np.median(flux)
    idx, props = find_peaks(flux, height=med * 1.5, distance=max(2, int(0.035 * FPS)))
    if len(idx) > max_n:  # keep the strongest
        keep = np.argsort(props["peak_heights"])[-max_n:]
        idx = np.sort(idx[keep])
    out = []
    for i in idx:
        dx = 0.0
        if 0 < i < len(flux) - 1:
            d = flux[i - 1] - 2 * flux[i] + flux[i + 1]
            if d < -1e-9:
                dx = float(np.clip(0.5 * (flux[i - 1] - flux[i + 1]) / d, -0.5, 0.5))
        out.append((i + dx) / FPS)
    return np.asarray(out)


def comb_score(onsets, period_beat, phase, sig=0.012):
    """Sum over onsets of hierarchy-weight x Gaussian(dist to nearest slot)."""
    step = period_beat / 24.0
    x = (onsets - phase) / step
    k = np.round(x)
    dist = (x - k) * step
    w = _W24[(k.astype(int)) % 24]
    return float(np.sum(w * np.exp(-0.5 * (dist / sig) ** 2)))


def onset_polish(grid, mel, span_bpm=0.6, span_ms=30, n_p=25, n_f=31):
    """Local (period, phase) refinement of a trusted fit on onset evidence.
    Returns an updated grid dict (or the original if no improvement)."""
    ons = onset_peaks(mel)
    if len(ons) < 30 or grid is None:
        return grid
    p0, f0 = grid["beat"], grid["phase"]
    base = comb_score(ons, p0, f0)
    best = (base, p0, f0)
    for p in np.linspace(60.0 / (grid["bpm"] + span_bpm), 60.0 / (grid["bpm"] - span_bpm), n_p):
        for f in f0 + np.linspace(-span_ms / 1000, span_ms / 1000, n_f):
            s = comb_score(ons, p, f)
            if s > best[0]:
                best = (s, float(p), float(f))
    s1, p1, f1 = best
    if s1 <= base * 1.02:  # <2% gain: keep the beat-head fit
        return grid
    out = dict(grid)
    meter = grid.get("meter", 4)
    out["beat"], out["bpm"], out["bar"] = p1, 60.0 / p1, meter * p1
    # keep the bar phase on the SAME barline, re-expressed on the new beat grid
    out["phase"] = f1 + round((grid["phase"] - f1) / p1) * p1
    dur = mel.shape[1] / FPS
    ph, bar = out["phase"], out["bar"]
    ph -= np.floor(ph / bar) * bar
    out["downbeats"] = np.arange(ph, dur, bar)
    out["beats"] = np.arange(ph - np.floor(ph / p1) * p1, dur, p1)
    out["onset_gain"] = s1 / max(comb_score(ons, p0, f0), 1e-9)
    return out


def onset_rescue(mel, act=None, bpm_range=(65, 210)):
    """Grid estimation from onsets alone (+ optional downbeat activation for
    the bar phase) — for songs where the beat-head fit failed. Returns a grid
    dict with ok=True only when the comb evidence is decisive."""
    ons = onset_peaks(mel)
    if len(ons) < 40:
        return None
    dur = mel.shape[1] / FPS
    # tempo candidates from the IOI histogram (mode + octaves)
    iois = np.diff(ons)
    iois = iois[(iois > 0.08) & (iois < 2.0)]
    if len(iois) < 20:
        return None
    hist, edges = np.histogram(iois, bins=192, range=(0.08, 2.0))
    cand_p = []
    for i in np.argsort(hist)[-6:]:
        c = 0.5 * (edges[i] + edges[i + 1])
        for mult in (0.5, 1.0, 2.0, 4.0):
            p = c * mult
            if 60.0 / bpm_range[1] <= p <= 60.0 / bpm_range[0]:
                cand_p.append(p)
    best = (0.0, None, None)
    for p in sorted(set(np.round(cand_p, 4))):
        for f in np.arange(0.0, p, p / 48):
            s = comb_score(ons, p, f, sig=0.015)
            if s > best[0]:
                best = (s, float(p), float(f))
    if best[1] is None:
        return None
    # fine polish around the winner
    g0 = {"beat": best[1], "bpm": 60.0 / best[1], "phase": best[2], "meter": 4,
          "bar": 4 * best[1]}
    # integer-BPM snap when it costs <2% of the comb score
    bpm = g0["bpm"]
    for candb in (round(bpm), round(bpm * 2) / 2):
        if abs(candb - bpm) < 0.6 and candb > 0:
            pb = 60.0 / candb
            fb = max(np.arange(0.0, pb, pb / 96),
                     key=lambda f: comb_score(ons, pb, f))
            if comb_score(ons, pb, fb) >= 0.98 * best[0]:
                g0.update(beat=pb, bpm=float(candb), phase=float(fb), bar=4 * pb)
                break
    # bar phase: downbeat-activation voting among the 4 beat offsets
    ph = g0["phase"]
    if act is not None:
        if isinstance(act, tuple):
            act, _bin = act
        else:
            _bin = BIN
        nb = act.shape[1]

        def _sc(o):
            ts = np.arange(o - np.floor(o / g0["bar"]) * g0["bar"], dur, g0["bar"])
            ix = np.clip(np.round(ts / _bin).astype(int), 0, nb - 1)
            return float(act[1][ix].sum())

        ph = max((g0["phase"] + jj * g0["beat"] for jj in range(4)), key=_sc)
    ph -= np.floor(ph / g0["bar"]) * g0["bar"]
    # decisiveness gate: winner must clearly beat the runner-up octave
    alt = comb_score(ons, g0["beat"] * 2, ph) + comb_score(ons, g0["beat"] / 2, ph)
    score = comb_score(ons, g0["beat"], ph)
    ok = bool(score > 0.25 * len(ons) and score > 0.75 * alt)
    return {"bpm": g0["bpm"], "beat": g0["beat"], "bar": g0["bar"], "meter": 4,
            "phase": float(ph),
            "downbeats": np.arange(ph, dur, g0["bar"]),
            "beats": np.arange(ph - np.floor(ph / g0["beat"]) * g0["beat"], dur, g0["beat"]),
            "n_db_peaks": 0, "n_beat_peaks": int(len(ons)),
            "inlier_frac": float(score / max(len(ons), 1)), "rms_ms": -1.0,
            "db_peaks": np.asarray([]), "ok": ok, "rescued": True}


def debias_to_grid(times, phase, step):
    """Remove the generator's systematic latency: median signed offset of
    `times` to the nearest grid subdivision. Returns (shifted_times, offset).
    Relative timing (groove, tuplets) is untouched — this is a global shift."""
    t = np.asarray(times, float)
    if len(t) == 0:
        return t, 0.0
    r = (t - phase + step / 2) % step - step / 2
    off = float(np.median(r))
    return t - off, off
