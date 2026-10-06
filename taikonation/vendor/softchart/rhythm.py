"""Rhythmic-structure utilities: adaptive rational-grid snapping and
tuplet-evenness / grid-consistency metrics.

The model emits note times on an 11.6 ms frame grid; musically, notes live at
rational subdivisions of the beat (halves, triplets, quintuplets, ...). These
utilities (a) snap generated notes onto a locally-selected rational grid, and
(b) measure how "musically exact" a chart's timing is.
"""

import numpy as np

# subdivisions considered per beat: binary, triple, quint, sept, and compounds
SUBDIVS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16)
COMPLEXITY_PENALTY = 0.002  # seconds of tolerated extra error per log2(subdiv)


def estimate_phase(times, beat, resolution=64):
    """Grid phase (sec) minimizing distance of notes to a fine subdivision grid."""
    times = np.asarray(times)
    if len(times) == 0:
        return 0.0
    fine = beat / 24
    cands = np.arange(0.0, beat, beat / resolution)
    errs = []
    for o in cands:
        d = (times - o) / fine
        errs.append(np.mean(np.abs(d - np.round(d))) * fine)
    return float(cands[int(np.argmin(errs))])


def snap_times(times, bpm, phase=None, local_beats=16, conf_gate_ms=18.0):
    """Snap each note to the best rational subdivision of its local beat.

    Improvements over naive global snapping (validated by the closed loop):
    - phase is re-estimated *locally* every `local_beats` beats, so tempo drift
      and mid-song BPM changes do not accumulate into cross-beat unevenness;
    - confidence gating: beats whose best-fit residual exceeds `conf_gate_ms`
      are left unsnapped (better raw than snapped-to-the-wrong-grid).
    Returns (snapped_times, info dict).
    """
    times = np.asarray(sorted(times), dtype=np.float64)
    if len(times) == 0 or not bpm or bpm <= 0:
        return times, {"phase": 0.0, "subdiv_hist": {}, "gated_frac": 0.0}
    beat = 60.0 / bpm
    if phase is None:
        phase = estimate_phase(times, beat)

    snapped = times.copy()
    subdiv_hist = {}
    n_gated = 0
    seg_len = local_beats * beat
    seg_idx = np.floor((times - phase) / seg_len).astype(int)
    for s in np.unique(seg_idx):
        seg_sel = seg_idx == s
        seg_t = times[seg_sel]
        # local phase refinement within this segment (bounded to +-1/8 beat)
        local_phase = phase
        if len(seg_t) >= 6:
            offs = np.arange(-beat / 8, beat / 8, beat / 64)
            fine = beat / 24
            errs = [np.mean(np.abs(((seg_t - phase - o) / fine)
                                   - np.round((seg_t - phase - o) / fine)) * fine)
                    for o in offs]
            local_phase = phase + float(offs[int(np.argmin(errs))])

        beat_idx = np.floor((seg_t - local_phase) / beat).astype(int)
        seg_snapped = seg_t.copy()
        for k in np.unique(beat_idx):
            sel = beat_idx == k
            t = seg_t[sel]
            t0 = local_phase + k * beat
            rel = (t - t0) / beat  # in [0, 1)
            best_d, best_err, best_raw = None, np.inf, np.inf
            for d in SUBDIVS:
                err = np.mean(np.abs(rel - np.round(rel * d) / d)) * beat
                score = err + COMPLEXITY_PENALTY * np.log2(d)
                if score < best_err:
                    best_err, best_d, best_raw = score, d, err
            if best_raw * 1000 > conf_gate_ms:  # low confidence: keep raw times
                n_gated += int(sel.sum())
                continue
            seg_snapped[sel] = t0 + np.round(rel * best_d) / best_d * beat
            subdiv_hist[int(best_d)] = subdiv_hist.get(int(best_d), 0) + int(sel.sum())
        snapped[seg_sel] = seg_snapped
    return snapped, {"phase": phase, "subdiv_hist": subdiv_hist,
                     "gated_frac": n_gated / len(times)}


def snap_chart(gen, bpm, min_gap=0.04):
    """Apply rational snapping to a generate_song() output dict (hits only).

    Post-snap min-gap collapse: snapping can push two notes into adjacent fine
    slots (< playable gap); merge them instead of emitting unplayable pairs.
    """
    if not gen["hits"]:
        return gen
    times = [h["t"] for h in gen["hits"]]
    snapped, info = snap_times(times, bpm)
    out = dict(gen)
    out["hits"] = []
    last = -1.0
    for h, t in zip(gen["hits"], snapped):
        if t - last < min_gap:  # collapse collisions created by snapping
            continue
        out["hits"].append({**h, "t": round(float(t), 4)})
        last = t
    out["snap_info"] = info
    return out


def grid_consistency(times, bpm, tol=0.015):
    """Fraction of notes within tol of the best rational grid + mean deviation.

    Uses the same per-beat adaptive subdivision selection as snap_times.
    """
    times = np.asarray(sorted(times), dtype=np.float64)
    if len(times) < 5 or not bpm or bpm <= 0:
        return None
    snapped, _ = snap_times(times, bpm)
    dev = np.abs(snapped - times)
    return {
        "on_grid_frac": float(np.mean(dev <= tol)),
        "mean_dev_ms": float(np.mean(dev) * 1000),
        "p95_dev_ms": float(np.percentile(dev, 95) * 1000),
    }


def tuplet_groups(times, bpm, min_len=3):
    """Find runs of >=min_len consecutive IOIs that form one subdivision class.

    Returns list of (subdiv_class, ioi_array). subdiv_class = round(beat/median_ioi),
    e.g. 2=8ths, 3=8th-triplets, 4=16ths, 5=quintuplets, 6=16th-triplets, 7=septuplets.
    """
    times = np.asarray(sorted(times), dtype=np.float64)
    if len(times) < min_len + 1 or not bpm or bpm <= 0:
        return []
    beat = 60.0 / bpm
    ioi = np.diff(times)
    groups = []
    i = 0
    while i < len(ioi):
        j = i
        while j + 1 < len(ioi) and 0.75 < ioi[j + 1] / max(ioi[i:j + 1].mean(), 1e-9) < 1.33:
            j += 1
        run = ioi[i : j + 1]
        if len(run) >= min_len:
            med = np.median(run)
            if med < 0.95 * beat:  # subdivision runs only
                cls = int(round(beat / med))
                if 2 <= cls <= 16:
                    groups.append((cls, run))
        i = j + 1
    return groups


def tuplet_evenness(times, bpm, min_len=3):
    """Within-group spacing consistency. Returns per-class and overall stats.

    CV = std(IOI)/mean(IOI) within a group (0 = perfectly even);
    max_dev_ms = worst absolute deviation from the group's mean IOI.
    """
    groups = tuplet_groups(times, bpm, min_len)
    if not groups:
        return None
    per_class = {}
    cvs, maxdevs = [], []
    for cls, run in groups:
        cv = float(np.std(run) / np.mean(run))
        md = float(np.max(np.abs(run - np.mean(run))) * 1000)
        cvs.append(cv)
        maxdevs.append(md)
        d = per_class.setdefault(cls, {"n": 0, "cv": [], "max_dev_ms": []})
        d["n"] += 1
        d["cv"].append(cv)
        d["max_dev_ms"].append(md)
    return {
        "n_groups": len(groups),
        "cv_mean": float(np.mean(cvs)),
        "max_dev_ms_mean": float(np.mean(maxdevs)),
        "per_class": {
            str(c): {"n": v["n"], "cv": float(np.mean(v["cv"])),
                     "max_dev_ms": float(np.mean(v["max_dev_ms"]))}
            for c, v in sorted(per_class.items())
        },
    }
