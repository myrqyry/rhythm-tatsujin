"""Automated proxies for playability, groove, and pattern structure.

Every proxy is computable without human input, from (events, bpm[, mel]).
Together with rhythm.py metrics they form the scorecard that closes the
evaluate -> diagnose -> improve loop.

Proxy design notes:
- playability = physical feasibility on the drum (min gaps, stream length,
  big-note crowding) + learnable structure (motif reuse, compressibility)
- groove = alignment with the metrical grid (strong-beat rate, accent placement)
  + audio grounding (onset-envelope energy at note times)
Reference values come from ground-truth charts of the same course (see scorecard).
"""

import zlib

import numpy as np

HIT_CLASSES = ("don", "ka", "don_big", "ka_big")
BIG_CLASSES = ("don_big", "ka_big")


def _hits(events):
    return sorted((t, c) for t, c in events if c in HIT_CLASSES)


def min_gap_violation_rate(events, min_gap=0.045):
    """Fraction of consecutive hits closer than a physically playable gap."""
    h = _hits(events)
    if len(h) < 2:
        return 0.0
    ioi = np.diff([t for t, _ in h])
    return float(np.mean(ioi < min_gap))


def stream_p99_nps(events, win=2.0):
    """99th percentile of local density (notes/sec in rolling windows) — stamina proxy."""
    t = np.array([x for x, _ in _hits(events)])
    if len(t) < 4:
        return 0.0
    counts = [np.sum((t >= x) & (t < x + win)) for x in np.arange(t[0], t[-1], 0.5)]
    return float(np.percentile(counts, 99) / win) if counts else 0.0


def big_crowding_rate(events, bpm):
    """Fraction of big notes followed by another hit within 0.45 beat.

    Big notes are played with both hands; crowding them against the next note
    is physically awkward and rarely done in official charts.
    """
    h = _hits(events)
    bigs = [(i, t) for i, (t, c) in enumerate(h) if c in BIG_CLASSES]
    if not bigs or not bpm or bpm <= 0:
        return None
    thresh = 0.45 * 60.0 / bpm
    crowded = sum(1 for i, t in bigs if i + 1 < len(h) and h[i + 1][0] - t < thresh)
    return crowded / len(bigs)


def big_note_rate(events):
    h = _hits(events)
    if not h:
        return 0.0
    return sum(1 for _, c in h if c in BIG_CLASSES) / len(h)


def _symbol_stream(events, bpm):
    """Quantized (ioi-class, note-type) symbol stream for structure metrics."""
    h = _hits(events)
    if len(h) < 3 or not bpm or bpm <= 0:
        return []
    beat = 60.0 / bpm
    syms = []
    for i in range(1, len(h)):
        ioi = h[i][0] - h[i - 1][0]
        frac = ioi / beat
        # classify IOI into musical classes
        classes = [0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, 1.5, 2.0]
        j = int(np.argmin([abs(frac - c) / c for c in classes]))
        ioi_cls = j if abs(frac - classes[j]) / classes[j] < 0.25 else len(classes)
        syms.append((ioi_cls, h[i][1]))
    return syms


def motif_reuse(events, bpm, n=4):
    """Fraction of n-grams (rhythm+type) that occur at least twice in the chart.

    Human charts repeat and transform motifs; random note salads do not.
    """
    syms = _symbol_stream(events, bpm)
    if len(syms) < n + 2:
        return None
    grams = [tuple(syms[i : i + n]) for i in range(len(syms) - n + 1)]
    from collections import Counter

    counts = Counter(grams)
    return float(sum(1 for g in grams if counts[g] >= 2) / len(grams))


def compression_ratio(events, bpm):
    """zlib-compressed size / raw size of the symbol stream. Lower = more structured."""
    syms = _symbol_stream(events, bpm)
    if len(syms) < 16:
        return None
    raw = bytes(b for s in syms for b in (s[0], HIT_CLASSES.index(s[1])))
    return float(len(zlib.compress(raw, 9)) / len(raw))


def strong_beat_rate(events, bpm, phase=None, tol=0.03):
    """Fraction of hits on integer beats (downbeat proxy needs measures; beat is enough)."""
    from .rhythm import estimate_phase

    t = np.array([x for x, _ in _hits(events)])
    if len(t) < 5 or not bpm or bpm <= 0:
        return None
    beat = 60.0 / bpm
    if phase is None:
        phase = estimate_phase(t, beat)
    rel = (t - phase) / beat
    return float(np.mean(np.abs(rel - np.round(rel)) * beat < tol))


def accent_on_strong(events, bpm, phase=None, tol=0.03):
    """Fraction of BIG notes that fall on integer beats — accent placement proxy."""
    from .rhythm import estimate_phase

    h = _hits(events)
    tb = np.array([t for t, c in h if c in BIG_CLASSES])
    if len(tb) < 3 or not bpm or bpm <= 0:
        return None
    beat = 60.0 / bpm
    if phase is None:
        phase = estimate_phase(np.array([t for t, _ in h]), beat)
    rel = (tb - phase) / beat
    return float(np.mean(np.abs(rel - np.round(rel)) * beat < tol))


def audio_energy_alignment(events, mel, fps):
    """Mean z-scored spectral-flux strength at hit times (GT-free audio grounding)."""
    t = np.array([x for x, _ in _hits(events)])
    if len(t) < 5:
        return None
    flux = np.maximum(0.0, np.diff(mel.astype(np.float32), axis=1)).sum(axis=0)
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    fr = np.clip(np.round(t * fps).astype(int), 1, len(flux) - 1)
    return float(np.mean(np.maximum.reduce([flux[fr - 1], flux[fr], flux[np.minimum(fr + 1, len(flux) - 1)]])))


def _pair_spans(events):
    """Pair span-start/end events into (t0, t1) tuples (for GT event streams)."""
    out = []
    open_t = None
    for t, c in sorted(events):
        if c in ("roll", "roll_big", "balloon") and open_t is None:
            open_t = t
        elif c == "end" and open_t is not None:
            out.append((open_t, t))
            open_t = None
    return out


def span_stats(events, spans=None):
    """Span-length distribution + rate. Added after a visual audit found
    generated balloons ~10s vs GT p50 ~1s (a metric blind spot)."""
    if spans is None:
        pairs = _pair_spans(events)
    else:
        pairs = [(s["t0"], s["t1"]) if isinstance(s, dict) else tuple(s) for s in spans]
    h = _hits(events)
    dur_min = max((h[-1][0] - h[0][0]) / 60.0, 1e-6) if len(h) > 1 else None
    if not pairs or dur_min is None:
        return {"span_p90_len": None, "span_per_min": 0.0 if dur_min else None}
    lens = np.array([b - a for a, b in pairs])
    return {"span_p90_len": float(np.percentile(lens, 90)),
            "span_per_min": float(len(pairs) / dur_min)}


def compute_proxies(events, bpm, mel=None, fps=None, spans=None):
    """All proxies for one chart. Returns dict (None where not computable)."""
    from .rhythm import grid_consistency, tuplet_evenness

    t = [x for x, _ in _hits(events)]
    out = {
        "nps": len(t) / max(t[-1] - t[0], 1e-6) if len(t) > 1 else 0.0,
        "min_gap_violation": min_gap_violation_rate(events),
        "stream_p99_nps": stream_p99_nps(events),
        "big_rate": big_note_rate(events),
        "big_crowding": big_crowding_rate(events, bpm),
        "motif_reuse": motif_reuse(events, bpm),
        "compression": compression_ratio(events, bpm),
        "strong_beat": strong_beat_rate(events, bpm),
        "accent_on_strong": accent_on_strong(events, bpm),
    }
    g = grid_consistency(t, bpm)
    if g:
        out["offgrid_rate"] = 1.0 - g["on_grid_frac"]
        out["grid_dev_ms"] = g["mean_dev_ms"]
    tp = tuplet_evenness(t, bpm)
    if tp:
        out["tuplet_cv"] = tp["cv_mean"]
    if mel is not None and fps:
        out["energy_align"] = audio_energy_alignment(events, mel, fps)
    out.update(span_stats(events, spans))
    return out


# direction: +1 higher is better, -1 lower is better, 0 match GT reference
PROXY_DIRECTION = {
    "nps": 0, "min_gap_violation": -1, "stream_p99_nps": 0, "big_rate": 0,
    "big_crowding": -1, "motif_reuse": 0, "compression": 0, "strong_beat": 0,
    "accent_on_strong": 0, "offgrid_rate": -1, "grid_dev_ms": -1,
    "tuplet_cv": -1, "energy_align": 1, "span_p90_len": 0, "span_per_min": 0,
}


def score_against_reference(proxy_row, ref_stats):
    """Normalized deviation of each proxy vs the GT reference distribution.

    ref_stats: {proxy: {"p25": .., "p50": .., "p75": ..}} per course.
    Returns {proxy: score} where score = robust z of |dev| (0 = at GT median;
    for directional proxies, only penalize the bad direction).
    """
    scores = {}
    for k, v in proxy_row.items():
        if v is None or k not in ref_stats:
            continue
        r = ref_stats[k]
        iqr = max(r["p75"] - r["p25"], 1e-6)
        d = PROXY_DIRECTION.get(k, 0)
        dev = (v - r["p50"]) / iqr
        if d == -1:
            dev = max(0.0, dev)  # only worse-than-GT (higher) penalized
        elif d == 1:
            dev = max(0.0, -dev)
        else:
            dev = abs(dev)
        scores[k] = float(dev)
    return scores
