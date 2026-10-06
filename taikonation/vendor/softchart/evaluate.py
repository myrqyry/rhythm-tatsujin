"""Objective evaluation metrics for generated charts."""

import numpy as np

HIT_CLASSES = ("don", "ka", "don_big", "ka_big")


def match_onsets(ref_times, est_times, tol=0.05):
    """Greedy bipartite matching (mir_eval style). Returns list of (ri, ei)."""
    matches = []
    ri, ei = 0, 0
    used_r = set()
    # standard approach: for each est, match nearest unmatched ref within tol
    ref = np.asarray(ref_times)
    est = np.asarray(est_times)
    order = np.argsort(est)
    ref_order = np.argsort(ref)
    ref_sorted = ref[ref_order]
    taken = np.zeros(len(ref), dtype=bool)
    for e_i in order:
        t = est[e_i]
        lo = np.searchsorted(ref_sorted, t - tol)
        hi = np.searchsorted(ref_sorted, t + tol)
        best, best_d = -1, tol + 1
        for k in range(lo, hi):
            rid = ref_order[k]
            if taken[rid]:
                continue
            d = abs(ref_sorted[k] - t)
            if d < best_d:
                best, best_d = rid, d
        if best >= 0:
            taken[best] = True
            matches.append((best, e_i))
    return matches


def onset_prf(ref_times, est_times, tol=0.05):
    if len(ref_times) == 0 and len(est_times) == 0:
        return 1.0, 1.0, 1.0
    if len(ref_times) == 0 or len(est_times) == 0:
        return 0.0, 0.0, 0.0
    m = len(match_onsets(ref_times, est_times, tol))
    p = m / len(est_times)
    r = m / len(ref_times)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def type_accuracy(ref, est, tol=0.05):
    """ref/est: list of (time, class). Accuracy + per-class stats on matched onsets."""
    if not ref or not est:
        return None
    rt = [t for t, _ in ref]
    et = [t for t, _ in est]
    matches = match_onsets(rt, et, tol)
    if not matches:
        return None
    correct = sum(1 for ri, ei in matches if ref[ri][1] == est[ei][1])
    per_class = {}
    for ri, ei in matches:
        c = ref[ri][1]
        d = per_class.setdefault(c, [0, 0])
        d[1] += 1
        if est[ei][1] == c:
            d[0] += 1
    # binary don/ka accuracy (big variants folded in)
    fold = lambda c: "don" if "don" in c else ("ka" if "ka" in c else c)
    correct2 = sum(1 for ri, ei in matches if fold(ref[ri][1]) == fold(est[ei][1]))
    return {
        "acc": correct / len(matches),
        "acc_donka": correct2 / len(matches),
        "n_matched": len(matches),
        "per_class": {c: v[0] / v[1] for c, v in per_class.items()},
    }


def density(events, span=None):
    times = [t for t, c in events if c in HIT_CLASSES]
    if len(times) < 2:
        return 0.0
    span = span or (max(times) - min(times))
    return len(times) / max(span, 1e-6) if span > 5 else 0.0


def ioi_hist(events, bpm, bins=None):
    """Inter-onset intervals in beat units, snapped to musical fractions."""
    if bins is None:
        bins = [0.125, 1 / 6, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    times = sorted(t for t, c in events if c in HIT_CLASSES)
    if len(times) < 3 or not bpm or bpm <= 0:
        return None
    ioi = np.diff(times) * bpm / 60.0  # beats
    hist = np.zeros(len(bins) + 1)
    for x in ioi:
        d = [abs(x - b) / b for b in bins]
        j = int(np.argmin(d))
        if d[j] < 0.2:
            hist[j] += 1
        else:
            hist[-1] += 1  # off-grid
    s = hist.sum()
    return hist / s if s > 0 else None


def js_divergence(p, q, eps=1e-9):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ngram_dist(events, n=2):
    """Distribution over note-class n-grams (hits only)."""
    seq = [c for _, c in sorted(events) if c in HIT_CLASSES]
    if len(seq) < n + 1:
        return {}
    counts = {}
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i : i + n])
        counts[g] = counts.get(g, 0) + 1
    tot = sum(counts.values())
    return {g: c / tot for g, c in counts.items()}


def ngram_js(ref_events, est_events, n=2):
    pr = ngram_dist(ref_events, n)
    pe = ngram_dist(est_events, n)
    if not pr or not pe:
        return None
    keys = sorted(set(pr) | set(pe))
    return js_divergence([pr.get(k, 0) for k in keys], [pe.get(k, 0) for k in keys])


def distinct_n(events, n=3):
    seq = [c for _, c in sorted(events) if c in HIT_CLASSES]
    if len(seq) < n:
        return None
    grams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
    return len(set(grams)) / len(grams)


def spearman(x, y):
    from scipy.stats import spearmanr

    if len(x) < 2:
        return None
    r = spearmanr(x, y).statistic
    return None if np.isnan(r) else float(r)
