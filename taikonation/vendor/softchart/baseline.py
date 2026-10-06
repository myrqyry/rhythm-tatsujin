"""Onset-detection baseline: spectral-flux peak picking on the cached log-mel
+ band-energy don/ka heuristic.

This is the "chart generation is just onset detection" strawman the model must beat.
Operates on the same cached mel features as the model, so no raw audio is needed.
"""

import numpy as np
from scipy.signal import find_peaks

from .vocab import FPS


def baseline_chart(mel, target_nps):
    """mel: (n_mels, T) log-mel (natural log). Returns list of (time_sec, class)."""
    m = mel.astype(np.float32)
    # spectral flux onset envelope
    flux = np.maximum(0.0, np.diff(m, axis=1)).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    # smooth lightly
    k = np.hanning(5)
    k /= k.sum()
    env = np.convolve(flux, k, mode="same")

    dur = m.shape[1] / FPS
    want = max(1, int(round(target_nps * dur)))
    peaks, props = find_peaks(env, distance=int(0.09 * FPS), height=0.0)
    if len(peaks) == 0:
        return []
    if len(peaks) > want:
        strongest = np.argsort(props["peak_heights"])[-want:]
        peaks = np.sort(peaks[strongest])

    # don/ka: low vs high mel-band energy at the onset (linear power domain)
    p = np.exp(m)
    low = p[:40].sum(axis=0)   # ~ <500 Hz
    high = p[64:].sum(axis=0)  # upper bands
    events = []
    for fr in peaks:
        fr2 = min(fr + 1, m.shape[1] - 1)
        ratio = high[fr2] / (low[fr2] + 1e-9)
        events.append((float(fr / FPS), "ka" if ratio > 0.35 else "don"))
    return events
