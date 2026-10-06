"""Serving-time BarScript grid built from an ESTIMATED beat fit.

Space copy
----------
Derived from the shipping tree's ``src/softchart/barscript_grid.py``
(md5 7e810833ed2e5c796c5bcfa6e3c376c3).  Two deliberate differences, both
forced by where this file now lives:

1. The two lazy imports point at the vendored 2.0 subpackage
   (``.sc2.generate``, ``.sc2.vocab``) instead of the sibling modules.  In the
   Space, ``softchart.generate`` and ``softchart.vocab`` are the SoftChart 1.x
   library with an incompatible token table, so the shipping tree's plain
   ``.generate`` / ``.vocab`` would silently bind to the wrong decoder.
2. The corpus figure quoted below is corrected.  The shipping tree's docstring
   said "62.3 % of corpus charts contain a variable-tempo or variable-meter
   bar"; recomputing from ``trace`` over 1 134 songs / 4 449 charts separates
   the two, and only the TEMPO half is what this module addresses:
   **29.0 % of charts change tempo**, 52.4 % change meter.  Variable meter is
   explicitly out of scope (see below), so 62.3 % overstated this module's
   reach by roughly a factor of two.

Nothing else is changed; the grid arithmetic and the gates are identical.

Why this module exists
----------------------
The BarScript decoder consumes a *supplied grid* dict --
``{bpm, downbeats, measure_edges, measure_num, measure_den, bar_denoms}`` --
and every note time it produces, plus every ``#BPMCHANGE`` the TJA writer
emits, is derived from ``measure_edges`` alone.  The representation is
therefore exact under tempo change: ``tja.write_tja_slots`` recomputes
``bpm_m = 60 * quarter_beats[m] / (edges[m+1] - edges[m])`` per bar and writes
a ``#BPMCHANGE`` whenever it moves.

The two shipped grid builders sit at opposite ends of that contract:

``generate.barscript_grid_from_trace``
    exact, but needs an AUTHORED chart -- unavailable when serving audio.
``generate.barscript_grid_from_bpm``
    the deployed serving path: ``bar_sec = (240/bpm)*num/den`` tiled uniformly
    from one scalar BPM.  It cannot express a tempo change at all, so on the
    29.0 % of corpus charts that contain a tempo change the served grid is
    wrong by construction and the error accumulates over the song.

``barscript_grid_from_fit`` closes that gap without touching either builder or
the decoder: ``grid.fit_grid_piecewise`` already returns ``downbeats`` that are
NON-UNIFORM (the spliced local tempi are encoded in the spacing), so the fix is
to hand those times through as the bar edges instead of re-tiling from
``fit["bpm"]``.

What this module does NOT do
----------------------------
Per-bar METER is not estimated.  ``fit_grid`` reports a single ``meter``
(quarter beats per bar, 3 or 4, from the bar/beat period ratio), so the grid
carries a CONSTANT ``measure_num``/``measure_den`` and ``meter_source`` is
recorded as ``"fit_estimate"``.  A time-signature change inside the song is
representable by the grid dict and by the TJA writer, but nothing in the
serving stack can currently detect one from audio; claiming otherwise would be
a fabricated ``#MEASURE``.  Variable TEMPO is supported; variable METER is not.
"""

import numpy as np

# Structural sanity bounds on a single bar, expressed as the implied tempo
# 60 * quarter_beats_per_bar / bar_seconds.  These are NOT a tempo prior: they
# bracket what a bar can be at all.  `grid.fit_grid` only ever snaps BPM inside
# [40, 320] and only accepts a snap within a factor 0.4-2.5 of the raw fit, so
# [30, 400] is a strict superset of everything the estimator can emit; a bar
# outside it is a broken fit, not a slow or fast song.  The upper bound also
# keeps every bar far inside the decoder's WINDOW (1728 frames ~ 20.06 s), which
# `generate_song_barscript` requires each single bar to fit in.
GRID_MIN_BPM = 30.0
GRID_MAX_BPM = 400.0

# Bars appended past the last fitted downbeat to reach ``duration_sec``.  The
# fit already tiles to the mel duration, so the normal case needs zero: a
# request for more means the fit stopped early and the tail would be invented.
MAX_TAIL_EXTRAP_BARS = 4

# Splice joints in `fit_grid_piecewise` can leave one implausibly short bar
# where a local segment hands back to the main grid (upstream only guarantees
# a 0.3 s separation, which is below GRID_MAX_BPM at 4/4).  Dropping that single
# barline is a bounded repair; needing it often means the downbeat sequence is
# not a bar sequence.
MAX_JOINT_MERGE_FRAC = 0.05

SHIP_DENOM = 16  # deployed bar lattice (SOFTCHART2_RELEASE: --grid-bpm-denom 16)


class GridFitError(ValueError):
    """A fit cannot be turned into a well-formed bar grid.

    Raised (fail-closed) for malformed downbeat sequences: non-increasing or
    non-finite times, bar spacings outside [GRID_MIN_BPM, GRID_MAX_BPM], or a
    fit that does not cover ``duration_sec``.  It is NOT raised for a fit that
    is merely low-quality -- ``fit["ok"] == False`` falls back to the uniform
    grid instead, which is a downgrade the caller can report rather than an
    error it must handle.
    """


def _meter_terms(fit):
    """(numerator, denominator) for the single estimated time signature."""
    meter = fit.get("meter", 4)
    if isinstance(meter, (tuple, list, np.ndarray)):
        if len(meter) != 2:
            raise GridFitError("fit meter must be an int or a (num, den) pair")
        num, den = int(meter[0]), int(meter[1])
    else:
        num, den = int(meter), 4
    if num <= 0 or den <= 0:
        raise GridFitError(f"fit meter {num}/{den} must have positive terms")
    return num, den


def _fit_quality(fit):
    """Fit diagnostics, JSON-ready, with the keys a raw `fit_grid` lacks."""
    segments = fit.get("segments")
    if segments is None:
        segments = [(float(fit.get("phase", 0.0)), float(fit.get("bpm", 0.0)))]
    return {
        "ok": bool(fit.get("ok", False)),
        "inlier_frac": (None if fit.get("inlier_frac") is None
                        else float(fit["inlier_frac"])),
        "rms_ms": (None if fit.get("rms_ms") is None
                   else float(fit["rms_ms"])),
        "n_segments": int(fit.get("n_segments", len(segments))),
        "piecewise": bool(fit.get("piecewise", False)),
        "bpm": float(fit.get("bpm", 0.0)),
        "segments": [[float(t), float(b)] for t, b in segments],
    }


def _uniform_fallback(fit, duration_sec, num, den, denom, reason,
                      fallback_bpm, fallback_offset, spec_version):
    """Re-tile uniformly from a scalar BPM (the shipped serving behaviour).

    This is the only path that loses tempo change, so it is always labelled
    ``grid_source == "uniform_fallback"`` and carries ``reason``.
    """
    from .sc2.generate import barscript_grid_from_bpm

    bpm = fallback_bpm
    if bpm is None and fit is not None:
        bpm = fit.get("bpm")
    if bpm is None or not np.isfinite(float(bpm)) or float(bpm) <= 0:
        raise GridFitError(
            f"uniform fallback needs a usable BPM ({reason}); pass "
            "fallback_bpm= explicitly")
    offset = fallback_offset
    if offset is None and fit is not None:
        offset = fit.get("phase", 0.0)
    offset = 0.0 if offset is None else float(offset)
    if not np.isfinite(offset) or not 0.0 <= offset < duration_sec:
        offset = 0.0
    grid = barscript_grid_from_bpm(float(bpm), offset, float(duration_sec),
                                   meter=(num, den), denom=int(denom))
    edges = np.asarray(grid["measure_edges"], np.float64)
    grid["grid_source"] = "uniform_fallback"
    grid["fit_quality"] = None if fit is None else _fit_quality(fit)
    grid["grid_meta"] = {
        "reason": reason,
        "n_bars": len(grid["bar_denoms"]),
        "meter_num": int(num), "meter_den": int(den),
        "meter_source": "fit_estimate" if fit is not None else "caller",
        "meter_varies": False,
        "bar_denom": int(denom),
        "duration_sec": float(duration_sec),
        "first_downbeat_sec": float(edges[0]),
        "terminal_edge_sec": float(edges[-1]),
        "tail_bars_extrapolated": 0,
        "joint_bars_merged": 0,
        "trailing_downbeats_dropped": 0,
        "bar_sec_min": float(np.min(np.diff(edges))),
        "bar_sec_max": float(np.max(np.diff(edges))),
        "spec_labels_version": spec_version,
    }
    return grid


def barscript_grid_from_fit(fit, duration_sec, *, denom=SHIP_DENOM, spec=None,
                            require_ok=True, repair_joints=True,
                            on_invalid="raise", fallback_bpm=None,
                            fallback_offset=None,
                            min_bar_bpm=GRID_MIN_BPM,
                            max_bar_bpm=GRID_MAX_BPM):
    """Build a BarScript supplied grid from ``grid.fit_grid_piecewise`` output.

    Parameters
    ----------
    fit : dict or None
        A ``fit_grid`` / ``fit_grid_fixed_bpm`` / ``fit_grid_piecewise`` result.
        Only ``downbeats``, ``meter``, ``ok``, ``bpm`` and ``phase`` are read
        (the rest is copied into ``fit_quality`` for display).  ``None`` means
        beat estimation failed outright: the call falls back to the uniform
        grid and therefore needs ``fallback_bpm``.
    duration_sec : float
        Audio length the grid must cover (normally ``mel.shape[1] / FPS``).
    denom : int
        Per-bar lattice denominator teacher-forced into the ``[BAR]`` headers,
        uniform over the song.  Default 16 = the shipped serving decision.
    spec : BarscriptSpec, optional
        Not needed to build the grid (bar geometry is spec-independent); when
        given, its ``labels_version`` is recorded in ``grid_meta`` so a served
        grid carries the label contract it was requested under.
    require_ok : bool
        When True (default) a fit whose own ``ok`` gate failed is not used.
    repair_joints : bool
        Drop a downbeat that sits closer than one ``max_bar_bpm`` bar to its
        predecessor (a splice-joint artefact), up to ``MAX_JOINT_MERGE_FRAC``
        of the bars.  The LATER barline is dropped, matching the upstream
        0.3 s de-duplication in ``fit_grid_piecewise``.
    on_invalid : {"raise", "fallback"}
        What to do when the downbeat sequence is malformed.  ``"raise"``
        (default) is fail-closed; ``"fallback"`` lets a serving path degrade to
        the uniform grid with the reason recorded instead of erroring.

    Returns
    -------
    dict
        The six keys the decoder consumes -- ``bpm``, ``downbeats``,
        ``measure_edges``, ``measure_num``, ``measure_den``, ``bar_denoms`` --
        exactly as ``barscript_grid_from_bpm`` returns them, plus three
        provenance keys the decoder ignores:

        ``grid_source``
            ``"piecewise"``  -- bar edges ARE the fitted downbeat times, so a
            tempo change in the fit survives into the chart and the TJA.  (The
            name follows the producer, ``fit_grid_piecewise``; a constant-tempo
            song takes this path too, with ``fit_quality["piecewise"] is
            False`` and ``n_segments == 1``.)
            ``"uniform_fallback"`` -- re-tiled from one scalar BPM; ANY tempo
            change in the song is lost.  ``grid_meta["reason"]`` says why.
        ``fit_quality``
            ``{ok, inlier_frac, rms_ms, n_segments, piecewise, bpm, segments}``
            (``None`` when ``fit is None``).
        ``grid_meta``
            Bar counts, meter and its source, coverage, and every repair that
            was applied (``joint_bars_merged``, ``tail_bars_extrapolated``,
            ``trailing_downbeats_dropped``).

    Notes
    -----
    ``bpm`` is the fit's MAIN tempo and is display/metadata only under
    ``"piecewise"``: the decoder reads times from ``measure_edges``, and
    ``tja.write_tja_slots`` uses it solely for the header ``BPM:`` field before
    emitting per-bar ``#BPMCHANGE`` from the edges.
    """
    if on_invalid not in ("raise", "fallback"):
        raise ValueError("on_invalid must be 'raise' or 'fallback'")
    duration_sec = float(duration_sec)
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("duration_sec must be finite and positive")
    from .sc2.vocab import MAX_BAR_DENOM
    if (isinstance(denom, bool) or not isinstance(denom, (int, np.integer))
            or not 1 <= int(denom) <= MAX_BAR_DENOM):
        raise ValueError(f"denom must be an int in [1, {MAX_BAR_DENOM}]")
    denom = int(denom)
    if not np.isfinite(min_bar_bpm) or not np.isfinite(max_bar_bpm) \
            or not 0 < min_bar_bpm < max_bar_bpm:
        raise ValueError("min_bar_bpm/max_bar_bpm must satisfy 0 < min < max")

    spec_version = None
    if spec is not None:
        spec_version = getattr(spec, "labels_version", None)
        if spec_version is None:
            raise ValueError("spec must be a BarscriptSpec (no labels_version)")

    if fit is None:
        return _uniform_fallback(None, duration_sec, 4, 4, denom,
                                 "beat fit unavailable", fallback_bpm,
                                 fallback_offset, spec_version)
    num, den = _meter_terms(fit)
    quarter_beats = 4.0 * num / den
    min_bar_sec = 60.0 * quarter_beats / max_bar_bpm
    max_bar_sec = 60.0 * quarter_beats / min_bar_bpm

    if require_ok and not bool(fit.get("ok", False)):
        return _uniform_fallback(
            fit, duration_sec, num, den, denom,
            "fit.ok is False (inlier_frac "
            f"{fit.get('inlier_frac')!r}, rms_ms {fit.get('rms_ms')!r})",
            fallback_bpm, fallback_offset, spec_version)

    def _bail(reason):
        if on_invalid == "raise":
            raise GridFitError(reason)
        return _uniform_fallback(fit, duration_sec, num, den, denom, reason,
                                 fallback_bpm, fallback_offset, spec_version)

    raw_db = fit.get("downbeats")
    if raw_db is None:
        return _bail("fit has no 'downbeats'")
    try:
        db = np.asarray(raw_db, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as e:
        return _bail(f"fit downbeats are not a float array: {e}")
    if db.size < 2:
        return _bail(f"fit has {db.size} downbeat(s); at least 2 are needed "
                     "to establish a bar period")
    if not np.all(np.isfinite(db)):
        return _bail("fit downbeats contain non-finite times")
    if np.any(np.diff(db) <= 0):
        bad = int(np.argmax(np.diff(db) <= 0))
        return _bail(f"fit downbeats are not strictly increasing at index "
                     f"{bad} ({db[bad]:.6f} -> {db[bad + 1]:.6f})")
    if db[0] < 0:
        return _bail(f"first downbeat {db[0]:.6f}s is negative")
    if db[0] >= duration_sec:
        return _bail(f"first downbeat {db[0]:.6f}s is at or past the audio end "
                     f"({duration_sec:.6f}s)")
    if db[0] > max_bar_sec:
        return _bail(f"first downbeat {db[0]:.6f}s leaves an unbarred head "
                     f"longer than one bar ({max_bar_sec:.6f}s)")

    # trailing downbeats past the requested duration: the caller's duration is
    # authoritative for what has to be charted
    n_trailing = int(np.count_nonzero(db >= duration_sec))
    if n_trailing:
        db = db[db < duration_sec]
        if db.size < 2:
            return _bail("fewer than 2 downbeats remain inside duration_sec")

    # splice-joint repair (drop the LATER barline of an impossibly short bar)
    n_merged = 0
    if repair_joints:
        keep = [0]
        for i in range(1, db.size):
            if db[i] - db[keep[-1]] < min_bar_sec:
                n_merged += 1
                continue
            keep.append(i)
        if n_merged:
            limit = max(1, int(np.floor(MAX_JOINT_MERGE_FRAC * db.size)))
            if n_merged > limit:
                return _bail(
                    f"{n_merged} of {db.size} downbeats sit closer than one "
                    f"{max_bar_bpm:g} BPM bar ({min_bar_sec * 1e3:.1f} ms); "
                    f"above the {limit}-bar joint-repair budget this is not a "
                    "bar sequence")
            if len(keep) < 2:
                return _bail("joint repair left fewer than 2 downbeats")
            db = db[np.asarray(keep)]

    # terminal edge + tail: extend at the LAST observed bar period.  The bar
    # after the final downbeat is the only extrapolated one in the normal case;
    # anything beyond that is counted and budgeted.
    tail_period = float(db[-1] - db[-2])
    edges = list(db)
    edges.append(edges[-1] + tail_period)
    n_tail = 0
    while edges[-1] < duration_sec - 1e-9:
        if n_tail >= MAX_TAIL_EXTRAP_BARS:
            covered = (edges[-1] - float(db[0])) / max(duration_sec, 1e-9)
            return _bail(
                f"fit covers only {covered * 100:.1f}% of the audio; "
                f"reaching {duration_sec:.3f}s would take more than "
                f"{MAX_TAIL_EXTRAP_BARS} extrapolated bars")
        edges.append(edges[-1] + tail_period)
        n_tail += 1
    edges = np.asarray(edges, dtype=np.float64)

    bar_sec = np.diff(edges)
    if np.any(bar_sec <= 0):
        return _bail("bar edges are not strictly increasing after repair")
    too_short = np.flatnonzero(bar_sec < min_bar_sec - 1e-12)
    if too_short.size:
        i = int(too_short[0])
        return _bail(
            f"bar {i} lasts {bar_sec[i] * 1e3:.1f} ms -> "
            f"{60.0 * quarter_beats / bar_sec[i]:.1f} BPM, above the "
            f"{max_bar_bpm:g} BPM structural bound "
            f"({too_short.size} bar(s) affected)")
    too_long = np.flatnonzero(bar_sec > max_bar_sec + 1e-12)
    if too_long.size:
        i = int(too_long[0])
        return _bail(
            f"bar {i} lasts {bar_sec[i]:.3f} s -> "
            f"{60.0 * quarter_beats / bar_sec[i]:.1f} BPM, below the "
            f"{min_bar_bpm:g} BPM structural bound "
            f"({too_long.size} bar(s) affected)")

    n_bars = len(edges) - 1
    grid = {
        "bpm": float(fit.get("bpm", 60.0 * quarter_beats / float(np.median(bar_sec)))),
        "downbeats": edges[:-1],
        "measure_edges": edges,
        "measure_num": [int(num)] * n_bars,
        "measure_den": [int(den)] * n_bars,
        "bar_denoms": [denom] * n_bars,
        "grid_source": "piecewise",
        "fit_quality": _fit_quality(fit),
        "grid_meta": {
            "reason": None,
            "n_bars": n_bars,
            "meter_num": int(num), "meter_den": int(den),
            # single estimated time signature: per-bar meter is NOT estimable
            # from audio in this stack (see the module docstring)
            "meter_source": "fit_estimate",
            "meter_varies": False,
            "bar_denom": denom,
            "duration_sec": duration_sec,
            "first_downbeat_sec": float(edges[0]),
            "terminal_edge_sec": float(edges[-1]),
            "tail_bars_extrapolated": int(n_tail),
            "joint_bars_merged": int(n_merged),
            "trailing_downbeats_dropped": int(n_trailing),
            "bar_sec_min": float(bar_sec.min()),
            "bar_sec_max": float(bar_sec.max()),
            "bar_bpm_min": float(60.0 * quarter_beats / bar_sec.max()),
            "bar_bpm_max": float(60.0 * quarter_beats / bar_sec.min()),
            "n_distinct_bar_sec": int(
                np.unique(np.round(bar_sec, 6)).size),
            "spec_labels_version": spec_version,
        },
    }
    if not np.isfinite(grid["bpm"]) or grid["bpm"] <= 0:
        grid["bpm"] = float(60.0 * quarter_beats / float(np.median(bar_sec)))
    return grid


def describe_grid(grid):
    """One honest line about where a served grid came from, for logs and UI."""
    meta = grid.get("grid_meta") or {}
    q = grid.get("fit_quality")
    src = grid.get("grid_source", "unknown")
    n = meta.get("n_bars", len(grid.get("bar_denoms", [])))
    head = (f"{n} bars, {meta.get('meter_num', '?')}/{meta.get('meter_den', '?')} "
            f"(estimated, constant)")
    if src == "piecewise":
        rng = (f"{meta.get('bar_bpm_min', float('nan')):.1f}-"
               f"{meta.get('bar_bpm_max', float('nan')):.1f} BPM")
        detail = (f"grid from estimated barlines, {rng} across bars, "
                  f"{q['n_segments']} tempo segment(s)" if q else
                  "grid from estimated barlines")
        if meta.get("joint_bars_merged"):
            detail += f", {meta['joint_bars_merged']} joint bar(s) merged"
        if meta.get("tail_bars_extrapolated"):
            detail += f", {meta['tail_bars_extrapolated']} tail bar(s) extrapolated"
    else:
        detail = (f"UNIFORM {grid['bpm']:.2f} BPM fallback -- tempo changes are "
                  f"not represented ({meta.get('reason')})")
    if q is not None:
        inl = "n/a" if q["inlier_frac"] is None else format(q["inlier_frac"], ".3f")
        rms = "n/a" if q["rms_ms"] is None else format(q["rms_ms"], ".1f") + "ms"
        detail += f" [fit ok={q['ok']}, inlier={inl}, rms={rms}]"
    return f"{head}; {detail}"
