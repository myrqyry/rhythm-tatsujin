"""Meter-aware TJA emission for slot-mode charts.

No quantization happens here — generate_song_slot() already produced exact
lattice positions (96 slots per 4/4 measure), so this writer just prints them.
Preview rendering and the .tja are identical by construction. Authored measure
edges determine exact per-measure tempo; meter changes emit #MEASURE directives.
"""

import numpy as np

from .meter import resolve_meter_layout
from .vocab import SLOTS

CHAR = {"don": "1", "ka": "2", "don_big": "3", "ka_big": "4",
        "roll": "5", "roll_big": "6", "balloon": "7"}


def _format_number(value):
    """Emit enough digits for a float to round-trip through the TJA text."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("TJA numeric fields must be finite")
    if value == 0.0:
        value = 0.0  # avoid a cosmetic negative zero in OFFSET
    return format(value, ".17g")


def write_tja_slots(gen, grid, title, course, level, wave, out_path=None,
                    balloon_count=10):
    """Serialize a slot generation on an explicit trusted meter timeline.

    ``grid`` must contain exact rational meter metadata and ``measure_edges``
    with one trusted terminal edge. Returns the TJA text and optionally writes
    it to ``out_path``.
    """
    if int(gen.get("slots_per_4_4", SLOTS)) != SLOTS:
        raise ValueError(f"expected {SLOTS} slots per 4/4 measure")
    n_meas = max(int(gen.get("n_measures", 0)),
                 (max((m for m, _, _ in gen.get("hits_slots", [])), default=-1) + 1),
                 (max((m1 for _, _, m1, _, _ in gen.get("spans_slots", [])),
                      default=-1) + 1), 1)
    fields = ("measure_slots", "measure_num", "measure_den")
    gen_has_layout = any(name in gen for name in fields)
    grid_has_layout = any(name in grid for name in fields) \
        or "meter" in grid or "beats_per_measure" in grid
    gen_layout = resolve_meter_layout(gen, n_meas) if gen_has_layout else None
    grid_layout = resolve_meter_layout(grid, n_meas) if grid_has_layout else None
    if gen_layout is not None and grid_layout is not None:
        gen_slots, gen_num, gen_den, _ = gen_layout
        grid_slots, grid_num, grid_den, _ = grid_layout
        same_fraction = all(
            int(gn) * int(gd) == int(gdn) * int(gnd)
            for gn, gnd, gdn, gd in zip(
                gen_num, gen_den, grid_num, grid_den)
        )
        if not np.array_equal(gen_slots, grid_slots) or not same_fraction:
            raise ValueError("generated meter layout disagrees with the timing grid")
    layout = grid_layout if grid_layout is not None else gen_layout
    if layout is None:
        raise ValueError(
            "TJA slot export requires exact meter metadata; refusing to assume 4/4")
    slots_per_measure, measure_num, measure_den, quarter_beats = layout
    offsets = np.concatenate([[0], np.cumsum(slots_per_measure)])

    def flat(measure, slot):
        measure, slot = int(measure), int(slot)
        if not 0 <= measure < n_meas:
            raise ValueError(f"measure index {measure} is outside 0..{n_meas - 1}")
        if not 0 <= slot < slots_per_measure[measure]:
            raise ValueError(
                f"slot {slot} is outside measure {measure}'s "
                f"0..{slots_per_measure[measure] - 1} range"
            )
        return int(offsets[measure] + slot)

    slots = {}
    for me, sl, cls in gen["hits_slots"]:
        position = flat(me, sl)
        if position in slots:
            raise ValueError(
                f"slot collision at measure {int(me)}, slot {int(sl)}")
        slots[position] = CHAR[cls]
    for m0, s0, m1, s1, typ in gen.get("spans_slots", []):
        a = flat(m0, s0)
        b = flat(m1, s1)
        if b <= a:
            raise ValueError("span end must be strictly after its start")
        if a in slots:
            raise ValueError(
                f"slot collision at span start measure {int(m0)}, slot {int(s0)}")
        if b in slots:
            raise ValueError(
                f"slot collision at span end measure {int(m1)}, slot {int(s1)}")
        slots[a] = CHAR[typ]
        slots[b] = "8"

    # Slot output is timing-exact only when every start and the terminal edge
    # are explicit.  Never fall back to BPM/downbeat extrapolation here: that
    # would make the serialized TJA disagree with the lattice used to decode.
    if "measure_edges" not in grid:
        raise ValueError(
            "slot TJA export requires explicit measure_edges including the "
            "trusted terminal edge"
        )
    tempo_edges = np.asarray(grid["measure_edges"], dtype=float)
    if (tempo_edges.ndim != 1 or len(tempo_edges) != n_meas + 1
            or not np.all(np.isfinite(tempo_edges))
            or np.any(np.diff(tempo_edges) <= 0)):
        raise ValueError(
            "measure_edges must contain one finite, increasing edge per "
            "measure plus the final edge")
    lines = []
    base_bpm = float(grid["bpm"])
    if not np.isfinite(base_bpm) or base_bpm <= 0:
        raise ValueError("grid bpm must be positive and finite")
    cur_bpm = base_bpm
    cur_meter = (4, 4)
    for m in range(n_meas):
        duration = float(tempo_edges[m + 1] - tempo_edges[m])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"measure {m} duration must be positive and finite")
        bpm_m = 60.0 * float(quarter_beats[m]) / duration
        # No integer snap and no perceptual threshold: even a small tempo
        # delta accumulates into a real chart/audio timing error.
        if bpm_m != cur_bpm:
            lines.append(f"#BPMCHANGE {_format_number(bpm_m)}")
            cur_bpm = bpm_m
        meter = (int(measure_num[m]), int(measure_den[m]))
        if meter != cur_meter:
            lines.append(f"#MEASURE {meter[0]}/{meter[1]}")
            cur_meter = meter
        start = int(offsets[m])
        lines.append("".join(slots.get(start + k, "0")
                             for k in range(int(slots_per_measure[m]))) + ",")
    balloons = [balloon_count] * sum(1 for s in gen.get("spans_slots", [])
                                     if s[4] == "balloon")
    offset = float(tempo_edges[0])
    tja = "\n".join([
        f"TITLE:{title} (SoftChart)", f"BPM:{_format_number(base_bpm)}",
        f"WAVE:{wave}", f"OFFSET:{_format_number(-offset)}",
        f"COURSE:{'Oni' if course == 'oni' else str(course).capitalize()}",
        f"LEVEL:{level}",
        f"BALLOON:{','.join(map(str, balloons))}" if balloons else "BALLOON:",
        "", "#START", *lines, "#END"]) + "\n"
    if out_path:
        with open(out_path, "w") as f:
            f.write(tja)
    return tja
