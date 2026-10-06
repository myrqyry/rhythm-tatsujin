"""TJA emission for slot-mode charts: slot indices ARE the chart.

No quantization happens here — generate_song_slot() already produced exact
lattice positions (measure, slot in 0..95), so this writer just prints them.
Preview rendering and the .tja are identical by construction. Piecewise
grids additionally emit #BPMCHANGE lines between measures.
"""

import numpy as np

from .vocab import SLOTS

CHAR = {"don": "1", "ka": "2", "don_big": "3", "ka_big": "4",
        "roll": "5", "roll_big": "6", "balloon": "7"}


def grid_measure_starts(grid, n_measures):
    """Return measure boundary times for a slot grid, extending if needed."""
    db = np.asarray(grid.get("downbeats", []), dtype=float)
    if len(db) >= n_measures + 1:
        return db[: n_measures + 1]
    if len(db) >= 2:
        step = float(np.median(np.diff(db)))
        start = float(db[0])
    else:
        step = 240.0 / float(grid["bpm"])
        start = float(db[0]) if len(db) else 0.0
    return start + np.arange(n_measures + 1, dtype=float) * step


def gogo_measure_mask(plan, measure_starts, n_measures):
    """Map plan climax blocks (flag == 2) to TJA measures."""
    mask = [False] * n_measures
    if not plan:
        return mask
    starts = np.asarray(measure_starts, dtype=float)
    if len(starts) < n_measures + 1:
        return mask
    for block in plan:
        if len(block) < 4 or int(block[3]) != 2:
            continue
        a, b = float(block[0]), float(block[1])
        if b <= a:
            continue
        for m in range(n_measures):
            if starts[m] < b and starts[m + 1] > a:
                mask[m] = True
    return mask


def append_measure_with_gogo(lines, measure_line, measure_idx, gogo_mask, in_gogo):
    """Append a measure line, opening/closing #GOGO commands at boundaries."""
    want_gogo = bool(gogo_mask[measure_idx]) if measure_idx < len(gogo_mask) else False
    if in_gogo and not want_gogo:
        lines.append("#GOGOEND")
        in_gogo = False
    if want_gogo and not in_gogo:
        lines.append("#GOGOSTART")
        in_gogo = True
    lines.append(measure_line)
    return in_gogo


def write_tja_slots(gen, grid, title, course, level, wave, out_path=None,
                    balloon_count=10, plan=None):
    """gen: generate_song_slot() result; grid: fit_grid() result (or a dict
    with downbeats+bpm). Returns the TJA text (and writes it if out_path)."""
    slots = {}
    for me, sl, cls in gen["hits_slots"]:
        slots.setdefault((me, sl), CHAR[cls])
    for m0, s0, m1, s1, typ in gen.get("spans_slots", []):
        a = (m0, s0)
        while a in slots:  # span start yields to hits: next free slot
            a = (a[0] + (a[1] + 1) // SLOTS, (a[1] + 1) % SLOTS)
        b = (m1, s1)
        while b in slots or b <= a:
            b = (b[0] + (b[1] + 1) // SLOTS, (b[1] + 1) % SLOTS)
        slots[a] = CHAR[typ]
        slots[b] = "8"
    n_meas = max(gen.get("n_measures", 0),
                 (max(m for m, _ in slots) + 1) if slots else 1)
    # piecewise-tempo grids: emit #BPMCHANGE whenever the per-measure BPM
    # (from consecutive fitted barlines) moves; TJA measure lines are unchanged
    db = np.asarray(grid.get("downbeats", []), float) if grid.get("piecewise") else None
    measure_starts = grid_measure_starts(grid, n_meas)
    gogo_mask = gogo_measure_mask(plan, measure_starts, n_meas)
    lines = []
    cur_bpm = float(grid["bpm"])
    in_gogo = False
    for m in range(n_meas):
        if in_gogo and not gogo_mask[m]:
            lines.append("#GOGOEND")
            in_gogo = False
        if db is not None and m + 1 < len(db):
            bpm_m = 240.0 / (db[m + 1] - db[m])
            if abs(bpm_m - round(bpm_m)) < 0.05:
                bpm_m = float(round(bpm_m))
            if abs(bpm_m - cur_bpm) > 0.05:
                lines.append(f"#BPMCHANGE {bpm_m:g}")
                cur_bpm = bpm_m
        in_gogo = append_measure_with_gogo(
            lines, "".join(slots.get((m, k), "0") for k in range(SLOTS)) + ",",
            m, gogo_mask, in_gogo)
    if in_gogo:
        lines.append("#GOGOEND")
    balloons = [balloon_count] * sum(1 for s in gen.get("spans_slots", [])
                                     if s[4] == "balloon")
    offset = float(grid["downbeats"][0]) if len(grid.get("downbeats", [])) else 0.0
    tja = "\n".join([
        f"TITLE:{title} (SoftChart)", f"BPM:{grid['bpm']:g}", f"WAVE:{wave}",
        f"OFFSET:{-offset:.3f}",
        f"COURSE:{'Oni' if course == 'oni' else str(course).capitalize()}",
        f"LEVEL:{level}",
        f"BALLOON:{','.join(map(str, balloons))}" if balloons else "BALLOON:",
        "", "#START", *lines, "#END"]) + "\n"
    if out_path:
        with open(out_path, "w") as f:
            f.write(tja)
    return tja
