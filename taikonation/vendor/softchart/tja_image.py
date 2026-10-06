"""Server-side TJA chart image rendering for the Space.

The layout follows the same practical convention as tja-tools: full-chart rows
with 16 beats per row, measure barlines, colored note heads, and long-note
spans. SoftChart emits simple single-course TJAs, so this renderer intentionally
supports the generated subset instead of bundling the browser/webpack canvas
app at runtime.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from .fonts import cjk_font_path


_NOTE_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_COURSE_NAME = {
    "0": "Easy",
    "1": "Normal",
    "2": "Hard",
    "3": "Oni",
    "4": "Ura",
}


@dataclass
class ParsedTJA:
    headers: dict[str, str]
    measures: list[str]
    bpm_changes: dict[int, str]
    gogo_ranges: list[tuple[int, int]]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cjk_path = cjk_font_path(bold=bold)
    names = (
        cjk_path,
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    )
    for name in names:
        if name is None:
            continue
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _strip_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def parse_tja(tja_text: str) -> ParsedTJA:
    headers: dict[str, str] = {}
    measures: list[str] = []
    bpm_changes: dict[int, str] = {}
    gogo_ranges: list[tuple[int, int]] = []
    in_chart = False
    in_gogo = False
    gogo_start = 0
    measure_buf: list[str] = []

    def flush_measure() -> None:
        measure = "".join(measure_buf).strip()
        measures.append(measure or "0")
        measure_buf.clear()

    for raw_line in tja_text.splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("#START"):
            in_chart = True
            continue
        if not in_chart:
            if ":" in line and not line.startswith("#"):
                key, value = line.split(":", 1)
                headers[key.strip().upper()] = value.strip()
            continue
        if upper.startswith("#END"):
            break
        if upper.startswith("#GOGOSTART"):
            if not in_gogo:
                gogo_start = len(measures)
                in_gogo = True
            continue
        if upper.startswith("#GOGOEND"):
            if in_gogo:
                gogo_ranges.append((gogo_start, len(measures)))
                in_gogo = False
            continue
        if upper.startswith("#BPMCHANGE"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                bpm_changes[len(measures)] = parts[1].strip()
            continue
        if line.startswith("#"):
            continue
        for ch in line:
            if ch == ",":
                flush_measure()
            elif ch.upper() in _NOTE_CHARS:
                measure_buf.append(ch.upper())

    if measure_buf:
        flush_measure()
    if in_gogo:
        gogo_ranges.append((gogo_start, len(measures)))
    if not measures:
        measures.append("0")
    return ParsedTJA(headers=headers, measures=measures, bpm_changes=bpm_changes,
                     gogo_ranges=gogo_ranges)


def _course_label(value: str) -> str:
    return _COURSE_NAME.get(value, value[:1].upper() + value[1:].lower() if value else "")


def _text_right(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
                font: ImageFont.ImageFont, fill: str) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]), y), text, font=font, fill=fill)


def _beat_xy(beat: float, row_beats: float, beat_width: int, left: int,
             top: int, row_h: int, note_y: int, rows: int) -> tuple[int, int, int]:
    row = max(0, min(rows - 1, int(math.floor(beat / row_beats))))
    x = int(round(left + (beat - row * row_beats) * beat_width))
    y = top + row * row_h + note_y
    return row, x, y


def _note_events(measures: list[str]) -> tuple[list[tuple[float, str]], list[tuple[float, float, str]]]:
    events: list[tuple[float, str]] = []
    rolls: list[tuple[float, float, str]] = []
    open_roll: tuple[float, str] | None = None

    for midx, measure in enumerate(measures):
        div = max(len(measure), 1)
        for pos, symbol in enumerate(measure):
            if symbol == "0":
                continue
            beat = midx * 4.0 + 4.0 * pos / div
            if symbol in {"5", "6", "7", "9"}:
                if open_roll is not None:
                    rolls.append((open_roll[0], beat, open_roll[1]))
                open_roll = (beat, symbol)
                events.append((beat, symbol))
            elif symbol == "8":
                if open_roll is not None:
                    rolls.append((open_roll[0], beat, open_roll[1]))
                    open_roll = None
                else:
                    events.append((beat, symbol))
            else:
                events.append((beat, symbol))

    if open_roll is not None:
        rolls.append((open_roll[0], max(4.0 * len(measures), open_roll[0] + 0.25), open_roll[1]))
    return events, rolls


def _draw_roll(draw: ImageDraw.ImageDraw, start: float, end: float, symbol: str,
               row_beats: float, beat_width: int, left: int, top: int, row_h: int,
               note_y: int, rows: int) -> None:
    color = "#b77de1" if symbol in {"7", "9"} else "#e9b83e"
    edge = "#6f5520" if symbol not in {"7", "9"} else "#6c4388"
    start_row = max(0, int(math.floor(start / row_beats)))
    end_row = max(0, int(math.floor(max(end - 1e-6, start) / row_beats)))
    for row in range(start_row, min(end_row, rows - 1) + 1):
        seg0 = max(start, row * row_beats)
        seg1 = min(end, (row + 1) * row_beats)
        if seg1 <= seg0:
            continue
        x0 = int(round(left + (seg0 - row * row_beats) * beat_width))
        x1 = int(round(left + (seg1 - row * row_beats) * beat_width))
        y = top + row * row_h + note_y
        draw.rounded_rectangle((x0, y - 8, max(x1, x0 + 3), y + 8),
                               radius=8, fill=color, outline=edge, width=2)


def _draw_note(draw: ImageDraw.ImageDraw, x: int, y: int, symbol: str,
               label_font: ImageFont.ImageFont) -> None:
    if symbol in {"1", "3", "A"}:
        fill = "#e34b3f"
    elif symbol in {"2", "4", "B"}:
        fill = "#4aa8c0"
    elif symbol in {"5", "6"}:
        fill = "#e9b83e"
    elif symbol in {"7", "9"}:
        fill = "#b77de1"
    elif symbol == "8":
        fill = "#555555"
    else:
        fill = "#8d96a0"

    radius = 14 if symbol in {"3", "4", "6", "A", "B"} else 10
    if symbol == "8":
        draw.rectangle((x - 5, y - 12, x + 5, y + 12), fill=fill)
        return

    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill=fill, outline="#202020", width=2)
    draw.ellipse((x - radius + 3, y - radius + 3, x - radius + 8, y - radius + 8),
                 fill="#ffffff")
    label = "B" if symbol in {"3", "4", "6", "A", "B"} else ("R" if symbol in {"5", "6"} else ("P" if symbol in {"7", "9"} else ""))
    if label:
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2 - 1),
                  label, font=label_font, fill="#1e1e1e")


def render_tja_image(tja_text: str, out_path: str | None = None) -> str:
    parsed = parse_tja(tja_text)
    headers = parsed.headers
    row_beats = 16.0
    try:
        row_beats = float(headers.get("TTROWBEAT", row_beats))
    except ValueError:
        row_beats = 16.0
    row_beats = min(max(row_beats, 4.0), 32.0)

    title = headers.get("TITLE", "SoftChart")
    course = _course_label(headers.get("COURSE", ""))
    level = headers.get("LEVEL", "")
    bpm = headers.get("BPM", "")
    wave = headers.get("WAVE", "")
    measures = parsed.measures
    total_beats = max(4.0 * len(measures), 4.0)
    rows = max(1, int(math.ceil(total_beats / row_beats)))

    left = 72
    right = 30
    beat_width = 56
    header_h = 90
    row_h = 74
    note_y = 45
    footer_h = 22
    width = int(left + row_beats * beat_width + right)
    height = int(header_h + rows * row_h + footer_h)

    img = Image.new("RGB", (width, height), "#f5f3ee")
    draw = ImageDraw.Draw(img)
    title_font = _font(24, bold=True)
    meta_font = _font(14)
    small_font = _font(11)
    note_font = _font(10, bold=True)

    draw.rectangle((0, 0, width, header_h - 8), fill="#23272f")
    draw.text((24, 16), title, font=title_font, fill="#ffffff")
    meta = " · ".join(p for p in (course, f"Lv.{level}" if level else "", f"BPM {bpm}" if bpm else "") if p)
    draw.text((24, 51), meta or "Generated TJA chart", font=meta_font, fill="#d8dde7")
    if wave:
        _text_right(draw, (width - 24, 52), f"WAVE {os.path.basename(wave)}", meta_font, "#d8dde7")
    _text_right(draw, (width - 24, 18), "red=don  blue=ka  yellow=roll  purple=balloon  pink=gogo",
                small_font, "#b8c0cc")

    for row in range(rows):
        y0 = header_h + row * row_h
        y1 = y0 + row_h - 12
        fill = "#d8d8d3" if row % 2 == 0 else "#cecec9"
        draw.rounded_rectangle((14, y0, width - 14, y1), radius=8, fill=fill)
        for start_measure, end_measure in parsed.gogo_ranges:
            start_beat = start_measure * 4.0
            end_beat = end_measure * 4.0
            row_start = row * row_beats
            row_end = (row + 1) * row_beats
            seg0 = max(start_beat, row_start)
            seg1 = min(end_beat, row_end)
            if seg1 <= seg0:
                continue
            x0 = int(round(left + (seg0 - row_start) * beat_width))
            x1 = int(round(left + (seg1 - row_start) * beat_width))
            draw.rectangle((x0, y0 + 6, x1, y1 - 6), fill="#eeb7b7")
        cy = y0 + note_y
        draw.line((left, cy, width - right, cy), fill="#6d6d68", width=1)
        for b in range(int(row_beats) + 1):
            x = int(round(left + b * beat_width))
            is_bar = b % 4 == 0
            draw.line((x, y0 + 8, x, y1 - 6),
                      fill="#5c5c58" if is_bar else "#eeeeea",
                      width=2 if is_bar else 1)
            if is_bar:
                measure_idx = int(row * row_beats / 4 + b / 4)
                if measure_idx < len(measures):
                    draw.text((x + 3, y0 + 9), str(measure_idx + 1),
                              font=small_font, fill="#4a4a46")

    for measure_idx, bpm_value in parsed.bpm_changes.items():
        beat = measure_idx * 4.0
        row, x, _ = _beat_xy(beat, row_beats, beat_width, left, header_h, row_h, note_y, rows)
        draw.text((x + 4, header_h + row * row_h + 26), f"BPM {bpm_value}",
                  font=small_font, fill="#8b3d3d")

    events, rolls = _note_events(measures)
    for start, end, symbol in rolls:
        _draw_roll(draw, start, end, symbol, row_beats, beat_width, left, header_h, row_h, note_y, rows)
    for beat, symbol in events:
        _, x, y = _beat_xy(beat, row_beats, beat_width, left, header_h, row_h, note_y, rows)
        _draw_note(draw, x, y, symbol, note_font)

    if out_path is None:
        out_path = tempfile.mktemp(suffix="_tja.png")
    img.save(out_path)
    return out_path
