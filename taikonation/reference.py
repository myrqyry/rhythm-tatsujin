"""Local osu!taiko reference-corpus import and difficulty analysis.

OSZ archives often contain copyrighted audio. Imports belong under ignored
runtime output and source audio should never be committed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import math
import re
import shutil
import statistics
import zipfile

DIFFICULTY_ORDER = {"easy": 0, "normal": 1, "hard": 2, "oni": 3, "ura": 4}
SNAP_CANDIDATES = (
    1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 3 / 8,
    1 / 2, 2 / 3, 3 / 4, 1.0, 1.5, 2.0, 3.0, 4.0,
)


def normalize_taiko_difficulty(version: str) -> str | None:
    value = re.sub(r"['’]s\b", " ", version.lower())
    tokens = re.findall(r"[a-z]+", value)
    words = set(tokens)

    # osu! diff names sometimes stylize standard labels as O.N.I. or
    # M.U.Z.U.K.A.S.H.I.I. Collapse only consecutive one-letter runs so
    # custom names such as "Expressionism" are never substring-matched.
    letter_runs = []
    run = []
    for token in tokens + ["__end__"]:
        if len(token) == 1:
            run.append(token)
        elif run:
            if len(run) >= 2:
                letter_runs.append("".join(run))
            run = []
    words.update(letter_runs)

    if (
        ("inner" in words and "oni" in words)
        or ("ura" in words and "oni" in words)
        or "ura" in words
        or "inneroni" in words
        or "uraoni" in words
    ):
        return "ura"
    if "oni" in words or "ni" in words:
        return "oni"
    if {"muzukashii", "muzukashi", "hard", "mzkshi"} & words:
        return "hard"
    if {"futsuu", "futsu", "normal", "ftsu"} & words:
        return "normal"
    if {"kantan", "easy"} & words:
        return "easy"
    return None


def difficulty_mapper(creator: str | None, version: str | None) -> str | None:
    """Recover guest mapper attribution from names such as "Nifty's Oni"."""
    if version:
        marker = version.lower().find("'s ")
        if marker > 0:
            return version[:marker].strip()
    return creator


def _parse_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def _key_values(lines: Iterable[str]) -> dict[str, str]:
    result = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def parse_osu(text: str) -> dict:
    sections = _parse_sections(text)
    general = _key_values(sections.get("General", []))
    metadata = _key_values(sections.get("Metadata", []))
    difficulty = _key_values(sections.get("Difficulty", []))

    timing_points = []
    for line in sections.get("TimingPoints", []):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            timing_points.append({
                "time": float(parts[0]),
                "beat_length": float(parts[1]),
                "uninherited": int(parts[6]) if len(parts) > 6 else 1,
            })
        except ValueError:
            continue

    hit_objects = []
    for line in sections.get("HitObjects", []):
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            hit_objects.append({
                "time": int(parts[2]),
                "type": int(parts[3]),
                "hitsound": int(parts[4]),
            })
        except ValueError:
            continue

    return {
        "general": general,
        "metadata": metadata,
        "difficulty_settings": difficulty,
        "timing_points": timing_points,
        "hit_objects": hit_objects,
    }


def _active_beat_length(red_points: list[dict], time_ms: int) -> float | None:
    active = None
    for point in red_points:
        if point["time"] <= time_ms:
            active = point
        else:
            break
    if active is None and red_points:
        active = red_points[0]
    if active is None or active["beat_length"] <= 0:
        return None
    return float(active["beat_length"])


def _nearest_snap(beats: float) -> float | str:
    nearest = min(SNAP_CANDIDATES, key=lambda candidate: abs(candidate - beats))
    return nearest if abs(nearest - beats) <= 0.03 else "other"


def analyze_taiko_map(parsed: dict) -> dict:
    metadata = parsed["metadata"]
    general = parsed["general"]
    if general.get("Mode") != "1":
        raise ValueError("Reference map is not osu!taiko (Mode:1)")

    difficulty = normalize_taiko_difficulty(metadata.get("Version", ""))
    red_points = sorted(
        (
            point
            for point in parsed["timing_points"]
            if point["uninherited"] == 1 and point["beat_length"] > 0
        ),
        key=lambda point: point["time"],
    )
    notes = [obj for obj in parsed["hit_objects"] if obj["type"] & 1]
    note_times = [obj["time"] for obj in notes]
    duration_s = (
        (max(note_times) - min(note_times)) / 1000.0 if len(note_times) >= 2 else 0.0
    )

    interval_ms = []
    interval_beats = []
    snap_counts: dict[str, int] = {}
    for previous, current in zip(notes, notes[1:]):
        delta = current["time"] - previous["time"]
        if delta < 0:
            continue
        interval_ms.append(delta)
        beat_length = _active_beat_length(red_points, previous["time"])
        if beat_length:
            beats = delta / beat_length
            interval_beats.append(beats)
            snap = _nearest_snap(beats)
            key = str(snap)
            snap_counts[key] = snap_counts.get(key, 0) + 1

    ka_count = sum(1 for note in notes if note["hitsound"] & 8)
    big_count = sum(1 for note in notes if note["hitsound"] & 4)
    fast_count = sum(1 for beats in interval_beats if beats <= 0.25 + 1e-6)
    half_or_faster = sum(1 for beats in interval_beats if beats <= 0.5 + 1e-6)

    return {
        "beatmap_id": metadata.get("BeatmapID"),
        "beatmap_set_id": metadata.get("BeatmapSetID"),
        "title": metadata.get("Title"),
        "artist": metadata.get("Artist"),
        "creator": metadata.get("Creator"),
        "version": metadata.get("Version"),
        "difficulty_mapper": difficulty_mapper(
            metadata.get("Creator"), metadata.get("Version")
        ),
        "difficulty": difficulty,
        "audio_filename": general.get("AudioFilename"),
        "note_count": len(notes),
        "duration_s": duration_s,
        "notes_per_second": (len(notes) / duration_s) if duration_s else 0.0,
        "median_ioi_ms": statistics.median(interval_ms) if interval_ms else None,
        "ka_ratio": (ka_count / len(notes)) if notes else 0.0,
        "big_note_ratio": (big_count / len(notes)) if notes else 0.0,
        "quarter_beat_or_faster_share": (
            fast_count / len(interval_beats) if interval_beats else 0.0
        ),
        "half_beat_or_faster_share": (
            half_or_faster / len(interval_beats) if interval_beats else 0.0
        ),
        "snap_counts": snap_counts,
    }


def _safe_member(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    return candidate


def import_osz(osz_path: str | Path, output_root: str | Path) -> dict:
    source = Path(osz_path)
    output_root = Path(output_root)
    maps = []
    taiko_sources = []

    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".osu"):
                continue
            text = archive.read(name).decode("utf-8-sig", errors="replace")
            parsed = parse_osu(text)
            if parsed["general"].get("Mode") != "1":
                continue
            metrics = analyze_taiko_map(parsed)
            taiko_sources.append((name, text, parsed))
            maps.append(metrics)

        if not maps:
            raise ValueError(f"No osu!taiko maps found in {source}")

        set_id = next(
            (item.get("beatmap_set_id") for item in maps if item.get("beatmap_set_id")),
            source.stem,
        )
        set_dir = output_root / str(set_id)
        map_dir = set_dir / "maps"
        map_dir.mkdir(parents=True, exist_ok=True)

        for name, text, _parsed in taiko_sources:
            safe = _safe_member(name).name
            (map_dir / safe).write_text(text, encoding="utf-8")

        audio_names = {
            item["audio_filename"] for item in maps if item.get("audio_filename")
        }
        extracted_audio = []
        members = set(archive.namelist())
        for audio_name in sorted(audio_names):
            safe = _safe_member(audio_name)
            if audio_name not in members:
                continue
            target = set_dir / safe.name
            with archive.open(audio_name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_audio.append(target.name)

    maps.sort(
        key=lambda item: (
            DIFFICULTY_ORDER.get(item.get("difficulty"), 99),
            item.get("version") or "",
        )
    )
    oni = next((m for m in maps if m.get("difficulty") == "oni"), None)
    if oni and oni["notes_per_second"] > 0:
        for item in maps:
            item["density_ratio_to_oni"] = (
                item["notes_per_second"] / oni["notes_per_second"]
            )

    standard_difficulties = {
        item["difficulty"] for item in maps if item.get("difficulty") is not None
    }
    if len(maps) == 1:
        reference_role = "style_exemplar"
    elif len(standard_difficulties) >= 2:
        reference_role = "difficulty_ladder"
    else:
        reference_role = "style_set"

    manifest = {
        "source_archive": source.name,
        "beatmap_set_id": str(set_id),
        "reference_role": reference_role,
        "audio_files": extracted_audio,
        "maps": maps,
    }
    (set_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _summarize_maps(items: Iterable[dict]) -> dict:
    by_difficulty: dict[str, list[dict]] = {}
    for item in items:
        difficulty = item.get("difficulty")
        if difficulty:
            by_difficulty.setdefault(difficulty, []).append(item)

    summary = {}
    for difficulty, difficulty_items in by_difficulty.items():
        ratios = [
            item["density_ratio_to_oni"]
            for item in difficulty_items
            if item.get("density_ratio_to_oni") is not None
        ]
        summary[difficulty] = {
            "maps": len(difficulty_items),
            "mean_notes_per_second": statistics.mean(
                item["notes_per_second"] for item in difficulty_items
            ),
            "mean_density_ratio_to_oni": statistics.mean(ratios) if ratios else None,
            "mean_quarter_beat_or_faster_share": statistics.mean(
                item["quarter_beat_or_faster_share"] for item in difficulty_items
            ),
            "mean_half_beat_or_faster_share": statistics.mean(
                item["half_beat_or_faster_share"] for item in difficulty_items
            ),
            "mean_ka_ratio": statistics.mean(
                item["ka_ratio"] for item in difficulty_items
            ),
            "mean_big_note_ratio": statistics.mean(
                item["big_note_ratio"] for item in difficulty_items
            ),
        }
    return {
        key: summary[key]
        for key in sorted(summary, key=lambda name: DIFFICULTY_ORDER.get(name, 99))
    }


def summarize_manifests(manifests: Iterable[dict]) -> dict:
    return _summarize_maps(
        item
        for manifest in manifests
        for item in manifest.get("maps", [])
    )


def summarize_mappers(manifests: Iterable[dict]) -> dict:
    """Build separate human-style profiles without blending mapper identities."""
    manifests = list(manifests)
    by_mapper: dict[str, list[dict]] = {}
    mapper_sets: dict[str, set[str]] = {}
    custom_counts: dict[str, int] = {}

    for manifest in manifests:
        set_id = str(manifest.get("beatmap_set_id", ""))
        for item in manifest.get("maps", []):
            mapper = item.get("difficulty_mapper")
            if not mapper:
                continue
            by_mapper.setdefault(mapper, []).append(item)
            mapper_sets.setdefault(mapper, set()).add(set_id)
            if item.get("difficulty") is None:
                custom_counts[mapper] = custom_counts.get(mapper, 0) + 1

    result = {}
    for mapper, items in sorted(by_mapper.items(), key=lambda pair: pair[0].lower()):
        result[mapper] = {
            "maps": len(items),
            "sets": len(mapper_sets.get(mapper, set())),
            "custom_maps": custom_counts.get(mapper, 0),
            "difficulty_profile": _summarize_maps(items),
        }
    return result
