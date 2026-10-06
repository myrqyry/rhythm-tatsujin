"""Human-reference candidate scoring for production chart generation.

This module deliberately scores plausibility, not "quality truth". It measures how
close a generated chart is to the local same-difficulty human reference corpus in
an explainable, robust feature space. Lower scores are better.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import statistics
from typing import Iterable

from taikonation.generation.osu_export import render_osu
from taikonation.reference import analyze_taiko_map, parse_osu


DEFAULT_REFERENCE_ROOT = Path("output/reference_corpus")

# One representative per feature family so correlated measurements do not get
# accidental extra weight merely because several variants exist.
QUALITY_FEATURES = (
    "notes_per_second",
    "median_ioi_ms",
    "ka_ratio",
    "big_note_ratio",
    "color_change_ratio",
    "repeated_motif_8_share",
    "subdivision_entropy",
    "mean_burst_length",
    "rest_interval_share",
    "section_density_change",
)


@dataclass(frozen=True)
class CandidateScore:
    score: float
    metrics: dict[str, float]
    neighbors: tuple[dict, ...]
    reference_count: int
    k: int


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _robust_center_scale(values: Iterable[float]) -> tuple[float, float]:
    items = [float(value) for value in values]
    center = statistics.median(items)
    deviations = [abs(value - center) for value in items]
    mad = statistics.median(deviations) if deviations else 0.0
    scale = 1.4826 * mad
    if scale <= 1e-9 and len(items) > 1:
        scale = statistics.stdev(items)
    if scale <= 1e-9:
        scale = 1.0
    return center, scale


@lru_cache(maxsize=32)
def _reference_rows(
    root_text: str,
    difficulty: str,
    mapper: str | None,
) -> tuple[dict, ...]:
    root = Path(root_text)
    rows: list[dict] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in manifest.get("maps", []):
            if item.get("difficulty") != difficulty:
                continue
            if mapper and item.get("difficulty_mapper") != mapper:
                continue
            metrics = {}
            valid = True
            for feature in QUALITY_FEATURES:
                value = _finite_number(item.get(feature))
                if value is None:
                    valid = False
                    break
                metrics[feature] = value
            if not valid:
                continue
            rows.append(
                {
                    "beatmap_set_id": str(item.get("beatmap_set_id") or ""),
                    "title": item.get("title") or "",
                    "version": item.get("version") or "",
                    "difficulty_mapper": item.get("difficulty_mapper") or "",
                    "metrics": metrics,
                }
            )
    return tuple(rows)


def clear_reference_cache() -> None:
    _reference_rows.cache_clear()


def score_metrics_against_reference(
    metrics: dict,
    *,
    difficulty: str,
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
    mapper: str | None = None,
    exclude_set_id: str | None = None,
) -> CandidateScore:
    rows = tuple(
        row
        for row in _reference_rows(str(Path(reference_root)), difficulty, mapper)
        if exclude_set_id is None or row["beatmap_set_id"] != str(exclude_set_id)
    )
    if len(rows) < 5:
        raise ValueError(
            f"Need at least 5 human {difficulty!r} reference maps; found {len(rows)}"
        )

    candidate: dict[str, float] = {}
    for feature in QUALITY_FEATURES:
        value = _finite_number(metrics.get(feature))
        if value is None:
            raise ValueError(f"Candidate is missing usable feature {feature!r}")
        candidate[feature] = value

    transforms = {
        feature: _robust_center_scale(
            row["metrics"][feature] for row in rows
        )
        for feature in QUALITY_FEATURES
    }

    def normalized_distance(row: dict) -> float:
        squared = 0.0
        for feature in QUALITY_FEATURES:
            center, scale = transforms[feature]
            candidate_z = (candidate[feature] - center) / scale
            reference_z = (row["metrics"][feature] - center) / scale
            # Prevent one pathological axis from completely dominating the
            # neighborhood while still penalizing clear outliers.
            delta = max(-4.0, min(4.0, candidate_z - reference_z))
            squared += delta * delta
        return math.sqrt(squared / len(QUALITY_FEATURES))

    ranked = sorted(
        ((normalized_distance(row), row) for row in rows),
        key=lambda pair: pair[0],
    )
    k = max(3, min(7, int(round(math.sqrt(len(rows))))))
    nearest = ranked[:k]
    score = statistics.mean(distance for distance, _ in nearest)
    neighbors = tuple(
        {
            "distance": distance,
            "beatmap_set_id": row["beatmap_set_id"],
            "title": row["title"],
            "version": row["version"],
            "difficulty_mapper": row["difficulty_mapper"],
        }
        for distance, row in nearest
    )
    return CandidateScore(
        score=score,
        metrics=candidate,
        neighbors=neighbors,
        reference_count=len(rows),
        k=k,
    )


def score_generated_events(
    events: dict,
    *,
    bpm: float,
    difficulty: str,
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
    mapper: str | None = None,
    exclude_set_id: str | None = None,
) -> CandidateScore:
    text = render_osu(
        events,
        audio_filename="candidate.wav",
        title="Candidate",
        artist="Rhythm Tatsujin",
        difficulty=difficulty,
        bpm=bpm,
        creator="Rhythm Tatsujin",
    )
    metrics = analyze_taiko_map(parse_osu(text))
    return score_metrics_against_reference(
        metrics,
        difficulty=difficulty,
        reference_root=reference_root,
        mapper=mapper,
        exclude_set_id=exclude_set_id,
    )
