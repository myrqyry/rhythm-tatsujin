"""Production audio-to-chart inference backed by the vendored SoftChart model.

The SoftChart source is vendored from a pinned public Hugging Face Space revision
under taikonation.vendor.softchart. This module owns the TaikoNation-facing
contract: raw audio in, timestamped Taiko events plus BPM out.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import torch

from taikonation.vendor.softchart.generate import generate_song, generate_song_slot, load_hf
from taikonation.vendor.softchart.grid import fit_grid_fixed_bpm, fit_grid_piecewise
from taikonation.vendor.softchart.vocab import HOP, N_FFT, N_MELS, SR

DEFAULT_MODEL_ID = "JacobLinCool/softchart-v17"
DEFAULT_LEVEL = {"easy": 3, "normal": 5, "hard": 7, "oni": 9, "ura": 10}
PAPER_MIN_HIT_GAP_SECONDS = 0.023


@dataclass(frozen=True)
class GeneratedChart:
    events: dict
    bpm: float
    timing_mode: str
    model_id: str
    device: str
    quality_score: float | None = None
    candidate_count: int = 1
    candidate_seed: int | None = None
    quality_neighbors: tuple[dict, ...] = ()
    candidate_scores: tuple[dict, ...] = ()


def preferred_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load_pretrained(model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
    resolved = device or preferred_device()
    return load_hf(model_id, device=resolved)


def load_log_mel(audio_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Match the pinned SoftChart v1.7 deployment preprocessing contract."""
    wav, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    if wav.size == 0:
        raise ValueError("Audio file decoded to an empty waveform")

    mel_filter = librosa.filters.mel(
        sr=SR,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=20.0,
        fmax=SR / 2,
    )
    window = torch.hann_window(N_FFT, periodic=True)
    spec = torch.stft(
        torch.from_numpy(wav),
        N_FFT,
        hop_length=HOP,
        window=window,
        center=True,
        return_complex=True,
    )
    mel = np.log(mel_filter @ spec.abs().pow(2).numpy() + 1e-5).astype(np.float32)
    return mel, wav


def _fallback_bpm(wav: np.ndarray) -> float:
    tempo = librosa.beat.beat_track(y=wav, sr=SR)[0]
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm <= 0:
        raise RuntimeError("Could not determine a usable BPM from the uploaded audio")
    return bpm


def postprocess_events(
    events: dict,
    *,
    min_hit_gap_seconds: float = PAPER_MIN_HIT_GAP_SECONDS,
) -> tuple[dict, dict[str, int]]:
    """Apply a minimal Taiko-specific quality guard without reshaping patterns.

    The original TaikoNation generation pipeline removed the later note when two
    discrete positives landed only one 23 ms frame apart. Keep that narrow rule
    here as a final sanity pass around the newer pretrained generator.
    """
    cleaned = dict(events)
    valid_hits: list[dict] = []
    invalid_hits_removed = 0

    for hit in events.get("hits", []):
        try:
            t = float(hit["t"])
        except (KeyError, TypeError, ValueError):
            invalid_hits_removed += 1
            continue
        if not np.isfinite(t) or t < 0:
            invalid_hits_removed += 1
            continue
        valid_hits.append(hit)

    valid_hits.sort(key=lambda hit: float(hit["t"]))
    kept_hits: list[dict] = []
    close_hits_removed = 0
    for hit in valid_hits:
        if kept_hits:
            delta = float(hit["t"]) - float(kept_hits[-1]["t"])
            if delta <= min_hit_gap_seconds + 1e-9:
                close_hits_removed += 1
                continue
        kept_hits.append(hit)

    cleaned["hits"] = kept_hits
    report = {
        "close_hits_removed": close_hits_removed,
        "invalid_hits_removed": invalid_hits_removed,
    }
    return cleaned, report


def generate_from_audio(
    audio_path: str | Path,
    *,
    course: str = "oni",
    level: int | None = None,
    bpm: float | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
    greedy: bool = True,
    seed: int = 0,
    temperature: float = 1.0,
    top_p: float = 0.95,
    on_progress: Callable[[int, str], None] | None = None,
) -> GeneratedChart:
    course = course.lower()
    if course not in DEFAULT_LEVEL:
        raise ValueError(f"Unsupported difficulty {course!r}; choose one of {sorted(DEFAULT_LEVEL)}")

    resolved = device or preferred_device()
    if on_progress:
        on_progress(5, "Loading pretrained model")
    model = load_pretrained(model_id, resolved)
    mel, wav = load_log_mel(audio_path)
    chart_level = int(level or DEFAULT_LEVEL[course])

    if on_progress:
        on_progress(15, "Analyzed audio")

    grid = None
    if bpm is not None and bpm > 0:
        candidate = fit_grid_fixed_bpm(model, mel, float(bpm), device=resolved)
        if candidate is not None and candidate.get("ok"):
            grid = candidate
    else:
        candidate = fit_grid_piecewise(model, mel, device=resolved)
        if candidate is not None and candidate.get("ok"):
            grid = candidate

    if on_progress:
        on_progress(30, "Estimated timing grid")

    def generation_progress(done: int, total: int) -> None:
        if on_progress:
            total = max(int(total), 1)
            done = max(0, min(int(done), total))
            percent = 30 + int(55 * done / total)
            on_progress(percent, f"Generating section {done}/{total}")

    if grid is not None:
        events = generate_song_slot(
            model,
            mel,
            grid,
            course,
            level=chart_level,
            greedy=greedy,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            device=resolved,
            on_progress=generation_progress,
        )
        resolved_bpm = float(bpm or grid["bpm"])
        timing_mode = "slot-exact"
    else:
        events = generate_song(
            model,
            mel,
            course,
            level=chart_level,
            greedy=greedy,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            device=resolved,
            on_progress=generation_progress,
        )
        resolved_bpm = float(bpm or _fallback_bpm(wav))
        timing_mode = "time"

    events, quality = postprocess_events(events)
    removed = quality["close_hits_removed"] + quality["invalid_hits_removed"]

    if on_progress:
        if removed:
            on_progress(90, f"Finalized chart events ({removed} implausible hit(s) removed)")
        else:
            on_progress(90, "Finalized chart events")

    return GeneratedChart(
        events=events,
        bpm=resolved_bpm,
        timing_mode=timing_mode,
        model_id=model_id,
        device=resolved,
    )


def generate_best_of_audio(
    audio_path: str | Path,
    *,
    course: str = "oni",
    level: int | None = None,
    bpm: float | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
    candidate_count: int = 4,
    reference_root: str | Path = "output/reference_corpus",
    mapper: str | None = None,
    sample_temperature: float = 0.65,
    sample_top_p: float = 0.90,
    on_progress: Callable[[int, str], None] | None = None,
) -> GeneratedChart:
    """Generate a deterministic baseline plus sampled alternatives and rerank them.

    If the local human-reference scorer is unavailable, return the baseline chart
    immediately instead of spending time on sampled candidates.
    """
    from taikonation.generation.quality import score_generated_events

    candidate_count = max(1, int(candidate_count))

    def scaled_progress(index: int):
        def report(percent: int, message: str) -> None:
            if not on_progress:
                return
            start = int(index * 88 / candidate_count)
            span = max(1, int(88 / candidate_count))
            scaled = min(88, start + int(span * max(0, min(percent, 100)) / 100))
            on_progress(
                scaled,
                f"Candidate {index + 1}/{candidate_count}: {message}",
            )
        return report

    baseline = generate_from_audio(
        audio_path,
        course=course,
        level=level,
        bpm=bpm,
        model_id=model_id,
        device=device,
        greedy=True,
        seed=0,
        on_progress=scaled_progress(0),
    )

    try:
        baseline_score = score_generated_events(
            baseline.events,
            bpm=baseline.bpm,
            difficulty=course,
            reference_root=reference_root,
            mapper=mapper,
        )
    except Exception:
        if on_progress:
            on_progress(90, "Human reference scorer unavailable; using deterministic chart")
        return baseline

    best_chart = baseline
    best_score = baseline_score
    best_seed = 0
    candidate_scores = [
        {"seed": 0, "greedy": True, "score": baseline_score.score}
    ]

    for index in range(1, candidate_count):
        seed = 2000 + index
        candidate = generate_from_audio(
            audio_path,
            course=course,
            level=level,
            bpm=bpm,
            model_id=model_id,
            device=device,
            greedy=False,
            seed=seed,
            temperature=sample_temperature,
            top_p=sample_top_p,
            on_progress=scaled_progress(index),
        )
        try:
            candidate_score = score_generated_events(
                candidate.events,
                bpm=candidate.bpm,
                difficulty=course,
                reference_root=reference_root,
                mapper=mapper,
            )
        except Exception:
            continue
        candidate_scores.append(
            {"seed": seed, "greedy": False, "score": candidate_score.score}
        )
        if candidate_score.score < best_score.score:
            best_chart = candidate
            best_score = candidate_score
            best_seed = seed

    if on_progress:
        on_progress(
            92,
            (
                f"Selected best of {candidate_count} candidates "
                f"(human-reference score {best_score.score:.3f})"
            ),
        )

    return replace(
        best_chart,
        quality_score=best_score.score,
        candidate_count=candidate_count,
        candidate_seed=best_seed,
        quality_neighbors=best_score.neighbors,
        candidate_scores=tuple(candidate_scores),
    )
