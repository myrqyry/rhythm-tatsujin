"""osu!taiko export for timestamped generated events."""

from __future__ import annotations

from pathlib import Path

from taikonation.utils.file_utils import atomic_write


def _hit_sound(note_type: str) -> int:
    if note_type == "ka":
        return 8
    if note_type == "don_big":
        return 4
    if note_type == "ka_big":
        return 12
    return 0


def render_osu(
    generated: dict,
    *,
    audio_filename: str,
    title: str,
    artist: str,
    difficulty: str,
    bpm: float,
    source: str = "",
    tags: str = "",
    creator: str = "TaikoNation",
    hitsound_volume: int = 70,
) -> str:
    """Render SoftChart-style timestamped hits/spans as an osu!taiko map."""
    bpm = max(float(bpm), 1.0)
    beat_length = 60000.0 / bpm
    slider_multiplier = 1.4
    hitsound_volume = max(0, min(int(hitsound_volume), 100))
    hit_sample = f"3:3:0:{hitsound_volume}:"

    objects: list[tuple[int, str]] = []

    for hit in generated.get("hits", []):
        note_type = str(hit.get("type", "don"))
        time_ms = max(0, int(round(float(hit["t"]) * 1000.0)))
        objects.append(
            (time_ms, f"256,192,{time_ms},1,{_hit_sound(note_type)},{hit_sample}")
        )

    for span in generated.get("spans", []):
        span_type = str(span.get("type", "roll"))
        start_ms = max(0, int(round(float(span["t0"]) * 1000.0)))
        end_ms = max(start_ms + 1, int(round(float(span["t1"]) * 1000.0)))
        if span_type == "balloon":
            objects.append(
                (start_ms, f"256,192,{start_ms},8,0,{end_ms},{hit_sample}")
            )
            continue

        duration_ms = end_ms - start_ms
        length = max((duration_ms * slider_multiplier * 100.0) / beat_length, 1.0)
        hit_sound = 4 if span_type == "roll_big" else 0
        objects.append(
            (
                start_ms,
                f"256,192,{start_ms},2,{hit_sound},B|256:192,1,{length:.2f},0|0,3:3|3:3,{hit_sample}",
            )
        )

    objects.sort(key=lambda item: item[0])
    version = difficulty.capitalize()
    header = f"""osu file format v14

[General]
AudioFilename: {Path(audio_filename).name}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Drum
StackLeniency: 0.7
Mode: 1
LetterboxInBreaks: 0

[Editor]
DistanceSpacing: 1
BeatDivisor: 4
GridSize: 4
TimelineZoom: 1

[Metadata]
Title:{title}
Artist:{artist}
Creator:{creator}
Version:{version}
Source:{source}
Tags:{tags}

[Difficulty]
HPDrainRate:5
CircleSize:5
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:{slider_multiplier}
SliderTickRate:1

[Events]
//Background and Video events

[TimingPoints]
0,{beat_length:.6f},4,3,1,{hitsound_volume},1,0

[HitObjects]
"""
    return header + "\n".join(line for _, line in objects) + ("\n" if objects else "")


def save_timestamped_osu(
    generated: dict,
    output_path: str | Path,
    *,
    audio_filename: str,
    title: str,
    artist: str,
    difficulty: str,
    bpm: float,
    source: str = "",
    tags: str = "",
    creator: str = "TaikoNation",
    hitsound_volume: int = 70,
) -> Path:
    path = Path(output_path)
    content = render_osu(
        generated,
        audio_filename=audio_filename,
        title=title,
        artist=artist,
        difficulty=difficulty,
        bpm=bpm,
        source=source,
        tags=tags,
        creator=creator,
        hitsound_volume=hitsound_volume,
    )
    with atomic_write(path, "w") as handle:
        handle.write(content)
    return path
