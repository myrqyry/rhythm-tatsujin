"""Deterministic modal-resonator Taiko preview mixing.

The chart is the timing authority: :func:`parse_tja_events` converts the
repo-emitted TJA subset to audio-clock events, including OFFSET, measure-local
subdivisions, and piecewise BPM changes. Drum timbres are three bundled,
hash-validated synthesized voices: Don, Big Don, and Katsu.  Private reference
audio is neither bundled nor required at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


_HIT_KIND = {"1": "don", "2": "ka", "3": "don_big", "4": "ka_big"}
_SPAN_KIND = {"5": "roll", "6": "roll_big", "7": "balloon"}
_ALLOWED_COMMANDS = {"#GOGOSTART", "#GOGOEND"}
_PCM16_CEILING = int(math.floor(32767 * 10.0 ** (-1.0 / 20.0)))
_ASSET_DIR = Path(__file__).with_name("assets") / "taiko_synth"


@dataclass(frozen=True)
class TaikoEvent:
    """One chart event on the source-audio clock.

    ``duration`` is non-zero for rolls and balloons.  ``hit_count`` is one for
    ordinary notes, the BALLOON header value for balloons, and zero for rolls
    whose autoplay cadence is derived from their duration.
    """

    time: float
    kind: str
    duration: float = 0.0
    hit_count: int = 1


@dataclass(frozen=True)
class _OpenSpan:
    time: float
    kind: str
    hit_count: int


@dataclass(frozen=True)
class _Strike:
    time: float
    voice_kind: str
    pan: float
    gain: float
    tail_seconds: float | None
    duck_depth: float
    autoplay: bool


@dataclass(frozen=True)
class _Voice:
    samples: np.ndarray
    anchor_frame: int


def _finite_float(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TJA {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"TJA {field} must be finite")
    return result


def _strip_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def parse_tja_events(tja_text: str) -> list[TaikoEvent]:
    """Parse the single-course TJA subset emitted by SoftChart.

    Event times use the source-audio clock.  TJA's OFFSET convention therefore
    becomes ``audio_time = score_time - OFFSET``.  Every measure is 4/4, and a
    measure's character count defines its exact subdivision.  ``#BPMCHANGE``
    is accepted at measure boundaries, matching :mod:`softchart.tja`.

    The parser is intentionally strict.  A malformed/unterminated span or an
    unsupported timing command raises ``ValueError`` instead of producing an
    audibly plausible but incorrectly synchronized preview.
    """

    if not isinstance(tja_text, str) or not tja_text.strip():
        raise ValueError("TJA text must be a non-empty string")

    lines = [_strip_comment(line) for line in tja_text.splitlines()]
    headers: dict[str, str] = {}
    start_index: int | None = None
    for index, line in enumerate(lines):
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("#START"):
            start_index = index + 1
            break
        if line.startswith("#"):
            raise ValueError(f"unsupported command before #START: {line}")
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().upper()] = value.strip()

    if start_index is None:
        raise ValueError("TJA chart is missing #START")
    if "BPM" not in headers:
        raise ValueError("TJA chart is missing BPM")
    if "OFFSET" not in headers:
        raise ValueError("TJA chart is missing OFFSET")

    bpm = _finite_float(headers["BPM"], "BPM")
    offset = _finite_float(headers["OFFSET"], "OFFSET")
    if bpm <= 0.0:
        raise ValueError("TJA BPM must be greater than zero")

    balloon_counts: list[int] = []
    balloon_field = headers.get("BALLOON", "").strip()
    if balloon_field:
        try:
            balloon_counts = [int(part.strip()) for part in balloon_field.split(",")]
        except ValueError as exc:
            raise ValueError("TJA BALLOON values must be integers") from exc
        if any(count <= 0 for count in balloon_counts):
            raise ValueError("TJA BALLOON values must be greater than zero")

    score_time = 0.0
    measure_buffer: list[str] = []
    events: list[TaikoEvent] = []
    open_span: _OpenSpan | None = None
    balloon_index = 0
    measure_count = 0
    found_end = False

    def finish_measure() -> None:
        nonlocal score_time, open_span, balloon_index, measure_count
        if not measure_buffer:
            raise ValueError("TJA contains an empty measure")
        measure = "".join(measure_buffer)
        measure_buffer.clear()
        division = len(measure)
        measure_seconds = 240.0 / bpm

        for position, symbol in enumerate(measure):
            if symbol == "0":
                continue
            event_time = score_time + measure_seconds * position / division - offset
            if symbol in _HIT_KIND:
                events.append(TaikoEvent(time=event_time, kind=_HIT_KIND[symbol]))
            elif symbol in _SPAN_KIND:
                if open_span is not None:
                    raise ValueError("TJA contains overlapping roll/balloon spans")
                hit_count = 0
                if symbol == "7":
                    if balloon_index >= len(balloon_counts):
                        raise ValueError("TJA balloon has no matching BALLOON count")
                    hit_count = balloon_counts[balloon_index]
                    balloon_index += 1
                open_span = _OpenSpan(event_time, _SPAN_KIND[symbol], hit_count)
            elif symbol == "8":
                if open_span is None:
                    raise ValueError("TJA contains a span end (8) without a start")
                duration = event_time - open_span.time
                if duration <= 0.0:
                    raise ValueError("TJA roll/balloon duration must be positive")
                events.append(TaikoEvent(open_span.time, open_span.kind,
                                         duration, open_span.hit_count))
                open_span = None
            else:
                raise ValueError(f"unsupported TJA note symbol: {symbol}")

        score_time += measure_seconds
        measure_count += 1

    for line in lines[start_index:]:
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("#END"):
            if measure_buffer:
                raise ValueError("TJA measure before #END is missing a comma")
            found_end = True
            break
        if upper.startswith("#BPMCHANGE"):
            if measure_buffer:
                raise ValueError("#BPMCHANGE is only supported at measure boundaries")
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError("#BPMCHANGE requires a BPM value")
            bpm = _finite_float(parts[1], "#BPMCHANGE")
            if bpm <= 0.0:
                raise ValueError("#BPMCHANGE BPM must be greater than zero")
            continue
        if upper in _ALLOWED_COMMANDS:
            if measure_buffer:
                raise ValueError(f"{line} is only supported at measure boundaries")
            continue
        if line.startswith("#"):
            raise ValueError(f"unsupported command in preview TJA: {line}")

        for char in line:
            if char.isspace():
                continue
            if char == ",":
                finish_measure()
            elif char in "012345678":
                measure_buffer.append(char)
            else:
                raise ValueError(f"unsupported character in TJA measure: {char!r}")

    if not found_end:
        raise ValueError("TJA chart is missing #END")
    if measure_count == 0:
        raise ValueError("TJA chart has no measures")
    if open_span is not None:
        raise ValueError("TJA roll/balloon span is missing its end marker (8)")
    return sorted(events, key=lambda event: event.time)


_BANKS: dict[int, dict[str, tuple[_Voice, ...]]] = {}
_REQUIRED_VOICES = {"don": 1, "don_big": 1, "katsu": 1}


def _measurable_anchor(samples: np.ndarray) -> int:
    peak = float(np.max(np.abs(samples)))
    if not math.isfinite(peak) or peak <= 1e-12:
        raise ValueError("Taiko asset must contain finite, non-silent audio")
    candidates = np.flatnonzero(np.abs(samples) >= peak * 10.0 ** (-30.0 / 20.0))
    if len(candidates) == 0:
        raise ValueError("Taiko asset has no measurable onset")
    return int(candidates[0])


def _load_pcm16_voice(
    path: Path,
    anchor_frame: int,
    sample_rate: int,
    source_rate: int,
    expected_frames: int,
    expected_pcm_hash: str,
) -> _Voice:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"Taiko asset must be mono PCM16: {path.name}")
        actual_rate = source.getframerate()
        actual_frames = source.getnframes()
        pcm = np.frombuffer(source.readframes(actual_frames), dtype="<i2")
    if actual_rate != source_rate:
        raise ValueError(f"Taiko asset sample-rate mismatch: {path.name}")
    if actual_frames != expected_frames:
        raise ValueError(f"Taiko asset frame-count mismatch: {path.name}")
    if hashlib.sha256(pcm.tobytes()).hexdigest() != expected_pcm_hash:
        raise ValueError(f"Taiko asset PCM hash mismatch: {path.name}")
    samples = pcm.astype(np.float32) / 32767.0
    measured_anchor = _measurable_anchor(samples)
    if measured_anchor != anchor_frame:
        raise ValueError(f"Taiko asset onset-anchor mismatch: {path.name}")
    if sample_rate != source_rate:
        ratio = Fraction(sample_rate, source_rate).limit_denominator()
        samples = resample_poly(samples, ratio.numerator, ratio.denominator).astype(np.float32)
        anchor_frame = _measurable_anchor(samples)
    if not 0 <= anchor_frame < len(samples):
        raise ValueError(f"Taiko asset anchor is outside the sample: {path.name}")
    return _Voice(np.ascontiguousarray(samples), anchor_frame)


def _sample_bank(sample_rate: int) -> dict[str, tuple[_Voice, ...]]:
    bank = _BANKS.get(sample_rate)
    if bank is not None:
        return bank

    manifest_path = _ASSET_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Taiko asset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("license") != "MIT":
        raise ValueError("Taiko synth manifest must be schema 2 and MIT licensed")
    source_rate = int(manifest.get("sample_rate", 0))
    if source_rate != 48_000:
        raise ValueError("Taiko synth assets must be generated at 48 kHz")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("Taiko synth manifest files must be a list")
    listed_files = {item.get("filename") for item in entries if isinstance(item, dict)}
    bundled_files = {path.name for path in _ASSET_DIR.glob("*.wav")}
    if listed_files != bundled_files:
        raise ValueError("Taiko synth manifest must cover every bundled WAV exactly")

    grouped: dict[str, list[tuple[int, _Voice]]] = {kind: [] for kind in _REQUIRED_VOICES}
    for item in entries:
        kind = item.get("kind")
        if kind not in grouped:
            raise ValueError(f"unsupported Taiko asset kind in manifest: {kind!r}")
        filename = item.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Taiko asset filename must be a plain basename")
        path = _ASSET_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Taiko asset is missing: {path}")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if file_hash != item.get("sha256"):
            raise ValueError(f"Taiko asset hash mismatch: {filename}")
        if item.get("sample_rate") != source_rate or item.get("channels") != 1:
            raise ValueError(f"Taiko asset format metadata mismatch: {filename}")
        voice = _load_pcm16_voice(
            path,
            int(item["anchor_frame"]),
            sample_rate,
            source_rate,
            int(item["frames"]),
            str(item["pcm_sha256"]),
        )
        grouped[kind].append((1, voice))

    bank = {}
    for kind, expected_count in _REQUIRED_VOICES.items():
        entries = sorted(grouped[kind], key=lambda entry: entry[0])
        if len(entries) != expected_count:
            raise ValueError(
                f"Taiko kit requires {expected_count} {kind} voice(s), got {len(entries)}"
            )
        bank[kind] = tuple(voice for _, voice in entries)
    _BANKS[sample_rate] = bank
    return bank


_DEFAULT_BANK = _sample_bank(44_100)


def _expand_strikes(events: list[TaikoEvent]) -> tuple[list[_Strike], int]:
    strikes: list[_Strike] = []
    autoplay_count = 0
    ordinary_index = 0
    for event in events:
        if event.kind in {"don", "ka", "don_big", "ka_big"}:
            pan_width = 0.08 if event.kind in {"ka", "ka_big"} else 0.015
            pan = 0.0 if event.kind == "don_big" else (
                -pan_width if ordinary_index % 2 == 0 else pan_width
            )
            voice_kind = {
                "don": "don",
                "ka": "katsu",
                "don_big": "don_big",
                "ka_big": "katsu",
            }[event.kind]
            gain = 1.12 if event.kind == "ka_big" else 1.0
            depth = 0.26 if event.kind.endswith("_big") else 0.22
            strikes.append(_Strike(
                event.time, voice_kind, pan, gain, None, depth, False,
            ))
            ordinary_index += 1
            continue

        if event.kind == "balloon":
            times = event.time + np.arange(event.hit_count, dtype=np.float64) * (
                event.duration / event.hit_count
            )
            voice_kind = "don"
            gain = 0.54
            tail_seconds = 0.135
            depth = 0.070
        else:
            interval = 0.074 if event.kind == "roll_big" else 0.086
            count = max(1, int(math.ceil(max(0.0, event.duration - 0.018) / interval)))
            times = event.time + np.arange(count, dtype=np.float64) * interval
            times = times[times < event.time + event.duration - 0.008]
            if len(times) == 0:
                times = np.asarray([event.time])
            voice_kind = "don_big" if event.kind == "roll_big" else "don"
            gain = 0.60 if event.kind == "roll_big" else 0.56
            tail_seconds = 0.175 if event.kind == "roll_big" else 0.150
            depth = 0.085 if event.kind == "roll_big" else 0.060
        for index, time in enumerate(times):
            pan_width = 0.10 if voice_kind == "don_big" else 0.06
            pan = -pan_width if index % 2 == 0 else pan_width
            strikes.append(_Strike(
                float(time), voice_kind, pan, gain, tail_seconds, depth, True,
            ))
            autoplay_count += 1
    strikes.sort(key=lambda strike: strike.time)
    return strikes, autoplay_count


def _load_stereo_audio(audio_path: str, sample_rate: int) -> np.ndarray:
    import soundfile as sf

    audio, source_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if source_rate <= 0 or audio.shape[0] == 0:
        raise ValueError("input audio must contain at least one sample")
    if not np.all(np.isfinite(audio)):
        raise ValueError("input audio contains non-finite samples")

    channels = audio.shape[1]
    if channels == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif channels > 2:
        ambience = np.mean(audio[:, 2:], axis=1)
        left = (audio[:, 0] + 0.5 * ambience) / 1.5
        right = (audio[:, 1] + 0.5 * ambience) / 1.5
        audio = np.stack((left, right), axis=1)
    else:
        audio = audio[:, :2]

    if source_rate != sample_rate:
        ratio = Fraction(sample_rate, int(source_rate)).limit_denominator()
        expected_frames = int(round(len(audio) * sample_rate / source_rate))
        audio = resample_poly(
            audio, ratio.numerator, ratio.denominator, axis=0,
            window=("kaiser", 10.0), padtype="line",
        )
        if len(audio) < expected_frames:
            audio = np.pad(audio, ((0, expected_frames - len(audio)), (0, 0)))
        audio = audio[:expected_frames]
    return np.ascontiguousarray(audio, dtype=np.float32)


def _place_strike(drums: np.ndarray, voice: _Voice, strike: _Strike,
                  sample_rate: int) -> bool:
    sample = voice.samples
    start = int(round(strike.time * sample_rate)) - voice.anchor_frame
    sample_start = max(0, -start)
    output_start = max(0, start)
    sample_end = len(sample)
    if strike.tail_seconds is not None:
        sample_end = min(
            sample_end,
            voice.anchor_frame + int(round(strike.tail_seconds * sample_rate)),
        )
    count = min(sample_end - sample_start, len(drums) - output_start)
    if count <= 0:
        return False

    angle = (float(np.clip(strike.pan, -1.0, 1.0)) + 1.0) * np.pi / 4.0
    left_gain, right_gain = math.cos(angle), math.sin(angle)
    segment = sample[sample_start:sample_start + count]
    if strike.tail_seconds is not None:
        source_indices = np.arange(sample_start, sample_start + count)
        active_frames = source_indices - voice.anchor_frame
        fade_start = int(round(strike.tail_seconds * sample_rate * 0.43))
        tail_frames = max(fade_start + 1, int(round(strike.tail_seconds * sample_rate)))
        fade_position = np.clip(
            (active_frames - fade_start) / (tail_frames - fade_start), 0.0, 1.0,
        )
        envelope = 0.5 + 0.5 * np.cos(np.pi * fade_position)
        envelope[active_frames < 0] = 1.0
        segment = segment * envelope.astype(np.float32)
    segment = segment * strike.gain
    drums[output_start:output_start + count, 0] += segment * left_gain
    drums[output_start:output_start + count, 1] += segment * right_gain
    return True


def _duck_envelope(frame_count: int, strikes: list[_Strike], sample_rate: int,
                   drum_gain: float) -> tuple[np.ndarray, float]:
    reduction = np.zeros(frame_count, dtype=np.float32)
    attack_frames = max(1, int(round(0.0025 * sample_rate)))
    release_frames = max(1, int(round(0.080 * sample_rate)))
    attack = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, attack_frames, dtype=np.float32))
    release = np.exp(-np.arange(release_frames, dtype=np.float32) /
                     max(1.0, 0.027 * sample_rate))
    shape = np.concatenate((attack, release))
    gain_scale = min(drum_gain / 0.8, 1.0) if drum_gain > 0.0 else 0.0

    for strike in strikes:
        hit_frame = int(round(strike.time * sample_rate))
        start = hit_frame - attack_frames
        source_start = max(0, -start)
        output_start = max(0, start)
        count = min(len(shape) - source_start, frame_count - output_start)
        if count <= 0:
            continue
        envelope = shape[source_start:source_start + count] * strike.duck_depth * gain_scale
        target = reduction[output_start:output_start + count]
        np.maximum(target, envelope, out=target)

    np.minimum(reduction, 0.38, out=reduction)
    gain = 1.0 - reduction
    max_duck_db = -20.0 * math.log10(max(float(np.min(gain)), 1e-12))
    return gain, max_duck_db


def _write_pcm16_wav(out_path: str, pcm: np.ndarray, sample_rate: int) -> None:
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with wave.open(out_path, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.ascontiguousarray(pcm.astype("<i2", copy=False)).tobytes())


def _soft_limit(signal: np.ndarray, threshold: float, ceiling: float) -> np.ndarray:
    """Limit only hot samples instead of turning down the entire preview."""

    magnitude = np.abs(signal)
    width = ceiling - threshold
    over = np.maximum(magnitude - threshold, 0.0) / width
    shaped = threshold + width * np.tanh(over)
    return np.copysign(np.where(magnitude <= threshold, magnitude, shaped), signal)


def synthesize_taiko_preview(
    audio_path: str,
    tja_text: str,
    out_path: str,
    drum_gain: float = 0.8,
    sample_rate: int = 44_100,
) -> dict[str, float | int]:
    """Mix a synchronized stereo PCM16 WAV preview and return numeric stats.

    Music is held at 3.5 dB of mix headroom and briefly ducked under each drum
    transient.  A local soft limiter controls only hot samples before PCM16
    quantization at a -1 dBFS sample-peak ceiling.
    """

    if not isinstance(sample_rate, int) or sample_rate < 16_000 or sample_rate > 192_000:
        raise ValueError("sample_rate must be an integer from 16000 to 192000")
    if not math.isfinite(drum_gain) or not 0.0 <= drum_gain <= 2.0:
        raise ValueError("drum_gain must be finite and between 0.0 and 2.0")
    if not out_path:
        raise ValueError("out_path must be provided")

    events = parse_tja_events(tja_text)
    music = _load_stereo_audio(audio_path, sample_rate)
    strikes, autoplay_count = _expand_strikes(events)
    bank = _DEFAULT_BANK if sample_rate == 44_100 else _sample_bank(sample_rate)
    source_frames = len(music)
    output_frames = source_frames
    for index, strike in enumerate(strikes):
        voices = bank[strike.voice_kind]
        voice = voices[index % len(voices)]
        start = int(round(strike.time * sample_rate)) - voice.anchor_frame
        voice_end = len(voice.samples)
        if strike.tail_seconds is not None:
            voice_end = min(
                voice_end,
                voice.anchor_frame + int(round(strike.tail_seconds * sample_rate)),
            )
        output_frames = max(
            output_frames,
            max(0, start + voice_end),
        )
    if output_frames > source_frames:
        music = np.pad(music, ((0, output_frames - source_frames), (0, 0)))
    drums = np.zeros_like(music)
    audible_strikes: list[_Strike] = []

    for index, strike in enumerate(strikes):
        voices = bank[strike.voice_kind]
        voice = voices[index % len(voices)]
        if _place_strike(drums, voice, strike, sample_rate):
            audible_strikes.append(strike)

    duck_gain, max_duck_db = _duck_envelope(
        len(music), audible_strikes, sample_rate, drum_gain,
    )
    music_gain = 10.0 ** (-3.5 / 20.0)
    mixed = music * (music_gain * duck_gain[:, None])
    mixed += drums * float(drum_gain)

    pre_limit_peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    float_ceiling = _PCM16_CEILING / 32767.0
    mixed = _soft_limit(mixed, threshold=0.68, ceiling=float_ceiling)
    np.clip(mixed, -float_ceiling, float_ceiling, out=mixed)
    limited_peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    limiter_gain = min(1.0, limited_peak / pre_limit_peak) if pre_limit_peak > 0.0 else 1.0
    pcm = np.rint(mixed * 32767.0).astype(np.int16)
    _write_pcm16_wav(out_path, pcm, sample_rate)

    pcm_peak = int(np.max(np.abs(pcm.astype(np.int32)))) if pcm.size else 0
    peak_dbfs = 20.0 * math.log10(pcm_peak / 32768.0) if pcm_peak else -120.0
    gain_reduction_db = -20.0 * math.log10(max(limiter_gain, 1e-12))
    audible_autoplay = sum(1 for strike in audible_strikes if strike.autoplay)
    return {
        "sample_rate": sample_rate,
        "frames": len(music),
        "duration_seconds": len(music) / sample_rate,
        "source_frames": source_frames,
        "event_count": len(events),
        "rendered_hit_count": len(audible_strikes),
        "autoplay_hit_count": audible_autoplay,
        "scheduled_autoplay_hit_count": autoplay_count,
        "peak_dbfs": peak_dbfs,
        "gain_reduction_db": gain_reduction_db,
        "max_duck_db": max_duck_db,
    }


__all__ = ["TaikoEvent", "parse_tja_events", "synthesize_taiko_preview"]
