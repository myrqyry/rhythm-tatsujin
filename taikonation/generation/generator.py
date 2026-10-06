# generate_chart.py
import argparse
import os
import torch
import yaml
import numpy as np

from taikonation.utils.quantization import apply_dynamic_quantization
from taikonation.models.transformer import TaikoTransformer
from taikonation.data.tokenization import TaikoTokenizer
from taikonation.data.audio_processing import get_audio_features
from taikonation.data.dataset import DIFFICULTY_MAP
from taikonation.data.mediapipe_audio import MediaPipeAudioAnalyzer
from taikonation.utils.file_utils import atomic_write

class EnhancedChartGenerator:
    """
    Chart generator using both standard audio features and MediaPipe classifications
    """

    def __init__(self, model, config_path='config/default.yaml', use_mediapipe=True):
        self.model = model
        self.use_mediapipe = use_mediapipe
        self.mediapipe_analyzer = MediaPipeAudioAnalyzer() if use_mediapipe else None
        self._config = load_config(config_path)
        self._data_config = getattr(model, "taikonation_data_config", self._config["data"])
        self._tokenizer = TaikoTokenizer(
            time_quantization=self._data_config["time_quantization_ms"],
            source_resolution=self._data_config["source_resolution_ms"],
        )

    def generate(self, audio_path, difficulty, mediapipe_data=None):
        """
        Generate chart with optional MediaPipe enhancement

        Args:
            audio_path: Path to audio file
            difficulty: Target difficulty level
            mediapipe_data: Optional dict with MediaPipe classifications
        """

        # Frame and normalize source features exactly as the checkpoint saw them.
        data_config = self._data_config
        mel_features = get_audio_features(
            audio_path,
            source_resolution_ms=data_config["source_resolution_ms"],
            frame_duration_ms=data_config["time_quantization_ms"],
        )
        if mel_features is None:
            raise ValueError("Could not frame input audio features")
        if data_config.get("feature_mean") is not None and data_config.get("feature_std") is not None:
            mean = np.asarray(data_config["feature_mean"], dtype=np.float32)
            std = np.maximum(np.asarray(data_config["feature_std"], dtype=np.float32), 1e-4)
            mel_features = (mel_features - mean) / std

        # Enhance with MediaPipe if available
        if mediapipe_data and self.use_mediapipe:
            features = self.mediapipe_analyzer.integrate_with_existing_features(
                mel_features,
                mediapipe_data['classifications']
            )

            # Detect sections for structural guidance
            sections = self.mediapipe_analyzer.detect_chart_sections(
                mediapipe_data['classifications']
            )

            # Use sections to guide generation
            chart = self.generate_with_sections(features, difficulty, sections)
        else:
            # Standard generation
            chart = self.generate_standard(mel_features, difficulty)

        return chart

    def generate_with_sections(self, features, difficulty, sections):
        """
        Generate chart respecting detected musical sections

        Percussion-heavy sections → more dense patterns
        Melodic sections → sparser, more melodic following
        """

        charts = []

        for section in sections:
            # Calculate frame indices at the checkpoint's actual model resolution.
            frame_seconds = float(self._data_config["time_quantization_ms"]) / 1000.0
            start_frame = int(section['start'] / frame_seconds)
            end_frame = int(section['end'] / frame_seconds)

            section_features = features[start_frame:end_frame]

            # Adjust generation parameters based on section type
            if section['type'] == 'instrumental':
                # Higher note density for percussion sections
                density_bias = 1.2
            else:
                # Lower density for melodic sections
                density_bias = 0.8

            section_chart = generate_chart(
                self.model,
                section_features,
                self._tokenizer,
                difficulty,
                self._config,
                'cpu',
                density_bias=density_bias
            )

            charts.append(section_chart)

        # Concatenate sections
        full_chart = np.concatenate(charts)

        return full_chart

    def generate_standard(self, features, difficulty):
        return generate_chart(self.model, features, self._tokenizer, difficulty, self._config, 'cpu')

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(checkpoint_path, config, device, quantize=False, allow_random=False):
    """Load a trained research checkpoint, reconstructing its saved architecture."""
    vocab_size = TaikoTokenizer().vocab_size
    num_difficulties = len(DIFFICULTY_MAP)
    num_genres = 10

    checkpoint = None
    model_config = dict(config['model'])
    max_sequence_length = int(config['data']['max_sequence_length'])
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            saved = dict(checkpoint['model_config'])
            num_genres = int(saved.pop('num_genres', num_genres))
            max_sequence_length = int(saved.pop('max_sequence_length', max_sequence_length))
            model_config.update(saved)
    elif not allow_random:
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Train a research model first or use the pretrained SoftChart backend."
        )

    model = TaikoTransformer(
        vocab_size=vocab_size,
        num_genres=num_genres,
        num_difficulties=num_difficulties,
        max_sequence_length=max_sequence_length,
        **model_config,
    ).to(device)

    if checkpoint:
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)

    if isinstance(checkpoint, dict):
        model.taikonation_data_config = dict(checkpoint.get("data_config", config["data"]))
    else:
        model.taikonation_data_config = dict(config["data"])

    model.eval()

    if quantize:
        # We can only quantize models on the CPU
        if device.type == 'cpu':
            model = apply_dynamic_quantization(model)
        else:
            print("Warning: Quantization is only supported on CPU. Skipping.")

    return model

from typing import Callable, Optional
import math

@torch.no_grad()
def generate_chart(
    model,
    audio_features,
    tokenizer,
    difficulty_id,
    config,
    device,
    temperature=1.0,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    density_bias=1.0,
    genre_id=0,
    initial_token_id=None,
):
    """
    Generate a chart with optional progress reporting.

    Args:
        progress_callback: Function to call with (progress_percent, status_message)
    """
    model.eval()
    encoder_input = torch.from_numpy(audio_features).float().unsqueeze(0).to(device)
    first_token = tokenizer.vocab["[CLS]"] if initial_token_id is None else int(initial_token_id)
    decoder_input = torch.tensor([[first_token]], dtype=torch.long).to(device)
    generated_tokens = []

    max_len = int(model.pos_encoder.pe.shape[1])
    target_steps = max(1, min(max_len, int(audio_features.shape[0])))

    if progress_callback:
        progress_callback(5, "Initialized generation pipeline")

    for step in range(target_steps):
        genre_id_tensor = torch.tensor([genre_id], dtype=torch.long).to(device)
        difficulty_tensor = torch.tensor([difficulty_id], dtype=torch.long).to(device)

        output_logits = model(encoder_input, decoder_input, genre_id_tensor, difficulty_tensor)
        next_token_logits = output_logits[:, -1, :]

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply density bias
        if density_bias != 1.0:
            note_tokens = [i for i, token in tokenizer.id_to_token.items() if token not in tokenizer.special_tokens]
            note_logits = next_token_logits[:, note_tokens]
            biased_note_logits = note_logits * density_bias
            next_token_logits[:, note_tokens] = biased_note_logits

        probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, 1).item()

        generated_tokens.append(next_token_id)
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
        decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)

        # Report progress every 10 steps
        if progress_callback and step % 10 == 0:
            progress = int(5 + (step / target_steps) * 85)
            progress_callback(progress, f"Generated {step}/{target_steps} tokens")

    if progress_callback:
        progress_callback(95, "Finalizing chart data")

    return generated_tokens

def save_osu_chart(
    token_ids,
    tokenizer,
    output_path,
    audio_filename,
    title=None,
    artist=None,
    source="",
    tags="",
    difficulty="normal",
    bpm=120.0,
    offset_ms=1000,
    hitsound_volume=70,
    time_quantization_ms=46.4,
):
    """Save generated fixed-time tokens to an osu!taiko file."""
    print(f"Saving chart to {output_path}...")

    # Infer title and artist from filename if not provided
    if title is None or artist is None:
        filename_no_ext = os.path.splitext(os.path.basename(audio_filename))[0]
        parts = filename_no_ext.split(' - ')
        if len(parts) == 2:
            inferred_artist, inferred_title = parts
            if artist is None:
                artist = inferred_artist
            if title is None:
                title = inferred_title
        else:
            if title is None:
                title = filename_no_ext
            if artist is None:
                artist = "Unknown Artist"

    beat_length = 60000.0 / max(float(bpm), 1.0)
    slider_velocity_multiplier = -100.0
    slider_multiplier = 1.4
    hitsound_volume = max(0, min(int(hitsound_volume), 100))
    osu_header = f"""osu file format v14
[General]
AudioFilename: {os.path.basename(audio_filename)}
AudioLeadIn: 0
Mode: 1
[Events]
//Background and Video events
[TimingPoints]
{offset_ms},{beat_length:.6f},4,2,1,70,1,0
{offset_ms},{slider_velocity_multiplier:.6f},4,2,1,70,0,0
[Metadata]
Title:{title}
Artist:{artist}
Creator:TaikoNationV1
Version:{str(difficulty).capitalize()}
Source:{source}
Tags:{tags}
[Difficulty]
HPDrainRate:5
CircleSize:5
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1
[HitObjects]
"""

    # Use the tokenizer to convert IDs back to human-readable token names
    token_names = tokenizer.detokenize(token_ids)
    token_names = normalize_export_tokens(token_names)

    try:
        with atomic_write(output_path, "w") as f:
            hitobject_lines = []

            # Token timestamps are fixed-time model frames, not beat subdivisions.
            # Derive every timestamp from the token index to avoid cumulative
            # rounding drift over a full song.
            quantum_ms = max(float(time_quantization_ms), 1e-3)

            roll_active = False
            roll_start_time = int(offset_ms)

            for step_index, token_name in enumerate(token_names):
                current_time = int(round(float(offset_ms) + step_index * quantum_ms))

                if token_name in tokenizer.special_tokens or token_name == "[EMPTY]":
                    continue

                if "roll_start" in token_name:
                    roll_active = True
                    roll_start_time = current_time
                    continue

                if "roll_end" in token_name and roll_active:
                    roll_duration = max(current_time - roll_start_time, int(round(quantum_ms)))
                    # slider length formula approximation:
                    # duration_ms ≈ (pixel_length * beat_length) / (slider_multiplier * 100)
                    slider_length = max((roll_duration * slider_multiplier * 100.0) / beat_length, 1.0)
                    hit_sample = f"1:0:0:{hitsound_volume}:"
                    hitobject_lines.append((
                        roll_start_time,
                        f"256,192,{roll_start_time},2,0,B|256:192,1,{slider_length:.2f},0|0,0:0|0:0,{hit_sample}\n"
                    ))
                    roll_active = False
                    continue

                if roll_active:
                    continue

                if "finisher" in token_name:
                    spinner_end = current_time + max(int(round(quantum_ms * 2.0)), 1)
                    hit_sample = f"1:0:0:{hitsound_volume}:"
                    hitobject_lines.append((
                        current_time,
                        f"256,192,{current_time},8,0,{spinner_end},{hit_sample}\n"
                    ))
                    continue

                # x,y,time,type,hitSound,objectParams,hitSample
                # For Taiko, x is always 256, y is always 192.
                note_type = 1

                # hitSound is a bitfield: 0=normal, 2=whistle, 4=finish, 8=clap.
                hit_sound = 0
                if "ka" in token_name:
                    hit_sound |= 8  # Clap for ka
                if "big" in token_name:
                    hit_sound |= 4  # Finish for big notes
                addition_set = 2 if "ka" in token_name else 0
                hit_sample = f"1:{addition_set}:0:{hitsound_volume}:"

                hitobject_lines.append((
                    current_time,
                    f"256,192,{current_time},{note_type},{hit_sound},{hit_sample}\n"
                ))

            hitobject_lines.sort(key=lambda item: item[0])
            content = osu_header + "".join(line for _, line in hitobject_lines)
            export_issues = validate_exported_osu(content)
            for issue in export_issues:
                print(f"Warning: export validation - {issue}")
            f.write(content)
        print("Chart saved successfully.")
    except IOError as e:
        print(f"Error saving chart file: {e}")


def normalize_export_tokens(token_names):
    """Normalize token stream for safer `.osu` object emission.

    - drops orphan `roll_end` tokens (no active roll to close)
    - auto-appends `roll_end` if stream ends with an open roll
    """
    normalized = []
    roll_depth = 0
    for token in token_names:
        if "roll_start" in token:
            if roll_depth > 0:
                # osu!taiko rolls do not nest; keep the first open span.
                continue
            roll_depth = 1
            normalized.append(token)
            continue
        if "roll_end" in token:
            if roll_depth <= 0:
                continue
            roll_depth = 0
            normalized.append(token)
            continue
        normalized.append(token)

    while roll_depth > 0:
        normalized.append("roll_end")
        roll_depth -= 1
        print("Warning: Auto-appended roll_end token during export normalization.")

    return normalized


def validate_exported_osu(content):
    """Validate structural constraints in exported `.osu` content."""
    issues = []
    required_sections = ["[General]", "[Metadata]", "[Difficulty]", "[TimingPoints]", "[HitObjects]"]
    for section in required_sections:
        if section not in content:
            issues.append(f"missing section: {section}")

    lines = content.splitlines()
    hitobject_start = None
    for idx, line in enumerate(lines):
        if line.strip() == "[HitObjects]":
            hitobject_start = idx + 1
            break
    if hitobject_start is None:
        return issues

    hit_lines = [line for line in lines[hitobject_start:] if line.strip()]
    prev_time = -1
    for i, line in enumerate(hit_lines):
        parts = line.split(",")
        if len(parts) < 6:
            issues.append(f"hitobject line {i+1} has too few fields")
            continue
        try:
            t = int(parts[2])
        except ValueError:
            issues.append(f"hitobject line {i+1} has non-integer time")
            continue
        if t < prev_time:
            issues.append(f"hitobject line {i+1} time is not monotonic")
        prev_time = t

        try:
            obj_type = int(parts[3])
        except ValueError:
            issues.append(f"hitobject line {i+1} has invalid type")
            continue
        if obj_type & 2 and len(parts) < 8:
            issues.append(f"slider line {i+1} missing slider fields")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Generate a Taiko chart from an audio file.")
    parser.add_argument("model_path", help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("audio_path", help="Path to the input audio features file (.npy).")
    parser.add_argument("output_path", help="Path to save the generated chart (.osu).")
    parser.add_argument("--difficulty", "-d", default="oni", help="Chart difficulty (e.g., easy, normal, hard, oni).")
    parser.add_argument("--config", default="config/default.yaml", help="Path to the configuration file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic generation.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling logits.")
    parser.add_argument("--title", default=None, help="Song title for .osu metadata.")
    parser.add_argument("--artist", default=None, help="Song artist for .osu metadata.")
    parser.add_argument("--source", default="", help="Source of the song for .osu metadata.")
    parser.add_argument("--tags", default="", help="Tags for .osu metadata.")
    parser.add_argument("--bpm", type=float, default=120.0, help="Base BPM for exported timing points.")
    parser.add_argument("--offset-ms", type=int, default=1000, help="First timing point offset in milliseconds.")
    parser.add_argument("--hitsound-volume", type=int, default=70, help="Per-object hitSample volume (0-100).")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 dynamic quantization (runs on CPU).")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. auto uses CUDA unless --quantize is requested.",
    )

    args = parser.parse_args()

    # --- Setup ---
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found at {args.audio_path}")
        return

    if not os.path.exists(args.model_path):
        parser.error(
            f"Model checkpoint not found at {args.model_path}. "
            "Train a research checkpoint first or use the pretrained SoftChart backend."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.device == "auto":
        selected_device = "cpu" if args.quantize else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = args.device
    if args.quantize and selected_device != "cpu":
        print("INT8 dynamic quantization requires CPU; switching to CPU.")
        selected_device = "cpu"
    device = torch.device(selected_device)
    print(f"Using device: {device}")

    # --- Load Everything ---
    config = load_config(args.config)
    model = load_model(args.model_path, config, device, quantize=args.quantize)
    data_config = getattr(model, "taikonation_data_config", config["data"])
    tokenizer = TaikoTokenizer(
        time_quantization=data_config["time_quantization_ms"],
        source_resolution=data_config["source_resolution_ms"],
    )

    difficulty_str = args.difficulty.lower()
    if difficulty_str not in DIFFICULTY_MAP:
        print(f"Error: Invalid difficulty '{args.difficulty}'. Please choose from {list(DIFFICULTY_MAP.keys())}.")
        return
    difficulty_id = DIFFICULTY_MAP[difficulty_str]

    try:
        audio_features = get_audio_features(
            args.audio_path,
            source_resolution_ms=data_config["source_resolution_ms"],
            frame_duration_ms=data_config["time_quantization_ms"],
        )
        if audio_features is None:
            raise ValueError("Could not frame input audio features")

        feature_mean = data_config.get("feature_mean")
        feature_std = data_config.get("feature_std")
        if feature_mean is not None and feature_std is not None:
            mean = np.asarray(feature_mean, dtype=np.float32)
            std = np.maximum(np.asarray(feature_std, dtype=np.float32), 1e-4)
            audio_features = (audio_features - mean) / std

        # Validate against the loaded checkpoint architecture, not merely
        # today's YAML defaults. Smoke/specialized checkpoints may differ.
        expected_feature_size = model.audio_input_projection.in_features
        if audio_features.shape[1] != expected_feature_size:
            print(f"Error: Audio feature size mismatch. Model expects {expected_feature_size}, but got {audio_features.shape[1]}.")
            print("Please re-run feature extraction with the correct settings.")
            return

        max_len = int(model.pos_encoder.pe.shape[1])
    except Exception as e:
        print(f"Error loading or processing audio features: {e}")
        return

    # --- Generate and Save ---
    # Walk the whole song in model-sized windows. Each window begins with the
    # last generated token from the previous window, matching the boundary
    # context used by the training dataset.
    generated_token_ids = []
    previous_token = None
    total_windows = max(1, math.ceil(audio_features.shape[0] / max_len))
    for window_index, start in enumerate(range(0, audio_features.shape[0], max_len), start=1):
        chunk = audio_features[start:start + max_len]
        print(f"Generating window {window_index}/{total_windows}...")
        chunk_tokens = generate_chart(
            model,
            chunk,
            tokenizer,
            difficulty_id,
            config,
            device,
            temperature=args.temperature,
            genre_id=0,
            initial_token_id=previous_token,
        )
        if not chunk_tokens:
            continue
        generated_token_ids.extend(chunk_tokens)
        previous_token = chunk_tokens[-1]

    save_osu_chart(
        generated_token_ids,
        tokenizer,
        args.output_path,
        args.audio_path,
        title=args.title,
        artist=args.artist,
        source=args.source,
        tags=args.tags,
        difficulty=args.difficulty,
        bpm=args.bpm,
        offset_ms=args.offset_ms,
        hitsound_volume=args.hitsound_volume,
        time_quantization_ms=data_config["time_quantization_ms"],
    )


if __name__ == "__main__":
    main()
