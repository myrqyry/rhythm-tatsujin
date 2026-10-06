"""TaikoNation command-line entry points."""

from __future__ import annotations

from pathlib import Path


def train() -> None:
    import argparse

    from taikonation.training.trainer import load_config, main as train_main

    parser = argparse.ArgumentParser(description="Train the TaikoNation research transformer.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny one-epoch training proof instead of a full experiment.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config["dry_run"] = True
        config["training"].update(
            {
                "num_epochs": 1,
                "k_folds": 2,
                "batch_size": 2,
                "effective_batch_size": 2,
                "num_workers": 0,
                "max_samples": 8,
                "use_wandb": False,
            }
        )
        config["data"].update(
            {
                "max_sequence_length": 64,
                "window_stride": 32,
                "validation_window_stride": 64,
            }
        )
        config["model"].update(
            {
                "d_model": 64,
                "nhead": 4,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "dim_feedforward": 128,
                "use_gradient_checkpointing": False,
            }
        )
    train_main(config)


def generate() -> None:
    import argparse

    from taikonation.generation.osu_export import save_timestamped_osu
    from taikonation.generation.softchart_backend import DEFAULT_MODEL_ID, generate_from_audio

    parser = argparse.ArgumentParser(
        description="Generate an osu!taiko chart directly from an audio file."
    )
    parser.add_argument("audio_path", help="Input audio file (wav/mp3/flac/ogg/m4a/aac).")
    parser.add_argument("output_path", help="Destination .osu file.")
    parser.add_argument("--difficulty", "-d", default="oni", choices=["easy", "normal", "hard", "oni", "ura"])
    parser.add_argument("--level", type=int, default=None, help="TJA-style level 1-10 conditioning.")
    parser.add_argument("--bpm", type=float, default=None, help="Optional trusted BPM override.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--artist", default="Unknown Artist")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--sample", action="store_true", help="Sample instead of deterministic greedy decoding.")
    args = parser.parse_args()

    audio = Path(args.audio_path)
    if not audio.is_file():
        parser.error(f"Audio file not found: {audio}")
    output = Path(args.output_path)
    if output.suffix.lower() != ".osu":
        parser.error("output_path must end in .osu")

    result = generate_from_audio(
        audio,
        course=args.difficulty,
        level=args.level,
        bpm=args.bpm,
        model_id=args.model,
        greedy=not args.sample,
    )
    save_timestamped_osu(
        result.events,
        output,
        audio_filename=audio.name,
        title=args.title or audio.stem,
        artist=args.artist,
        difficulty=args.difficulty,
        bpm=result.bpm,
    )
    print(f"Generated {output} with {result.model_id} on {result.device} ({result.timing_mode}, {result.bpm:.2f} BPM)")


def generate_research() -> None:
    from taikonation.generation.generator import main

    main()


def import_reference() -> None:
    import argparse
    import json
    from pathlib import Path

    from taikonation.reference import (
        import_osz,
        summarize_manifests,
        summarize_mappers,
    )

    parser = argparse.ArgumentParser(
        description="Import local .osz files as an ignored Taiko reference/evaluation corpus."
    )
    parser.add_argument("archives", nargs="+", help="One or more .osz files.")
    parser.add_argument(
        "--output",
        default="output/reference_corpus",
        help="Ignored destination for extracted audio, Taiko maps, and manifests.",
    )
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for path in args.archives:
        import_osz(path, output_root)

    # Build profiles from the canonical per-set manifests on disk. This makes
    # incremental imports include the existing corpus and prevents duplicate
    # downloads of the same beatmap set from inflating mapper statistics.
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_root.glob("*/manifest.json"))
    ]
    difficulty_profile = summarize_manifests(manifests)
    mapper_profiles = summarize_mappers(manifests)
    (output_root / "difficulty_profile.json").write_text(
        json.dumps(difficulty_profile, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "mapper_profiles.json").write_text(
        json.dumps(mapper_profiles, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "difficulty_profile": difficulty_profile,
                "mappers": {
                    mapper: {
                        "maps": profile["maps"],
                        "sets": profile["sets"],
                        "custom_maps": profile["custom_maps"],
                    }
                    for mapper, profile in mapper_profiles.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def serve() -> None:
    import uvicorn

    uvicorn.run("web.server_fastapi:socket_app", host="127.0.0.1", port=5000, reload=False)
