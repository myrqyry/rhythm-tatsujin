# rhythm-tatsujin

Audio-to-chart generation for osu!taiko, plus a retrainable TaikoNation research lane.

This repository started from Emily Halina and Matthew Guzdial's TaikoNation work and now has two deliberately separate paths:

1. **Use it:** upload or pass a normal audio file and generate a Taiko chart with a pretrained, Taiko-specific model.
2. **Research it:** train the repository's PyTorch transformer on the bundled TaikoNation paired feature/chart corpus.

The old TensorFlow/tflearn checkpoints under model/ are retained as historical artifacts. They are not part of the current runtime.

## What works now

- Raw WAV/MP3/FLAC/OGG/M4A/AAC upload in the web UI; runtime uploads stay under ignored `output/uploads/` and never mix into the training corpus.
- Pretrained audio-to-Taiko generation using the pinned SoftChart v1.7 release used by the validated upstream Space, with optional local human-reference best-of-4 reranking that always keeps the deterministic baseline in the candidate pool.
- Automatic CUDA use when available; CPU fallback is supported.
- GPU-aware mixed precision: Ampere+ can use BF16, older CUDA GPUs use FP16.
- Easy, Normal, Hard, Oni, and Ura conditioning.
- Beat/grid fitting with an optional manual BPM override.
- osu!taiko .osu export with normal/big Don/Ka plus rolls and balloons/spinners.
- Persistent generated-chart metadata and task state in SQLite.
- Live generation/training progress over Socket.IO.
- Ready-to-import `.osz` downloads bundle the generated `.osu` with its exact source audio; raw `.osu` remains available separately.
- Edit, rate, and multi-criteria evaluation endpoints.
- Retrainable PyTorch transformer using the 100 paired TaikoNation samples in input_songs/ and input_charts_nr/.
- Finer 46.4ms research tokens instead of the old 100ms buckets, preserving much more onset/rhythm detail from the original ~23ms arrays.
- Full-song overlapping training windows, training-fold-only 80-band normalization, padding-aware attention, and checkpointed preprocessing metadata for matching inference.
- Unlabeled curated TaikoNation maps are treated as expert/Oni material rather than assigned synthetic difficulty labels from density.
- A fast smoke-training mode for proving the research pipeline before committing to a long run.

## Install

uv is the canonical environment manager.

~~~bash
git clone https://github.com/myrqyry/rhythm-tatsujin.git
cd rhythm-tatsujin
uv sync --group dev
~~~

The first pretrained generation downloads the model weights from Hugging Face and caches them locally.

## Generate from audio

~~~bash
uv run taikonation-generate song.mp3 output/song-oni.osu --difficulty oni
~~~

Optional flags include:

~~~text
--difficulty easy|normal|hard|oni|ura
--level 1..10
--bpm 174            # trusted manual override; omit for automatic timing
--title "Song title"
--artist "Artist"
--sample             # sampled decoding instead of deterministic greedy decoding
~~~

By default generation uses JacobLinCool/softchart-v17, the default model in the pinned upstream Space revision. The inference source used by this repository is vendored from the public SoftChart Hugging Face Space at a pinned revision; provenance and the upstream MIT license are in third_party/softchart/.

## Web studio

~~~bash
uv run taikonation-serve
~~~

Open http://127.0.0.1:5000.

The generation flow is:

~~~text
audio upload
  -> audio inspection
  -> SoftChart log-mel preprocessing
  -> beat/grid fit
  -> constrained Taiko decoding
  -> osu!taiko export
  -> persisted chart/task record
  -> download/editor/evaluation
~~~

If TAIKONATION_API_TOKEN is set, protected API requests require that token.

## Research transformer

The bundled research data contains 100 paired examples:

- input_songs/: float32 arrays with 80 audio bands at roughly 23.2ms source resolution
- input_charts_nr/: 7-channel Taiko event arrays on the same source timeline
- event channels: Don, Ka, big Don, big Ka, roll start, roll end, finisher

The default trainer groups two source frames into a 46.4ms token, uses 512-token (~23.76s) windows with 50% overlap for training, and computes per-band mean/std only from the training fold. Validation uses held-out songs with non-overlapping windows. This keeps substantially more rhythmic detail than the previous 100ms representation without increasing the Transformer's 512-token attention length.

Before a long training run, prove the full path with:

~~~bash
uv run taikonation-train --smoke
~~~

For the configured full experiment:

~~~bash
uv run taikonation-train --config config/default.yaml
~~~

Best checkpoints are exported under output/ and experiment metadata/artifacts are kept under experiments/.

To run the repository's research model explicitly instead of the pretrained production backend:

~~~bash
uv run taikonation-generate-research \
  output/taiko_transformer.pth \
  input_songs/example.npy \
  output/research-chart.osu \
  --difficulty oni
~~~

The research generator fails closed when no trained checkpoint exists; it no longer silently produces a "successful" chart from random weights. It reloads the checkpoint's exact feature normalization/timing contract and walks the full song in model-sized windows while carrying the previous generated token across window boundaries.

## Human reference ladders

Real multi-difficulty Taiko sets are especially useful for evaluating difficulty conditioning because the audio stays fixed while a human mapper changes density, subdivision complexity, and patterns.

Keep local `.osz` archives under an ignored directory such as `reference_osz/`, then import them with:

~~~bash
uv run taikonation-import-reference reference_osz/*.osz
~~~

The importer:

- keeps only `Mode:1` osu!taiko difficulties;
- recognizes Kantan/Easy, Futsuu/Normal, Muzukashii/Hard, Oni, and Inner Oni/Ura labels;
- groups all difficulties by beatmap set/audio instead of pretending they are independent songs;
- extracts the source audio and Taiko maps only into ignored `output/reference_corpus/`;
- records note density, Don/Ka and big-note ratios, inter-onset timing, beat-subdivision complexity, and density relative to the human Oni chart;
- writes both aggregate difficulty profiles and separate per-mapper style profiles under `output/reference_corpus/`;
- keeps human-Taiko plausibility, requested-difficulty plausibility, and mapper-style similarity as separate evaluation concepts rather than collapsing them into one score.

These references are calibration/evaluation data by default. They are not automatically mixed into the original 100-map training corpus.

## Architecture

- taikonation/generation/softchart_backend.py — production raw-audio inference adapter
- taikonation/generation/osu_export.py — timestamped osu!taiko exporter
- taikonation/vendor/softchart/ — pinned MIT-licensed upstream inference source
- taikonation/models/transformer.py — local research transformer
- taikonation/data/ — paired-corpus tokenization/features/dataset
- taikonation/training/trainer.py — local training/checkpoint pipeline
- web/server_fastapi.py — sole backend runtime
- web/persistence.py — SQLite persistence
- web/static/ — studio frontend

tools/route_audit.py inspects the actual FastAPI routes and rejects stale Flask-era runtime imports. It does not hard-code a "migration complete" result.

## Validation

~~~bash
uv run pytest -q
uv run python tools/route_audit.py
node --check web/static/js/app.js
~~~

For a real end-to-end proof, generate a chart from an audio file and validate/open the resulting .osu in an osu!taiko-compatible client.

## Provenance

TaikoNation research:

Halina, Emily and Guzdial, Matthew. "TaikoNation: Patterning-focused Chart Generation for Rhythm Action Games." Proceedings of the Twelfth Workshop on Procedural Content Generation, 2021.

Original repository: https://github.com/emily-halina/TaikoNationV1

Pretrained production inference uses SoftChart source by JacobLinCool under the MIT license. See third_party/softchart/PROVENANCE.md.

## License

See LICENSE for this repository and third_party/softchart/LICENSE for the vendored SoftChart source.
