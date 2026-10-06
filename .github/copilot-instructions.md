# rhythm-tatsujin — coding-agent instructions

This repository has two intentionally separate ML paths. Do not collapse them or silently substitute one for the other.

## Runtime architecture

Production generation
- Raw audio -> pretrained Taiko-specific SoftChart model -> timestamped events -> osu!taiko .osu.
- Adapter: taikonation/generation/softchart_backend.py.
- Exporter: taikonation/generation/osu_export.py.
- Pinned upstream inference source: taikonation/vendor/softchart/.
- Provenance/license: third_party/softchart/.
- Default model: JacobLinCool/softchart-v17.

Research/training
- Bundled 80-band .npy features + 7-channel chart arrays -> local PyTorch transformer.
- Model: taikonation/models/transformer.py.
- Dataset/tokenization: taikonation/data/.
- Training/checkpoints: taikonation/training/trainer.py.
- Research generation: taikonation/generation/generator.py.

Web studio
- web/server_fastapi.py is the only backend runtime.
- web/persistence.py owns SQLite state.
- web/static/js/app.js is the browser application.
- There is no Flask server and no web.server runtime.

## Canonical commands

    uv sync --group dev
    uv run pytest -q
    uv run python tools/route_audit.py
    node --check web/static/js/app.js

    uv run taikonation-serve
    uv run taikonation-generate song.mp3 output/song.osu --difficulty oni
    uv run taikonation-train --smoke
    uv run taikonation-train --config config/default.yaml

Use uv, not a second ad-hoc virtualenv or a web-only dependency environment.

## Checkpoint discipline

- After a coherent change batch has passed its supported validation, commit it to the current working branch unless the user explicitly asks not to.
- Treat the commit as part of completing the work, not as an optional follow-up.
- Never stage runtime/user assets, uploaded audio, local reference corpora, model weights, databases, caches, or generated outputs just to make the tree clean.
- Do not push commits unless the user explicitly asks for a push.

## Non-negotiable correctness rules

- Never generate with random weights when a research checkpoint is missing. Fail closed.
- A saved training checkpoint is not considered valid until it reloads under weights_only=True.
- Preserve the checkpoint's architecture metadata; do not reconstruct from current YAML defaults when a checkpoint records different dimensions.
- Dataset cache keys must identify the source song and preprocessing parameters. Never key cache files by split-local integer index.
- Do not force a CUDA attention kernel. Let PyTorch choose a supported backend.
- Mixed precision must be hardware-aware. The vendored production path uses BF16 only on sm80+ and FP16 on older CUDA hardware.
- Do not invent research metrics, generated-chart metadata, task success, or model status.
- A generated .osu must be structurally validated and hit-object timestamps must be monotonic.
- Preserve existing SQLite data when schemas evolve. Use additive migrations rather than deleting the database.
- The production model and the research transformer are different systems; tests and UI labels must say which path they exercise.

## Taiko data contracts

Bundled research arrays:
- input_songs/*.npy: 2-D float arrays, 80 audio bands.
- input_charts_nr/*.npy: 2-D binary arrays, 7 event channels.
- Event order: don, ka, big_don, big_ka, roll_start, roll_end, finisher.

Difficulty aliases accepted at UI boundaries:
- Kantan -> easy
- Futsuu -> normal
- Muzukashii -> hard
- Oni -> oni
- Ura Oni -> ura

Store canonical difficulty values internally.

## Web/API checks

tools/route_audit.py parses the actual FastAPI module and rejects stale Flask/root-script references. Keep it executable; do not replace it with hard-coded migration text.

High-value web smoke flow:
1. GET /api/status
2. multipart POST /api/upload-audio
3. JSON POST /api/generate-chart
4. poll GET /api/tasks/{id}
5. inspect GET /api/chart-data?id=...
6. download GET /api/download-chart?id=...
7. validate downloaded .osu

Long-running generation/training currently executes in the server's worker thread pool with task state persisted in SQLite. A future restart-safe external worker is hardening work, not a reason to reintroduce the removed Flask/task registry.

## Vendored SoftChart

Treat taikonation/vendor/softchart/ as pinned third-party source. Keep upstream changes isolated and documented in third_party/softchart/PROVENANCE.md. Prefer adapting through softchart_backend.py; patch vendored code only for a concrete compatibility defect and document why.

Do not casually upgrade the pretrained model because a newer repository number exists. The default should track a model whose capabilities are verified against the pinned upstream application.
