<!-- meristem-template:v1 -->
# Project

## Purpose

Rhythm Tatsujin generates osu!taiko charts from ordinary audio using pretrained SoftChart inference, while retaining a separate retrainable TaikoNation PyTorch research lane. Human multi-difficulty reference sets support calibration and experimental candidate selection, not automatic training-corpus expansion.

`README.md` owns the user-facing overview, installation, commands, and architecture map. `.github/copilot-instructions.md` owns existing domain coding rules. This file is a continuity index, not a replacement for those authorities.

## Invariants

- Keep production raw-audio inference and research feature-array training/generation distinct in implementation, tests, and UI labels.
- Preserve pinned SoftChart provenance/license and verified model selection; adapt through the production backend before patching vendor code.
- Research generation must fail closed without a trained checkpoint. Preserve checkpoint architecture, normalization, and timing metadata; checkpoints must reload with `weights_only=True`.
- Preserve existing SQLite data through additive schema changes. Never turn missing wiring or random weights into reported success.
- Uploaded audio, generated outputs, reference archives/corpora, model weights, and databases remain local runtime assets; do not add them to Git or the original research corpus.
- Preserve the deterministic baseline in reference-guided candidate selection. Reference distance is a heuristic, not evidence of player enjoyment.
- Detailed Taiko array/event contracts and hardware-aware precision rules remain in `.github/copilot-instructions.md`.

## Architecture and boundaries

- Production: `taikonation/generation/softchart_backend.py` adapts pinned source in `taikonation/vendor/softchart/`; `taikonation/generation/osu_export.py` exports timestamped events. Provenance lives in `third_party/softchart/`.
- Research: `taikonation/data/`, `taikonation/models/transformer.py`, and `taikonation/training/trainer.py` own paired-array preprocessing, the transformer, and training/checkpoints.
- Web: the installed serve entrypoint in `taikonation/cli.py` targets `web.server_fastapi:socket_app`. `web/persistence.py` owns SQLite records; `web/static/` owns the browser UI. The server uses a `ThreadPoolExecutor` for ML work, not a restart-safe external worker.
- Frontend: `web/index.html` and `web/editor.html` load the shared legacy `web/static/css/styles.css`, followed by scoped `web/static/css/expressive.css`. The latter owns the current studio/editor visual treatment; `web/static/js/app.js` and `editor.js` own interaction behavior. There is no React/Vite build in this checkout. Keep current UI delivery/acceptance state in `ACTIVE_WORK.md` rather than treating a visual implementation as an approved permanent design.
- Packaging: `pyproject.toml` defines Python >=3.10, package version 2.0.0, dependencies, dev group, and CLI entrypoints; `uv.lock` records the environment. `requirements.txt` explicitly defers to `pyproject.toml`.
- Research/calibration rationale and evidence remain in `docs/reference-ladder-calibration.md`, `docs/mapper-style-profiles.md`, `docs/wolfram-mapper-analysis.md`, and `docs/reference-reranker.md`. Do not duplicate their statistical or runtime receipts here.
- `.meristem/DECISIONS.md`, `LEARNINGS.md`, `ERRORS.md`, and `FEATURE_REQUESTS.md` provide future evidence-shaped recording lanes; the existing documents retain their authority. No separate ADR/task-system authority was identified in the bounded inspection.

## Known-good workflows

Use the uv workflow and commands in `README.md` and `.github/copilot-instructions.md`: `uv sync --group dev`, `uv run taikonation-serve`, `uv run taikonation-generate`, `uv run taikonation-train --smoke`, and the explicit research/reference-import commands. Initialization did not install dependencies or rerun generation/training.

## Verification

Established checks: `uv run pytest -q`, `uv run python tools/route_audit.py`, and `node --check web/static/js/app.js`.

User-visible generation requires a real audio-to-chart smoke flow, persisted task/chart inspection, export validation, and opening the result in an osu!taiko-compatible client. Keep automated checks, historical runtime receipts, and human playtesting distinct. `docs/reference-reranker.md` records prior route/archive evidence, not a fresh verification of a future working tree.

## Open structural uncertainty

- `docs/consolidation-roadmap.md` and `docs/next-steps-checklist.md` still describe retiring `web/server.py` and adding package CLI metadata as unfinished. Current serve wiring targets FastAPI and `pyproject.toml` already declares the CLI/Python floor. The checklist also describes `asyncio.create_task` execution, while the current server declares a `ThreadPoolExecutor`. Treat these documents as partially stale planning evidence; reconcile remaining items individually before treating them as the active backlog.
- `setup.py` still declares version 1.0.0, Python >=3.8, and an older dependency list, conflicting with `pyproject.toml` (2.0.0, >=3.10). Canonical uv/package metadata is the latter; legacy installation behavior has not been tested during initialization.
- `.github/copilot-instructions.md` requests automatic local commits, whereas the current host requires explicit user authorization. Preserve the source and follow the host requirement rather than silently changing checkpoint policy.
- Worker restart recovery and exporter gameplay fidelity require task-specific verification; substrate readiness does not establish them.
