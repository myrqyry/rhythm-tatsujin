# Rhythm Tatsujin

FastAPI is the only web backend. The studio uses the same package/runtime as the CLI.

## Start

From the repository root:

~~~bash
uv sync --group dev
uv run taikonation-serve
~~~

Open http://127.0.0.1:5000.

Do not start the server from web/ and do not install a second web-only environment.

## Generation flow

1. Upload WAV/MP3/FLAC/OGG/M4A/AAC.
2. The server stores the upload and performs best-effort duration/BPM inspection for display.
3. Leave the BPM field blank to let the Taiko model fit timing itself, or type a BPM to use it as a trusted override.
4. Pick a difficulty and generate.
5. The pretrained Taiko model performs log-mel preprocessing, timing/grid fitting, constrained chart decoding, and osu!taiko export.
6. Progress arrives through Socket.IO; task and chart metadata are persisted in output/studio.sqlite3.
7. Generated .osu files can be downloaded, edited, rated, or evaluated.

The old browser-side MediaPipe classifier and Flask backend were removed because neither was part of a real generation path.

## Main API

- GET /api/status
- POST /api/upload-audio
- POST /api/generate-chart
- GET /api/tasks
- GET /api/tasks/{task_id}
- POST /api/tasks/{task_id}/cancel
- GET /api/charts
- GET /api/chart-data
- GET /api/download-chart
- POST /api/save-chart
- POST /api/submit-rating
- POST /api/submit-evaluation
- POST /api/start-training
- POST /api/stop-training
- GET /api/research/experiments
- GET /api/research/export-dataset

Run uv run python tools/route_audit.py to regenerate the authoritative route inventory in docs/flask-migration-status.md.

## Socket.IO events

- generation_progress
- generation_failed
- chart_generated
- training_progress
- training_failed
- system_log

## Authentication

Local development works without a token. To protect mutable endpoints:

~~~bash
export TAIKONATION_API_TOKEN="your-token"
uv run taikonation-serve
~~~

The browser sends the token from local storage as Authorization: Bearer <token> when configured.

## Training

The Training page controls the local PyTorch research transformer, not the pretrained production generator. It uses the paired arrays in input_songs/ and input_charts_nr/.

Before a long run, use the CLI smoke proof:

~~~bash
uv run taikonation-train --smoke
~~~

## Validation

~~~bash
uv run pytest -q
uv run python tools/route_audit.py
node --check web/static/js/app.js
~~~

For the generation runtime, the final acceptance check is an actual audio file through the model to a valid .osu artifact.
