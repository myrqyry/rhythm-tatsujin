#!/usr/bin/env python3
"""Rhythm Tatsujin - FastAPI ASGI Server

This file provides a FastAPI-based server with python-socketio AsyncServer
mounted as an ASGI app. It mirrors the main API surface from the Flask server
but uses async handling and lazy imports for heavy ML dependencies.
"""
import os
import io
import json
import zipfile
import asyncio
import logging
import tempfile
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile as StarletteUploadFile
import socketio
from web.persistence import StudioStore

logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
CHART_OUTPUT_FOLDER = REPO_ROOT / 'output'
UPLOAD_FOLDER = CHART_OUTPUT_FOLDER / 'uploads'
LEGACY_AUDIO_FOLDER = REPO_ROOT / 'input_songs'
CONFIG_FOLDER = REPO_ROOT / 'config'
MODEL_FOLDER = REPO_ROOT / 'model'

for p in (CHART_OUTPUT_FOLDER, UPLOAD_FOLDER):
    p.mkdir(parents=True, exist_ok=True)

# Config
API_TOKEN = os.environ.get('TAIKONATION_API_TOKEN')

# Socket.IO Async server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Serve static files
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'static')), name='static')

# Compatibility caches backed by SQLite plus the runtime model catalog.
system_logs = []
generated_charts = []
human_evaluations = []
model_configurations = []
audio_features = []
active_models: Dict[str, Dict[str, Any]] = {
    "softchart-v17": {
        "name": "SoftChart v1.7",
        "type": "Pretrained audio-to-Taiko generator",
        "model_id": "JacobLinCool/softchart-v17",
        "status": "available",
    }
}
store = StudioStore(CHART_OUTPUT_FOLDER / "studio.sqlite3")
active_tasks = []
# One ML worker keeps GPU-heavy training/generation serialized on local hardware.
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="taikonation-ml")


class UploadAudioResponse(BaseModel):
    success: bool
    filename: str
    title: str
    duration: Optional[float] = None
    detected_bpm: Optional[float] = None
    ready_for_generation: bool = True


class GenerateChartRequest(BaseModel):
    title: str = Field(default="Untitled", min_length=1, max_length=120)
    artist: str = Field(default="Unknown", min_length=1, max_length=120)
    audio_filename: str = Field(min_length=1, max_length=255)
    bpm: Optional[float] = Field(default=None, ge=40, le=320)
    genre: str = Field(default="electronic", min_length=1, max_length=64)
    difficulty: str = Field(default="oni", pattern="^(easy|normal|hard|oni|ura|kantan|futsuu|muzukashii)$")
    pattern_style: str = Field(default="balanced", min_length=1, max_length=64)


class PaginatedTasksResponse(BaseModel):
    tasks: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


class PaginatedChartsResponse(BaseModel):
    charts: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any]


class SaveChartRequest(BaseModel):
    id: int
    notes: List[Dict[str, Any]]


class StartTrainingRequest(BaseModel):
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    learning_rate: float = 0.0001
    batch_size: int = 4


class SubmitRatingRequest(BaseModel):
    chart_id: int
    rating: int = Field(ge=1, le=5)


class SubmitEvaluationRequest(BaseModel):
    chart_id: int
    fun: int
    musicality: int
    playability: int
    coherence: int
    comments: str = ''


def _load_persisted_state():
    """Load persisted state and close out work that could not survive restart."""
    store.interrupt_unfinished_tasks(updated_at=datetime.now(timezone.utc).isoformat())
    system_logs[:] = store.list_logs(limit=200)
    generated_charts[:] = store.list_charts()
    human_evaluations[:] = store.list_evaluations()
    model_configurations[:] = store.get_json("model_configurations", [])
    audio_features[:] = store.get_json("audio_features", [])
    active_tasks[:] = store.list_tasks(limit=100)


_load_persisted_state()


def _config_path() -> Path:
    return CONFIG_FOLDER / "default.yaml"


def _load_runtime_config() -> Dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def require_token(token: Optional[str] = None):
    if API_TOKEN is None:
        return True
    if token == API_TOKEN:
        return True
    raise HTTPException(status_code=401, detail='Unauthorized')


def token_auth(request: Request, authorization: Optional[str] = Header(None)):
    """Accept the API token from Bearer auth or the download-link query parameter."""
    bearer = (authorization or '').replace('Bearer ', '').strip()
    query_token = request.query_params.get('api_token')
    return require_token(bearer or query_token)


async def add_system_log(level: str, message: str):
    entry = {'timestamp': datetime.now(timezone.utc).isoformat(), 'level': level, 'message': message}
    system_logs.insert(0, entry)
    # Keep last 200
    del system_logs[200:]
    store.append_log(entry)
    await sio.emit('system_log', entry)


@app.get('/')
async def index():
    index_path = BASE_DIR / 'index.html'
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return JSONResponse({'error': 'index.html not found'}, status_code=404)


@app.get('/editor')
async def editor():
    editor_path = BASE_DIR / 'editor.html'
    if editor_path.exists():
        return HTMLResponse(editor_path.read_text())
    return JSONResponse({'error': 'editor.html not found'}, status_code=404)


@app.get('/favicon.ico')
async def favicon():
    fav_path = BASE_DIR / 'static' / 'favicon.ico'
    if fav_path.exists():
        return FileResponse(fav_path)
    return Response(status_code=204)


@app.get('/static/{filename:path}')
async def static_files(filename: str):
    static_path = BASE_DIR / 'static' / filename
    if not static_path.exists():
        return JSONResponse(status_code=404, content="")
    return FileResponse(static_path, headers={'Cache-Control': 'max-age=3600'})


@app.get('/api/status')
async def api_status():
    active_tasks[:] = store.list_tasks(limit=100)
    generation_active = any(
        t.get('status') in {'queued', 'running'} and t.get('task_type') == 'generation'
        for t in active_tasks
    )
    training_active = any(
        t.get('status') in {'queued', 'running'} and t.get('task_type') == 'training'
        for t in active_tasks
    )
    return {
        'status': 'ready',
        'models_loaded': len([m for m in active_models.values() if m.get('status') == 'ready']),
        'models_available': len(active_models),
        'training_active': training_active,
        'generation_active': generation_active,
    }


@app.get('/api/dashboard')
async def api_dashboard():
    charts = store.list_charts(limit=500)
    generated_charts[:] = charts
    rated = [chart for chart in charts if float(chart.get('rating', 0) or 0) > 0]
    average_rating = (
        sum(float(chart.get('rating', 0) or 0) for chart in rated) / len(rated)
        if rated else None
    )
    return {
        'metrics': {
            'generated_charts': store.count_charts(),
            'rated_charts': len(rated),
            'avg_rating': round(average_rating, 2) if average_rating is not None else None,
        },
        'recent_logs': system_logs[:10],
    }


@app.get('/api/models')
async def api_models():
    return {'models': active_models}


@app.get('/api/config')
async def api_get_config(_auth: bool = Depends(token_auth)):
    return _load_runtime_config()


@app.post('/api/config')
async def api_update_config(payload: ConfigUpdateRequest, _auth: bool = Depends(token_auth)):
    existing = _load_runtime_config()
    merged = _deep_merge(existing, payload.config)
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    await add_system_log("success", "Configuration updated")
    return {"success": True, "config": merged}


def _inspect_audio(path: Path) -> tuple[Optional[float], Optional[float]]:
    """Best-effort metadata for the upload UI; generation does its own preprocessing."""
    try:
        import librosa
        import numpy as np

        wav, sr = librosa.load(str(path), sr=22050, mono=True)
        if wav.size == 0:
            return None, None
        duration = float(librosa.get_duration(y=wav, sr=sr))
        tempo = librosa.beat.beat_track(y=wav, sr=sr)[0]
        bpm = float(np.atleast_1d(tempo)[0])
        if not np.isfinite(bpm) or bpm <= 0:
            bpm = None
        return duration, bpm
    except Exception:
        logger.warning("Could not inspect uploaded audio %s", path, exc_info=True)
        return None, None


@app.post('/api/upload-audio')
async def api_upload_audio(
    filename: Optional[str] = None,
    content: Optional[bytes] = None,
    audio: Optional[UploadFile] = File(default=None),
    file: Optional[UploadFile] = File(default=None),
    _auth: bool = Depends(token_auth),
) -> UploadAudioResponse:
    upload = (
        audio if isinstance(audio, StarletteUploadFile)
        else file if isinstance(file, StarletteUploadFile)
        else None
    )
    if upload is not None:
        filename = upload.filename
        content = await upload.read()

    if not filename or content is None:
        raise HTTPException(status_code=400, detail='No file provided')

    filename = Path(filename).name
    allowed = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'}
    if Path(filename).suffix.lower() not in allowed:
        raise HTTPException(status_code=400, detail='Unsupported file type')
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='File too large (max 100MB)')

    dest = UPLOAD_FOLDER / filename
    with open(dest, 'wb') as handle:
        handle.write(content)

    duration, detected_bpm = await asyncio.to_thread(_inspect_audio, dest)
    title = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
    await add_system_log('success', f'Audio uploaded: {filename}')
    return UploadAudioResponse(
        success=True,
        filename=filename,
        title=title,
        duration=duration,
        detected_bpm=detected_bpm,
        ready_for_generation=True,
    )


class GenerationCancelled(RuntimeError):
    pass


def _emit_from_worker(loop: asyncio.AbstractEventLoop, event: str, payload: dict) -> None:
    try:
        future = asyncio.run_coroutine_threadsafe(sio.emit(event, payload), loop)
        future.add_done_callback(
            lambda done: logger.error("Socket emit failed: %s", done.exception())
            if done.exception() is not None else None
        )
    except Exception:
        logger.warning("Could not emit %s from worker", event, exc_info=True)


def _run_generation_job(
    task_id: int,
    audio_filename: str,
    title: str,
    artist: str,
    bpm: Optional[float],
    genre: str,
    difficulty: str,
    pattern_style: str,
    loop: asyncio.AbstractEventLoop,
):
    from taikonation.generation.osu_export import save_timestamped_osu
    from taikonation.generation.softchart_backend import generate_best_of_audio

    audio_path = UPLOAD_FOLDER / Path(audio_filename).name

    def progress(percent: int, message: str) -> None:
        current = store.get_task(task_id)
        if current and current.get('status') == 'cancelled':
            raise GenerationCancelled("Generation cancelled by user")
        store.update_task(
            task_id,
            status='running',
            progress=max(0, min(int(percent), 99)),
            message=message,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        _emit_from_worker(
            loop,
            'generation_progress',
            {'task_id': task_id, 'progress': int(percent), 'message': message},
        )

    try:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Uploaded audio not found: {audio_filename}")

        progress(1, 'Loading pretrained Taiko model')
        result = generate_best_of_audio(
            audio_path,
            course=difficulty,
            bpm=bpm,
            candidate_count=4,
            reference_root=REPO_ROOT / "output" / "reference_corpus",
            on_progress=progress,
        )
        active_models["softchart-v17"].update({
            "status": "ready",
            "device": result.device,
            "model_id": result.model_id,
        })

        safe_title = ''.join(
            ch if ch.isalnum() or ch in {'-', '_'} else '_'
            for ch in title
        ).strip('_')[:60] or 'chart'
        output_filename = f"{task_id}_{safe_title}_{difficulty}.osu"
        output_path = CHART_OUTPUT_FOLDER / output_filename
        save_timestamped_osu(
            result.events,
            output_path,
            audio_filename=audio_path.name,
            title=title,
            artist=artist,
            difficulty=difficulty,
            bpm=result.bpm,
        )

        chart = store.create_chart({
            'title': title,
            'artist': artist,
            'difficulty': difficulty,
            'bpm': result.bpm,
            'genre': genre,
            'filename': output_filename,
            'audio_filename': audio_path.name,
            'model_id': result.model_id,
            'timing_mode': result.timing_mode,
            'quality_score': result.quality_score,
            'candidate_count': result.candidate_count,
            'candidate_seed': result.candidate_seed,
            'candidate_scores': list(result.candidate_scores),
            'rating': 0,
            'plays': 0,
            'created_at': datetime.now(timezone.utc).isoformat(),
        })

        store.update_task(
            task_id,
            status='completed',
            progress=100,
            message='Generation completed',
            result={'chart_id': chart['id'], 'filename': output_filename},
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        active_tasks[:] = store.list_tasks(limit=100)
        generated_charts[:] = store.list_charts()
        _emit_from_worker(
            loop,
            'generation_progress',
            {'task_id': task_id, 'progress': 100, 'message': 'Generation completed'},
        )
        _emit_from_worker(loop, 'chart_generated', {'chart': chart})
    except GenerationCancelled:
        active_tasks[:] = store.list_tasks(limit=100)
        _emit_from_worker(
            loop,
            'generation_progress',
            {'task_id': task_id, 'progress': 0, 'message': 'Generation cancelled', 'cancelled': True},
        )
    except Exception as exc:
        logger.error("Generation task %s failed", task_id, exc_info=True)
        current = store.get_task(task_id)
        if not current or current.get('status') != 'cancelled':
            store.update_task(
                task_id,
                status='failed',
                message=str(exc),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        active_tasks[:] = store.list_tasks(limit=100)
        _emit_from_worker(
            loop,
            'generation_failed',
            {'task_id': task_id, 'error': str(exc)},
        )


class TrainingCancelled(RuntimeError):
    pass


def _run_training_job(
    task_id: int,
    model_dump: dict,
    loop: asyncio.AbstractEventLoop,
):
    from taikonation.training.trainer import load_config as load_training_config
    from taikonation.training.trainer import main as train_main

    try:
        current = store.get_task(task_id)
        if current and current.get('status') == 'cancelled':
            return

        config = load_training_config(str(CONFIG_FOLDER / 'default.yaml'))
        model_keys = {'d_model', 'nhead', 'num_encoder_layers', 'num_decoder_layers'}
        training_keys = {'learning_rate', 'batch_size'}
        config['model'].update({k: v for k, v in model_dump.items() if k in model_keys})
        config['training'].update({k: v for k, v in model_dump.items() if k in training_keys})
        config['training']['use_wandb'] = False

        store.update_task(
            task_id,
            status='running',
            progress=1,
            message='Training started',
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        def progress(info: dict) -> None:
            current = store.get_task(task_id)
            if current and current.get('status') == 'cancelled':
                raise TrainingCancelled("Training cancelled by user")
            folds = max(int(info.get('folds', 1)), 1)
            epochs = max(int(info.get('epochs', 1)), 1)
            fold = max(int(info.get('fold', 1)), 1)
            epoch = max(int(info.get('epoch', 1)), 1)
            completed = (fold - 1) * epochs + epoch
            total = folds * epochs
            percent = min(99, 5 + int(90 * completed / total))
            val = info.get('val', {})
            message = (
                f"Fold {fold}/{folds}, epoch {epoch}/{epochs}"
                f" - val loss {float(val.get('loss', 0.0)):.4f}"
            )
            store.update_task(
                task_id,
                status='running',
                progress=percent,
                message=message,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            _emit_from_worker(
                loop,
                'training_progress',
                {
                    'task_id': task_id,
                    'progress': percent,
                    'metrics': {
                        'train_loss': info.get('train', {}).get('loss'),
                        'val_loss': val.get('loss'),
                        'val_accuracy': val.get('accuracy'),
                    },
                },
            )

        results = train_main(config, progress_callback=progress)

        store.update_task(
            task_id,
            status='completed',
            progress=100,
            message='Training completed',
            result={'folds': results},
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        active_tasks[:] = store.list_tasks(limit=100)
        _emit_from_worker(
            loop,
            'training_progress',
            {'task_id': task_id, 'progress': 100, 'metrics': {}},
        )
    except TrainingCancelled:
        active_tasks[:] = store.list_tasks(limit=100)
        _emit_from_worker(
            loop,
            'training_progress',
            {'task_id': task_id, 'progress': 0, 'metrics': {}, 'cancelled': True},
        )
    except Exception as exc:
        logger.error("Training task %s failed", task_id, exc_info=True)
        current = store.get_task(task_id)
        if not current or current.get('status') != 'cancelled':
            store.update_task(
                task_id,
                status='failed',
                message=str(exc),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        active_tasks[:] = store.list_tasks(limit=100)
        _emit_from_worker(
            loop,
            'training_failed',
            {'task_id': task_id, 'error': str(exc)},
        )


@app.post('/api/generate-chart')
async def api_generate_chart(
    request: GenerateChartRequest,
    _auth: bool = Depends(token_auth),
):
    audio_filename = Path(request.audio_filename).name
    audio_path = UPLOAD_FOLDER / audio_filename
    difficulty = {
        'kantan': 'easy',
        'futsuu': 'normal',
        'muzukashii': 'hard',
    }.get(request.difficulty, request.difficulty)
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail='Uploaded audio not found')

    await add_system_log('info', f'Chart generation started: {request.title}')
    task = store.create_task(
        task_type='generation',
        payload={
            'audio_filename': audio_filename,
            'title': request.title,
            'artist': request.artist,
            'difficulty': difficulty,
            'bpm': request.bpm,
            'genre': request.genre,
            'pattern_style': request.pattern_style,
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    active_tasks.insert(0, task)

    loop = asyncio.get_running_loop()
    executor.submit(
        _run_generation_job,
        task['id'],
        audio_filename,
        request.title,
        request.artist,
        request.bpm,
        request.genre,
        difficulty,
        request.pattern_style,
        loop,
    )
    return {'success': True, 'message': 'Chart generation started', 'task_id': task['id']}


@app.get('/api/tasks')
async def api_tasks(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _auth: bool = Depends(token_auth),
) -> PaginatedTasksResponse:
    active_tasks[:] = store.list_tasks(limit=limit, offset=offset)
    return PaginatedTasksResponse(tasks=active_tasks, total=store.count_tasks(), limit=limit, offset=offset)


@app.get('/api/tasks/{task_id}')
async def api_task(task_id: int, _auth: bool = Depends(token_auth)):
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail='Task not found')
    return task


@app.post('/api/tasks/{task_id}/cancel')
async def api_cancel_task(task_id: int, _auth: bool = Depends(token_auth)):
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail='Task not found')
    if task.get('status') in {'completed', 'failed', 'cancelled', 'interrupted'}:
        return {'success': False, 'message': f"Task already {task.get('status')}"}
    store.update_task(
        task_id,
        status='cancelled',
        message='Cancelled by user',
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    active_tasks[:] = store.list_tasks(limit=100)
    await add_system_log('warning', f'Task cancelled: {task_id}')
    return {'success': True, 'task_id': task_id, 'status': 'cancelled'}


@app.get('/api/charts')
async def api_charts(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _auth: bool = Depends(token_auth),
) -> PaginatedChartsResponse:
    generated_charts[:] = store.list_charts(limit=limit, offset=offset)
    return PaginatedChartsResponse(charts=generated_charts, total=store.count_charts(), limit=limit, offset=offset)


@app.get('/api/get-chart-for-evaluation')
async def api_get_chart_for_evaluation():
    chart = store.get_unrated_chart()
    if chart:
        return chart
    raise HTTPException(status_code=404, detail='No charts available for evaluation')


@app.get('/api/chart-data')
async def api_chart_data(id: int = Query(...)):
    chart = store.get_chart(id)
    if not chart or not chart.get('filename'):
        raise HTTPException(status_code=404, detail='Chart not found')

    filepath = CHART_OUTPUT_FOLDER / chart['filename']
    if not filepath.exists():
        raise HTTPException(status_code=404, detail='Chart file not found on disk')

    # Helper to parse .osu file for editor
    def parse_osu_file(filepath):
        notes = []
        in_hit_objects = False
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == '[HitObjects]':
                        in_hit_objects = True
                        continue
                    if line.startswith('['):
                        in_hit_objects = False
                        continue
                    if in_hit_objects and line:
                        parts = line.split(',')
                        if len(parts) >= 5:
                            time = int(parts[2])
                            obj_type = int(parts[3])
                            hit_sound = int(parts[4])
                            if not (obj_type & 1):
                                continue
                            is_big = bool(hit_sound & 4)
                            is_ka = bool(hit_sound & 8 or hit_sound & 2)
                            if is_ka:
                                note_type = 'big_ka' if is_big else 'ka'
                            else:
                                note_type = 'big_don' if is_big else 'don'
                            notes.append({'time': time, 'type': note_type})
        except Exception as e:
            # Assuming `logger` is a configured logging instance at the module level
            logger.error(f"Failed to parse .osu file {filepath}: {e}", exc_info=True)
            return None
        return notes

    notes = parse_osu_file(filepath)
    if notes is None:
        raise HTTPException(status_code=500, detail='Failed to parse chart file')

    return {'notes': notes, 'metadata': chart}


def _resolve_chart_audio_path(audio_filename: str) -> Optional[Path]:
    """Resolve current uploads first, then the pre-revival legacy audio location."""
    safe_name = Path(audio_filename).name
    if not safe_name:
        return None
    for folder in (UPLOAD_FOLDER, LEGACY_AUDIO_FOLDER):
        candidate = folder / safe_name
        if candidate.is_file():
            return candidate
    return None


def _build_osz_archive(chart: Dict[str, Any]) -> Path:
    """Build a temporary osu! beatmap archive from a persisted chart + source audio."""
    osu_filename = Path(str(chart.get('filename') or '')).name
    audio_filename = Path(str(chart.get('audio_filename') or '')).name
    if not osu_filename:
        raise HTTPException(status_code=404, detail='Chart file is not recorded')
    if not audio_filename:
        raise HTTPException(
            status_code=409,
            detail='Source audio is not recorded for this chart; download the raw .osu instead',
        )

    osu_path = CHART_OUTPUT_FOLDER / osu_filename
    audio_path = _resolve_chart_audio_path(audio_filename)
    if not osu_path.is_file():
        raise HTTPException(status_code=404, detail='Chart file not found on disk')
    if audio_path is None:
        raise HTTPException(
            status_code=404,
            detail='Source audio not found on disk; download the raw .osu instead',
        )

    temp = tempfile.NamedTemporaryFile(
        prefix='rhythm-tatsujin-',
        suffix='.osz',
        delete=False,
    )
    temp_path = Path(temp.name)
    temp.close()
    try:
        with zipfile.ZipFile(temp_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(osu_path, arcname=osu_filename)
            archive.write(audio_path, arcname=audio_filename)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return temp_path


@app.get('/api/download-chart')
async def download_chart(
    id: int = Query(...),
    format: str = Query('osz', pattern='^(osz|osu)$'),
    _auth: bool = Depends(token_auth),
):
    chart = store.get_chart(id)
    if not chart or not chart.get('filename'):
        raise HTTPException(status_code=404, detail='Chart not found')

    filename = Path(chart['filename']).name
    filepath = CHART_OUTPUT_FOLDER / filename
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail='Chart file not found on disk')

    if format == 'osu':
        return FileResponse(filepath, filename=filename, media_type='text/plain')

    archive_path = _build_osz_archive(chart)
    archive_filename = f"{Path(filename).stem}.osz"
    return FileResponse(
        archive_path,
        filename=archive_filename,
        media_type='application/zip',
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@app.post('/api/submit-rating')
async def api_submit_rating(request: SubmitRatingRequest, _auth: bool = Depends(token_auth)):
    if not store.set_chart_rating(request.chart_id, request.rating):
        raise HTTPException(status_code=404, detail='Chart not found')
    generated_charts[:] = store.list_charts()
    await add_system_log('success', f'Rating submitted for chart {request.chart_id}')
    return {'success': True, 'rating': request.rating}


@app.post('/api/submit-evaluation')
async def api_submit_evaluation(request: SubmitEvaluationRequest, _auth: bool = Depends(token_auth)):
    ok = store.submit_evaluation(
        chart_id=request.chart_id,
        fun=request.fun,
        musicality=request.musicality,
        playability=request.playability,
        coherence=request.coherence,
        comments=request.comments,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    if not ok:
        raise HTTPException(status_code=404, detail='Chart not found')
    generated_charts[:] = store.list_charts()
    human_evaluations[:] = store.list_evaluations()
    await add_system_log('success', f'Evaluation submitted for chart {request.chart_id}')
    return {'success': True}




def _read_experiment_metadata() -> list[dict]:
    experiment_root = REPO_ROOT / "experiments"
    rows = []
    if not experiment_root.is_dir():
        return rows
    for metadata_path in experiment_root.glob("*/experiment.json"):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if isinstance(metadata, dict):
                rows.append(metadata)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read experiment metadata %s", metadata_path, exc_info=True)
    rows.sort(key=lambda row: row.get("start_time", ""), reverse=True)
    return rows


@app.get('/api/research/experiments')
async def api_research_experiments():
    """Return only persisted experiment/chart/evaluation evidence."""
    experiments = _read_experiment_metadata()
    model_runs = []
    for experiment in experiments:
        metrics = experiment.get("metrics", {})
        val = metrics.get("final_val_results", {}) if isinstance(metrics, dict) else {}
        model_runs.append({
            "experiment_id": experiment.get("experiment_id"),
            "name": experiment.get("experiment_name"),
            "status": experiment.get("status"),
            "start_time": experiment.get("start_time"),
            "val_loss": val.get("loss") if isinstance(val, dict) else None,
            "val_accuracy": val.get("accuracy") if isinstance(val, dict) else None,
            "best_checkpoint": metrics.get("best_checkpoint") if isinstance(metrics, dict) else None,
        })

    charts = store.list_charts(limit=500)
    evaluations = store.list_evaluations()
    rated = [chart for chart in charts if float(chart.get('rating', 0) or 0) > 0]
    return {
        'experiments': experiments,
        'training_runs': model_runs,
        'chart_history': {
            'timestamps': [c.get('created_at', '') for c in charts],
            'ratings': [float(c.get('rating', 0) or 0) for c in charts],
            'difficulties': [c.get('difficulty') for c in charts],
        },
        'human_feedback_stats': {
            'total_evaluations': len(evaluations),
            'rated_charts': len(rated),
            'average_rating': (
                sum(float(c.get('rating', 0) or 0) for c in rated) / len(rated)
                if rated else None
            ),
        },
    }


@app.post('/api/save-chart')
async def api_save_chart(request: SaveChartRequest, _auth: bool = Depends(token_auth)):
    chart_id = request.id
    chart = store.get_chart(chart_id)
    if not chart or not chart.get('filename'):
        raise HTTPException(status_code=404, detail='Chart not found')

    filepath = CHART_OUTPUT_FOLDER / chart['filename']
    if not filepath.exists():
        raise HTTPException(status_code=404, detail='Chart file not found on disk')

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        try:
            start_index = lines.index('[HitObjects]\n') + 1
        except ValueError:
            lines.append('\n[HitObjects]\n')
            start_index = len(lines)

        end_index = start_index
        while end_index < len(lines) and not lines[end_index].startswith('['):
            end_index += 1

        new_hit_objects = []
        for note in request.notes:
            time = note.get('time', 0)
            note_type = note.get('type', 'don')
            x, y, obj_type, hit_sound, extras = 256, 192, 1, 0, '3:3:0:70:'

            if note_type == 'don':
                hit_sound = 0
            elif note_type == 'ka':
                hit_sound = 8
            elif note_type == 'big_don':
                hit_sound = 4
            elif note_type == 'big_ka':
                hit_sound = 12

            new_hit_objects.append(f"{x},{y},{time},{obj_type},{hit_sound},{extras}\n")

        new_lines = lines[:start_index] + new_hit_objects + lines[end_index:]

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        await add_system_log('success', f"Chart '{chart.get('title', 'Unknown')}' updated from editor.")
        return {'success': True, 'message': 'Chart saved successfully.'}
    except Exception as e:
        # Assuming `logger` is a configured logging instance at the module level
        logger.error(f"Failed to save chart file {filepath}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail='Failed to save chart file')


@app.post('/api/start-training')
async def api_start_training(request: StartTrainingRequest, _auth: bool = Depends(token_auth)):
    task = store.create_task(
        task_type='training',
        payload=request.model_dump(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    active_tasks.insert(0, task)
    loop = asyncio.get_running_loop()
    executor.submit(_run_training_job, task['id'], request.model_dump(), loop)
    return {'success': True, 'task_id': task['id'], 'message': 'Training started.'}


@app.post('/api/stop-training')
async def api_stop_training(_auth: bool = Depends(token_auth)):
    cancelled = []
    for task in store.list_tasks(limit=500):
        if task.get('task_type') == 'training' and task.get('status') in {'queued', 'running'}:
            store.update_task(
                task['id'],
                status='cancelled',
                message='Cancelled by user',
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            cancelled.append(task['id'])
    active_tasks[:] = store.list_tasks(limit=100)
    await add_system_log('info', f'Training stop requested for {len(cancelled)} task(s)')
    return {'success': True, 'cancelled_task_ids': cancelled}


@app.get('/api/research/export-dataset')
async def api_research_export_dataset():
    """Export the current research dataset as a zip archive."""
    # Ensure we use fresh data from the store for the export
    current_charts = store.list_charts()
    current_evals = store.list_evaluations()
    current_model_configs = store.get_json("model_configurations", [])
    current_audio_features = store.get_json("audio_features", [])

    # Sync global variables for compatibility
    generated_charts[:] = current_charts
    human_evaluations[:] = current_evals
    model_configurations[:] = current_model_configs
    audio_features[:] = current_audio_features

    export_payload = {
        'generated_charts': current_charts,
        'human_evaluations': current_evals,
        'model_configurations': current_model_configs,
        'audio_features': current_audio_features,
        'exported_at': datetime.now(timezone.utc).isoformat()
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('dataset.json', json.dumps(export_payload, indent=2))

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type='application/zip',
        headers={'Content-Disposition': 'attachment; filename=research_dataset.zip'}
    )

@sio.event
async def connect(sid, environ):
    await add_system_log('info', f'Client connected: {sid}')


@sio.event
async def disconnect(sid):
    await add_system_log('info', f'Client disconnected: {sid}')


if __name__ == '__main__':
    # Run with: uvicorn server_fastapi:socket_app --reload --port 5000
    import uvicorn
    uvicorn.run('server_fastapi:socket_app', host='127.0.0.1', port=5000, reload=True)
