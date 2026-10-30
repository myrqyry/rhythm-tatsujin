import asyncio
import os
import sys
import io
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from web import server_fastapi as web_server
from web.persistence import StudioStore
from fastapi import UploadFile


@pytest.fixture(autouse=True)
def isolated_web_runtime(tmp_path: Path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "output"
    upload_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(web_server, "UPLOAD_FOLDER", upload_dir)
    monkeypatch.setattr(web_server, "CHART_OUTPUT_FOLDER", output_dir)
    monkeypatch.setattr(web_server, "store", StudioStore(output_dir / "studio.sqlite3"))
    web_server.system_logs.clear()
    web_server.generated_charts.clear()
    web_server.human_evaluations.clear()
    web_server.active_tasks.clear()


def test_status_endpoint():
    payload = asyncio.run(web_server.api_status())
    assert payload['status'] == 'ready'


def test_audio_upload():
    web_server.system_logs.clear()
    web_server.generated_charts.clear()

    content = b'RIFF....WAVEfmt '
    response = asyncio.run(web_server.api_upload_audio(filename='test_song.wav', content=content))

    assert response.success is True
    assert response.filename == 'test_song.wav'
    assert any(log['message'].startswith('Audio uploaded:') for log in web_server.system_logs)


def test_cancel_task_endpoint():
    task = web_server.store.create_task(
        task_type='generation',
        payload={'title': 'cancel me'},
        created_at='2026-03-21T00:00:00'
    )
    response = asyncio.run(web_server.api_cancel_task(task['id']))
    assert response['success'] is True
    task_after = web_server.store.get_task(task['id'])
    assert task_after is not None
    assert task_after['status'] == 'cancelled'

def test_api_tasks_persistence():
    # Insert multiple tasks
    for i in range(5):
        web_server.store.create_task(
            task_type='training',
            payload={'run': i},
            created_at='2026-03-21T00:00:00'
        )

    # Retrieve paginated tasks
    tasks_page = asyncio.run(web_server.api_tasks(limit=3, offset=0))
    assert len(tasks_page.tasks) == 3
    assert tasks_page.total >= 5

    # Ensure they reflect state from the DB, not just in-memory dicts
    task_id = tasks_page.tasks[0]['id']
    single_task = asyncio.run(web_server.api_task(task_id))
    assert single_task['id'] == task_id
    assert single_task['task_type'] == 'training'

    # Ensure task endpoints actually query the persistent store correctly
    # Check that pagination works
    next_page = asyncio.run(web_server.api_tasks(limit=3, offset=3))
    assert len(next_page.tasks) >= 2


def test_audio_upload_with_multipart_file():
    upload = UploadFile(filename='multipart.wav', file=io.BytesIO(b'RIFF....WAVEfmt '))
    response = asyncio.run(web_server.api_upload_audio(file=upload))
    assert response.success is True
    assert response.filename == 'multipart.wav'


def test_tasks_endpoint_pagination_shape():
    payload = asyncio.run(web_server.api_tasks(limit=5, offset=0))
    assert payload.limit == 5
    assert payload.offset == 0
    assert isinstance(payload.total, int)
    assert isinstance(payload.tasks, list)


def test_charts_endpoint_pagination_shape():
    payload = asyncio.run(web_server.api_charts(limit=3, offset=0))
    assert payload.limit == 3
    assert payload.offset == 0
    assert isinstance(payload.total, int)
    assert isinstance(payload.charts, list)


def test_config_get_and_update(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "default.yaml"
    cfg.write_text("model:\n  d_model: 128\ntraining:\n  batch_size: 8\n")
    monkeypatch.setattr(web_server, "_config_path", lambda: cfg)

    initial = asyncio.run(web_server.api_get_config())
    assert initial["model"]["d_model"] == 128

    updated = asyncio.run(
        web_server.api_update_config(
            web_server.ConfigUpdateRequest(config={"model": {"d_model": 256}})
        )
    )
    assert updated["success"] is True
    assert updated["config"]["model"]["d_model"] == 256

    final = asyncio.run(web_server.api_get_config())
    assert final["model"]["d_model"] == 256
    assert final["training"]["batch_size"] == 8


def test_save_chart_preserves_existing_sliders_and_spinners(tmp_path: Path):
    """Saving circle edits must never erase osu! objects the editor can't represent."""
    filename = "Mixed Objects.osu"
    chart_file = web_server.CHART_OUTPUT_FOLDER / filename
    chart_file.write_text(
        "osu file format v14\n[HitObjects]\n"
        "256,192,100,1,0,3:3:0:70:\n"
        "128,192,200,2,0,B|200:192,1,100\n"
        "256,192,500,8,0,600\n",
        encoding="utf-8",
    )
    chart = web_server.store.create_chart({
        "title": "Mixed Objects", "artist": "Test", "difficulty": "oni",
        "bpm": 180, "genre": "test", "filename": filename,
        "created_at": "2026-10-30T00:00:00",
    })
    response = asyncio.run(web_server.api_save_chart(
        web_server.SaveChartRequest(id=chart["id"], notes=[{"time": 300, "type": "ka"}]),
        _auth=True,
    ))
    assert response["success"] is True
    saved = chart_file.read_text(encoding="utf-8")
    section = saved.split("[HitObjects]\n", 1)[1].splitlines()
    assert [int(line.split(",")[2]) for line in section] == [200, 300, 500]
    assert "128,192,200,2,0,B|200:192,1,100" in section
    assert "256,192,500,8,0,600" in section
    assert "256,192,100,1,0,3:3:0:70:" not in section
