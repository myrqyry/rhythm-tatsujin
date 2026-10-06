import asyncio
import io
import json
import os
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web import server_fastapi as web_server
from web.persistence import StudioStore


async def _response_bytes(response):
    payload = b''
    async for chunk in response.body_iterator:
        payload += chunk
    return payload


def test_export_dataset_returns_zip(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(web_server, "CHART_OUTPUT_FOLDER", output_dir)
    monkeypatch.setattr(web_server, "store", StudioStore(output_dir / "studio.sqlite3"))
    web_server.generated_charts.clear()
    web_server.human_evaluations.clear()

    web_server.store.create_chart({
        'title': 'Sample',
        'artist': 'Artist',
        'difficulty': 'oni',
        'bpm': 180,
        'genre': 'electronic',
        'rating': 0,
        'plays': 0,
        'created_at': '2026-03-21T00:00:00',
    })

    resp = asyncio.run(web_server.api_research_export_dataset())
    assert resp.media_type == 'application/zip'

    data = asyncio.run(_response_bytes(resp))
    assert data[:2] == b'PK'

    z = zipfile.ZipFile(io.BytesIO(data))
    assert 'dataset.json' in z.namelist()
    content = json.loads(z.read('dataset.json').decode('utf-8'))
    assert len(content['generated_charts']) > 0
    assert content['generated_charts'][0]['title'] == 'Sample'
