import asyncio
import zipfile
from pathlib import Path

from web import server_fastapi as web_server
from web.persistence import StudioStore


def _make_chart(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    upload_dir = output_dir / "uploads"
    legacy_dir = tmp_path / "input_songs"
    output_dir.mkdir()
    upload_dir.mkdir()
    legacy_dir.mkdir()

    monkeypatch.setattr(web_server, "CHART_OUTPUT_FOLDER", output_dir)
    monkeypatch.setattr(web_server, "UPLOAD_FOLDER", upload_dir)
    monkeypatch.setattr(web_server, "LEGACY_AUDIO_FOLDER", legacy_dir)
    monkeypatch.setattr(web_server, "store", StudioStore(output_dir / "studio.sqlite3"))

    osu_name = "7_My_Song_oni.osu"
    audio_name = "My Song.mp3"
    (output_dir / osu_name).write_text(
        "osu file format v14\n\n[General]\n"
        f"AudioFilename: {audio_name}\n"
        "Mode: 1\n\n[Metadata]\nTitle:My Song\nVersion:Oni\n",
        encoding="utf-8",
    )
    (upload_dir / audio_name).write_bytes(b"fake-mp3-audio")

    chart = web_server.store.create_chart(
        {
            "title": "My Song",
            "artist": "Artist",
            "difficulty": "oni",
            "bpm": 180,
            "genre": "electronic",
            "filename": osu_name,
            "audio_filename": audio_name,
            "model_id": "JacobLinCool/softchart-v17",
            "timing_mode": "fixed",
            "rating": 0,
            "plays": 0,
            "created_at": "2026-10-05T00:00:00+00:00",
        }
    )
    return chart, output_dir, upload_dir


def test_osz_builder_packages_exact_chart_and_audio_names(tmp_path: Path, monkeypatch):
    chart, _, _ = _make_chart(tmp_path, monkeypatch)
    archive_path = web_server._build_osz_archive(chart)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            assert set(archive.namelist()) == {"7_My_Song_oni.osu", "My Song.mp3"}
            osu = archive.read("7_My_Song_oni.osu").decode("utf-8")
            assert "AudioFilename: My Song.mp3" in osu
            assert archive.read("My Song.mp3") == b"fake-mp3-audio"
    finally:
        archive_path.unlink(missing_ok=True)


def test_download_chart_defaults_to_osz_and_keeps_raw_osu_option(tmp_path: Path, monkeypatch):
    chart, output_dir, _ = _make_chart(tmp_path, monkeypatch)

    osz_response = asyncio.run(web_server.download_chart(id=chart["id"], format="osz"))
    osz_path = Path(osz_response.path)
    try:
        assert osz_response.media_type == "application/zip"
        assert ".osz" in osz_response.headers["content-disposition"]
        with zipfile.ZipFile(osz_path) as archive:
            assert "My Song.mp3" in archive.namelist()
            assert "7_My_Song_oni.osu" in archive.namelist()
    finally:
        osz_path.unlink(missing_ok=True)

    osu_response = asyncio.run(web_server.download_chart(id=chart["id"], format="osu"))
    assert Path(osu_response.path) == output_dir / "7_My_Song_oni.osu"
    assert ".osu" in osu_response.headers["content-disposition"]


def test_osz_builder_falls_back_to_legacy_audio_location(tmp_path: Path, monkeypatch):
    chart, _, upload_dir = _make_chart(tmp_path, monkeypatch)
    legacy_dir = web_server.LEGACY_AUDIO_FOLDER
    source = upload_dir / "My Song.mp3"
    legacy_audio = legacy_dir / source.name
    source.replace(legacy_audio)

    archive_path = web_server._build_osz_archive(chart)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.read("My Song.mp3") == b"fake-mp3-audio"
    finally:
        archive_path.unlink(missing_ok=True)
