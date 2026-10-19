import asyncio
import io
import math
import wave
from pathlib import Path

import pytest
from fastapi import HTTPException

from web import server_fastapi as server
from web.persistence import StudioStore


def wav_bytes() -> bytes:
    """A real, decodable 0.25-second PCM WAV, not a magic-byte stub."""
    audio = io.BytesIO()
    with wave.open(audio, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        samples = [int(12000 * math.sin(2 * math.pi * 220 * n / 8000)) for n in range(2000)]
        from struct import pack
        writer.writeframes(pack("<" + "h" * len(samples), *samples))
    return audio.getvalue()


@pytest.fixture(autouse=True)
def isolated_uploads(tmp_path: Path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(server, "UPLOAD_FOLDER", uploads)
    monkeypatch.setattr(server, "CHART_OUTPUT_FOLDER", tmp_path)
    monkeypatch.setattr(server, "store", StudioStore(tmp_path / "studio.sqlite3"))
    server.system_logs.clear()
    return uploads


def test_rejects_undecodable_payload_without_creating_a_saved_audio_file(isolated_uploads):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.api_upload_audio(filename="broken.wav", content=b"RIFF....WAVEfmt "))
    assert exc.value.status_code == 400
    assert list(isolated_uploads.iterdir()) == []


def test_empty_upload_is_not_ready_for_generation(isolated_uploads):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.api_upload_audio(filename="empty.wav", content=b""))
    assert exc.value.status_code == 400
    assert list(isolated_uploads.iterdir()) == []


def test_uploaded_valid_audio_is_decoded_before_being_marked_ready(isolated_uploads):
    result = asyncio.run(server.api_upload_audio(filename="valid.wav", content=wav_bytes()))
    assert result.success is True
    assert result.ready_for_generation is True
    assert result.duration == pytest.approx(0.25, rel=0.06)
    assert (isolated_uploads / result.filename).read_bytes() == wav_bytes()


def test_same_filename_preserves_existing_audio_reference(isolated_uploads):
    original = wav_bytes()
    first = asyncio.run(server.api_upload_audio(filename="song.wav", content=original))
    second = asyncio.run(server.api_upload_audio(filename="song.wav", content=original + b"padding"))
    assert first.filename == "song.wav"
    assert second.filename != first.filename
    assert (isolated_uploads / first.filename).read_bytes() == original
    assert (isolated_uploads / second.filename).read_bytes() == original + b"padding"


def test_real_multipart_http_contract_rejects_invalid_and_keeps_distinct_assets(isolated_uploads):
    from fastapi.testclient import TestClient

    with TestClient(server.app) as client:
        invalid = client.post(
            "/api/upload-audio",
            files={"audio": ("unreadable.wav", b"broken", "audio/wav")},
        )
        assert invalid.status_code == 400
        assert not (isolated_uploads / "unreadable.wav").exists()

        audio = wav_bytes()
        first = client.post(
            "/api/upload-audio",
            files={"audio": ("track.wav", audio, "audio/wav")},
        )
        assert first.status_code == 200, first.text
        one = first.json()
        assert one["ready_for_generation"] is True
        assert one["duration"] == pytest.approx(0.25, rel=0.06)

        second = client.post(
            "/api/upload-audio",
            files={"audio": ("track.wav", audio + b"extra", "audio/wav")},
        )
        assert second.status_code == 200, second.text
        two = second.json()
        assert two["filename"] != one["filename"]
        assert (isolated_uploads / one["filename"]).read_bytes() == audio
        assert (isolated_uploads / two["filename"]).read_bytes() == audio + b"extra"


def test_invalid_duplicate_does_not_damage_original(isolated_uploads):
    original = wav_bytes()
    result = asyncio.run(server.api_upload_audio(filename="taken.wav", content=original))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.api_upload_audio(filename="taken.wav", content=b"corrupt"))
    assert exc.value.status_code == 400
    assert sorted(p.name for p in isolated_uploads.iterdir()) == [result.filename]
    assert (isolated_uploads / result.filename).read_bytes() == original
