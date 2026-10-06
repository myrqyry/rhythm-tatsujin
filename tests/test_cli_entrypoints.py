from types import SimpleNamespace
import sys

from taikonation import cli


def test_generate_entrypoint(monkeypatch, tmp_path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")
    output = tmp_path / "chart.osu"
    calls = {}

    def fake_generate_from_audio(audio_path, **kwargs):
        calls["audio"] = audio_path
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            events={"hits": [], "spans": []},
            bpm=123.0,
            model_id="test/model",
            device="cpu",
            timing_mode="time",
        )

    def fake_save(events, output_path, **kwargs):
        calls["output"] = output_path
        calls["save_kwargs"] = kwargs

    import taikonation.generation.softchart_backend as backend
    import taikonation.generation.osu_export as exporter

    monkeypatch.setattr(backend, "generate_from_audio", fake_generate_from_audio)
    monkeypatch.setattr(exporter, "save_timestamped_osu", fake_save)
    monkeypatch.setattr(
        sys,
        "argv",
        ["taikonation-generate", str(audio), str(output), "--difficulty", "oni"],
    )

    cli.generate()

    assert calls["audio"] == audio
    assert calls["output"] == output
    assert calls["kwargs"]["course"] == "oni"
    assert calls["save_kwargs"]["bpm"] == 123.0


def test_serve_entrypoint(monkeypatch):
    calls = []

    def _fake_run(app, host, port, reload):
        calls.append((app, host, port, reload))

    monkeypatch.setattr("uvicorn.run", _fake_run)
    cli.serve()
    assert calls == [("web.server_fastapi:socket_app", "127.0.0.1", 5000, False)]
