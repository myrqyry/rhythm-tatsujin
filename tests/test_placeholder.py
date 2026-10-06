from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _sample_id(path: Path) -> str:
    return path.name.split("_", 1)[0].split(" ", 1)[0]


def test_bundled_corpus_has_paired_audio_and_chart_arrays():
    songs = sorted((ROOT / "input_songs").glob("*.npy"))
    charts = sorted((ROOT / "input_charts_nr").glob("*.npy"))

    assert songs, "Bundled audio-feature corpus is empty"
    assert charts, "Bundled chart corpus is empty"
    assert {_sample_id(path) for path in songs} == {_sample_id(path) for path in charts}

    for song in songs[:5]:
        features = np.load(song, mmap_mode="r")
        assert features.ndim == 2
        assert features.shape[1] == 80
        assert np.issubdtype(features.dtype, np.floating)

    for chart in charts[:5]:
        events = np.load(chart, mmap_mode="r")
        assert events.ndim == 2
        assert events.shape[1] == 7
        assert set(np.unique(events)).issubset({0, 1})
