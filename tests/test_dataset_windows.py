from pathlib import Path

import numpy as np

from taikonation.data.dataset import TaikoTransformerDataset
from taikonation.data.tokenization import TaikoTokenizer


def _write_pair(tmp_path: Path, *, frames: int = 12):
    features = np.arange(frames * 80, dtype=np.float32).reshape(frames, 80)
    chart = np.zeros((frames, 7), dtype=np.uint8)
    for position, event in [(0, 0), (2, 1), (4, 0), (6, 2), (8, 3), (10, 0)]:
        if position < frames:
            chart[position, event] = 1

    song_path = tmp_path / "song.npy"
    chart_path = tmp_path / "chart.npy"
    np.save(song_path, features)
    np.save(chart_path, chart)
    return {
        "song_path": str(song_path),
        "chart_path": str(chart_path),
        "genre": "unknown",
        "difficulty": "oni",
    }


def test_windowed_dataset_uses_entire_song_and_previous_token_context(tmp_path: Path):
    sample = _write_pair(tmp_path, frames=12)
    tokenizer = TaikoTokenizer(time_quantization=46.4, source_resolution=23.2)
    dataset = TaikoTransformerDataset(
        [sample],
        [0],
        tokenizer,
        {"unknown": 0},
        is_train=False,
        max_sequence_length=4,
        time_quantization_ms=46.4,
        source_resolution_ms=23.2,
        window_stride=2,
        cache_dir=str(tmp_path / "cache"),
        prefetch=False,
    )

    # 12 source frames -> 6 model tokens. A 4-token window with stride 2
    # covers the whole song with two overlapping windows.
    assert len(dataset) == 2
    assert dataset.windows == [(0, 0, 6), (0, 2, 6)]

    mean, std = dataset.compute_feature_stats()
    dataset.set_feature_stats(mean, std)

    first = dataset[0]
    second = dataset[1]
    assert first["encoder_input"].shape == (4, 80)
    assert second["encoder_input"].shape == (4, 80)
    assert not first["padding_mask"].any()
    assert first["decoder_input"][0].item() == tokenizer.vocab["[CLS]"]

    all_tokens = dataset._get_tokens(0)
    assert second["decoder_input"][0].item() == all_tokens[1]


def test_short_window_is_mean_normalized_and_padding_is_masked(tmp_path: Path):
    sample = _write_pair(tmp_path, frames=6)
    tokenizer = TaikoTokenizer(time_quantization=46.4, source_resolution=23.2)
    dataset = TaikoTransformerDataset(
        [sample],
        [0],
        tokenizer,
        {"unknown": 0},
        is_train=False,
        max_sequence_length=4,
        time_quantization_ms=46.4,
        source_resolution_ms=23.2,
        cache_dir=str(tmp_path / "cache"),
        prefetch=False,
    )

    mean, std = dataset.compute_feature_stats()
    dataset.set_feature_stats(mean, std)
    item = dataset[0]

    # 6 source frames -> 3 model tokens, then one padded position.
    assert item["padding_mask"].tolist() == [False, False, False, True]
    assert np.allclose(item["encoder_input"][-1].numpy(), 0.0)
    assert np.isfinite(item["encoder_input"].numpy()).all()
