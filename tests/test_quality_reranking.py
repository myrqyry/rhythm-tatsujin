import json

from taikonation.generation import quality
from taikonation.generation import softchart_backend as backend


BASE_METRICS = {
    "notes_per_second": 5.0,
    "median_ioi_ms": 180.0,
    "ka_ratio": 0.5,
    "big_note_ratio": 0.04,
    "color_change_ratio": 0.55,
    "repeated_motif_8_share": 0.35,
    "subdivision_entropy": 0.42,
    "mean_burst_length": 4.0,
    "rest_interval_share": 0.08,
    "section_density_change": 0.35,
}


def _reference_manifest(tmp_path):
    root = tmp_path / "reference"
    set_dir = root / "1"
    set_dir.mkdir(parents=True)
    maps = []
    for index, offset in enumerate((-0.2, -0.1, 0.0, 0.1, 0.2), start=1):
        metrics = {
            name: value + offset * (0.1 if name != "median_ioi_ms" else 10.0)
            for name, value in BASE_METRICS.items()
        }
        maps.append(
            {
                "beatmap_set_id": str(index),
                "title": f"Human {index}",
                "version": "Oni",
                "difficulty_mapper": "Nifty",
                "difficulty": "oni",
                **metrics,
            }
        )
    (set_dir / "manifest.json").write_text(
        json.dumps({"beatmap_set_id": "1", "maps": maps}),
        encoding="utf-8",
    )
    quality.clear_reference_cache()
    return root


def test_human_reference_score_prefers_nearby_candidate(tmp_path):
    root = _reference_manifest(tmp_path)

    near = quality.score_metrics_against_reference(
        BASE_METRICS,
        difficulty="oni",
        reference_root=root,
    )
    far_metrics = dict(BASE_METRICS)
    far_metrics["subdivision_entropy"] = 1.0
    far_metrics["notes_per_second"] = 12.0
    far = quality.score_metrics_against_reference(
        far_metrics,
        difficulty="oni",
        reference_root=root,
    )

    assert near.reference_count == 5
    assert near.k == 3
    assert near.score < far.score
    assert near.neighbors


def test_best_of_keeps_baseline_in_pool_and_selects_lower_score(monkeypatch):
    generated = [
        backend.GeneratedChart({"hits": [{"t": 0.1, "type": "don"}]}, 180, "time", "m", "cpu"),
        backend.GeneratedChart({"hits": [{"t": 0.2, "type": "don"}]}, 180, "time", "m", "cpu"),
        backend.GeneratedChart({"hits": [{"t": 0.3, "type": "don"}]}, 180, "time", "m", "cpu"),
    ]
    calls = []

    def fake_generate(*args, **kwargs):
        calls.append((kwargs["greedy"], kwargs["seed"]))
        return generated[len(calls) - 1]

    scores = iter([0.8, 0.3, 0.5])

    class FakeScore:
        def __init__(self, score):
            self.score = score
            self.neighbors = ({"title": "Human"},)

    monkeypatch.setattr(backend, "generate_from_audio", fake_generate)
    monkeypatch.setattr(
        quality,
        "score_generated_events",
        lambda *args, **kwargs: FakeScore(next(scores)),
    )

    result = backend.generate_best_of_audio(
        "unused.wav",
        course="oni",
        candidate_count=3,
    )

    assert calls == [(True, 0), (False, 2001), (False, 2002)]
    assert result.events == generated[1].events
    assert result.quality_score == 0.3
    assert result.candidate_count == 3
    assert result.candidate_seed == 2001
    assert [item["seed"] for item in result.candidate_scores] == [0, 2001, 2002]
