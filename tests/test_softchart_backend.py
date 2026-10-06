from taikonation.generation.softchart_backend import postprocess_events


def test_postprocess_events_removes_later_hits_within_23ms():
    events = {
        "hits": [
            {"t": 1.000, "type": "don"},
            {"t": 1.010, "type": "ka"},
            {"t": 1.023, "type": "don_big"},
            {"t": 1.200, "type": "ka_big"},
        ],
        "spans": [],
    }

    cleaned, report = postprocess_events(events)

    assert [hit["t"] for hit in cleaned["hits"]] == [1.000, 1.200]
    assert report["close_hits_removed"] == 2


def test_postprocess_events_sorts_hits_and_preserves_extra_fields():
    events = {
        "hits": [
            {"t": 2.0, "type": "ka"},
            {"t": 1.0, "type": "don"},
        ],
        "spans": [{"t0": 3.0, "t1": 4.0, "type": "roll"}],
        "meta": {"keep": True},
    }

    cleaned, report = postprocess_events(events)

    assert [hit["t"] for hit in cleaned["hits"]] == [1.0, 2.0]
    assert cleaned["spans"] == events["spans"]
    assert cleaned["meta"] == {"keep": True}
    assert report["close_hits_removed"] == 0


def test_postprocess_events_drops_nonfinite_hits_without_touching_valid_hits():
    events = {
        "hits": [
            {"t": float("nan"), "type": "don"},
            {"t": 0.5, "type": "ka"},
            {"t": float("inf"), "type": "don"},
        ],
        "spans": [],
    }

    cleaned, report = postprocess_events(events)

    assert cleaned["hits"] == [{"t": 0.5, "type": "ka"}]
    assert report["invalid_hits_removed"] == 2
