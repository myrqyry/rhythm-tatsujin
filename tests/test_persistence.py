from pathlib import Path

from web.persistence import StudioStore


def test_store_chart_and_evaluation(tmp_path: Path):
    store = StudioStore(tmp_path / "studio.sqlite3")

    chart = store.create_chart(
        {
            "title": "Song",
            "artist": "Artist",
            "difficulty": "oni",
            "bpm": 180,
            "genre": "electronic",
            "quality_score": 0.91,
            "candidate_count": 4,
            "candidate_seed": 2002,
            "candidate_scores": [
                {"seed": 0, "greedy": True, "score": 1.1},
                {"seed": 2002, "greedy": False, "score": 0.91},
            ],
            "rating": 0,
            "plays": 0,
            "created_at": "2026-03-21T00:00:00",
        }
    )

    assert chart["id"] > 0
    unrated = store.get_unrated_chart()
    assert unrated is not None
    assert unrated["id"] == chart["id"]

    ok = store.submit_evaluation(
        chart_id=chart["id"],
        fun=4,
        musicality=5,
        playability=3,
        coherence=4,
        comments="good flow",
        created_at="2026-03-21T00:01:00",
    )
    assert ok is True

    charts = store.list_charts()
    assert charts[0]["rating"] == 4.0
    assert charts[0]["quality_score"] == 0.91
    assert charts[0]["candidate_count"] == 4
    assert charts[0]["candidate_seed"] == 2002
    assert charts[0]["candidate_scores"][1]["score"] == 0.91
    evaluations = store.list_evaluations()
    assert evaluations[0]["comments"] == "good flow"


def test_store_task_lifecycle(tmp_path: Path):
    store = StudioStore(tmp_path / "studio.sqlite3")
    task = store.create_task(
        task_type="generation",
        payload={"title": "Song"},
        created_at="2026-03-21T00:00:00",
    )
    assert task["status"] == "queued"
    assert task["progress"] == 0
    assert task["payload"]["title"] == "Song"

    updated = store.update_task(
        task["id"],
        status="running",
        progress=40,
        message="working",
        updated_at="2026-03-21T00:01:00",
    )
    assert updated is True
    task_after = store.get_task(task["id"])
    assert task_after is not None
    assert task_after["status"] == "running"
    assert task_after["progress"] == 40

    store.update_task(
        task["id"],
        status="completed",
        progress=100,
        message="done",
        result={"chart_id": 7},
        updated_at="2026-03-21T00:02:00",
    )
    task_done = store.get_task(task["id"])
    assert task_done is not None
    assert task_done["result"]["chart_id"] == 7


def test_store_pagination_and_counts(tmp_path: Path):
    store = StudioStore(tmp_path / "studio.sqlite3")
    for i in range(3):
        store.create_chart(
            {
                "title": f"Song {i}",
                "artist": "Artist",
                "difficulty": "oni",
                "bpm": 180,
                "genre": "electronic",
                "rating": 0,
                "plays": 0,
                "created_at": "2026-03-21T00:00:00",
            }
        )
    assert store.count_charts() == 3
    page = store.list_charts(limit=2, offset=1)
    assert len(page) == 2

    for i in range(4):
        store.create_task(task_type="generation", payload={"i": i}, created_at="2026-03-21T00:00:00")
    assert store.count_tasks() == 4
    tasks_page = store.list_tasks(limit=2, offset=1)
    assert len(tasks_page) == 2

def test_unfinished_tasks_are_marked_interrupted_after_restart(tmp_path: Path):
    store = StudioStore(tmp_path / "studio.sqlite3")
    created = "2026-10-05T00:00:00+00:00"

    queued = store.create_task(task_type="generation", payload={}, created_at=created)
    running = store.create_task(task_type="training", payload={}, created_at=created)
    complete = store.create_task(task_type="generation", payload={}, created_at=created)

    store.update_task(
        running["id"],
        status="running",
        message="working",
        updated_at=created,
    )
    store.update_task(
        complete["id"],
        status="completed",
        progress=100,
        message="done",
        updated_at=created,
    )

    interrupted = store.interrupt_unfinished_tasks(updated_at="2026-10-05T00:01:00+00:00")

    assert interrupted == 2
    assert store.get_task(queued["id"])["status"] == "interrupted"
    assert store.get_task(running["id"])["status"] == "interrupted"
    assert store.get_task(complete["id"])["status"] == "completed"
