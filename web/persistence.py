"""SQLite persistence helpers for FastAPI demo state.

This module provides a small persistence layer for server_fastapi.py so
generated charts, evaluations, and system logs survive process restarts.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


_LOCK = Lock()


class StudioStore:
    """Simple SQLite-backed store for studio entities."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with _LOCK:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS charts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        artist TEXT NOT NULL,
                        difficulty TEXT NOT NULL,
                        bpm REAL NOT NULL,
                        genre TEXT NOT NULL,
                        filename TEXT,
                        audio_filename TEXT,
                        model_id TEXT,
                        timing_mode TEXT,
                        rating REAL NOT NULL DEFAULT 0,
                        plays INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS evaluations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chart_id INTEGER NOT NULL,
                        fun INTEGER NOT NULL,
                        musicality INTEGER NOT NULL,
                        playability INTEGER NOT NULL,
                        coherence INTEGER NOT NULL,
                        comments TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(chart_id) REFERENCES charts(id)
                    );

                    CREATE TABLE IF NOT EXISTS key_value (
                        key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        progress INTEGER NOT NULL DEFAULT 0,
                        message TEXT NOT NULL DEFAULT '',
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        result_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                existing = {row["name"] for row in conn.execute("PRAGMA table_info(charts)")}
                for name, sql_type in {
                    "filename": "TEXT",
                    "audio_filename": "TEXT",
                    "model_id": "TEXT",
                    "timing_mode": "TEXT",
                }.items():
                    if name not in existing:
                        conn.execute(f"ALTER TABLE charts ADD COLUMN {name} {sql_type}")

    def append_log(self, entry: Dict[str, Any]) -> None:
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO system_logs(timestamp, level, message) VALUES(?, ?, ?)",
                    (entry["timestamp"], entry["level"], entry["message"]),
                )

    def list_logs(self, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, level, message FROM system_logs ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def create_chart(self, chart: Dict[str, Any]) -> Dict[str, Any]:
        with _LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO charts(
                        title, artist, difficulty, bpm, genre, filename,
                        audio_filename, model_id, timing_mode, rating, plays, created_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chart["title"],
                        chart["artist"],
                        chart["difficulty"],
                        float(chart["bpm"]),
                        chart["genre"],
                        chart.get("filename"),
                        chart.get("audio_filename"),
                        chart.get("model_id"),
                        chart.get("timing_mode"),
                        float(chart.get("rating", 0)),
                        int(chart.get("plays", 0)),
                        chart["created_at"],
                    ),
                )
                chart_id = int(cur.lastrowid)
        stored = dict(chart)
        stored["id"] = chart_id
        return stored

    def get_chart(self, chart_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, artist, difficulty, bpm, genre, filename,
                       audio_filename, model_id, timing_mode, rating, plays, created_at
                FROM charts WHERE id = ?
                """,
                (chart_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_charts(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, artist, difficulty, bpm, genre, filename,
                       audio_filename, model_id, timing_mode, rating, plays, created_at
                FROM charts ORDER BY id DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_charts(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM charts").fetchone()
        return int(row["count"]) if row else 0

    def get_unrated_chart(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, artist, difficulty, bpm, genre, filename,
                       audio_filename, model_id, timing_mode, rating, plays, created_at
                FROM charts
                WHERE rating <= 0
                ORDER BY id ASC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    def set_chart_rating(self, chart_id: int, rating: float) -> bool:
        with _LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE charts SET rating = ? WHERE id = ?",
                    (float(rating), chart_id),
                )
                return cur.rowcount > 0

    def submit_evaluation(
        self,
        *,
        chart_id: int,
        fun: int,
        musicality: int,
        playability: int,
        coherence: int,
        comments: str,
        created_at: str,
    ) -> bool:
        rating = (fun + musicality + playability + coherence) / 4
        with _LOCK:
            with self._connect() as conn:
                existing = conn.execute("SELECT id FROM charts WHERE id = ?", (chart_id,)).fetchone()
                if not existing:
                    return False
                conn.execute(
                    """
                    INSERT INTO evaluations(chart_id, fun, musicality, playability, coherence, comments, created_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chart_id, fun, musicality, playability, coherence, comments, created_at),
                )
                conn.execute("UPDATE charts SET rating = ? WHERE id = ?", (rating, chart_id))
        return True

    def list_evaluations(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chart_id, fun, musicality, playability, coherence, comments, created_at
                FROM evaluations ORDER BY id DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def set_json(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO key_value(key, value_json) VALUES(?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                    """,
                    (key, payload),
                )

    def get_json(self, key: str, default: Any) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM key_value WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return default

    def create_task(self, *, task_type: str, payload: Dict[str, Any], created_at: str) -> Dict[str, Any]:
        payload_json = json.dumps(payload)
        with _LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO tasks(task_type, status, progress, message, payload_json, result_json, created_at, updated_at)
                    VALUES(?, 'queued', 0, '', ?, '{}', ?, ?)
                    """,
                    (task_type, payload_json, created_at, created_at),
                )
                task_id = int(cur.lastrowid)
        return self.get_task(task_id) or {}

    def update_task(
        self,
        task_id: int,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        updated_at: str,
    ) -> bool:
        fields = []
        params: List[Any] = []
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if progress is not None:
            fields.append("progress = ?")
            params.append(int(progress))
        if message is not None:
            fields.append("message = ?")
            params.append(message)
        if result is not None:
            fields.append("result_json = ?")
            params.append(json.dumps(result))
        fields.append("updated_at = ?")
        params.append(updated_at)
        params.append(task_id)
        with _LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?",
                    tuple(params),
                )
                return cur.rowcount > 0

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, task_type, status, progress, message, payload_json, result_json, created_at, updated_at
                FROM tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
        if not row:
            return None
        task = dict(row)
        task["payload"] = json.loads(task.pop("payload_json") or "{}")
        task["result"] = json.loads(task.pop("result_json") or "{}")
        return task

    def list_tasks(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, task_type, status, progress, message, payload_json, result_json, created_at, updated_at
                FROM tasks
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        tasks: List[Dict[str, Any]] = []
        for row in rows:
            task = dict(row)
            task["payload"] = json.loads(task.pop("payload_json") or "{}")
            task["result"] = json.loads(task.pop("result_json") or "{}")
            tasks.append(task)
        return tasks

    def interrupt_unfinished_tasks(self, *, updated_at: str) -> int:
        """Mark work that cannot survive a process restart as interrupted."""
        with _LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'interrupted',
                        message = 'Interrupted by a previous server shutdown',
                        updated_at = ?
                    WHERE status IN ('queued', 'running')
                    """,
                    (updated_at,),
                )
                return int(cur.rowcount)

    def count_tasks(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM tasks").fetchone()
        return int(row["count"]) if row else 0
