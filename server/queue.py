from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

from server.jobs import JobData

_DB_PATH = Path.home() / ".config" / "parallax" / "jobs.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    status     TEXT,
    data       TEXT,
    result     TEXT,
    created_at TEXT,
    updated_at TEXT
)
"""

_instance: "JobQueue | None" = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: "aiosqlite.Row") -> dict:
    return dict(row)


class JobQueue:
    def __init__(self, db: "aiosqlite.Connection") -> None:
        self._db = db

    async def enqueue(self, data: JobData) -> str:
        job_id = str(uuid.uuid4())
        now = _now()
        await self._db.execute(
            "INSERT INTO jobs (id, status, data, result, created_at, updated_at) VALUES (?, ?, ?, NULL, ?, ?)",
            (job_id, "pending", data.model_dump_json(), now, now),
        )
        await self._db.commit()
        return job_id

    async def get(self, job_id: str) -> dict | None:
        async with self._db.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    async def update_status(
        self, job_id: str, status: str, result: dict | None = None
    ) -> None:
        result_json = json.dumps(result) if result is not None else None
        await self._db.execute(
            "UPDATE jobs SET status = ?, result = ?, updated_at = ? WHERE id = ?",
            (status, result_json, _now(), job_id),
        )
        await self._db.commit()

    async def list_jobs(self, limit: int = 50) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def cancel(self, job_id: str) -> bool:
        async with self._db.execute(
            "SELECT status FROM jobs WHERE id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return False
        current_status = row["status"]
        if current_status in ("done", "failed", "cancelled"):
            return False
        await self._db.execute(
            "UPDATE jobs SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (_now(), job_id),
        )
        await self._db.commit()
        return True


async def get_queue(db_path: Path | None = None) -> "JobQueue":
    """Return the shared async JobQueue singleton, initialising it on first call."""
    global _instance
    if _instance is None:
        import aiosqlite

        path = db_path or _DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        db: aiosqlite.Connection = await aiosqlite.connect(path)
        db.row_factory = aiosqlite.Row
        await db.execute(_CREATE_TABLE)
        await db.commit()
        _instance = JobQueue(db)
    return _instance


def _reset_singleton() -> None:
    """Reset the singleton for testing purposes only."""
    global _instance
    _instance = None
