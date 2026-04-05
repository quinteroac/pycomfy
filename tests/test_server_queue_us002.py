"""Tests for server/queue.py — US-002: SQLite job queue singleton."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from server.jobs import JobData
from server.queue import JobQueue, _reset_singleton, get_queue


def _sample_job_data() -> JobData:
    return JobData(
        action="generate",
        media="image",
        model="sdxl",
        script="image/sdxl",
        args={"width": 512, "height": 512},
        script_base="/tmp",
        uv_path="/usr/bin/uv",
    )


def _run(coro):
    """Helper to run a coroutine in a fresh event loop."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _tmp_queue(tmp_path: Path) -> JobQueue:
    """Return a fresh JobQueue backed by a temp DB, resetting the singleton."""
    _reset_singleton()

    async def _init():
        return await get_queue(db_path=tmp_path / "jobs.db")

    return _run(_init())


# ---------------------------------------------------------------------------
# AC01 — get_queue() returns a JobQueue wrapping aiosqlite
# ---------------------------------------------------------------------------

class TestAC01GetQueue:
    def test_get_queue_returns_job_queue_instance(self, tmp_path):
        q = _tmp_queue(tmp_path)
        assert isinstance(q, JobQueue)

    def test_job_queue_has_db_connection(self, tmp_path):
        import aiosqlite

        q = _tmp_queue(tmp_path)
        assert isinstance(q._db, aiosqlite.Connection)


# ---------------------------------------------------------------------------
# AC02 — lazy async singleton
# ---------------------------------------------------------------------------

class TestAC02Singleton:
    def test_same_instance_on_second_call(self, tmp_path):
        _reset_singleton()
        db_path = tmp_path / "jobs.db"

        async def _two_calls():
            q1 = await get_queue(db_path=db_path)
            q2 = await get_queue(db_path=db_path)
            return q1, q2

        q1, q2 = _run(_two_calls())
        assert q1 is q2

    def test_db_file_created_on_first_call(self, tmp_path):
        db_path = tmp_path / "sub" / "jobs.db"
        _reset_singleton()

        async def _init():
            return await get_queue(db_path=db_path)

        _run(_init())
        assert db_path.exists()

    def test_parent_dirs_created_automatically(self, tmp_path):
        db_path = tmp_path / "deep" / "nested" / "jobs.db"
        _reset_singleton()

        async def _init():
            return await get_queue(db_path=db_path)

        _run(_init())
        assert db_path.parent.exists()

    def test_reset_singleton_allows_new_instance(self, tmp_path):
        db_path = tmp_path / "jobs.db"
        _reset_singleton()

        async def _get():
            return await get_queue(db_path=db_path)

        q1 = _run(_get())
        _reset_singleton()
        q2 = _run(_get())
        assert q1 is not q2


# ---------------------------------------------------------------------------
# AC03 — jobs table schema
# ---------------------------------------------------------------------------

class TestAC03TableSchema:
    def test_jobs_table_exists(self, tmp_path):
        q = _tmp_queue(tmp_path)

        async def _check():
            async with q._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
            ) as cur:
                return await cur.fetchone()

        row = _run(_check())
        assert row is not None
        assert row["name"] == "jobs"

    def _get_columns(self, q: JobQueue) -> dict[str, str]:
        async def _fetch():
            async with q._db.execute("PRAGMA table_info(jobs)") as cur:
                rows = await cur.fetchall()
            return {r["name"]: r["type"] for r in rows}

        return _run(_fetch())

    def test_column_id_exists(self, tmp_path):
        q = _tmp_queue(tmp_path)
        cols = self._get_columns(q)
        assert "id" in cols

    def test_column_status_exists(self, tmp_path):
        q = _tmp_queue(tmp_path)
        cols = self._get_columns(q)
        assert "status" in cols

    def test_column_data_exists(self, tmp_path):
        q = _tmp_queue(tmp_path)
        cols = self._get_columns(q)
        assert "data" in cols

    def test_column_result_exists(self, tmp_path):
        q = _tmp_queue(tmp_path)
        cols = self._get_columns(q)
        assert "result" in cols

    def test_column_created_at_exists(self, tmp_path):
        q = _tmp_queue(tmp_path)
        cols = self._get_columns(q)
        assert "created_at" in cols

    def test_column_updated_at_exists(self, tmp_path):
        q = _tmp_queue(tmp_path)
        cols = self._get_columns(q)
        assert "updated_at" in cols

    def test_id_is_primary_key(self, tmp_path):
        q = _tmp_queue(tmp_path)

        async def _fetch():
            async with q._db.execute("PRAGMA table_info(jobs)") as cur:
                rows = await cur.fetchall()
            return {r["name"]: r["pk"] for r in rows}

        pk_map = _run(_fetch())
        assert pk_map["id"] == 1  # pk=1 means it is the primary key

    def test_result_column_is_nullable(self, tmp_path):
        """result should accept NULL — verified by inserting with no result."""
        q = _tmp_queue(tmp_path)
        data = _sample_job_data()

        async def _check():
            job_id = await q.enqueue(data)
            row = await q.get(job_id)
            return row["result"]

        assert _run(_check()) is None


# ---------------------------------------------------------------------------
# AC04 — JobQueue methods
# ---------------------------------------------------------------------------

class TestAC04JobQueueMethods:
    # enqueue
    def test_enqueue_returns_string_id(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_enqueue_stores_pending_status(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        row = _run(q.get(job_id))
        assert row["status"] == "pending"

    def test_enqueue_stores_data_json(self, tmp_path):
        import json

        q = _tmp_queue(tmp_path)
        data = _sample_job_data()
        job_id = _run(q.enqueue(data))
        row = _run(q.get(job_id))
        stored = json.loads(row["data"])
        assert stored["action"] == "generate"
        assert stored["model"] == "sdxl"

    def test_enqueue_sets_created_at(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        row = _run(q.get(job_id))
        assert row["created_at"] is not None

    def test_enqueue_sets_updated_at(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        row = _run(q.get(job_id))
        assert row["updated_at"] is not None

    def test_enqueue_result_is_none_initially(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        row = _run(q.get(job_id))
        assert row["result"] is None

    # get
    def test_get_returns_dict_for_existing_job(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        row = _run(q.get(job_id))
        assert isinstance(row, dict)
        assert row["id"] == job_id

    def test_get_returns_none_for_missing_job(self, tmp_path):
        q = _tmp_queue(tmp_path)
        assert _run(q.get("nonexistent-id")) is None

    # update_status
    def test_update_status_changes_status(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        _run(q.update_status(job_id, "running"))
        row = _run(q.get(job_id))
        assert row["status"] == "running"

    def test_update_status_stores_result_json(self, tmp_path):
        import json

        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        result = {"output_path": "/tmp/out.png"}
        _run(q.update_status(job_id, "done", result=result))
        row = _run(q.get(job_id))
        assert json.loads(row["result"]) == result

    def test_update_status_result_none_leaves_null(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        _run(q.update_status(job_id, "running"))
        row = _run(q.get(job_id))
        assert row["result"] is None

    def test_update_status_updates_updated_at(self, tmp_path):
        import time

        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        row_before = _run(q.get(job_id))
        time.sleep(0.01)
        _run(q.update_status(job_id, "done"))
        row_after = _run(q.get(job_id))
        assert row_after["updated_at"] >= row_before["updated_at"]

    # list_jobs
    def test_list_jobs_returns_list(self, tmp_path):
        q = _tmp_queue(tmp_path)
        result = _run(q.list_jobs())
        assert isinstance(result, list)

    def test_list_jobs_contains_enqueued_jobs(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        jobs = _run(q.list_jobs())
        ids = [j["id"] for j in jobs]
        assert job_id in ids

    def test_list_jobs_respects_limit(self, tmp_path):
        q = _tmp_queue(tmp_path)
        for _ in range(5):
            _run(q.enqueue(_sample_job_data()))
        jobs = _run(q.list_jobs(limit=3))
        assert len(jobs) <= 3

    def test_list_jobs_default_limit_is_50(self, tmp_path):
        q = _tmp_queue(tmp_path)
        for _ in range(60):
            _run(q.enqueue(_sample_job_data()))
        jobs = _run(q.list_jobs())
        assert len(jobs) == 50

    def test_list_jobs_ordered_newest_first(self, tmp_path):
        q = _tmp_queue(tmp_path)
        ids = [_run(q.enqueue(_sample_job_data())) for _ in range(3)]
        jobs = _run(q.list_jobs())
        returned_ids = [j["id"] for j in jobs]
        # Newest first means last-enqueued id appears before first-enqueued
        assert returned_ids.index(ids[-1]) < returned_ids.index(ids[0])

    # cancel
    def test_cancel_returns_true_for_pending_job(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        assert _run(q.cancel(job_id)) is True

    def test_cancel_sets_status_to_cancelled(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        _run(q.cancel(job_id))
        row = _run(q.get(job_id))
        assert row["status"] == "cancelled"

    def test_cancel_returns_false_for_done_job(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        _run(q.update_status(job_id, "done"))
        assert _run(q.cancel(job_id)) is False

    def test_cancel_returns_false_for_failed_job(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        _run(q.update_status(job_id, "failed"))
        assert _run(q.cancel(job_id)) is False

    def test_cancel_returns_false_for_nonexistent_job(self, tmp_path):
        q = _tmp_queue(tmp_path)
        assert _run(q.cancel("nonexistent")) is False

    def test_cancel_returns_false_for_already_cancelled_job(self, tmp_path):
        q = _tmp_queue(tmp_path)
        job_id = _run(q.enqueue(_sample_job_data()))
        _run(q.cancel(job_id))
        assert _run(q.cancel(job_id)) is False
