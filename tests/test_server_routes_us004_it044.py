"""Tests for GET /jobs and DELETE /jobs/{job_id} — US-004 it_000044: Job list and cancel.

Covers:
  AC01 — GET /jobs returns a JSON array of up to 50 jobs with id, status, created_at, updated_at
  AC02 — DELETE /jobs/{job_id} cancels a queued job and returns {"cancelled": true}
  AC03 — DELETE /jobs/{job_id} returns HTTP 409 when job is not in "queued" status
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from server.app import app

_QUEUE_PATCH = "server.app.get_queue"


def _make_row(
    *,
    job_id: str = "job-001",
    status: str = "pending",
    created_at: str = "2024-01-01T00:00:00+00:00",
    updated_at: str = "2024-01-01T00:01:00+00:00",
) -> dict:
    return {
        "id": job_id,
        "status": status,
        "data": "{}",
        "result": None,
        "progress": None,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _mock_queue_for_list(rows: list[dict]):
    mock_queue = MagicMock()
    mock_queue.list_jobs = AsyncMock(return_value=rows)

    async def _get_queue_fn(*args, **kwargs):
        return mock_queue

    return patch(_QUEUE_PATCH, side_effect=_get_queue_fn)


def _mock_queue_for_delete(row: dict | None, *, expect_cancel: bool = False):
    mock_queue = MagicMock()
    mock_queue.get = AsyncMock(return_value=row)
    mock_queue.update_status = AsyncMock(return_value=None)

    # get_queue is called twice in cancel_job (once for get, once for update_status)
    call_count = 0

    async def _get_queue_fn(*args, **kwargs):
        return mock_queue

    return patch(_QUEUE_PATCH, side_effect=_get_queue_fn)


# ---------------------------------------------------------------------------
# AC01 — GET /jobs
# ---------------------------------------------------------------------------

class TestListJobs:
    def test_returns_empty_array_when_no_jobs(self):
        with _mock_queue_for_list([]):
            client = TestClient(app)
            resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_array_with_required_fields(self):
        rows = [_make_row(job_id=f"job-{i:03d}") for i in range(3)]
        with _mock_queue_for_list(rows):
            client = TestClient(app)
            resp = client.get("/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        for item in data:
            assert "id" in item
            assert "status" in item
            assert "created_at" in item
            assert "updated_at" in item

    def test_maps_pending_status_to_queued(self):
        rows = [_make_row(status="pending")]
        with _mock_queue_for_list(rows):
            client = TestClient(app)
            resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json()[0]["status"] == "queued"

    def test_passes_through_non_pending_statuses(self):
        statuses = ["running", "completed", "failed", "cancelled"]
        rows = [_make_row(job_id=f"job-{s}", status=s) for s in statuses]
        with _mock_queue_for_list(rows):
            client = TestClient(app)
            resp = client.get("/jobs")
        assert resp.status_code == 200
        result_statuses = {item["id"]: item["status"] for item in resp.json()}
        for s in statuses:
            assert result_statuses[f"job-{s}"] == s

    def test_returns_at_most_50_jobs(self):
        rows = [_make_row(job_id=f"job-{i:03d}") for i in range(50)]
        with _mock_queue_for_list(rows):
            client = TestClient(app)
            resp = client.get("/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 50

    def test_correct_id_in_response(self):
        rows = [_make_row(job_id="my-special-job")]
        with _mock_queue_for_list(rows):
            client = TestClient(app)
            resp = client.get("/jobs")
        assert resp.json()[0]["id"] == "my-special-job"

    def test_correct_timestamps_in_response(self):
        rows = [_make_row(
            created_at="2024-06-01T10:00:00+00:00",
            updated_at="2024-06-01T10:05:00+00:00",
        )]
        with _mock_queue_for_list(rows):
            client = TestClient(app)
            resp = client.get("/jobs")
        item = resp.json()[0]
        assert item["created_at"] == "2024-06-01T10:00:00+00:00"
        assert item["updated_at"] == "2024-06-01T10:05:00+00:00"

    def test_no_extra_fields_in_response(self):
        rows = [_make_row()]
        with _mock_queue_for_list(rows):
            client = TestClient(app)
            resp = client.get("/jobs")
        item = resp.json()[0]
        assert set(item.keys()) == {"id", "status", "created_at", "updated_at"}

    def test_list_jobs_called_with_limit_50(self):
        mock_queue = MagicMock()
        mock_queue.list_jobs = AsyncMock(return_value=[])

        async def _get_queue_fn(*args, **kwargs):
            return mock_queue

        with patch(_QUEUE_PATCH, side_effect=_get_queue_fn):
            client = TestClient(app)
            client.get("/jobs")

        mock_queue.list_jobs.assert_called_once_with(limit=50)


# ---------------------------------------------------------------------------
# AC02 — DELETE /jobs/{job_id} — queued job
# ---------------------------------------------------------------------------

class TestCancelJob:
    def test_cancel_queued_job_returns_cancelled_true(self):
        row = _make_row(status="pending")
        with _mock_queue_for_delete(row):
            client = TestClient(app)
            resp = client.delete("/jobs/job-001")
        assert resp.status_code == 200
        assert resp.json() == {"cancelled": True}

    def test_cancel_queued_job_calls_update_status(self):
        mock_queue = MagicMock()
        row = _make_row(status="pending")
        mock_queue.get = AsyncMock(return_value=row)
        mock_queue.update_status = AsyncMock(return_value=None)

        async def _get_queue_fn(*args, **kwargs):
            return mock_queue

        with patch(_QUEUE_PATCH, side_effect=_get_queue_fn):
            client = TestClient(app)
            client.delete("/jobs/job-001")

        mock_queue.update_status.assert_called_once_with("job-001", "cancelled")

    def test_cancel_nonexistent_job_returns_404(self):
        with _mock_queue_for_delete(None):
            client = TestClient(app)
            resp = client.delete("/jobs/does-not-exist")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "job not found"

    def test_cancel_response_body_has_boolean_not_string(self):
        row = _make_row(status="pending")
        with _mock_queue_for_delete(row):
            client = TestClient(app)
            resp = client.delete("/jobs/job-001")
        assert resp.json()["cancelled"] is True


# ---------------------------------------------------------------------------
# AC03 — DELETE /jobs/{job_id} — non-queued job → 409
# ---------------------------------------------------------------------------

class TestCancelNonQueuedJob:
    @pytest.mark.parametrize("status", ["running", "completed", "failed", "cancelled"])
    def test_non_queued_status_returns_409(self, status: str):
        row = _make_row(status=status)
        with _mock_queue_for_delete(row):
            client = TestClient(app)
            resp = client.delete(f"/jobs/job-001")
        assert resp.status_code == 409

    @pytest.mark.parametrize("status", ["running", "completed", "failed", "cancelled"])
    def test_non_queued_status_returns_correct_detail(self, status: str):
        row = _make_row(status=status)
        with _mock_queue_for_delete(row):
            client = TestClient(app)
            resp = client.delete("/jobs/job-001")
        assert resp.json()["detail"] == "job is already running or completed"

    @pytest.mark.parametrize("status", ["running", "completed", "failed", "cancelled"])
    def test_non_queued_status_does_not_call_update_status(self, status: str):
        mock_queue = MagicMock()
        row = _make_row(status=status)
        mock_queue.get = AsyncMock(return_value=row)
        mock_queue.update_status = AsyncMock(return_value=None)

        async def _get_queue_fn(*args, **kwargs):
            return mock_queue

        with patch(_QUEUE_PATCH, side_effect=_get_queue_fn):
            client = TestClient(app)
            client.delete("/jobs/job-001")

        mock_queue.update_status.assert_not_called()
