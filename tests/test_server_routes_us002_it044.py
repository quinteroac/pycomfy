"""Tests for GET /jobs/{job_id} — US-002 it_000044: Job status polling.

Covers:
  AC01 — Returns JSON with fields: id, status, created_at, updated_at, result (null or JobResult)
  AC02 — Returns HTTP 404 with {"detail": "job not found"} when job ID does not exist
  AC03 — Status values match: "queued", "running", "completed", "failed", "cancelled"
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from server.app import app
from server.schemas import JobResult, JobStatusResponse

_QUEUE_PATCH = "server.app.get_queue"


def _make_row(
    *,
    job_id: str = "abc-123",
    status: str = "completed",
    result: dict | None = None,
    created_at: str = "2024-01-01T00:00:00+00:00",
    updated_at: str = "2024-01-01T00:01:00+00:00",
) -> dict:
    return {
        "id": job_id,
        "status": status,
        "data": "{}",
        "result": json.dumps(result) if result is not None else None,
        "progress": None,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _mock_queue(row: dict | None):
    """Return a context manager that patches get_queue to return a mock queue."""
    mock_queue = MagicMock()
    mock_queue.get = AsyncMock(return_value=row)

    async def _get_queue_fn(*args, **kwargs):
        return mock_queue

    return patch(_QUEUE_PATCH, side_effect=_get_queue_fn)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# AC01 — Response shape
# ---------------------------------------------------------------------------

class TestAC01ResponseShape:
    def test_returns_all_required_fields(self, client):
        row = _make_row(result={"output_path": "/out/result.png"})
        with _mock_queue(row):
            resp = client.get("/jobs/abc-123")
        assert resp.status_code == 200
        body = resp.json()
        assert "id" in body
        assert "status" in body
        assert "created_at" in body
        assert "updated_at" in body
        assert "result" in body

    def test_id_matches_requested_job(self, client):
        row = _make_row(job_id="my-job-xyz")
        with _mock_queue(row):
            resp = client.get("/jobs/my-job-xyz")
        assert resp.json()["id"] == "my-job-xyz"

    def test_result_is_null_when_not_set(self, client):
        row = _make_row(status="queued", result=None)
        with _mock_queue(row):
            resp = client.get("/jobs/abc-123")
        assert resp.json()["result"] is None

    def test_result_contains_output_path(self, client):
        row = _make_row(result={"output_path": "/srv/output/video.mp4"})
        with _mock_queue(row):
            resp = client.get("/jobs/abc-123")
        assert resp.json()["result"]["output_path"] == "/srv/output/video.mp4"

    def test_result_contains_error(self, client):
        row = _make_row(status="failed", result={"error": "OOM"})
        with _mock_queue(row):
            resp = client.get("/jobs/abc-123")
        assert resp.json()["result"]["error"] == "OOM"

    def test_response_is_valid_job_status_response(self, client):
        row = _make_row(result={"output_path": "/out/img.png"})
        with _mock_queue(row):
            resp = client.get("/jobs/abc-123")
        parsed = JobStatusResponse(**resp.json())
        assert isinstance(parsed.result, JobResult)

    def test_timestamps_are_preserved(self, client):
        row = _make_row(
            created_at="2025-06-01T10:00:00+00:00",
            updated_at="2025-06-01T10:05:30+00:00",
        )
        with _mock_queue(row):
            resp = client.get("/jobs/abc-123")
        body = resp.json()
        assert body["created_at"] == "2025-06-01T10:00:00+00:00"
        assert body["updated_at"] == "2025-06-01T10:05:30+00:00"


# ---------------------------------------------------------------------------
# AC02 — 404 for missing job
# ---------------------------------------------------------------------------

class TestAC02NotFound:
    def test_returns_404_when_job_does_not_exist(self, client):
        with _mock_queue(None):
            resp = client.get("/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_404_body_has_detail_field(self, client):
        with _mock_queue(None):
            resp = client.get("/jobs/nonexistent-job-id")
        assert resp.json() == {"detail": "job not found"}


# ---------------------------------------------------------------------------
# AC03 — Status values
# ---------------------------------------------------------------------------

class TestAC03StatusValues:
    @pytest.mark.parametrize("stored_status,expected_status", [
        ("pending", "queued"),
        ("queued", "queued"),
        ("running", "running"),
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
    ])
    def test_status_mapping(self, client, stored_status, expected_status):
        row = _make_row(status=stored_status)
        with _mock_queue(row):
            resp = client.get("/jobs/abc-123")
        assert resp.status_code == 200
        assert resp.json()["status"] == expected_status

    def test_queued_status_when_pending_in_db(self, client):
        """Newly submitted jobs have status 'pending' in DB; should appear as 'queued'."""
        row = _make_row(status="pending")
        with _mock_queue(row):
            resp = client.get("/jobs/abc-123")
        assert resp.json()["status"] == "queued"
