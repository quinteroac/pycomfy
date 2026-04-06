"""Tests for US-003: SSE progress stream — GET /jobs/{job_id}/stream."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from server.app import app
from server.jobs import PythonProgress

client = TestClient(app)

_JOB_ID = "test-job-stream-001"


def _make_row(status: str, progress: str | None = None, result: str | None = None) -> dict:
    return {
        "id": _JOB_ID,
        "status": status,
        "data": "{}",
        "result": result,
        "progress": progress,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:01+00:00",
    }


def _mock_queue(get_side_effect) -> MagicMock:
    queue = MagicMock()
    queue.get = AsyncMock(side_effect=get_side_effect)
    return queue


# ---------------------------------------------------------------------------
# AC04: 404 when job does not exist
# ---------------------------------------------------------------------------

def test_stream_returns_404_for_unknown_job():
    """AC04 — endpoint returns HTTP 404 immediately if job does not exist."""
    queue = _mock_queue([None])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/nonexistent-job/stream")
    assert response.status_code == 404


def test_stream_404_does_not_open_stream():
    """AC04 — no streaming body is returned for a missing job."""
    queue = _mock_queue([None])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/missing/stream")
    assert response.status_code == 404
    detail = response.json()
    assert "not found" in detail["detail"]


# ---------------------------------------------------------------------------
# AC01: text/event-stream content type
# ---------------------------------------------------------------------------

def test_stream_returns_text_event_stream_content_type():
    """AC01 — response content-type is text/event-stream."""
    queue = _mock_queue([
        _make_row("completed"),   # initial check
        _make_row("completed"),   # inside generator
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/{_JOB_ID}/stream")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_stream_uses_streaming_response():
    """AC01 — the endpoint returns a StreamingResponse (via content-type header)."""
    queue = _mock_queue([
        _make_row("completed"),
        _make_row("completed"),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/{_JOB_ID}/stream")
    # StreamingResponse sets text/event-stream for SSE endpoints
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/event-stream" in content_type


# ---------------------------------------------------------------------------
# AC02: each event is JSON-encoded PythonProgress as "data: <json>\n\n"
# ---------------------------------------------------------------------------

def test_stream_events_are_sse_formatted():
    """AC02 — emitted lines follow the 'data: <json>\\n\\n' SSE format."""
    progress = PythonProgress(step="sampling", pct=0.5)
    queue = _mock_queue([
        _make_row("running", progress=progress.model_dump_json()),  # initial
        _make_row("running", progress=progress.model_dump_json()),  # poll 1
        _make_row("completed"),  # poll 2 — terminates stream
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    assert response.status_code == 200
    raw = response.text
    # All non-empty lines must start with "data: "
    data_lines = [ln for ln in raw.split("\n") if ln.strip()]
    for line in data_lines:
        assert line.startswith("data: "), f"Expected SSE 'data: ' prefix, got: {line!r}"


def test_stream_events_contain_valid_python_progress_json():
    """AC02 — the payload after 'data: ' is a valid JSON-encoded PythonProgress."""
    progress = PythonProgress(step="sampling", pct=0.3, frame=10, total=50)
    queue = _mock_queue([
        _make_row("running", progress=progress.model_dump_json()),
        _make_row("running", progress=progress.model_dump_json()),
        _make_row("completed"),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    raw = response.text
    events = [ln[len("data: "):] for ln in raw.split("\n") if ln.startswith("data: ")]
    # At least the progress event and the final "done" event
    assert len(events) >= 2
    for ev in events:
        parsed = json.loads(ev)
        # Must have step and pct fields (PythonProgress schema)
        assert "step" in parsed
        assert "pct" in parsed


def test_stream_progress_event_matches_stored_progress():
    """AC02 — the emitted progress JSON matches what is stored in the DB row."""
    progress = PythonProgress(step="vae_decode", pct=0.8, frame=40, total=50)
    queue = _mock_queue([
        _make_row("running", progress=progress.model_dump_json()),
        _make_row("running", progress=progress.model_dump_json()),
        _make_row("completed"),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    events = [
        json.loads(ln[len("data: "):])
        for ln in response.text.split("\n")
        if ln.startswith("data: ")
    ]
    progress_events = [e for e in events if e["step"] != "done" and e["step"] != "error"]
    assert len(progress_events) >= 1
    assert progress_events[0]["step"] == "vae_decode"
    assert progress_events[0]["pct"] == pytest.approx(0.8)
    assert progress_events[0]["frame"] == 40
    assert progress_events[0]["total"] == 50


# ---------------------------------------------------------------------------
# AC03: final event with step "done" for completed jobs
# ---------------------------------------------------------------------------

def test_stream_emits_done_event_when_job_completes():
    """AC03 — a final 'step: done' event is emitted when the job reaches 'completed'."""
    queue = _mock_queue([
        _make_row("completed"),
        _make_row("completed"),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    events = [
        json.loads(ln[len("data: "):])
        for ln in response.text.split("\n")
        if ln.startswith("data: ")
    ]
    steps = [e["step"] for e in events]
    assert "done" in steps


def test_stream_closes_after_done_event():
    """AC03 — stream closes (no further events) after the 'done' event."""
    queue = _mock_queue([
        _make_row("completed"),
        _make_row("completed"),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    events = [
        json.loads(ln[len("data: "):])
        for ln in response.text.split("\n")
        if ln.startswith("data: ")
    ]
    # "done" must be the last event
    assert events[-1]["step"] == "done"


def test_stream_done_event_has_pct_1():
    """AC03 — the 'done' final event has pct=1.0."""
    queue = _mock_queue([
        _make_row("completed"),
        _make_row("completed"),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    events = [
        json.loads(ln[len("data: "):])
        for ln in response.text.split("\n")
        if ln.startswith("data: ")
    ]
    done_events = [e for e in events if e["step"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["pct"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AC03: final event with step "error" for failed jobs
# ---------------------------------------------------------------------------

def test_stream_emits_error_event_when_job_fails():
    """AC03 — a final 'step: error' event is emitted when the job reaches 'failed'."""
    queue = _mock_queue([
        _make_row("failed", result=json.dumps({"error": "OOM"})),
        _make_row("failed", result=json.dumps({"error": "OOM"})),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    events = [
        json.loads(ln[len("data: "):])
        for ln in response.text.split("\n")
        if ln.startswith("data: ")
    ]
    steps = [e["step"] for e in events]
    assert "error" in steps


def test_stream_error_event_includes_error_message():
    """AC03 — the 'error' final event includes the error message from the job result."""
    error_text = "CUDA out of memory"
    queue = _mock_queue([
        _make_row("failed", result=json.dumps({"error": error_text})),
        _make_row("failed", result=json.dumps({"error": error_text})),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    events = [
        json.loads(ln[len("data: "):])
        for ln in response.text.split("\n")
        if ln.startswith("data: ")
    ]
    error_events = [e for e in events if e["step"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["error"] == error_text


def test_stream_closes_after_error_event():
    """AC03 — stream closes (no further events) after the 'error' event."""
    queue = _mock_queue([
        _make_row("failed", result=json.dumps({"error": "timeout"})),
        _make_row("failed", result=json.dumps({"error": "timeout"})),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    events = [
        json.loads(ln[len("data: "):])
        for ln in response.text.split("\n")
        if ln.startswith("data: ")
    ]
    assert events[-1]["step"] == "error"


def test_stream_error_event_without_result_field():
    """AC03 — 'error' event is still emitted even if job result column is NULL."""
    queue = _mock_queue([
        _make_row("failed", result=None),
        _make_row("failed", result=None),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    events = [
        json.loads(ln[len("data: "):])
        for ln in response.text.split("\n")
        if ln.startswith("data: ")
    ]
    error_events = [e for e in events if e["step"] == "error"]
    assert len(error_events) == 1


# ---------------------------------------------------------------------------
# AC02 + AC03 combined: progress then done
# ---------------------------------------------------------------------------

def test_stream_progress_then_done_sequence():
    """AC02+AC03 — progress events precede the terminal 'done' event."""
    progress1 = PythonProgress(step="sampling", pct=0.4)
    progress2 = PythonProgress(step="sampling", pct=0.9)
    queue = _mock_queue([
        _make_row("running", progress=progress1.model_dump_json()),  # initial
        _make_row("running", progress=progress1.model_dump_json()),  # poll 1
        _make_row("running", progress=progress2.model_dump_json()),  # poll 2
        _make_row("completed"),                                        # poll 3
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    events = [
        json.loads(ln[len("data: "):])
        for ln in response.text.split("\n")
        if ln.startswith("data: ")
    ]
    steps = [e["step"] for e in events]
    assert steps[-1] == "done"
    assert "sampling" in steps


def test_stream_deduplicates_progress_events():
    """AC02 — identical consecutive progress events are not duplicated."""
    progress = PythonProgress(step="sampling", pct=0.5)
    progress_json = progress.model_dump_json()
    queue = _mock_queue([
        _make_row("running", progress=progress_json),
        _make_row("running", progress=progress_json),  # same progress — should not re-emit
        _make_row("running", progress=progress_json),  # same again
        _make_row("completed"),
    ])
    mock_get_queue = AsyncMock(return_value=queue)
    with patch("server.gateway.get_queue", mock_get_queue), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        response = client.get(f"/jobs/{_JOB_ID}/stream")

    events = [
        json.loads(ln[len("data: "):])
        for ln in response.text.split("\n")
        if ln.startswith("data: ")
    ]
    sampling_events = [e for e in events if e["step"] == "sampling"]
    # Only one unique progress event should be emitted despite three identical polls
    assert len(sampling_events) == 1
