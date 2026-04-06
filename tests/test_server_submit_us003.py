"""Tests for server/submit.py — US-003: submit_job() helper."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.jobs import JobData
from server.submit import submit_job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_JOB_DATA = JobData(
    action="generate",
    media="image",
    model="sdxl",
    cmd=["/usr/bin/uv", "run", "python", "/tmp/pipelines/image/sdxl/t2i.py", "--prompt", "a red apple"],
)

_FAKE_JOB_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _make_mock_queue(job_id: str = _FAKE_JOB_ID) -> MagicMock:
    """Return a mock JobQueue whose enqueue() is an async coroutine."""
    mock_queue = MagicMock()
    mock_queue.enqueue = AsyncMock(return_value=job_id)
    return mock_queue


def _patched(mock_queue: MagicMock):
    """Context-manager pair: AsyncMock get_queue + mocked Popen."""
    return (
        patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)),
        patch("server.submit.subprocess.Popen"),
    )


# ---------------------------------------------------------------------------
# US-003-AC01: submit_job is exported from server/submit.py
# ---------------------------------------------------------------------------

class TestAC01:
    def test_submit_job_importable(self):
        from server.submit import submit_job as fn  # noqa: F401

        assert callable(fn)

    def test_signature_accepts_job_data(self):
        """submit_job(data) should accept a JobData argument."""
        import inspect

        from server.submit import submit_job as fn

        sig = inspect.signature(fn)
        params = list(sig.parameters)
        assert "data" in params

    def test_return_annotation_is_str(self):
        import inspect
        import typing

        from server.submit import submit_job as fn

        hints = typing.get_type_hints(fn)
        assert hints.get("return") is str


# ---------------------------------------------------------------------------
# US-003-AC02: enqueues via get_queue() and spawns worker with start_new_session
# ---------------------------------------------------------------------------

class TestAC02:
    def test_get_queue_is_called(self):
        mock_queue = _make_mock_queue()

        with patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)):
            with patch("server.submit.subprocess.Popen"):
                submit_job(_SAMPLE_JOB_DATA)

        mock_queue.enqueue.assert_awaited_once_with(_SAMPLE_JOB_DATA)

    def test_popen_called_with_start_new_session(self):
        mock_queue = _make_mock_queue()

        with patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)):
            with patch("server.submit.subprocess.Popen") as mock_popen:
                submit_job(_SAMPLE_JOB_DATA)

        assert mock_popen.call_count == 1
        _, kwargs = mock_popen.call_args
        assert kwargs.get("start_new_session") is True

    def test_popen_spawns_worker_script(self):
        mock_queue = _make_mock_queue()

        with patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)):
            with patch("server.submit.subprocess.Popen") as mock_popen:
                submit_job(_SAMPLE_JOB_DATA)

        args_list, _ = mock_popen.call_args
        cmd = args_list[0]
        assert cmd[1].endswith("worker.py"), f"Expected worker.py, got: {cmd[1]}"


# ---------------------------------------------------------------------------
# US-003-AC03: returns job ID, wall-clock time < 100ms with mock queue
# ---------------------------------------------------------------------------

class TestAC03:
    def test_returns_job_id_string(self):
        mock_queue = _make_mock_queue(job_id=_FAKE_JOB_ID)

        with patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)):
            with patch("server.submit.subprocess.Popen"):
                result = submit_job(_SAMPLE_JOB_DATA)

        assert result == _FAKE_JOB_ID

    def test_returns_str_type(self):
        mock_queue = _make_mock_queue()

        with patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)):
            with patch("server.submit.subprocess.Popen"):
                result = submit_job(_SAMPLE_JOB_DATA)

        assert isinstance(result, str)

    def test_wall_clock_under_100ms(self):
        mock_queue = _make_mock_queue()

        with patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)):
            with patch("server.submit.subprocess.Popen"):
                start = time.perf_counter()
                submit_job(_SAMPLE_JOB_DATA)
                elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"submit_job took {elapsed_ms:.1f}ms (limit: 100ms)"

    def test_wall_clock_multiple_calls(self):
        """Each independent call should complete in < 100ms."""
        for _ in range(3):
            mock_queue = _make_mock_queue()
            with patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)):
                with patch("server.submit.subprocess.Popen"):
                    start = time.perf_counter()
                    submit_job(_SAMPLE_JOB_DATA)
                    elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 100, f"Call took {elapsed_ms:.1f}ms"


# ---------------------------------------------------------------------------
# US-003-AC04: worker subprocess receives job ID as its only CLI argument
# ---------------------------------------------------------------------------

class TestAC04:
    def test_worker_receives_job_id_as_sole_arg(self):
        mock_queue = _make_mock_queue(job_id=_FAKE_JOB_ID)

        with patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)):
            with patch("server.submit.subprocess.Popen") as mock_popen:
                submit_job(_SAMPLE_JOB_DATA)

        args_list, _ = mock_popen.call_args
        cmd = args_list[0]
        # cmd structure: [python_executable, worker_path, job_id]
        assert len(cmd) == 3, f"Expected 3-element command, got: {cmd}"
        assert cmd[2] == _FAKE_JOB_ID

    def test_worker_receives_exact_returned_job_id(self):
        """The job ID passed to worker must match the one returned to caller."""
        mock_queue = _make_mock_queue(job_id=_FAKE_JOB_ID)

        with patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)):
            with patch("server.submit.subprocess.Popen") as mock_popen:
                returned_id = submit_job(_SAMPLE_JOB_DATA)

        args_list, _ = mock_popen.call_args
        cmd = args_list[0]
        assert cmd[2] == returned_id

    def test_no_extra_cli_args_to_worker(self):
        """Worker command must be exactly [python, worker.py, job_id] — no extras."""
        mock_queue = _make_mock_queue()

        with patch("server.submit.get_queue", new=AsyncMock(return_value=mock_queue)):
            with patch("server.submit.subprocess.Popen") as mock_popen:
                submit_job(_SAMPLE_JOB_DATA)

        args_list, _ = mock_popen.call_args
        cmd = args_list[0]
        assert len(cmd) == 3, f"Worker received unexpected extra args: {cmd}"
