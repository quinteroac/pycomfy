"""Tests for server/worker.py — US-004: Python worker process."""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.jobs import JobData, PythonProgress
from server.queue import _reset_singleton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _sample_job_data(**overrides) -> JobData:
    defaults = dict(
        action="generate",
        media="image",
        model="sdxl",
        script="image/sdxl/t2i",
        args={"prompt": "a red apple", "width": "512"},
        script_base="/tmp/pipelines",
        uv_path="/usr/bin/uv",
    )
    defaults.update(overrides)
    return JobData(**defaults)


def _make_mock_queue(
    *,
    job_status: str = "queued",
    job_data: JobData | None = None,
) -> MagicMock:
    """Return a mock JobQueue suitable for patching get_queue."""
    data = job_data or _sample_job_data()
    job_row = {
        "id": "test-job-id",
        "status": job_status,
        "data": data.model_dump_json(),
        "result": None,
        "progress": None,
    }
    q = MagicMock()
    q.get = AsyncMock(return_value=job_row)
    q.update_status = AsyncMock()
    q.update_progress = AsyncMock()
    return q


def _make_proc(stdout_lines: list[str], stderr: str = "", returncode: int = 0):
    """Return a mock Popen process."""
    proc = MagicMock()
    proc.stdout = iter(line + "\n" for line in stdout_lines)
    proc.stderr = io.StringIO(stderr)
    proc.wait = MagicMock(return_value=returncode)
    return proc


def _patch_queue(mock_queue):
    return patch("server.worker.get_queue", new=AsyncMock(return_value=mock_queue))


def _patch_popen(proc):
    return patch("server.worker.subprocess.Popen", return_value=proc)


# ---------------------------------------------------------------------------
# AC01 — worker runs without error when job exists with status "queued"
# ---------------------------------------------------------------------------

class TestAC01JobQueued:
    def test_worker_runs_cleanly_for_queued_job(self):
        q = _make_mock_queue(job_status="queued")
        proc = _make_proc([])
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        # no exception raised

    def test_worker_accepts_pending_status(self):
        """submit_job creates jobs with 'pending'; worker should accept it too."""
        q = _make_mock_queue(job_status="pending")
        proc = _make_proc([])
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))

    def test_worker_exits_nonzero_when_job_not_found(self):
        q = MagicMock()
        q.get = AsyncMock(return_value=None)
        with _patch_queue(q), pytest.raises(SystemExit) as exc:
            from server.worker import _run_worker
            _run(_run_worker("missing-id"))
        assert exc.value.code == 1

    def test_worker_exits_nonzero_when_status_is_completed(self):
        q = _make_mock_queue(job_status="completed")
        with _patch_queue(q), pytest.raises(SystemExit) as exc:
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        assert exc.value.code == 1

    def test_worker_exits_nonzero_when_status_is_failed(self):
        q = _make_mock_queue(job_status="failed")
        with _patch_queue(q), pytest.raises(SystemExit) as exc:
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        assert exc.value.code == 1

    def test_worker_exits_nonzero_when_status_is_running(self):
        q = _make_mock_queue(job_status="running")
        with _patch_queue(q), pytest.raises(SystemExit) as exc:
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# AC02 — worker reads JobData, spawns subprocess with uv run python <script>
# ---------------------------------------------------------------------------

class TestAC02SubprocessSpawn:
    def test_popen_called_once(self):
        q = _make_mock_queue()
        proc = _make_proc([])
        with _patch_queue(q), _patch_popen(proc) as mock_popen:
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        assert mock_popen.call_count == 1

    def test_popen_command_starts_with_uv_path(self):
        job_data = _sample_job_data(uv_path="/custom/uv")
        q = _make_mock_queue(job_data=job_data)
        proc = _make_proc([])
        with _patch_queue(q), _patch_popen(proc) as mock_popen:
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "/custom/uv"
        assert cmd[1] == "run"
        assert cmd[2] == "python"

    def test_popen_command_includes_script_path(self):
        job_data = _sample_job_data(script_base="/tmp/base", script="image/sdxl/t2i")
        q = _make_mock_queue(job_data=job_data)
        proc = _make_proc([])
        with _patch_queue(q), _patch_popen(proc) as mock_popen:
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        cmd = mock_popen.call_args[0][0]
        assert str(Path("/tmp/base/image/sdxl/t2i")) in cmd

    def test_popen_command_includes_args(self):
        job_data = _sample_job_data(args={"prompt": "hello", "width": "512"})
        q = _make_mock_queue(job_data=job_data)
        proc = _make_proc([])
        with _patch_queue(q), _patch_popen(proc) as mock_popen:
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        cmd = mock_popen.call_args[0][0]
        assert "--prompt" in cmd
        assert "hello" in cmd
        assert "--width" in cmd
        assert "512" in cmd

    def test_popen_stdout_pipe(self):
        q = _make_mock_queue()
        proc = _make_proc([])
        with _patch_queue(q), _patch_popen(proc) as mock_popen:
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        kwargs = mock_popen.call_args[1]
        assert kwargs.get("stdout") is not None

    def test_popen_stderr_pipe(self):
        q = _make_mock_queue()
        proc = _make_proc([])
        with _patch_queue(q), _patch_popen(proc) as mock_popen:
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        kwargs = mock_popen.call_args[1]
        assert kwargs.get("stderr") is not None

    def test_status_set_to_running_before_spawn(self):
        q = _make_mock_queue()
        proc = _make_proc([])
        update_calls = []
        original = q.update_status.side_effect

        async def _capture(job_id, status, *args, **kwargs):
            update_calls.append(status)

        q.update_status.side_effect = _capture
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        assert "running" in update_calls
        assert update_calls[0] == "running"


# ---------------------------------------------------------------------------
# AC03 — valid PythonProgress JSON lines parsed and stored as progress updates
# ---------------------------------------------------------------------------

class TestAC03ProgressParsing:
    def _make_progress_line(self, **kwargs) -> str:
        defaults = {"step": "encode", "pct": 0.5}
        defaults.update(kwargs)
        return json.dumps(defaults)

    def test_valid_progress_line_calls_update_progress(self):
        line = self._make_progress_line(step="encode", pct=0.5)
        q = _make_mock_queue()
        proc = _make_proc([line])
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        q.update_progress.assert_awaited_once()

    def test_multiple_progress_lines_all_stored(self):
        lines = [
            self._make_progress_line(step="encode", pct=0.25),
            self._make_progress_line(step="sample", pct=0.75),
        ]
        q = _make_mock_queue()
        proc = _make_proc(lines)
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        assert q.update_progress.await_count == 2

    def test_non_json_line_does_not_call_update_progress(self):
        q = _make_mock_queue()
        proc = _make_proc(["plain text output"])
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        q.update_progress.assert_not_awaited()

    def test_invalid_json_line_does_not_call_update_progress(self):
        q = _make_mock_queue()
        proc = _make_proc(["{not valid json"])
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        q.update_progress.assert_not_awaited()

    def test_json_missing_required_fields_does_not_call_update_progress(self):
        """JSON that is not a valid PythonProgress (missing step/pct) is skipped."""
        q = _make_mock_queue()
        proc = _make_proc(['{"key": "value"}'])
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        q.update_progress.assert_not_awaited()

    def test_update_progress_receives_python_progress_object(self):
        line = self._make_progress_line(step="decode", pct=0.9)
        q = _make_mock_queue()
        proc = _make_proc([line])
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        call_args = q.update_progress.await_args
        _, progress_arg = call_args[0]
        assert isinstance(progress_arg, PythonProgress)
        assert progress_arg.step == "decode"
        assert progress_arg.pct == 0.9


# ---------------------------------------------------------------------------
# AC04 — on exit code 0: status "completed" with output_path from last progress
# ---------------------------------------------------------------------------

class TestAC04CompletedStatus:
    def _make_progress_with_output(self, output: str) -> str:
        return json.dumps({"step": "done", "pct": 1.0, "output": output})

    def test_status_completed_on_exit_0(self):
        q = _make_mock_queue()
        proc = _make_proc([], returncode=0)
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        # find the "completed" call
        completed_call = None
        for call in q.update_status.await_args_list:
            if call[0][1] == "completed":
                completed_call = call
        assert completed_call is not None

    def test_output_path_from_last_progress_with_output(self):
        lines = [
            self._make_progress_with_output("/out/frame001.png"),
            self._make_progress_with_output("/out/frame002.png"),
        ]
        q = _make_mock_queue()
        proc = _make_proc(lines, returncode=0)
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        completed_call = None
        for call in q.update_status.await_args_list:
            if call[0][1] == "completed":
                completed_call = call
        assert completed_call is not None
        result = completed_call[0][2]
        assert result["output_path"] == "/out/frame002.png"

    def test_completed_without_output_has_empty_result(self):
        """If no PythonProgress line has output set, result dict has no output_path."""
        line = json.dumps({"step": "encode", "pct": 0.5})
        q = _make_mock_queue()
        proc = _make_proc([line], returncode=0)
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        for call in q.update_status.await_args_list:
            if call[0][1] == "completed":
                result = call[0][2]
                assert "output_path" not in result or result.get("output_path") is None
                return
        pytest.fail("No 'completed' status update found")

    def test_no_failed_update_on_exit_0(self):
        q = _make_mock_queue()
        proc = _make_proc([], returncode=0)
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        for call in q.update_status.await_args_list:
            assert call[0][1] != "failed", "unexpected 'failed' status on clean exit"


# ---------------------------------------------------------------------------
# AC05 — on non-zero exit: status "failed" with stderr in result.error
# ---------------------------------------------------------------------------

class TestAC05FailedStatus:
    def test_status_failed_on_nonzero_exit(self):
        q = _make_mock_queue()
        proc = _make_proc([], stderr="CUDA error", returncode=1)
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        failed_call = None
        for call in q.update_status.await_args_list:
            if call[0][1] == "failed":
                failed_call = call
        assert failed_call is not None

    def test_failed_result_contains_error_key(self):
        q = _make_mock_queue()
        proc = _make_proc([], stderr="RuntimeError: out of memory", returncode=2)
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        for call in q.update_status.await_args_list:
            if call[0][1] == "failed":
                result = call[0][2]
                assert "error" in result
                assert "RuntimeError" in result["error"]
                return
        pytest.fail("No 'failed' status update found")

    def test_stderr_content_stored_in_error(self):
        stderr_msg = "Traceback (most recent call last):\n  File ...\nValueError: bad input"
        q = _make_mock_queue()
        proc = _make_proc([], stderr=stderr_msg, returncode=1)
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        for call in q.update_status.await_args_list:
            if call[0][1] == "failed":
                result = call[0][2]
                assert result["error"] == stderr_msg
                return
        pytest.fail("No 'failed' status update found")

    def test_no_completed_update_on_nonzero_exit(self):
        q = _make_mock_queue()
        proc = _make_proc([], returncode=99)
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        for call in q.update_status.await_args_list:
            assert call[0][1] != "completed", "unexpected 'completed' on failed exit"

    def test_partial_stdout_before_failure_does_not_set_output_path(self):
        """Progress lines before failure should not end up as output_path."""
        lines = [json.dumps({"step": "encode", "pct": 0.5})]
        q = _make_mock_queue()
        proc = _make_proc(lines, stderr="crash", returncode=1)
        with _patch_queue(q), _patch_popen(proc):
            from server.worker import _run_worker
            _run(_run_worker("test-job-id"))
        for call in q.update_status.await_args_list:
            if call[0][1] == "failed":
                result = call[0][2]
                assert "output_path" not in result
                return
        pytest.fail("No 'failed' status update found")
