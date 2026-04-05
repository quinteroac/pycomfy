"""Tests for US-003 — ``parallax jobs`` subcommand group.

AC01: ``parallax jobs list`` prints a table of the 20 most recent jobs with
      columns: ID, STATUS, MODEL, CREATED.
AC02: ``parallax jobs status <job_id>`` prints the full job record as formatted JSON.
AC03: ``parallax jobs watch <job_id>`` renders a live progress bar until terminal
      state, then prints the output path or error.
AC04: ``parallax jobs cancel <job_id>`` cancels a queued job and prints
      ``Cancelled <job_id>`` or an appropriate error.
AC05: ``parallax jobs open <job_id>`` opens the output file with the OS default app.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

_PATCH_LIST = "cli.commands.jobs._call_list_jobs"
_PATCH_GET = "cli.commands.jobs._call_get_job"
_PATCH_CANCEL = "cli.commands.jobs._call_cancel_job"

_JOB_ID = "11111111-2222-3333-4444-555555555555"
_CREATED_AT = "2026-04-05T12:00:00+00:00"
_UPDATED_AT = "2026-04-05T12:01:00+00:00"


def _make_job(
    job_id: str = _JOB_ID,
    status: str = "pending",
    model: str = "sdxl",
    result: dict | None = None,
    progress: dict | None = None,
) -> dict[str, Any]:
    data_payload = json.dumps({"model": model, "prompt": "a cat"})
    return {
        "id": job_id,
        "status": status,
        "data": data_payload,
        "result": json.dumps(result) if result else None,
        "progress": json.dumps(progress) if progress else None,
        "created_at": _CREATED_AT,
        "updated_at": _UPDATED_AT,
    }


@pytest.fixture
def runner():
    return CliRunner()


def _app():
    from cli.main import app
    return app


# ── AC01: parallax jobs list ──────────────────────────────────────────────────


class TestJobsList:
    def test_list_prints_table_columns(self, runner):
        jobs = [
            _make_job("id-001", status="pending", model="sdxl"),
            _make_job("id-002", status="completed", model="anima"),
        ]
        with patch(_PATCH_LIST, return_value=jobs):
            result = runner.invoke(_app(), ["jobs", "list"])

        assert result.exit_code == 0
        output = result.output
        assert "ID" in output
        assert "STATUS" in output
        assert "MODEL" in output
        assert "CREATED" in output

    def test_list_shows_job_ids(self, runner):
        jobs = [_make_job("id-abc", status="pending", model="sdxl")]
        with patch(_PATCH_LIST, return_value=jobs):
            result = runner.invoke(_app(), ["jobs", "list"])

        assert result.exit_code == 0
        assert "id-abc" in result.output

    def test_list_maps_pending_to_queued(self, runner):
        jobs = [_make_job(status="pending")]
        with patch(_PATCH_LIST, return_value=jobs):
            result = runner.invoke(_app(), ["jobs", "list"])

        assert "queued" in result.output
        assert "pending" not in result.output

    def test_list_shows_model_name(self, runner):
        jobs = [_make_job(model="flux_klein")]
        with patch(_PATCH_LIST, return_value=jobs):
            result = runner.invoke(_app(), ["jobs", "list"])

        assert "flux_klein" in result.output

    def test_list_shows_created_at(self, runner):
        jobs = [_make_job()]
        with patch(_PATCH_LIST, return_value=jobs):
            result = runner.invoke(_app(), ["jobs", "list"])

        # Rich may truncate the timestamp — check the date portion only
        assert "2026-04-05" in result.output

    def test_list_empty_queue(self, runner):
        with patch(_PATCH_LIST, return_value=[]):
            result = runner.invoke(_app(), ["jobs", "list"])

        assert result.exit_code == 0

    def test_list_calls_limit_20(self, runner):
        mock_fn = MagicMock(return_value=[])
        with patch(_PATCH_LIST, mock_fn):
            runner.invoke(_app(), ["jobs", "list"])

        mock_fn.assert_called_once_with(limit=20)


# ── AC02: parallax jobs status <job_id> ──────────────────────────────────────


class TestJobsStatus:
    def test_status_prints_json(self, runner):
        job = _make_job(status="completed", result={"output_path": "/tmp/out.png"})
        with patch(_PATCH_GET, return_value=job):
            result = runner.invoke(_app(), ["jobs", "status", _JOB_ID])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["id"] == _JOB_ID

    def test_status_maps_pending_to_queued(self, runner):
        job = _make_job(status="pending")
        with patch(_PATCH_GET, return_value=job):
            result = runner.invoke(_app(), ["jobs", "status", _JOB_ID])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "queued"

    def test_status_deserialises_result_field(self, runner):
        job = _make_job(status="completed", result={"output_path": "/tmp/out.png"})
        with patch(_PATCH_GET, return_value=job):
            result = runner.invoke(_app(), ["jobs", "status", _JOB_ID])

        parsed = json.loads(result.output)
        assert isinstance(parsed["result"], dict)
        assert parsed["result"]["output_path"] == "/tmp/out.png"

    def test_status_not_found_exits_1(self, runner):
        with patch(_PATCH_GET, return_value=None):
            result = runner.invoke(_app(), ["jobs", "status", "no-such-id"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_status_includes_all_keys(self, runner):
        job = _make_job(status="completed")
        with patch(_PATCH_GET, return_value=job):
            result = runner.invoke(_app(), ["jobs", "status", _JOB_ID])

        parsed = json.loads(result.output)
        for key in ("id", "status", "created_at", "updated_at"):
            assert key in parsed


# ── AC03: parallax jobs watch <job_id> ───────────────────────────────────────


class TestJobsWatch:
    def test_watch_completed_prints_output_path(self, runner):
        job = _make_job(
            status="completed",
            result={"output_path": "/tmp/output.png"},
            progress={"pct": 1.0},
        )
        with patch(_PATCH_GET, return_value=job):
            result = runner.invoke(_app(), ["jobs", "watch", _JOB_ID])

        assert result.exit_code == 0
        assert "/tmp/output.png" in result.output

    def test_watch_failed_exits_1_with_error(self, runner):
        job = _make_job(
            status="failed",
            result={"error": "OOM"},
        )
        with patch(_PATCH_GET, return_value=job):
            result = runner.invoke(_app(), ["jobs", "watch", _JOB_ID])

        assert result.exit_code == 1
        assert "OOM" in result.output

    def test_watch_cancelled_prints_message(self, runner):
        job = _make_job(status="cancelled")
        with patch(_PATCH_GET, return_value=job):
            result = runner.invoke(_app(), ["jobs", "watch", _JOB_ID])

        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()

    def test_watch_not_found_exits_1(self, runner):
        with patch(_PATCH_GET, return_value=None):
            result = runner.invoke(_app(), ["jobs", "watch", "no-such-id"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_watch_polls_until_terminal(self, runner):
        """Verify that watch keeps polling: first returns in-progress, then completed."""
        running_job = _make_job(status="running", progress={"pct": 0.5})
        done_job = _make_job(
            status="completed",
            result={"output_path": "/tmp/done.mp4"},
            progress={"pct": 1.0},
        )
        side_effects = [running_job, running_job, done_job]
        mock_get = MagicMock(side_effect=side_effects)

        with patch(_PATCH_GET, mock_get), patch("time.sleep"):
            result = runner.invoke(_app(), ["jobs", "watch", _JOB_ID])

        assert result.exit_code == 0
        assert "/tmp/done.mp4" in result.output
        assert mock_get.call_count == 3


# ── AC04: parallax jobs cancel <job_id> ──────────────────────────────────────


class TestJobsCancel:
    def test_cancel_success_prints_cancelled(self, runner):
        job = _make_job(status="pending")
        with patch(_PATCH_GET, return_value=job), patch(_PATCH_CANCEL, return_value=True):
            result = runner.invoke(_app(), ["jobs", "cancel", _JOB_ID])

        assert result.exit_code == 0
        assert f"Cancelled {_JOB_ID}" in result.output

    def test_cancel_not_found_exits_1(self, runner):
        with patch(_PATCH_GET, return_value=None):
            result = runner.invoke(_app(), ["jobs", "cancel", "no-such-id"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_cancel_already_terminal_exits_1(self, runner):
        job = _make_job(status="completed")
        with patch(_PATCH_GET, return_value=job), patch(_PATCH_CANCEL, return_value=False):
            result = runner.invoke(_app(), ["jobs", "cancel", _JOB_ID])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_cancel_calls_cancel_with_job_id(self, runner):
        job = _make_job(status="pending")
        mock_cancel = MagicMock(return_value=True)
        with patch(_PATCH_GET, return_value=job), patch(_PATCH_CANCEL, mock_cancel):
            runner.invoke(_app(), ["jobs", "cancel", _JOB_ID])

        mock_cancel.assert_called_once_with(_JOB_ID)


# ── AC05: parallax jobs open <job_id> ────────────────────────────────────────


class TestJobsOpen:
    def test_open_calls_os_opener(self, runner):
        job = _make_job(
            status="completed",
            result={"output_path": "/tmp/result.png"},
        )
        mock_run = MagicMock()
        with patch(_PATCH_GET, return_value=job), patch("subprocess.run", mock_run):
            result = runner.invoke(_app(), ["jobs", "open", _JOB_ID])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "/tmp/result.png" in call_args

    def test_open_not_found_exits_1(self, runner):
        with patch(_PATCH_GET, return_value=None):
            result = runner.invoke(_app(), ["jobs", "open", "no-such-id"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_open_no_result_exits_1(self, runner):
        job = _make_job(status="running")  # no result yet
        with patch(_PATCH_GET, return_value=job):
            result = runner.invoke(_app(), ["jobs", "open", _JOB_ID])

        assert result.exit_code == 1
        assert "no output" in result.output

    def test_open_no_output_path_exits_1(self, runner):
        job = _make_job(status="completed", result={"error": "boom"})
        with patch(_PATCH_GET, return_value=job):
            result = runner.invoke(_app(), ["jobs", "open", _JOB_ID])

        assert result.exit_code == 1
        assert "no output path" in result.output

    def test_open_uses_xdg_open_on_linux(self, runner):
        job = _make_job(status="completed", result={"output_path": "/tmp/out.png"})
        mock_run = MagicMock()
        with (
            patch(_PATCH_GET, return_value=job),
            patch("subprocess.run", mock_run),
            patch("sys.platform", "linux"),
        ):
            result = runner.invoke(_app(), ["jobs", "open", _JOB_ID])

        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "xdg-open"

    def test_open_uses_open_on_macos(self, runner):
        job = _make_job(status="completed", result={"output_path": "/tmp/out.png"})
        mock_run = MagicMock()
        with (
            patch(_PATCH_GET, return_value=job),
            patch("subprocess.run", mock_run),
            patch("sys.platform", "darwin"),
        ):
            result = runner.invoke(_app(), ["jobs", "open", _JOB_ID])

        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "open"
