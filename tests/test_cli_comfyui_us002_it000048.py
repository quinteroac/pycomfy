"""Tests for US-002 it_000048 — parallax comfyui stop CLI command.

Covers:
  AC01 — reads PID from ~/.config/parallax/comfyui.pid and terminates the process
  AC02 — PID file removed after successful stop
  AC03 — if no instance running, prints informative message and exits with code 0
  AC04 — cross-platform: SIGTERM on Linux/macOS, TerminateProcess on Windows
"""
from __future__ import annotations

import signal
from pathlib import Path
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from cli.commands.comfyui import _terminate_process, app

runner = CliRunner()

# Wrap sub-app in a parent so ["comfyui", "stop"] routes correctly.
_cli = typer.Typer()
_cli.add_typer(app, name="comfyui")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _run_stop(
    tmp_path: Path,
    *,
    pid: int = 12345,
    is_running: bool = True,
    terminate_side_effect: Exception | None = None,
) -> tuple:
    """Invoke ``parallax comfyui stop`` with all I/O mocked."""
    pid_file = tmp_path / "comfyui.pid"
    if is_running:
        pid_file.write_text(str(pid), encoding="utf-8")

    kwargs: dict = {}
    if terminate_side_effect is not None:
        kwargs["side_effect"] = terminate_side_effect

    with (
        patch("cli.commands.comfyui._pid_file", return_value=pid_file),
        patch("cli.commands.comfyui._is_running", return_value=is_running),
        patch("cli.commands.comfyui._terminate_process", **kwargs) as mock_term,
    ):
        result = runner.invoke(_cli, ["comfyui", "stop"])

    return result, pid_file, mock_term


# ---------------------------------------------------------------------------
# AC01 — reads PID and terminates the process
# ---------------------------------------------------------------------------


class TestAC01TerminateProcess:
    def test_terminate_called_with_correct_pid(self, tmp_path: Path) -> None:
        result, _, mock_term = _run_stop(tmp_path, pid=55555)
        assert result.exit_code == 0
        mock_term.assert_called_once_with(55555)

    def test_exit_code_zero_on_success(self, tmp_path: Path) -> None:
        result, _, _ = _run_stop(tmp_path)
        assert result.exit_code == 0

    def test_error_message_on_permission_error(self, tmp_path: Path) -> None:
        result, _, _ = _run_stop(tmp_path, terminate_side_effect=PermissionError("denied"))
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_error_message_on_process_lookup_error(self, tmp_path: Path) -> None:
        result, _, _ = _run_stop(tmp_path, terminate_side_effect=ProcessLookupError("gone"))
        assert result.exit_code != 0
        assert "Error" in result.output


# ---------------------------------------------------------------------------
# AC02 — PID file removed after successful stop
# ---------------------------------------------------------------------------


class TestAC02PidFileRemoved:
    def test_pid_file_deleted_after_stop(self, tmp_path: Path) -> None:
        _, pid_file, _ = _run_stop(tmp_path)
        assert not pid_file.exists()

    def test_pid_file_retained_on_terminate_error(self, tmp_path: Path) -> None:
        _, pid_file, _ = _run_stop(
            tmp_path, terminate_side_effect=PermissionError("denied")
        )
        # PID file should still exist when termination failed
        assert pid_file.exists()


# ---------------------------------------------------------------------------
# AC03 — no instance running → informative message + exit 0
# ---------------------------------------------------------------------------


class TestAC03NoInstance:
    def test_informative_message_when_not_running(self, tmp_path: Path) -> None:
        result, _, _ = _run_stop(tmp_path, is_running=False)
        assert "No ComfyUI instance" in result.output

    def test_exit_code_zero_when_not_running(self, tmp_path: Path) -> None:
        result, _, _ = _run_stop(tmp_path, is_running=False)
        assert result.exit_code == 0

    def test_terminate_not_called_when_not_running(self, tmp_path: Path) -> None:
        _, _, mock_term = _run_stop(tmp_path, is_running=False)
        mock_term.assert_not_called()


# ---------------------------------------------------------------------------
# AC04 — cross-platform: SIGTERM on POSIX, TerminateProcess on Windows
# ---------------------------------------------------------------------------


class TestAC04CrossPlatform:
    def test_posix_uses_sigterm(self) -> None:
        """On POSIX, _terminate_process must call os.kill with SIGTERM."""
        with (
            patch("sys.platform", "linux"),
            patch("os.kill") as mock_kill,
        ):
            _terminate_process(42)
        mock_kill.assert_called_once_with(42, signal.SIGTERM)

    def test_windows_uses_os_kill_sigterm(self) -> None:
        """On Windows, _terminate_process must call os.kill (→ TerminateProcess) with SIGTERM."""
        with (
            patch("sys.platform", "win32"),
            patch("os.kill") as mock_kill,
        ):
            _terminate_process(42)
        mock_kill.assert_called_once_with(42, signal.SIGTERM)

    def test_macos_uses_sigterm(self) -> None:
        """On macOS, _terminate_process must call os.kill with SIGTERM."""
        with (
            patch("sys.platform", "darwin"),
            patch("os.kill") as mock_kill,
        ):
            _terminate_process(99)
        mock_kill.assert_called_once_with(99, signal.SIGTERM)
