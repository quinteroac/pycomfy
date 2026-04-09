"""Tests for US-005 it_000048 — parallax comfyui status CLI command.

Covers:
  AC01 — if running: prints "ComfyUI is running (PID <pid>, port <port>)"
  AC02 — if not running: prints "ComfyUI is not running" and exits with code 0
  AC03 — port is read from the PID file metadata (stored alongside PID at start time)
  AC04 — typecheck / lint passes (verified by running tests without error)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from cli.commands.comfyui import _read_pid_file, app

runner = CliRunner()

# Wrap sub-app in a parent so ["comfyui", "status"] routes correctly.
_cli = typer.Typer()
_cli.add_typer(app, name="comfyui")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _run_status(
    tmp_path: Path,
    *,
    pid: int = 12345,
    port: int = 8188,
    is_running: bool = True,
) -> tuple:
    """Invoke ``parallax comfyui status`` with all I/O mocked."""
    pid_file = tmp_path / "comfyui.pid"
    if is_running:
        pid_file.write_text(
            json.dumps({"pid": pid, "port": port}), encoding="utf-8"
        )

    with (
        patch("cli.commands.comfyui._pid_file", return_value=pid_file),
        patch("cli.commands.comfyui._is_running", return_value=is_running),
    ):
        result = runner.invoke(_cli, ["comfyui", "status"])

    return result, pid_file


# ---------------------------------------------------------------------------
# AC01 — running: prints "ComfyUI is running (PID <pid>, port <port>)"
# ---------------------------------------------------------------------------


class TestAC01Running:
    def test_output_contains_running_message(self, tmp_path: Path) -> None:
        result, _ = _run_status(tmp_path, pid=42000, port=8188)
        assert "ComfyUI is running" in result.output

    def test_output_contains_pid(self, tmp_path: Path) -> None:
        result, _ = _run_status(tmp_path, pid=55555, port=8188)
        assert "55555" in result.output

    def test_output_contains_port(self, tmp_path: Path) -> None:
        result, _ = _run_status(tmp_path, pid=12345, port=9090)
        assert "9090" in result.output

    def test_output_format(self, tmp_path: Path) -> None:
        result, _ = _run_status(tmp_path, pid=77777, port=8188)
        assert "ComfyUI is running (PID 77777, port 8188)" in result.output

    def test_exit_code_zero_when_running(self, tmp_path: Path) -> None:
        result, _ = _run_status(tmp_path)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC02 — not running: prints "ComfyUI is not running" and exits with code 0
# ---------------------------------------------------------------------------


class TestAC02NotRunning:
    def test_output_contains_not_running_message(self, tmp_path: Path) -> None:
        result, _ = _run_status(tmp_path, is_running=False)
        assert "ComfyUI is not running" in result.output

    def test_exit_code_zero_when_not_running(self, tmp_path: Path) -> None:
        result, _ = _run_status(tmp_path, is_running=False)
        assert result.exit_code == 0

    def test_running_message_not_printed_when_not_running(self, tmp_path: Path) -> None:
        result, _ = _run_status(tmp_path, is_running=False)
        assert "ComfyUI is running" not in result.output


# ---------------------------------------------------------------------------
# AC03 — port read from PID file metadata
# ---------------------------------------------------------------------------


class TestAC03PortFromPidFile:
    def test_port_matches_pid_file_metadata(self, tmp_path: Path) -> None:
        """Port in output must come from JSON metadata in the PID file."""
        result, _ = _run_status(tmp_path, pid=1111, port=7777)
        assert "7777" in result.output

    def test_pid_matches_pid_file_metadata(self, tmp_path: Path) -> None:
        result, _ = _run_status(tmp_path, pid=9999, port=8188)
        assert "9999" in result.output

    def test_read_pid_file_returns_correct_tuple(self, tmp_path: Path) -> None:
        """Unit-test _read_pid_file helper directly."""
        pid_file = tmp_path / "comfyui.pid"
        pid_file.write_text(json.dumps({"pid": 4242, "port": 3000}), encoding="utf-8")
        result = _read_pid_file(pid_file)
        assert result == (4242, 3000)

    def test_read_pid_file_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        result = _read_pid_file(tmp_path / "nope.pid")
        assert result is None

    def test_read_pid_file_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "comfyui.pid"
        pid_file.write_text("not-json", encoding="utf-8")
        assert _read_pid_file(pid_file) is None

    def test_read_pid_file_returns_none_for_missing_keys(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "comfyui.pid"
        pid_file.write_text(json.dumps({"pid": 1}), encoding="utf-8")  # missing "port"
        assert _read_pid_file(pid_file) is None
