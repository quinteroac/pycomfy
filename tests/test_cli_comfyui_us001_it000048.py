"""Tests for US-001 it_000048 — parallax comfyui start CLI command.

Covers:
  AC01 — launches ComfyUI using the runtime at ~/.parallax/env
  AC02 — default port 8188
  AC03 — writes PID to ~/.config/parallax/comfyui.pid
  AC04 — prints URL once the server is ready
  AC05 — if already running, prints warning and exits without starting a second instance
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from cli.commands.comfyui import (
    _get_comfyui_main,
    _is_running,
    _wait_until_ready,
    app,
)

runner = CliRunner()

# Wrap sub-app in a parent so `["comfyui", "start"]` routes correctly.
_cli = typer.Typer()
_cli.add_typer(app, name="comfyui")

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _fake_popen(pid: int = 12345) -> MagicMock:
    """Return a mock subprocess.Popen instance with a given pid."""
    mock = MagicMock()
    mock.pid = pid
    return mock


def _run_start(
    tmp_path: Path,
    *,
    port: int = 8188,
    pid: int = 12345,
    comfyui_main: Path | None = None,
    ready: bool = True,
    already_running: bool = False,
) -> tuple:
    """Invoke ``parallax comfyui start`` with all I/O mocked."""
    if comfyui_main is None:
        comfyui_main = tmp_path / "comfyui_main.py"
        comfyui_main.write_text("# fake", encoding="utf-8")

    fake_python = tmp_path / "bin" / "python"
    fake_python.parent.mkdir(parents=True, exist_ok=True)
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_python.chmod(0o755)

    pid_file = tmp_path / "comfyui.pid"
    if already_running:
        import json

        pid_file.write_text(json.dumps({"pid": pid, "port": port}), encoding="utf-8")
    fake_proc = _fake_popen(pid)

    with (
        patch("cli.commands.comfyui._python_path", return_value=fake_python),
        patch("cli.commands.comfyui._pid_file", return_value=pid_file),
        patch("cli.commands.comfyui._is_running", return_value=already_running),
        patch("cli.commands.comfyui._get_comfyui_main", return_value=comfyui_main),
        patch("cli.commands.comfyui._wait_until_ready", return_value=ready),
        patch("subprocess.Popen", return_value=fake_proc),
    ):
        result = runner.invoke(_cli, ["comfyui", "start", "--port", str(port)])

    return result, pid_file


# ---------------------------------------------------------------------------
# AC01 — launches ComfyUI using ~/.parallax/env python
# ---------------------------------------------------------------------------


class TestAC01Launch:
    def test_popen_called_with_env_python(self, tmp_path: Path) -> None:
        fake_python = tmp_path / "bin" / "python"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_python.chmod(0o755)

        comfyui_main = tmp_path / "main.py"
        comfyui_main.write_text("# fake", encoding="utf-8")
        pid_file = tmp_path / "comfyui.pid"
        fake_proc = _fake_popen(99)

        popen_calls: list = []

        def tracking_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            return fake_proc

        with (
            patch("cli.commands.comfyui._python_path", return_value=fake_python),
            patch("cli.commands.comfyui._pid_file", return_value=pid_file),
            patch("cli.commands.comfyui._is_running", return_value=False),
            patch("cli.commands.comfyui._get_comfyui_main", return_value=comfyui_main),
            patch("cli.commands.comfyui._wait_until_ready", return_value=True),
            patch("subprocess.Popen", side_effect=tracking_popen),
        ):
            runner.invoke(_cli, ["comfyui", "start"])

        assert popen_calls, "subprocess.Popen was never called"
        cmd = popen_calls[0]
        assert str(fake_python) == cmd[0], "Popen must use the env python"

    def test_popen_includes_comfyui_main(self, tmp_path: Path) -> None:
        fake_python = tmp_path / "bin" / "python"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.write_text("#!/bin/sh\n")
        fake_python.chmod(0o755)

        comfyui_main = tmp_path / "main.py"
        comfyui_main.write_text("# fake")
        pid_file = tmp_path / "comfyui.pid"
        fake_proc = _fake_popen(99)

        popen_calls: list = []

        def tracking_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            return fake_proc

        with (
            patch("cli.commands.comfyui._python_path", return_value=fake_python),
            patch("cli.commands.comfyui._pid_file", return_value=pid_file),
            patch("cli.commands.comfyui._is_running", return_value=False),
            patch("cli.commands.comfyui._get_comfyui_main", return_value=comfyui_main),
            patch("cli.commands.comfyui._wait_until_ready", return_value=True),
            patch("subprocess.Popen", side_effect=tracking_popen),
        ):
            runner.invoke(_cli, ["comfyui", "start"])

        cmd = popen_calls[0]
        assert str(comfyui_main) in cmd

    def test_exits_with_error_if_env_not_installed(self, tmp_path: Path) -> None:
        missing_python = tmp_path / "bin" / "python"  # does not exist
        with patch("cli.commands.comfyui._python_path", return_value=missing_python):
            result = runner.invoke(_cli, ["comfyui", "start"])
        assert result.exit_code != 0
        assert "parallax install" in result.output


# ---------------------------------------------------------------------------
# AC02 — default port is 8188
# ---------------------------------------------------------------------------


class TestAC02DefaultPort:
    def test_default_port_8188(self, tmp_path: Path) -> None:
        popen_calls: list = []
        fake_python = tmp_path / "bin" / "python"
        fake_python.parent.mkdir(parents=True)
        fake_python.write_text("#!/bin/sh\n")
        fake_python.chmod(0o755)
        comfyui_main = tmp_path / "main.py"
        comfyui_main.write_text("# fake")
        pid_file = tmp_path / "comfyui.pid"

        def tracking_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            return _fake_popen()

        with (
            patch("cli.commands.comfyui._python_path", return_value=fake_python),
            patch("cli.commands.comfyui._pid_file", return_value=pid_file),
            patch("cli.commands.comfyui._is_running", return_value=False),
            patch("cli.commands.comfyui._get_comfyui_main", return_value=comfyui_main),
            patch("cli.commands.comfyui._wait_until_ready", return_value=True),
            patch("subprocess.Popen", side_effect=tracking_popen),
        ):
            runner.invoke(_cli, ["comfyui", "start"])

        cmd = popen_calls[0]
        assert "--port" in cmd
        port_idx = cmd.index("--port")
        assert cmd[port_idx + 1] == "8188"

    def test_custom_port_used(self, tmp_path: Path) -> None:
        result, _ = _run_start(tmp_path, port=9090)
        assert result.exit_code == 0

    def test_url_contains_correct_port(self, tmp_path: Path) -> None:
        result, _ = _run_start(tmp_path, port=9191)
        assert "9191" in result.output


# ---------------------------------------------------------------------------
# AC03 — PID file written
# ---------------------------------------------------------------------------


class TestAC03PidFile:
    def test_pid_file_created(self, tmp_path: Path) -> None:
        _, pid_file = _run_start(tmp_path, pid=42000)
        assert pid_file.exists()

    def test_pid_file_contains_process_pid(self, tmp_path: Path) -> None:
        import json

        _, pid_file = _run_start(tmp_path, pid=77777)
        data = json.loads(pid_file.read_text())
        assert data["pid"] == 77777

    def test_pid_file_parent_dir_created(self, tmp_path: Path) -> None:
        """Parent dirs for the PID file are created if they don't exist."""
        deep_pid = tmp_path / "deep" / "nested" / "comfyui.pid"
        fake_python = tmp_path / "bin" / "python"
        fake_python.parent.mkdir(parents=True)
        fake_python.write_text("#!/bin/sh\n")
        fake_python.chmod(0o755)
        comfyui_main = tmp_path / "main.py"
        comfyui_main.write_text("# fake")

        with (
            patch("cli.commands.comfyui._python_path", return_value=fake_python),
            patch("cli.commands.comfyui._pid_file", return_value=deep_pid),
            patch("cli.commands.comfyui._is_running", return_value=False),
            patch("cli.commands.comfyui._get_comfyui_main", return_value=comfyui_main),
            patch("cli.commands.comfyui._wait_until_ready", return_value=True),
            patch("subprocess.Popen", return_value=_fake_popen(1234)),
        ):
            runner.invoke(_cli, ["comfyui", "start"])

        assert deep_pid.exists()


# ---------------------------------------------------------------------------
# AC04 — URL printed once ready
# ---------------------------------------------------------------------------


class TestAC04PrintUrl:
    def test_url_printed_when_ready(self, tmp_path: Path) -> None:
        result, _ = _run_start(tmp_path, port=8188, ready=True)
        assert "http://localhost:8188" in result.output

    def test_warning_printed_when_not_ready(self, tmp_path: Path) -> None:
        result, _ = _run_start(tmp_path, port=8188, ready=False)
        assert "Warning" in result.output
        assert "http://localhost:8188" in result.output

    def test_exit_code_zero_on_success(self, tmp_path: Path) -> None:
        result, _ = _run_start(tmp_path)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC05 — already running guard
# ---------------------------------------------------------------------------


class TestAC05AlreadyRunning:
    def test_warning_printed_if_running(self, tmp_path: Path) -> None:
        result, _ = _run_start(tmp_path, already_running=True)
        assert "Warning" in result.output
        assert "already running" in result.output

    def test_popen_not_called_if_running(self, tmp_path: Path) -> None:
        popen_calls: list = []
        fake_python = tmp_path / "bin" / "python"
        fake_python.parent.mkdir(parents=True)
        fake_python.write_text("#!/bin/sh\n")
        fake_python.chmod(0o755)
        pid_file = tmp_path / "comfyui.pid"
        pid_file.write_text("99999")

        with (
            patch("cli.commands.comfyui._python_path", return_value=fake_python),
            patch("cli.commands.comfyui._pid_file", return_value=pid_file),
            patch("cli.commands.comfyui._is_running", return_value=True),
            patch("subprocess.Popen", side_effect=lambda *a, **k: popen_calls.append(a)),
        ):
            runner.invoke(_cli, ["comfyui", "start"])

        assert not popen_calls, "Popen must not be called when ComfyUI is already running"

    def test_exit_zero_if_already_running(self, tmp_path: Path) -> None:
        result, _ = _run_start(tmp_path, already_running=True)
        # warning + graceful exit (exit code 0)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestIsRunning:
    def test_returns_false_when_no_pid_file(self, tmp_path: Path) -> None:
        assert _is_running(tmp_path / "nope.pid") is False

    def test_returns_false_when_pid_file_is_empty(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "comfyui.pid"
        pid_file.write_text("")
        assert _is_running(pid_file) is False

    def test_returns_false_when_process_not_found(self, tmp_path: Path) -> None:
        import json

        pid_file = tmp_path / "comfyui.pid"
        pid_file.write_text(json.dumps({"pid": 999999999, "port": 8188}))
        with patch("os.kill", side_effect=ProcessLookupError):
            assert _is_running(pid_file) is False

    def test_returns_true_when_process_alive(self, tmp_path: Path) -> None:
        import json

        pid_file = tmp_path / "comfyui.pid"
        pid_file.write_text(json.dumps({"pid": 1234, "port": 8188}))
        with patch("os.kill", return_value=None):
            assert _is_running(pid_file) is True

    def test_returns_false_on_permission_error(self, tmp_path: Path) -> None:
        import json

        pid_file = tmp_path / "comfyui.pid"
        pid_file.write_text(json.dumps({"pid": 1, "port": 8188}))
        with patch("os.kill", side_effect=PermissionError):
            assert _is_running(pid_file) is False


class TestGetComfyuiMain:
    def test_raises_when_subprocess_fails(self, tmp_path: Path) -> None:
        fake_python = tmp_path / "python"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error", stdout="")
            with pytest.raises(RuntimeError, match="Could not resolve ComfyUI root"):
                _get_comfyui_main(fake_python)

    def test_raises_when_main_py_missing(self, tmp_path: Path) -> None:
        fake_python = tmp_path / "python"
        # subprocess returns a path where main.py does NOT exist
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=str(tmp_path / "comfyui"), stderr=""
            )
            with pytest.raises(RuntimeError, match="main.py not found"):
                _get_comfyui_main(fake_python)

    def test_returns_main_py_path(self, tmp_path: Path) -> None:
        fake_python = tmp_path / "python"
        comfyui_dir = tmp_path / "ComfyUI"
        comfyui_dir.mkdir()
        main_py = comfyui_dir / "main.py"
        main_py.write_text("# fake")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=str(comfyui_dir), stderr=""
            )
            result = _get_comfyui_main(fake_python)

        assert result == main_py


class TestWaitUntilReady:
    def test_returns_true_on_first_success(self) -> None:
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_cm = MagicMock()
            mock_cm.__enter__ = lambda s: s
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_cm
            result = _wait_until_ready(port=8188, timeout=5)
        assert result is True

    def test_returns_false_on_timeout(self) -> None:
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with patch("time.sleep"):  # skip actual sleeping
                result = _wait_until_ready(port=8188, timeout=0)
        assert result is False
