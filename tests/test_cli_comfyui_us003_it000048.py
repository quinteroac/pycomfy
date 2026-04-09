"""Tests for US-003 it_000048 — parallax comfyui start --port option.

Covers:
  AC01 — --port <N> starts ComfyUI on the given port
  AC02 — printed URL reflects the custom port
  AC03 — invalid port (< 1 or > 65535) prints error and exits with non-zero code
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from cli.commands.comfyui import app

runner = CliRunner()

# Wrap sub-app in a parent so ["comfyui", "start"] routes correctly.
_cli = typer.Typer()
_cli.add_typer(app, name="comfyui")


def _fake_popen(pid: int = 12345) -> MagicMock:
    mock = MagicMock()
    mock.pid = pid
    return mock


def _run_start(tmp_path: Path, *, port: int) -> object:
    """Invoke ``parallax comfyui start --port <port>`` with all I/O mocked."""
    fake_python = tmp_path / "bin" / "python"
    fake_python.parent.mkdir(parents=True, exist_ok=True)
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_python.chmod(0o755)

    comfyui_main = tmp_path / "main.py"
    comfyui_main.write_text("# fake", encoding="utf-8")

    pid_file = tmp_path / "comfyui.pid"

    with (
        patch("cli.commands.comfyui._python_path", return_value=fake_python),
        patch("cli.commands.comfyui._pid_file", return_value=pid_file),
        patch("cli.commands.comfyui._is_running", return_value=False),
        patch("cli.commands.comfyui._get_comfyui_main", return_value=comfyui_main),
        patch("cli.commands.comfyui._wait_until_ready", return_value=True),
        patch("subprocess.Popen", return_value=_fake_popen()),
    ):
        result = runner.invoke(_cli, ["comfyui", "start", "--port", str(port)])

    return result


# ---------------------------------------------------------------------------
# AC01 — custom port is passed to the subprocess
# ---------------------------------------------------------------------------


class TestAC01CustomPort:
    def test_custom_port_passed_to_popen(self, tmp_path: Path) -> None:
        popen_calls: list = []
        fake_python = tmp_path / "bin" / "python"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_python.chmod(0o755)
        comfyui_main = tmp_path / "main.py"
        comfyui_main.write_text("# fake", encoding="utf-8")
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
            result = runner.invoke(_cli, ["comfyui", "start", "--port", "8189"])

        assert result.exit_code == 0
        assert popen_calls, "subprocess.Popen was never called"
        cmd = popen_calls[0]
        assert "--port" in cmd
        port_idx = cmd.index("--port")
        assert cmd[port_idx + 1] == "8189"

    def test_start_exits_zero_with_custom_port(self, tmp_path: Path) -> None:
        result = _run_start(tmp_path, port=8189)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC02 — printed URL reflects the custom port
# ---------------------------------------------------------------------------


class TestAC02UrlReflectsPort:
    def test_url_contains_custom_port(self, tmp_path: Path) -> None:
        result = _run_start(tmp_path, port=8189)
        assert "http://localhost:8189" in result.output

    def test_url_does_not_contain_default_port_when_overridden(self, tmp_path: Path) -> None:
        result = _run_start(tmp_path, port=9999)
        assert "http://localhost:9999" in result.output
        # Should not show default port 8188 in the URL when overridden
        assert "http://localhost:8188" not in result.output


# ---------------------------------------------------------------------------
# AC03 — invalid port prints error and exits non-zero
# ---------------------------------------------------------------------------


class TestAC03InvalidPort:
    def test_port_zero_exits_nonzero(self, tmp_path: Path) -> None:
        result = _run_start(tmp_path, port=0)
        assert result.exit_code != 0

    def test_port_zero_prints_error(self, tmp_path: Path) -> None:
        result = _run_start(tmp_path, port=0)
        assert "Error" in result.output or "error" in result.output.lower()

    def test_port_negative_exits_nonzero(self, tmp_path: Path) -> None:
        result = _run_start(tmp_path, port=-1)
        assert result.exit_code != 0

    def test_port_65536_exits_nonzero(self, tmp_path: Path) -> None:
        result = _run_start(tmp_path, port=65536)
        assert result.exit_code != 0

    def test_port_65536_prints_error(self, tmp_path: Path) -> None:
        result = _run_start(tmp_path, port=65536)
        assert "Error" in result.output or "error" in result.output.lower()

    def test_port_boundary_1_is_valid(self, tmp_path: Path) -> None:
        result = _run_start(tmp_path, port=1)
        assert result.exit_code == 0

    def test_port_boundary_65535_is_valid(self, tmp_path: Path) -> None:
        result = _run_start(tmp_path, port=65535)
        assert result.exit_code == 0
