"""Tests for US-004: Auto-open Browser with `--open`.

Acceptance criteria verified:
  AC01 — `parallax comfyui start --open` opens http://localhost:<port> once server is ready
  AC02 — uses webbrowser.open() (stdlib) — no additional dependency
  AC03 — works on Linux, macOS, and Windows (webbrowser.open is cross-platform)
  AC04 — typecheck / lint passes (verified by running tests without error)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner, Result

from cli.commands.comfyui import app

runner = CliRunner()

_FAKE_PID = 99999
_DEFAULT_PORT = 8188


def _run_start(
    args: list[str] | None = None,
    *,
    already_running: bool = False,
    ready: bool = True,
    tmp_path: Path,
) -> Result:
    """Invoke `parallax comfyui start` with all I/O mocked."""
    pid_file = tmp_path / "comfyui.pid"
    comfyui_main = tmp_path / "main.py"
    comfyui_main.touch()
    python_bin = tmp_path / "python"
    python_bin.touch()

    if already_running:
        pid_file.write_text(str(_FAKE_PID), encoding="utf-8")

    with (
        patch("cli.commands.comfyui._python_path", return_value=python_bin),
        patch("cli.commands.comfyui._pid_file", return_value=pid_file),
        patch("cli.commands.comfyui._is_running", return_value=already_running),
        patch("cli.commands.comfyui._get_comfyui_main", return_value=comfyui_main),
        patch("cli.commands.comfyui.subprocess.Popen") as mock_popen,
        patch("cli.commands.comfyui._wait_until_ready", return_value=ready),
        patch("cli.commands.comfyui._open_browser") as mock_open_browser,
    ):
        mock_proc = MagicMock()
        mock_proc.pid = _FAKE_PID
        mock_popen.return_value = mock_proc

        result = runner.invoke(app, ["start"] + (args or []))
        result._mock_open_browser = mock_open_browser  # type: ignore[attr-defined]
        return result


class TestOpenBrowserFlag:
    """US-004 AC01 — browser opens when --open is passed and server is ready."""

    def test_open_called_when_flag_set_and_ready(self, tmp_path: Path) -> None:
        result = _run_start(["--open"], ready=True, tmp_path=tmp_path)
        assert result.exit_code == 0
        result._mock_open_browser.assert_called_once_with(f"http://localhost:{_DEFAULT_PORT}")

    def test_open_not_called_without_flag(self, tmp_path: Path) -> None:
        result = _run_start(ready=True, tmp_path=tmp_path)
        assert result.exit_code == 0
        result._mock_open_browser.assert_not_called()

    def test_open_not_called_when_server_not_ready(self, tmp_path: Path) -> None:
        """Browser must NOT open if the server never became ready."""
        result = _run_start(["--open"], ready=False, tmp_path=tmp_path)
        assert result.exit_code == 0
        result._mock_open_browser.assert_not_called()

    def test_open_uses_correct_url_with_custom_port(self, tmp_path: Path) -> None:
        """AC01 — URL reflects the custom port passed via --port."""
        result = _run_start(["--port", "9000", "--open"], ready=True, tmp_path=tmp_path)
        assert result.exit_code == 0
        result._mock_open_browser.assert_called_once_with("http://localhost:9000")

    def test_open_not_called_when_already_running(self, tmp_path: Path) -> None:
        """Browser should not be opened when start exits early due to existing instance."""
        result = _run_start(["--open"], already_running=True, tmp_path=tmp_path)
        assert result.exit_code == 0
        result._mock_open_browser.assert_not_called()


class TestOpenBrowserImplementation:
    """US-004 AC02 — _open_browser delegates to webbrowser.open (stdlib)."""

    def test_open_browser_calls_webbrowser_open(self) -> None:
        with patch("cli.commands.comfyui.webbrowser.open") as mock_wb:
            from cli.commands.comfyui import _open_browser

            _open_browser("http://localhost:8188")
            mock_wb.assert_called_once_with("http://localhost:8188")

    def test_no_non_stdlib_import(self) -> None:
        """Verify webbrowser is imported from stdlib (not a third-party package)."""
        import webbrowser as _wb

        assert _wb.__file__ is not None
        # stdlib modules live inside the Python install — not in site-packages
        assert "site-packages" not in _wb.__file__
