"""Tests for US-001 (it_000045) — ``parallax install`` command.

AC01: detect uv; if absent, download+install via urllib.request (stdlib only).
AC02: create ~/.parallax/env via uv venv; install comfy-diffusion[cuda|cpu].
AC03: call check_runtime() in venv; error dict → print + exit 1.
AC04: on success, print version and next step message.
AC05: on failure, print failing step, subprocess stderr, suggest --verbose.
AC06: already installed → print "Already installed …" message, exit 0.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _app():
    from cli.main import app
    return app


def _runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# AC01 — uv detection and auto-install
# ---------------------------------------------------------------------------


class TestUvDetection:
    """AC01: uv is detected or installed automatically."""

    def test_find_uv_returns_path_when_on_path(self, tmp_path):
        """_find_uv() returns the path when uv is on PATH."""
        from cli.commands.install import _find_uv

        with patch("shutil.which", return_value="/usr/local/bin/uv"):
            result = _find_uv()
        assert result == "/usr/local/bin/uv"

    def test_find_uv_returns_none_when_absent(self, tmp_path):
        """_find_uv() returns None when uv is not found."""
        from cli.commands.install import _find_uv

        with patch("shutil.which", return_value=None):
            # Patch the fallback candidate paths to not exist
            with patch.object(Path, "is_file", return_value=False):
                result = _find_uv()
        assert result is None

    def test_install_command_calls_download_when_uv_missing(self, tmp_path):
        """If uv is not found, install command downloads+installs it (AC01)."""
        runner = _runner()
        app = _app()

        fake_uv = str(tmp_path / "fake_uv")
        Path(fake_uv).write_text("#!/bin/sh\nexit 0\n")

        with patch("cli.commands.install._find_uv", return_value=None):
            with patch(
                "cli.commands.install._download_and_install_uv", return_value=fake_uv
            ) as mock_dl:
                with patch(
                    "cli.commands.install._installed_version", return_value=None
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                        runner.invoke(app, ["install"])

        mock_dl.assert_called_once()

    def test_install_command_skips_download_when_uv_present(self, tmp_path):
        """If uv is already present, _download_and_install_uv is never called (AC01)."""
        runner = _runner()
        app = _app()

        with patch("cli.commands.install._find_uv", return_value="/usr/bin/uv"):
            with patch(
                "cli.commands.install._download_and_install_uv"
            ) as mock_dl:
                with patch(
                    "cli.commands.install._installed_version", return_value=None
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                        runner.invoke(app, ["install"])

        mock_dl.assert_not_called()

    def test_uv_installer_uses_urllib_not_subprocess(self):
        """_download_and_install_uv uses urllib.request (stdlib); subprocess only for sh."""
        import urllib.request

        from cli.commands.install import _download_and_install_uv

        installer_content = b"#!/bin/sh\necho installed\n"
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = installer_content

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            with patch("subprocess.run", return_value=MagicMock(returncode=0)):
                with patch(
                    "cli.commands.install._find_uv", return_value="/home/user/.local/bin/uv"
                ):
                    result = _download_and_install_uv()

        mock_urlopen.assert_called_once()
        # Confirm the URL points to astral.sh (official uv installer)
        called_url = mock_urlopen.call_args[0][0]
        assert "astral.sh" in called_url
        assert result == "/home/user/.local/bin/uv"


# ---------------------------------------------------------------------------
# AC02 — venv creation and package installation
# ---------------------------------------------------------------------------


class TestVenvAndPackageInstall:
    """AC02: uv venv + uv pip install into ~/.parallax/env."""

    def _invoke_install(self, extra_args: list[str] | None = None) -> Any:
        runner = _runner()
        app = _app()

        fake_uv = "/usr/bin/uv"
        called: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            called.append(list(cmd))
            return MagicMock(returncode=0, stdout="1.3.0", stderr="")

        with patch("cli.commands.install._find_uv", return_value=fake_uv):
            with patch("cli.commands.install._installed_version", side_effect=[None, "1.3.0"]):
                with patch("subprocess.run", side_effect=fake_run):
                    result = runner.invoke(app, ["install"] + (extra_args or []))

        return result, called

    def test_creates_venv_at_parallax_env(self):
        """uv venv ~/.parallax/env is called (AC02)."""
        result, called = self._invoke_install()
        venv_calls = [c for c in called if "venv" in c]
        assert len(venv_calls) >= 1
        assert any(str(Path.home() / ".parallax" / "env") in " ".join(c) for c in venv_calls)

    def test_installs_cuda_package_by_default(self):
        """comfy-diffusion[cuda] is installed by default (AC02)."""
        result, called = self._invoke_install()
        install_calls = [c for c in called if "pip" in c and "install" in c]
        assert any("comfy-diffusion[cuda]" in " ".join(c) for c in install_calls)

    def test_installs_cpu_package_with_flag(self):
        """comfy-diffusion[cpu] is installed when --cpu is passed (AC02)."""
        result, called = self._invoke_install(["--cpu"])
        install_calls = [c for c in called if "pip" in c and "install" in c]
        assert any("comfy-diffusion[cpu]" in " ".join(c) for c in install_calls)

    def test_install_targets_venv_python(self):
        """pip install uses --python pointing into the venv (AC02)."""
        result, called = self._invoke_install()
        install_calls = [c for c in called if "pip" in c and "install" in c]
        assert len(install_calls) >= 1
        joined = " ".join(install_calls[0])
        assert "--python" in joined
        assert ".parallax" in joined


# ---------------------------------------------------------------------------
# AC03 — check_runtime() error handling
# ---------------------------------------------------------------------------


class TestCheckRuntime:
    """AC03: check_runtime() error dict causes exit 1."""

    def test_check_runtime_error_exits_1(self):
        """If check_runtime() returns an error dict, exit code is 1 (AC03)."""
        runner = _runner()
        app = _app()

        def fake_run(cmd, **kwargs):
            # First two calls succeed (venv, pip install)
            # Third call (bootstrap) returns error
            if "venv" in cmd or ("pip" in cmd and "install" in cmd):
                return MagicMock(returncode=0, stdout="", stderr="")
            # bootstrap check_runtime
            return MagicMock(returncode=1, stdout='{"error": "ComfyUI not found"}', stderr="")

        with patch("cli.commands.install._find_uv", return_value="/usr/bin/uv"):
            with patch("cli.commands.install._installed_version", return_value=None):
                with patch("subprocess.run", side_effect=fake_run):
                    result = runner.invoke(app, ["install"])

        assert result.exit_code == 1

    def test_check_runtime_success_continues(self):
        """If check_runtime() succeeds, command continues to AC04 (AC03)."""
        runner = _runner()
        app = _app()

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch("cli.commands.install._find_uv", return_value="/usr/bin/uv"):
            with patch(
                "cli.commands.install._installed_version", side_effect=[None, "1.3.0"]
            ):
                with patch("subprocess.run", side_effect=fake_run):
                    result = runner.invoke(app, ["install"])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC04 — success message
# ---------------------------------------------------------------------------


class TestSuccessMessage:
    """AC04: on success, print installed version and next step."""

    def test_prints_installed_version(self):
        """Prints the installed version after success (AC04)."""
        runner = _runner()
        app = _app()

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch("cli.commands.install._find_uv", return_value="/usr/bin/uv"):
            with patch(
                "cli.commands.install._installed_version", side_effect=[None, "1.3.0"]
            ):
                with patch("subprocess.run", side_effect=fake_run):
                    result = runner.invoke(app, ["install"])

        assert result.exit_code == 0, result.output
        assert "1.3.0" in result.output

    def test_prints_next_step_message(self):
        """Prints 'Run `parallax ms install`' after success (AC04)."""
        runner = _runner()
        app = _app()

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch("cli.commands.install._find_uv", return_value="/usr/bin/uv"):
            with patch(
                "cli.commands.install._installed_version", side_effect=[None, "1.3.0"]
            ):
                with patch("subprocess.run", side_effect=fake_run):
                    result = runner.invoke(app, ["install"])

        assert "parallax ms install" in result.output


# ---------------------------------------------------------------------------
# AC05 — failure reporting
# ---------------------------------------------------------------------------


class TestFailureReporting:
    """AC05: on failure, print step name, stderr, and --verbose suggestion."""

    def test_venv_failure_prints_step_and_suggests_verbose(self):
        """uv venv failure prints step name and --verbose hint (AC05)."""
        runner = _runner()
        app = _app()

        def fake_run(cmd, **kwargs):
            if "venv" in cmd:
                return MagicMock(returncode=1, stdout="", stderr="Permission denied")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("cli.commands.install._find_uv", return_value="/usr/bin/uv"):
            with patch("cli.commands.install._installed_version", return_value=None):
                with patch("subprocess.run", side_effect=fake_run):
                    result = runner.invoke(app, ["install"])

        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "uv venv" in combined
        assert "--verbose" in combined

    def test_pip_install_failure_prints_stderr(self):
        """pip install failure prints step and subprocess error output (AC05)."""
        runner = _runner()
        app = _app()

        def fake_run(cmd, **kwargs):
            if "pip" in cmd and "install" in cmd:
                return MagicMock(returncode=1, stdout="", stderr="No matching distribution")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("cli.commands.install._find_uv", return_value="/usr/bin/uv"):
            with patch("cli.commands.install._installed_version", return_value=None):
                with patch("subprocess.run", side_effect=fake_run):
                    result = runner.invoke(app, ["install"])

        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "--verbose" in combined

    def test_uv_ensure_failure_exits_1(self):
        """Failure to obtain uv (no download possible) exits 1 (AC05)."""
        runner = _runner()
        app = _app()

        with patch("cli.commands.install._find_uv", return_value=None):
            with patch(
                "cli.commands.install._download_and_install_uv",
                side_effect=RuntimeError("Network unreachable"),
            ):
                with patch("cli.commands.install._installed_version", return_value=None):
                    result = runner.invoke(app, ["install"])

        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "--verbose" in combined


# ---------------------------------------------------------------------------
# AC06 — already-installed guard
# ---------------------------------------------------------------------------


class TestAlreadyInstalled:
    """AC06: re-running when installed prints message and exits 0."""

    def test_already_installed_prints_message(self):
        """Prints 'Already installed (v...)' when env exists (AC06)."""
        runner = _runner()
        app = _app()

        with patch("cli.commands.install._installed_version", return_value="1.3.0"):
            result = runner.invoke(app, ["install"])

        assert result.exit_code == 0
        assert "Already installed" in result.output
        assert "1.3.0" in result.output

    def test_already_installed_mentions_upgrade_flag(self):
        """Message includes 'parallax install --upgrade' (AC06)."""
        runner = _runner()
        app = _app()

        with patch("cli.commands.install._installed_version", return_value="1.2.0"):
            result = runner.invoke(app, ["install"])

        assert "--upgrade" in result.output

    def test_already_installed_performs_no_work(self):
        """subprocess.run is never called when already installed (AC06)."""
        runner = _runner()
        app = _app()

        with patch("cli.commands.install._installed_version", return_value="1.3.0"):
            with patch("subprocess.run") as mock_run:
                runner.invoke(app, ["install"])

        mock_run.assert_not_called()

    def test_upgrade_flag_bypasses_guard(self):
        """--upgrade flag skips the 'already installed' guard (AC06)."""
        runner = _runner()
        app = _app()

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch("cli.commands.install._installed_version", side_effect=["1.3.0", "1.3.0"]):
            with patch("cli.commands.install._find_uv", return_value="/usr/bin/uv"):
                with patch("subprocess.run", side_effect=fake_run):
                    result = runner.invoke(app, ["install", "--upgrade"])

        # Should not print "Already installed"
        assert "Already installed" not in result.output

    def test_install_help_shows_flags(self):
        """--help shows --cpu, --upgrade, --verbose flags."""
        runner = _runner()
        app = _app()
        result = runner.invoke(app, ["install", "--help"])
        assert result.exit_code == 0
        assert "--cpu" in result.output
        assert "--upgrade" in result.output
        assert "--verbose" in result.output
