"""Tests for US-003: Register the FastAPI server as a system service.

Covers:
  AC01 — exits 1 with "Run `parallax install` first." when env missing
  AC02 — Linux: writes systemd unit file, runs systemctl --user enable --now
  AC03 — macOS: writes launchd plist, runs launchctl load
  AC04 — unit/plist uses python -m uvicorn server.main:app --host 0.0.0.0 --port 5000
          with the Python interpreter from ~/.parallax/env
  AC05 — prints "Inference server running on http://localhost:5000" + status line
  AC06 — re-running prints "Already registered." and exits 0
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

runner = CliRunner()


def _app():
    from cli.main import app
    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_python(tmp_path: Path) -> Path:
    """Create a fake python binary inside a temp env dir."""
    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    return python


def _make_subprocess_ok(stdout: str = "active\n") -> MagicMock:
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = stdout
    proc.stderr = ""
    return proc


# ---------------------------------------------------------------------------
# AC01 — env not installed
# ---------------------------------------------------------------------------


class TestAC01EnvNotInstalled:
    def test_exits_1_when_env_missing(self, tmp_path):
        """AC01: no ~/.parallax/env → exit 1 and correct message."""
        env_dir = tmp_path / "env"  # does NOT exist

        with (
            patch("cli.commands.ms._ENV_DIR", env_dir),
            patch("platform.system", return_value="Linux"),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 1
        assert "Run `parallax install` first." in result.output

    def test_exits_1_when_python_binary_missing(self, tmp_path):
        """AC01: env dir exists but python binary absent → exit 1."""
        env_dir = tmp_path / "env"
        env_dir.mkdir(parents=True)  # no bin/python inside

        with (
            patch("cli.commands.ms._ENV_DIR", env_dir),
            patch("platform.system", return_value="Linux"),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 1
        assert "Run `parallax install` first." in result.output


# ---------------------------------------------------------------------------
# AC02 — Linux: systemd unit
# ---------------------------------------------------------------------------


class TestAC02Linux:
    def test_writes_systemd_unit_file(self, tmp_path):
        """AC02: unit file is created at the expected path."""
        _fake_python(tmp_path)
        unit_path = tmp_path / "parallax-ms.service"

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._systemd_unit_path", return_value=unit_path),
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", return_value=_make_subprocess_ok()),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 0, result.output
        assert unit_path.exists()

    def test_runs_systemctl_enable_now(self, tmp_path):
        """AC02: systemctl --user enable --now parallax-ms is called."""
        _fake_python(tmp_path)
        unit_path = tmp_path / "parallax-ms.service"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return _make_subprocess_ok()

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._systemd_unit_path", return_value=unit_path),
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 0, result.output
        enable_calls = [c for c in calls if "systemctl" in c and "enable" in c and "--now" in c]
        assert enable_calls, f"systemctl enable --now not called; calls: {calls}"
        assert "parallax-ms" in enable_calls[0]

    def test_exits_1_on_systemctl_failure(self, tmp_path):
        """AC02: if systemctl returns non-zero, command exits 1."""
        _fake_python(tmp_path)
        unit_path = tmp_path / "parallax-ms.service"

        fail_proc = MagicMock()
        fail_proc.returncode = 1
        fail_proc.stdout = ""
        fail_proc.stderr = "Failed to enable unit"

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._systemd_unit_path", return_value=unit_path),
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", return_value=fail_proc),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# AC03 — macOS: launchd plist
# ---------------------------------------------------------------------------


class TestAC03macOS:
    def test_writes_launchd_plist(self, tmp_path):
        """AC03: plist file is created at the expected path."""
        _fake_python(tmp_path)
        plist_path = tmp_path / "run.parallax.ms.plist"

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._launchd_plist_path", return_value=plist_path),
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", return_value=_make_subprocess_ok("loaded\n")),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 0, result.output
        assert plist_path.exists()

    def test_runs_launchctl_load(self, tmp_path):
        """AC03: launchctl load <plist> is called."""
        _fake_python(tmp_path)
        plist_path = tmp_path / "run.parallax.ms.plist"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return _make_subprocess_ok("loaded\n")

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._launchd_plist_path", return_value=plist_path),
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 0, result.output
        load_calls = [c for c in calls if "launchctl" in c and "load" in c]
        assert load_calls, f"launchctl load not called; calls: {calls}"
        assert str(plist_path) in load_calls[0]

    def test_exits_1_on_launchctl_failure(self, tmp_path):
        """AC03: if launchctl returns non-zero, command exits 1."""
        _fake_python(tmp_path)
        plist_path = tmp_path / "run.parallax.ms.plist"

        fail_proc = MagicMock()
        fail_proc.returncode = 1
        fail_proc.stdout = ""
        fail_proc.stderr = "Failed to load"

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._launchd_plist_path", return_value=plist_path),
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", return_value=fail_proc),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# AC04 — unit/plist uses correct command with env python
# ---------------------------------------------------------------------------


class TestAC04ServiceCommand:
    def test_systemd_unit_contains_python_and_uvicorn(self, tmp_path):
        """AC04: unit file uses ~/.parallax/env/bin/python -m uvicorn ..."""
        python = _fake_python(tmp_path)
        unit_path = tmp_path / "parallax-ms2.service"
        unit_path.parent.mkdir(parents=True, exist_ok=True)

        with patch("cli.commands.ms._systemd_unit_path", return_value=unit_path):
            from cli.commands.ms import _write_systemd_unit
            _write_systemd_unit(python)

        content = unit_path.read_text()
        assert str(python) in content
        assert "-m uvicorn" in content
        assert "server.main:app" in content
        assert "--host 0.0.0.0" in content
        assert "--port 5000" in content

    def test_launchd_plist_contains_python_and_uvicorn(self, tmp_path):
        """AC04: plist uses ~/.parallax/env/bin/python -m uvicorn ..."""
        python = _fake_python(tmp_path)
        plist_path = tmp_path / "run.parallax.ms.plist"
        plist_path.parent.mkdir(parents=True, exist_ok=True)

        with patch("cli.commands.ms._launchd_plist_path", return_value=plist_path):
            from cli.commands.ms import _write_launchd_plist
            _write_launchd_plist(python)

        content = plist_path.read_text()
        assert str(python) in content
        assert "uvicorn" in content
        assert "server.main:app" in content
        assert "0.0.0.0" in content
        assert "5000" in content

    def test_python_path_is_from_parallax_env(self, tmp_path):
        """AC04: _python_path() points to ~/.parallax/env/bin/python."""
        env_dir = tmp_path / "env"
        with patch("cli.commands.ms._ENV_DIR", env_dir):
            from cli.commands.ms import _python_path
            result = _python_path()
        assert result == env_dir / "bin" / "python"


# ---------------------------------------------------------------------------
# AC05 — success output
# ---------------------------------------------------------------------------


class TestAC05SuccessOutput:
    def test_prints_inference_server_url_linux(self, tmp_path):
        """AC05: success prints the inference server URL on Linux."""
        _fake_python(tmp_path)
        unit_path = tmp_path / "parallax-ms.service"

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._systemd_unit_path", return_value=unit_path),
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", return_value=_make_subprocess_ok()),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 0, result.output
        assert "Inference server running on http://localhost:5000" in result.output

    def test_prints_service_status_line_linux(self, tmp_path):
        """AC05: a service status line is printed after the URL on Linux."""
        _fake_python(tmp_path)
        unit_path = tmp_path / "parallax-ms.service"

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._systemd_unit_path", return_value=unit_path),
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", return_value=_make_subprocess_ok()),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 0, result.output
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert len(lines) >= 2, f"Expected at least 2 output lines, got: {result.output!r}"

    def test_prints_inference_server_url_macos(self, tmp_path):
        """AC05: success prints the inference server URL on macOS."""
        _fake_python(tmp_path)
        plist_path = tmp_path / "run.parallax.ms.plist"

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._launchd_plist_path", return_value=plist_path),
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", return_value=_make_subprocess_ok("loaded\n")),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 0, result.output
        assert "Inference server running on http://localhost:5000" in result.output


# ---------------------------------------------------------------------------
# AC06 — idempotent re-run
# ---------------------------------------------------------------------------


class TestAC06AlreadyRegistered:
    def test_linux_already_registered(self, tmp_path):
        """AC06: unit file exists → 'Already registered.' exit 0."""
        _fake_python(tmp_path)
        unit_path = tmp_path / "parallax-ms.service"
        unit_path.touch()  # simulate already registered

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._systemd_unit_path", return_value=unit_path),
            patch("platform.system", return_value="Linux"),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 0
        assert "Already registered." in result.output

    def test_linux_already_registered_no_subprocess(self, tmp_path):
        """AC06: if already registered, no systemctl is called."""
        _fake_python(tmp_path)
        unit_path = tmp_path / "parallax-ms.service"
        unit_path.touch()

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._systemd_unit_path", return_value=unit_path),
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run") as mock_run,
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        mock_run.assert_not_called()
        assert result.exit_code == 0

    def test_macos_already_registered(self, tmp_path):
        """AC06: plist exists → 'Already registered.' exit 0."""
        _fake_python(tmp_path)
        plist_path = tmp_path / "run.parallax.ms.plist"
        plist_path.touch()

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._launchd_plist_path", return_value=plist_path),
            patch("platform.system", return_value="Darwin"),
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        assert result.exit_code == 0
        assert "Already registered." in result.output

    def test_macos_already_registered_no_subprocess(self, tmp_path):
        """AC06: if already registered on macOS, no launchctl is called."""
        _fake_python(tmp_path)
        plist_path = tmp_path / "run.parallax.ms.plist"
        plist_path.touch()

        with (
            patch("cli.commands.ms._ENV_DIR", tmp_path / "env"),
            patch("cli.commands.ms._launchd_plist_path", return_value=plist_path),
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run") as mock_run,
        ):
            result = runner.invoke(_app(), ["ms", "install"])

        mock_run.assert_not_called()
        assert result.exit_code == 0
