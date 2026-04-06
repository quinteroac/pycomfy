"""``parallax ms install`` — register the inference server as a system service.

Acceptance criteria implemented:
  AC01 — check ~/.parallax/env exists (parallax install must run first); exit 1 if not
  AC02 — Linux: write systemd user unit + systemctl --user enable --now
  AC03 — macOS: write launchd plist + launchctl load
  AC04 — service runs python -m uvicorn server.main:app --host 0.0.0.0 --port 5000
          using the Python interpreter from ~/.parallax/env
  AC05 — print "Inference server running on http://localhost:5000" + one status line
  AC06 — already registered → print "Already registered." and exit 0
"""

from __future__ import annotations

import platform
import subprocess
import textwrap
from pathlib import Path

import typer

app = typer.Typer(name="ms", help="Manage inference server.", no_args_is_help=True)

_ENV_DIR = Path.home() / ".parallax" / "env"
_SERVICE_NAME = "parallax-ms"
_LAUNCHD_LABEL = "run.parallax.ms"


# ---------------------------------------------------------------------------
# Internal helpers (extracted for testability)
# ---------------------------------------------------------------------------


def _env_installed() -> bool:
    """Return True if ``parallax install`` has been run (AC01)."""
    return (_ENV_DIR / "bin" / "python").exists()


def _python_path() -> Path:
    """Return the path to the Python interpreter in the parallax env (AC04)."""
    return _ENV_DIR / "bin" / "python"


def _systemd_unit_path() -> Path:
    """Return the path for the systemd user unit file (AC02)."""
    return Path.home() / ".config" / "systemd" / "user" / f"{_SERVICE_NAME}.service"


def _launchd_plist_path() -> Path:
    """Return the path for the launchd plist file (AC03)."""
    return Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCHD_LABEL}.plist"


def _write_systemd_unit(python: Path) -> Path:
    """Write the systemd user unit file; return its path (AC02, AC04)."""
    unit_path = _systemd_unit_path()
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    content = textwrap.dedent(f"""\
        [Unit]
        Description=Parallax Inference Server
        After=network.target

        [Service]
        ExecStart={python} -m uvicorn server.main:app --host 0.0.0.0 --port 5000
        Restart=on-failure

        [Install]
        WantedBy=default.target
    """)
    unit_path.write_text(content, encoding="utf-8")
    return unit_path


def _write_launchd_plist(python: Path) -> Path:
    """Write the launchd plist file; return its path (AC03, AC04)."""
    plist_path = _launchd_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    content = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
            "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{_LAUNCHD_LABEL}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{python}</string>
                <string>-m</string>
                <string>uvicorn</string>
                <string>server.main:app</string>
                <string>--host</string>
                <string>0.0.0.0</string>
                <string>--port</string>
                <string>5000</string>
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
        </dict>
        </plist>
    """)
    plist_path.write_text(content, encoding="utf-8")
    return plist_path


def _get_service_status_line(system: str) -> str:
    """Return one line describing service status for AC05."""
    if system == "Linux":
        result = subprocess.run(
            ["systemctl", "--user", "is-active", _SERVICE_NAME],
            capture_output=True,
            text=True,
        )
        return f"Service status: {result.stdout.strip() or 'unknown'}"
    if system == "Darwin":
        result = subprocess.run(
            ["launchctl", "list", _LAUNCHD_LABEL],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().splitlines()
        return f"Service status: {lines[0] if lines else 'loaded'}"
    return ""


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------


@app.command("install")
def install() -> None:
    """Register the Parallax inference server as a system service."""
    # AC01 — ensure parallax install has been run first
    if not _env_installed():
        typer.echo("Run `parallax install` first.")
        raise typer.Exit(1)

    system = platform.system()
    python = _python_path()

    if system == "Linux":
        unit_path = _systemd_unit_path()

        # AC06 — already registered
        if unit_path.exists():
            typer.echo("Already registered.")
            return

        # AC02 + AC04 — write unit file
        _write_systemd_unit(python)

        # AC02 — enable and start via systemctl
        result = subprocess.run(
            ["systemctl", "--user", "enable", "--now", _SERVICE_NAME],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            typer.echo(result.stderr.strip() or result.stdout.strip(), err=True)
            raise typer.Exit(1)

    elif system == "Darwin":
        plist_path = _launchd_plist_path()

        # AC06 — already registered
        if plist_path.exists():
            typer.echo("Already registered.")
            return

        # AC03 + AC04 — write plist file
        _write_launchd_plist(python)

        # AC03 — register via launchctl load
        result = subprocess.run(
            ["launchctl", "load", str(plist_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            typer.echo(result.stderr.strip() or result.stdout.strip(), err=True)
            raise typer.Exit(1)

    else:
        typer.echo(f"Unsupported platform: {system}", err=True)
        raise typer.Exit(1)

    # AC05 — success message + one line of service status
    typer.echo("Inference server running on http://localhost:5000")
    status_line = _get_service_status_line(system)
    if status_line:
        typer.echo(status_line)
