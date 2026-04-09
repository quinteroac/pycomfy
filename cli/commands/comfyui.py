"""``parallax comfyui`` — manage ComfyUI web UI as a background process.

US-001 acceptance criteria:
  AC01 — launches ComfyUI using the runtime at ~/.parallax/env
  AC02 — default port 8188
  AC03 — writes PID to ~/.config/parallax/comfyui.pid
  AC04 — prints URL once the server is ready
  AC05 — if already running, prints warning and exits without starting a second instance
  AC06 — typecheck / lint passes

US-002 acceptance criteria:
  AC01 — reads PID from ~/.config/parallax/comfyui.pid and terminates the process
  AC02 — PID file removed after successful stop
  AC03 — if no instance is running, prints informative message and exits with code 0
  AC04 — cross-platform: SIGTERM on Linux/macOS, TerminateProcess on Windows
  AC05 — typecheck / lint passes

US-003 acceptance criteria:
  AC01 — --port <N> starts ComfyUI on the given port
  AC02 — printed URL reflects the custom port
  AC03 — invalid port (< 1 or > 65535) prints error and exits with non-zero code
  AC04 — typecheck / lint passes

US-004 acceptance criteria:
  AC01 — --open opens http://localhost:<port> in the default browser once server is ready
  AC02 — uses webbrowser.open() (stdlib) — no additional dependency
  AC03 — works on Linux, macOS, and Windows
  AC04 — typecheck / lint passes
  AC05 — visually verified in browser: ComfyUI interface loads
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from cli.commands._common import ENV_DIR as _ENV_DIR

app = typer.Typer(name="comfyui", help="Manage ComfyUI web UI.", no_args_is_help=True)

_DEFAULT_PORT = 8188
_READY_TIMEOUT = 60  # seconds


def _python_path() -> Path:
    """Return path to the Python interpreter in the parallax env."""
    return _ENV_DIR / "bin" / "python"


def _pid_file() -> Path:
    """Return the canonical PID file path for the ComfyUI process."""
    return Path.home() / ".config" / "parallax" / "comfyui.pid"


def _get_comfyui_main(python: Path) -> Path:
    """Resolve ComfyUI main.py path using the installed env."""
    result = subprocess.run(
        [
            str(python),
            "-c",
            "from comfy_diffusion._runtime import _comfyui_root; print(_comfyui_root())",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not resolve ComfyUI root: {result.stderr.strip() or result.stdout.strip()}"
        )
    comfyui_root = Path(result.stdout.strip())
    main_py = comfyui_root / "main.py"
    if not main_py.exists():
        raise RuntimeError(f"ComfyUI main.py not found at {main_py}")
    return main_py


def _is_running(pid_file: Path) -> bool:
    """Return True if the PID in *pid_file* refers to a live process (AC05)."""
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _wait_until_ready(port: int, timeout: int = _READY_TIMEOUT) -> bool:
    """Poll http://localhost:<port> until it responds or *timeout* expires (AC04)."""
    url = f"http://localhost:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):  # noqa: S310
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def _terminate_process(pid: int) -> None:
    """Terminate a process: SIGTERM on POSIX, TerminateProcess on Windows (US-002 AC04)."""
    if sys.platform == "win32":
        # On Windows, os.kill with SIGTERM calls TerminateProcess via the CRT.
        os.kill(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)


def _open_browser(url: str) -> None:
    """Open *url* in the default browser (US-004 AC02)."""
    webbrowser.open(url)


@app.command("start")
def start(
    port: Annotated[
        int,
        typer.Option("--port", help="Port for ComfyUI to listen on."),
    ] = _DEFAULT_PORT,
    timeout: Annotated[
        int,
        typer.Option("--timeout", help="Seconds to wait for ComfyUI to become ready."),
    ] = _READY_TIMEOUT,
    open_browser: Annotated[
        bool,
        typer.Option("--open", help="Open the ComfyUI URL in the default browser once ready."),
    ] = False,
) -> None:
    """Launch ComfyUI web UI as a background process."""
    # US-003 AC03 — validate port range
    if port < 1 or port > 65535:
        typer.echo(f"Error: invalid port {port}. Must be between 1 and 65535.", err=True)
        raise typer.Exit(1)

    python = _python_path()
    if not python.exists():
        typer.echo("Run `parallax install` first.", err=True)
        raise typer.Exit(1)

    pid_file = _pid_file()

    # AC05 — already running guard
    if _is_running(pid_file):
        pid_str = pid_file.read_text(encoding="utf-8").strip()
        typer.echo(
            f"Warning: ComfyUI is already running (PID {pid_str}). "
            "Stop it first before starting a new instance."
        )
        return

    # Resolve ComfyUI main.py from the installed env
    try:
        comfyui_main = _get_comfyui_main(python)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    # AC01 + AC02 — launch ComfyUI as a detached background process
    proc = subprocess.Popen(
        [str(python), str(comfyui_main), "--port", str(port), "--listen", "0.0.0.0"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # AC03 — write PID file so the process can be tracked
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(proc.pid), encoding="utf-8")

    # AC04 — wait for the server to be ready, then print the URL
    typer.echo(f"Starting ComfyUI on port {port}…")
    ready = _wait_until_ready(port, timeout=timeout)
    url = f"http://localhost:{port}"
    if ready:
        typer.echo(f"ComfyUI is running at {url}")
        # US-004 AC01 — open browser if requested and server is ready
        if open_browser:
            _open_browser(url)
    else:
        typer.echo(
            f"Warning: ComfyUI did not respond within {timeout}s. "
            f"It may still be loading — check {url}"
        )


@app.command("stop")
def stop() -> None:
    """Stop the running ComfyUI process."""
    pid_file = _pid_file()

    # US-002 AC03 — no instance running
    if not _is_running(pid_file):
        typer.echo("No ComfyUI instance is currently running.")
        return

    pid = int(pid_file.read_text(encoding="utf-8").strip())

    # US-002 AC01 + AC04 — terminate the process
    try:
        _terminate_process(pid)
    except (ProcessLookupError, PermissionError) as exc:
        typer.echo(f"Error: could not terminate process {pid}: {exc}", err=True)
        raise typer.Exit(1)

    # US-002 AC02 — remove PID file after successful stop
    try:
        pid_file.unlink()
    except OSError:
        pass

    typer.echo(f"ComfyUI (PID {pid}) stopped.")
