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
  AC04 — cross-platform: SIGTERM on Linux/macOS, taskkill on Windows
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

US-005 acceptance criteria:
  AC01 — if running: prints "ComfyUI is running (PID <pid>, port <port>)"
  AC02 — if not running: prints "ComfyUI is not running" and exits with code 0
  AC03 — port is read from the PID file metadata (stored alongside PID at start time)
  AC04 — typecheck / lint passes
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from cli.commands._common import ENV_DIR as _ENV_DIR

app = typer.Typer(name="comfyui", help="Manage ComfyUI web UI.", no_args_is_help=True)

_DEFAULT_PORT = 8188
_READY_TIMEOUT = 30  # seconds — FR-5


def _python_path() -> Path:
    """Return path to the Python interpreter in the parallax env."""
    return _ENV_DIR / "bin" / "python"


def _config_dir() -> Path:
    """Return the parallax config directory, honouring ``PARALLAX_CONFIG_DIR`` if set."""
    override = os.environ.get("PARALLAX_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "parallax"


def _pid_file() -> Path:
    """Return the canonical PID file path for the ComfyUI process."""
    return _config_dir() / "comfyui.pid"


def _log_file() -> Path:
    """Return the canonical log file path for the ComfyUI process (FR-8)."""
    return _config_dir() / "comfyui.log"


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


def _write_pid_file(pid_file: Path, pid: int, port: int) -> None:
    """Write *pid* and *port* to *pid_file* in ``pid:<N>\\nport:<P>`` format (FR-4)."""
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"pid:{pid}\nport:{port}", encoding="utf-8")


def _read_pid_file(pid_file: Path) -> tuple[int, int] | None:
    """Parse *pid_file* and return ``(pid, port)``, or ``None`` on any error.

    The file is written as ``pid:<N>\\nport:<P>`` plain text (FR-4).
    """
    if not pid_file.exists():
        return None
    try:
        text = pid_file.read_text(encoding="utf-8")
        data: dict[str, int] = {}
        for line in text.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                data[key.strip()] = int(val.strip())
        return data["pid"], data["port"]
    except (ValueError, KeyError, OSError):
        return None


def _is_running(pid_file: Path) -> bool:
    """Return True if the PID in *pid_file* refers to a live process (AC05)."""
    result = _read_pid_file(pid_file)
    if result is None:
        return False
    pid, _ = result
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _wait_until_ready(port: int, timeout: int = _READY_TIMEOUT) -> bool:
    """Poll localhost:<port> via TCP connect until it responds or *timeout* expires.

    Uses a raw TCP connect (FR-5) with 0.5 s polling interval.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _terminate_process(pid: int) -> None:
    """Terminate a process: SIGTERM on POSIX, taskkill on Windows (US-002 AC04, FR-7)."""
    if sys.platform == "win32":
        subprocess.call(["taskkill", "/F", "/PID", str(pid)])
    else:
        os.kill(pid, signal.SIGTERM)


def _open_browser(url: str) -> None:
    """Open *url* in the default browser (US-004 AC02)."""
    webbrowser.open(url)


def _write_extra_model_paths_config(config_dir: Path, models_dir: Path) -> Path:
    """Write an extra_model_paths.yaml for ComfyUI pointing at the parallax models dir.

    Returns the path to the written YAML file.
    """
    yaml_path = config_dir / "extra_model_paths.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "parallax:\n"
        f"  base_path: {models_dir}/\n"
        "  checkpoints: checkpoints/\n"
        "  diffusion_models: |\n"
        "    unet/\n"
        "    diffusion_models/\n"
        "  text_encoders: |\n"
        "    text_encoders/\n"
        "    clip/\n"
        "  vae: vae/\n"
        "  embeddings: embeddings/\n"
        "  loras: loras/\n"
        "  upscale_models: |\n"
        "    upscale_models/\n"
        "    upscale/\n"
        "  audio_encoders: audio_encoders/\n"
        "  llm: llm/\n"
        "  clip_vision: clip_vision/\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


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
    models_dir: Annotated[
        str | None,
        typer.Option("--models-dir", help="Models directory (overrides PYCOMFY_MODELS_DIR)."),
    ] = None,
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
        existing = _read_pid_file(pid_file)
        pid_str = str(existing[0]) if existing else "unknown"
        typer.echo(
            f"Warning: ComfyUI is already running (PID {pid_str}). "
            "Stop it first before starting a new instance."
        )
        return

    # Resolve ComfyUI main.py and its parent directory (FR-3 cwd)
    try:
        comfyui_main = _get_comfyui_main(python)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    comfyui_root = comfyui_main.parent

    # FR-8 — redirect stdout/stderr to the log file in append mode
    log_path = _log_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fd = log_path.open("a", encoding="utf-8")

    # Build base command
    cmd = [str(python), str(comfyui_main), "--port", str(port), "--listen", "0.0.0.0"]

    # Point ComfyUI at the models directory via extra_model_paths config.
    # Resolution order: --models-dir flag → PYCOMFY_MODELS_DIR env var.
    resolved_models = models_dir or os.environ.get("PYCOMFY_MODELS_DIR")
    if resolved_models and Path(resolved_models).is_dir():
        extra_paths_yaml = _write_extra_model_paths_config(_config_dir(), Path(resolved_models))
        cmd += ["--extra-model-paths-config", str(extra_paths_yaml)]
    elif resolved_models:
        typer.echo(
            f"Warning: models directory '{resolved_models}' does not exist — "
            "ComfyUI will use its default model paths.",
            err=True,
        )

    # AC01 + AC02 — launch ComfyUI as a detached background process
    proc = subprocess.Popen(
        cmd,
        cwd=comfyui_root,  # FR-3
        start_new_session=True,
        stdout=log_fd,  # FR-8
        stderr=log_fd,  # FR-8
    )

    # Close our copy of the log fd — the child process holds its own reference
    log_fd.close()

    # AC03 — write PID file (FR-4 plain-text format)
    _write_pid_file(pid_file, proc.pid, port)

    # AC04 — wait for the server to be ready, then print the URL
    typer.echo(f"Starting ComfyUI on port {port}…")
    ready = _wait_until_ready(port, timeout=timeout)
    url = f"http://localhost:{port}"
    if ready:
        typer.echo(f"ComfyUI is running at {url}")
        typer.echo(f"Logs: {log_path}")  # FR-8
        # US-004 AC01 — open browser if requested and server is ready
        if open_browser:
            _open_browser(url)
    else:
        typer.echo(
            f"Warning: ComfyUI did not respond within {timeout}s. "
            f"It may still be loading — check {url}"
        )
        typer.echo(f"Logs: {log_path}")  # FR-8


@app.command("stop")
def stop() -> None:
    """Stop the running ComfyUI process."""
    pid_file = _pid_file()

    # US-002 AC03 — no instance running
    if not _is_running(pid_file):
        typer.echo("No ComfyUI instance is currently running.")
        return

    existing = _read_pid_file(pid_file)
    pid = existing[0] if existing else 0

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


@app.command("status")
def status() -> None:
    """Show whether ComfyUI is currently running and on which port."""
    pid_file = _pid_file()

    # US-005 AC02 — not running
    if not _is_running(pid_file):
        typer.echo("ComfyUI is not running")
        return

    # US-005 AC01 + AC03 — running: read pid and port from PID file metadata
    existing = _read_pid_file(pid_file)
    if existing is None:
        typer.echo("ComfyUI is not running")
        return
    pid, port = existing
    typer.echo(f"ComfyUI is running (PID {pid}, port {port})")
