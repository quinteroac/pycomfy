"""``parallax install`` — install comfy-diffusion with torch and ComfyUI engine.

Acceptance criteria implemented:
  AC01 — detect uv; if absent, download+install via urllib.request (stdlib only)
  AC02 — create ~/.parallax/env via uv venv; install comfy-diffusion[cuda|cpu]
  AC03 — call check_runtime() in the venv; error dict → print + exit 1
  AC04 — on success, print version and next step
  AC05 — on failure, print step name + stderr + suggest --verbose
  AC06 — already installed → print message + exit 0 (no --upgrade)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer

from cli.commands._common import ENV_DIR as _ENV_DIR

_PACKAGE_CUDA = "comfy-diffusion[cuda]"
_PACKAGE_CPU = "comfy-diffusion[cpu]"

# UV installer URL (official astral.sh installer — stdlib urllib.request only)
_UV_INSTALLER_URL = "https://astral.sh/uv/install.sh"


# ---------------------------------------------------------------------------
# Internal helpers (extracted for testability)
# ---------------------------------------------------------------------------


def _find_uv() -> str | None:
    """Return the absolute path to the ``uv`` binary, or *None* if not found."""
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (
        Path.home() / ".cargo" / "bin" / "uv",
        Path.home() / ".local" / "bin" / "uv",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _download_and_install_uv() -> str:
    """Download the official uv installer and execute it (stdlib-only)."""
    import tempfile
    import urllib.request

    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with urllib.request.urlopen(_UV_INSTALLER_URL) as resp:  # noqa: S310
            Path(tmp_path).write_bytes(resp.read())
        os.chmod(tmp_path, 0o755)
        subprocess.run(["sh", tmp_path], check=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    uv = _find_uv()
    if uv is None:
        raise RuntimeError(
            "uv installer finished but the binary was not found. "
            "Try adding ~/.local/bin or ~/.cargo/bin to your PATH."
        )
    return uv


def _ensure_uv(verbose: bool = False) -> str:
    """Return path to ``uv``, downloading and installing it first if needed."""
    uv = _find_uv()
    if uv is not None:
        return uv
    typer.echo("uv not found — downloading and installing uv via urllib.request…")
    uv = _download_and_install_uv()
    if verbose:
        typer.echo(f"uv installed at {uv}")
    return uv


def _installed_version(env_dir: Path) -> str | None:
    """Return the installed ``comfy-diffusion`` version string, or *None*."""
    python = env_dir / "bin" / "python"
    if not python.exists():
        return None
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import importlib.metadata; print(importlib.metadata.version('comfy-diffusion'))",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return version if version else None
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _run_step(
    cmd: list[str],
    step_name: str,
    verbose: bool,
) -> None:
    """Execute *cmd* and raise ``typer.Exit(1)`` on failure (AC05)."""
    kwargs: dict[str, Any] = {"text": True}
    if not verbose:
        kwargs["capture_output"] = True

    if not verbose:
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description=step_name, total=None)
            result = subprocess.run(cmd, **kwargs)
    else:
        result = subprocess.run(cmd, **kwargs)

    if result.returncode != 0:
        typer.echo(f"\nError during step: {step_name}", err=True)
        if not verbose:
            stderr = getattr(result, "stderr", "") or ""
            if stderr:
                typer.echo(stderr, err=True)
        typer.echo("Re-run with --verbose for full output.", err=True)
        raise typer.Exit(1)


def _bootstrap_comfyui(env_dir: Path, verbose: bool) -> None:
    """Run ``check_runtime()`` inside the venv; exit 1 if it returns an error (AC03)."""
    python = env_dir / "bin" / "python"
    script = (
        "import comfy_diffusion, json, sys; "
        "r = comfy_diffusion.check_runtime(); "
        "err = r.get('error') if isinstance(r, dict) else None; "
        "print(json.dumps(r) if err else 'ok'); "
        "sys.exit(1) if err else sys.exit(0)"
    )
    result = subprocess.run(
        [str(python), "-c", script],
        capture_output=not verbose,
        text=True,
    )
    if result.returncode != 0:
        typer.echo("\nError during step: check_runtime()", err=True)
        if not verbose:
            output = result.stdout.strip() or result.stderr.strip()
            if output:
                typer.echo(output, err=True)
        typer.echo("Re-run with --verbose for full output.", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------


def install(
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Install CPU-only variant (no CUDA).")
    ] = False,
    upgrade: Annotated[
        bool, typer.Option("--upgrade", help="Upgrade an existing installation.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show full subprocess output.")
    ] = False,
) -> None:
    """Install comfy-diffusion, torch, and the ComfyUI engine into ~/.parallax/env."""
    env_dir = _ENV_DIR

    # AC06 — already installed guard
    if not upgrade:
        version = _installed_version(env_dir)
        if version is not None:
            typer.echo(
                f"Already installed (v{version}). "
                "Run `parallax install --upgrade` to update."
            )
            return

    # AC01 — ensure uv is available
    try:
        uv = _ensure_uv(verbose=verbose)
    except Exception as exc:
        typer.echo(f"\nError during step: ensure uv\n{exc}", err=True)
        typer.echo("Re-run with --verbose for full output.", err=True)
        raise typer.Exit(1)

    # AC02 — create dedicated virtual environment
    _run_step(
        [uv, "venv", str(env_dir)],
        step_name="uv venv",
        verbose=verbose,
    )

    # AC02 — install comfy-diffusion into the new venv
    package = _PACKAGE_CPU if cpu else _PACKAGE_CUDA
    _run_step(
        [uv, "pip", "install", package, "--python", str(env_dir / "bin" / "python")],
        step_name=f"install {package}",
        verbose=verbose,
    )

    # AC03 — bootstrap ComfyUI engine
    _bootstrap_comfyui(env_dir, verbose=verbose)

    # AC04 — success
    version = _installed_version(env_dir) or "unknown"
    typer.echo(f"Installed comfy-diffusion v{version}.")
    typer.echo("Run `parallax ms install` to set up the inference server.")
