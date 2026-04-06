"""Async job submission helper for the parallax CLI.

Provides ``enqueue_cmd()`` which stores a command list into the job queue
and prints the standard queued message:

    Job <job_id> queued
      → parallax jobs watch <job_id>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer

_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _uv_path() -> str:
    import os

    env_path = os.environ.get("PARALLAX_UV_PATH")
    if env_path:
        return env_path
    found = shutil.which("uv")
    return found if found else sys.executable


def _call_submit_job(data: object) -> str:
    """Thin wrapper around submit_job — isolated here so tests can mock it."""
    from server.submit import submit_job
    return submit_job(data)  # type: ignore[arg-type]


def enqueue_cmd(cmd: list[str]) -> None:
    """Queue *cmd* as an async job and print the queued message.

    *cmd* is the full command to execute, e.g.::

        ["uv", "run", "parallax", "create", "image", "--model", "anima", ...]
    """
    from server.jobs import JobData

    # Infer action/media/model from the cmd for metadata only.
    # Expects: [uv, run, parallax, <action>, <media>, --model, <model>, ...]
    action = cmd[3] if len(cmd) > 3 else "unknown"
    media = cmd[4] if len(cmd) > 4 else "unknown"
    model = "unknown"
    try:
        model = cmd[cmd.index("--model") + 1]
    except (ValueError, IndexError):
        pass

    data = JobData(
        action=action,
        media=media,
        model=model,
        cmd=cmd,
    )
    job_id = _call_submit_job(data)
    typer.echo(f"Job {job_id} queued")
    typer.echo(f"  → parallax jobs watch {job_id}")


# ---------------------------------------------------------------------------
# Legacy helper — kept for backward compat with --async call sites.
# ---------------------------------------------------------------------------

def run_async(*, action: str, media: str, model: str, args: dict) -> None:
    """Build the CLI command from parts and enqueue it."""
    uv = _uv_path()
    cmd: list[str] = [uv, "run", "parallax", action, media, "--model", model]
    for k, v in args.items():
        if v is not None:
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    enqueue_cmd(cmd)
