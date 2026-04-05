"""Async job submission helper for the parallax CLI.

Provides ``run_async()`` which builds a ``JobData`` payload, calls
``submit_job()`` from ``server/submit.py``, and prints the standard
queued message:

    Job <job_id> queued
      → parallax jobs watch <job_id>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import typer

_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _uv_path() -> str:
    found = shutil.which("uv")
    return found if found else sys.executable


def _call_submit_job(data: object) -> str:
    """Thin wrapper around submit_job — isolated here so tests can mock it."""
    from server.submit import submit_job
    return submit_job(data)  # type: ignore[arg-type]


def run_async(
    *,
    action: str,
    media: str,
    model: str,
    args: dict[str, Any],
) -> None:
    """Submit *args* as an async job and print the queued message, then exit 0."""
    from server.jobs import JobData

    script = f"comfy_diffusion/pipelines/{media}/{model}/run.py"
    data = JobData(
        action=action,
        media=media,
        model=model,
        script=script,
        args={k: v for k, v in args.items() if v is not None},
        script_base=_REPO_ROOT,
        uv_path=_uv_path(),
    )
    job_id = _call_submit_job(data)
    typer.echo(f"Job {job_id} queued")
    typer.echo(f"  \u2192 parallax jobs watch {job_id}")
