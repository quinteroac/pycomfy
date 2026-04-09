from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

from server.jobs import JobData
from server.job_queue import close_queue, get_queue

_WORKER_PATH = Path(__file__).parent / "worker.py"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_uv() -> str:
    found = shutil.which("uv")
    if found is None:
        raise RuntimeError(
            "uv not found on PATH. Install uv or add it to PATH before starting the server."
        )
    return found


def submit_job(data: JobData) -> str:
    """Enqueue *data*, spawn server/worker.py as a detached subprocess, return job ID."""

    async def _enqueue() -> str:
        queue = await get_queue()
        try:
            return await queue.enqueue(data)
        finally:
            await close_queue()

    job_id = asyncio.run(_enqueue())

    subprocess.Popen(
        [_resolve_uv(), "run", "python", str(_WORKER_PATH), job_id],
        cwd=str(_REPO_ROOT),
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return job_id
