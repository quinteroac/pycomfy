from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from server.jobs import JobData
from server.queue import get_queue

_WORKER_PATH = Path(__file__).parent / "worker.py"


def submit_job(data: JobData) -> str:
    """Enqueue *data*, spawn server/worker.py as a detached subprocess, return job ID."""

    async def _enqueue() -> str:
        queue = await get_queue()
        return await queue.enqueue(data)

    job_id = asyncio.run(_enqueue())

    subprocess.Popen(
        [sys.executable, str(_WORKER_PATH), job_id],
        start_new_session=True,
    )

    return job_id
