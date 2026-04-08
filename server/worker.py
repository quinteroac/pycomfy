"""Worker: reads a queued job by ID, spawns the pipeline subprocess, streams NDJSON progress."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

_LOG_FILE = Path("/tmp/comfy_diffusion_worker.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE),
        logging.StreamHandler(sys.stderr),
    ],
)
_log = logging.getLogger(__name__)

from server.jobs import JobData, PythonProgress
from server.job_queue import close_queue, get_queue

_PROCESSABLE_STATUSES = {"queued"}


async def _run_worker(job_id: str) -> None:
    queue = await get_queue()
    try:
        _log.info("Starting job %s", job_id)
        job = await queue.get(job_id)
        if job is None:
            _log.error("Job %s not found", job_id)
            sys.exit(1)

        status = job["status"]
        if status not in _PROCESSABLE_STATUSES:
            _log.error(
                "Job %s has status %r, expected one of %s",
                job_id,
                status,
                sorted(_PROCESSABLE_STATUSES),
            )
            sys.exit(1)

        data = JobData.model_validate_json(job["data"])
        await queue.update_status(job_id, "running")

        _log.debug("Spawning subprocess: %s", data.cmd)

        proc = subprocess.Popen(
            data.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        last_output: str | None = None

        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            _log.debug("[stdout] %s", line)
            print(line, flush=True)
            try:
                progress = PythonProgress.model_validate_json(line)
                await queue.update_progress(job_id, progress)
                if progress.output is not None:
                    last_output = progress.output
            except Exception:
                # Not valid PythonProgress JSON — treat as plain output path if it looks like a file
                if line.startswith("/") and Path(line).exists():
                    last_output = line

        assert proc.stderr is not None
        stderr_content = proc.stderr.read()
        exit_code = proc.wait()

        if exit_code == 0:
            result: dict = {}
            if last_output is not None:
                result["output_path"] = last_output
            _log.info("Job %s completed, output: %s", job_id, last_output)
            await queue.update_status(job_id, "completed", result)
        else:
            _log.error("Job %s failed (exit %d):\n%s", job_id, exit_code, stderr_content)
            await queue.update_status(job_id, "failed", {"error": stderr_content})
    finally:
        await close_queue()


def main() -> None:
    if len(sys.argv) < 2:
        _log.error("Usage: worker.py <job_id>")
        sys.exit(1)
    job_id = sys.argv[1]
    asyncio.run(_run_worker(job_id))


if __name__ == "__main__":
    main()
