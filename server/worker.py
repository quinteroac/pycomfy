"""Worker: reads a queued job by ID, spawns the pipeline subprocess, streams NDJSON progress."""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from server.jobs import JobData, PythonProgress
from server.queue import get_queue

_PROCESSABLE_STATUSES = {"pending", "queued"}


async def _run_worker(job_id: str) -> None:
    queue = await get_queue()

    job = await queue.get(job_id)
    if job is None:
        print(f"Job {job_id} not found", file=sys.stderr)
        sys.exit(1)

    status = job["status"]
    if status not in _PROCESSABLE_STATUSES:
        print(
            f"Job {job_id} has status {status!r}, expected one of {sorted(_PROCESSABLE_STATUSES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    data = JobData.model_validate_json(job["data"])
    await queue.update_status(job_id, "running")

    script_path = Path(data.script_base) / data.script
    cmd = [data.uv_path, "run", "python", str(script_path)]
    for key, value in data.args.items():
        cmd.extend([f"--{key}", str(value)])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    last_output: str | None = None

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        print(line, flush=True)
        try:
            progress = PythonProgress.model_validate_json(line)
            await queue.update_progress(job_id, progress)
            if progress.output is not None:
                last_output = progress.output
        except Exception:
            pass  # not valid PythonProgress JSON — stream as plain text

    assert proc.stderr is not None
    stderr_content = proc.stderr.read()
    exit_code = proc.wait()

    if exit_code == 0:
        result: dict = {}
        if last_output is not None:
            result["output_path"] = last_output
        await queue.update_status(job_id, "completed", result)
    else:
        await queue.update_status(job_id, "failed", {"error": stderr_content})


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: worker.py <job_id>", file=sys.stderr)
        sys.exit(1)
    job_id = sys.argv[1]
    asyncio.run(_run_worker(job_id))


if __name__ == "__main__":
    main()
