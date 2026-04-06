"""MCP tool handlers for job status and polling (US-002, US-003)."""

from __future__ import annotations

import asyncio
import json

from server.job_queue import get_queue


async def get_job_status(job_id: str) -> str:
    """Return the current status and metadata of a job by its ID.

    Checks the job queue and returns fields: status, model, created_at, and
    either output_path (when completed) or error (when failed). Returns
    ``status: not_found`` when the job ID does not exist — never raises.
    """
    queue = await get_queue()
    row = await queue.get(job_id)

    if row is None:
        return "status: not_found"

    data_dict = json.loads(row["data"]) if row.get("data") else {}
    model = data_dict.get("model", "unknown")
    status = row["status"]
    created_at = row.get("created_at", "")

    lines = [
        f"status: {status}",
        f"model: {model}",
        f"created_at: {created_at}",
    ]

    result_raw = row.get("result")
    if result_raw:
        result = json.loads(result_raw)
        if result.get("output_path"):
            lines.append(f"output_path: {result['output_path']}")
        if result.get("error"):
            lines.append(f"error: {result['error']}")

    return "\n".join(lines)


async def wait_for_job(job_id: str, timeout_seconds: int = 600) -> str:
    """Block until a job completes or times out, then return the output path.

    Polls the job queue every 2 seconds until the job status is ``completed``
    or ``failed``. On completion returns ``output: <path>``; on failure
    returns ``error: <message>``; on timeout returns
    ``error: timeout after <N>s`` without raising.
    """
    elapsed = 0
    while elapsed < timeout_seconds:
        queue = await get_queue()
        row = await queue.get(job_id)

        if row is None:
            return "error: job not found"

        status = row["status"]

        if status == "completed":
            result_raw = row.get("result")
            output_path = ""
            if result_raw:
                result = json.loads(result_raw)
                output_path = result.get("output_path", "")
            return f"output: {output_path}"

        if status == "failed":
            result_raw = row.get("result")
            error_msg = "unknown error"
            if result_raw:
                result = json.loads(result_raw)
                error_msg = result.get("error", "unknown error")
            return f"error: {error_msg}"

        await asyncio.sleep(2)
        elapsed += 2

    return f"error: timeout after {timeout_seconds}s"
