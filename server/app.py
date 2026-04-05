"""FastAPI application — inference job submission endpoints."""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from server.jobs import JobData, PythonProgress
from server.queue import get_queue
from server.schemas import (
    CreateAudioRequest,
    CreateImageRequest,
    CreateVideoRequest,
    EditImageRequest,
    JobResponse,
    JobResult,
    JobStatusResponse,
    UpscaleImageRequest,
)
from server.submit import submit_job

app = FastAPI(title="comfy-diffusion server")

_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _uv_path() -> str:
    found = shutil.which("uv")
    return found if found else sys.executable


def _make_args(req_dict: dict) -> dict:
    """Return a flat args dict from the request, dropping None values and 'model'."""
    return {k: v for k, v in req_dict.items() if v is not None and k != "model"}


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    async def _get() -> dict | None:
        queue = await get_queue()
        return await queue.get(job_id)

    row = asyncio.run(_get())
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")

    status = row["status"]
    if status == "pending":
        status = "queued"

    result: JobResult | None = None
    if row.get("result") is not None:
        result = JobResult(**json.loads(row["result"]))

    return JobStatusResponse(
        id=row["id"],
        status=status,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        result=result,
    )


@app.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str) -> StreamingResponse:
    queue = await get_queue()
    row = await queue.get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        last_progress_json: str | None = None
        while True:
            current = await queue.get(job_id)
            if current is None:
                break

            progress_json = current.get("progress")
            status = current["status"]

            if progress_json and progress_json != last_progress_json:
                last_progress_json = progress_json
                yield f"data: {progress_json}\n\n"

            if status == "completed":
                final = PythonProgress(step="done", pct=1.0)
                yield f"data: {final.model_dump_json()}\n\n"
                break
            elif status == "failed":
                error_msg: str | None = None
                if current.get("result"):
                    result_data = json.loads(current["result"])
                    error_msg = result_data.get("error")
                final = PythonProgress(step="error", pct=0.0, error=error_msg)
                yield f"data: {final.model_dump_json()}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/jobs/create/image", response_model=JobResponse)
def create_image(req: CreateImageRequest) -> JobResponse:
    data = JobData(
        action="create",
        media="image",
        model=req.model,
        script=f"comfy_diffusion/pipelines/image/{req.model}/run.py",
        args=_make_args(req.model_dump()),
        script_base=_REPO_ROOT,
        uv_path=_uv_path(),
    )
    job_id = submit_job(data)
    return JobResponse(job_id=job_id, status="queued")


@app.post("/jobs/create/video", response_model=JobResponse)
def create_video(req: CreateVideoRequest) -> JobResponse:
    data = JobData(
        action="create",
        media="video",
        model=req.model,
        script=f"comfy_diffusion/pipelines/video/{req.model}/run.py",
        args=_make_args(req.model_dump()),
        script_base=_REPO_ROOT,
        uv_path=_uv_path(),
    )
    job_id = submit_job(data)
    return JobResponse(job_id=job_id, status="queued")


@app.post("/jobs/create/audio", response_model=JobResponse)
def create_audio(req: CreateAudioRequest) -> JobResponse:
    data = JobData(
        action="create",
        media="audio",
        model=req.model,
        script=f"comfy_diffusion/pipelines/audio/{req.model}/run.py",
        args=_make_args(req.model_dump()),
        script_base=_REPO_ROOT,
        uv_path=_uv_path(),
    )
    job_id = submit_job(data)
    return JobResponse(job_id=job_id, status="queued")


@app.post("/jobs/edit/image", response_model=JobResponse)
def edit_image(req: EditImageRequest) -> JobResponse:
    data = JobData(
        action="edit",
        media="image",
        model=req.model,
        script=f"comfy_diffusion/pipelines/image/{req.model}/run.py",
        args=_make_args(req.model_dump()),
        script_base=_REPO_ROOT,
        uv_path=_uv_path(),
    )
    job_id = submit_job(data)
    return JobResponse(job_id=job_id, status="queued")


@app.post("/jobs/upscale/image", response_model=JobResponse)
def upscale_image(req: UpscaleImageRequest) -> JobResponse:
    data = JobData(
        action="upscale",
        media="image",
        model=req.model,
        script=f"comfy_diffusion/pipelines/image/{req.model}/run.py",
        args=_make_args(req.model_dump()),
        script_base=_REPO_ROOT,
        uv_path=_uv_path(),
    )
    job_id = submit_job(data)
    return JobResponse(job_id=job_id, status="queued")
