"""APIRouter — inference job submission endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import shutil
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

import tempfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from server.jobs import JobData, PythonProgress
from server.job_queue import get_queue
from server.schemas import (
    CreateAudioRequest,
    CreateImageRequest,
    CreateVideoRequest,
    EditImageRequest,
    JobListItem,
    JobResponse,
    JobResult,
    JobStatusResponse,
    UpscaleImageRequest,
)
from server.submit import submit_job

router = APIRouter()

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_SSE_POLL_S = float(os.environ.get("PARALLAX_SSE_POLL_MS", "500")) / 1000.0

_VIDEO_FPS: dict[str, int] = {
    "ltx2": 24,
    "ltx23": 25,
    "wan21": 16,
    "wan22": 16,
}

# Models whose frame count must satisfy (length - 1) % 8 == 0
_LTX_MODELS = {"ltx2", "ltx23"}


def _duration_to_length(duration_s: float, model: str) -> int:
    """Convert duration in seconds to frame count, respecting model constraints."""
    fps = _VIDEO_FPS.get(model, 24)
    raw = duration_s * fps
    if model in _LTX_MODELS:
        # Round to nearest valid length: 8k + 1
        k = max(0, round((raw - 1) / 8))
        return 8 * k + 1
    return max(1, round(raw))


def _uv_path() -> str:
    found = shutil.which("uv")
    if found is None:
        raise RuntimeError(
            "uv not found on PATH. Install uv or add it to PATH before starting the server."
        )
    return found


def _build_cmd(action: str, media: str, model: str, req_dict: dict) -> list[str]:
    """Build the full CLI command to queue as a subprocess job."""
    uv = _uv_path()
    cmd = [uv, "run", "parallax", action, media, "--model", model]
    for key, value in req_dict.items():
        if value is not None and key != "model":
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    logger.info("Built command: %s", cmd)
    return cmd


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": _pkg_version("comfy-diffusion")}


@router.get("/jobs", response_model=list[JobListItem])
def list_jobs() -> list[JobListItem]:
    async def _list() -> list[dict]:
        queue = await get_queue()
        return await queue.list_jobs(limit=50)

    rows = asyncio.run(_list())
    return [
        JobListItem(
            id=r["id"],
            status=r["status"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> dict:
    async def _get() -> dict | None:
        queue = await get_queue()
        return await queue.get(job_id)

    row = asyncio.run(_get())
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")

    if row["status"] != "queued":
        raise HTTPException(status_code=409, detail="job is already running or completed")

    async def _cancel() -> None:
        queue = await get_queue()
        await queue.update_status(job_id, "cancelled")

    asyncio.run(_cancel())
    return {"cancelled": True}


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    async def _get() -> dict | None:
        queue = await get_queue()
        return await queue.get(job_id)

    row = asyncio.run(_get())
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")

    result: JobResult | None = None
    if row.get("result") is not None:
        result = JobResult(**json.loads(row["result"]))

    return JobStatusResponse(
        id=row["id"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        result=result,
    )


@router.get("/jobs/{job_id}/stream")
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

            await asyncio.sleep(_SSE_POLL_S)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/jobs/{job_id}/result")
def get_job_result_file(job_id: str) -> FileResponse:
    async def _get() -> dict | None:
        queue = await get_queue()
        return await queue.get(job_id)

    row = asyncio.run(_get())
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    if row["status"] != "completed":
        raise HTTPException(status_code=409, detail="job not yet completed")

    result: dict | None = None
    if row.get("result"):
        result = json.loads(row["result"])

    if not result or not result.get("output_path"):
        raise HTTPException(status_code=404, detail="no output file recorded")

    output_path = result["output_path"]
    if not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="output file not found on disk")

    media_type, _ = mimetypes.guess_type(output_path)
    return FileResponse(
        path=output_path,
        media_type=media_type or "application/octet-stream",
        filename=Path(output_path).name,
        headers={"Content-Disposition": f'attachment; filename="{Path(output_path).name}"'},
    )


@router.post("/jobs/create/image", response_model=JobResponse)
def create_image(req: CreateImageRequest) -> JobResponse:
    data = JobData(
        action="create",
        media="image",
        model=req.model,
        cmd=_build_cmd("create", "image", req.model, req.model_dump()),
    )
    job_id = submit_job(data)
    return JobResponse(job_id=job_id, status="queued")


@router.post("/jobs/create/video", response_model=JobResponse)
async def create_video(request: Request) -> JobResponse:
    content_type = request.headers.get("content-type", "")
    input_path: str | None = None

    if "multipart/form-data" in content_type:
        form = await request.form()

        def _str(key: str) -> str | None:
            v = form.get(key)
            return str(v) if v is not None else None

        def _int(key: str) -> int | None:
            v = form.get(key)
            return int(v) if v is not None else None  # type: ignore[arg-type]

        def _float(key: str) -> float | None:
            v = form.get(key)
            return float(v) if v is not None else None  # type: ignore[arg-type]

        image_field = form.get("image")
        if image_field is not None and hasattr(image_field, "read"):
            filename = getattr(image_field, "filename", None) or "upload"
            suffix = Path(filename).suffix or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(await image_field.read())  # type: ignore[union-attr]
            tmp.close()
            input_path = tmp.name

        req = CreateVideoRequest(
            model=_str("model") or "",
            prompt=_str("prompt") or "",
            pipeline=_str("pipeline"),  # type: ignore[call-arg]
            width=_int("width"),
            height=_int("height"),
            duration=_float("duration"),
            seed=_int("seed"),
            input=input_path,
        )
    else:
        body = await request.json()
        req = CreateVideoRequest(**body)

    req_dict = req.model_dump()
    if req_dict.get("duration") is not None and req_dict.get("length") is None:
        req_dict["length"] = _duration_to_length(req_dict["duration"], req.model)
    req_dict.pop("duration", None)
    req_dict.pop("pipeline", None)
    data = JobData(
        action="create",
        media="video",
        model=req.model,
        cmd=_build_cmd("create", "video", req.model, req_dict),
    )
    job_id = await asyncio.to_thread(submit_job, data)
    return JobResponse(job_id=job_id, status="queued")


@router.post("/jobs/create/audio", response_model=JobResponse)
def create_audio(req: CreateAudioRequest) -> JobResponse:
    req_dict = req.model_dump()
    if req_dict.get("duration") is not None and req_dict.get("length") is None:
        req_dict["length"] = req_dict["duration"]
    req_dict.pop("duration", None)
    data = JobData(
        action="create",
        media="audio",
        model=req.model,
        cmd=_build_cmd("create", "audio", req.model, req_dict),
    )
    job_id = submit_job(data)
    return JobResponse(job_id=job_id, status="queued")


@router.post("/jobs/edit/image", response_model=JobResponse)
def edit_image(req: EditImageRequest) -> JobResponse:
    data = JobData(
        action="edit",
        media="image",
        model=req.model,
        cmd=_build_cmd("edit", "image", req.model, req.model_dump()),
    )
    job_id = submit_job(data)
    return JobResponse(job_id=job_id, status="queued")


@router.post("/jobs/upscale/image", response_model=JobResponse)
def upscale_image(req: UpscaleImageRequest) -> JobResponse:
    data = JobData(
        action="upscale",
        media="image",
        model=req.model,
        cmd=_build_cmd("upscale", "image", req.model, req.model_dump()),
    )
    job_id = submit_job(data)
    return JobResponse(job_id=job_id, status="queued")
