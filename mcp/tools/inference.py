"""MCP tool handlers for non-blocking inference job submission (US-001).

Each tool enqueues a job and returns immediately with a job ID so that
AI agents never hit tool-call timeouts on long-running inference tasks.
Use ``get_job_status`` to poll or ``wait_for_job`` to block until done.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from server.jobs import JobData
from server.job_queue import get_queue

_WORKER_PATH = Path(__file__).resolve().parents[2] / "server" / "worker.py"


def _uv_path() -> str:
    found = shutil.which("uv")
    if found is None:
        raise RuntimeError(
            "uv not found on PATH. Install uv or add it to PATH before starting the server."
        )
    return found


def _spawn_worker(job_id: str) -> None:
    subprocess.Popen(
        [_uv_path(), "run", "python", str(_WORKER_PATH), job_id],
        start_new_session=True,
    )


def _build_cmd(action: str, media: str, model: str, **kwargs: object) -> list[str]:
    uv = _uv_path()
    cmd = [uv, "run", "parallax", action, media, "--model", model]
    for k, v in kwargs.items():
        if v is not None:
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    return cmd


async def create_image(
    model: str,
    prompt: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    input: Optional[str] = None,
) -> str:
    """Submit an image generation job and return a job ID immediately.

    Enqueues inference using the given model and prompt, then returns within
    200ms with a job ID. Use ``wait_for_job(job_id)`` to block until the
    output image path is available.
    """
    data = JobData(
        action="create",
        media="image",
        model=model,
        cmd=_build_cmd(
            "create", "image", model,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            negative_prompt=negative_prompt,
            input=input,
        ),
    )
    queue = await get_queue()
    job_id = await queue.enqueue(data)
    _spawn_worker(job_id)
    return f"job_id: {job_id}\nstatus: queued\nmodel: {model}"


async def create_video(
    model: str,
    prompt: str,
    input: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    length: Optional[int] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
) -> str:
    """Submit a video generation job and return a job ID immediately.

    Enqueues inference using the given model and prompt, then returns within
    200ms with a job ID. Use ``wait_for_job(job_id)`` to block until the
    output video path is available.
    """
    data = JobData(
        action="create",
        media="video",
        model=model,
        cmd=_build_cmd(
            "create", "video", model,
            prompt=prompt,
            input=input,
            width=width,
            height=height,
            length=length,
            seed=seed,
            steps=steps,
            cfg=cfg,
        ),
    )
    queue = await get_queue()
    job_id = await queue.enqueue(data)
    _spawn_worker(job_id)
    return f"job_id: {job_id}\nstatus: queued\nmodel: {model}"


async def create_audio(
    model: str,
    prompt: str,
    lyrics: Optional[str] = None,
    bpm: Optional[int] = None,
    length: Optional[float] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
) -> str:
    """Submit an audio generation job and return a job ID immediately.

    Enqueues inference using the given model and prompt, then returns within
    200ms with a job ID. Use ``wait_for_job(job_id)`` to block until the
    output audio path is available.
    """
    data = JobData(
        action="create",
        media="audio",
        model=model,
        cmd=_build_cmd(
            "create", "audio", model,
            prompt=prompt,
            lyrics=lyrics,
            bpm=bpm,
            length=length,
            seed=seed,
            steps=steps,
        ),
    )
    queue = await get_queue()
    job_id = await queue.enqueue(data)
    _spawn_worker(job_id)
    return f"job_id: {job_id}\nstatus: queued\nmodel: {model}"


async def edit_image(
    model: str,
    prompt: str,
    input: str,
) -> str:
    """Submit an image editing job and return a job ID immediately.

    Enqueues inference using the given model, prompt, and input image path,
    then returns within 200ms with a job ID. Use ``wait_for_job(job_id)`` to
    block until the output image path is available.
    """
    data = JobData(
        action="edit",
        media="image",
        model=model,
        cmd=_build_cmd("edit", "image", model, prompt=prompt, input=input),
    )
    queue = await get_queue()
    job_id = await queue.enqueue(data)
    _spawn_worker(job_id)
    return f"job_id: {job_id}\nstatus: queued\nmodel: {model}"


async def upscale_image(
    model: str,
    prompt: str,
    input: str,
) -> str:
    """Submit an image upscaling job and return a job ID immediately.

    Enqueues inference using the given model, prompt, and input image path,
    then returns within 200ms with a job ID. Use ``wait_for_job(job_id)`` to
    block until the output image path is available.
    """
    data = JobData(
        action="upscale",
        media="image",
        model=model,
        cmd=_build_cmd("upscale", "image", model, prompt=prompt, input=input),
    )
    queue = await get_queue()
    job_id = await queue.enqueue(data)
    _spawn_worker(job_id)
    return f"job_id: {job_id}\nstatus: queued\nmodel: {model}"
