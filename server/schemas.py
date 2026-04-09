"""Pydantic request/response schemas for the inference job REST API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class JobResult(BaseModel):
    output_path: Optional[str] = None
    error: Optional[str] = None


class CreateImageRequest(BaseModel):
    model: str
    prompt: str
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    input: Optional[str] = None


class CreateVideoRequest(BaseModel):
    model: str
    prompt: str
    input: Optional[str] = None
    pipeline: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    length: Optional[int] = None
    duration: Optional[float] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None


class CreateAudioRequest(BaseModel):
    model: str
    prompt: str
    lyrics: Optional[str] = None
    bpm: Optional[int] = None
    length: Optional[float] = None
    duration: Optional[float] = None
    seed: Optional[int] = None
    steps: Optional[int] = None


class EditImageRequest(BaseModel):
    model: str
    prompt: str
    input: str


class UpscaleImageRequest(BaseModel):
    model: str
    prompt: str
    input: str


class JobStatusResponse(BaseModel):
    id: str
    status: str
    created_at: str
    updated_at: str
    result: Optional[JobResult] = None


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobListItem(BaseModel):
    id: str
    status: str
    created_at: str
    updated_at: str
