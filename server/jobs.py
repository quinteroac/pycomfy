from __future__ import annotations

from pydantic import BaseModel


class JobData(BaseModel):
    action: str
    media: str
    model: str
    script: str
    args: dict
    script_base: str
    uv_path: str


class JobResult(BaseModel):
    output_path: str


class PythonProgress(BaseModel):
    step: str
    pct: float
    frame: int | None = None
    total: int | None = None
    output: str | None = None
    error: str | None = None
