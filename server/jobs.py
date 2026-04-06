from __future__ import annotations

from pydantic import BaseModel

from server.schemas import JobResult as JobResult  # noqa: F401 — re-exported for consumers


class JobData(BaseModel):
    action: str
    media: str
    model: str
    cmd: list[str]


class PythonProgress(BaseModel):
    step: str
    pct: float
    frame: int | None = None
    total: int | None = None
    output: str | None = None
    error: str | None = None
