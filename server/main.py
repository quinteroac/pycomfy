"""FastAPI application entry point — attaches CORS middleware and mounts gateway routes."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.gateway import router

app = FastAPI(title="comfy-diffusion server")

_cors_origins = [o.strip() for o in os.environ.get("PARALLAX_CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(router)

_FRONTEND_DIST = Path(os.environ.get("PARALLAX_FRONTEND_DIR", "")).expanduser() or (
    Path(__file__).parent.parent / "frontend" / "dist"
)


def _mount_frontend_ui(target_app: FastAPI, dist_dir: Path) -> None:
    """Mount the compiled frontend SPA at /ui."""

    @target_app.get("/ui", include_in_schema=False)
    async def _ui_root() -> FileResponse:
        return FileResponse(dist_dir / "index.html")

    target_app.mount("/ui", StaticFiles(directory=dist_dir, html=True), name="frontend")


if _FRONTEND_DIST.is_dir():
    _mount_frontend_ui(app, _FRONTEND_DIST)
