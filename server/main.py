"""FastAPI application entry point — attaches CORS middleware and mounts gateway routes."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.gateway import router

logger = logging.getLogger(__name__)

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


def _resolve_frontend_dist() -> Path:
    """Return the frontend dist path from env or repo-relative default."""
    env_val = os.environ.get("PARALLAX_FRONTEND_PATH", "")
    if env_val:
        return Path(env_val).expanduser()
    return Path(__file__).parent.parent / "frontend" / "dist"


def _mount_frontend_ui(target_app: FastAPI, dist_dir: Path) -> None:
    """Mount the compiled frontend SPA at /ui."""

    @target_app.get("/ui", include_in_schema=False)
    async def _ui_root() -> FileResponse:
        return FileResponse(dist_dir / "index.html")

    target_app.mount("/ui", StaticFiles(directory=dist_dir, html=True), name="frontend")


_FRONTEND_DIST = _resolve_frontend_dist()

if _FRONTEND_DIST.is_dir():
    _mount_frontend_ui(app, _FRONTEND_DIST)
else:
    logger.warning("Frontend not found at %s — /ui will not be served.", _FRONTEND_DIST)
