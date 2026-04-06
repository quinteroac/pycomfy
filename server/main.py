"""FastAPI application entry point — attaches CORS middleware and mounts gateway routes."""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
