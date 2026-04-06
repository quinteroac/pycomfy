"""Backward-compatible re-export shim.

All application logic lives in ``server.gateway`` (APIRouter) and
``server.main`` (FastAPI app + middleware).  This module re-exports the
``app`` object so existing imports and test patches continue to work.
"""
from __future__ import annotations

# Re-export the assembled FastAPI application.
from server.main import app  # noqa: F401

# Re-export symbols that tests patch via "server.app.<name>" so that
# monkeypatching these names still takes effect in gateway handlers.
# Tests should migrate to patching "server.gateway.<name>" directly.
from server.gateway import (  # noqa: F401
    _pkg_version,
    _uv_path,
    get_queue,
    submit_job,
)
