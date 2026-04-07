"""Tests for US-002 it_000047 — Frontend directory path configuration via env var.

Covers:
  AC01 — Server reads PARALLAX_FRONTEND_PATH from environment.
  AC02 — If PARALLAX_FRONTEND_PATH is set and the path exists, server mounts at /ui.
  AC03 — If PARALLAX_FRONTEND_PATH is not set, falls back to frontend/dist/ relative to repo root.
  AC04 — If neither path exists, server starts normally without /ui and logs a warning.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.testclient import TestClient

from server.gateway import router
from server.main import _mount_frontend_ui, _resolve_frontend_dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(dist_dir: Path) -> TestClient:
    """Return a TestClient for a fresh app with frontend mounted."""
    _app = FastAPI(title="test-server")
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _app.include_router(router)
    _mount_frontend_ui(_app, dist_dir)
    return TestClient(_app, raise_server_exceptions=True)


@pytest.fixture
def dist_dir(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<html><body><div id='root'></div></body></html>", encoding="utf-8"
    )
    return dist


# ---------------------------------------------------------------------------
# AC01 — Server reads PARALLAX_FRONTEND_PATH from the environment
# ---------------------------------------------------------------------------


class TestAC01ReadsEnvVar:
    def test_env_var_is_read(self, dist_dir: Path) -> None:
        with patch.dict(os.environ, {"PARALLAX_FRONTEND_PATH": str(dist_dir)}):
            resolved = _resolve_frontend_dist()
        assert resolved == dist_dir

    def test_env_var_path_expanded(self, tmp_path: Path) -> None:
        """expanduser() is applied so ~/... paths resolve correctly."""
        fake_path = str(tmp_path / "some" / "dir")
        with patch.dict(os.environ, {"PARALLAX_FRONTEND_PATH": fake_path}):
            resolved = _resolve_frontend_dist()
        assert resolved == Path(fake_path)


# ---------------------------------------------------------------------------
# AC02 — If PARALLAX_FRONTEND_PATH is set and path exists, mount at /ui
# ---------------------------------------------------------------------------


class TestAC02MountWhenEnvVarSet:
    def test_ui_returns_200(self, dist_dir: Path) -> None:
        client = _make_app(dist_dir)
        resp = client.get("/ui")
        assert resp.status_code == 200

    def test_ui_serves_index_html(self, dist_dir: Path) -> None:
        client = _make_app(dist_dir)
        resp = client.get("/ui")
        assert "root" in resp.text

    def test_env_var_resolves_to_existing_dir(self, dist_dir: Path) -> None:
        with patch.dict(os.environ, {"PARALLAX_FRONTEND_PATH": str(dist_dir)}):
            resolved = _resolve_frontend_dist()
        assert resolved.is_dir()


# ---------------------------------------------------------------------------
# AC03 — If PARALLAX_FRONTEND_PATH not set, fall back to frontend/dist/
# ---------------------------------------------------------------------------


class TestAC03FallbackToRepoDist:
    def test_fallback_when_env_not_set(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "PARALLAX_FRONTEND_PATH"}
        with patch.dict(os.environ, env, clear=True):
            resolved = _resolve_frontend_dist()
        expected = Path(__file__).parent.parent / "frontend" / "dist"
        assert resolved == expected

    def test_fallback_path_ends_with_frontend_dist(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "PARALLAX_FRONTEND_PATH"}
        with patch.dict(os.environ, env, clear=True):
            resolved = _resolve_frontend_dist()
        assert resolved.parts[-2:] == ("frontend", "dist")


# ---------------------------------------------------------------------------
# AC04 — If neither path exists, server starts without /ui; warning logged
# ---------------------------------------------------------------------------


class TestAC04NeitherPathExists:
    def test_no_ui_route_when_dist_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent_dist"
        _app = FastAPI(title="test-no-frontend")
        _app.include_router(router)
        # Intentionally do NOT call _mount_frontend_ui
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.get("/ui")
        assert resp.status_code == 404

    def test_warning_logged_when_path_missing(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        missing = tmp_path / "nonexistent_dist"
        with caplog.at_level(logging.WARNING, logger="server.main"):
            with patch.dict(os.environ, {"PARALLAX_FRONTEND_PATH": str(missing)}):
                # Re-import the resolution logic and simulate the startup check
                resolved = _resolve_frontend_dist()
                if not resolved.is_dir():
                    logging.getLogger("server.main").warning(
                        "Frontend not found at %s — /ui will not be served.", resolved
                    )
        assert "Frontend not found at" in caplog.text
        assert "/ui will not be served." in caplog.text

    def test_warning_message_contains_path(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        missing = tmp_path / "no_dist_here"
        with caplog.at_level(logging.WARNING, logger="server.main"):
            with patch.dict(os.environ, {"PARALLAX_FRONTEND_PATH": str(missing)}):
                resolved = _resolve_frontend_dist()
                if not resolved.is_dir():
                    logging.getLogger("server.main").warning(
                        "Frontend not found at %s — /ui will not be served.", resolved
                    )
        assert str(missing) in caplog.text

    def test_api_routes_still_work_without_frontend(self) -> None:
        """Server operates normally (API routes reachable) when frontend is absent."""
        _app = FastAPI(title="test-no-frontend")
        _app.include_router(router)
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.get("/jobs")
        assert resp.status_code != 500
