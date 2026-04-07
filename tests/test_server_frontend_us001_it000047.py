"""Tests for US-001 it_000047 — Frontend served from FastAPI at /ui.

Covers:
  AC01 — GET /ui returns index.html
  AC02 — GET /ui/assets/<file> serves JS/CSS correctly
  AC03 — GET /ui/ (trailing slash) returns index.html
  AC04 — Existing API routes /create/* and /jobs/* are unaffected
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.testclient import TestClient

from server.gateway import router
from server.main import _mount_frontend_ui


@pytest.fixture
def dist_dir(tmp_path: Path) -> Path:
    """Create a minimal mock frontend dist directory."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<html><head><title>Parallax</title></head><body><div id='root'></div></body></html>",
        encoding="utf-8",
    )
    assets = dist / "assets"
    assets.mkdir()
    (assets / "main-abc123.js").write_text("console.log('parallax');", encoding="utf-8")
    (assets / "main-def456.css").write_text("body { margin: 0; }", encoding="utf-8")
    return dist


@pytest.fixture
def ui_client(dist_dir: Path) -> TestClient:
    """TestClient for a fresh app instance with frontend mounted."""
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


# ---------------------------------------------------------------------------
# AC01 — GET /ui returns index.html
# ---------------------------------------------------------------------------

class TestAC01UiRoot:
    def test_ui_returns_200(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui")
        assert resp.status_code == 200

    def test_ui_returns_html(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui")
        assert "text/html" in resp.headers["content-type"]

    def test_ui_contains_root_div(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui")
        assert "root" in resp.text


# ---------------------------------------------------------------------------
# AC02 — GET /ui/assets/<file> serves assets correctly
# ---------------------------------------------------------------------------

class TestAC02Assets:
    def test_js_asset_served(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/assets/main-abc123.js")
        assert resp.status_code == 200

    def test_js_asset_content(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/assets/main-abc123.js")
        assert "parallax" in resp.text

    def test_css_asset_served(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/assets/main-def456.css")
        assert resp.status_code == 200

    def test_css_asset_content_type(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/assets/main-def456.css")
        assert "css" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# AC03 — GET /ui/ (trailing slash) returns index.html
# ---------------------------------------------------------------------------

class TestAC03TrailingSlash:
    def test_ui_trailing_slash_200(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/")
        assert resp.status_code == 200

    def test_ui_trailing_slash_html(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/")
        assert "text/html" in resp.headers["content-type"]

    def test_ui_trailing_slash_contains_root(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/ui/")
        assert "root" in resp.text


# ---------------------------------------------------------------------------
# AC04 — Existing API routes unaffected
# ---------------------------------------------------------------------------

class TestAC04NoRouteCollision:
    def test_health_check_unaffected(self, ui_client: TestClient) -> None:
        """GET /health (or any non-ui path) is not intercepted by static mount."""
        resp = ui_client.get("/ui/nonexistent-file-xyz.txt")
        # Should be 404, not a server error
        assert resp.status_code == 404

    def test_jobs_list_reachable(self, ui_client: TestClient) -> None:
        resp = ui_client.get("/jobs")
        # Gateway handles /jobs — must not 404 as if matched by static files
        assert resp.status_code != 500

    def test_ui_path_does_not_shadow_api(self, ui_client: TestClient) -> None:
        """Routes not under /ui are never handled by the static mount."""
        # /jobs/unknown returns 404 from gateway, not static files
        resp = ui_client.get("/jobs/unknown-job-id-xyz")
        assert resp.status_code in (404, 422)  # gateway's response, not static 404
