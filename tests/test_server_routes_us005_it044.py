"""Tests for US-005: GET /health endpoint."""
from __future__ import annotations

from unittest.mock import patch

from starlette.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_health_returns_200() -> None:
    """AC01: GET /health returns HTTP 200."""
    with patch("server.app._pkg_version", return_value="1.3.0"):
        response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_status_ok() -> None:
    """AC01: response body contains status=ok."""
    with patch("server.app._pkg_version", return_value="1.3.0"):
        response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_health_returns_version() -> None:
    """AC01: response body contains version matching the package version."""
    with patch("server.app._pkg_version", return_value="1.3.0"):
        response = client.get("/health")
    assert response.json()["version"] == "1.3.0"


def test_health_no_db_call(monkeypatch: object) -> None:
    """AC02: GET /health never calls get_queue (no DB dependency)."""
    called: list[bool] = []

    async def fake_get_queue():  # type: ignore[return]
        called.append(True)

    monkeypatch.setattr("server.app.get_queue", fake_get_queue)
    with patch("server.app._pkg_version", return_value="1.3.0"):
        response = client.get("/health")
    assert response.status_code == 200
    assert called == [], "get_queue must not be called by /health"
