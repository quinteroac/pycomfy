"""Tests for US-001, US-002, US-003: MCP inference and job tools."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_queue(job_id: str = "test-job-123") -> MagicMock:
    """Return a mock JobQueue that returns *job_id* from enqueue."""
    queue = MagicMock()
    queue.enqueue = AsyncMock(return_value=job_id)
    return queue


# ---------------------------------------------------------------------------
# US-001: Non-blocking inference tools
# ---------------------------------------------------------------------------

class TestCreateImage:
    def _run(self, **kwargs):
        from mcp.tools.inference import create_image
        return asyncio.run(create_image(**kwargs))

    def test_returns_job_id_string(self):
        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = _make_mock_queue("img-001")
            result = self._run(model="flux_klein", prompt="a red ball")
        assert "job_id: img-001" in result
        assert "status: queued" in result
        assert "model: flux_klein" in result

    def test_completes_under_500ms(self):
        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = _make_mock_queue("img-fast")
            t0 = time.monotonic()
            self._run(model="sdxl", prompt="test")
            elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500, f"create_image took {elapsed_ms:.1f}ms (limit 500ms)"

    def test_optional_params_excluded_when_none(self):
        captured_data = {}

        async def fake_enqueue(data):
            captured_data.update(data.args)
            return "j1"

        queue = MagicMock()
        queue.enqueue = fake_enqueue

        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = queue
            asyncio.run(
                __import__("mcp.tools.inference", fromlist=["create_image"]).create_image(
                    model="sdxl", prompt="hello"
                )
            )
        assert "width" not in captured_data
        assert "height" not in captured_data
        assert captured_data["prompt"] == "hello"


class TestCreateVideo:
    def test_returns_job_id_string(self):
        from mcp.tools.inference import create_video
        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = _make_mock_queue("vid-001")
            result = asyncio.run(create_video(model="ltx2", prompt="a river"))
        assert "job_id: vid-001" in result
        assert "status: queued" in result
        assert "model: ltx2" in result

    def test_completes_under_500ms(self):
        from mcp.tools.inference import create_video
        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = _make_mock_queue("vid-fast")
            t0 = time.monotonic()
            asyncio.run(create_video(model="ltx2", prompt="test"))
            elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500


class TestCreateAudio:
    def test_returns_job_id_string(self):
        from mcp.tools.inference import create_audio
        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = _make_mock_queue("aud-001")
            result = asyncio.run(create_audio(model="ace_step", prompt="jazz music"))
        assert "job_id: aud-001" in result
        assert "status: queued" in result
        assert "model: ace_step" in result

    def test_completes_under_500ms(self):
        from mcp.tools.inference import create_audio
        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = _make_mock_queue("aud-fast")
            t0 = time.monotonic()
            asyncio.run(create_audio(model="ace_step", prompt="test"))
            elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500


class TestEditImage:
    def test_returns_job_id_string(self):
        from mcp.tools.inference import edit_image
        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = _make_mock_queue("edit-001")
            result = asyncio.run(
                edit_image(model="qwen", prompt="make it blue", input="/tmp/img.png")
            )
        assert "job_id: edit-001" in result
        assert "status: queued" in result
        assert "model: qwen" in result

    def test_completes_under_500ms(self):
        from mcp.tools.inference import edit_image
        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = _make_mock_queue("edit-fast")
            t0 = time.monotonic()
            asyncio.run(edit_image(model="qwen", prompt="test", input="/tmp/x.png"))
            elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500


class TestUpscaleImage:
    def test_returns_job_id_string(self):
        from mcp.tools.inference import upscale_image
        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = _make_mock_queue("up-001")
            result = asyncio.run(
                upscale_image(model="esrgan", prompt="upscale", input="/tmp/low.png")
            )
        assert "job_id: up-001" in result
        assert "status: queued" in result
        assert "model: esrgan" in result

    def test_completes_under_500ms(self):
        from mcp.tools.inference import upscale_image
        with (
            patch("mcp.tools.inference.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.inference._spawn_worker"),
            patch("mcp.tools.inference._uv_path", return_value="/usr/bin/uv"),
        ):
            mock_gq.return_value = _make_mock_queue("up-fast")
            t0 = time.monotonic()
            asyncio.run(
                upscale_image(model="esrgan", prompt="test", input="/tmp/x.png")
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500


# ---------------------------------------------------------------------------
# US-002: get_job_status tool
# ---------------------------------------------------------------------------

class TestGetJobStatus:
    def _make_row(self, status: str, model: str = "sdxl", result: dict | None = None) -> dict:
        return {
            "id": "job-abc",
            "status": status,
            "data": json.dumps({"model": model, "action": "create", "media": "image",
                                "script": "", "args": {}, "script_base": "", "uv_path": ""}),
            "result": json.dumps(result) if result is not None else None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:01:00+00:00",
        }

    def _run(self, row: dict | None) -> str:
        from mcp.tools.jobs import get_job_status
        queue = MagicMock()
        queue.get = AsyncMock(return_value=row)
        with patch("mcp.tools.jobs.get_queue", new_callable=AsyncMock) as mock_gq:
            mock_gq.return_value = queue
            return asyncio.run(get_job_status("job-abc"))

    def test_not_found(self):
        result = self._run(None)
        assert result == "status: not_found"

    def test_queued_status(self):
        result = self._run(self._make_row("queued"))
        assert "status: queued" in result
        assert "model: sdxl" in result
        assert "created_at:" in result

    def test_completed_includes_output_path(self):
        result = self._run(self._make_row("completed", result={"output_path": "/out/img.png"}))
        assert "status: completed" in result
        assert "output_path: /out/img.png" in result

    def test_failed_includes_error(self):
        result = self._run(self._make_row("failed", result={"error": "OOM"}))
        assert "status: failed" in result
        assert "error: OOM" in result

    def test_accepts_only_job_id_param(self):
        """get_job_status signature must accept job_id: str as its only parameter."""
        import inspect
        from mcp.tools.jobs import get_job_status
        sig = inspect.signature(get_job_status)
        params = list(sig.parameters.keys())
        assert params == ["job_id"]


# ---------------------------------------------------------------------------
# US-003: wait_for_job tool
# ---------------------------------------------------------------------------

class TestWaitForJob:
    def _make_row(self, status: str, result: dict | None = None) -> dict:
        return {
            "id": "job-xyz",
            "status": status,
            "data": json.dumps({"model": "sdxl", "action": "create", "media": "image",
                                "script": "", "args": {}, "script_base": "", "uv_path": ""}),
            "result": json.dumps(result) if result is not None else None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:01:00+00:00",
        }

    def test_completed_returns_output_path(self):
        from mcp.tools.jobs import wait_for_job
        row = self._make_row("completed", result={"output_path": "/output/video.mp4"})
        queue = MagicMock()
        queue.get = AsyncMock(return_value=row)
        with patch("mcp.tools.jobs.get_queue", new_callable=AsyncMock) as mock_gq:
            mock_gq.return_value = queue
            result = asyncio.run(wait_for_job("job-xyz"))
        assert result == "output: /output/video.mp4"

    def test_failed_returns_error(self):
        from mcp.tools.jobs import wait_for_job
        row = self._make_row("failed", result={"error": "CUDA OOM"})
        queue = MagicMock()
        queue.get = AsyncMock(return_value=row)
        with patch("mcp.tools.jobs.get_queue", new_callable=AsyncMock) as mock_gq:
            mock_gq.return_value = queue
            result = asyncio.run(wait_for_job("job-xyz"))
        assert result == "error: CUDA OOM"

    def test_not_found_returns_error(self):
        from mcp.tools.jobs import wait_for_job
        queue = MagicMock()
        queue.get = AsyncMock(return_value=None)
        with patch("mcp.tools.jobs.get_queue", new_callable=AsyncMock) as mock_gq:
            mock_gq.return_value = queue
            result = asyncio.run(wait_for_job("missing-job"))
        assert result == "error: job not found"

    def test_timeout_returns_error_text(self):
        from mcp.tools.jobs import wait_for_job
        row = self._make_row("running")
        queue = MagicMock()
        queue.get = AsyncMock(return_value=row)
        with (
            patch("mcp.tools.jobs.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.jobs.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_gq.return_value = queue
            result = asyncio.run(wait_for_job("job-xyz", timeout_seconds=2))
        assert result == "error: timeout after 2s"

    def test_polls_with_sleep_between_attempts(self):
        """wait_for_job must use asyncio.sleep(2) — never busy-wait."""
        from mcp.tools.jobs import wait_for_job
        rows = [self._make_row("running"), self._make_row("running"), self._make_row("completed", result={"output_path": "/out/x.png"})]
        queue = MagicMock()
        queue.get = AsyncMock(side_effect=rows)
        sleep_calls = []

        async def fake_sleep(n):
            sleep_calls.append(n)

        with (
            patch("mcp.tools.jobs.get_queue", new_callable=AsyncMock) as mock_gq,
            patch("mcp.tools.jobs.asyncio.sleep", side_effect=fake_sleep),
        ):
            mock_gq.return_value = queue
            result = asyncio.run(wait_for_job("job-xyz", timeout_seconds=60))
        assert result == "output: /out/x.png"
        assert all(s == 2 for s in sleep_calls), f"sleep called with non-2 values: {sleep_calls}"
        assert len(sleep_calls) == 2  # slept twice before seeing completed

    def test_default_timeout_is_600(self):
        import inspect
        from mcp.tools.jobs import wait_for_job
        sig = inspect.signature(wait_for_job)
        assert sig.parameters["timeout_seconds"].default == 600
