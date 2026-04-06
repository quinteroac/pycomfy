"""Tests for US-002 — ``--async`` flag on generation commands.

AC01: All five generation commands accept an ``--async`` flag (boolean, default False).
AC02: When ``--async`` is provided, the command calls ``submit_job()`` from
      ``server/submit.py`` instead of blocking.
AC03: The command prints exactly:
        Job <job_id> queued
          → parallax jobs watch <job_id>
      and exits with code 0.
AC04: When ``--async`` is NOT provided, behavior is identical to current sync mode.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

_FAKE_JOB_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

# Patch targets — both live in cli._async which loads without pydantic/server deps.
# _call_submit_job is a module-level function wrapping submit_job, patchable without
# importing server.submit (which would pull in pydantic).
_SUBMIT_PATCH = "cli._async._call_submit_job"
# server.jobs is patched as a whole module so JobData creation doesn't need pydantic.
_SERVER_JOBS_MODULE = "server.jobs"


def _inject_server_jobs_mock():
    """Insert a fake server.jobs module into sys.modules so cli._async can import it."""
    if "server.jobs" not in sys.modules:
        fake_jobs = ModuleType("server.jobs")
        fake_jobs.JobData = MagicMock(return_value=MagicMock())
        sys.modules["server"] = sys.modules.get("server") or ModuleType("server")
        sys.modules["server.jobs"] = fake_jobs


@pytest.fixture(autouse=True)
def mock_server_jobs():
    """Ensure server.jobs is available (mocked) for all tests in this file."""
    _inject_server_jobs_mock()


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def models_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a tiny PNG for tests that need an input image."""
    img_path = tmp_path / "input.png"
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(str(img_path))
    except ImportError:
        import struct, zlib
        def _png():
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF
            idat_data = zlib.compress(b"\x00\xff\xff\xff")
            idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            return (
                sig
                + struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", ihdr_crc)
                + struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc)
                + struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            )
        img_path.write_bytes(_png())
    return str(img_path)


def _import_app():
    from cli.main import app
    return app


# ===========================================================================
# AC01 — All five commands accept --async flag (shown in --help)
# ===========================================================================

class TestAsyncFlagExists:
    """AC01: --async appears in --help for all five commands."""

    def test_create_image_help_shows_async(self, runner):
        result = runner.invoke(_import_app(), ["create", "image", "--help"])
        assert result.exit_code == 0
        assert "--async" in result.output

    def test_create_video_help_shows_async(self, runner):
        result = runner.invoke(_import_app(), ["create", "video", "--help"])
        assert result.exit_code == 0
        assert "--async" in result.output

    def test_create_audio_help_shows_async(self, runner):
        result = runner.invoke(_import_app(), ["create", "audio", "--help"])
        assert result.exit_code == 0
        assert "--async" in result.output

    def test_edit_image_help_shows_async(self, runner):
        result = runner.invoke(_import_app(), ["edit", "image", "--help"])
        assert result.exit_code == 0
        assert "--async" in result.output

    def test_upscale_image_help_shows_async(self, runner):
        result = runner.invoke(_import_app(), ["upscale", "image", "--help"])
        assert result.exit_code == 0
        assert "--async" in result.output


# ===========================================================================
# AC02 — --async calls submit_job() instead of blocking pipeline
# ===========================================================================

class TestAsyncCallsSubmitJob:
    """AC02: submit_job() is called when --async is provided; pipeline is NOT called."""

    def test_create_image_async_calls_submit_job(self, runner, models_dir):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID) as mock_submit:
            result = runner.invoke(_import_app(), [
                "create", "image",
                "--model", "sdxl",
                "--prompt", "a test image",
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0, result.output
        mock_submit.assert_called_once()

    def test_create_image_async_does_not_call_pipeline(self, runner, models_dir):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID), \
             patch("comfy_diffusion.pipelines.image.sdxl.t2i.run") as mock_run:
            runner.invoke(_import_app(), [
                "create", "image",
                "--model", "sdxl",
                "--prompt", "test",
                "--models-dir", models_dir,
                "--async",
            ])
        mock_run.assert_not_called()

    def test_create_video_async_calls_submit_job(self, runner, models_dir):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID) as mock_submit:
            result = runner.invoke(_import_app(), [
                "create", "video",
                "--model", "wan21",
                "--prompt", "a test video",
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0, result.output
        mock_submit.assert_called_once()

    def test_create_audio_async_calls_submit_job(self, runner, models_dir):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID) as mock_submit:
            result = runner.invoke(_import_app(), [
                "create", "audio",
                "--model", "ace_step",
                "--prompt", "jazz piano",
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0, result.output
        mock_submit.assert_called_once()

    def test_edit_image_async_calls_submit_job(self, runner, models_dir, sample_image_path):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID) as mock_submit:
            result = runner.invoke(_import_app(), [
                "edit", "image",
                "--model", "flux_4b_base",
                "--prompt", "make it bright",
                "--input", sample_image_path,
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0, result.output
        mock_submit.assert_called_once()

    def test_upscale_image_async_calls_submit_job(self, runner, models_dir, sample_image_path):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID) as mock_submit:
            result = runner.invoke(_import_app(), [
                "upscale", "image",
                "--model", "esrgan",
                "--input", sample_image_path,
                "--esrgan-checkpoint", "RealESRGAN_x4plus.pth",
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0, result.output
        mock_submit.assert_called_once()


# ===========================================================================
# AC03 — Exact output message and exit code 0
# ===========================================================================

class TestAsyncOutputFormat:
    """AC03: prints correct queued message and exits 0."""

    def test_create_image_async_output_format(self, runner, models_dir):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID):
            result = runner.invoke(_import_app(), [
                "create", "image",
                "--model", "sdxl",
                "--prompt", "test",
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0, result.output
        assert f"Job {_FAKE_JOB_ID} queued" in result.output
        assert f"parallax jobs watch {_FAKE_JOB_ID}" in result.output

    def test_create_video_async_output_format(self, runner, models_dir):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID):
            result = runner.invoke(_import_app(), [
                "create", "video",
                "--model", "wan21",
                "--prompt", "test video",
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0, result.output
        assert f"Job {_FAKE_JOB_ID} queued" in result.output
        assert f"parallax jobs watch {_FAKE_JOB_ID}" in result.output

    def test_create_audio_async_output_format(self, runner, models_dir):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID):
            result = runner.invoke(_import_app(), [
                "create", "audio",
                "--model", "ace_step",
                "--prompt", "jazz",
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0, result.output
        assert f"Job {_FAKE_JOB_ID} queued" in result.output
        assert f"parallax jobs watch {_FAKE_JOB_ID}" in result.output

    def test_edit_image_async_output_format(self, runner, models_dir, sample_image_path):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID):
            result = runner.invoke(_import_app(), [
                "edit", "image",
                "--model", "qwen",
                "--prompt", "brighten",
                "--input", sample_image_path,
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0, result.output
        assert f"Job {_FAKE_JOB_ID} queued" in result.output
        assert f"parallax jobs watch {_FAKE_JOB_ID}" in result.output

    def test_upscale_image_async_output_format(self, runner, models_dir, sample_image_path):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID):
            result = runner.invoke(_import_app(), [
                "upscale", "image",
                "--model", "esrgan",
                "--input", sample_image_path,
                "--esrgan-checkpoint", "RealESRGAN_x4plus.pth",
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0, result.output
        assert f"Job {_FAKE_JOB_ID} queued" in result.output
        assert f"parallax jobs watch {_FAKE_JOB_ID}" in result.output

    def test_async_exit_code_is_zero(self, runner, models_dir):
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID):
            result = runner.invoke(_import_app(), [
                "create", "image",
                "--model", "sdxl",
                "--prompt", "test",
                "--models-dir", models_dir,
                "--async",
            ])
        assert result.exit_code == 0

    def test_async_output_two_lines(self, runner, models_dir):
        """The output must contain both the queued line and the watch hint line."""
        with patch(_SUBMIT_PATCH, return_value=_FAKE_JOB_ID):
            result = runner.invoke(_import_app(), [
                "create", "image",
                "--model", "sdxl",
                "--prompt", "test",
                "--models-dir", models_dir,
                "--async",
            ])
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert any(_FAKE_JOB_ID in ln and "queued" in ln for ln in lines)
        assert any(_FAKE_JOB_ID in ln and "watch" in ln for ln in lines)


# ===========================================================================
# AC04 — Without --async, sync behavior is unchanged
# ===========================================================================

class TestSyncModeUnchanged:
    """AC04: without --async, commands behave identically to sync mode."""

    def test_create_image_sync_still_calls_pipeline(self, runner, tmp_path, models_dir):
        from PIL import Image
        fake_image = Image.new("RGB", (16, 16))
        out_file = str(tmp_path / "out.png")
        with patch("comfy_diffusion.pipelines.image.sdxl.t2i.run", return_value=[fake_image]):
            result = runner.invoke(_import_app(), [
                "create", "image",
                "--model", "sdxl",
                "--prompt", "test",
                "--output", out_file,
                "--models-dir", models_dir,
            ])
        assert result.exit_code == 0, result.output

    def test_create_image_sync_does_not_call_submit_job(self, runner, tmp_path, models_dir):
        from PIL import Image
        fake_image = Image.new("RGB", (16, 16))
        out_file = str(tmp_path / "out.png")
        with patch("comfy_diffusion.pipelines.image.sdxl.t2i.run", return_value=[fake_image]), \
             patch(_SUBMIT_PATCH) as mock_submit:
            runner.invoke(_import_app(), [
                "create", "image",
                "--model", "sdxl",
                "--prompt", "test",
                "--output", out_file,
                "--models-dir", models_dir,
            ])
        mock_submit.assert_not_called()

    def test_create_video_sync_no_submit_job(self, runner, tmp_path, models_dir):
        """Sync mode for create video does not call submit_job."""
        out_file = str(tmp_path / "out.mp4")
        with patch("comfy_diffusion.pipelines.video.wan.wan21.t2v.run", return_value=MagicMock()), \
             patch("cli._io.save_video_frames", return_value=out_file), \
             patch(_SUBMIT_PATCH) as mock_submit:
            runner.invoke(_import_app(), [
                "create", "video",
                "--model", "wan21",
                "--prompt", "test",
                "--output", out_file,
                "--models-dir", models_dir,
            ])
        mock_submit.assert_not_called()
