"""Tests for US-001 — Generation commands (sync mode).

AC01: parallax create image/video/audio, parallax edit image, parallax upscale image
       are implemented as Typer commands under cli/commands/.
AC02: In sync mode (default, no --async), each command calls the pipeline run()
       directly and blocks until completion.
AC03: On success, the command prints the output file path and exits with code 0.
AC04: On failure, the command prints the error to stderr and exits with code 1.
AC05: --help on each command shows all available options with descriptions.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def models_dir(tmp_path):
    """A temporary directory that stands in as models_dir."""
    return str(tmp_path)


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a tiny PNG for tests that need an existing input image."""
    img_path = tmp_path / "input.png"
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(str(img_path))
    except ImportError:
        # PIL not available — create a minimal 1×1 PNG bytes manually
        import struct, zlib
        def _png():
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF
            idat_data = zlib.compress(b"\x00\xff\xff\xff")
            idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            return (sig
                    + struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", ihdr_crc)
                    + struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc)
                    + struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc))
        img_path.write_bytes(_png())
    return str(img_path)


def _pil_image():
    """Return a tiny PIL Image object (or mock if PIL unavailable)."""
    try:
        from PIL import Image
        return Image.new("RGB", (16, 16), color=(0, 0, 0))
    except ImportError:
        return MagicMock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_app():
    from cli.main import app
    return app


# ===========================================================================
# AC01 — Commands exist as Typer commands
# ===========================================================================

class TestCommandsExist:
    """AC01: all five commands exist in the Typer app."""

    def test_create_image_exists(self, runner):
        app = _import_app()
        result = runner.invoke(app, ["create", "image", "--help"])
        assert result.exit_code == 0, result.output

    def test_create_video_exists(self, runner):
        app = _import_app()
        result = runner.invoke(app, ["create", "video", "--help"])
        assert result.exit_code == 0, result.output

    def test_create_audio_exists(self, runner):
        app = _import_app()
        result = runner.invoke(app, ["create", "audio", "--help"])
        assert result.exit_code == 0, result.output

    def test_edit_image_exists(self, runner):
        app = _import_app()
        result = runner.invoke(app, ["edit", "image", "--help"])
        assert result.exit_code == 0, result.output

    def test_upscale_image_exists(self, runner):
        app = _import_app()
        result = runner.invoke(app, ["upscale", "image", "--help"])
        assert result.exit_code == 0, result.output


# ===========================================================================
# AC05 — --help shows all available options
# ===========================================================================

class TestHelpMessages:
    """AC05: --help on each command shows available options with descriptions."""

    def test_create_image_help_shows_model_and_prompt(self, runner):
        app = _import_app()
        result = runner.invoke(app, ["create", "image", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--prompt" in result.output
        assert "--output" in result.output
        assert "--models-dir" in result.output

    def test_create_video_help_shows_options(self, runner):
        app = _import_app()
        result = runner.invoke(app, ["create", "video", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--prompt" in result.output
        assert "--length" in result.output
        assert "--fps" in result.output

    def test_create_audio_help_shows_options(self, runner):
        app = _import_app()
        result = runner.invoke(app, ["create", "audio", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--prompt" in result.output
        assert "--bpm" in result.output
        assert "--lyrics" in result.output

    def test_edit_image_help_shows_options(self, runner):
        app = _import_app()
        result = runner.invoke(app, ["edit", "image", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--input" in result.output
        assert "--subject-image" in result.output

    def test_upscale_image_help_shows_options(self, runner):
        app = _import_app()
        result = runner.invoke(app, ["upscale", "image", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--input" in result.output
        assert "--esrgan-checkpoint" in result.output
        assert "--checkpoint" in result.output


# ===========================================================================
# AC02 + AC03 — Sync mode calls pipeline directly; on success prints path
# ===========================================================================

class TestSyncModeSuccess:
    """AC02: sync mode (default) calls pipeline run() directly.
    AC03: on success, prints output file path and exits 0.
    """

    def test_create_image_calls_pipeline_and_prints_path(self, runner, tmp_path, models_dir):
        """create image calls the sdxl pipeline and saves output, printing the path."""
        app = _import_app()
        out_file = str(tmp_path / "out.png")
        fake_image = _pil_image()

        with patch("comfy_diffusion.pipelines.image.sdxl.t2i.run", return_value=[fake_image]) as mock_run:
            result = runner.invoke(app, [
                "create", "image",
                "--model", "sdxl",
                "--prompt", "a test image",
                "--output", out_file,
                "--models-dir", models_dir,
            ])

        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
        mock_run.assert_called_once()
        # The output path is printed to stdout
        assert out_file in result.output or str(Path(out_file).resolve()) in result.output

    def test_create_video_calls_pipeline_and_prints_path(self, runner, tmp_path, models_dir):
        """create video calls the wan21 pipeline and saves output."""
        app = _import_app()
        out_file = str(tmp_path / "out.mp4")
        fake_frame = _pil_image()

        with patch("comfy_diffusion.pipelines.video.wan.wan21.t2v.run", return_value=[fake_frame]) as mock_run:
            with patch("cli.commands.create.save_video_frames", return_value=str(tmp_path / "out.mp4")) as mock_save:
                result = runner.invoke(app, [
                    "create", "video",
                    "--model", "wan21",
                    "--prompt", "a test video",
                    "--output", out_file,
                    "--models-dir", models_dir,
                ])

        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
        mock_run.assert_called_once()
        assert str(tmp_path / "out.mp4") in result.output

    def test_create_audio_calls_pipeline_and_prints_path(self, runner, tmp_path, models_dir):
        """create audio calls the ace_step pipeline and saves output."""
        app = _import_app()
        out_file = str(tmp_path / "out.wav")
        fake_audio = {"waveform": MagicMock(), "sample_rate": 44100}

        with patch("comfy_diffusion.pipelines.audio.ace_step.v1_5.checkpoint.run",
                   return_value={"audio": fake_audio}) as mock_run:
            with patch("cli.commands.create.save_audio", return_value=str(tmp_path / "out.wav")) as mock_save:
                result = runner.invoke(app, [
                    "create", "audio",
                    "--model", "ace_step",
                    "--prompt", "upbeat jazz",
                    "--output", out_file,
                    "--models-dir", models_dir,
                ])

        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
        mock_run.assert_called_once()
        assert str(tmp_path / "out.wav") in result.output

    def test_edit_image_calls_pipeline_and_prints_path(
        self, runner, tmp_path, models_dir, sample_image_path
    ):
        """edit image calls the qwen pipeline and saves output."""
        app = _import_app()
        out_file = str(tmp_path / "edited.png")
        fake_image = _pil_image()

        with patch("comfy_diffusion.pipelines.image.qwen.edit_2511.run", return_value=[fake_image]) as mock_run:
            result = runner.invoke(app, [
                "edit", "image",
                "--model", "qwen",
                "--prompt", "make it blue",
                "--input", sample_image_path,
                "--output", out_file,
                "--models-dir", models_dir,
            ])

        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
        mock_run.assert_called_once()
        assert out_file in result.output or str(Path(out_file).resolve()) in result.output

    def test_upscale_image_esrgan_calls_model_and_prints_path(
        self, runner, tmp_path, models_dir, sample_image_path
    ):
        """upscale image (esrgan) loads upscale model and saves output."""
        app = _import_app()
        out_file = str(tmp_path / "upscaled.png")
        fake_image = _pil_image()

        with patch("comfy_diffusion.models.ModelManager.load_upscale_model", return_value=MagicMock()):
            with patch("comfy_diffusion.image.load_image", return_value=(MagicMock(), MagicMock())):
                with patch("comfy_diffusion.image.image_upscale_with_model", return_value=MagicMock()):
                    with patch("cli.commands.upscale._tensor_to_pil_list", return_value=[fake_image]):
                        result = runner.invoke(app, [
                            "upscale", "image",
                            "--model", "esrgan",
                            "--input", sample_image_path,
                            "--esrgan-checkpoint", "RealESRGAN_x4plus.pth",
                            "--output", out_file,
                            "--models-dir", models_dir,
                        ])

        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
        assert out_file in result.output or str(Path(out_file).resolve()) in result.output


# ===========================================================================
# AC04 — On failure, print error to stderr and exit 1
# ===========================================================================

class TestSyncModeFailure:
    """AC04: on failure, error goes to stderr and exit code is 1."""

    def test_create_image_pipeline_exception_exits_1(self, runner, tmp_path, models_dir):
        """If the pipeline raises, exit code is 1 and error message goes to stderr."""
        app = _import_app()

        with patch("comfy_diffusion.pipelines.image.sdxl.t2i.run",
                   side_effect=RuntimeError("CUDA OOM")):
            result = runner.invoke(app, [
                "create", "image",
                "--model", "sdxl",
                "--prompt", "test",
                "--output", str(tmp_path / "out.png"),
                "--models-dir", models_dir,
            ])

        assert result.exit_code == 1
        assert "CUDA OOM" in result.output

    def test_create_image_unknown_model_exits_1(self, runner, tmp_path, models_dir):
        """Unknown model name → exit 1 with error message."""
        app = _import_app()
        result = runner.invoke(app, [
            "create", "image",
            "--model", "nonexistent_model",
            "--prompt", "test",
            "--output", str(tmp_path / "out.png"),
            "--models-dir", models_dir,
        ])
        assert result.exit_code == 1

    def test_create_video_unknown_model_exits_1(self, runner, tmp_path, models_dir):
        """Unknown video model → exit 1."""
        app = _import_app()
        result = runner.invoke(app, [
            "create", "video",
            "--model", "badmodel",
            "--prompt", "test",
            "--output", str(tmp_path / "out.mp4"),
            "--models-dir", models_dir,
        ])
        assert result.exit_code == 1

    def test_create_audio_unknown_model_exits_1(self, runner, tmp_path, models_dir):
        """Unknown audio model → exit 1."""
        app = _import_app()
        result = runner.invoke(app, [
            "create", "audio",
            "--model", "badmodel",
            "--prompt", "test",
            "--output", str(tmp_path / "out.wav"),
            "--models-dir", models_dir,
        ])
        assert result.exit_code == 1

    def test_edit_image_missing_input_exits_1(self, runner, tmp_path, models_dir):
        """edit image with non-existent --input → exit 1."""
        app = _import_app()
        result = runner.invoke(app, [
            "edit", "image",
            "--model", "qwen",
            "--prompt", "test",
            "--input", str(tmp_path / "no_such_file.png"),
            "--output", str(tmp_path / "out.png"),
            "--models-dir", models_dir,
        ])
        assert result.exit_code == 1

    def test_upscale_image_missing_esrgan_checkpoint_exits_1(
        self, runner, tmp_path, models_dir, sample_image_path
    ):
        """upscale image (esrgan) without --esrgan-checkpoint → exit 1."""
        app = _import_app()
        result = runner.invoke(app, [
            "upscale", "image",
            "--model", "esrgan",
            "--input", sample_image_path,
            "--output", str(tmp_path / "out.png"),
            "--models-dir", models_dir,
        ])
        assert result.exit_code == 1

    def test_upscale_image_pipeline_exception_exits_1(
        self, runner, tmp_path, models_dir, sample_image_path
    ):
        """If ESRGAN pipeline raises, exit 1."""
        app = _import_app()

        with patch("comfy_diffusion.models.ModelManager.load_upscale_model",
                   side_effect=RuntimeError("model not found")):
            result = runner.invoke(app, [
                "upscale", "image",
                "--model", "esrgan",
                "--input", sample_image_path,
                "--esrgan-checkpoint", "fake.pth",
                "--output", str(tmp_path / "out.png"),
                "--models-dir", models_dir,
            ])

        assert result.exit_code == 1

    def test_invalid_models_dir_exits_1(self, runner, tmp_path):
        """If --models-dir does not exist, exit 1."""
        app = _import_app()
        result = runner.invoke(app, [
            "create", "image",
            "--model", "sdxl",
            "--prompt", "test",
            "--models-dir", str(tmp_path / "no_such_dir"),
        ])
        assert result.exit_code == 1


# TestAsyncModeNotYetAvailable has been removed.
# --async is now fully implemented in US-002 (it_000044).
# See tests/test_cli_us002_it044.py for --async tests.
