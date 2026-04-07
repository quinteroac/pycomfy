"""Tests for US-001 (it_000046) — ``parallax create video --audio`` option.

AC01: ltx23 + --audio + --input runs without error and produces a valid .mp4.
AC02: --audio is optional; omitting it leaves existing t2v / i2v behaviour.
AC03: --audio with a model other than ltx23 → error + exit 1.
AC04: --audio path does not exist → error + exit 1.
AC05: --audio without --input → error + exit 1.
AC06: typecheck / lint passes (structural test — imports and signatures checked).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _app():
    from cli.main import app
    return app


def _runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# AC01 — ltx23 + --audio runs and produces output
# ---------------------------------------------------------------------------


class TestAudioOption:
    """AC01: running with --audio routes to ia2v and saves output."""

    def test_ltx23_audio_calls_ia2v_pipeline(self, tmp_path):
        """create video --model ltx23 --audio routes to ia2v pipeline (AC01)."""
        runner = _runner()
        app = _app()

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        image_file = tmp_path / "frame.png"
        audio_file = tmp_path / "track.wav"
        image_file.write_bytes(b"PNG")
        audio_file.write_bytes(b"RIFF")
        output_file = str(tmp_path / "out.mp4")

        fake_frames = [MagicMock()]
        mock_runner = MagicMock(return_value=fake_frames)

        with patch("cli.commands.create.ensure_env_on_path"):
            with patch("PIL.Image.open") as mock_open:
                mock_open.return_value.convert.return_value = MagicMock()
                with patch("cli._runners.video.RUNNERS", {"ltx23": mock_runner}):
                    with patch("cli.commands.create.save_video_frames", return_value=output_file):
                        result = runner.invoke(app, [
                            "create", "video",
                            "--model", "ltx23",
                            "--prompt", "test prompt",
                            "--input", str(image_file),
                            "--audio", str(audio_file),
                            "--output", output_file,
                            "--models-dir", str(models_dir),
                        ])

        assert result.exit_code == 0, result.output
        mock_runner.assert_called_once()
        call_kwargs = mock_runner.call_args.kwargs
        assert call_kwargs["audio"] == str(audio_file)

    def test_ltx23_audio_ia2v_run_invoked(self, tmp_path):
        """create video --model ltx23 --audio invokes ia2v.run() with audio_path (AC01)."""
        runner = _runner()
        app = _app()

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        image_file = tmp_path / "frame.png"
        audio_file = tmp_path / "track.wav"
        image_file.write_bytes(b"PNG")
        audio_file.write_bytes(b"RIFF")
        output_file = str(tmp_path / "out.mp4")

        fake_frames = [MagicMock()]
        fake_result = {"frames": fake_frames, "audio": {"waveform": MagicMock(), "sample_rate": 44100}}

        with patch("cli.commands.create.ensure_env_on_path"):
            with patch("PIL.Image.open") as mock_open:
                mock_open.return_value.convert.return_value = MagicMock()
                with patch(
                    "comfy_diffusion.pipelines.video.ltx.ltx23.ia2v.run",
                    return_value=fake_result,
                ) as mock_run:
                    with patch("cli.commands.create.save_video_frames", return_value=output_file):
                        result = runner.invoke(app, [
                            "create", "video",
                            "--model", "ltx23",
                            "--prompt", "test prompt",
                            "--input", str(image_file),
                            "--audio", str(audio_file),
                            "--output", output_file,
                            "--models-dir", str(models_dir),
                        ])

        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["audio_path"] == str(audio_file)


# ---------------------------------------------------------------------------
# AC02 — --audio is optional; existing behaviour unchanged
# ---------------------------------------------------------------------------


class TestAudioOptional:
    """AC02: omitting --audio keeps t2v / i2v behaviour for all models."""

    def test_ltx23_t2v_without_audio(self, tmp_path):
        """ltx23 t2v still works when --audio is omitted (AC02)."""
        runner = _runner()
        app = _app()

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mock_runner = MagicMock(return_value=[MagicMock()])
        with patch("cli.commands.create.ensure_env_on_path"):
            with patch("cli._runners.video.RUNNERS", {"ltx23": mock_runner}):
                with patch("cli.commands.create.save_video_frames", return_value="out.mp4"):
                    result = runner.invoke(app, [
                        "create", "video",
                        "--model", "ltx23",
                        "--prompt", "a test",
                        "--models-dir", str(models_dir),
                    ])

        assert result.exit_code == 0, result.output
        call_kwargs = mock_runner.call_args.kwargs
        assert call_kwargs.get("audio") is None

    def test_wan22_without_audio(self, tmp_path):
        """wan22 t2v still works when --audio is omitted (AC02)."""
        runner = _runner()
        app = _app()

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mock_runner = MagicMock(return_value=[MagicMock()])
        with patch("cli.commands.create.ensure_env_on_path"):
            with patch("cli._runners.video.RUNNERS", {"wan22": mock_runner}):
                with patch("cli.commands.create.save_video_frames", return_value="out.mp4"):
                    result = runner.invoke(app, [
                        "create", "video",
                        "--model", "wan22",
                        "--prompt", "a test",
                        "--models-dir", str(models_dir),
                    ])

        assert result.exit_code == 0, result.output

    def test_ltx23_i2v_without_audio(self, tmp_path):
        """ltx23 i2v (--input provided, no --audio) still works (AC02)."""
        runner = _runner()
        app = _app()

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        image_file = tmp_path / "frame.png"
        image_file.write_bytes(b"PNG")

        mock_runner = MagicMock(return_value=[MagicMock()])
        with patch("cli.commands.create.ensure_env_on_path"):
            with patch("PIL.Image.open") as mock_open:
                mock_open.return_value.convert.return_value = MagicMock()
                with patch("cli._runners.video.RUNNERS", {"ltx23": mock_runner}):
                    with patch("cli.commands.create.save_video_frames", return_value="out.mp4"):
                        result = runner.invoke(app, [
                            "create", "video",
                            "--model", "ltx23",
                            "--prompt", "a test",
                            "--input", str(image_file),
                            "--models-dir", str(models_dir),
                        ])

        assert result.exit_code == 0, result.output
        call_kwargs = mock_runner.call_args.kwargs
        assert call_kwargs.get("audio") is None


# ---------------------------------------------------------------------------
# AC03 — --audio with unsupported model → error
# ---------------------------------------------------------------------------


class TestAudioUnsupportedModel:
    """AC03: --audio with a model other than ltx23 → exit 1 with error message."""

    @pytest.mark.parametrize("model", ["ltx2", "wan21", "wan22"])
    def test_audio_unsupported_model_exits_1(self, tmp_path, model):
        """--audio with model other than ltx23 prints error and exits 1 (AC03)."""
        runner = _runner()
        app = _app()

        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"RIFF")
        image_file = tmp_path / "frame.png"
        image_file.write_bytes(b"PNG")

        result = runner.invoke(app, [
            "create", "video",
            "--model", model,
            "--prompt", "test",
            "--input", str(image_file),
            "--audio", str(audio_file),
        ])

        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "--audio is only supported for model 'ltx23'" in combined

    def test_audio_unsupported_model_error_to_stderr(self, tmp_path):
        """Error message is sent to stderr (AC03)."""
        runner = _runner()
        app = _app()

        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"RIFF")
        image_file = tmp_path / "frame.png"
        image_file.write_bytes(b"PNG")

        result = runner.invoke(app, [
            "create", "video",
            "--model", "wan21",
            "--prompt", "test",
            "--input", str(image_file),
            "--audio", str(audio_file),
        ])

        assert result.exit_code == 1
        assert "--audio is only supported for model 'ltx23'" in result.output


# ---------------------------------------------------------------------------
# AC04 — --audio path does not exist
# ---------------------------------------------------------------------------


class TestAudioFileNotFound:
    """AC04: non-existent audio path → exit 1 with error message."""

    def test_missing_audio_file_exits_1(self, tmp_path):
        """Non-existent --audio path prints error and exits 1 (AC04)."""
        runner = _runner()
        app = _app()

        image_file = tmp_path / "frame.png"
        image_file.write_bytes(b"PNG")
        missing_audio = str(tmp_path / "nonexistent.wav")

        result = runner.invoke(app, [
            "create", "video",
            "--model", "ltx23",
            "--prompt", "test",
            "--input", str(image_file),
            "--audio", missing_audio,
        ])

        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert f"audio file not found: {missing_audio}" in combined

    def test_missing_audio_file_message_contains_path(self, tmp_path):
        """Error message contains the provided path (AC04)."""
        runner = _runner()
        app = _app()

        image_file = tmp_path / "frame.png"
        image_file.write_bytes(b"PNG")
        missing_audio = "/absolutely/not/there/track.mp3"

        result = runner.invoke(app, [
            "create", "video",
            "--model", "ltx23",
            "--prompt", "test",
            "--input", str(image_file),
            "--audio", missing_audio,
        ])

        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert missing_audio in combined


# ---------------------------------------------------------------------------
# AC05 — --audio without --input
# ---------------------------------------------------------------------------


class TestAudioRequiresInput:
    """AC05: --audio without --input → exit 1 with error message."""

    def test_audio_without_input_exits_1(self, tmp_path):
        """--audio without --input prints error and exits 1 (AC05)."""
        runner = _runner()
        app = _app()

        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"RIFF")

        result = runner.invoke(app, [
            "create", "video",
            "--model", "ltx23",
            "--prompt", "test",
            "--audio", str(audio_file),
        ])

        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "--audio requires --input (image)" in combined

    def test_audio_without_input_error_message(self, tmp_path):
        """Error message matches expected text exactly (AC05)."""
        runner = _runner()
        app = _app()

        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"RIFF")

        result = runner.invoke(app, [
            "create", "video",
            "--model", "ltx23",
            "--prompt", "test",
            "--audio", str(audio_file),
        ])

        assert result.exit_code == 1
        assert "Error: --audio requires --input (image)." in result.output


# ---------------------------------------------------------------------------
# AC06 — structural / import checks (typecheck / lint)
# ---------------------------------------------------------------------------


class TestStructuralChecks:
    """AC06: CLI imports cleanly; --audio flag is present in help."""

    def test_create_video_help_shows_audio_option(self):
        """--help output lists --audio option (AC06)."""
        runner = _runner()
        app = _app()
        result = runner.invoke(app, ["create", "video", "--help"])
        assert result.exit_code == 0
        assert "--audio" in result.output

    def test_runner_accepts_audio_kwarg(self):
        """_ltx23 runner accepts audio keyword argument without TypeError (AC06)."""
        from cli._runners.video import _ltx23

        fake_result = {"frames": [MagicMock()], "audio": {}}
        with patch(
            "comfy_diffusion.pipelines.video.ltx.ltx23.ia2v.run",
            return_value=fake_result,
        ):
            # Verify the function signature accepts audio without raising TypeError.
            import inspect
            sig = inspect.signature(_ltx23)
            assert "audio" in sig.parameters

    def test_create_module_imports_cleanly(self):
        """cli.commands.create imports without errors (AC06)."""
        import importlib
        mod = importlib.import_module("cli.commands.create")
        assert hasattr(mod, "create_video")

    def test_video_runner_module_imports_cleanly(self):
        """cli._runners.video imports without errors (AC06)."""
        import importlib
        mod = importlib.import_module("cli._runners.video")
        assert hasattr(mod, "_ltx23")
        assert hasattr(mod, "RUNNERS")
