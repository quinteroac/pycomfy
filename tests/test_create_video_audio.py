"""CPU-only tests for ``parallax create video --audio`` guard validations (RF-2, it_000046).

Three validation scenarios that do not require a GPU:
  (a) --audio with a non-ltx23 model  → exit 1 + error message
  (b) --audio with a non-existent file → exit 1 + error message
  (c) --audio without --input          → exit 1 + error message
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner


def _app():
    from cli.main import app
    return app


class TestAudioUnsupportedModelGuard:
    """(a) --audio with a non-ltx23 model prints error and exits 1."""

    @pytest.mark.parametrize("model", ["ltx2", "wan21", "wan22"])
    def test_exits_1_for_unsupported_model(self, tmp_path, model):
        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"RIFF")
        image_file = tmp_path / "frame.png"
        image_file.write_bytes(b"PNG")

        result = CliRunner().invoke(_app(), [
            "create", "video",
            "--model", model,
            "--prompt", "test",
            "--input", str(image_file),
            "--audio", str(audio_file),
        ])

        assert result.exit_code == 1
        assert "--audio is only supported for model 'ltx23'" in result.output


class TestAudioMissingFileGuard:
    """(b) --audio with a path that does not exist prints error and exits 1."""

    def test_exits_1_when_audio_file_missing(self, tmp_path):
        image_file = tmp_path / "frame.png"
        image_file.write_bytes(b"PNG")
        missing = str(tmp_path / "nonexistent.wav")

        result = CliRunner().invoke(_app(), [
            "create", "video",
            "--model", "ltx23",
            "--prompt", "test",
            "--input", str(image_file),
            "--audio", missing,
        ])

        assert result.exit_code == 1
        assert f"audio file not found: {missing}" in result.output


class TestAudioRequiresInputGuard:
    """(c) --audio without --input prints error and exits 1."""

    def test_exits_1_when_input_missing(self, tmp_path):
        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"RIFF")

        result = CliRunner().invoke(_app(), [
            "create", "video",
            "--model", "ltx23",
            "--prompt", "test",
            "--audio", str(audio_file),
        ])

        assert result.exit_code == 1
        assert "--audio requires --input (image)" in result.output
