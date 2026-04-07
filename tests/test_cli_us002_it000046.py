"""Tests for US-002 (it_000046) — Route to ``ia2v`` pipeline in the ``_ltx23`` runner.

AC01: ``_ltx23`` accepts ``audio`` keyword argument (default ``None``).
AC02: When ``audio`` is not ``None``, calls ``ia2v.run()`` with the correct args
      including ``models_dir``, ``image``, ``audio_path``, ``prompt``, ``width``,
      ``height``, ``length``, ``fps``, ``cfg``, and ``seed``.
AC03: When ``audio`` is ``None`` and ``image`` is set, existing ``i2v`` path is unchanged.
AC04: When both are ``None``, existing ``t2v`` path is unchanged.
AC05: ``RUNNERS["ltx23"]`` signature accepts ``audio=None`` without breaking existing
      call sites (uses ``**_`` for unknown kwargs).
AC06: Typecheck / lint passes (structural / import checks).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ltx23():
    from cli._runners.video import _ltx23 as fn
    return fn


def _runners():
    from cli._runners.video import RUNNERS
    return RUNNERS


# ---------------------------------------------------------------------------
# AC01 — _ltx23 accepts audio kwarg with default None
# ---------------------------------------------------------------------------


class TestAC01AudioKwarg:
    """AC01: ``_ltx23`` signature includes ``audio`` with default ``None``."""

    def test_audio_param_in_signature(self):
        sig = inspect.signature(_ltx23())
        assert "audio" in sig.parameters

    def test_audio_param_default_is_none(self):
        sig = inspect.signature(_ltx23())
        assert sig.parameters["audio"].default is None


# ---------------------------------------------------------------------------
# AC02 — ia2v.run() called with correct args when audio is provided
# ---------------------------------------------------------------------------


class TestAC02IaRoute:
    """AC02: audio provided → ``ia2v.run()`` called with all required args."""

    def test_ia2v_run_called_with_cfg(self, tmp_path):
        """``cfg`` forwarded from ``c`` parameter to ``ia2v.run()`` (AC02)."""
        fake_result = {
            "frames": [MagicMock()],
            "audio": {"waveform": MagicMock(), "sample_rate": 44100},
        }

        with patch(
            "comfy_diffusion.pipelines.video.ltx.ltx23.ia2v.run",
            return_value=fake_result,
        ) as mock_run:
            _ltx23()(
                mdir=str(tmp_path),
                prompt="test",
                image=MagicMock(),
                audio="/tmp/track.wav",
                w=768,
                h=512,
                n=97,
                f=24,
                c=2.5,
                seed=42,
            )

        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        assert kwargs["cfg"] == 2.5

    def test_ia2v_run_called_with_all_required_args(self, tmp_path):
        """All required args passed to ``ia2v.run()`` (AC02)."""
        image_obj = MagicMock()
        fake_result = {"frames": [MagicMock()], "audio": {}}

        with patch(
            "comfy_diffusion.pipelines.video.ltx.ltx23.ia2v.run",
            return_value=fake_result,
        ) as mock_run:
            _ltx23()(
                mdir=str(tmp_path),
                prompt="hello world",
                image=image_obj,
                audio="/tmp/audio.mp3",
                w=640,
                h=480,
                n=65,
                f=25,
                c=1.5,
                seed=7,
            )

        mock_run.assert_called_once()
        kw = mock_run.call_args.kwargs
        assert kw["models_dir"] == str(tmp_path)
        assert kw["image"] is image_obj
        assert kw["audio_path"] == "/tmp/audio.mp3"
        assert kw["prompt"] == "hello world"
        assert kw["width"] == 640
        assert kw["height"] == 480
        assert kw["length"] == 65
        assert kw["fps"] == 25
        assert kw["cfg"] == 1.5
        assert kw["seed"] == 7

    def test_ia2v_returns_frames(self, tmp_path):
        """``_ltx23`` returns ``frames`` from the ia2v result dict (AC02)."""
        expected_frames = [MagicMock(), MagicMock()]
        fake_result = {"frames": expected_frames, "audio": {}}

        with patch(
            "comfy_diffusion.pipelines.video.ltx.ltx23.ia2v.run",
            return_value=fake_result,
        ):
            frames = _ltx23()(
                mdir=str(tmp_path),
                prompt="p",
                image=MagicMock(),
                audio="/tmp/x.wav",
                w=768,
                h=512,
                n=97,
                f=24,
                c=1.0,
                seed=0,
            )

        assert frames is expected_frames


# ---------------------------------------------------------------------------
# AC03 — i2v path unchanged when audio is None and image is set
# ---------------------------------------------------------------------------


class TestAC03I2vPathUnchanged:
    """AC03: audio=None + image set → existing i2v path invoked, not ia2v."""

    def test_i2v_invoked_not_ia2v(self, tmp_path):
        """i2v.run() is called when audio=None and image is provided (AC03)."""
        image_obj = MagicMock()
        fake_result = {"frames": [MagicMock()]}

        with patch(
            "comfy_diffusion.pipelines.video.ltx.ltx23.i2v.run",
            return_value=fake_result,
        ) as mock_i2v, patch(
            "comfy_diffusion.pipelines.video.ltx.ltx23.ia2v.run",
        ) as mock_ia2v:
            _ltx23()(
                mdir=str(tmp_path),
                prompt="p",
                image=image_obj,
                w=768,
                h=512,
                n=97,
                f=25,
                c=1.0,
                seed=0,
                audio=None,
            )

        mock_i2v.assert_called_once()
        mock_ia2v.assert_not_called()


# ---------------------------------------------------------------------------
# AC04 — t2v path unchanged when both audio and image are None
# ---------------------------------------------------------------------------


class TestAC04T2vPathUnchanged:
    """AC04: audio=None + image=None → existing t2v path invoked, not ia2v/i2v."""

    def test_t2v_invoked_not_ia2v(self, tmp_path):
        """t2v.run() is called when both audio and image are None (AC04)."""
        fake_result = {"frames": [MagicMock()]}

        with patch(
            "comfy_diffusion.pipelines.video.ltx.ltx23.t2v.run",
            return_value=fake_result,
        ) as mock_t2v, patch(
            "comfy_diffusion.pipelines.video.ltx.ltx23.ia2v.run",
        ) as mock_ia2v:
            _ltx23()(
                mdir=str(tmp_path),
                prompt="p",
                image=None,
                w=768,
                h=512,
                n=97,
                f=25,
                c=1.0,
                seed=0,
                audio=None,
            )

        mock_t2v.assert_called_once()
        mock_ia2v.assert_not_called()


# ---------------------------------------------------------------------------
# AC05 — RUNNERS["ltx23"] signature accepts audio=None and **_ absorbs extras
# ---------------------------------------------------------------------------


class TestAC05RunnersSignature:
    """AC05: RUNNERS dict entry for ltx23 handles audio=None and unknown kwargs."""

    def test_runners_dict_has_ltx23(self):
        """RUNNERS["ltx23"] key exists (AC05)."""
        assert "ltx23" in _runners()

    def test_runners_ltx23_accepts_audio_none(self, tmp_path):
        """RUNNERS["ltx23"] called with audio=None does not raise TypeError (AC05)."""
        fake_result = {"frames": [MagicMock()]}
        with patch(
            "comfy_diffusion.pipelines.video.ltx.ltx23.t2v.run",
            return_value=fake_result,
        ):
            result = _runners()["ltx23"](
                mdir=str(tmp_path),
                prompt="p",
                image=None,
                w=768,
                h=512,
                n=97,
                f=25,
                c=1.0,
                seed=0,
                audio=None,
            )
        assert isinstance(result, list)

    def test_runners_ltx23_absorbs_unknown_kwargs(self, tmp_path):
        """RUNNERS["ltx23"] absorbs unknown kwargs via **_ without TypeError (AC05)."""
        fake_result = {"frames": [MagicMock()]}
        with patch(
            "comfy_diffusion.pipelines.video.ltx.ltx23.t2v.run",
            return_value=fake_result,
        ):
            # extra_param is an unknown kwarg — must be silently absorbed by **_
            result = _runners()["ltx23"](
                mdir=str(tmp_path),
                prompt="p",
                image=None,
                w=768,
                h=512,
                n=97,
                f=25,
                c=1.0,
                seed=0,
                extra_param="ignored",
            )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# AC06 — Structural / import checks
# ---------------------------------------------------------------------------


class TestAC06Structural:
    """AC06: imports are clean and signature is correct."""

    def test_video_runner_module_imports_cleanly(self):
        """cli._runners.video imports without errors (AC06)."""
        import importlib
        mod = importlib.import_module("cli._runners.video")
        assert hasattr(mod, "_ltx23")
        assert hasattr(mod, "RUNNERS")

    def test_ltx23_runner_has_c_param(self):
        """``_ltx23`` signature includes ``c`` for cfg (AC06)."""
        sig = inspect.signature(_ltx23())
        assert "c" in sig.parameters

    def test_ia2v_module_exports_run(self):
        """``ia2v`` module has a callable ``run`` export (AC06)."""
        import importlib
        mod = importlib.import_module("comfy_diffusion.pipelines.video.ltx.ltx23.ia2v")
        assert callable(getattr(mod, "run", None))

    def test_ia2v_run_accepts_cfg(self):
        """``ia2v.run`` accepts a ``cfg`` keyword argument (AC06)."""
        import importlib
        mod = importlib.import_module("comfy_diffusion.pipelines.video.ltx.ltx23.ia2v")
        sig = inspect.signature(mod.run)
        assert "cfg" in sig.parameters
