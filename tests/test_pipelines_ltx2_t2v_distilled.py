"""Tests for comfy_diffusion/pipelines/video/ltx/ltx2/t2v_distilled.py pipeline.

Covers:
  AC01: manifest() returns 3 HFModelEntry items with correct dest paths
  AC02: run() accepts models_dir, prompt, negative_prompt, width, height, fps, etc.
  AC03: pipeline calls AV chain (ltxv_conditioning, ltxv_concat_av_latent, sample_custom,
        ltxv_separate_av_latent, ltxv_latent_upsample) in correct order
  AC04: CPU test passes with mocked ModelManager and no GPU
  AC05: typecheck / lint — file parses without syntax errors; no top-level comfy imports
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Paths under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINES_PKG = _REPO_ROOT / "comfy_diffusion" / "pipelines"
_PIPELINE_FILE = _PIPELINES_PKG / "video" / "ltx" / "ltx2" / "t2v_distilled.py"


# ---------------------------------------------------------------------------
# AC05 — file parses without syntax errors; conventions
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), (
        "comfy_diffusion/pipelines/video/ltx/ltx2/t2v_distilled.py must exist"
    )


def test_pipeline_parses_without_syntax_errors() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PIPELINE_FILE))
    assert isinstance(tree, ast.Module)


def test_pipeline_has_future_annotations() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert "from __future__ import annotations" in source


def test_pipeline_has_module_docstring() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PIPELINE_FILE))
    docstring = ast.get_docstring(tree)
    assert docstring, "t2v_distilled.py must have a module-level docstring"


def test_pipeline_exports_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '"manifest"' in source or "'manifest'" in source
    assert '"run"' in source or "'run'" in source


def test_no_top_level_comfy_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), (
                f"Top-level comfy import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


# ---------------------------------------------------------------------------
# AC01 — manifest() returns 3 HFModelEntry items with correct dest paths
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_manifest_returns_exactly_three_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 3, f"manifest() must return exactly 3 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any("diffusion_models" in d and "ltx-2-19b-distilled" in d for d in dests), (
        "manifest() must include an entry with dest under diffusion_models/"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include a text_encoders/gemma_3_12B_it_fp4_mixed entry"
    )


def test_manifest_upscaler_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any("upscale_models" in d and "ltx-2-spatial-upscaler" in d for d in dests), (
        "manifest() must include an upscale_models/ltx-2-spatial-upscaler entry"
    )


def test_manifest_all_from_lightricks_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest

    for entry in manifest():
        assert entry.repo_id == "Lightricks/LTX-Video", (
            f"Entry {entry!r} must use repo_id 'Lightricks/LTX-Video'"
        )


def test_manifest_entries_have_nonempty_dest() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest

    for entry in manifest():
        assert entry.dest, f"manifest() entry {entry!r} must have a non-empty dest"


# ---------------------------------------------------------------------------
# AC02 — run() accepts required parameters
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    import inspect
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import run

    sig = inspect.signature(run)
    params = set(sig.parameters.keys())
    required = {"models_dir", "prompt", "negative_prompt", "width", "height"}
    assert required <= params, f"run() is missing parameters: {required - params}"


def test_run_default_steps_is_eight() -> None:
    import inspect
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 8, (
        "run() default steps must be 8 for the distilled model"
    )


def test_run_has_fps_param_with_default_24() -> None:
    import inspect
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import run

    sig = inspect.signature(run)
    assert "fps" in sig.parameters
    assert sig.parameters["fps"].default == 24


def test_run_has_unet_filename_override() -> None:
    import inspect
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import run

    sig = inspect.signature(run)
    assert "unet_filename" in sig.parameters
    assert "text_encoder_filename" in sig.parameters
    assert "upscaler_filename" in sig.parameters


# ---------------------------------------------------------------------------
# AC03 + AC04 — AV pipeline order and CPU test with mocked ModelManager
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    """Return a ModelManager mock with sensible return values."""
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_ltxv_audio_vae.return_value = MagicMock(name="audio_vae")
    mm.load_ltxav_text_encoder.return_value = MagicMock(name="clip")
    mm.load_latent_upscale_model.return_value = MagicMock(name="upscale_model")
    return mm


_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_CONDITIONING_PATCH = "comfy_diffusion.conditioning.ltxv_conditioning"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_UPSAMPLE_PATCH = "comfy_diffusion.latent.ltxv_latent_upsample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode_batch_tiled"
_SAMPLE_CUSTOM_PATCH = "comfy_diffusion.sampling.sample_custom"
_EMPTY_AUDIO_PATCH = "comfy_diffusion.audio.ltxv_empty_latent_audio"
_CONCAT_AV_PATCH = "comfy_diffusion.audio.ltxv_concat_av_latent"
_SEPARATE_AV_PATCH = "comfy_diffusion.audio.ltxv_separate_av_latent"
_AUDIO_DECODE_PATCH = "comfy_diffusion.audio.ltxv_audio_vae_decode"
_MANUAL_SIGMAS_PATCH = "comfy_diffusion.sampling.manual_sigmas"
_CFG_GUIDER_PATCH = "comfy_diffusion.sampling.cfg_guider"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_RANDOM_NOISE_PATCH = "comfy_diffusion.sampling.random_noise"


def test_run_calls_ltxv_latent_upsample_before_vae_decode(
    tmp_path: Path,
) -> None:
    """AC03 + AC04: pipeline calls AV chain in the correct order."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v_distilled as pipeline_mod

    call_order: list[str] = []
    fake_frames = [MagicMock(name="frame")]
    fake_audio = MagicMock(name="audio")
    fake_video_out = MagicMock(name="video_out")
    fake_audio_out = MagicMock(name="audio_out")
    fake_video_up = MagicMock(name="video_up")

    mm = _build_mock_mm()

    def _track(name: str, rv: Any) -> Any:
        call_order.append(name)
        return rv

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_CONDITIONING_PATCH, side_effect=lambda p, n, **kw: (
            _track("ltxv_conditioning", (p, n))
        )),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, side_effect=lambda av, **kw: (
            _track("ltxv_empty_latent_audio", MagicMock())
        )),
        patch(_CONCAT_AV_PATCH, side_effect=lambda vl, al: (
            _track("ltxv_concat_av_latent", MagicMock())
        )),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, side_effect=lambda n, g, s, sig, lat: (
            _track("sample_custom", (MagicMock(), MagicMock()))
        )),
        patch(_SEPARATE_AV_PATCH, side_effect=lambda d: (
            _track("ltxv_separate_av_latent", (fake_video_out, fake_audio_out))
        )),
        patch(_UPSAMPLE_PATCH, side_effect=lambda s, **kw: (
            _track("ltxv_latent_upsample", fake_video_up)
        )),
        patch(_VAE_DECODE_PATCH, side_effect=lambda v, s: (
            _track("vae_decode_batch_tiled", fake_frames)
        )),
        patch(_AUDIO_DECODE_PATCH, side_effect=lambda av, al: (
            _track("ltxv_audio_vae_decode", fake_audio)
        )),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            prompt="a golden retriever running through a sunlit park",
        )

    expected_order = [
        "ltxv_conditioning",
        "ltxv_empty_latent_audio",
        "ltxv_concat_av_latent",
        "sample_custom",
        "ltxv_separate_av_latent",
        "ltxv_latent_upsample",
        "vae_decode_batch_tiled",
        "ltxv_audio_vae_decode",
    ]
    assert call_order == expected_order, (
        f"Expected {expected_order}, got {call_order}"
    )
    assert result == {"frames": fake_frames, "audio": fake_audio}


def test_run_passes_upsampled_samples_to_vae_decode(
    tmp_path: Path,
) -> None:
    """Upsampled latent is passed to vae_decode_batch_tiled."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v_distilled as pipeline_mod

    fake_video_out = MagicMock(name="video_out")
    fake_video_up = MagicMock(name="upsampled")
    received_for_decode: list[Any] = []

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_CONDITIONING_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(fake_video_out, MagicMock())),
        patch(_UPSAMPLE_PATCH, side_effect=lambda s, **kw: fake_video_up),
        patch(_VAE_DECODE_PATCH, side_effect=lambda v, s: received_for_decode.append(s) or []),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test prompt")

    assert len(received_for_decode) == 1
    assert received_for_decode[0] is fake_video_up, (
        "vae_decode_batch_tiled must receive the upsampled latent"
    )


def test_run_returns_dict_with_frames_and_audio(tmp_path: Path) -> None:
    """run() returns dict with 'frames' and 'audio' keys."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v_distilled as pipeline_mod

    fake_frames = [MagicMock(name="frame")]
    fake_audio = MagicMock(name="audio")
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_CONDITIONING_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=fake_frames),
        patch(_AUDIO_DECODE_PATCH, return_value=fake_audio),
    ):
        result = pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert isinstance(result, dict)
    assert result["frames"] is fake_frames
    assert result["audio"] is fake_audio


def test_run_calls_ltxv_conditioning(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v_distilled as pipeline_mod

    mm = _build_mock_mm()
    cond_called = []

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_CONDITIONING_PATCH, side_effect=lambda p, n, **kw: cond_called.append(True) or (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert cond_called, "ltxv_conditioning must be called"


def test_run_calls_sample_custom_not_sample(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v_distilled as pipeline_mod

    mm = _build_mock_mm()
    sample_custom_called = []

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_CONDITIONING_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, side_effect=lambda n, g, s, sig, lat: (
            sample_custom_called.append(True) or (MagicMock(), MagicMock())
        )),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert sample_custom_called, "sample_custom must be called"


def test_run_calls_ltxv_separate_av_latent(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v_distilled as pipeline_mod

    mm = _build_mock_mm()
    sep_called = []

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_CONDITIONING_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, side_effect=lambda d: (
            sep_called.append(True) or (MagicMock(), MagicMock())
        )),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert sep_called, "ltxv_separate_av_latent must be called"


def test_run_uses_load_ltxv_audio_vae(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v_distilled as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_CONDITIONING_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_ltxv_audio_vae.assert_called_once()


def test_run_uses_load_ltxav_text_encoder_not_load_clip(
    tmp_path: Path,
) -> None:
    """Gemma 3 text encoder must be loaded via load_ltxav_text_encoder."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v_distilled as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_CONDITIONING_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_ltxav_text_encoder.assert_called_once()
    mm.load_clip.assert_not_called()


def test_run_uses_load_latent_upscale_model(
    tmp_path: Path,
) -> None:
    """Upscale model must be loaded via load_latent_upscale_model."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v_distilled as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_CONDITIONING_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_latent_upscale_model.assert_called_once()


def test_run_raises_on_runtime_error(
    tmp_path: Path,
) -> None:
    """run() must raise RuntimeError when check_runtime() returns an error."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v_distilled as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    """download_models(manifest()) completes without error when all 3 files are present."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest

    entries = manifest()
    assert len(entries) == 3

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
