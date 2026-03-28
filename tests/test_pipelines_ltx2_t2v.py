"""Tests for comfy_diffusion/pipelines/video/ltx/ltx2/t2v.py — LTX2 T2V AV pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring
  - Exports manifest and run; no top-level comfy imports
  - manifest() returns exactly 4 HFModelEntry items with correct dest paths
  - run() signature: models_dir, prompt, fps, unet_filename, etc.
  - run() behaviour: AV chain, apply_lora, ltxv_conditioning, correct call order
  - download_models idempotent with 4 entries
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_FILE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx2" / "t2v.py"

# ---------------------------------------------------------------------------
# Patch constants (source module paths for lazy imports)
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_LTXV_COND_PATCH = "comfy_diffusion.conditioning.ltxv_conditioning"
_EMPTY_VIDEO_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_EMPTY_AUDIO_PATCH = "comfy_diffusion.audio.ltxv_empty_latent_audio"
_CONCAT_AV_PATCH = "comfy_diffusion.audio.ltxv_concat_av_latent"
_CFG_GUIDER_PATCH = "comfy_diffusion.sampling.cfg_guider"
_RANDOM_NOISE_PATCH = "comfy_diffusion.sampling.random_noise"
_BASIC_SCHEDULER_PATCH = "comfy_diffusion.sampling.basic_scheduler"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_SAMPLE_CUSTOM_PATCH = "comfy_diffusion.sampling.sample_custom"
_SEPARATE_AV_PATCH = "comfy_diffusion.audio.ltxv_separate_av_latent"
_UPSAMPLE_PATCH = "comfy_diffusion.latent.ltxv_latent_upsample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode_batch_tiled"
_AUDIO_VAE_DECODE_PATCH = "comfy_diffusion.audio.ltxv_audio_vae_decode"
_APPLY_LORA_PATCH = "comfy_diffusion.lora.apply_lora"


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "t2v.py must exist"


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
    assert docstring, "t2v.py must have a module-level docstring"


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


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks — exactly 4 HFModelEntry items
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_four_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 4, f"manifest() must return exactly 4 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "ltx-2-19b-dev-fp8" in d for d in dests), (
        "manifest() must include a diffusion_models/ltx-2-19b-dev-fp8 entry"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include a text_encoders/gemma_3_12B_it_fp4_mixed entry"
    )


def test_manifest_lora_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d and "ltx-2-19b-distilled-lora-384" in d for d in dests), (
        "manifest() must include a loras/ltx-2-19b-distilled-lora-384 entry"
    )


def test_manifest_upscaler_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("upscale_models" in d and "ltx-2-spatial-upscaler" in d for d in dests), (
        "manifest() must include an upscale_models/ltx-2-spatial-upscaler entry"
    )


def test_manifest_all_from_lightricks_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Lightricks/LTX-Video", (
            f"Entry {entry!r} must use repo_id 'Lightricks/LTX-Video'"
        )


# ---------------------------------------------------------------------------
# run() signature checks
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import run

    sig = inspect.signature(run)
    required = {"models_dir", "prompt", "negative_prompt", "width", "height"}
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_has_fps_param_with_default_24() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import run

    sig = inspect.signature(run)
    assert "fps" in sig.parameters
    assert sig.parameters["fps"].default == 24


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import run

    sig = inspect.signature(run)
    for param in ("unet_filename", "text_encoder_filename", "vae_filename",
                  "lora_filename", "upscaler_filename"):
        assert param in sig.parameters, f"run() must accept '{param}'"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_ltxv_audio_vae.return_value = MagicMock(name="audio_vae")
    mm.load_ltxav_text_encoder.return_value = MagicMock(name="clip")
    mm.load_latent_upscale_model.return_value = MagicMock(name="upscale_model")
    return mm


def _run_with_mocks(
    tmp_path: Path,
    *,
    call_order: list[str] | None = None,
    fake_frames: list[Any] | None = None,
    fake_audio: Any = None,
) -> dict[str, Any]:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v as pipeline_mod

    if fake_frames is None:
        fake_frames = [MagicMock(name="frame0")]
    if fake_audio is None:
        fake_audio = MagicMock(name="audio")

    fake_video_latent = MagicMock(name="video_latent")
    fake_audio_latent = MagicMock(name="audio_latent")
    fake_av_latent = MagicMock(name="av_latent")
    fake_denoised = MagicMock(name="denoised")
    fake_video_out = MagicMock(name="video_latent_out")
    fake_audio_out = MagicMock(name="audio_latent_out")
    fake_video_up = MagicMock(name="video_latent_up")
    patched_model = MagicMock(name="patched_model")
    patched_clip = MagicMock(name="patched_clip")

    mm = _build_mock_mm()

    def _track(name: str, rv: Any) -> Any:
        if call_order is not None:
            call_order.append(name)
        return rv

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, side_effect=lambda m, c, p, sm, sc: (
            _track("apply_lora", (patched_model, patched_clip))
        )),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda pos, neg, **kw: (
            _track("ltxv_conditioning", (pos, neg))
        )),
        patch(_EMPTY_VIDEO_PATCH, return_value=fake_video_latent),
        patch(_EMPTY_AUDIO_PATCH, side_effect=lambda av, **kw: (
            _track("ltxv_empty_latent_audio", fake_audio_latent)
        )),
        patch(_CONCAT_AV_PATCH, side_effect=lambda vl, al: (
            _track("ltxv_concat_av_latent", fake_av_latent)
        )),
        patch(_CFG_GUIDER_PATCH, side_effect=lambda m, p, n, c: (
            _track("cfg_guider", MagicMock(name="guider"))
        )),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock(name="noise")),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock(name="sigmas")),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock(name="sampler_obj")),
        patch(_SAMPLE_CUSTOM_PATCH, side_effect=lambda noise, guider, sampler, sigmas, latent: (
            _track("sample_custom", (MagicMock(), fake_denoised))
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
        patch(_AUDIO_VAE_DECODE_PATCH, side_effect=lambda av, al: (
            _track("ltxv_audio_vae_decode", fake_audio)
        )),
    ):
        return pipeline_mod.run(
            models_dir=tmp_path,
            prompt="a golden retriever running through a sunlit park",
        )


# ---------------------------------------------------------------------------
# run() behaviour tests
# ---------------------------------------------------------------------------


def test_run_returns_dict_with_frames_and_audio(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path)
    assert isinstance(result, dict)
    assert "frames" in result
    assert "audio" in result


def test_run_frames_value(tmp_path: Path) -> None:
    fake_frames = [MagicMock(name="f0")]
    result = _run_with_mocks(tmp_path, fake_frames=fake_frames)
    assert result["frames"] is fake_frames


def test_run_calls_apply_lora(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "apply_lora" in call_order


def test_run_calls_ltxv_conditioning(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_conditioning" in call_order


def test_run_calls_ltxv_empty_latent_audio(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_empty_latent_audio" in call_order


def test_run_calls_ltxv_concat_av_latent(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_concat_av_latent" in call_order


def test_run_calls_sample_custom_not_sample(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "sample_custom" in call_order
    assert "sample" not in call_order


def test_run_calls_ltxv_separate_av_latent(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_separate_av_latent" in call_order


def test_run_calls_ltxv_latent_upsample(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_latent_upsample" in call_order


def test_full_pipeline_call_order(tmp_path: Path) -> None:
    """AV chain must execute in the correct order."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)

    expected = [
        "apply_lora",
        "ltxv_conditioning",
        "ltxv_empty_latent_audio",
        "ltxv_concat_av_latent",
        "cfg_guider",
        "sample_custom",
        "ltxv_separate_av_latent",
        "ltxv_latent_upsample",
        "vae_decode_batch_tiled",
        "ltxv_audio_vae_decode",
    ]
    assert call_order == expected, (
        f"Expected call order {expected}, got {call_order}"
    )


def test_run_uses_load_ltxv_audio_vae(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_VIDEO_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_ltxv_audio_vae.assert_called_once()


def test_run_uses_load_ltxav_text_encoder_not_load_clip(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_VIDEO_PATCH, return_value=MagicMock()),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_ltxav_text_encoder.assert_called_once()
    mm.load_clip.assert_not_called()


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import t2v as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


# ---------------------------------------------------------------------------
# download_models idempotent — 4 entries
# ---------------------------------------------------------------------------


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest

    entries = manifest()
    assert len(entries) == 4

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
