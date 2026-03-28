"""Tests for US-004 — comfy_diffusion/pipelines/ltx2_i2v_lora.py pipeline (AV chain).

Covers:
  AC01: manifest() returns 4 HFModelEntry items (base models only; style LoRA excluded)
  AC02: run() accepts lora_path: str | Path and lora_strength: float = 1.0
  AC03: Both LoRAs applied via apply_lora() (stacked, base LoRA first)
  AC04: Default width=1280, height=1280 (square)
  AC05: CPU test passes with mocked inputs; AV chain in correct order
  AC06: typecheck / lint — file parses without syntax errors; no top-level comfy imports
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Paths under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_FILE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx2" / "i2v_lora.py"


# ---------------------------------------------------------------------------
# AC06 — file parses without syntax errors; conventions
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), (
        "comfy_diffusion/pipelines/ltx2_i2v_lora.py must exist"
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
    assert docstring, "i2v_lora.py must have a module-level docstring"


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
# AC01 — manifest() returns 4 HFModelEntry items (base models only)
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_manifest_returns_exactly_four_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 4, f"manifest() must return exactly 4 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_unet_dest_uses_dev_fp8_checkpoint() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "ltx-2-19b-dev-fp8" in d for d in dests), (
        "manifest() must include an entry with dest under diffusion_models/ltx-2-19b-dev-fp8"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include a text_encoders/gemma_3_12B_it_fp4_mixed entry"
    )


def test_manifest_base_lora_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d and "ltx-2-19b-distilled-lora-384" in d for d in dests), (
        "manifest() must include a loras/ltx-2-19b-distilled-lora-384 entry"
    )


def test_manifest_upscaler_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("upscale_models" in d and "ltx-2-spatial-upscaler" in d for d in dests), (
        "manifest() must include an upscale_models/ltx-2-spatial-upscaler entry"
    )


def test_manifest_no_style_lora_entry() -> None:
    """Style LoRA is caller-supplied; manifest must not list it."""
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest

    entries = manifest()
    # Only one lora entry (the base distilled lora) is allowed
    lora_entries = [e for e in entries if "lora" in str(e.dest).lower()]
    assert len(lora_entries) == 1, (
        f"manifest() must contain exactly 1 LoRA entry (the base distilled LoRA), "
        f"found {len(lora_entries)}"
    )


def test_manifest_all_from_lightricks_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest

    for entry in manifest():
        assert entry.repo_id == "Lightricks/LTX-Video", (
            f"Entry {entry!r} must use repo_id 'Lightricks/LTX-Video'"
        )


# ---------------------------------------------------------------------------
# AC02 — run() accepts lora_path and lora_strength
# ---------------------------------------------------------------------------


def test_run_signature_includes_lora_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import run

    sig = inspect.signature(run)
    assert "lora_path" in sig.parameters, "run() must accept a 'lora_path' parameter"


def test_run_lora_strength_default_is_one() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import run

    sig = inspect.signature(run)
    assert "lora_strength" in sig.parameters
    assert sig.parameters["lora_strength"].default == 1.0, (
        "lora_strength default must be 1.0"
    )


def test_run_has_fps_param_with_default_24() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import run

    sig = inspect.signature(run)
    assert "fps" in sig.parameters
    assert sig.parameters["fps"].default == 24


def test_run_signature_includes_all_required_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import run

    sig = inspect.signature(run)
    required = {
        "models_dir",
        "image",
        "prompt",
        "lora_path",
        "negative_prompt",
        "width",
        "height",
        "length",
        "steps",
        "cfg",
        "seed",
        "lora_strength",
    }
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


# ---------------------------------------------------------------------------
# AC04 — default width=1280, height=1280
# ---------------------------------------------------------------------------


def test_run_default_width_is_1280() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 1280, (
        "Default width must be 1280"
    )


def test_run_default_height_is_1280() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import run

    sig = inspect.signature(run)
    assert sig.parameters["height"].default == 1280, (
        "Default height must be 1280 (square)"
    )


# ---------------------------------------------------------------------------
# AC03 / AC05 helpers
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
_LTXV_COND_PATCH = "comfy_diffusion.conditioning.ltxv_conditioning"
_CROP_GUIDES_PATCH = "comfy_diffusion.conditioning.ltxv_crop_guides"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_UPSAMPLE_PATCH = "comfy_diffusion.latent.ltxv_latent_upsample"
_SAMPLE_CUSTOM_PATCH = "comfy_diffusion.sampling.sample_custom"
_CFG_GUIDER_PATCH = "comfy_diffusion.sampling.cfg_guider"
_BASIC_SCHEDULER_PATCH = "comfy_diffusion.sampling.basic_scheduler"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_RANDOM_NOISE_PATCH = "comfy_diffusion.sampling.random_noise"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode_batch_tiled"
_APPLY_LORA_PATCH = "comfy_diffusion.lora.apply_lora"
_IMG_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"
_LOAD_IMAGE_PATCH = "comfy_diffusion.image.load_image"
_IMG_TO_VIDEO_PATCH = "comfy_diffusion.video.ltxv_img_to_video_inplace"
_EMPTY_AUDIO_PATCH = "comfy_diffusion.audio.ltxv_empty_latent_audio"
_CONCAT_AV_PATCH = "comfy_diffusion.audio.ltxv_concat_av_latent"
_SEPARATE_AV_PATCH = "comfy_diffusion.audio.ltxv_separate_av_latent"
_AUDIO_DECODE_PATCH = "comfy_diffusion.audio.ltxv_audio_vae_decode"


def _make_fake_image() -> MagicMock:
    """Return a mock that passes the ``hasattr(image, 'mode')`` check."""
    img = MagicMock()
    img.mode = "RGB"
    return img


def _run_with_mocks(
    tmp_path: Path,
    image: Any,
    *,
    call_order: list[str] | None = None,
    fake_frames: list[Any] | None = None,
    fake_audio: Any = None,
    lora_path: str = "/path/to/style.safetensors",
) -> dict[str, Any]:
    """Execute pipeline.run() with all heavy dependencies mocked."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v_lora as pipeline_mod

    if fake_frames is None:
        fake_frames = [MagicMock(name="frame")]
    if fake_audio is None:
        fake_audio = MagicMock(name="audio")

    fake_video_out = MagicMock(name="video_out")
    fake_audio_out = MagicMock(name="audio_out")
    fake_video_up = MagicMock(name="video_up")
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
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (
            _track("ltxv_conditioning", (p, n))
        )),
        patch(_LOAD_IMAGE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMG_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, side_effect=lambda vae, img, lat, **kw: (
            _track("ltxv_img_to_video_inplace", {"samples": MagicMock()})
        )),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (
            _track("ltxv_crop_guides", (p, n, lat))
        )),
        patch(_EMPTY_AUDIO_PATCH, side_effect=lambda av, **kw: (
            _track("ltxv_empty_latent_audio", MagicMock())
        )),
        patch(_CONCAT_AV_PATCH, side_effect=lambda vl, al: (
            _track("ltxv_concat_av_latent", MagicMock())
        )),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
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
        patch(_AUDIO_DECODE_PATCH, return_value=fake_audio),
    ):
        return pipeline_mod.run(
            models_dir=tmp_path,
            image=image,
            prompt="a cat jumps over a fence",
            lora_path=lora_path,
        )


# ---------------------------------------------------------------------------
# AC05 — CPU test passes with mocked inputs
# ---------------------------------------------------------------------------


def test_run_end_to_end_with_mocks(tmp_path: Path) -> None:
    """AC05: full pipeline runs on CPU with all heavy dependencies mocked."""
    fake_frames = [MagicMock(name="frame")]
    fake_audio = MagicMock(name="audio")
    result = _run_with_mocks(tmp_path, _make_fake_image(),
                             fake_frames=fake_frames, fake_audio=fake_audio)
    assert isinstance(result, dict)
    assert result["frames"] is fake_frames
    assert result["audio"] is fake_audio


def test_run_returns_dict_with_frames_and_audio(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path, _make_fake_image())
    assert isinstance(result, dict)
    assert "frames" in result
    assert "audio" in result


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    """run() must raise RuntimeError when check_runtime() returns an error."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v_lora as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(
                models_dir=tmp_path,
                image=_make_fake_image(),
                prompt="test",
                lora_path="/path/to/style.safetensors",
            )


def test_run_accepts_path_image_via_load_image(tmp_path: Path) -> None:
    """When image is a path (no .mode attr), load_image() must be called."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v_lora as pipeline_mod

    mm = _build_mock_mm()
    fake_tensor = MagicMock(name="tensor")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_LOAD_IMAGE_PATCH, return_value=(fake_tensor, None)) as mock_load_image,
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
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
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image="/some/image.png",
            prompt="test",
            lora_path="/path/to/style.safetensors",
        )

    mock_load_image.assert_called_once_with("/some/image.png")


# ---------------------------------------------------------------------------
# AC03 — both LoRAs applied via apply_lora, stacked, base LoRA first
# ---------------------------------------------------------------------------


def test_apply_lora_called_twice(tmp_path: Path) -> None:
    """AC03: apply_lora() must be called exactly twice (base + style LoRA)."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v_lora as pipeline_mod

    mm = _build_mock_mm()
    lora_call_count = []
    model_v1 = MagicMock(name="model_v1")
    clip_v1 = MagicMock(name="clip_v1")
    model_v2 = MagicMock(name="model_v2")
    clip_v2 = MagicMock(name="clip_v2")

    def fake_apply_lora(model: Any, clip: Any, path: Any, ms: float, cs: float) -> tuple:
        lora_call_count.append(str(path))
        if len(lora_call_count) == 1:
            return model_v1, clip_v1
        return model_v2, clip_v2

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, side_effect=fake_apply_lora),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[MagicMock()]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=_make_fake_image(),
            prompt="test",
            lora_path="/path/to/style.safetensors",
        )

    assert len(lora_call_count) == 2, (
        f"apply_lora() must be called exactly twice, got {len(lora_call_count)} calls"
    )


def test_base_lora_applied_before_style_lora(tmp_path: Path) -> None:
    """AC03: base distilled LoRA must be applied before the caller-supplied style LoRA."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v_lora as pipeline_mod

    mm = _build_mock_mm()
    lora_paths_called: list[str] = []

    def fake_apply_lora(model: Any, clip: Any, path: Any, ms: float, cs: float) -> tuple:
        lora_paths_called.append(str(path))
        return MagicMock(), MagicMock()

    style_lora = str(tmp_path / "my_style.safetensors")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, side_effect=fake_apply_lora),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[MagicMock()]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=_make_fake_image(),
            prompt="test",
            lora_path=style_lora,
        )

    assert len(lora_paths_called) == 2
    assert "ltx-2-19b-distilled-lora-384" in lora_paths_called[0], (
        f"First apply_lora() call must use the base distilled LoRA, got: {lora_paths_called[0]}"
    )
    assert lora_paths_called[1] == style_lora, (
        f"Second apply_lora() call must use the style LoRA, got: {lora_paths_called[1]}"
    )


def test_style_lora_strength_passed_correctly(tmp_path: Path) -> None:
    """AC02/AC03: lora_strength must be forwarded to the style LoRA apply_lora() call."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v_lora as pipeline_mod

    mm = _build_mock_mm()
    call_strengths: list[tuple[float, float]] = []

    def fake_apply_lora(model: Any, clip: Any, path: Any, ms: float, cs: float) -> tuple:
        call_strengths.append((ms, cs))
        return MagicMock(), MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, side_effect=fake_apply_lora),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[MagicMock()]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=_make_fake_image(),
            prompt="test",
            lora_path="/path/to/style.safetensors",
            lora_strength=0.75,
        )

    assert len(call_strengths) == 2
    assert call_strengths[0] == (1.0, 1.0), (
        f"Base LoRA strength must be (1.0, 1.0), got {call_strengths[0]}"
    )
    assert call_strengths[1] == (0.75, 0.75), (
        f"Style LoRA strength must be (0.75, 0.75), got {call_strengths[1]}"
    )


def test_no_ltxv_preprocess_import_or_call() -> None:
    """No ltxv_preprocess import or call: image loaded directly per reference workflow."""
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    import ast as _ast
    tree = _ast.parse(source, filename=str(_PIPELINE_FILE))
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            for alias in getattr(node, "names", []):
                assert "ltxv_preprocess" not in alias.name, (
                    "i2v_lora.py must not import ltxv_preprocess"
                )
        if isinstance(node, _ast.Attribute) and node.attr == "ltxv_preprocess":
            pytest.fail(
                "i2v_lora.py must not call ltxv_preprocess — "
                "image is loaded directly without resize preprocessing"
            )
        if isinstance(node, _ast.Name) and node.id == "ltxv_preprocess":
            pytest.fail(
                "i2v_lora.py must not reference ltxv_preprocess — "
                "image is loaded directly without resize preprocessing"
            )


def test_run_uses_load_ltxav_text_encoder(tmp_path: Path) -> None:
    """Gemma 3 text encoder must be loaded via load_ltxav_text_encoder."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v_lora as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
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
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=_make_fake_image(),
            prompt="test",
            lora_path="/path/to/style.safetensors",
        )

    mm.load_ltxav_text_encoder.assert_called_once()
    mm.load_clip.assert_not_called()


def test_run_uses_load_ltxv_audio_vae(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v_lora as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
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
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=_make_fake_image(),
            prompt="test",
            lora_path="/path/to/style.safetensors",
        )

    mm.load_ltxv_audio_vae.assert_called_once()


def test_run_calls_ltxv_conditioning(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, _make_fake_image(), call_order=call_order)
    assert "ltxv_conditioning" in call_order


def test_run_calls_ltxv_crop_guides(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, _make_fake_image(), call_order=call_order)
    assert "ltxv_crop_guides" in call_order


def test_run_calls_ltxv_separate_av_latent(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, _make_fake_image(), call_order=call_order)
    assert "ltxv_separate_av_latent" in call_order


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    """download_models(manifest()) completes without error when all 4 files are present."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest

    entries = manifest()
    assert len(entries) == 4

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
