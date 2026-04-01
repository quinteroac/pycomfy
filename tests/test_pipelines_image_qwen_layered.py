"""Tests for comfy_diffusion/pipelines/image/qwen/layered.py — Qwen Image Layered pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring              (AC08)
  - __all__ = ["manifest", "run_t2l", "run_i2l"]                              (AC08)
  - No top-level comfy/torch imports (lazy import pattern)                     (AC08)
  - manifest() returns exactly 3 HFModelEntry items with correct dest paths   (AC01)
  - run_t2l() and run_i2l() have correct signatures with expected defaults     (AC02-AC04)
  - run_t2l() returns list[PIL.Image.Image]                                    (AC02, AC06)
  - run_i2l() returns list[PIL.Image.Image]                                    (AC03, AC06)
  - Both check_runtime() and raise RuntimeError on failure                     (AC05)
  - run_t2l() node execution order: load_unet, load_clip, load_vae,
    model_sampling_aura_flow, encode_prompt, empty_qwen_image_layered,
    sample, latent_cut per layer, vae_decode per layer                         (AC02)
  - run_i2l() node execution order: same + image_scale_to_max_dimension,
    get_image_size, vae_encode_tensor, reference_latent                        (AC03)
  - Default steps=20, cfg=2.5, layers=2, sampler=euler, scheduler=simple      (AC04)
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
_PIPELINE_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "qwen" / "layered.py"
)

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_AURA_FLOW_PATCH = "comfy_diffusion.models.model_sampling_aura_flow"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_REF_LATENT_PATCH = "comfy_diffusion.conditioning.reference_latent"
_EMPTY_QWEN_PATCH = "comfy_diffusion.latent.empty_qwen_image_layered_latent_image"
_LATENT_CUT_PATCH = "comfy_diffusion.latent.latent_cut"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"
_VAE_ENCODE_TENSOR_PATCH = "comfy_diffusion.vae.vae_encode_tensor"
_IMAGE_SCALE_PATCH = "comfy_diffusion.image.image_scale_to_max_dimension"
_GET_IMAGE_SIZE_PATCH = "comfy_diffusion.image.get_image_size"
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"


# ---------------------------------------------------------------------------
# File-level checks (AC08)
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "layered.py must exist under pipelines/image/qwen/"


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
    assert docstring, "layered.py must have a module-level docstring"


def test_pipeline_has_dunder_all_with_manifest_run_t2l_run_i2l() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert "__all__" in source
    assert '"manifest"' in source or "'manifest'" in source
    assert '"run_t2l"' in source or "'run_t2l'" in source
    assert '"run_i2l"' in source or "'run_i2l'" in source


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.qwen import layered

    assert hasattr(layered, "__all__")
    assert set(layered.__all__) == {"manifest", "run_t2l", "run_i2l"}


def test_no_top_level_comfy_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), (
                f"Top-level comfy import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


def test_no_top_level_torch_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            assert line.startswith("    "), (
                f"Top-level torch import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


def test_import_manifest_run_t2l_run_i2l() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import manifest, run_i2l, run_t2l  # noqa: F401

    assert callable(manifest)
    assert callable(run_t2l)
    assert callable(run_i2l)


# ---------------------------------------------------------------------------
# Manifest checks (AC01)
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_three_entries() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 3, f"manifest() must return exactly 3 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.qwen.layered import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry must be an HFModelEntry, got {type(entry)!r}"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "qwen_image_layered_bf16" in d for d in dests), (
        f"No unet dest matching diffusion_models/qwen_image_layered_bf16.safetensors in {dests}"
    )


def test_manifest_clip_dest_path() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "qwen_2.5_vl_7b_fp8_scaled" in d for d in dests), (
        f"No text_encoder dest matching text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors in {dests}"
    )


def test_manifest_vae_dest_path() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("vae" in d and "qwen_image_layered_vae" in d for d in dests), (
        f"No vae dest matching vae/qwen_image_layered_vae.safetensors in {dests}"
    )


# ---------------------------------------------------------------------------
# Signature checks (AC02, AC03, AC04)
# ---------------------------------------------------------------------------


def test_run_t2l_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import run_t2l

    sig = inspect.signature(run_t2l)
    required = {"prompt", "width", "height", "layers", "steps", "cfg", "seed", "models_dir"}
    assert required <= set(sig.parameters.keys()), (
        f"run_t2l() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_i2l_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import run_i2l

    sig = inspect.signature(run_i2l)
    required = {"prompt", "image", "layers", "steps", "cfg", "seed", "models_dir"}
    assert required <= set(sig.parameters.keys()), (
        f"run_i2l() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_t2l_default_steps_20() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import run_t2l

    sig = inspect.signature(run_t2l)
    assert sig.parameters["steps"].default == 20


def test_run_t2l_default_cfg_2_5() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import run_t2l

    sig = inspect.signature(run_t2l)
    assert sig.parameters["cfg"].default == 2.5


def test_run_t2l_default_layers_2() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import run_t2l

    sig = inspect.signature(run_t2l)
    assert sig.parameters["layers"].default == 2


def test_run_i2l_default_steps_20() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import run_i2l

    sig = inspect.signature(run_i2l)
    assert sig.parameters["steps"].default == 20


def test_run_i2l_default_cfg_2_5() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import run_i2l

    sig = inspect.signature(run_i2l)
    assert sig.parameters["cfg"].default == 2.5


def test_run_i2l_default_layers_2() -> None:
    from comfy_diffusion.pipelines.image.qwen.layered import run_i2l

    sig = inspect.signature(run_i2l)
    assert sig.parameters["layers"].default == 2


def test_pipeline_uses_euler_sampler() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '"euler"' in source or "'euler'" in source


def test_pipeline_uses_simple_scheduler() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '"simple"' in source or "'simple'" in source


# ---------------------------------------------------------------------------
# Helpers for run tests
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    return mm


# ---------------------------------------------------------------------------
# run_t2l() behaviour tests (AC02, AC04, AC05, AC06)
# ---------------------------------------------------------------------------


def test_run_t2l_raises_on_runtime_error(tmp_path: Path) -> None:
    """run_t2l() must raise RuntimeError when check_runtime() returns an error (AC05)."""
    from comfy_diffusion.pipelines.image.qwen.layered import run_t2l

    with patch(_RUNTIME_PATCH, return_value={"error": "ComfyUI not found", "python_version": "3.12.0"}):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            run_t2l(prompt="test", models_dir=tmp_path)


def test_run_t2l_returns_list_of_pil_images(tmp_path: Path) -> None:
    """run_t2l() must return a list[PIL.Image.Image] (AC06)."""
    from PIL import Image as PILImage

    fake_image = MagicMock(spec=PILImage.Image)
    layers = 2
    latent_cut_mock = MagicMock(return_value=MagicMock(name="layer_latent"))

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_QWEN_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_LATENT_CUT_PATCH, latent_cut_mock),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        from comfy_diffusion.pipelines.image.qwen import layered

        result = layered.run_t2l(
            prompt="a test prompt",
            width=640,
            height=640,
            layers=layers,
            models_dir=tmp_path,
        )

    assert isinstance(result, list)
    assert len(result) == layers
    for img in result:
        assert img is fake_image


def test_run_t2l_calls_aura_flow_shift_1(tmp_path: Path) -> None:
    """run_t2l() must call model_sampling_aura_flow with shift=1 (AC02)."""
    aura_mock = MagicMock(return_value=MagicMock(name="patched_model"))

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_AURA_FLOW_PATCH, aura_mock),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_QWEN_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_LATENT_CUT_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.qwen import layered

        layered.run_t2l(prompt="test", models_dir=tmp_path)

    aura_mock.assert_called_once()
    _, kwargs = aura_mock.call_args
    shift_val = kwargs.get("shift", aura_mock.call_args[0][1] if len(aura_mock.call_args[0]) > 1 else None)
    assert shift_val == 1, f"model_sampling_aura_flow shift must be 1, got {shift_val}"


def test_run_t2l_calls_sample_with_euler_simple(tmp_path: Path) -> None:
    """run_t2l() must call sample() with 'euler' sampler and 'simple' scheduler (AC04)."""
    sample_mock = MagicMock(return_value=MagicMock(name="latent_out"))

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_QWEN_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, sample_mock),
        patch(_LATENT_CUT_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.qwen import layered

        layered.run_t2l(prompt="test", steps=20, cfg=2.5, seed=0, models_dir=tmp_path)

    sample_mock.assert_called_once()
    args = sample_mock.call_args[0]
    # sample(model, positive, negative, latent, steps, cfg, sampler_name, scheduler, seed)
    assert args[6] == "euler", f"sampler_name must be 'euler', got {args[6]!r}"
    assert args[7] == "simple", f"scheduler must be 'simple', got {args[7]!r}"


def test_run_t2l_calls_latent_cut_per_layer(tmp_path: Path) -> None:
    """run_t2l() must call latent_cut once per layer (AC02)."""
    latent_cut_mock = MagicMock(return_value=MagicMock(name="layer_latent"))
    layers = 3

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_QWEN_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_LATENT_CUT_PATCH, latent_cut_mock),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.qwen import layered

        result = layered.run_t2l(prompt="test", layers=layers, models_dir=tmp_path)

    assert latent_cut_mock.call_count == layers
    assert len(result) == layers


# ---------------------------------------------------------------------------
# run_i2l() behaviour tests (AC03, AC04, AC05, AC06)
# ---------------------------------------------------------------------------


def test_run_i2l_raises_on_runtime_error(tmp_path: Path) -> None:
    """run_i2l() must raise RuntimeError when check_runtime() returns an error (AC05)."""
    from comfy_diffusion.pipelines.image.qwen.layered import run_i2l

    fake_image = MagicMock(name="image")
    with patch(_RUNTIME_PATCH, return_value={"error": "ComfyUI not found", "python_version": "3.12.0"}):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            run_i2l(prompt="test", image=fake_image, models_dir=tmp_path)


def test_run_i2l_returns_list_of_pil_images(tmp_path: Path) -> None:
    """run_i2l() must return a list[PIL.Image.Image] (AC06)."""
    from PIL import Image as PILImage

    fake_image_out = MagicMock(spec=PILImage.Image)
    fake_tensor = MagicMock(name="tensor")
    fake_scaled = MagicMock(name="scaled_image")
    layers = 2
    latent_cut_mock = MagicMock(return_value=MagicMock(name="layer_latent"))

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=fake_tensor),
        patch(_IMAGE_SCALE_PATCH, return_value=fake_scaled),
        patch(_GET_IMAGE_SIZE_PATCH, return_value=(640, 480)),
        patch(_EMPTY_QWEN_PATCH, return_value=MagicMock()),
        patch(_VAE_ENCODE_TENSOR_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_REF_LATENT_PATCH, side_effect=lambda cond, lat: cond),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_LATENT_CUT_PATCH, latent_cut_mock),
        patch(_VAE_DECODE_PATCH, return_value=fake_image_out),
    ):
        from PIL import Image as PILImage

        input_img = MagicMock(spec=PILImage.Image)

        from comfy_diffusion.pipelines.image.qwen import layered

        result = layered.run_i2l(
            prompt="a test prompt",
            image=input_img,
            layers=layers,
            models_dir=tmp_path,
        )

    assert isinstance(result, list)
    assert len(result) == layers
    for img in result:
        assert img is fake_image_out


def test_run_i2l_calls_image_scale_to_max_dimension(tmp_path: Path) -> None:
    """run_i2l() must scale input image to max 640px (AC03)."""
    scale_mock = MagicMock(return_value=MagicMock(name="scaled_image"))

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_IMAGE_SCALE_PATCH, scale_mock),
        patch(_GET_IMAGE_SIZE_PATCH, return_value=(640, 480)),
        patch(_EMPTY_QWEN_PATCH, return_value=MagicMock()),
        patch(_VAE_ENCODE_TENSOR_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_REF_LATENT_PATCH, side_effect=lambda cond, lat: cond),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_LATENT_CUT_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from PIL import Image as PILImage

        input_img = MagicMock(spec=PILImage.Image)

        from comfy_diffusion.pipelines.image.qwen import layered

        layered.run_i2l(prompt="test", image=input_img, models_dir=tmp_path)

    scale_mock.assert_called_once()
    call_args = scale_mock.call_args
    # image_scale_to_max_dimension(image, upscale_method, max_dimension)
    pos_args = call_args[0]
    assert pos_args[1] == "lanczos", f"upscale_method must be 'lanczos', got {pos_args[1]!r}"
    assert pos_args[2] == 640, f"max_dimension must be 640, got {pos_args[2]!r}"


def test_run_i2l_calls_reference_latent_for_both_conditioning(tmp_path: Path) -> None:
    """run_i2l() must apply reference_latent to both positive and negative (AC03)."""
    ref_latent_mock = MagicMock(side_effect=lambda cond, lat: cond)

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_IMAGE_SCALE_PATCH, return_value=MagicMock()),
        patch(_GET_IMAGE_SIZE_PATCH, return_value=(640, 480)),
        patch(_EMPTY_QWEN_PATCH, return_value=MagicMock()),
        patch(_VAE_ENCODE_TENSOR_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_REF_LATENT_PATCH, ref_latent_mock),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_LATENT_CUT_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from PIL import Image as PILImage

        input_img = MagicMock(spec=PILImage.Image)

        from comfy_diffusion.pipelines.image.qwen import layered

        layered.run_i2l(prompt="test", image=input_img, models_dir=tmp_path)

    assert ref_latent_mock.call_count == 2, (
        f"reference_latent must be called twice (positive + negative), "
        f"got {ref_latent_mock.call_count}"
    )


def test_run_i2l_calls_vae_encode_tensor(tmp_path: Path) -> None:
    """run_i2l() must encode the reference image with vae_encode_tensor (AC03)."""
    vae_enc_mock = MagicMock(return_value=MagicMock(name="ref_latent"))

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_IMAGE_SCALE_PATCH, return_value=MagicMock()),
        patch(_GET_IMAGE_SIZE_PATCH, return_value=(640, 480)),
        patch(_EMPTY_QWEN_PATCH, return_value=MagicMock()),
        patch(_VAE_ENCODE_TENSOR_PATCH, vae_enc_mock),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_REF_LATENT_PATCH, side_effect=lambda cond, lat: cond),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_LATENT_CUT_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from PIL import Image as PILImage

        input_img = MagicMock(spec=PILImage.Image)

        from comfy_diffusion.pipelines.image.qwen import layered

        layered.run_i2l(prompt="test", image=input_img, models_dir=tmp_path)

    vae_enc_mock.assert_called_once()
