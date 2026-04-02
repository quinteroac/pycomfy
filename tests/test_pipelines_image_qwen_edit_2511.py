"""Tests for comfy_diffusion/pipelines/image/qwen/edit_2511.py — Qwen Image Edit 2511 pipeline.

Covers:
  - File exists and is importable                                               (AC01)
  - __all__ = ["manifest", "run"]                                               (AC05)
  - No top-level comfy/torch imports (lazy import pattern)                      (AC06)
  - manifest() returns exactly 4 HFModelEntry items with correct dest paths    (AC02)
  - run() has correct signature with expected defaults                          (AC03)
  - run() returns non-empty list[PIL.Image.Image]                              (AC03, AC04)
  - run() check_runtime() raises RuntimeError on failure                        (AC03)
  - run() node execution order: load_unet, load_clip, load_vae,
    model_sampling_aura_flow, apply_cfg_norm, optional apply_lora,
    flux_kontext_image_scale, encode_qwen_image_edit_plus (×2),
    apply_flux_kontext_multi_reference (×2), vae_encode_tensor,
    sample, vae_decode                                                          (AC03)
  - Default steps=40, cfg=3.0, use_lora=True, seed=0                          (AC03)
  - With use_lora=False, apply_lora is NOT called                              (AC03)
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
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "qwen" / "edit_2511.py"
)

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_AURA_FLOW_PATCH = "comfy_diffusion.models.model_sampling_aura_flow"
_CFG_NORM_PATCH = "comfy_diffusion.video.apply_cfg_norm"
_APPLY_LORA_PATCH = "comfy_diffusion.lora.apply_lora"
_FLUX_SCALE_PATCH = "comfy_diffusion.image.flux_kontext_image_scale"
_ENCODE_QWEN_PATCH = "comfy_diffusion.conditioning.encode_qwen_image_edit_plus"
_FLUX_MULTI_REF_PATCH = "comfy_diffusion.conditioning.apply_flux_kontext_multi_reference"
_VAE_ENCODE_TENSOR_PATCH = "comfy_diffusion.vae.vae_encode_tensor"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"


# ---------------------------------------------------------------------------
# File-level checks (AC01, AC05, AC06)
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "edit_2511.py must exist under pipelines/image/qwen/"


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
    assert docstring, "edit_2511.py must have a module-level docstring"


def test_pipeline_has_dunder_all_with_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert "__all__" in source
    assert '"manifest"' in source or "'manifest'" in source
    assert '"run"' in source or "'run'" in source


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.qwen import edit_2511

    assert hasattr(edit_2511, "__all__")
    assert set(edit_2511.__all__) == {"manifest", "run"}


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


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks (AC02)
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_four_entries() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 4, f"manifest() must return exactly 4 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry must be an HFModelEntry, got {type(entry)!r}"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "qwen_image_edit_2511_bf16" in d for d in dests), (
        f"No unet dest matching diffusion_models/qwen_image_edit_2511_bf16.safetensors in {dests}"
    )


def test_manifest_clip_dest_path() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "qwen_2.5_vl_7b_fp8_scaled" in d for d in dests), (
        f"No text_encoder dest matching text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors in {dests}"
    )


def test_manifest_vae_dest_path() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("vae" in d and "qwen_image_vae" in d for d in dests), (
        f"No vae dest matching vae/qwen_image_vae.safetensors in {dests}"
    )


def test_manifest_lora_dest_path() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d and "Qwen-Image-Edit-2511-Lightning" in d for d in dests), (
        f"No lora dest matching loras/Qwen-Image-Edit-2511-Lightning*.safetensors in {dests}"
    )


# ---------------------------------------------------------------------------
# Signature checks (AC03)
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run

    sig = inspect.signature(run)
    required = {"prompt", "image", "models_dir", "steps", "cfg", "use_lora", "seed"}
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_signature_includes_optional_images() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run

    sig = inspect.signature(run)
    assert "image2" in sig.parameters
    assert "image3" in sig.parameters
    assert sig.parameters["image2"].default is None
    assert sig.parameters["image3"].default is None


def test_run_default_steps_40() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 40


def test_run_default_cfg_3_0() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run

    sig = inspect.signature(run)
    assert sig.parameters["cfg"].default == 3.0


def test_run_default_use_lora_true() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run

    sig = inspect.signature(run)
    assert sig.parameters["use_lora"].default is True


def test_run_default_seed_0() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run

    sig = inspect.signature(run)
    assert sig.parameters["seed"].default == 0


def test_pipeline_uses_euler_sampler() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '"euler"' in source or "'euler'" in source


def test_pipeline_uses_simple_scheduler() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '"simple"' in source or "'simple'" in source


# ---------------------------------------------------------------------------
# Helpers for run() integration tests
# ---------------------------------------------------------------------------


def _make_mock_image() -> MagicMock:
    """Return a mock PIL image."""
    from unittest.mock import MagicMock

    img = MagicMock()
    img.__class__ = type("Image", (), {})  # not a real PIL image
    return img


def _build_run_patches(
    *,
    runtime_ok: bool = True,
    pil_image: bool = True,
) -> dict[str, Any]:
    """Build a dict of patch targets → mock return values for run()."""
    import PIL.Image

    fake_pil = MagicMock(spec=PIL.Image.Image)
    fake_tensor = MagicMock(name="tensor")
    fake_model = MagicMock(name="model")
    fake_clip = MagicMock(name="clip")
    fake_vae = MagicMock(name="vae")
    fake_conditioning = MagicMock(name="conditioning")
    fake_latent = MagicMock(name="latent")

    mm_instance = MagicMock(name="mm")
    mm_instance.load_unet.return_value = fake_model
    mm_instance.load_clip.return_value = fake_clip
    mm_instance.load_vae.return_value = fake_vae

    return {
        "mm_cls": mm_instance,
        "model": fake_model,
        "clip": fake_clip,
        "vae": fake_vae,
        "tensor": fake_tensor,
        "conditioning": fake_conditioning,
        "latent": fake_latent,
        "pil": fake_pil,
        "runtime_ok": runtime_ok,
    }


# ---------------------------------------------------------------------------
# Execution order tests (AC03)
# ---------------------------------------------------------------------------


def test_run_raises_on_runtime_error() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run

    with patch(_RUNTIME_PATCH, return_value={"error": "submodule missing", "python_version": "3.12"}):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            run("edit this", MagicMock(), models_dir="/tmp/models")


def _run_with_mocks(
    prompt: str = "edit this",
    image: Any = None,
    image2: Any = None,
    image3: Any = None,
    models_dir: str = "/tmp/models",
    *,
    steps: int = 40,
    cfg: float = 3.0,
    use_lora: bool = True,
    seed: int = 0,
) -> tuple[list[Any], dict[str, MagicMock]]:
    """Run the pipeline with all external calls mocked; return (result, mocks)."""
    import PIL.Image

    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run

    fake_pil = MagicMock(spec=PIL.Image.Image)
    fake_pil_input = MagicMock(spec=PIL.Image.Image)
    if image is None:
        image = fake_pil_input

    fake_tensor = MagicMock(name="image_tensor")
    fake_scaled = MagicMock(name="scaled_image")
    fake_model_raw = MagicMock(name="model_raw")
    fake_model_aura = MagicMock(name="model_aura")
    fake_model_cfg = MagicMock(name="model_cfg")
    fake_model_lora = MagicMock(name="model_lora")
    fake_clip = MagicMock(name="clip")
    fake_vae = MagicMock(name="vae")
    fake_cond_neg = MagicMock(name="cond_neg")
    fake_cond_pos = MagicMock(name="cond_pos")
    fake_cond_neg2 = MagicMock(name="cond_neg_ref")
    fake_cond_pos2 = MagicMock(name="cond_pos_ref")
    fake_latent = MagicMock(name="latent")
    fake_latent_out = MagicMock(name="latent_out")

    mm_instance = MagicMock(name="mm")
    mm_instance.load_unet.return_value = fake_model_raw
    mm_instance.load_clip.return_value = fake_clip
    mm_instance.load_vae.return_value = fake_vae

    mocks: dict[str, MagicMock] = {
        "mm": mm_instance,
        "model_raw": fake_model_raw,
        "model_aura": fake_model_aura,
        "model_cfg": fake_model_cfg,
        "model_lora": fake_model_lora,
        "clip": fake_clip,
        "vae": fake_vae,
        "tensor": fake_tensor,
        "scaled": fake_scaled,
        "cond_neg": fake_cond_neg,
        "cond_pos": fake_cond_pos,
        "cond_neg2": fake_cond_neg2,
        "cond_pos2": fake_cond_pos2,
        "latent": fake_latent,
        "latent_out": fake_latent_out,
        "pil_out": fake_pil,
    }

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12"}),
        patch(_MM_PATCH, return_value=mm_instance),
        patch(_AURA_FLOW_PATCH, return_value=fake_model_aura) as mock_aura,
        patch(_CFG_NORM_PATCH, return_value=fake_model_cfg) as mock_cfg_norm,
        patch(_APPLY_LORA_PATCH, return_value=(fake_model_lora, fake_clip)) as mock_lora,
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=fake_tensor) as mock_to_tensor,
        patch(_FLUX_SCALE_PATCH, return_value=fake_scaled) as mock_scale,
        patch(
            _ENCODE_QWEN_PATCH,
            side_effect=[fake_cond_neg, fake_cond_pos],
        ) as mock_encode,
        patch(
            _FLUX_MULTI_REF_PATCH,
            side_effect=[fake_cond_neg2, fake_cond_pos2],
        ) as mock_multi_ref,
        patch(_VAE_ENCODE_TENSOR_PATCH, return_value=fake_latent) as mock_vae_enc,
        patch(_SAMPLE_PATCH, return_value=fake_latent_out) as mock_sample,
        patch(_VAE_DECODE_PATCH, return_value=fake_pil) as mock_vae_dec,
    ):
        result = run(
            prompt,
            image,
            image2=image2,
            image3=image3,
            models_dir=models_dir,
            steps=steps,
            cfg=cfg,
            use_lora=use_lora,
            seed=seed,
        )
        mocks.update({
            "mock_aura": mock_aura,
            "mock_cfg_norm": mock_cfg_norm,
            "mock_lora": mock_lora,
            "mock_to_tensor": mock_to_tensor,
            "mock_scale": mock_scale,
            "mock_encode": mock_encode,
            "mock_multi_ref": mock_multi_ref,
            "mock_vae_enc": mock_vae_enc,
            "mock_sample": mock_sample,
            "mock_vae_dec": mock_vae_dec,
        })

    return result, mocks


def test_run_returns_non_empty_list() -> None:
    """AC04: run() returns a non-empty list."""
    result, _ = _run_with_mocks()
    assert isinstance(result, list)
    assert len(result) > 0


def test_run_returns_pil_images() -> None:
    """AC04: each element is a PIL.Image.Image (or mock spec thereof)."""
    import PIL.Image

    result, _ = _run_with_mocks()
    for img in result:
        assert isinstance(img, PIL.Image.Image)


def test_run_loads_unet() -> None:
    _, mocks = _run_with_mocks()
    mocks["mm"].load_unet.assert_called_once()


def test_run_loads_clip_with_qwen_image_type() -> None:
    _, mocks = _run_with_mocks()
    mocks["mm"].load_clip.assert_called_once()
    call_kwargs = mocks["mm"].load_clip.call_args
    assert call_kwargs.kwargs.get("clip_type") == "qwen_image"


def test_run_loads_vae() -> None:
    _, mocks = _run_with_mocks()
    mocks["mm"].load_vae.assert_called_once()


def test_run_applies_aura_flow_shift_3_1() -> None:
    _, mocks = _run_with_mocks()
    mocks["mock_aura"].assert_called_once_with(mocks["model_raw"], shift=3.1)


def test_run_applies_cfg_norm() -> None:
    _, mocks = _run_with_mocks()
    mocks["mock_cfg_norm"].assert_called_once_with(mocks["model_aura"], strength=1.0)


def test_run_applies_lora_when_use_lora_true() -> None:
    _, mocks = _run_with_mocks(use_lora=True)
    mocks["mock_lora"].assert_called_once()
    lora_call_args = mocks["mock_lora"].call_args
    # LoraLoaderModelOnly: clip strength = 0.0
    assert lora_call_args.args[3] == 1.0  # strength_model
    assert lora_call_args.args[4] == 0.0  # strength_clip


def test_run_skips_lora_when_use_lora_false() -> None:
    _, mocks = _run_with_mocks(use_lora=False)
    mocks["mock_lora"].assert_not_called()


def test_run_scales_input_image() -> None:
    _, mocks = _run_with_mocks()
    mocks["mock_scale"].assert_called_once_with(mocks["tensor"])


def test_run_encodes_negative_conditioning_first() -> None:
    _, mocks = _run_with_mocks()
    assert mocks["mock_encode"].call_count == 2
    neg_call = mocks["mock_encode"].call_args_list[0]
    assert neg_call.kwargs.get("prompt", neg_call.args[5] if len(neg_call.args) > 5 else "") == ""


def test_run_encodes_positive_conditioning_second() -> None:
    _, mocks = _run_with_mocks(prompt="change to fur")
    pos_call = mocks["mock_encode"].call_args_list[1]
    # prompt may be positional (index 5) or keyword
    prompt_val = pos_call.kwargs.get("prompt") or (
        pos_call.args[5] if len(pos_call.args) > 5 else None
    )
    assert prompt_val == "change to fur"


def test_run_applies_flux_kontext_multi_reference_twice() -> None:
    _, mocks = _run_with_mocks()
    assert mocks["mock_multi_ref"].call_count == 2


def test_run_calls_vae_encode_tensor() -> None:
    _, mocks = _run_with_mocks()
    mocks["mock_vae_enc"].assert_called_once_with(mocks["vae"], mocks["scaled"])


def test_run_calls_sample_with_correct_args() -> None:
    result, mocks = _run_with_mocks(steps=40, cfg=3.0, seed=7)
    mocks["mock_sample"].assert_called_once()
    call_args = mocks["mock_sample"].call_args
    assert call_args.args[4] == 40   # steps
    assert call_args.args[5] == 3.0  # cfg
    assert call_args.args[6] == "euler"
    assert call_args.args[7] == "simple"
    assert call_args.args[8] == 7    # seed


def test_run_calls_vae_decode() -> None:
    _, mocks = _run_with_mocks()
    mocks["mock_vae_dec"].assert_called_once_with(mocks["vae"], mocks["latent_out"])


def test_run_passes_image2_to_conditioning() -> None:
    import PIL.Image

    fake_img2 = MagicMock(spec=PIL.Image.Image)
    _, mocks = _run_with_mocks(image2=fake_img2)
    # Both conditioning calls should receive image2 tensor
    for c in mocks["mock_encode"].call_args_list:
        # image2 is the 4th positional arg or keyword
        img2_val = c.kwargs.get("image2") or (c.args[3] if len(c.args) > 3 else None)
        assert img2_val is not None


def test_run_passes_none_image3_when_not_provided() -> None:
    _, mocks = _run_with_mocks()
    for c in mocks["mock_encode"].call_args_list:
        img3_val = c.kwargs.get("image3") or (c.args[4] if len(c.args) > 4 else None)
        assert img3_val is None
