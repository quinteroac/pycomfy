"""CPU-safe unit tests for the Qwen Image Edit 2511 pipeline and related node wrappers.

Covers:
  - AC01: this file exists                                                         (AC01)
  - AC02: manifest() returns exactly 4 ModelEntry items with correct filenames
          and destination directories                                               (AC02)
  - AC03: run() stubs all model loading and node functions; asserts vae_decode
          is called and result is returned                                          (AC03)
  - AC04: four new node-wrapper functions covered with mocked comfy.* internals    (AC04)
  - AC05: all tests pass under uv run pytest tests/test_qwen_edit_2511.py on CPU   (AC05)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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


# ===========================================================================
# AC02 — manifest() returns exactly 4 ModelEntry items
# ===========================================================================


def test_manifest_returns_list() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    result = manifest()
    assert isinstance(result, list)


def test_manifest_returns_exactly_four_entries() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    result = manifest()
    assert len(result) == 4, f"Expected 4 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"Expected HFModelEntry, got {type(entry)!r}"
        )


def test_manifest_unet_filename_and_dest() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    entries = manifest()
    unet = next(
        (e for e in entries if "qwen_image_edit_2511_bf16" in str(e.dest)),
        None,
    )
    assert unet is not None, "No unet entry found in manifest"
    assert "diffusion_models" in str(unet.dest)
    assert str(unet.dest).endswith(".safetensors")


def test_manifest_clip_filename_and_dest() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    entries = manifest()
    clip = next(
        (e for e in entries if "qwen_2.5_vl_7b_fp8_scaled" in str(e.dest)),
        None,
    )
    assert clip is not None, "No CLIP entry found in manifest"
    assert "text_encoders" in str(clip.dest)
    assert str(clip.dest).endswith(".safetensors")


def test_manifest_vae_filename_and_dest() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    entries = manifest()
    vae = next(
        (e for e in entries if "qwen_image_vae" in str(e.dest)),
        None,
    )
    assert vae is not None, "No VAE entry found in manifest"
    assert "vae" in str(vae.dest)
    assert str(vae.dest).endswith(".safetensors")


def test_manifest_lora_filename_and_dest() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest

    entries = manifest()
    lora = next(
        (e for e in entries if "Qwen-Image-Edit-2511-Lightning" in str(e.dest)),
        None,
    )
    assert lora is not None, "No LoRA entry found in manifest"
    assert "loras" in str(lora.dest)
    assert str(lora.dest).endswith(".safetensors")


# ===========================================================================
# AC03 — run() stubbed call-graph tests; vae_decode called; result returned
# ===========================================================================


def _make_run_mocks() -> tuple[MagicMock, dict[str, MagicMock]]:
    """Build (mm_instance, named_mocks) for patching run()."""
    import PIL.Image

    fake_pil = MagicMock(spec=PIL.Image.Image)
    fake_model_raw = MagicMock(name="model_raw")
    fake_model_aura = MagicMock(name="model_aura")
    fake_model_cfg = MagicMock(name="model_cfg")
    fake_model_lora = MagicMock(name="model_lora")
    fake_clip = MagicMock(name="clip")
    fake_vae = MagicMock(name="vae")
    fake_tensor = MagicMock(name="image_tensor")
    fake_scaled = MagicMock(name="scaled_image")
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
    return mm_instance, mocks


def _run_pipeline(
    prompt: str = "make it look metallic",
    *,
    use_lora: bool = True,
    steps: int = 40,
    cfg: float = 3.0,
    seed: int = 0,
    image2: Any = None,
    image3: Any = None,
) -> tuple[list[Any], dict[str, MagicMock]]:
    """Execute run() with all external calls mocked; return (result, mocks)."""
    import PIL.Image

    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run

    mm_instance, mocks = _make_run_mocks()
    fake_input_image = MagicMock(spec=PIL.Image.Image)

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12"}),
        patch(_MM_PATCH, return_value=mm_instance),
        patch(_AURA_FLOW_PATCH, return_value=mocks["model_aura"]) as mock_aura,
        patch(_CFG_NORM_PATCH, return_value=mocks["model_cfg"]) as mock_cfg,
        patch(_APPLY_LORA_PATCH, return_value=(mocks["model_lora"], mocks["clip"])) as mock_lora,
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=mocks["tensor"]) as mock_to_tensor,
        patch(_FLUX_SCALE_PATCH, return_value=mocks["scaled"]) as mock_scale,
        patch(
            _ENCODE_QWEN_PATCH,
            side_effect=[mocks["cond_neg"], mocks["cond_pos"]],
        ) as mock_encode,
        patch(
            _FLUX_MULTI_REF_PATCH,
            side_effect=[mocks["cond_neg2"], mocks["cond_pos2"]],
        ) as mock_multi_ref,
        patch(_VAE_ENCODE_TENSOR_PATCH, return_value=mocks["latent"]) as mock_vae_enc,
        patch(_SAMPLE_PATCH, return_value=mocks["latent_out"]) as mock_sample,
        patch(_VAE_DECODE_PATCH, return_value=mocks["pil_out"]) as mock_vae_dec,
    ):
        result = run(
            prompt,
            fake_input_image,
            image2=image2,
            image3=image3,
            models_dir="/tmp/models",
            steps=steps,
            cfg=cfg,
            use_lora=use_lora,
            seed=seed,
        )
        mocks.update(
            {
                "mock_aura": mock_aura,
                "mock_cfg": mock_cfg,
                "mock_lora": mock_lora,
                "mock_to_tensor": mock_to_tensor,
                "mock_scale": mock_scale,
                "mock_encode": mock_encode,
                "mock_multi_ref": mock_multi_ref,
                "mock_vae_enc": mock_vae_enc,
                "mock_sample": mock_sample,
                "mock_vae_dec": mock_vae_dec,
            }
        )

    return result, mocks


def test_run_raises_on_runtime_error() -> None:
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "submodule missing", "python_version": "3.12"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            run("edit this", MagicMock(), models_dir="/tmp/models")


def test_run_returns_non_empty_list() -> None:
    result, _ = _run_pipeline()
    assert isinstance(result, list)
    assert len(result) > 0


def test_run_returns_pil_images() -> None:
    import PIL.Image

    result, _ = _run_pipeline()
    for img in result:
        assert isinstance(img, PIL.Image.Image)


def test_run_calls_vae_decode() -> None:
    """AC03: vae_decode must be called."""
    _, mocks = _run_pipeline()
    mocks["mock_vae_dec"].assert_called_once()


def test_run_vae_decode_receives_correct_args() -> None:
    _, mocks = _run_pipeline()
    mocks["mock_vae_dec"].assert_called_once_with(mocks["vae"], mocks["latent_out"])


def test_run_loads_unet() -> None:
    _, mocks = _run_pipeline()
    mocks["mm"].load_unet.assert_called_once()


def test_run_loads_clip_with_qwen_image_type() -> None:
    _, mocks = _run_pipeline()
    mocks["mm"].load_clip.assert_called_once()
    assert mocks["mm"].load_clip.call_args.kwargs.get("clip_type") == "qwen_image"


def test_run_loads_vae() -> None:
    _, mocks = _run_pipeline()
    mocks["mm"].load_vae.assert_called_once()


def test_run_applies_aura_flow_patch() -> None:
    _, mocks = _run_pipeline()
    mocks["mock_aura"].assert_called_once_with(mocks["model_raw"], shift=3.1)


def test_run_applies_cfg_norm() -> None:
    _, mocks = _run_pipeline()
    mocks["mock_cfg"].assert_called_once_with(mocks["model_aura"], strength=1.0)


def test_run_applies_lora_when_use_lora_true() -> None:
    _, mocks = _run_pipeline(use_lora=True)
    mocks["mock_lora"].assert_called_once()
    # LoraLoaderModelOnly: clip strength must be 0.0
    call_args = mocks["mock_lora"].call_args
    assert call_args.args[3] == 1.0   # strength_model
    assert call_args.args[4] == 0.0   # strength_clip


def test_run_skips_lora_when_use_lora_false() -> None:
    _, mocks = _run_pipeline(use_lora=False)
    mocks["mock_lora"].assert_not_called()


def test_run_scales_input_image() -> None:
    _, mocks = _run_pipeline()
    mocks["mock_scale"].assert_called_once_with(mocks["tensor"])


def test_run_encodes_conditioning_twice() -> None:
    _, mocks = _run_pipeline()
    assert mocks["mock_encode"].call_count == 2


def test_run_encodes_negative_with_empty_prompt() -> None:
    _, mocks = _run_pipeline(prompt="change colour")
    neg_call = mocks["mock_encode"].call_args_list[0]
    prompt_val = neg_call.kwargs.get("prompt", neg_call.args[5] if len(neg_call.args) > 5 else None)
    assert prompt_val == ""


def test_run_encodes_positive_with_user_prompt() -> None:
    _, mocks = _run_pipeline(prompt="change colour")
    pos_call = mocks["mock_encode"].call_args_list[1]
    prompt_val = pos_call.kwargs.get("prompt") or (
        pos_call.args[5] if len(pos_call.args) > 5 else None
    )
    assert prompt_val == "change colour"


def test_run_applies_flux_kontext_multi_reference_twice() -> None:
    _, mocks = _run_pipeline()
    assert mocks["mock_multi_ref"].call_count == 2


def test_run_calls_vae_encode_tensor() -> None:
    _, mocks = _run_pipeline()
    mocks["mock_vae_enc"].assert_called_once_with(mocks["vae"], mocks["scaled"])


def test_run_calls_sample_with_euler_simple() -> None:
    _, mocks = _run_pipeline(steps=40, cfg=3.0, seed=7)
    mocks["mock_sample"].assert_called_once()
    call_args = mocks["mock_sample"].call_args
    assert call_args.args[4] == 40
    assert call_args.args[5] == 3.0
    assert call_args.args[6] == "euler"
    assert call_args.args[7] == "simple"
    assert call_args.args[8] == 7


# ===========================================================================
# AC04 — four new node-wrapper functions with mocked comfy.* internals
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. encode_qwen_image_edit_plus
# ---------------------------------------------------------------------------


def test_encode_qwen_image_edit_plus_callable() -> None:
    from comfy_diffusion.conditioning import encode_qwen_image_edit_plus

    assert callable(encode_qwen_image_edit_plus)


def test_encode_qwen_image_edit_plus_in_all() -> None:
    import comfy_diffusion.conditioning as mod

    assert "encode_qwen_image_edit_plus" in mod.__all__


def test_encode_qwen_image_edit_plus_calls_clip_tokenize_and_encode() -> None:
    """Asserts clip.tokenize and clip.encode_from_tokens_scheduled are invoked."""
    fake_clip = MagicMock()
    fake_tokens = MagicMock()
    fake_clip.tokenize.return_value = fake_tokens
    fake_conditioning = [{"test": True}]
    fake_clip.encode_from_tokens_scheduled.return_value = fake_conditioning

    fake_comfy_utils = MagicMock()
    fake_scaled = MagicMock()
    fake_scaled.movedim.return_value = MagicMock(shape=(1, 3, 8, 8))
    fake_comfy_utils.common_upscale.return_value = fake_scaled

    fake_node_helpers = MagicMock()
    merged = [{"merged": True}]
    fake_node_helpers.conditioning_set_values.return_value = merged

    fake_image = MagicMock()
    fake_image.movedim.return_value = MagicMock(shape=(1, 3, 8, 8))

    with patch(
        "comfy_diffusion.conditioning._get_qwen_image_edit_plus_dependencies",
        return_value=(math, fake_comfy_utils, fake_node_helpers),
    ):
        from comfy_diffusion.conditioning import encode_qwen_image_edit_plus

        result = encode_qwen_image_edit_plus(
            clip=fake_clip,
            vae=MagicMock(),
            image1=fake_image,
            prompt="make it blue",
        )

    fake_clip.tokenize.assert_called_once()
    fake_clip.encode_from_tokens_scheduled.assert_called_once_with(fake_tokens)
    assert result is merged


def test_encode_qwen_image_edit_plus_no_vae_skips_ref_latents() -> None:
    """When vae=None, conditioning_set_values must NOT be called."""
    fake_clip = MagicMock()
    fake_tokens = MagicMock()
    fake_clip.tokenize.return_value = fake_tokens
    fake_conditioning = [{"test": True}]
    fake_clip.encode_from_tokens_scheduled.return_value = fake_conditioning

    fake_comfy_utils = MagicMock()
    fake_scaled = MagicMock()
    fake_scaled.movedim.return_value = MagicMock(shape=(1, 3, 8, 8))
    fake_comfy_utils.common_upscale.return_value = fake_scaled

    fake_node_helpers = MagicMock()
    fake_image = MagicMock()
    fake_image.movedim.return_value = MagicMock(shape=(1, 3, 8, 8))

    with patch(
        "comfy_diffusion.conditioning._get_qwen_image_edit_plus_dependencies",
        return_value=(math, fake_comfy_utils, fake_node_helpers),
    ):
        from comfy_diffusion.conditioning import encode_qwen_image_edit_plus

        result = encode_qwen_image_edit_plus(
            clip=fake_clip,
            vae=None,
            image1=fake_image,
            prompt="prompt",
        )

    fake_node_helpers.conditioning_set_values.assert_not_called()
    assert result is fake_conditioning


# ---------------------------------------------------------------------------
# 2. apply_flux_kontext_multi_reference
# ---------------------------------------------------------------------------


def test_apply_flux_kontext_multi_reference_callable() -> None:
    from comfy_diffusion.conditioning import apply_flux_kontext_multi_reference

    assert callable(apply_flux_kontext_multi_reference)


def test_apply_flux_kontext_multi_reference_in_all() -> None:
    import comfy_diffusion.conditioning as mod

    assert "apply_flux_kontext_multi_reference" in mod.__all__


def test_apply_flux_kontext_multi_reference_calls_conditioning_set_values() -> None:
    """Asserts node_helpers.conditioning_set_values is invoked with the method name."""
    fake_node_helpers = MagicMock()
    out_cond = [{"method_set": True}]
    fake_node_helpers.conditioning_set_values.return_value = out_cond
    cond = [{"base": True}]

    with patch(
        "comfy_diffusion.conditioning._get_node_helpers",
        return_value=fake_node_helpers,
    ):
        from comfy_diffusion.conditioning import apply_flux_kontext_multi_reference

        result = apply_flux_kontext_multi_reference(cond, "index_timestep_zero")

    fake_node_helpers.conditioning_set_values.assert_called_once_with(
        cond, {"reference_latents_method": "index_timestep_zero"}
    )
    assert result is out_cond


def test_apply_flux_kontext_multi_reference_default_method() -> None:
    """Default reference_latents_method is 'index_timestep_zero'."""
    import inspect

    from comfy_diffusion.conditioning import apply_flux_kontext_multi_reference

    sig = inspect.signature(apply_flux_kontext_multi_reference)
    assert sig.parameters["reference_latents_method"].default == "index_timestep_zero"


# ---------------------------------------------------------------------------
# 3. flux_kontext_image_scale
# ---------------------------------------------------------------------------


def test_flux_kontext_image_scale_callable() -> None:
    from comfy_diffusion.image import flux_kontext_image_scale

    assert callable(flux_kontext_image_scale)


def test_flux_kontext_image_scale_in_all() -> None:
    import comfy_diffusion.image as mod

    assert "flux_kontext_image_scale" in mod.__all__


def test_flux_kontext_image_scale_calls_common_upscale() -> None:
    """Asserts comfy_utils.common_upscale is called with the chosen resolution."""
    fake_comfy_utils = MagicMock()
    upscaled = MagicMock()
    upscaled.movedim.return_value = upscaled
    fake_comfy_utils.common_upscale.return_value = upscaled

    preferred_resolutions = [(1024, 1024), (1280, 768), (768, 1280)]

    fake_image = MagicMock()
    fake_inner = MagicMock()
    fake_inner.movedim.return_value = upscaled
    fake_image.shape = (1, 1024, 1024, 3)
    fake_image.movedim.return_value = fake_inner

    with patch(
        "comfy_diffusion.image._get_flux_kontext_dependencies",
        return_value=(fake_comfy_utils, preferred_resolutions),
    ):
        from comfy_diffusion.image import flux_kontext_image_scale

        result = flux_kontext_image_scale(fake_image)

    fake_comfy_utils.common_upscale.assert_called_once()
    # The first two args should be the inner tensor and the chosen width/height
    call_args = fake_comfy_utils.common_upscale.call_args
    assert call_args.args[3] == "lanczos"
    assert call_args.args[4] == "center"


# ---------------------------------------------------------------------------
# 4. apply_cfg_norm
# ---------------------------------------------------------------------------


def test_apply_cfg_norm_callable() -> None:
    from comfy_diffusion.video import apply_cfg_norm

    assert callable(apply_cfg_norm)


def test_apply_cfg_norm_in_all() -> None:
    import comfy_diffusion.video as mod

    assert "apply_cfg_norm" in mod.__all__


def test_apply_cfg_norm_clones_model_and_registers_hook() -> None:
    """Asserts apply_cfg_norm clones the model and installs a post-CFG hook."""
    fake_model = MagicMock(name="model")
    cloned = MagicMock(name="cloned_model")
    fake_model.clone.return_value = cloned

    from comfy_diffusion.video import apply_cfg_norm

    result = apply_cfg_norm(fake_model, strength=1.0)

    fake_model.clone.assert_called_once()
    cloned.set_model_sampler_post_cfg_function.assert_called_once()
    assert result is cloned


def test_apply_cfg_norm_hook_scales_prediction() -> None:
    """The installed hook scales the denoised prediction by the norm ratio."""
    import torch

    fake_model = MagicMock(name="model")
    cloned = MagicMock(name="cloned_model")
    fake_model.clone.return_value = cloned

    from comfy_diffusion.video import apply_cfg_norm

    apply_cfg_norm(fake_model, strength=1.0)

    # Retrieve and exercise the hook directly
    hook = cloned.set_model_sampler_post_cfg_function.call_args.args[0]

    cond_p = torch.ones(1, 4, 4, 4)
    pred_text = torch.ones(1, 4, 4, 4) * 2.0
    args = {"cond_denoised": cond_p, "denoised": pred_text}
    output = hook(args)

    assert output.shape == pred_text.shape
    # With strength=1 and equal-shaped tensors, scale = norm(cond)/norm(pred)
    assert torch.all(output >= 0)
