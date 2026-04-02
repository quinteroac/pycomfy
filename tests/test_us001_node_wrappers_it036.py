"""Tests for US-001 (it_000036) — Missing node wrappers.

Covers:
  - AC01: encode_qwen_image_edit_plus in conditioning module, in __all__, correct signature, lazy import
  - AC02: apply_flux_kontext_multi_reference in conditioning module, in __all__, correct signature, lazy import
  - AC03: flux_kontext_image_scale in image module, in __all__, correct signature, lazy import
  - AC04: apply_cfg_norm in video module, in __all__, correct signature, lazy import
  - AC05: no top-level comfy.* / torch imports in any of the three modules
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONDITIONING_FILE = _REPO_ROOT / "comfy_diffusion" / "conditioning.py"
_IMAGE_FILE = _REPO_ROOT / "comfy_diffusion" / "image.py"
_VIDEO_FILE = _REPO_ROOT / "comfy_diffusion" / "video.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_top_level_imports(filepath: Path) -> list[str]:
    """Return module names imported at module top level (col_offset == 0)."""
    tree = ast.parse(filepath.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if node.col_offset != 0:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


# ---------------------------------------------------------------------------
# AC01: encode_qwen_image_edit_plus
# ---------------------------------------------------------------------------


def test_encode_qwen_image_edit_plus_exists() -> None:
    import comfy_diffusion.conditioning as mod

    assert hasattr(mod, "encode_qwen_image_edit_plus")
    assert callable(mod.encode_qwen_image_edit_plus)


def test_encode_qwen_image_edit_plus_in_dunder_all() -> None:
    import comfy_diffusion.conditioning as mod

    assert "encode_qwen_image_edit_plus" in mod.__all__


def test_encode_qwen_image_edit_plus_signature() -> None:
    from comfy_diffusion.conditioning import encode_qwen_image_edit_plus

    sig = inspect.signature(encode_qwen_image_edit_plus)
    params = sig.parameters
    assert "clip" in params
    assert "vae" in params
    assert "image1" in params
    assert "image2" in params
    assert "image3" in params
    assert "prompt" in params
    assert params["image2"].default is None
    assert params["image3"].default is None
    assert params["prompt"].default == ""


def test_encode_qwen_image_edit_plus_calls_clip_tokenize_and_encode() -> None:
    """Smoke: function calls clip.tokenize and clip.encode_from_tokens_scheduled."""
    import math

    fake_latent = MagicMock()
    fake_vae = MagicMock()
    fake_vae.encode.return_value = fake_latent

    fake_clip = MagicMock()
    fake_tokens = MagicMock()
    fake_clip.tokenize.return_value = fake_tokens
    fake_conditioning = [{"test": True}]
    fake_clip.encode_from_tokens_scheduled.return_value = fake_conditioning

    fake_comfy_utils = MagicMock()
    scaled = MagicMock()
    scaled.movedim.return_value = MagicMock()
    fake_comfy_utils.common_upscale.return_value = scaled

    fake_node_helpers = MagicMock()
    merged_cond = [{"merged": True}]
    fake_node_helpers.conditioning_set_values.return_value = merged_cond

    # Build a minimal fake image tensor (1, 8, 8, 3) via shape attribute
    fake_image = MagicMock()
    fake_image.movedim.return_value = MagicMock(shape=(1, 3, 8, 8))

    with patch(
        "comfy_diffusion.conditioning._get_qwen_image_edit_plus_dependencies",
        return_value=(math, fake_comfy_utils, fake_node_helpers),
    ):
        from comfy_diffusion.conditioning import encode_qwen_image_edit_plus

        result = encode_qwen_image_edit_plus(
            clip=fake_clip,
            vae=fake_vae,
            image1=fake_image,
            prompt="make it blue",
        )

    fake_clip.tokenize.assert_called_once()
    fake_clip.encode_from_tokens_scheduled.assert_called_once_with(fake_tokens)
    # reference latent was set because vae is not None
    fake_node_helpers.conditioning_set_values.assert_called_once()
    assert result is merged_cond


def test_encode_qwen_image_edit_plus_no_vae_skips_ref_latents() -> None:
    """When vae is None, reference_latents are not attached to conditioning."""
    import math

    fake_clip = MagicMock()
    fake_tokens = MagicMock()
    fake_clip.tokenize.return_value = fake_tokens
    fake_conditioning = [{"test": True}]
    fake_clip.encode_from_tokens_scheduled.return_value = fake_conditioning

    fake_comfy_utils = MagicMock()
    scaled = MagicMock()
    scaled.movedim.return_value = MagicMock()
    fake_comfy_utils.common_upscale.return_value = scaled

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
            prompt="make it blue",
        )

    # conditioning_set_values must NOT be called when vae is None
    fake_node_helpers.conditioning_set_values.assert_not_called()
    assert result is fake_conditioning


# ---------------------------------------------------------------------------
# AC02: apply_flux_kontext_multi_reference
# ---------------------------------------------------------------------------


def test_apply_flux_kontext_multi_reference_exists() -> None:
    import comfy_diffusion.conditioning as mod

    assert hasattr(mod, "apply_flux_kontext_multi_reference")
    assert callable(mod.apply_flux_kontext_multi_reference)


def test_apply_flux_kontext_multi_reference_in_dunder_all() -> None:
    import comfy_diffusion.conditioning as mod

    assert "apply_flux_kontext_multi_reference" in mod.__all__


def test_apply_flux_kontext_multi_reference_signature() -> None:
    from comfy_diffusion.conditioning import apply_flux_kontext_multi_reference

    sig = inspect.signature(apply_flux_kontext_multi_reference)
    params = sig.parameters
    assert "conditioning" in params
    assert "reference_latents_method" in params
    assert params["reference_latents_method"].default == "index_timestep_zero"


def test_apply_flux_kontext_multi_reference_sets_method() -> None:
    fake_node_helpers = MagicMock()
    out_cond = [{"method_set": True}]
    fake_node_helpers.conditioning_set_values.return_value = out_cond
    cond = [{"base": True}]

    with patch(
        "comfy_diffusion.conditioning._get_node_helpers",
        return_value=fake_node_helpers,
    ):
        from comfy_diffusion.conditioning import apply_flux_kontext_multi_reference

        result = apply_flux_kontext_multi_reference(cond, "index")

    fake_node_helpers.conditioning_set_values.assert_called_once_with(
        cond, {"reference_latents_method": "index"}
    )
    assert result is out_cond


def test_apply_flux_kontext_multi_reference_normalises_uxo() -> None:
    """uxo/uno variant should be normalised to 'uxo'."""
    fake_node_helpers = MagicMock()
    fake_node_helpers.conditioning_set_values.return_value = []
    cond: list = []

    with patch(
        "comfy_diffusion.conditioning._get_node_helpers",
        return_value=fake_node_helpers,
    ):
        from comfy_diffusion.conditioning import apply_flux_kontext_multi_reference

        apply_flux_kontext_multi_reference(cond, "uxo/uno")

    _call_kwargs = fake_node_helpers.conditioning_set_values.call_args
    passed_value = _call_kwargs[0][1]["reference_latents_method"]
    assert passed_value == "uxo"


def test_apply_flux_kontext_multi_reference_normalises_uso() -> None:
    """Values containing 'uso' are also normalised to 'uxo'."""
    fake_node_helpers = MagicMock()
    fake_node_helpers.conditioning_set_values.return_value = []

    with patch(
        "comfy_diffusion.conditioning._get_node_helpers",
        return_value=fake_node_helpers,
    ):
        from comfy_diffusion.conditioning import apply_flux_kontext_multi_reference

        apply_flux_kontext_multi_reference([], "uso_test")

    passed_value = fake_node_helpers.conditioning_set_values.call_args[0][1][
        "reference_latents_method"
    ]
    assert passed_value == "uxo"


# ---------------------------------------------------------------------------
# AC03: flux_kontext_image_scale
# ---------------------------------------------------------------------------


def test_flux_kontext_image_scale_exists() -> None:
    import comfy_diffusion.image as mod

    assert hasattr(mod, "flux_kontext_image_scale")
    assert callable(mod.flux_kontext_image_scale)


def test_flux_kontext_image_scale_in_dunder_all() -> None:
    import comfy_diffusion.image as mod

    assert "flux_kontext_image_scale" in mod.__all__


def test_flux_kontext_image_scale_signature() -> None:
    from comfy_diffusion.image import flux_kontext_image_scale

    sig = inspect.signature(flux_kontext_image_scale)
    assert "image" in sig.parameters


def test_flux_kontext_image_scale_selects_nearest_resolution() -> None:
    """Should pick (1024, 1024) for a square image and call common_upscale."""
    fake_comfy_utils = MagicMock()
    upscaled = MagicMock()
    upscaled.movedim.return_value = MagicMock()
    fake_comfy_utils.common_upscale.return_value = upscaled

    # Square 512x512 image — nearest square resolution is (1024, 1024)
    fake_image = MagicMock()
    fake_image.shape = (1, 512, 512, 3)  # B H W C

    preferred = [(1024, 1024), (672, 1568)]

    with patch(
        "comfy_diffusion.image._get_flux_kontext_dependencies",
        return_value=(fake_comfy_utils, preferred),
    ):
        from comfy_diffusion.image import flux_kontext_image_scale

        result = flux_kontext_image_scale(fake_image)

    fake_comfy_utils.common_upscale.assert_called_once()
    call_args = fake_comfy_utils.common_upscale.call_args[0]
    # width and height should be 1024
    assert call_args[1] == 1024
    assert call_args[2] == 1024
    assert call_args[3] == "lanczos"
    assert call_args[4] == "center"
    assert result is upscaled.movedim.return_value


# ---------------------------------------------------------------------------
# AC04: apply_cfg_norm
# ---------------------------------------------------------------------------


def test_apply_cfg_norm_exists() -> None:
    import comfy_diffusion.video as mod

    assert hasattr(mod, "apply_cfg_norm")
    assert callable(mod.apply_cfg_norm)


def test_apply_cfg_norm_in_dunder_all() -> None:
    import comfy_diffusion.video as mod

    assert "apply_cfg_norm" in mod.__all__


def test_apply_cfg_norm_signature() -> None:
    from comfy_diffusion.video import apply_cfg_norm

    sig = inspect.signature(apply_cfg_norm)
    params = sig.parameters
    assert "model" in params
    assert "strength" in params
    assert params["strength"].default == 1.0


def test_apply_cfg_norm_clones_model_and_installs_hook() -> None:
    """Should clone the model and register a post-cfg function."""
    fake_model = MagicMock()
    cloned = MagicMock()
    fake_model.clone.return_value = cloned

    import comfy_diffusion.video as video_mod

    result = video_mod.apply_cfg_norm(fake_model, strength=1.0)

    fake_model.clone.assert_called_once()
    cloned.set_model_sampler_post_cfg_function.assert_called_once()
    assert result is cloned


def test_apply_cfg_norm_hook_normalises_magnitude() -> None:
    """The installed hook should scale denoised by cond_norm / pred_norm."""
    import torch

    fake_model = MagicMock()
    cloned = MagicMock()
    fake_model.clone.return_value = cloned

    captured_hook = None

    def capture_hook(fn: object) -> None:
        nonlocal captured_hook
        captured_hook = fn

    cloned.set_model_sampler_post_cfg_function.side_effect = capture_hook

    import comfy_diffusion.video as video_mod

    video_mod.apply_cfg_norm(fake_model, strength=1.0)

    assert captured_hook is not None, "Hook was not registered"

    # Build simple tensors: cond_denoised and denoised both (1, 4, 2, 2)
    cond_p = torch.ones(1, 4, 2, 2)
    pred = torch.ones(1, 4, 2, 2) * 2.0

    out = captured_hook({"cond_denoised": cond_p, "denoised": pred})
    # norm(cond_p) == 4.0, norm(pred) == 8.0 → scale = 0.5, clamped to 0.5
    # result should be pred * 0.5 * 1.0 == ones
    assert torch.allclose(out, torch.ones(1, 4, 2, 2), atol=1e-5)


# ---------------------------------------------------------------------------
# AC05: No top-level comfy.* or torch imports in modified modules
# ---------------------------------------------------------------------------


def test_conditioning_no_top_level_comfy_import() -> None:
    top_level = _get_top_level_imports(_CONDITIONING_FILE)
    bad = [m for m in top_level if m.startswith("comfy") or m == "torch"]
    assert not bad, f"conditioning.py has top-level comfy/torch imports: {bad}"


def test_image_no_top_level_comfy_import() -> None:
    top_level = _get_top_level_imports(_IMAGE_FILE)
    bad = [m for m in top_level if m.startswith("comfy") or m == "torch"]
    assert not bad, f"image.py has top-level comfy/torch imports: {bad}"


def test_video_no_top_level_comfy_import() -> None:
    top_level = _get_top_level_imports(_VIDEO_FILE)
    bad = [m for m in top_level if m.startswith("comfy") or m == "torch"]
    assert not bad, f"video.py has top-level comfy/torch imports: {bad}"
