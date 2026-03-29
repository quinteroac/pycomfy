"""CPU-safe tests for image wrapper functions added in iteration 000029.

Covers:
  - image_resize_kj
  - image_batch_extend_with_overlap
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# __all__ membership
# ---------------------------------------------------------------------------


def test_image_resize_kj_in_all() -> None:
    import comfy_diffusion.image as img

    assert "image_resize_kj" in img.__all__


def test_image_batch_extend_with_overlap_in_all() -> None:
    import comfy_diffusion.image as img

    assert "image_batch_extend_with_overlap" in img.__all__


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_image_wrappers_importable() -> None:
    from comfy_diffusion.image import image_batch_extend_with_overlap, image_resize_kj  # noqa: F401


# ---------------------------------------------------------------------------
# image_resize_kj — signature
# ---------------------------------------------------------------------------


def test_image_resize_kj_signature() -> None:
    from comfy_diffusion.image import image_resize_kj

    sig = inspect.signature(image_resize_kj)
    params = list(sig.parameters)
    assert "image" in params
    assert "width" in params
    assert "height" in params
    assert "upscale_method" in params
    assert "keep_proportion" in params
    assert "pad_color" in params
    assert "crop_position" in params
    assert "divisible_by" in params
    assert "device" in params

    assert sig.parameters["upscale_method"].default == "lanczos"
    assert sig.parameters["keep_proportion"].default == "crop"
    assert sig.parameters["divisible_by"].default == 2
    assert sig.parameters["device"].default == "cpu"


def test_image_resize_kj_returns_tuple() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_resize_kj

    image = torch.zeros(1, 64, 64, 3)
    result = image_resize_kj(image, width=32, height=32)

    assert isinstance(result, tuple)
    assert len(result) == 3


def test_image_resize_kj_stretch_mode_output_dimensions() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_resize_kj

    image = torch.zeros(1, 64, 64, 3)
    out_image, out_w, out_h = image_resize_kj(image, width=32, height=16, keep_proportion="stretch")

    assert out_w == 32
    assert out_h == 16
    assert out_image.shape == (1, 16, 32, 3)


def test_image_resize_kj_output_tensor_has_correct_channels() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_resize_kj

    image = torch.rand(1, 128, 128, 3)
    out_image, out_w, out_h = image_resize_kj(image, width=64, height=64, keep_proportion="stretch")

    assert out_image.shape[-1] == 3


def test_image_resize_kj_resize_mode_preserves_aspect_ratio() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_resize_kj

    # 64x32 image (2:1 aspect ratio) resized to 64x64 target with "resize" mode
    # should fit within 64x64 → output should be 64x32
    image = torch.zeros(1, 32, 64, 3)
    out_image, out_w, out_h = image_resize_kj(image, width=64, height=64, keep_proportion="resize")

    # Width should be 64, height should preserve ratio → 32
    assert out_w == 64
    assert out_h == 32


def test_image_resize_kj_divisible_by_constraint() -> None:
    """Output dimensions must be divisible by divisible_by."""
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_resize_kj

    image = torch.zeros(1, 64, 64, 3)
    out_image, out_w, out_h = image_resize_kj(
        image, width=30, height=30, keep_proportion="stretch", divisible_by=8
    )

    assert out_w % 8 == 0
    assert out_h % 8 == 0


# ---------------------------------------------------------------------------
# image_batch_extend_with_overlap — signature
# ---------------------------------------------------------------------------


def test_image_batch_extend_with_overlap_signature() -> None:
    from comfy_diffusion.image import image_batch_extend_with_overlap

    sig = inspect.signature(image_batch_extend_with_overlap)
    params = list(sig.parameters)
    assert "source_images" in params
    assert "new_images" in params
    assert "overlap" in params
    assert "overlap_side" in params
    assert "overlap_mode" in params

    assert sig.parameters["new_images"].default is None
    assert sig.parameters["overlap"].default == 13
    assert sig.parameters["overlap_side"].default == "source"
    assert sig.parameters["overlap_mode"].default == "filmic_crossfade"


# ---------------------------------------------------------------------------
# image_batch_extend_with_overlap — behaviour
# ---------------------------------------------------------------------------


def test_image_batch_extend_no_new_images_returns_zeros() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_batch_extend_with_overlap

    source = torch.zeros(5, 64, 64, 3)
    # Use overlap < len(source) so the early-return-source branch is skipped
    result = image_batch_extend_with_overlap(source, new_images=None, overlap=1)

    assert result.shape == (1, 64, 64, 3)


def test_image_batch_extend_linear_blend() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_batch_extend_with_overlap

    source = torch.zeros(10, 32, 32, 3)
    new_imgs = torch.ones(10, 32, 32, 3)
    overlap = 3

    result = image_batch_extend_with_overlap(
        source, new_images=new_imgs, overlap=overlap, overlap_mode="linear_blend"
    )

    # Total frames = (10 - 3) + 10 = 17
    expected_frames = (source.shape[0] - overlap) + new_imgs.shape[0]
    assert result.shape[0] == expected_frames
    assert result.shape[1:] == (32, 32, 3)


def test_image_batch_extend_filmic_crossfade() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_batch_extend_with_overlap

    source = torch.zeros(8, 16, 16, 3)
    new_imgs = torch.ones(8, 16, 16, 3)
    overlap = 2

    result = image_batch_extend_with_overlap(
        source, new_images=new_imgs, overlap=overlap, overlap_mode="filmic_crossfade"
    )

    expected_frames = (source.shape[0] - overlap) + new_imgs.shape[0]
    assert result.shape[0] == expected_frames


def test_image_batch_extend_cut_mode() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_batch_extend_with_overlap

    source = torch.zeros(10, 16, 16, 3)
    new_imgs = torch.ones(10, 16, 16, 3)
    overlap = 4

    result = image_batch_extend_with_overlap(
        source, new_images=new_imgs, overlap=overlap, overlap_mode="cut"
    )

    expected_frames = (source.shape[0] - overlap) + new_imgs.shape[0]
    assert result.shape[0] == expected_frames


def test_image_batch_extend_ease_in_out_mode() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_batch_extend_with_overlap

    source = torch.zeros(6, 8, 8, 3)
    new_imgs = torch.ones(6, 8, 8, 3)
    overlap = 2

    result = image_batch_extend_with_overlap(
        source, new_images=new_imgs, overlap=overlap, overlap_mode="ease_in_out"
    )

    expected_frames = (source.shape[0] - overlap) + new_imgs.shape[0]
    assert result.shape[0] == expected_frames


def test_image_batch_extend_mismatched_spatial_dims_raises() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_batch_extend_with_overlap

    source = torch.zeros(5, 64, 64, 3)
    new_imgs = torch.zeros(5, 32, 32, 3)

    with pytest.raises(ValueError, match="same spatial dimensions"):
        image_batch_extend_with_overlap(source, new_images=new_imgs, overlap=2)


def test_image_batch_extend_overlap_greater_than_source_returns_source() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.image import image_batch_extend_with_overlap

    source = torch.zeros(3, 16, 16, 3)
    new_imgs = torch.ones(5, 16, 16, 3)

    result = image_batch_extend_with_overlap(source, new_images=new_imgs, overlap=10)

    # overlap > len(source_images) → return source unchanged
    assert result is source
