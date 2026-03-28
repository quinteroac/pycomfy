"""CPU-only tests for ltxv_img_to_video_inplace().

Mirrors LTXVImgToVideoInplace.execute() behaviour; all tests run without
a GPU or real model weights by mocking comfy.utils and providing a fake VAE.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from comfy_diffusion.video import ltxv_img_to_video_inplace


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_fake_comfy_utils(monkeypatch: Any, upscale_fn: Any = None) -> None:
    """Inject a fake ``comfy`` + ``comfy.utils`` into sys.modules."""
    fake_comfy = types.ModuleType("comfy")
    fake_utils = types.ModuleType("comfy.utils")

    if upscale_fn is None:
        # Default: return the tensor unchanged (same as no-resize case)
        def upscale_fn(img: Any, w: int, h: int, mode: str, crop: str) -> Any:
            return img

    fake_utils.common_upscale = upscale_fn
    fake_comfy.utils = fake_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)


class _FakeVAE:
    """Minimal VAE stub with configurable downscale_index_formula."""

    def __init__(self, scale: int = 8, encoded_frames: int = 1) -> None:
        self.downscale_index_formula = (1, scale, scale)
        self._scale = scale
        self._encoded_frames: int | None = encoded_frames

    def encode(self, pixels: Any) -> Any:
        import torch

        batch = pixels.shape[0]
        lh = pixels.shape[1] // self._scale
        lw = pixels.shape[2] // self._scale
        frames = self._encoded_frames or 1
        # Mimic encoded latent shape: (batch, 16, frames, lh, lw)
        return torch.zeros(batch, 16, frames, lh, lw)


# ---------------------------------------------------------------------------
# AC: bypass=True path — returns unmodified latent without calling vae
# ---------------------------------------------------------------------------


def test_bypass_returns_input_latent_unchanged() -> None:
    """bypass=True must return the exact latent dict passed in."""
    sentinel: dict[str, Any] = {"samples": object(), "extra_key": "preserved"}
    result = ltxv_img_to_video_inplace(
        vae=object(), image=object(), latent=sentinel, bypass=True
    )
    assert result is sentinel


def test_bypass_does_not_invoke_vae_encode(monkeypatch: Any) -> None:
    """bypass=True must not call vae.encode under any circumstances."""

    class StrictVAE:
        downscale_index_formula = (1, 8, 8)

        def encode(self, _pixels: Any) -> Any:
            raise AssertionError("vae.encode must not be called when bypass=True")

    latent: dict[str, Any] = {"samples": object()}
    result = ltxv_img_to_video_inplace(
        vae=StrictVAE(), image=object(), latent=latent, bypass=True
    )
    assert result is latent


# ---------------------------------------------------------------------------
# AC: normal path — returns dict with `samples` and `noise_mask`
# ---------------------------------------------------------------------------


def test_normal_path_returns_samples_and_noise_mask_keys(monkeypatch: Any) -> None:
    """Normal call must return a dict containing 'samples' and 'noise_mask'."""
    torch = pytest.importorskip("torch")
    _make_fake_comfy_utils(monkeypatch)

    batch, channels, latent_frames, lh, lw = 1, 16, 6, 4, 4
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}
    image = torch.zeros(batch, lh * 8, lw * 8, 3)

    result = ltxv_img_to_video_inplace(
        vae=_FakeVAE(scale=8, encoded_frames=1),
        image=image,
        latent=latent,
        strength=1.0,
    )

    assert "samples" in result
    assert "noise_mask" in result


def test_normal_path_noise_mask_shape(monkeypatch: Any) -> None:
    """noise_mask shape must be (batch, 1, latent_frames, 1, 1)."""
    torch = pytest.importorskip("torch")
    _make_fake_comfy_utils(monkeypatch)

    batch, channels, latent_frames, lh, lw = 2, 16, 8, 3, 3
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}
    image = torch.zeros(batch, lh * 8, lw * 8, 3)

    result = ltxv_img_to_video_inplace(
        vae=_FakeVAE(scale=8, encoded_frames=1),
        image=image,
        latent=latent,
        strength=1.0,
    )

    mask = result["noise_mask"]
    assert mask.shape == (batch, 1, latent_frames, 1, 1)


def test_normal_path_noise_mask_values_reflect_strength(monkeypatch: Any) -> None:
    """First encoded frames in noise_mask must equal 1.0 - strength; rest 1.0."""
    torch = pytest.importorskip("torch")
    _make_fake_comfy_utils(monkeypatch)

    batch, channels, latent_frames, lh, lw = 1, 16, 5, 2, 2
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}
    image = torch.zeros(batch, lh * 8, lw * 8, 3)
    strength = 0.7

    result = ltxv_img_to_video_inplace(
        vae=_FakeVAE(scale=8, encoded_frames=1),
        image=image,
        latent=latent,
        strength=strength,
    )

    mask = result["noise_mask"]
    # Frame 0 (encoded): 1.0 - strength
    assert float(mask[0, 0, 0, 0, 0]) == pytest.approx(1.0 - strength, abs=1e-5)
    # Remaining frames: 1.0
    for f in range(1, latent_frames):
        assert float(mask[0, 0, f, 0, 0]) == pytest.approx(1.0, abs=1e-5)


def test_normal_path_samples_are_modified_inplace(monkeypatch: Any) -> None:
    """samples tensor in the returned dict must be the same object (inplace)."""
    torch = pytest.importorskip("torch")
    _make_fake_comfy_utils(monkeypatch)

    batch, channels, latent_frames, lh, lw = 1, 16, 4, 2, 2
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}
    image = torch.zeros(batch, lh * 8, lw * 8, 3)

    result = ltxv_img_to_video_inplace(
        vae=_FakeVAE(scale=8, encoded_frames=1),
        image=image,
        latent=latent,
    )

    # The returned samples must be the same tensor object (modified in-place)
    assert result["samples"] is samples


# ---------------------------------------------------------------------------
# AC: auto-resize path — image shape differs from latent spatial dims
# ---------------------------------------------------------------------------


def test_autoresize_calls_common_upscale_when_image_size_differs(
    monkeypatch: Any,
) -> None:
    """When image.shape[1:3] != (height, width), common_upscale must be called."""
    torch = pytest.importorskip("torch")

    upscale_calls: list[tuple[int, int, str, str]] = []

    def tracking_upscale(img: Any, w: int, h: int, mode: str, crop: str) -> Any:
        upscale_calls.append((w, h, mode, crop))
        # Return img with the target spatial dimensions after movedim
        # img is (batch, channels, img_h, img_w) after movedim(-1,1)
        batch, ch = img.shape[0], img.shape[1]
        return torch.zeros(batch, ch, h, w)

    _make_fake_comfy_utils(monkeypatch, upscale_fn=tracking_upscale)

    batch, channels, latent_frames, lh, lw = 1, 16, 4, 4, 4
    scale = 8
    target_h = lh * scale  # 32
    target_w = lw * scale  # 32

    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}

    # Image is smaller than required (16x16 instead of 32x32)
    small_image = torch.zeros(batch, 16, 16, 3)

    result = ltxv_img_to_video_inplace(
        vae=_FakeVAE(scale=scale, encoded_frames=1),
        image=small_image,
        latent=latent,
        strength=1.0,
    )

    assert len(upscale_calls) == 1
    assert upscale_calls[0] == (target_w, target_h, "bilinear", "center")
    assert "samples" in result
    assert "noise_mask" in result


def test_autoresize_not_called_when_image_already_correct_size(
    monkeypatch: Any,
) -> None:
    """When image.shape[1:3] == (height, width), common_upscale must NOT be called."""
    torch = pytest.importorskip("torch")

    upscale_calls: list[Any] = []

    def tracking_upscale(img: Any, w: int, h: int, mode: str, crop: str) -> Any:
        upscale_calls.append((w, h))
        return img

    _make_fake_comfy_utils(monkeypatch, upscale_fn=tracking_upscale)

    batch, channels, latent_frames, lh, lw = 1, 16, 4, 4, 4
    scale = 8
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}

    # Image already at the correct pixel size
    correct_image = torch.zeros(batch, lh * scale, lw * scale, 3)

    ltxv_img_to_video_inplace(
        vae=_FakeVAE(scale=scale, encoded_frames=1),
        image=correct_image,
        latent=latent,
    )

    assert upscale_calls == [], "common_upscale must not be called when image is correct size"
