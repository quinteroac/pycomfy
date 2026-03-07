"""VAE decode helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from PIL import Image


class _VaeDecoder(Protocol):
    def decode(self, samples: Any) -> Any: ...


def _clip_to_uint8(value: float) -> int:
    scaled = int(value * 255.0)
    if scaled < 0:
        return 0
    if scaled > 255:
        return 255
    return scaled


def _tensor_like_to_pil(image: Any) -> Image.Image:
    values = image.tolist()

    if not values or not values[0]:
        raise ValueError("decoded image tensor is empty")

    if isinstance(values[0][0], list):
        height = len(values)
        width = len(values[0])
        channels = len(values[0][0])

        if channels == 1:
            gray_pixels = [_clip_to_uint8(pixel[0]) for row in values for pixel in row]
            result = Image.new("L", (width, height))
            result.putdata(gray_pixels)
            return result

        if channels == 3:
            rgb_pixels: list[tuple[int, int, int]] = []
            for row in values:
                for pixel in row:
                    rgb_pixels.append(
                        (
                            _clip_to_uint8(pixel[0]),
                            _clip_to_uint8(pixel[1]),
                            _clip_to_uint8(pixel[2]),
                        )
                    )
            result = Image.new("RGB", (width, height))
            result.putdata(rgb_pixels)
            return result

        if channels == 4:
            rgba_pixels: list[tuple[int, int, int, int]] = []
            for row in values:
                for pixel in row:
                    rgba_pixels.append(
                        (
                            _clip_to_uint8(pixel[0]),
                            _clip_to_uint8(pixel[1]),
                            _clip_to_uint8(pixel[2]),
                            _clip_to_uint8(pixel[3]),
                        )
                    )
            result = Image.new("RGBA", (width, height))
            result.putdata(rgba_pixels)
            return result

        raise ValueError(f"unsupported channel count: {channels}")

    if isinstance(values[0], list):
        height = len(values)
        width = len(values[0])
        pixels = [_clip_to_uint8(pixel) for row in values for pixel in row]
        result = Image.new("L", (width, height))
        result.putdata(pixels)
        return result

    raise ValueError("unsupported decoded image shape")


def vae_decode(vae: _VaeDecoder, latent: Mapping[str, Any]) -> Image.Image:
    """Decode a ComfyUI LATENT dict into a PIL image."""
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[0]

    images = vae.decode(samples)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    image = images[0]
    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    return _tensor_like_to_pil(image)


__all__ = ["vae_decode"]
