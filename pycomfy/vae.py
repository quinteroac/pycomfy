"""VAE decode helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from PIL import Image


class _VaeDecoder(Protocol):
    def decode(self, samples: Any) -> Any: ...


class _VaeEncoder(Protocol):
    def encode(self, pixel_samples: Any) -> Any: ...


class _VaeEncoderTiled(Protocol):
    def encode_tiled(
        self,
        pixel_samples: Any,
        *,
        tile_x: int,
        tile_y: int,
        overlap: int,
    ) -> Any: ...


class _VaeDecoderTiled(Protocol):
    def decode_tiled(
        self,
        samples: Any,
        *,
        tile_x: int,
        tile_y: int,
        overlap: int,
    ) -> Any: ...


class _ListTensor:
    """Minimal tensor-like wrapper used when torch is unavailable."""

    def __init__(self, data: list[list[list[list[float]]]]) -> None:
        self._data = data

    def tolist(self) -> list[list[list[list[float]]]]:
        return self._data


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


def vae_decode_tiled(
    vae: _VaeDecoderTiled,
    latent: Mapping[str, Any],
    tile_size: int = 512,
    overlap: int = 64,
) -> Image.Image:
    """Decode a ComfyUI LATENT dict into a PIL image using tiled decode."""
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[0]

    images = vae.decode_tiled(samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    image = images[0]
    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    return _tensor_like_to_pil(image)


def _pil_to_batched_hwc(image: Image.Image) -> list[list[list[list[float]]]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    if pixels is None:
        raise ValueError("unable to access image pixels")

    rows: list[list[list[float]]] = []
    for y in range(height):
        row: list[list[float]] = []
        for x in range(width):
            r, g, b = cast(tuple[int, int, int], pixels[x, y])
            row.append([r / 255.0, g / 255.0, b / 255.0])
        rows.append(row)

    return [rows]


def _image_to_tensor_like(image: Image.Image) -> Any:
    batched_hwc = _pil_to_batched_hwc(image)
    try:
        import torch
    except ModuleNotFoundError:
        return _ListTensor(batched_hwc)

    return torch.tensor(batched_hwc, dtype=torch.float32)


def vae_encode(vae: _VaeEncoder, image: Image.Image) -> dict[str, Any]:
    """Encode a PIL image into a ComfyUI LATENT dict."""
    pixel_samples = _image_to_tensor_like(image)
    samples = vae.encode(pixel_samples)
    return {"samples": samples}


def vae_encode_tiled(
    vae: _VaeEncoderTiled,
    image: Image.Image,
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, Any]:
    """Encode a PIL image into a ComfyUI LATENT dict using tiled encode."""
    pixel_samples = _image_to_tensor_like(image)
    samples = vae.encode_tiled(pixel_samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
    return {"samples": samples}


__all__ = ["vae_decode", "vae_decode_tiled", "vae_encode", "vae_encode_tiled"]
