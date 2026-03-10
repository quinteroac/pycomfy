"""Image loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


def _get_load_image_dependencies() -> tuple[Any, Any, Any]:
    import torch
    from PIL import Image, ImageOps

    return Image, ImageOps, torch


def _pixels_to_float_rows(image: Any, *, channels: int) -> list[list[list[float]]]:
    width, height = image.size
    pixels = image.load()
    if pixels is None:
        raise ValueError("unable to access image pixels")

    rows: list[list[list[float]]] = []
    for y in range(height):
        row: list[list[float]] = []
        for x in range(width):
            pixel = cast(tuple[int, ...], pixels[x, y])
            row.append([channel / 255.0 for channel in pixel[:channels]])
        rows.append(row)
    return rows


def _opaque_mask(height: int, width: int) -> list[list[float]]:
    return [[0.0 for _ in range(width)] for _ in range(height)]


def _alpha_to_mask_rows(rgba_image: Any) -> list[list[float]]:
    width, height = rgba_image.size
    pixels = rgba_image.load()
    if pixels is None:
        raise ValueError("unable to access image pixels")

    rows: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        for x in range(width):
            alpha = cast(tuple[int, int, int, int], pixels[x, y])[3]
            row.append((255 - alpha) / 255.0)
        rows.append(row)
    return rows


def load_image(path: str | Path) -> tuple[Any, Any]:
    """Load an image file into ComfyUI-compatible IMAGE/MASK tensors."""
    image_path = Path(path)
    Image, ImageOps, torch = _get_load_image_dependencies()

    with Image.open(image_path) as source:
        oriented = ImageOps.exif_transpose(source)
        rgb = oriented.convert("RGB")
        width, height = rgb.size
        image_rows = _pixels_to_float_rows(rgb, channels=3)

        has_alpha = "A" in oriented.getbands() or (
            oriented.mode == "P" and "transparency" in oriented.info
        )
        if has_alpha:
            mask_values = _alpha_to_mask_rows(oriented.convert("RGBA"))
        else:
            mask_values = _opaque_mask(height, width)

    image_tensor = torch.tensor([image_rows], dtype=torch.float32)
    mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
    return image_tensor, mask_tensor


__all__ = ["load_image"]
