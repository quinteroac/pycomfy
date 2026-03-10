"""Mask loading and manipulation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from .image import (
    _alpha_to_mask_rows,
    _get_load_image_dependencies,
    _get_torch_module,
    _unwrap_node_output,
)


def _get_grow_mask_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_mask import GrowMask

    return GrowMask


def _rgb_channel_to_mask_rows(rgb_image: Any, channel_index: int) -> list[list[float]]:
    width, height = rgb_image.size
    pixels = rgb_image.load()
    if pixels is None:
        raise ValueError("unable to access image pixels")

    rows: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        for x in range(width):
            channel_value = cast(tuple[int, int, int], pixels[x, y])[channel_index]
            row.append(channel_value / 255.0)
        rows.append(row)
    return rows


def load_image_mask(path: str | Path, channel: str) -> Any:
    """Load one image channel as a ComfyUI-compatible MASK tensor with shape (1, H, W)."""
    channel_indices = {
        "red": 0,
        "green": 1,
        "blue": 2,
    }
    if channel != "alpha" and channel not in channel_indices:
        raise ValueError("channel must be one of: alpha, red, green, blue")

    image_path = Path(path)
    Image, ImageOps, torch = _get_load_image_dependencies()

    with Image.open(image_path) as source:
        oriented = ImageOps.exif_transpose(source)
        if channel == "alpha":
            mask_values = _alpha_to_mask_rows(oriented.convert("RGBA"))
        else:
            channel_index = channel_indices[channel]
            mask_values = _rgb_channel_to_mask_rows(oriented.convert("RGB"), channel_index)

    return torch.tensor([mask_values], dtype=torch.float32)


def image_to_mask(image: Any, channel: str) -> Any:
    """Convert a BHWC image tensor to a BHW float32 mask tensor by channel."""
    channel_indices = {
        "red": 0,
        "green": 1,
        "blue": 2,
    }
    if channel not in channel_indices:
        raise ValueError("channel must be one of: red, green, blue")

    if len(image.shape) != 4:
        raise ValueError("image tensor must have shape (B, H, W, C)")

    channel_index = channel_indices[channel]
    if image.shape[3] <= channel_index:
        raise ValueError("image tensor channel dimension must be at least 3")

    values = image.tolist()
    mask_values = [
        [
            [float(pixel[channel_index]) for pixel in row]
            for row in batch
        ]
        for batch in values
    ]

    torch = _get_torch_module()
    return torch.tensor(mask_values, dtype=torch.float32)


def mask_to_image(mask: Any) -> Any:
    """Convert a BHW float32 mask tensor to a BHWC float32 image tensor."""
    if len(mask.shape) != 3:
        raise ValueError("mask tensor must have shape (B, H, W)")

    mask_values = mask.tolist()
    image_values = [
        [
            [[float(value), float(value), float(value)] for value in row]
            for row in batch
        ]
        for batch in mask_values
    ]

    torch = _get_torch_module()
    return torch.tensor(image_values, dtype=torch.float32)


def grow_mask(mask: Any, expand: int, tapered_corners: bool) -> Any:
    """Grow or shrink a mask by pixel radius using ComfyUI's GrowMask semantics."""
    grow_mask_type = _get_grow_mask_type()
    return _unwrap_node_output(
        grow_mask_type.execute(mask=mask, expand=expand, tapered_corners=tapered_corners)
    )


def feather_mask(mask: Any, left: int, top: int, right: int, bottom: int) -> Any:
    """Feather mask edges by independent pixel amounts for each side."""
    if left < 0 or top < 0 or right < 0 or bottom < 0:
        raise ValueError("left, top, right, and bottom must be non-negative integers")
    if len(mask.shape) != 3:
        raise ValueError("mask tensor must have shape (B, H, W)")

    _, height, width = mask.shape
    left = min(left, width)
    right = min(right, width)
    top = min(top, height)
    bottom = min(bottom, height)

    torch = _get_torch_module()
    if getattr(torch, "arange", None) is not None and getattr(torch, "where", None) is not None:
        return _feather_mask_native(mask, left, top, right, bottom, height, width, torch)

    return _feather_mask_loop(mask, left, top, right, bottom, height, width, torch)


def _feather_mask_native(
    mask: Any,
    left: int,
    top: int,
    right: int,
    bottom: int,
    height: int,
    width: int,
    torch: Any,
) -> Any:
    device = getattr(mask, "device", None)
    dtype = getattr(mask, "dtype", torch.float32)
    y_grid = torch.arange(height, device=device, dtype=dtype).unsqueeze(1).expand(height, width)
    x_grid = torch.arange(width, device=device, dtype=dtype).unsqueeze(0).expand(height, width)

    feather = torch.ones((height, width), device=device, dtype=dtype)

    if left > 0:
        left_factor = torch.where(
            x_grid < left,
            (x_grid + 1.0) / left,
            torch.ones((height, width), device=device, dtype=dtype),
        )
        feather = feather * left_factor

    if right > 0:
        dist_right = width - 1 - x_grid
        right_factor = torch.where(
            dist_right < right,
            (dist_right + 1.0) / right,
            torch.ones((height, width), device=device, dtype=dtype),
        )
        feather = feather * right_factor

    if top > 0:
        top_factor = torch.where(
            y_grid < top,
            (y_grid + 1.0) / top,
            torch.ones((height, width), device=device, dtype=dtype),
        )
        feather = feather * top_factor

    if bottom > 0:
        dist_bottom = height - 1 - y_grid
        bottom_factor = torch.where(
            dist_bottom < bottom,
            (dist_bottom + 1.0) / bottom,
            torch.ones((height, width), device=device, dtype=dtype),
        )
        feather = feather * bottom_factor

    feathered = mask * feather.unsqueeze(0)
    return feathered.clamp(0.0, 1.0)


def _feather_mask_loop(
    mask: Any,
    left: int,
    top: int,
    right: int,
    bottom: int,
    height: int,
    width: int,
    torch: Any,
) -> Any:
    mask_values = mask.tolist()
    feathered_values: list[list[list[float]]] = []

    for batch in mask_values:
        feathered_batch: list[list[float]] = []
        for y, row in enumerate(batch):
            feathered_row: list[float] = []
            for x, value in enumerate(row):
                feather_rate = 1.0

                if left > 0 and x < left:
                    feather_rate *= (x + 1.0) / left
                if right > 0:
                    distance_from_right = width - 1 - x
                    if distance_from_right < right:
                        feather_rate *= (distance_from_right + 1.0) / right
                if top > 0 and y < top:
                    feather_rate *= (y + 1.0) / top
                if bottom > 0:
                    distance_from_bottom = height - 1 - y
                    if distance_from_bottom < bottom:
                        feather_rate *= (distance_from_bottom + 1.0) / bottom

                feathered_value = float(value) * feather_rate
                feathered_row.append(min(1.0, max(0.0, feathered_value)))
            feathered_batch.append(feathered_row)
        feathered_values.append(feathered_batch)

    return torch.tensor(feathered_values, dtype=torch.float32)


__all__ = [
    "load_image_mask",
    "image_to_mask",
    "mask_to_image",
    "grow_mask",
    "feather_mask",
]
