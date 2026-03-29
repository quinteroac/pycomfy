"""Image loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from PIL import Image as PILImage


def _get_load_image_dependencies() -> tuple[Any, Any, Any]:
    import torch
    from PIL import Image, ImageOps

    return Image, ImageOps, torch


def _get_torch_module() -> Any:
    import torch

    return torch


def _get_image_upscale_with_model_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

    return ImageUpscaleWithModel


def _get_repeat_image_batch_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_images import RepeatImageBatch

    return RepeatImageBatch


def _get_image_from_batch_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_images import ImageFromBatch

    return ImageFromBatch


def _get_image_composite_masked_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_mask import ImageCompositeMasked

    return ImageCompositeMasked


def _get_resize_image_mask_node_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_post_processing import ResizeImageMaskNode

    return ResizeImageMaskNode


def _get_empty_image_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from nodes import EmptyImage

    return EmptyImage


def _get_canny_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_canny import Canny

    return Canny


def _get_image_invert_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from nodes import ImageInvert

    return ImageInvert


def _get_math_expression_node_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_math import MathExpressionNode

    return MathExpressionNode


def _get_resize_images_by_longer_edge_node_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_dataset import ResizeImagesByLongerEdgeNode

    return ResizeImagesByLongerEdgeNode


def _get_ltxv_preprocess_dependencies() -> tuple[Any, Any]:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.utils
    from comfy_extras.nodes_lt import LTXVPreprocess

    return comfy.utils, LTXVPreprocess


def _unwrap_node_output(output: Any) -> Any:
    result = getattr(output, "result", output)
    return result[0]


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


def image_pad_for_outpaint(
    image: Any, left: int, top: int, right: int, bottom: int, feathering: int
) -> tuple[Any, Any]:
    """Pad an image and return a corresponding outpaint mask."""
    if left < 0 or top < 0 or right < 0 or bottom < 0:
        raise ValueError("left, top, right, and bottom must be non-negative integers")
    if feathering < 0:
        raise ValueError("feathering must be a non-negative integer")

    torch = _get_torch_module()

    batch, height, width, channels = image.size()

    padded_image = torch.ones(
        (batch, height + top + bottom, width + left + right, channels),
        dtype=torch.float32,
    ) * 0.5
    padded_image[:, top : top + height, left : left + width, :] = image

    padded_mask = torch.ones(
        (height + top + bottom, width + left + right),
        dtype=torch.float32,
    )
    inner_mask = torch.zeros((height, width), dtype=torch.float32)

    if feathering > 0 and feathering * 2 < height and feathering * 2 < width:
        for row in range(height):
            for col in range(width):
                distance_top = row if top != 0 else height
                distance_bottom = height - row if bottom != 0 else height
                distance_left = col if left != 0 else width
                distance_right = width - col if right != 0 else width

                distance = min(distance_top, distance_bottom, distance_left, distance_right)
                if distance >= feathering:
                    continue

                blend = (feathering - distance) / feathering
                inner_mask[row, col] = blend * blend

    padded_mask[top : top + height, left : left + width] = inner_mask
    return padded_image, padded_mask.unsqueeze(0)


def image_upscale_with_model(upscale_model: Any, image: Any) -> Any:
    """Upscale a BHWC image tensor using a ComfyUI-compatible upscale model."""
    image_upscale_with_model_type = _get_image_upscale_with_model_type()
    return _unwrap_node_output(image_upscale_with_model_type.execute(upscale_model, image))


def image_from_batch(image: Any, batch_index: int, length: int = 1) -> Any:
    """Extract a contiguous subset from the image batch dimension."""
    image_from_batch_type = _get_image_from_batch_type()
    return _unwrap_node_output(
        image_from_batch_type.execute(image=image, batch_index=batch_index, length=length)
    )


def repeat_image_batch(image: Any, amount: int) -> Any:
    """Repeat image tensors along the batch dimension."""
    repeat_image_batch_type = _get_repeat_image_batch_type()
    return _unwrap_node_output(repeat_image_batch_type.execute(image=image, amount=amount))


def image_composite_masked(destination: Any, source: Any, mask: Any, x: int, y: int) -> Any:
    """Composite source image onto destination image with mask-guided blending."""
    image_composite_masked_type = _get_image_composite_masked_type()
    return _unwrap_node_output(
        image_composite_masked_type.execute(
            destination=destination,
            source=source,
            x=x,
            y=y,
            resize_source=False,
            mask=mask,
        )
    )


def image_to_tensor(image: PILImage.Image) -> Any:
    """Convert a PIL Image to a BHWC float32 tensor with shape (1, H, W, 3)."""
    torch = _get_torch_module()
    rgb = image.convert("RGB")
    rows = _pixels_to_float_rows(rgb, channels=3)
    return torch.tensor([rows], dtype=torch.float32)


def ltxv_preprocess(image: Any, width: int, height: int) -> Any:
    """Preprocess image batch for LTXV img2vid with center resize and node compression."""
    comfy_utils, ltxv_preprocess_type = _get_ltxv_preprocess_dependencies()
    resized = comfy_utils.common_upscale(
        image.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    return _unwrap_node_output(ltxv_preprocess_type.execute(resized, img_compression=35))


def resize_image_mask(
    image: Any,
    mask: Any,
    width: int,
    height: int,
    interpolation: str = "bilinear",
) -> tuple[Any, Any]:
    """Resize an image and its mask to the given dimensions in a single call."""
    resize_image_mask_node_type = _get_resize_image_mask_node_type()
    result = resize_image_mask_node_type.execute(
        image=image,
        mask=mask,
        width=width,
        height=height,
        interpolation=interpolation,
    )
    raw = getattr(result, "result", result)
    return raw[0], raw[1]


def resize_images_by_longer_edge(images: Any, size: int) -> Any:
    """Resize images so the longer dimension equals `size`, preserving aspect ratio."""
    resize_images_by_longer_edge_node_type = _get_resize_images_by_longer_edge_node_type()
    return _unwrap_node_output(
        resize_images_by_longer_edge_node_type.execute(images=images, size=size)
    )


def empty_image(width: int, height: int, batch_size: int = 1, color: int = 0) -> Any:
    """Create a solid-color blank image tensor with shape (batch_size, height, width, 3)."""
    empty_image_type = _get_empty_image_type()
    return _unwrap_node_output(
        empty_image_type.execute(width=width, height=height, batch_size=batch_size, color=color)
    )


def canny(image: Any, low_threshold: int = 100, high_threshold: int = 200) -> Any:
    """Apply Canny edge detection and return an image tensor of the same spatial dimensions."""
    canny_type = _get_canny_type()
    return _unwrap_node_output(
        canny_type.execute(
            image=image,
            low_threshold=low_threshold / 255.0,
            high_threshold=high_threshold / 255.0,
        )
    )


def image_invert(image: Any) -> Any:
    """Invert pixel values (1 − pixel) of an image tensor."""
    image_invert_type = _get_image_invert_type()
    return _unwrap_node_output(image_invert_type.execute(image=image))


def math_expression(expression: str, **kwargs: float) -> int | float:
    """Evaluate a parameterised math expression via ComfyMathExpression."""
    math_expression_node_type = _get_math_expression_node_type()
    result = math_expression_node_type.execute(expression=expression, values=kwargs)
    raw = getattr(result, "result", result)
    return raw[0]


__all__ = [
    "load_image",
    "image_to_tensor",
    "image_pad_for_outpaint",
    "image_upscale_with_model",
    "image_from_batch",
    "repeat_image_batch",
    "image_composite_masked",
    "ltxv_preprocess",
    "resize_image_mask",
    "resize_images_by_longer_edge",
    "empty_image",
    "math_expression",
    "canny",
    "image_invert",
]
