"""VAE decode helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from PIL import Image


class _VaeDecoder(Protocol):
    def decode(self, samples: Any) -> Any: ...


class _VaeEncoder(Protocol):
    def encode(self, pixel_samples: Any) -> Any: ...


class _VaeEncoderForInpaint(Protocol):
    def spacial_compression_encode(self) -> int: ...

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
    """Decode a ComfyUI LATENT dict into a PIL image using tiled decode.

    Args:
        vae: VAE model with a ``decode_tiled`` method.
        latent: ComfyUI LATENT dict containing a ``"samples"`` tensor.
        tile_size: Width and height of each decode tile in pixels. Defaults to 512.
        overlap: Number of pixels of overlap between adjacent tiles. Defaults to 64.

    Returns:
        Decoded image as a PIL ``Image``.
    """
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


def vae_decode_batch(vae: _VaeDecoder, latent: Mapping[str, Any]) -> list[Image.Image]:
    """Decode a ComfyUI LATENT dict into a flat list of PIL images."""
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[0]

    sample_dims = len(samples.shape)
    if sample_dims not in (4, 5):
        raise ValueError("latent samples must be 4D or 5D")

    images = vae.decode(samples)
    image_dims = len(images.shape)
    if image_dims == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    elif image_dims != 4:
        raise ValueError("unsupported decoded image shape")

    if images.shape[0] == 0:
        raise ValueError("decoded image tensor is empty")

    result: list[Image.Image] = []
    for index in range(images.shape[0]):
        image = images[index]
        if hasattr(image, "detach"):
            image = image.detach()
        if hasattr(image, "cpu"):
            image = image.cpu()
        result.append(_tensor_like_to_pil(image))

    if not result:
        raise ValueError("decoded image tensor is empty")
    return result


def vae_decode_batch_tiled(
    vae: _VaeDecoderTiled,
    latent: Mapping[str, Any],
    tile_size: int = 512,
    overlap: int = 64,
) -> list[Image.Image]:
    """Decode a ComfyUI LATENT dict into a flat list of PIL images using tiled decode."""
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[0]

    sample_dims = len(samples.shape)
    if sample_dims not in (4, 5):
        raise ValueError("latent samples must be 4D or 5D")

    if samples.shape[0] == 0:
        raise ValueError("latent samples must not be empty")

    result: list[Image.Image] = []

    # process_output uses in-place ops that fail on inference tensors.
    # Patch the instance attribute temporarily to use non-in-place equivalents.
    _orig_process_output = getattr(vae, "process_output", None)
    if _orig_process_output is not None:
        vae.process_output = lambda image: (image + 1.0).div_(2.0).clamp_(0.0, 1.0)

    try:
        if sample_dims == 5:
            # (B, T, C, H, W) video latent — flatten batch×time, then decode frame-by-frame.
            samples = samples.reshape(-1, samples.shape[-3], samples.shape[-2], samples.shape[-1])
            sample_dims = 4

        # 4D latents: (B, C, H, W). Decode one item at a time, adding a
        # temporal dim so video VAEs can call decode_tiled_3d internally.
        for index in range(samples.shape[0]):
            frame_samples = samples[index : index + 1]
            if hasattr(frame_samples, "unsqueeze"):
                # (1, C, H, W) -> (1, C, 1, H, W)
                frame_samples = frame_samples.unsqueeze(2)
            images = vae.decode_tiled(frame_samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
            image_dims = len(images.shape)
            if image_dims == 5:
                # (B, T, H, W, C) — extract frames
                if hasattr(images, "detach"):
                    images = images.detach().cpu()
                for t in range(images.shape[1]):
                    result.append(_tensor_like_to_pil(images[0][t]))
            elif image_dims == 4:
                # (B, H, W, C) — single frame
                if hasattr(images, "detach"):
                    images = images.detach().cpu()
                result.append(_tensor_like_to_pil(images[0]))
            else:
                raise ValueError("unsupported decoded image shape")
    finally:
        if _orig_process_output is not None:
            vae.process_output = _orig_process_output

    if not result:
        raise ValueError("decoded image tensor is empty")

    return result


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


def _mask_to_tensor(mask: Any) -> Any:
    if isinstance(mask, Image.Image):
        width, height = mask.size
        grayscale = mask.convert("L")
        pixels = grayscale.load()
        if pixels is None:
            raise ValueError("unable to access mask pixels")

        rows: list[list[float]] = []
        for y in range(height):
            row: list[float] = []
            for x in range(width):
                value = cast(int, pixels[x, y])
                row.append(value / 255.0)
            rows.append(row)

        import torch

        return torch.tensor(rows, dtype=torch.float32)

    return mask


def _images_to_tensor_like(images: list[Image.Image]) -> Any:
    stacked_hwc: list[list[list[list[float]]]] = []
    for image in images:
        stacked_hwc.append(_pil_to_batched_hwc(image)[0])

    try:
        import torch
    except ModuleNotFoundError:
        return _ListTensor(stacked_hwc)

    return torch.tensor(stacked_hwc, dtype=torch.float32)


def _concat_batch_tensors(samples_list: list[Any]) -> Any:
    if not samples_list:
        raise ValueError("samples_list must not be empty")

    try:
        import torch as torch_module
    except ModuleNotFoundError:
        torch_module = None

    if torch_module is not None and all(
        isinstance(samples, torch_module.Tensor) for samples in samples_list
    ):
        return torch_module.cat(samples_list, dim=0)

    stacked_samples: list[Any] = []
    for samples in samples_list:
        values = samples.tolist()
        if not isinstance(values, list):
            raise ValueError("encoded samples must be batch-first tensors")
        stacked_samples.extend(values)
    return _ListTensor(stacked_samples)


def vae_encode(vae: _VaeEncoder, image: Image.Image) -> dict[str, Any]:
    """Encode a PIL image into a ComfyUI LATENT dict."""
    pixel_samples = _image_to_tensor_like(image)
    samples = vae.encode(pixel_samples)
    return {"samples": samples}


def vae_encode_for_inpaint(
    vae: _VaeEncoderForInpaint,
    image: Image.Image,
    mask: Any,
    grow_mask_by: int = 6,
) -> dict[str, Any]:
    """Encode an inpaint latent and attach a noise mask."""
    import torch

    pixels = _image_to_tensor_like(image)
    if not isinstance(pixels, torch.Tensor):
        raise TypeError("vae_encode_for_inpaint requires torch to be installed")

    normalized_mask = _mask_to_tensor(mask)
    if not isinstance(normalized_mask, torch.Tensor):
        raise TypeError("mask must be a PIL image or torch.Tensor")

    downscale_ratio = vae.spacial_compression_encode()
    x = (pixels.shape[1] // downscale_ratio) * downscale_ratio
    y = (pixels.shape[2] // downscale_ratio) * downscale_ratio

    resized_mask = torch.nn.functional.interpolate(
        normalized_mask.reshape((-1, 1, normalized_mask.shape[-2], normalized_mask.shape[-1])),
        size=(pixels.shape[1], pixels.shape[2]),
        mode="bilinear",
    )

    masked_pixels = pixels.clone()
    if masked_pixels.shape[1] != x or masked_pixels.shape[2] != y:
        x_offset = (masked_pixels.shape[1] % downscale_ratio) // 2
        y_offset = (masked_pixels.shape[2] % downscale_ratio) // 2
        masked_pixels = masked_pixels[:, x_offset : x + x_offset, y_offset : y + y_offset, :]
        resized_mask = resized_mask[:, :, x_offset : x + x_offset, y_offset : y + y_offset]

    if grow_mask_by == 0:
        mask_erosion = resized_mask
    else:
        kernel_tensor = torch.ones(
            (1, 1, grow_mask_by, grow_mask_by),
            dtype=resized_mask.dtype,
            device=resized_mask.device,
        )
        mask_erosion = torch.clamp(
            torch.nn.functional.conv2d(
                resized_mask.round(),
                kernel_tensor,
                padding=grow_mask_by // 2,
            ),
            0,
            1,
        )

    inverse_mask = (1.0 - resized_mask.round()).squeeze(1)
    for channel in range(3):
        masked_pixels[:, :, :, channel] -= 0.5
        masked_pixels[:, :, :, channel] *= inverse_mask
        masked_pixels[:, :, :, channel] += 0.5

    samples = vae.encode(masked_pixels)
    return {"samples": samples, "noise_mask": mask_erosion[:, :, :x, :y].round()}


def vae_encode_batch(vae: _VaeEncoder, images: list[Image.Image]) -> dict[str, Any]:
    """Encode a list of PIL images into a ComfyUI LATENT dict."""
    if not images:
        raise ValueError("images must not be empty")

    pixel_samples = _images_to_tensor_like(images)
    samples = vae.encode(pixel_samples)
    return {"samples": samples}


def vae_encode_tiled(
    vae: _VaeEncoderTiled,
    image: Image.Image,
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, Any]:
    """Encode a PIL image into a ComfyUI LATENT dict using tiled encode.

    Args:
        vae: VAE model with an ``encode_tiled`` method.
        image: Input PIL image to encode.
        tile_size: Width and height of each encode tile in pixels. Defaults to 512.
        overlap: Number of pixels of overlap between adjacent tiles. Defaults to 64.

    Returns:
        ComfyUI LATENT dict containing a ``"samples"`` tensor.
    """
    pixel_samples = _image_to_tensor_like(image)
    samples = vae.encode_tiled(pixel_samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
    return {"samples": samples}


def vae_encode_batch_tiled(
    vae: _VaeEncoderTiled,
    images: list[Image.Image],
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, Any]:
    """Encode a list of PIL images into a ComfyUI LATENT dict using tiled encode."""
    if not images:
        raise ValueError("images must not be empty")

    samples_list: list[Any] = []
    for image in images:
        pixel_samples = _image_to_tensor_like(image)
        samples = vae.encode_tiled(
            pixel_samples,
            tile_x=tile_size,
            tile_y=tile_size,
            overlap=overlap,
        )
        samples_list.append(samples)

    return {"samples": _concat_batch_tensors(samples_list)}


__all__ = [
    "vae_decode",
    "vae_decode_tiled",
    "vae_decode_batch",
    "vae_decode_batch_tiled",
    "vae_encode",
    "vae_encode_for_inpaint",
    "vae_encode_batch",
    "vae_encode_tiled",
    "vae_encode_batch_tiled",
]
