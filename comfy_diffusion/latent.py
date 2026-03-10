"""Latent image helpers."""

from __future__ import annotations

from typing import Any, cast

_SUPPORTED_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")


def _get_empty_latent_image_type() -> Any:
    """Resolve ComfyUI EmptyLatentImage node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.EmptyLatentImage


def _get_latent_upscale_type() -> Any:
    """Resolve ComfyUI LatentUpscale node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.LatentUpscale


def _get_latent_upscale_by_type() -> Any:
    """Resolve ComfyUI LatentUpscaleBy node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.LatentUpscaleBy


def _get_latent_crop_type() -> Any:
    """Resolve ComfyUI LatentCrop node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.LatentCrop


def _get_latent_composite_type() -> Any:
    """Resolve ComfyUI LatentComposite node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.LatentComposite


def _get_latent_composite_masked_type() -> Any:
    """Resolve ComfyUI LatentCompositeMasked node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_mask import LatentCompositeMasked

    return LatentCompositeMasked


def _unwrap_node_output(output: Any) -> Any:
    """Return first output for ComfyUI V3 nodes and tuple-style APIs."""
    if hasattr(output, "result"):
        return output.result[0]
    if isinstance(output, tuple):
        return output[0]
    return output


def _validate_upscale_method(method: str) -> None:
    """Validate latent upscale methods against the ComfyUI set."""
    if method not in _SUPPORTED_UPSCALE_METHODS:
        raise ValueError(
            f"method must be one of {list(_SUPPORTED_UPSCALE_METHODS)!r}; got {method!r}"
        )


def empty_latent_image(width: int, height: int, batch_size: int = 1) -> dict[str, Any]:
    """Create empty image latents compatible with ComfyUI samplers."""
    empty_latent_image_type = _get_empty_latent_image_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            empty_latent_image_type().generate(width=width, height=height, batch_size=batch_size)
        ),
    )


def latent_upscale(
    latent: dict[str, Any], method: str, width: int, height: int
) -> dict[str, Any]:
    """Upscale a LATENT dict to target pixel dimensions."""
    _validate_upscale_method(method)
    latent_upscale_type = _get_latent_upscale_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_upscale_type().upscale(
                samples=latent,
                upscale_method=method,
                width=width,
                height=height,
                crop="disabled",
            )
        ),
    )


def latent_upscale_by(latent: dict[str, Any], method: str, scale_by: float) -> dict[str, Any]:
    """Upscale a LATENT dict by floating-point factor."""
    _validate_upscale_method(method)
    latent_upscale_by_type = _get_latent_upscale_by_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_upscale_by_type().upscale(
                samples=latent,
                upscale_method=method,
                scale_by=scale_by,
            )
        ),
    )


def latent_crop(
    latent: dict[str, Any], x: int, y: int, width: int, height: int
) -> dict[str, Any]:
    """Crop a LATENT dict using pixel-space coordinates and dimensions."""
    latent_crop_type = _get_latent_crop_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_crop_type().crop(samples=latent, width=width, height=height, x=x, y=y)
        ),
    )


def latent_composite(
    destination: dict[str, Any], source: dict[str, Any], x: int, y: int
) -> dict[str, Any]:
    """Composite source LATENT onto destination LATENT at pixel-space coordinates."""
    latent_composite_type = _get_latent_composite_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_composite_type().composite(
                samples_to=destination,
                samples_from=source,
                x=x,
                y=y,
                feather=0,
            )
        ),
    )


def latent_composite_masked(
    destination: dict[str, Any], source: dict[str, Any], mask: Any, x: int = 0, y: int = 0
) -> dict[str, Any]:
    """Composite source LATENT onto destination LATENT with mask-guided blending."""
    latent_composite_masked_type = _get_latent_composite_masked_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_composite_masked_type().composite(
                destination=destination,
                source=source,
                x=x,
                y=y,
                resize_source=False,
                mask=mask,
            )
        ),
    )


__all__ = [
    "empty_latent_image",
    "latent_upscale",
    "latent_upscale_by",
    "latent_crop",
    "latent_composite",
    "latent_composite_masked",
]
