"""Latent image helpers."""

from __future__ import annotations

from typing import Any, cast


def _get_empty_latent_image_type() -> Any:
    """Resolve ComfyUI EmptyLatentImage node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.EmptyLatentImage


def _unwrap_node_output(output: Any) -> Any:
    """Return first output for ComfyUI V3 nodes and tuple-style APIs."""
    if hasattr(output, "result"):
        return output.result[0]
    if isinstance(output, tuple):
        return output[0]
    return output


def empty_latent_image(width: int, height: int, batch_size: int = 1) -> dict[str, Any]:
    """Create empty image latents compatible with ComfyUI samplers."""
    empty_latent_image_type = _get_empty_latent_image_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            empty_latent_image_type().generate(width=width, height=height, batch_size=batch_size)
        ),
    )


__all__ = ["empty_latent_image"]
