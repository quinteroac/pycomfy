"""Sampling helpers that wrap ComfyUI's KSampler behavior."""

from __future__ import annotations

from typing import Any

from ._runtime import ensure_comfyui_on_path


def sample(
    model: Any,
    positive: Any,
    negative: Any,
    latent: Any,
    steps: Any,
    cfg: Any,
    sampler_name: str,
    scheduler: str,
    seed: Any,
    *,
    denoise: float = 1.0,
) -> Any:
    """Run denoising through ComfyUI and return the denoised LATENT object.

    The `latent` input follows ComfyUI's `common_ksampler` contract: a LATENT dict
    containing `"samples"` and optional metadata keys.
    """
    ensure_comfyui_on_path()
    import nodes

    return nodes.common_ksampler(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=denoise,
    )[0]


__all__ = ["sample"]
