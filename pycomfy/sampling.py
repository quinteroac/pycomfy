"""Sampling helpers that wrap ComfyUI's KSampler behavior."""

from __future__ import annotations

from typing import Any


def _get_common_ksampler() -> Any:
    """Resolve ComfyUI sampling entrypoint at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.common_ksampler


def _get_basic_guider_type() -> Any:
    """Resolve ComfyUI BasicGuider implementation at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_custom_sampler import Guider_Basic

    return Guider_Basic


def _get_cfg_guider_type() -> Any:
    """Resolve ComfyUI CFGGuider implementation at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy.samplers import CFGGuider

    return CFGGuider


def basic_guider(model: Any, conditioning: Any) -> Any:
    """Create a BasicGuider compatible with ``sample_custom()``."""
    guider_type = _get_basic_guider_type()
    guider = guider_type(model)
    guider.set_conds(conditioning)
    return guider


def cfg_guider(model: Any, positive: Any, negative: Any, cfg: Any) -> Any:
    """Create a CFGGuider compatible with ``sample_custom()``."""
    guider_type = _get_cfg_guider_type()
    guider = guider_type(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    return guider


def sample(
    model: Any,
    positive: Any,
    negative: Any,
    latent: Any,
    steps: Any,
    cfg: Any,
    sampler_name: str,
    scheduler: str,
    seed: int,
    *,
    denoise: float = 1.0,
) -> Any:
    """Run denoising through ComfyUI and return the denoised LATENT object.

    The `latent` input follows ComfyUI's `common_ksampler` contract: a LATENT dict
    containing `"samples"` and optional metadata keys.
    """
    common_ksampler = _get_common_ksampler()

    return common_ksampler(
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


def sample_advanced(
    model: Any,
    positive: Any,
    negative: Any,
    latent: Any,
    steps: Any,
    cfg: Any,
    sampler_name: str,
    scheduler: str,
    noise_seed: int,
    *,
    add_noise: bool = True,
    return_with_leftover_noise: bool = False,
    denoise: float = 1.0,
    start_at_step: int = 0,
    end_at_step: int = 10000,
) -> Any:
    """Run advanced denoising with explicit noise and final-step control.

    Mirrors ComfyUI `KSamplerAdvanced` semantics by mapping:
    - `add_noise=False` -> `disable_noise=True`
    - `return_with_leftover_noise=True` -> `force_full_denoise=False`
    """
    common_ksampler = _get_common_ksampler()

    return common_ksampler(
        model,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=denoise,
        disable_noise=not add_noise,
        start_step=start_at_step,
        last_step=end_at_step,
        force_full_denoise=not return_with_leftover_noise,
    )[0]


__all__ = ["sample", "sample_advanced", "basic_guider", "cfg_guider"]
