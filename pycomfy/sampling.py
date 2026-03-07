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


def _get_random_noise_type() -> Any:
    """Resolve ComfyUI RandomNoise implementation at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_custom_sampler import RandomNoise

    return RandomNoise


def _get_disable_noise_type() -> Any:
    """Resolve ComfyUI DisableNoise implementation at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_custom_sampler import DisableNoise

    return DisableNoise


def _get_basic_scheduler_type() -> Any:
    """Resolve ComfyUI BasicScheduler implementation at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_custom_sampler import BasicScheduler

    return BasicScheduler


def _get_karras_scheduler_type() -> Any:
    """Resolve ComfyUI KarrasScheduler implementation at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_custom_sampler import KarrasScheduler

    return KarrasScheduler


def _get_ays_scheduler_type() -> Any:
    """Resolve ComfyUI AlignYourStepsScheduler implementation at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler

    return AlignYourStepsScheduler


def _get_flux2_scheduler_type() -> Any:
    """Resolve ComfyUI Flux2Scheduler implementation at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_flux import Flux2Scheduler

    return Flux2Scheduler


def _get_ltxv_scheduler_type() -> Any:
    """Resolve ComfyUI LTXVScheduler implementation at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_lt import LTXVScheduler

    return LTXVScheduler


def _get_sampler_custom_advanced_type() -> Any:
    """Resolve ComfyUI SamplerCustomAdvanced implementation at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

    return SamplerCustomAdvanced


def _unwrap_node_output(output: Any) -> Any:
    """Return the first node output value from ComfyUI V3 or tuple-style APIs."""
    result = getattr(output, "result", output)
    return result[0]


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


def random_noise(noise_seed: int) -> Any:
    """Create a RandomNoise object compatible with ``sample_custom()``."""
    random_noise_type = _get_random_noise_type()
    return random_noise_type(noise_seed)


def disable_noise() -> Any:
    """Create a DisableNoise object compatible with ``sample_custom()``."""
    disable_noise_type = _get_disable_noise_type()
    return disable_noise_type()


def basic_scheduler(
    model: Any, scheduler_name: str, steps: int, denoise: float = 1.0
) -> Any:
    """Create SIGMAS using ComfyUI BasicScheduler."""
    basic_scheduler_type = _get_basic_scheduler_type()
    return _unwrap_node_output(
        basic_scheduler_type.execute(model, scheduler_name, steps, denoise)
    )


def karras_scheduler(
    steps: int, sigma_max: float, sigma_min: float, rho: float = 7.0
) -> Any:
    """Create SIGMAS using ComfyUI KarrasScheduler."""
    karras_scheduler_type = _get_karras_scheduler_type()
    return _unwrap_node_output(
        karras_scheduler_type.execute(steps, sigma_max, sigma_min, rho)
    )


def ays_scheduler(model_type: str, steps: int, denoise: float = 1.0) -> Any:
    """Create SIGMAS using ComfyUI AlignYourStepsScheduler."""
    allowed_model_types = {"SD1", "SDXL", "SVD"}
    if model_type not in allowed_model_types:
        raise ValueError(
            f"model_type must be one of {sorted(allowed_model_types)!r}; got {model_type!r}"
        )

    ays_scheduler_type = _get_ays_scheduler_type()
    return _unwrap_node_output(ays_scheduler_type.execute(model_type, steps, denoise))


def flux2_scheduler(steps: int, width: int, height: int) -> Any:
    """Create SIGMAS using ComfyUI Flux2Scheduler."""
    flux2_scheduler_type = _get_flux2_scheduler_type()
    return _unwrap_node_output(flux2_scheduler_type.execute(steps, width, height))


def ltxv_scheduler(
    steps: int,
    max_shift: float,
    base_shift: float,
    *,
    stretch: bool = True,
    terminal: float = 0.1,
    latent: Any = None,
) -> Any:
    """Create SIGMAS using ComfyUI LTXVScheduler."""
    ltxv_scheduler_type = _get_ltxv_scheduler_type()
    return _unwrap_node_output(
        ltxv_scheduler_type.execute(
            steps,
            max_shift,
            base_shift,
            stretch,
            terminal,
            latent,
        )
    )


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


def sample_custom(
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: Any,
    latent_image: Any,
) -> tuple[Any, Any]:
    """Run custom sampling with explicit noise/guider/sampler/sigmas inputs.

    The `model` is not a direct argument here: it is already embedded in the
    provided `guider` object (for example from `basic_guider()` or `cfg_guider()`).
    """
    sampler_custom_advanced_type = _get_sampler_custom_advanced_type()
    output = sampler_custom_advanced_type.execute(
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
    )
    result = getattr(output, "result", output)
    return result[0], result[1]


__all__ = [
    "sample",
    "sample_advanced",
    "sample_custom",
    "basic_guider",
    "cfg_guider",
    "random_noise",
    "disable_noise",
    "basic_scheduler",
    "karras_scheduler",
    "ays_scheduler",
    "flux2_scheduler",
    "ltxv_scheduler",
]
