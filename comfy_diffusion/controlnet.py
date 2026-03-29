"""ControlNet helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_controlnet(path: str | Path) -> Any:
    """Load a ControlNet checkpoint from a local file path."""
    from ._runtime import ensure_comfyui_on_path

    controlnet_path = Path(path).resolve()
    if not controlnet_path.is_file():
        raise FileNotFoundError(f"controlnet file not found: {controlnet_path}")

    ensure_comfyui_on_path()

    import comfy.controlnet as comfy_controlnet

    controlnet = comfy_controlnet.load_controlnet(str(controlnet_path))
    if controlnet is None:
        raise RuntimeError(
            "controlnet file is invalid and does not contain a valid controlnet model"
        )
    return controlnet


def load_diff_controlnet(model: Any, path: str | Path) -> Any:
    """Load a diff ControlNet checkpoint paired with a specific base model."""
    from ._runtime import ensure_comfyui_on_path

    controlnet_path = Path(path).resolve()
    if not controlnet_path.is_file():
        raise FileNotFoundError(f"controlnet file not found: {controlnet_path}")

    ensure_comfyui_on_path()

    import comfy.controlnet as comfy_controlnet

    controlnet = comfy_controlnet.load_controlnet(str(controlnet_path), model=model)
    if controlnet is None:
        raise RuntimeError(
            "controlnet file is invalid and does not contain a valid controlnet model"
        )
    return controlnet


def apply_controlnet(
    positive: Any,
    negative: Any,
    control_net: Any,
    image: Any,
    strength: float = 1.0,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    vae: Any = None,
) -> tuple[Any, Any]:
    """Apply ControlNet to positive and negative conditioning.

    ``image`` should be a torch Tensor control hint map.
    Mirrors ComfyUI's ``ControlNetApplyAdvanced`` behavior.
    """
    if strength == 0:
        return positive, negative

    control_hint = image.movedim(-1, 1)
    cached_controlnets: dict[Any, Any] = {}
    outputs: list[Any] = []

    for conditioning in (positive, negative):
        updated_conditioning: list[Any] = []
        for token, metadata in conditioning:
            updated_metadata = metadata.copy()
            previous_controlnet = updated_metadata.get("control")

            if previous_controlnet in cached_controlnets:
                controlnet_instance = cached_controlnets[previous_controlnet]
            else:
                controlnet_instance = control_net.copy().set_cond_hint(
                    control_hint,
                    strength,
                    (start_percent, end_percent),
                    vae=vae,
                    extra_concat=[],
                )
                controlnet_instance.set_previous_controlnet(previous_controlnet)
                cached_controlnets[previous_controlnet] = controlnet_instance

            updated_metadata["control"] = controlnet_instance
            updated_metadata["control_apply_to_uncond"] = False
            updated_conditioning.append([token, updated_metadata])

        outputs.append(updated_conditioning)

    return outputs[0], outputs[1]


def set_union_controlnet_type(control_net: Any, type: str) -> Any:
    """Configure a union ControlNet with the requested control type."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()

    from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

    control_net = control_net.copy()
    if type == "auto":
        control_net.set_extra_arg("control_type", [])
        return control_net

    type_number = UNION_CONTROLNET_TYPES.get(type)
    if type_number is None:
        supported_types = ["auto", *UNION_CONTROLNET_TYPES]
        supported_values = ", ".join(repr(item) for item in supported_types)
        raise ValueError(
            f"unsupported union controlnet type {type!r}; supported types: {supported_values}"
        )

    control_net.set_extra_arg("control_type", [type_number])
    return control_net


def _get_lotus_conditioning_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_lotus import LotusConditioning

    return LotusConditioning


def _get_ltxv_add_guide_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_lt import LTXVAddGuide

    return LTXVAddGuide


def _unwrap_node_output(output: Any) -> Any:
    result = getattr(output, "result", output)
    return result[0]


def lotus_conditioning(model: Any, image: Any) -> Any:
    """Apply Lotus depth-model conditioning for depth-to-video generation.

    Wraps ComfyUI's ``LotusConditioning`` node, which uses a frozen CLIP encoder
    and inlined null-conditioning tensors to produce the required conditioning
    format for Lotus-based depth pipelines (e.g. LTX depth-to-video).

    Args:
        model: The diffusion model (accepted for API consistency; Lotus conditioning
            is model-architecture-agnostic and uses hardcoded frozen encoder values).
        image: The input image tensor (accepted for API consistency; conditioning
            values are pre-computed from the frozen Lotus encoder).

    Returns:
        Conditioning tensor compatible with standard ComfyUI conditioning inputs.
    """
    lotus_conditioning_type = _get_lotus_conditioning_type()
    return _unwrap_node_output(lotus_conditioning_type.execute())


def ltxv_add_guide(
    conditioning: Any,
    image: Any,
    mask: Any,
    strength: float,
    start_percent: float,
    end_percent: float,
) -> Any:
    """Add guide-frame conditioning for spatially controlled LTXV video generation.

    Wraps ComfyUI's ``LTXVAddGuide`` node. Injects a guide image into the
    conditioning tensor so the sampler respects the spatial reference frames
    (e.g. canny, depth, or pose control images).

    Args:
        conditioning: Positive conditioning tensor to attach guide frames to.
        image: Guide image tensor (B, H, W, C).
        mask: Guide mask tensor, or ``None`` for full-frame guidance.
        strength: Guide strength in [0.0, 1.0].
        start_percent: Temporal start fraction in [0.0, 1.0].
        end_percent: Temporal end fraction in [0.0, 1.0].

    Returns:
        Updated conditioning tensor with guide frame metadata injected.
    """
    ltxv_add_guide_type = _get_ltxv_add_guide_type()
    return _unwrap_node_output(
        ltxv_add_guide_type.execute(
            conditioning, image, mask, strength, start_percent, end_percent
        )
    )


def lotus_depth_pass(lotus_model: Any, lotus_vae: Any, image: Any) -> Any:
    """Run the full Lotus depth-estimation pass on an image batch.

    Wraps the "Image to Depth Map (Lotus)" subgraph from the LTX-2 depth
    workflow:  VAEEncode → LotusConditioning + DisableNoise + BasicGuider +
    BasicScheduler(1 step, normal) + SetFirstSigma(999) →
    SamplerCustomAdvanced(euler) → VAEDecode → ImageInvert.

    Parameters
    ----------
    lotus_model :
        Lotus depth diffusion model loaded via ``ModelManager.load_unet()``.
    lotus_vae :
        Standard SD VAE (``vae-ft-mse-840000-ema-pruned``) loaded via
        ``ModelManager.load_vae()``.
    image :
        ComfyUI IMAGE tensor (B, H, W, C) of input frames at half resolution.

    Returns
    -------
    Any
        ComfyUI IMAGE tensor of inverted depth maps, same resolution as input.
    """
    from comfy_diffusion.image import image_invert
    from comfy_diffusion.sampling import (
        basic_guider,
        basic_scheduler,
        disable_noise,
        get_sampler,
        sample_custom,
        set_first_sigma,
    )

    # Encode input image using the standard SD VAE (Lotus uses 4-channel latents).
    latent: dict[str, Any] = {"samples": lotus_vae.encode(image[:, :, :, :3])}

    # Lotus conditioning uses a frozen encoder with hardcoded null tensors.
    cond = lotus_conditioning(lotus_model, image)

    # BasicScheduler (1 step, normal schedule) + override first sigma to 999.
    sigmas = basic_scheduler(lotus_model, "normal", 1, 1.0)
    sigmas = set_first_sigma(sigmas, 999.0)

    # BasicGuider (unconditioned, matches the reference workflow).
    guider = basic_guider(lotus_model, cond)

    # DisableNoise: tells the sampler to skip noise injection.
    noise = disable_noise()

    # Run one denoising step with the euler sampler.
    sampler = get_sampler("euler")
    output, _ = sample_custom(noise, guider, sampler, sigmas, latent)

    # Decode latent → image tensor using the same lotus VAE.
    depth_tensor = lotus_vae.decode(output["samples"])

    # Invert the depth map to match LTX-2 depth-control LoRA convention.
    return image_invert(depth_tensor)


__all__ = [
    "load_controlnet",
    "load_diff_controlnet",
    "apply_controlnet",
    "set_union_controlnet_type",
    "lotus_conditioning",
    "ltxv_add_guide",
    "lotus_depth_pass",
]
