"""Prompt conditioning helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol


class _ClipTextEncoder(Protocol):
    def tokenize(self, text: str) -> Any: ...

    def encode_from_tokens_scheduled(self, tokens: Any) -> Any: ...


class _FluxClipTextEncoder(Protocol):
    def tokenize(self, text: str) -> Any: ...

    def encode_from_tokens_scheduled(
        self,
        tokens: Any,
        add_dict: dict[str, Any] | None = None,
    ) -> Any: ...


class _ClipVisionEncoder(Protocol):
    def encode_image(self, image: Any, crop: bool = True) -> Any: ...


class _VaeEncoder(Protocol):
    latent_channels: int

    def spacial_compression_encode(self) -> int: ...

    def encode(self, image: Any) -> Any: ...


def _get_wan_conditioning_dependencies() -> tuple[Any, Any, Any, Any]:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management
    import comfy.utils
    import node_helpers
    import torch

    return torch, comfy.model_management, comfy.utils, node_helpers


def _get_clip_vision_output_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.clip_vision

    return comfy.clip_vision.Output


def _normalize_prompt_text(text: str) -> str:
    return " " if text == "" else text


def encode_prompt(clip: _ClipTextEncoder, text: str) -> Any:
    """Encode prompt text with a ComfyUI-compatible CLIP object.

    Positive and negative prompts use the same encoding path; prompt
    semantics are owned by the caller.
    """
    normalized_text = _normalize_prompt_text(text)
    tokens = clip.tokenize(normalized_text)
    return clip.encode_from_tokens_scheduled(tokens)


def encode_prompt_flux(
    clip: _FluxClipTextEncoder,
    text: str,
    clip_l_text: str,
    guidance: float = 3.5,
) -> Any:
    """Encode prompts with Flux dual-encoder token layout."""
    clip_l_tokens = clip.tokenize(_normalize_prompt_text(clip_l_text))
    clip_l_tokens["t5xxl"] = clip.tokenize(_normalize_prompt_text(text))["t5xxl"]
    return clip.encode_from_tokens_scheduled(clip_l_tokens, add_dict={"guidance": guidance})


def encode_clip_vision(
    clip_vision: _ClipVisionEncoder,
    image: Any,
    crop: Literal["center", "none"] = "center",
) -> Any:
    """Encode image features with a CLIP-vision model."""
    return clip_vision.encode_image(image, crop=(crop == "center"))


def wan_image_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    start_image: Any | None = None,
    clip_vision_output: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN image-to-video conditioning and empty latent samples."""
    torch, model_management, comfy_utils, node_helpers = _get_wan_conditioning_dependencies()

    latent = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=model_management.intermediate_device(),
    )

    if start_image is not None:
        start_image = comfy_utils.common_upscale(
            start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        image = torch.ones(
            (length, height, width, start_image.shape[-1]),
            device=start_image.device,
            dtype=start_image.dtype,
        ) * 0.5
        image[: start_image.shape[0]] = start_image

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = torch.ones(
            (1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
            device=start_image.device,
            dtype=start_image.dtype,
        )
        mask[:, :, : ((start_image.shape[0] - 1) // 4) + 1] = 0.0

        values = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)

    if clip_vision_output is not None:
        values = {"clip_vision_output": clip_vision_output}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)

    return positive, negative, {"samples": latent}


def wan_first_last_frame_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    start_image: Any | None = None,
    end_image: Any | None = None,
    clip_vision_start_image: Any | None = None,
    clip_vision_end_image: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN first/last-frame conditioning and empty latent samples."""
    torch, model_management, comfy_utils, node_helpers = _get_wan_conditioning_dependencies()

    spacial_scale = vae.spacial_compression_encode()
    latent = torch.zeros(
        [
            batch_size,
            vae.latent_channels,
            ((length - 1) // 4) + 1,
            height // spacial_scale,
            width // spacial_scale,
        ],
        device=model_management.intermediate_device(),
    )

    if start_image is not None:
        start_image = comfy_utils.common_upscale(
            start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
    if end_image is not None:
        end_image = comfy_utils.common_upscale(
            end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

    image = torch.ones((length, height, width, 3)) * 0.5
    mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

    if start_image is not None:
        image[: start_image.shape[0]] = start_image
        mask[:, :, : start_image.shape[0] + 3] = 0.0

    if end_image is not None:
        image[-end_image.shape[0] :] = end_image
        mask[:, :, -end_image.shape[0] :] = 0.0

    concat_latent_image = vae.encode(image[:, :, :, :3])
    mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
    values = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
    positive = node_helpers.conditioning_set_values(positive, values)
    negative = node_helpers.conditioning_set_values(negative, values)

    clip_vision_output = None
    if clip_vision_start_image is not None:
        clip_vision_output = clip_vision_start_image

    if clip_vision_end_image is not None:
        if clip_vision_output is not None:
            states = torch.cat(
                [
                    clip_vision_output.penultimate_hidden_states,
                    clip_vision_end_image.penultimate_hidden_states,
                ],
                dim=-2,
            )
            clip_vision_output = _get_clip_vision_output_type()()
            clip_vision_output.penultimate_hidden_states = states
        else:
            clip_vision_output = clip_vision_end_image

    if clip_vision_output is not None:
        values = {"clip_vision_output": clip_vision_output}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)

    return positive, negative, {"samples": latent}


def conditioning_combine(
    cond_a: Any,
    cond_b: Any | None = None,
    *additional: Any,
) -> list[Any]:
    """Combine two or more conditioning objects into one list.

    Supports both:
    - ``conditioning_combine(cond_a, cond_b, cond_c, ...)``
    - ``conditioning_combine([cond_a, cond_b, cond_c, ...])``
    """
    sources: list[Any]
    if cond_b is None and not additional:
        if not isinstance(cond_a, Sequence) or isinstance(cond_a, (str, bytes)):
            raise ValueError(
                "conditioning_combine() requires at least two conditioning objects"
            )

        sources = list(cond_a)
        if len(sources) < 2:
            raise ValueError(
                "conditioning_combine() requires at least two conditioning objects"
            )
    else:
        if cond_b is None:
            raise ValueError(
                "conditioning_combine() requires at least two conditioning objects"
            )
        sources = [cond_a, cond_b, *additional]

    merged: list[Any] = []
    for source in sources:
        merged.extend(source)

    return merged


def conditioning_set_mask(
    conditioning: Any,
    mask: Any,
    strength: float = 1.0,
    set_cond_area: Literal["default", "mask bounds"] = "default",
) -> list[Any]:
    """Attach mask metadata to each conditioning entry.

    Mirrors ComfyUI's ``ConditioningSetMask`` node behavior.
    """
    set_area_to_bounds = set_cond_area != "default"
    normalized_mask = mask if len(mask.shape) >= 3 else mask.unsqueeze(0)

    output: list[Any] = []
    for item in conditioning:
        updated = [item[0], item[1].copy()]
        updated[1]["mask"] = normalized_mask
        updated[1]["set_area_to_bounds"] = set_area_to_bounds
        updated[1]["mask_strength"] = strength
        output.append(updated)

    return output


def _validate_timestep_percent(name: str, value: float) -> None:
    if not isinstance(value, float):
        raise TypeError(f"{name} must be a float")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def conditioning_set_timestep_range(
    conditioning: Any,
    start: float,
    end: float,
) -> list[Any]:
    """Attach timestep bounds metadata to each conditioning entry."""
    _validate_timestep_percent("start", start)
    _validate_timestep_percent("end", end)

    output: list[Any] = []
    for item in conditioning:
        updated = [item[0], item[1].copy()]
        updated[1]["start_percent"] = start
        updated[1]["end_percent"] = end
        output.append(updated)

    return output


def flux_guidance(conditioning: Any, guidance: float = 3.5) -> list[Any]:
    """Apply Flux guidance value metadata to each conditioning entry."""
    output: list[Any] = []
    for item in conditioning:
        updated = [item[0], item[1].copy()]
        updated[1]["guidance"] = guidance
        output.append(updated)

    return output


__all__ = [
    "encode_prompt",
    "encode_prompt_flux",
    "encode_clip_vision",
    "wan_image_to_video",
    "wan_first_last_frame_to_video",
    "conditioning_combine",
    "conditioning_set_mask",
    "conditioning_set_timestep_range",
    "flux_guidance",
]
