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


class _LtxvVaeEncoder(Protocol):
    def encode(self, image: Any) -> Any: ...


def _get_ltxv_conditioning_dependencies() -> tuple[Any, Any, Any, Any]:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management
    import comfy.utils
    import node_helpers
    import torch

    return torch, comfy.model_management, comfy.utils, node_helpers


def _get_wan_conditioning_dependencies() -> tuple[Any, Any, Any, Any]:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management
    import comfy.utils
    import node_helpers
    import torch

    return torch, comfy.model_management, comfy.utils, node_helpers


def _get_wan_vace_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.latent_formats
    import comfy.model_management
    import comfy.utils
    import node_helpers
    import torch

    return torch, comfy.model_management, comfy.utils, node_helpers, comfy.latent_formats


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


def ltxv_img_to_video(
    positive: Any,
    negative: Any,
    image: Any,
    vae: _LtxvVaeEncoder,
    width: int = 768,
    height: int = 512,
    length: int = 97,
    batch_size: int = 1,
    strength: float = 1.0,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build LTXV image-to-video conditioning and latent with frame noise mask."""
    torch, model_management, comfy_utils, _ = _get_ltxv_conditioning_dependencies()

    pixels = comfy_utils.common_upscale(
        image.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    latent_conditioning = vae.encode(pixels[:, :, :, :3])

    latent = torch.zeros(
        [batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32],
        device=model_management.intermediate_device(),
    )
    latent[:, :, : latent_conditioning.shape[2]] = latent_conditioning

    conditioning_latent_frames_mask = torch.ones(
        (batch_size, 1, latent.shape[2], 1, 1),
        dtype=torch.float32,
        device=latent.device,
    )
    conditioning_latent_frames_mask[:, :, : latent_conditioning.shape[2]] = 1.0 - strength

    latent_with_noise_mask = {
        "samples": latent,
        "noise_mask": conditioning_latent_frames_mask,
    }
    return positive, negative, latent_with_noise_mask


def ltxv_conditioning(
    positive: Any,
    negative: Any,
    frame_rate: float = 25.0,
) -> tuple[Any, Any]:
    """Inject frame rate metadata into LTXV positive and negative conditioning."""
    _, _, _, node_helpers = _get_ltxv_conditioning_dependencies()
    values = {"frame_rate": frame_rate}
    return (
        node_helpers.conditioning_set_values(positive, values),
        node_helpers.conditioning_set_values(negative, values),
    )


def ltxv_crop_guides(
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Crop keyframe guide frames from LTXV conditioning and latent before the main sampling pass.

    Removes ``keyframe_idxs`` and ``guide_attention_entries`` from both conditioning
    objects and slices the corresponding frames off the end of the latent tensor.
    When there are no keyframe guides (``num_keyframes == 0``) the inputs are
    returned unchanged.
    """
    # Extract keyframe_idxs without importing torch so the function is import-safe.
    keyframe_idxs = None
    for entry in positive:
        if "keyframe_idxs" in entry[1]:
            keyframe_idxs = entry[1]["keyframe_idxs"]
            break

    if keyframe_idxs is None:
        return positive, negative, latent

    torch, _, _, node_helpers = _get_ltxv_conditioning_dependencies()

    num_keyframes: int = torch.unique(keyframe_idxs[:, 0, :, 0]).shape[0]

    if num_keyframes == 0:
        return positive, negative, latent

    latent_image = latent["samples"].clone()
    noise_mask = latent.get("noise_mask", None)
    if noise_mask is None:
        batch_size, _, latent_length, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
    else:
        noise_mask = noise_mask.clone()

    latent_image = latent_image[:, :, :-num_keyframes]
    noise_mask = noise_mask[:, :, :-num_keyframes]

    guide_values: dict[str, Any] = {"keyframe_idxs": None, "guide_attention_entries": None}
    positive = node_helpers.conditioning_set_values(positive, guide_values)
    negative = node_helpers.conditioning_set_values(negative, guide_values)

    return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}


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


def wan_vace_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    strength: float = 1.0,
    control_video: Any | None = None,
    control_masks: Any | None = None,
    reference_image: Any | None = None,
) -> tuple[Any, Any, dict[str, Any], int]:
    """Build WAN VACE conditioning and empty latent samples.

    Args:
        positive: Positive conditioning.
        negative: Negative conditioning.
        vae: VAE model used to encode control frames and reference image.
        width: Video width in pixels (multiple of 8).
        height: Video height in pixels (multiple of 8).
        length: Number of video frames.
        batch_size: Latent batch size.
        strength: VACE conditioning strength.
        control_video: Optional control video tensor ``[T, H, W, C]`` in [0, 1].
        control_masks: Optional mask tensor ``[T, H, W]`` or ``[T, 1, H, W]`` in [0, 1].
        reference_image: Optional reference image tensor ``[1, H, W, C]`` in [0, 1].

    Returns:
        ``(positive, negative, latent, trim_latent)`` where ``trim_latent`` is
        the number of latent frames prepended for the reference image (0 if none).
    """
    torch, model_management, comfy_utils, node_helpers, comfy_latent_formats = _get_wan_vace_dependencies()

    latent_length = ((length - 1) // 4) + 1

    if control_video is not None:
        control_video = comfy_utils.common_upscale(
            control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        if control_video.shape[0] < length:
            control_video = torch.nn.functional.pad(
                control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
            )
    else:
        control_video = torch.ones((length, height, width, 3)) * 0.5

    if reference_image is not None:
        reference_image = comfy_utils.common_upscale(
            reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        ref_latent = vae.encode(reference_image[:, :, :, :3])
        reference_image = torch.cat(
            [ref_latent, comfy_latent_formats.Wan21().process_out(torch.zeros_like(ref_latent))], dim=1
        )

    if control_masks is None:
        mask = torch.ones((length, height, width, 1))
    else:
        mask = control_masks
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = comfy_utils.common_upscale(
            mask[:length], width, height, "bilinear", "center"
        ).movedim(1, -1)
        if mask.shape[0] < length:
            mask = torch.nn.functional.pad(
                mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0
            )

    control_video = control_video - 0.5
    inactive = (control_video * (1 - mask)) + 0.5
    reactive = (control_video * mask) + 0.5

    inactive = vae.encode(inactive[:, :, :, :3])
    reactive = vae.encode(reactive[:, :, :, :3])
    control_video_latent = torch.cat((inactive, reactive), dim=1)
    if reference_image is not None:
        control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

    vae_stride = 8
    height_mask = height // vae_stride
    width_mask = width // vae_stride
    mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
    mask = mask.permute(2, 4, 0, 1, 3)
    mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode="nearest-exact"
    ).squeeze(0)

    trim_latent = 0
    if reference_image is not None:
        mask_pad = torch.zeros_like(mask[:, : reference_image.shape[2], :, :])
        mask = torch.cat((mask_pad, mask), dim=1)
        latent_length += reference_image.shape[2]
        trim_latent = reference_image.shape[2]

    mask = mask.unsqueeze(0)

    positive = node_helpers.conditioning_set_values(
        positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True
    )
    negative = node_helpers.conditioning_set_values(
        negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True
    )

    latent = torch.zeros(
        [batch_size, 16, latent_length, height // 8, width // 8],
        device=model_management.intermediate_device(),
    )
    return positive, negative, {"samples": latent}, trim_latent


__all__ = [
    "encode_prompt",
    "encode_prompt_flux",
    "encode_clip_vision",
    "wan_image_to_video",
    "wan_first_last_frame_to_video",
    "ltxv_img_to_video",
    "ltxv_conditioning",
    "ltxv_crop_guides",
    "conditioning_combine",
    "conditioning_set_mask",
    "conditioning_set_timestep_range",
    "flux_guidance",
]
