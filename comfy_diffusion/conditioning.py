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
    "conditioning_combine",
    "conditioning_set_mask",
    "conditioning_set_timestep_range",
    "flux_guidance",
]
