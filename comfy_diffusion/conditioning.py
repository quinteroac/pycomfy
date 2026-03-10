"""Prompt conditioning helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class _ClipTextEncoder(Protocol):
    def tokenize(self, text: str) -> Any: ...

    def encode_from_tokens_scheduled(self, tokens: Any) -> Any: ...


def encode_prompt(clip: _ClipTextEncoder, text: str) -> Any:
    """Encode prompt text with a ComfyUI-compatible CLIP object.

    Positive and negative prompts use the same encoding path; prompt
    semantics are owned by the caller.
    """
    normalized_text = " " if text == "" else text
    tokens = clip.tokenize(normalized_text)
    return clip.encode_from_tokens_scheduled(tokens)


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


__all__ = ["encode_prompt", "conditioning_combine"]
