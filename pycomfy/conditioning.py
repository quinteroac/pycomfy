"""Prompt conditioning helpers."""

from __future__ import annotations

from typing import Any, Protocol


class _ClipTextEncoder(Protocol):
    def tokenize(self, text: str) -> Any: ...

    def encode_from_tokens_scheduled(self, tokens: Any) -> Any: ...


def encode_prompt(clip: _ClipTextEncoder, text: str) -> Any:
    """Encode prompt text with a ComfyUI-compatible CLIP object."""
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


__all__ = ["encode_prompt"]
