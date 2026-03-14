"""Text generation helpers that wrap ComfyUI-compatible CLIP textgen objects."""

from __future__ import annotations

from typing import Any, cast


def generate_text(
    clip: Any,
    prompt: str,
    *,
    image: Any | None = None,
    max_length: int = 256,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_k: int = 64,
    top_p: float = 0.95,
    min_p: float = 0.05,
    repetition_penalty: float = 1.05,
    seed: int = 0,
) -> str:
    """Generate text with a ComfyUI-compatible text encoder."""
    tokens = clip.tokenize(prompt, image=image, skip_template=False, min_length=1)

    if do_sample:
        generated_ids = clip.generate(
            tokens,
            do_sample=True,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
    else:
        generated_ids = clip.generate(tokens, do_sample=False, max_length=max_length)

    return cast(str, clip.decode(generated_ids, skip_special_tokens=True))


__all__ = ["generate_text"]
