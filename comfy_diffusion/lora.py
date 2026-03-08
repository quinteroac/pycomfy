"""LoRA application helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


def apply_lora(
    model: Any,
    clip: Any,
    path: str | Path,
    strength_model: float,
    strength_clip: float,
) -> tuple[Any, Any]:
    """Apply a LoRA file to a model/CLIP pair and return patched copies.

    The returned pair can be passed back into ``apply_lora`` to stack
    multiple LoRAs by chaining calls.
    """
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()

    import comfy.sd
    import comfy.utils

    lora_path = str(Path(path))
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    patched = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
    return cast(tuple[Any, Any], patched)


__all__ = ["apply_lora"]
