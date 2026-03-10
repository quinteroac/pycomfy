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


__all__ = ["load_controlnet"]
