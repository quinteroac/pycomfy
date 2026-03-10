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


__all__ = ["load_controlnet", "load_diff_controlnet", "apply_controlnet"]
