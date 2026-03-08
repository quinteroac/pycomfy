"""Internal runtime bootstrap for comfy_diffusion.

Path insertion is intentionally lightweight and import-safe: this module must not
import torch or comfy internals just to make ComfyUI discoverable.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _comfyui_root() -> Path:
    """Return the absolute path to the vendored ComfyUI directory."""
    return Path(__file__).resolve().parents[1] / "vendor" / "ComfyUI"


def ensure_comfyui_on_path() -> Path:
    """Ensure vendored ComfyUI is importable and return the inserted path."""
    comfyui_root = _comfyui_root()
    comfyui_root_str = str(comfyui_root)

    if comfyui_root_str not in sys.path:
        sys.path.insert(0, comfyui_root_str)

    return comfyui_root
