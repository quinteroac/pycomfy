"""Internal runtime bootstrap for pycomfy."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _comfyui_root() -> Path:
    """Return the absolute path to the vendored ComfyUI directory."""
    return Path(__file__).resolve().parents[1] / "vendor" / "ComfyUI"


def _ensure_cpu_mode_if_no_cuda() -> None:
    """If torch is CPU-only, add --cpu so ComfyUI uses CPU (it reads cli_args at import time)."""
    if "--cpu" in sys.argv:
        return
    try:
        import torch
        if not torch.cuda.is_available():
            sys.argv.append("--cpu")
    except ImportError:
        pass


def ensure_comfyui_on_path() -> Path:
    """Ensure vendored ComfyUI is importable and return the inserted path."""
    comfyui_root = _comfyui_root()
    comfyui_root_str = str(comfyui_root)

    if comfyui_root_str not in sys.path:
        sys.path.insert(0, comfyui_root_str)

    _ensure_cpu_mode_if_no_cuda()

    # When we forced --cpu (torch has no CUDA), ComfyUI must see it. ComfyUI parses
    # from [] when used as a library, so we enable parsing and force a single parse
    # with a minimal argv to avoid pytest/other args breaking the parser; then restore.
    if "--cpu" in sys.argv:
        try:
            options = importlib.import_module("comfy.options")
            options.enable_args_parsing(True)
            saved_argv = sys.argv
            sys.argv = [saved_argv[0] if saved_argv else "pycomfy", "--cpu"]
            importlib.import_module("comfy.cli_args")  # trigger parse with minimal argv
            sys.argv = saved_argv
        except (ImportError, AttributeError):
            pass

    return comfyui_root
