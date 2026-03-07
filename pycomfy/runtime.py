"""Runtime diagnostics for pycomfy."""

from __future__ import annotations

import importlib
import sys
from typing import Any


def _python_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def _runtime_not_found(python_version: str) -> dict[str, Any]:
    return {
        "error": "ComfyUI runtime not found. Run: git submodule update --init",
        "comfyui_version": None,
        "device": None,
        "vram_total_mb": None,
        "vram_free_mb": None,
        "python_version": python_version,
    }


def _runtime_not_responsive(python_version: str, message: str) -> dict[str, Any]:
    return {
        "error": f"ComfyUI runtime is not responsive: {message}",
        "comfyui_version": None,
        "device": None,
        "vram_total_mb": None,
        "vram_free_mb": None,
        "python_version": python_version,
    }


def _bytes_to_mb(value: int) -> int:
    return value // (1024 * 1024)


def check_runtime() -> dict[str, Any]:
    """Return structured runtime diagnostics for the current Python process."""
    python_version = _python_version()

    try:
        comfyui_version_module = importlib.import_module("comfyui_version")
        model_management = importlib.import_module("comfy.model_management")
    except Exception:
        return _runtime_not_found(python_version)

    try:
        device = model_management.get_torch_device()
    except Exception as exc:
        return _runtime_not_responsive(python_version, str(exc))

    comfyui_version = str(getattr(comfyui_version_module, "__version__", "unknown"))
    device_str = str(device)
    device_type = getattr(device, "type", "")

    if device_type == "cpu" or device_str == "cpu":
        return {
            "comfyui_version": comfyui_version,
            "device": "cpu",
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "python_version": python_version,
        }

    try:
        total_memory_bytes = model_management.get_total_memory(device)
        free_memory_bytes = model_management.get_free_memory(device)
    except Exception as exc:
        return _runtime_not_responsive(python_version, str(exc))

    return {
        "comfyui_version": comfyui_version,
        "device": device_str,
        "vram_total_mb": _bytes_to_mb(int(total_memory_bytes)),
        "vram_free_mb": _bytes_to_mb(int(free_memory_bytes)),
        "python_version": python_version,
    }
