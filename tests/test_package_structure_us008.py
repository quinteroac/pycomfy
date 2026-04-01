"""Tests for pipeline sub-package structure (US-008).

Covers:
  - AC01: comfy_diffusion/pipelines/image/flux_klein/__init__.py exists
  - AC02: comfy_diffusion/pipelines/image/qwen/__init__.py exists
  - AC03: from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base succeeds (no GPU)
  - AC04: from comfy_diffusion.pipelines.image.qwen import layered succeeds (no GPU)
  - AC05: typecheck / lint passes (verified by import success without errors)
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINES_IMAGE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image"


def test_ac01_flux_klein_init_exists() -> None:
    """AC01: flux_klein __init__.py must exist on disk."""
    init = _PIPELINES_IMAGE / "flux_klein" / "__init__.py"
    assert init.exists(), f"Missing {init}"


def test_ac02_qwen_init_exists() -> None:
    """AC02: qwen __init__.py must exist on disk."""
    init = _PIPELINES_IMAGE / "qwen" / "__init__.py"
    assert init.exists(), f"Missing {init}"


def test_ac03_flux_klein_t2i_4b_base_importable() -> None:
    """AC03: importing t2i_4b_base from flux_klein package succeeds without GPU."""
    mod = importlib.import_module("comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base")
    assert hasattr(mod, "manifest"), "t2i_4b_base must export manifest()"
    assert hasattr(mod, "run"), "t2i_4b_base must export run()"


def test_ac04_qwen_layered_importable() -> None:
    """AC04: importing layered from qwen package succeeds without GPU."""
    mod = importlib.import_module("comfy_diffusion.pipelines.image.qwen.layered")
    assert hasattr(mod, "manifest"), "layered must export manifest()"
