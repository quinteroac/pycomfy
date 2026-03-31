"""Tests for US-005 — Pipeline sub-package wiring.

Verifies that the `wan` sub-package is properly wired into the
`pipelines.video` namespace, mirroring the LTX family structure.

Covers:
  - AC01: comfy_diffusion/pipelines/video/wan/__init__.py exports "wan21"
  - AC02: comfy_diffusion/pipelines/video/wan/wan21/__init__.py exports
          "t2v", "i2v", and "flf2v"
  - AC03: comfy_diffusion/pipelines/video/__init__.py __all__ includes "wan"
  - AC04: All sub-package __init__ files parse without syntax errors and
          the namespace is importable
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINES_VIDEO = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video"


# ---------------------------------------------------------------------------
# AC01 — wan/__init__.py exports "wan21"
# ---------------------------------------------------------------------------


def test_wan_init_exists() -> None:
    assert (_PIPELINES_VIDEO / "wan" / "__init__.py").is_file()


def test_wan_init_parses() -> None:
    source = (_PIPELINES_VIDEO / "wan" / "__init__.py").read_text(encoding="utf-8")
    ast.parse(source)


def test_wan_init_exports_wan21() -> None:
    import comfy_diffusion.pipelines.video.wan as wan_pkg

    assert "wan21" in wan_pkg.__all__, (
        f"wan/__init__.py __all__ must contain 'wan21', got {wan_pkg.__all__!r}"
    )


def test_wan_init_all_is_list() -> None:
    import comfy_diffusion.pipelines.video.wan as wan_pkg

    assert isinstance(wan_pkg.__all__, list)


# ---------------------------------------------------------------------------
# AC02 — wan/wan21/__init__.py exports "t2v", "i2v", "flf2v"
# ---------------------------------------------------------------------------


def test_wan21_init_exists() -> None:
    assert (_PIPELINES_VIDEO / "wan" / "wan21" / "__init__.py").is_file()


def test_wan21_init_parses() -> None:
    source = (_PIPELINES_VIDEO / "wan" / "wan21" / "__init__.py").read_text(encoding="utf-8")
    ast.parse(source)


def test_wan21_init_exports_t2v() -> None:
    import comfy_diffusion.pipelines.video.wan.wan21 as wan21_pkg

    assert "t2v" in wan21_pkg.__all__, (
        f"wan21/__init__.py __all__ must contain 't2v', got {wan21_pkg.__all__!r}"
    )


def test_wan21_init_exports_i2v() -> None:
    import comfy_diffusion.pipelines.video.wan.wan21 as wan21_pkg

    assert "i2v" in wan21_pkg.__all__, (
        f"wan21/__init__.py __all__ must contain 'i2v', got {wan21_pkg.__all__!r}"
    )


def test_wan21_init_exports_flf2v() -> None:
    import comfy_diffusion.pipelines.video.wan.wan21 as wan21_pkg

    assert "flf2v" in wan21_pkg.__all__, (
        f"wan21/__init__.py __all__ must contain 'flf2v', got {wan21_pkg.__all__!r}"
    )


def test_wan21_init_all_is_list() -> None:
    import comfy_diffusion.pipelines.video.wan.wan21 as wan21_pkg

    assert isinstance(wan21_pkg.__all__, list)


# ---------------------------------------------------------------------------
# AC03 — video/__init__.py __all__ includes "wan"
# ---------------------------------------------------------------------------


def test_video_init_exists() -> None:
    assert (_PIPELINES_VIDEO / "__init__.py").is_file()


def test_video_init_parses() -> None:
    source = (_PIPELINES_VIDEO / "__init__.py").read_text(encoding="utf-8")
    ast.parse(source)


def test_video_init_all_includes_wan() -> None:
    import comfy_diffusion.pipelines.video as video_pkg

    assert "wan" in video_pkg.__all__, (
        f"video/__init__.py __all__ must include 'wan', got {video_pkg.__all__!r}"
    )


def test_video_init_all_includes_ltx() -> None:
    import comfy_diffusion.pipelines.video as video_pkg

    assert "ltx" in video_pkg.__all__, (
        "video/__init__.py __all__ must also include 'ltx' (consistency check)"
    )


# ---------------------------------------------------------------------------
# AC04 — Importability / typecheck
# ---------------------------------------------------------------------------


def test_wan_sub_package_importable() -> None:
    mod = importlib.import_module("comfy_diffusion.pipelines.video.wan")
    assert mod is not None


def test_wan21_sub_package_importable() -> None:
    mod = importlib.import_module("comfy_diffusion.pipelines.video.wan.wan21")
    assert mod is not None


def test_wan21_t2v_module_importable() -> None:
    mod = importlib.import_module("comfy_diffusion.pipelines.video.wan.wan21.t2v")
    assert callable(mod.manifest)
    assert callable(mod.run)


def test_wan21_i2v_module_importable() -> None:
    mod = importlib.import_module("comfy_diffusion.pipelines.video.wan.wan21.i2v")
    assert callable(mod.manifest)
    assert callable(mod.run)


def test_wan21_flf2v_module_importable() -> None:
    mod = importlib.import_module("comfy_diffusion.pipelines.video.wan.wan21.flf2v")
    assert callable(mod.manifest)
    assert callable(mod.run)


def test_video_namespace_consistent_with_ltx() -> None:
    """wan and ltx must both appear in video.__all__ for namespace symmetry."""
    import comfy_diffusion.pipelines.video as video_pkg

    assert set(["ltx", "wan"]).issubset(set(video_pkg.__all__)), (
        f"video.__all__ must include both 'ltx' and 'wan', got {video_pkg.__all__!r}"
    )
