"""Tests for US-005 — package scaffolding for audio pipelines.

Covers:
  - __init__.py files exist for each audio sub-package                    (AC01)
  - comfy_diffusion.pipelines.__all__ lists "audio"                       (AC02)
  - All audio pipeline sub-packages are importable without errors          (AC03)
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# AC01 — __init__.py files exist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path", [
    "comfy_diffusion/pipelines/__init__.py",
    "comfy_diffusion/pipelines/audio/__init__.py",
    "comfy_diffusion/pipelines/audio/ace_step/__init__.py",
    "comfy_diffusion/pipelines/audio/ace_step/v1_5/__init__.py",
])
def test_init_files_exist(rel_path: str) -> None:
    assert (_REPO_ROOT / rel_path).is_file(), f"Missing: {rel_path}"


# ---------------------------------------------------------------------------
# AC02 — pipelines.__all__ lists "audio"
# ---------------------------------------------------------------------------

def test_pipelines_all_contains_audio() -> None:
    import comfy_diffusion.pipelines as pipelines_pkg

    assert "audio" in pipelines_pkg.__all__


# ---------------------------------------------------------------------------
# AC03 — sub-packages are importable and __all__ is correct
# ---------------------------------------------------------------------------

def test_audio_package_importable() -> None:
    import comfy_diffusion.pipelines.audio as audio

    assert "ace_step" in audio.__all__


def test_ace_step_package_importable() -> None:
    import comfy_diffusion.pipelines.audio.ace_step as ace_step

    assert "v1_5" in ace_step.__all__


def test_v1_5_package_importable() -> None:
    import comfy_diffusion.pipelines.audio.ace_step.v1_5 as v1_5

    assert "checkpoint" in v1_5.__all__
    assert "split" in v1_5.__all__
    assert "split_4b" in v1_5.__all__


def test_pipeline_modules_importable() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5 import checkpoint
    from comfy_diffusion.pipelines.audio.ace_step.v1_5 import split
    from comfy_diffusion.pipelines.audio.ace_step.v1_5 import split_4b

    for mod in (checkpoint, split, split_4b):
        assert hasattr(mod, "manifest")
        assert hasattr(mod, "run")
