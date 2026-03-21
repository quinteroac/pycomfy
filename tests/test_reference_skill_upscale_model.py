"""Tests for US-004 — reference skill documents load_upscale_model."""

from __future__ import annotations

from pathlib import Path


def _skill_md() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root
        / "comfy_diffusion"
        / "skills"
        / "comfy-diffusion-reference"
        / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_skill_md_documents_load_upscale_model() -> None:
    """AC01: SKILL.md contains a load_upscale_model entry."""
    assert "load_upscale_model" in _skill_md()


def test_skill_md_documents_path_resolution_order() -> None:
    """AC02: The entry documents the path resolution order."""
    content = _skill_md()
    assert "Absolute path" in content or "absolute path" in content
    assert "upscale_models" in content


def test_skill_md_load_upscale_model_style_consistent_with_other_loaders() -> None:
    """AC03: Style is consistent with other loader entries (load_vae, load_llm, etc.)."""
    content = _skill_md()
    # All loaders use 'str | Path' type annotation
    assert "str | Path" in content
    # The entry uses a docstring style matching other loaders
    assert "def load_upscale_model(self, path: str | Path) -> Any:" in content
