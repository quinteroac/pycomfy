"""Tests for US-003 — example file updated to use public API."""

from __future__ import annotations

import ast
from pathlib import Path

EXAMPLE_PATH = (
    Path(__file__).parent.parent / "examples" / "simple_checkpoint_esrgan_upscale_example.py"
)


def _example_source() -> str:
    return EXAMPLE_PATH.read_text(encoding="utf-8")


def _example_ast() -> ast.Module:
    return ast.parse(_example_source())


# ---------------------------------------------------------------------------
# AC01 — _load_image_upscale_model private helper is removed
# ---------------------------------------------------------------------------


def test_private_load_image_upscale_model_not_defined() -> None:
    """The private reimplementation must not exist in the example file."""
    source = _example_source()
    assert "_load_image_upscale_model" not in source, (
        "Private helper _load_image_upscale_model should have been removed"
    )


# ---------------------------------------------------------------------------
# AC02 — ModelManager instance is constructed and load_upscale_model is called
# ---------------------------------------------------------------------------


def test_model_manager_instantiated_in_example() -> None:
    """ModelManager(...) must appear in the example source."""
    source = _example_source()
    assert "ModelManager(" in source, "Example must construct a ModelManager instance"


def test_load_upscale_model_called_on_manager() -> None:
    """manager.load_upscale_model() must be called in the example."""
    source = _example_source()
    assert "load_upscale_model(" in source, (
        "Example must call manager.load_upscale_model() instead of the private helper"
    )


# ---------------------------------------------------------------------------
# AC03 — example still imports image_upscale_with_model and saves upscaled image
# ---------------------------------------------------------------------------


def test_example_still_calls_image_upscale_with_model() -> None:
    """Observable behaviour: image_upscale_with_model must still be invoked."""
    source = _example_source()
    assert "image_upscale_with_model(" in source, (
        "Example must still call image_upscale_with_model to run the upscale step"
    )


def test_example_still_saves_upscaled_image() -> None:
    """Observable behaviour: upscaled image must still be saved."""
    source = _example_source()
    assert "output_upscaled" in source and ".save(" in source, (
        "Example must still save the upscaled image"
    )


# ---------------------------------------------------------------------------
# AC04 — example parses without syntax errors (basic typecheck / lint proxy)
# ---------------------------------------------------------------------------


def test_example_parses_without_syntax_errors() -> None:
    """The example file must be valid Python (no syntax errors)."""
    source = _example_source()
    # ast.parse raises SyntaxError if the file is invalid
    tree = ast.parse(source, filename=str(EXAMPLE_PATH))
    assert isinstance(tree, ast.Module)
