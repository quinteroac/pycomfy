"""Tests for it_000034 image wrappers: image_scale_to_total_pixels,
image_scale_to_max_dimension, and get_image_size."""

from __future__ import annotations

import ast
import pathlib
from typing import Any
from unittest.mock import MagicMock

import pytest

import comfy_diffusion
import comfy_diffusion.image as image_module
from comfy_diffusion.image import (
    get_image_size,
    image_scale_to_max_dimension,
    image_scale_to_total_pixels,
)

_IMAGE_FILE = pathlib.Path(image_module.__file__)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_image_tensor(batch: int = 1, height: int = 64, width: int = 64) -> Any:
    """Return a minimal fake IMAGE tensor (BHWC layout) backed by a MagicMock."""
    tensor = MagicMock()
    tensor.shape = (batch, height, width, 3)
    return tensor


def _make_fake_node_output(*values: Any) -> Any:
    output = MagicMock()
    output.result = list(values)
    return output


# ---------------------------------------------------------------------------
# image_scale_to_total_pixels — AC01
# ---------------------------------------------------------------------------


def test_image_scale_to_total_pixels_is_callable() -> None:
    assert callable(image_scale_to_total_pixels)


def test_image_scale_to_total_pixels_in_dunder_all() -> None:
    """AC04: appears in image.__all__."""
    assert "image_scale_to_total_pixels" in image_module.__all__


def test_image_scale_to_total_pixels_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_scale_to_total_pixels")


def test_image_scale_to_total_pixels_returns_tensor(monkeypatch: Any) -> None:
    """AC01/AC05: returns the tensor produced by the node."""
    expected_output = MagicMock(name="scaled_tensor")

    class _FakeNode:
        @classmethod
        def execute(cls, **kwargs: Any) -> Any:
            return _make_fake_node_output(expected_output)

    monkeypatch.setattr(
        image_module,
        "_get_image_scale_to_total_pixels_type",
        lambda: _FakeNode,
    )

    result = image_scale_to_total_pixels(
        image=_make_fake_image_tensor(),
        upscale_method="lanczos",
        megapixels=1.0,
        smallest_side=8,
    )
    assert result is expected_output


def test_image_scale_to_total_pixels_passes_args_correctly(monkeypatch: Any) -> None:
    """AC01: node receives the correct keyword arguments."""
    captured: dict[str, Any] = {}
    fake_output = MagicMock(name="output")

    class _FakeNode:
        @classmethod
        def execute(cls, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _make_fake_node_output(fake_output)

    monkeypatch.setattr(
        image_module,
        "_get_image_scale_to_total_pixels_type",
        lambda: _FakeNode,
    )

    fake_image = _make_fake_image_tensor()
    image_scale_to_total_pixels(
        image=fake_image,
        upscale_method="bilinear",
        megapixels=2.5,
        smallest_side=16,
    )

    assert captured["image"] is fake_image
    assert captured["upscale_method"] == "bilinear"
    assert captured["megapixels"] == 2.5
    assert captured["resolution_steps"] == 16


# ---------------------------------------------------------------------------
# image_scale_to_max_dimension — AC02
# ---------------------------------------------------------------------------


def test_image_scale_to_max_dimension_is_callable() -> None:
    assert callable(image_scale_to_max_dimension)


def test_image_scale_to_max_dimension_in_dunder_all() -> None:
    """AC04: appears in image.__all__."""
    assert "image_scale_to_max_dimension" in image_module.__all__


def test_image_scale_to_max_dimension_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_scale_to_max_dimension")


def test_image_scale_to_max_dimension_returns_tensor(monkeypatch: Any) -> None:
    """AC02/AC05: returns the tensor produced by the node."""
    expected_output = MagicMock(name="scaled_tensor")

    class _FakeNode:
        @classmethod
        def execute(cls, **kwargs: Any) -> Any:
            return _make_fake_node_output(expected_output)

    monkeypatch.setattr(
        image_module,
        "_get_image_scale_to_max_dimension_type",
        lambda: _FakeNode,
    )

    result = image_scale_to_max_dimension(
        image=_make_fake_image_tensor(),
        upscale_method="lanczos",
        max_dimension=512,
    )
    assert result is expected_output


def test_image_scale_to_max_dimension_passes_args_correctly(monkeypatch: Any) -> None:
    """AC02: node receives the correct keyword arguments."""
    captured: dict[str, Any] = {}
    fake_output = MagicMock(name="output")

    class _FakeNode:
        @classmethod
        def execute(cls, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _make_fake_node_output(fake_output)

    monkeypatch.setattr(
        image_module,
        "_get_image_scale_to_max_dimension_type",
        lambda: _FakeNode,
    )

    fake_image = _make_fake_image_tensor()
    image_scale_to_max_dimension(
        image=fake_image,
        upscale_method="bicubic",
        max_dimension=1024,
    )

    assert captured["image"] is fake_image
    assert captured["upscale_method"] == "bicubic"
    assert captured["largest_size"] == 1024


# ---------------------------------------------------------------------------
# get_image_size — AC03
# ---------------------------------------------------------------------------


def test_get_image_size_is_callable() -> None:
    assert callable(get_image_size)


def test_get_image_size_in_dunder_all() -> None:
    """AC04: appears in image.__all__."""
    assert "get_image_size" in image_module.__all__


def test_get_image_size_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "get_image_size")


def test_get_image_size_returns_width_height_tuple(monkeypatch: Any) -> None:
    """AC03/AC05: returns (width, height) as a 2-tuple of ints."""

    class _FakeNode:
        @classmethod
        def execute(cls, **kwargs: Any) -> Any:
            return _make_fake_node_output(320, 240, 1)  # width, height, batch_size

    monkeypatch.setattr(
        image_module,
        "_get_get_image_size_type",
        lambda: _FakeNode,
    )

    result = get_image_size(image=_make_fake_image_tensor(height=240, width=320))

    assert result == (320, 240)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)


def test_get_image_size_passes_image_to_node(monkeypatch: Any) -> None:
    """AC03: node receives the image argument."""
    captured: dict[str, Any] = {}

    class _FakeNode:
        @classmethod
        def execute(cls, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _make_fake_node_output(100, 200, 1)

    monkeypatch.setattr(
        image_module,
        "_get_get_image_size_type",
        lambda: _FakeNode,
    )

    fake_image = _make_fake_image_tensor(height=200, width=100)
    get_image_size(image=fake_image)

    assert captured["image"] is fake_image


# ---------------------------------------------------------------------------
# Lazy-import validation — AC01/AC02/AC03/AC06
# ---------------------------------------------------------------------------


def _get_top_level_imports(filepath: pathlib.Path) -> list[str]:
    tree = ast.parse(filepath.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if node.col_offset != 0:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_image_no_top_level_comfy_or_torch_import() -> None:
    """AC06: image.py has no top-level comfy.* or torch imports."""
    top_level = _get_top_level_imports(_IMAGE_FILE)
    bad = [m for m in top_level if m.startswith("comfy") or m == "torch"]
    assert not bad, f"image.py has top-level comfy/torch imports: {bad}"
