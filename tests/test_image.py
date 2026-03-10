"""Tests for image loading helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageOps, features

import comfy_diffusion
import comfy_diffusion.image as image_module
from comfy_diffusion.image import load_image


class _FakeTensor:
    def __init__(self, data: Any, dtype: object) -> None:
        self._data = data
        self.dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        shape: list[int] = []
        current = self._data
        while isinstance(current, list):
            shape.append(len(current))
            current = current[0] if current else []
        return tuple(shape)

    def tolist(self) -> Any:
        return self._data


class _FakeTorch:
    float32 = "torch.float32"

    @staticmethod
    def tensor(data: Any, *, dtype: object) -> _FakeTensor:
        return _FakeTensor(data=data, dtype=dtype)


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        flattened: list[float] = []
        for value in values:
            flattened.extend(_flatten(value))
        return flattened
    return [float(values)]


@pytest.fixture(autouse=True)
def _fake_torch_dependency(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        image_module,
        "_get_load_image_dependencies",
        lambda: (Image, ImageOps, _FakeTorch),
    )


def test_image_module_exports_load_image_only() -> None:
    assert image_module.__all__ == ["load_image"]


def test_load_image_signature_matches_contract() -> None:
    signature = inspect.signature(load_image)
    assert str(signature) == "(path: 'str | Path') -> 'tuple[Any, Any]'"


def test_load_image_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "load_image")


def test_load_image_returns_bhwc_float32_image_and_hw_mask_from_alpha(tmp_path: Path) -> None:
    image_path = tmp_path / "alpha.png"
    source = Image.new("RGBA", (2, 2))
    source.putdata(
        [
            (255, 0, 0, 255),
            (0, 255, 0, 128),
            (0, 0, 255, 0),
            (255, 255, 255, 64),
        ]
    )
    source.save(image_path, format="PNG")

    image_tensor, mask_tensor = load_image(image_path)

    assert image_tensor.dtype == _FakeTorch.float32
    assert mask_tensor.dtype == _FakeTorch.float32
    assert image_tensor.shape == (1, 2, 2, 3)
    assert mask_tensor.shape == (2, 2)
    assert all(0.0 <= value <= 1.0 for value in _flatten(image_tensor.tolist()))
    assert mask_tensor.tolist() == [
        [0.0, 127 / 255.0],
        [1.0, 191 / 255.0],
    ]


@pytest.mark.parametrize(
    ("suffix", "format_name"),
    [
        (".png", "PNG"),
        (".jpg", "JPEG"),
        (".webp", "WEBP"),
    ],
)
def test_load_image_supports_common_formats(tmp_path: Path, suffix: str, format_name: str) -> None:
    if format_name == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow WEBP support is unavailable in this environment")

    image_path = tmp_path / f"sample{suffix}"
    Image.new("RGB", (3, 2), color=(12, 34, 56)).save(image_path, format=format_name)

    image_tensor, mask_tensor = load_image(image_path)

    assert image_tensor.shape == (1, 2, 3, 3)
    assert mask_tensor.shape == (2, 3)


def test_load_image_without_alpha_returns_fully_opaque_zero_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.jpg"
    Image.new("RGB", (2, 3), color=(11, 22, 33)).save(image_path, format="JPEG")

    _, mask_tensor = load_image(image_path)

    assert mask_tensor.shape == (3, 2)
    assert mask_tensor.tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_load_image_applies_exif_orientation(tmp_path: Path) -> None:
    image_path = tmp_path / "oriented.jpg"
    source = Image.new("RGB", (2, 3))
    source.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
    )
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    source.save(image_path, format="JPEG", exif=exif)

    image_tensor, mask_tensor = load_image(image_path)

    assert image_tensor.shape == (1, 2, 3, 3)
    assert mask_tensor.shape == (2, 3)
