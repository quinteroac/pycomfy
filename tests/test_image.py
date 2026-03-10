"""Tests for image loading helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageOps, features

import comfy_diffusion
import comfy_diffusion.image as image_module
from comfy_diffusion.image import image_pad_for_outpaint, image_upscale_with_model, load_image


class _FakeTensor:
    def __init__(self, data: Any, dtype: object, shape: tuple[int, ...] | None = None) -> None:
        if shape is None:
            shape = self._infer_shape(data)
        self._shape = shape
        self._data = self._flatten(data)
        self.dtype = dtype

    @classmethod
    def full(cls, shape: tuple[int, ...], fill: float, dtype: object) -> _FakeTensor:
        total = 1
        for dim in shape:
            total *= dim
        return cls(data=[float(fill)] * total, dtype=dtype, shape=shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def size(self) -> tuple[int, ...]:
        return self._shape

    def tolist(self) -> Any:
        return self._unflatten(self._data, self._shape)

    def unsqueeze(self, dim: int) -> _FakeTensor:
        if dim < 0:
            dim += len(self._shape) + 1
        if dim < 0 or dim > len(self._shape):
            raise IndexError("dimension out of range")
        new_shape = self._shape[:dim] + (1,) + self._shape[dim:]
        return _FakeTensor(data=self._data.copy(), dtype=self.dtype, shape=new_shape)

    def __mul__(self, scalar: float) -> _FakeTensor:
        return _FakeTensor(
            data=[value * float(scalar) for value in self._data],
            dtype=self.dtype,
            shape=self._shape,
        )

    __rmul__ = __mul__

    def __setitem__(self, key: Any, value: Any) -> None:
        index_lists = self._key_to_indices(key)
        flat_indices = list(self._iter_flat_indices(index_lists))
        if isinstance(value, _FakeTensor):
            source_values = value._data
            assert len(source_values) == len(flat_indices)
            for destination, source in zip(flat_indices, source_values):
                self._data[destination] = source
            return

        scalar = float(value)
        for destination in flat_indices:
            self._data[destination] = scalar

    def _key_to_indices(self, key: Any) -> list[list[int]]:
        if not isinstance(key, tuple):
            key = (key,)
        expanded_key = list(key) + [slice(None)] * (len(self._shape) - len(key))
        index_lists: list[list[int]] = []
        for dim, selector in zip(self._shape, expanded_key):
            if isinstance(selector, int):
                normalized = selector if selector >= 0 else dim + selector
                index_lists.append([normalized])
                continue
            start, stop, step = selector.indices(dim)
            index_lists.append(list(range(start, stop, step)))
        return index_lists

    def _iter_flat_indices(self, index_lists: list[list[int]]) -> Any:
        shape = self._shape

        def walk(dimension: int, coordinates: list[int]) -> Any:
            if dimension == len(index_lists):
                yield self._coords_to_flat_index(coordinates, shape)
                return
            for index in index_lists[dimension]:
                coordinates.append(index)
                yield from walk(dimension + 1, coordinates)
                coordinates.pop()

        yield from walk(0, [])

    @staticmethod
    def _coords_to_flat_index(coords: list[int], shape: tuple[int, ...]) -> int:
        index = 0
        stride = 1
        for dim_size, coord in zip(reversed(shape), reversed(coords)):
            index += coord * stride
            stride *= dim_size
        return index

    @staticmethod
    def _flatten(data: Any) -> list[float]:
        if isinstance(data, list):
            flattened: list[float] = []
            for value in data:
                flattened.extend(_FakeTensor._flatten(value))
            return flattened
        return [float(data)]

    @staticmethod
    def _infer_shape(data: Any) -> tuple[int, ...]:
        shape: list[int] = []
        current = data
        while isinstance(current, list):
            shape.append(len(current))
            current = current[0] if current else []
        return tuple(shape)

    @staticmethod
    def _unflatten(values: list[float], shape: tuple[int, ...]) -> Any:
        if not shape:
            return values[0]
        if len(shape) == 1:
            return values.copy()

        chunk = 1
        for dim in shape[1:]:
            chunk *= dim

        nested: list[Any] = []
        for offset in range(0, len(values), chunk):
            nested.append(_FakeTensor._unflatten(values[offset : offset + chunk], shape[1:]))
        return nested


class _FakeTorch:
    float32 = "torch.float32"

    @staticmethod
    def tensor(data: Any, *, dtype: object) -> _FakeTensor:
        return _FakeTensor(data=data, dtype=dtype)

    @staticmethod
    def ones(shape: tuple[int, ...], *, dtype: object) -> _FakeTensor:
        return _FakeTensor.full(shape=shape, fill=1.0, dtype=dtype)

    @staticmethod
    def zeros(shape: tuple[int, ...], *, dtype: object) -> _FakeTensor:
        return _FakeTensor.full(shape=shape, fill=0.0, dtype=dtype)


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        flattened: list[float] = []
        for value in values:
            flattened.extend(_flatten(value))
        return flattened
    return [float(values)]


def _constant_bhwc_image(
    batch: int, height: int, width: int, color: tuple[float, float, float]
) -> _FakeTensor:
    pixel = list(color)
    return _FakeTorch.tensor(
        [[[[pixel[0], pixel[1], pixel[2]] for _ in range(width)] for _ in range(height)]
         for _ in range(batch)],
        dtype=_FakeTorch.float32,
    )


@pytest.fixture(autouse=True)
def _fake_torch_dependency(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        image_module,
        "_get_load_image_dependencies",
        lambda: (Image, ImageOps, _FakeTorch),
    )
    monkeypatch.setattr(image_module, "_get_torch_module", lambda: _FakeTorch)


def test_image_module_exports_expected_entrypoints() -> None:
    assert image_module.__all__ == [
        "load_image",
        "image_pad_for_outpaint",
        "image_upscale_with_model",
    ]


def test_load_image_signature_matches_contract() -> None:
    signature = inspect.signature(load_image)
    assert str(signature) == "(path: 'str | Path') -> 'tuple[Any, Any]'"


def test_load_image_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "load_image")


def test_image_upscale_with_model_signature_matches_contract() -> None:
    signature = inspect.signature(image_upscale_with_model)
    assert str(signature) == "(upscale_model: 'Any', image: 'Any') -> 'Any'"


def test_image_upscale_with_model_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_upscale_with_model")


def test_image_upscale_with_model_wraps_comfyui_node_and_returns_bhwc_tensor(
    monkeypatch: Any,
) -> None:
    upscale_model = object()
    image = object()
    expected_upscaled = object()

    class FakeImageUpscaleWithModel:
        @classmethod
        def execute(cls, received_upscale_model: Any, received_image: Any) -> tuple[Any]:
            assert received_upscale_model is upscale_model
            assert received_image is image
            return (expected_upscaled,)

    monkeypatch.setattr(
        image_module,
        "_get_image_upscale_with_model_type",
        lambda: FakeImageUpscaleWithModel,
    )

    upscaled = image_upscale_with_model(upscale_model, image)

    assert upscaled is expected_upscaled


def test_image_upscale_with_model_supports_comfyui_v3_result_output(
    monkeypatch: Any,
) -> None:
    upscale_model = object()
    image = object()
    expected_upscaled = object()

    class FakeNodeOutput:
        def __init__(self, result: tuple[Any, ...]) -> None:
            self.result = result

    class FakeImageUpscaleWithModel:
        @classmethod
        def execute(cls, received_upscale_model: Any, received_image: Any) -> Any:
            assert received_upscale_model is upscale_model
            assert received_image is image
            return FakeNodeOutput((expected_upscaled,))

    monkeypatch.setattr(
        image_module,
        "_get_image_upscale_with_model_type",
        lambda: FakeImageUpscaleWithModel,
    )

    upscaled = image_upscale_with_model(upscale_model, image)

    assert upscaled is expected_upscaled


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


def test_image_pad_for_outpaint_signature_matches_contract() -> None:
    signature = inspect.signature(image_pad_for_outpaint)
    expected_signature = (
        "(image: 'Any', left: 'int', top: 'int', right: 'int', "
        "bottom: 'int', feathering: 'int') -> 'tuple[Any, Any]'"
    )
    assert str(signature) == expected_signature


def test_image_pad_for_outpaint_returns_padded_image_and_mask_in_bhwc() -> None:
    source = _FakeTorch.tensor(
        [
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]],
            ]
        ],
        dtype=_FakeTorch.float32,
    )

    padded_image, padded_mask = image_pad_for_outpaint(
        source, left=1, top=2, right=3, bottom=1, feathering=0
    )

    assert padded_image.shape == (1, 5, 6, 3)
    assert padded_mask.shape == (1, 5, 6)
    assert padded_image.dtype == _FakeTorch.float32
    assert padded_mask.dtype == _FakeTorch.float32

    padded_image_values = padded_image.tolist()
    assert padded_image_values[0][2][1] == [0.1, 0.2, 0.3]
    assert padded_image_values[0][3][2] == [0.0, 0.1, 0.2]
    assert padded_image_values[0][0][0] == [0.5, 0.5, 0.5]

    expected_mask = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]
    assert padded_mask.tolist()[0] == expected_mask


def test_image_pad_for_outpaint_applies_feathering_gradient() -> None:
    source = _constant_bhwc_image(batch=1, height=6, width=6, color=(0.1, 0.1, 0.1))

    _, padded_mask = image_pad_for_outpaint(source, left=1, top=1, right=1, bottom=1, feathering=2)
    mask_values = padded_mask.tolist()[0]

    assert mask_values[1][1] == 1.0
    assert mask_values[2][2] == 0.25
    assert mask_values[3][3] == 0.0
