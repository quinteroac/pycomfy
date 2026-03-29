"""Tests for mask loading and manipulation helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageOps

import comfy_diffusion
import comfy_diffusion.mask as mask_module
from comfy_diffusion.mask import (
    feather_mask,
    grow_mask,
    image_to_mask,
    load_image_mask,
    mask_to_image,
)


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
        mask_module,
        "_get_load_image_dependencies",
        lambda: (Image, ImageOps, _FakeTorch),
    )
    monkeypatch.setattr(mask_module, "_get_torch_module", lambda: _FakeTorch)


def test_mask_module_exports_expected_entrypoints() -> None:
    assert mask_module.__all__ == [
        "load_image_mask",
        "image_to_mask",
        "mask_to_image",
        "grow_mask",
        "feather_mask",
        "solid_mask",
    ]


def test_load_image_mask_signature_matches_contract() -> None:
    signature = inspect.signature(load_image_mask)
    assert str(signature) == "(path: 'str | Path', channel: 'str') -> 'Any'"


def test_image_to_mask_signature_matches_contract() -> None:
    signature = inspect.signature(image_to_mask)
    assert str(signature) == "(image: 'Any', channel: 'str') -> 'Any'"


def test_mask_to_image_signature_matches_contract() -> None:
    signature = inspect.signature(mask_to_image)
    assert str(signature) == "(mask: 'Any') -> 'Any'"


def test_load_image_mask_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "load_image_mask")


def test_image_to_mask_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_to_mask")


def test_mask_to_image_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "mask_to_image")


def test_grow_mask_signature_matches_contract() -> None:
    signature = inspect.signature(grow_mask)
    assert str(signature) == "(mask: 'Any', expand: 'int', tapered_corners: 'bool') -> 'Any'"


def test_grow_mask_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "grow_mask")


def test_feather_mask_signature_matches_contract() -> None:
    signature = inspect.signature(feather_mask)
    expected_signature = (
        "(mask: 'Any', left: 'int', top: 'int', right: 'int', bottom: 'int') -> 'Any'"
    )
    assert str(signature) == expected_signature


def test_feather_mask_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "feather_mask")


@pytest.mark.parametrize(
    ("expand", "tapered_corners", "expected_output"),
    [
        (2, True, [[[0.0, 1.0], [1.0, 1.0]]]),
        (-1, False, [[[0.0, 0.0], [0.0, 0.0]]]),
    ],
)
def test_grow_mask_forwards_expand_and_tapered_corners_to_comfyui_node(
    monkeypatch: Any,
    expand: int,
    tapered_corners: bool,
    expected_output: list[list[list[float]]],
) -> None:
    mask = _FakeTorch.tensor(
        [[[0.0, 0.0], [0.0, 1.0]]],
        dtype=_FakeTorch.float32,
    )
    calls: list[dict[str, Any]] = []

    class FakeGrowMask:
        @classmethod
        def execute(cls, *, mask: Any, expand: int, tapered_corners: bool) -> tuple[Any]:
            calls.append(
                {
                    "mask": mask,
                    "expand": expand,
                    "tapered_corners": tapered_corners,
                }
            )
            return (_FakeTorch.tensor(expected_output, dtype=_FakeTorch.float32),)

    monkeypatch.setattr(mask_module, "_get_grow_mask_type", lambda: FakeGrowMask)

    result = grow_mask(mask=mask, expand=expand, tapered_corners=tapered_corners)

    assert calls == [
        {
            "mask": mask,
            "expand": expand,
            "tapered_corners": tapered_corners,
        }
    ]
    assert result.shape == (1, 2, 2)
    assert result.tolist() == expected_output


def test_grow_mask_tapered_corners_affects_corner_growth(
    monkeypatch: Any,
) -> None:
    mask = _FakeTorch.tensor(
        [[[1.0, 0.0], [0.0, 0.0]]],
        dtype=_FakeTorch.float32,
    )

    class FakeGrowMask:
        @classmethod
        def execute(cls, *, mask: Any, expand: int, tapered_corners: bool) -> tuple[Any]:
            assert expand == 1
            if tapered_corners:
                return (_FakeTorch.tensor([[[1.0, 1.0], [1.0, 0.0]]], dtype=_FakeTorch.float32),)
            return (_FakeTorch.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=_FakeTorch.float32),)

    monkeypatch.setattr(mask_module, "_get_grow_mask_type", lambda: FakeGrowMask)

    rounded = grow_mask(mask=mask, expand=1, tapered_corners=True)
    square = grow_mask(mask=mask, expand=1, tapered_corners=False)

    assert rounded.tolist()[0][1][1] == 0.0
    assert square.tolist()[0][1][1] == 1.0


def test_feather_mask_applies_side_amounts() -> None:
    mask = _FakeTorch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ],
        dtype=_FakeTorch.float32,
    )

    feathered = feather_mask(mask=mask, left=2, top=1, right=3, bottom=2)
    values = feathered.tolist()[0]

    assert feathered.shape == (1, 5, 6)
    assert values[0][0] == pytest.approx(0.5)
    assert values[2][2] == pytest.approx(1.0)
    assert values[4][5] == pytest.approx(1.0 / 6.0)


def test_feather_mask_supports_independent_edge_amounts() -> None:
    mask = _FakeTorch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ],
        dtype=_FakeTorch.float32,
    )

    left_only = feather_mask(mask=mask, left=2, top=0, right=0, bottom=0).tolist()[0]
    top_only = feather_mask(mask=mask, left=0, top=2, right=0, bottom=0).tolist()[0]

    assert left_only[0][0] == pytest.approx(0.5)
    assert left_only[0][3] == pytest.approx(1.0)
    assert top_only[0][0] == pytest.approx(0.5)
    assert top_only[3][0] == pytest.approx(1.0)


def test_feather_mask_clamps_output_values_to_zero_one_range() -> None:
    mask = _FakeTorch.tensor(
        [[[1.2, -0.2], [0.6, 2.0]]],
        dtype=_FakeTorch.float32,
    )

    feathered = feather_mask(mask=mask, left=0, top=0, right=0, bottom=0)
    flattened = _flatten(feathered.tolist())

    assert all(0.0 <= value <= 1.0 for value in flattened)
    assert feathered.tolist() == [[[1.0, 0.0], [0.6, 1.0]]]


def test_load_image_mask_alpha_returns_comfyui_mask_convention(tmp_path: Path) -> None:
    image_path = tmp_path / "alpha_mask.png"
    source = Image.new("RGBA", (2, 2))
    source.putdata(
        [
            (100, 50, 25, 255),  # opaque -> 0.0
            (100, 50, 25, 128),  # partial -> (255 - 128) / 255
            (100, 50, 25, 0),    # transparent -> 1.0
            (100, 50, 25, 64),   # partial -> (255 - 64) / 255
        ]
    )
    source.save(image_path, format="PNG")

    mask_tensor = load_image_mask(image_path, channel="alpha")

    assert mask_tensor.dtype == _FakeTorch.float32
    assert mask_tensor.shape == (1, 2, 2)
    assert mask_tensor.tolist() == [
        [
            [0.0, 127 / 255.0],
            [1.0, 191 / 255.0],
        ]
    ]
    assert all(0.0 <= value <= 1.0 for value in _flatten(mask_tensor.tolist()))


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        ("red", [[[1.0, 0.0], [64 / 255.0, 1.0]]]),
        ("green", [[[0.0, 1.0], [128 / 255.0, 1.0]]]),
        ("blue", [[[0.0, 0.0], [1.0, 1.0]]]),
    ],
)
def test_load_image_mask_color_channels_map_0_255_to_0_1(
    tmp_path: Path,
    channel: str,
    expected: list[list[list[float]]],
) -> None:
    image_path = tmp_path / "color_mask.png"
    source = Image.new("RGB", (2, 2))
    source.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (64, 128, 255),
            (255, 255, 255),
        ]
    )
    source.save(image_path, format="PNG")

    mask_tensor = load_image_mask(image_path, channel=channel)

    assert mask_tensor.dtype == _FakeTorch.float32
    assert mask_tensor.shape == (1, 2, 2)
    assert mask_tensor.tolist() == expected
    assert all(0.0 <= value <= 1.0 for value in _flatten(mask_tensor.tolist()))


@pytest.mark.parametrize("channel", ["alpha", "red", "green", "blue"])
def test_load_image_mask_supports_expected_channels(tmp_path: Path, channel: str) -> None:
    image_path = tmp_path / "channels.png"
    Image.new("RGBA", (1, 1), color=(16, 32, 64, 128)).save(image_path, format="PNG")

    mask_tensor = load_image_mask(image_path, channel=channel)

    assert mask_tensor.shape == (1, 1, 1)


def test_load_image_mask_rejects_unsupported_channel(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (1, 1), color=(0, 0, 0)).save(image_path, format="PNG")

    with pytest.raises(ValueError, match="channel must be one of: alpha, red, green, blue"):
        load_image_mask(image_path, channel="luma")


def test_load_image_mask_raises_file_not_found_for_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        load_image_mask(missing_path, channel="alpha")


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        ("red", [[[1.0, 0.1]], [[0.5, 0.0]]]),
        ("green", [[[0.0, 0.2]], [[0.6, 0.0]]]),
        ("blue", [[[0.4, 0.3]], [[0.7, 1.0]]]),
    ],
)
def test_image_to_mask_extracts_supported_channels_to_bhw_float32(
    channel: str,
    expected: list[list[list[float]]],
) -> None:
    image = _FakeTorch.tensor(
        [
            [[[1.0, 0.0, 0.4], [0.1, 0.2, 0.3]]],
            [[[0.5, 0.6, 0.7], [0.0, 0.0, 1.0]]],
        ],
        dtype=_FakeTorch.float32,
    )

    mask = image_to_mask(image=image, channel=channel)

    assert mask.dtype == _FakeTorch.float32
    assert mask.shape == (2, 1, 2)
    assert mask.tolist() == expected


def test_image_to_mask_rejects_unsupported_channel() -> None:
    image = _constant_bhwc_image(batch=1, height=1, width=1, color=(0.1, 0.2, 0.3))

    with pytest.raises(ValueError, match="channel must be one of: red, green, blue"):
        image_to_mask(image=image, channel="alpha")


def test_image_to_mask_rejects_non_bhwc_shape() -> None:
    image = _FakeTorch.tensor([[[0.1, 0.2, 0.3]]], dtype=_FakeTorch.float32)

    with pytest.raises(ValueError, match="image tensor must have shape \\(B, H, W, C\\)"):
        image_to_mask(image=image, channel="red")


def test_mask_to_image_replicates_mask_values_across_rgb_channels() -> None:
    mask = _FakeTorch.tensor(
        [
            [[0.0, 0.25], [0.5, 1.0]],
            [[1.0, 0.5], [0.25, 0.0]],
        ],
        dtype=_FakeTorch.float32,
    )

    image = mask_to_image(mask)

    assert image.dtype == _FakeTorch.float32
    assert image.shape == (2, 2, 2, 3)
    assert image.tolist() == [
        [
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        ],
        [
            [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
            [[0.25, 0.25, 0.25], [0.0, 0.0, 0.0]],
        ],
    ]


def test_mask_to_image_rejects_non_bhw_shape() -> None:
    mask = _FakeTorch.tensor([[[[0.5]]]], dtype=_FakeTorch.float32)

    with pytest.raises(ValueError, match="mask tensor must have shape \\(B, H, W\\)"):
        mask_to_image(mask)
