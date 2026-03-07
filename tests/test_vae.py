"""Tests for latent-to-image VAE decode helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

import pycomfy
import pycomfy.vae as vae_module
from pycomfy import vae_decode
from pycomfy.models import CheckpointResult


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    repo_root = str(_repo_root())
    env["PYTHONPATH"] = repo_root if not existing else f"{repo_root}{os.pathsep}{existing}"

    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=env,
        cwd=_repo_root(),
    )


class _FakeTensor:
    def __init__(self, data: Any, *, is_nested: bool = False) -> None:
        self._data = data
        self.is_nested = is_nested

    @property
    def shape(self) -> tuple[int, ...]:
        shape: list[int] = []
        current = self._data
        while isinstance(current, list):
            shape.append(len(current))
            current = current[0] if current else []
        return tuple(shape)

    def unbind(self) -> tuple[_FakeTensor, ...]:
        return tuple(_FakeTensor(self._data[idx]) for idx in range(len(self._data)))

    def reshape(self, *shape: int) -> _FakeTensor:
        if shape == (-1, self.shape[-3], self.shape[-2], self.shape[-1]):
            flat: list[Any] = []
            for batch in self._data:
                flat.extend(batch)
            return _FakeTensor(flat)
        raise AssertionError(f"Unexpected reshape request: {shape}")

    def __getitem__(self, key: Any) -> _FakeTensor:
        return _FakeTensor(self._data[key])

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> Any:
        return self._data


def test_vae_module_exports_only_vae_decode() -> None:
    assert vae_module.__all__ == ["vae_decode"]


def test_vae_decode_is_re_exported_from_package_root() -> None:
    assert callable(vae_decode)
    assert pycomfy.vae_decode is vae_decode
    assert "vae_decode" in pycomfy.__all__


def test_vae_decode_accepts_checkpoint_result_vae_and_returns_pil_image() -> None:
    samples = _FakeTensor([[[[0.0, 0.0, 0.0]]]])
    decoded = _FakeTensor([[[[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]]])
    calls: list[object] = []

    class _FakeVae:
        def decode(self, value: object) -> _FakeTensor:
            calls.append(value)
            return decoded

    checkpoint = CheckpointResult(model=object(), clip=None, vae=_FakeVae())

    image = vae_decode(checkpoint.vae, {"samples": samples})

    assert isinstance(image, Image.Image)
    assert calls == [samples]


def test_vae_decode_outputs_uint8_pixels_in_0_to_255_range() -> None:
    decoded = _FakeTensor(
        [
            [
                [[-0.2, 0.0, 0.5], [1.0, 1.2, 0.25]],
                [[0.75, 0.1, 0.9], [0.4, 0.6, 1.5]],
            ]
        ]
    )

    class _FakeVae:
        def decode(self, _value: object) -> _FakeTensor:
            return decoded

    image = vae_decode(_FakeVae(), {"samples": _FakeTensor([[[[0.0]]]])})
    channels = list(image.tobytes())

    assert image.mode == "RGB"
    assert min(channels) == 0
    assert max(channels) == 255
    assert channels[2] == 127
    assert channels[4] == 255


def test_import_pycomfy_vae_has_no_heavy_import_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import pycomfy\n"
        "baseline_modules = set(sys.modules)\n"
        "from pycomfy.vae import vae_decode\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'func_name': vae_decode.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'nodes_loaded': 'nodes' in sys.modules,\n"
        "  'folder_paths_loaded': 'folder_paths' in sys.modules,\n"
        "  'comfy_sd_loaded': 'comfy.sd' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["func_name"] == "vae_decode"
    assert payload["torch_loaded"] is False
    assert payload["nodes_loaded"] is False
    assert payload["folder_paths_loaded"] is False
    assert payload["comfy_sd_loaded"] is False
    heavy = [
        module
        for module in payload["new_modules"]
        if module == "nodes" or module.startswith(("torch", "folder_paths", "comfy.sd"))
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
