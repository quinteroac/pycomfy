"""Tests for LTXV conditioning helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import comfy_diffusion.conditioning as conditioning_module
from comfy_diffusion.conditioning import ltxv_conditioning, ltxv_img_to_video


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
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        device: str = "fake-device",
        dtype: str = "fake-dtype",
    ) -> None:
        self.shape = shape
        self.device = device
        self.dtype = dtype

    def __getitem__(self, _: Any) -> _FakeTensor:
        return self

    def __setitem__(self, _: Any, __: Any) -> None:
        return None

    def movedim(self, _: int, __: int) -> _FakeTensor:
        return self


class _FakeTorch:
    float32 = "float32"

    def __init__(self) -> None:
        self.zeros_calls: list[tuple[list[int], str]] = []
        self.ones_calls: list[tuple[tuple[int, ...], str, str]] = []

    def zeros(self, shape: list[int], *, device: str) -> _FakeTensor:
        self.zeros_calls.append((shape, device))
        return _FakeTensor(tuple(shape), device=device)

    def ones(self, shape: tuple[int, ...], *, dtype: str, device: str) -> _FakeTensor:
        self.ones_calls.append((shape, dtype, device))
        return _FakeTensor(shape, dtype=dtype, device=device)


class _FakeModelManagement:
    @staticmethod
    def intermediate_device() -> str:
        return "fake-device"


class _FakeComfyUtils:
    @staticmethod
    def common_upscale(
        image: _FakeTensor,
        width: int,
        height: int,
        upscale_method: str,
        crop_mode: str,
    ) -> _FakeTensor:
        assert image.shape == (3, 512, 768, 3)
        assert width == 768
        assert height == 512
        assert upscale_method == "bilinear"
        assert crop_mode == "center"
        return image


class _FakeNodeHelpers:
    calls: list[tuple[Any, dict[str, Any]]] = []

    @classmethod
    def conditioning_set_values(cls, conditioning: Any, values: dict[str, Any]) -> list[Any]:
        cls.calls.append((conditioning, values))
        updated: list[Any] = []
        for token, metadata in conditioning:
            copied = metadata.copy()
            copied.update(values)
            updated.append([token, copied])
        return updated


class _FakeVae:
    def __init__(self) -> None:
        self.encode_calls: list[Any] = []

    def encode(self, image: Any) -> _FakeTensor:
        self.encode_calls.append(image)
        return _FakeTensor((1, 128, 2, 16, 24), device="latent-device")


def test_ltxv_img_to_video_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_img_to_video)
    assert (
        str(signature)
        == "(positive: 'Any', negative: 'Any', image: 'Any', vae: '_LtxvVaeEncoder', "
        "width: 'int' = 768, height: 'int' = 512, length: 'int' = 97, "
        "batch_size: 'int' = 1, strength: 'float' = 1.0) "
        "-> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_ltxv_conditioning_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_conditioning)
    assert (
        str(signature)
        == "(positive: 'Any', negative: 'Any', frame_rate: 'float' = 25.0) "
        "-> 'tuple[Any, Any]'"
    )


def test_ltxv_img_to_video_returns_positive_negative_and_latent_with_noise_mask(
    monkeypatch: Any,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(
        conditioning_module,
        "_get_ltxv_conditioning_dependencies",
        lambda: (fake_torch, _FakeModelManagement, _FakeComfyUtils, _FakeNodeHelpers),
    )

    positive = [["p", {"seed": 1}]]
    negative = [["n", {"seed": 2}]]
    vae = _FakeVae()
    image = _FakeTensor((3, 512, 768, 3))

    output_positive, output_negative, latent_with_noise_mask = ltxv_img_to_video(
        positive=positive,
        negative=negative,
        image=image,
        vae=vae,
    )

    assert output_positive == positive
    assert output_negative == negative
    assert len(vae.encode_calls) == 1
    assert latent_with_noise_mask["samples"].shape == (1, 128, 13, 16, 24)
    assert latent_with_noise_mask["noise_mask"].shape == (1, 1, 13, 1, 1)
    assert fake_torch.zeros_calls == [([1, 128, 13, 16, 24], "fake-device")]
    assert fake_torch.ones_calls == [((1, 1, 13, 1, 1), "float32", "fake-device")]


def test_ltxv_conditioning_injects_frame_rate_metadata(monkeypatch: Any) -> None:
    _FakeNodeHelpers.calls = []
    monkeypatch.setattr(
        conditioning_module,
        "_get_ltxv_conditioning_dependencies",
        lambda: (_FakeTorch(), _FakeModelManagement, _FakeComfyUtils, _FakeNodeHelpers),
    )

    positive = [["p", {"source": "pos"}]]
    negative = [["n", {"source": "neg"}]]

    output_positive, output_negative = ltxv_conditioning(
        positive=positive,
        negative=negative,
        frame_rate=23.976,
    )

    assert output_positive == [["p", {"source": "pos", "frame_rate": 23.976}]]
    assert output_negative == [["n", {"source": "neg", "frame_rate": 23.976}]]
    assert len(_FakeNodeHelpers.calls) == 2
    assert _FakeNodeHelpers.calls[0][1] == {"frame_rate": 23.976}
    assert _FakeNodeHelpers.calls[1][1] == {"frame_rate": 23.976}
    assert "frame_rate" not in positive[0][1]
    assert "frame_rate" not in negative[0][1]


def test_ltxv_conditioning_helpers_do_not_use_clip_vision() -> None:
    assert "clip_vision" not in inspect.getsource(ltxv_img_to_video)
    assert "clip_vision" not in inspect.getsource(ltxv_conditioning)


def test_ltxv_functions_import_without_torch_or_comfy_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.conditioning import ltxv_conditioning, ltxv_img_to_video\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'ltxv_img_to_video': ltxv_img_to_video.__name__,\n"
        "  'ltxv_conditioning': ltxv_conditioning.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'comfy_loaded': 'comfy' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["ltxv_img_to_video"] == "ltxv_img_to_video"
    assert payload["ltxv_conditioning"] == "ltxv_conditioning"
    assert payload["torch_loaded"] is False
    assert payload["comfy_loaded"] is False
    heavy = [
        module
        for module in payload["new_modules"]
        if module.startswith(("torch", "comfy.")) or module == "comfy"
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
