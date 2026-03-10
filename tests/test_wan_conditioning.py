"""Tests for WAN conditioning helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import comfy_diffusion.conditioning as conditioning_module
from comfy_diffusion.conditioning import (
    encode_clip_vision,
    wan_first_last_frame_to_video,
    wan_image_to_video,
)


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

    def view(self, *shape: int) -> _FakeTensor:
        return _FakeTensor(shape, device=self.device, dtype=self.dtype)

    def transpose(self, _: int, __: int) -> _FakeTensor:
        return self

    def __mul__(self, _: float) -> _FakeTensor:
        return self


class _FakeTorch:
    def __init__(self) -> None:
        self.zeros_calls: list[tuple[list[int], str]] = []
        self.ones_calls: list[tuple[tuple[int, ...], str | None, str | None]] = []
        self.cat_calls: list[tuple[list[Any], int]] = []

    def zeros(self, shape: list[int], *, device: str) -> _FakeTensor:
        self.zeros_calls.append((shape, device))
        return _FakeTensor(tuple(shape), device=device)

    def ones(
        self,
        shape: tuple[int, ...],
        *,
        device: str | None = None,
        dtype: str | None = None,
    ) -> _FakeTensor:
        self.ones_calls.append((shape, device, dtype))
        return _FakeTensor(shape, device=device or "fake-device", dtype=dtype or "fake-dtype")

    def cat(self, values: list[Any], dim: int) -> _FakeTensor:
        self.cat_calls.append((values, dim))
        return _FakeTensor((1, 2, 3), device="cat-device", dtype="cat-dtype")


class _FakeComfyUtils:
    @staticmethod
    def common_upscale(
        image: _FakeTensor,
        width: int,
        height: int,
        upscale_method: str,
        crop_mode: str,
    ) -> _FakeTensor:
        assert width == 832
        assert height == 480
        assert upscale_method == "bilinear"
        assert crop_mode == "center"
        return image


class _FakeModelManagement:
    @staticmethod
    def intermediate_device() -> str:
        return "fake-device"


class _FakeNodeHelpers:
    @staticmethod
    def conditioning_set_values(conditioning: Any, values: dict[str, Any]) -> list[Any]:
        updated: list[Any] = []
        for token, metadata in conditioning:
            copied = metadata.copy()
            copied.update(values)
            updated.append([token, copied])
        return updated


class _FakeVaeForI2V:
    def __init__(self) -> None:
        self.encode_calls: list[Any] = []

    def encode(self, image: Any) -> _FakeTensor:
        self.encode_calls.append(image)
        return _FakeTensor((1, 16, 5, 60, 104), device="latent-device", dtype="latent-dtype")


class _FakeVaeForFirstLast:
    latent_channels = 16

    def __init__(self) -> None:
        self.encode_calls: list[Any] = []

    def spacial_compression_encode(self) -> int:
        return 8

    def encode(self, image: Any) -> _FakeTensor:
        self.encode_calls.append(image)
        return _FakeTensor((1, 16, 5, 60, 104), device="latent-device", dtype="latent-dtype")


class _FakeClipVisionOutput:
    def __init__(self, penultimate_hidden_states: Any = None) -> None:
        self.penultimate_hidden_states = penultimate_hidden_states


def test_encode_clip_vision_signature_matches_contract() -> None:
    signature = inspect.signature(encode_clip_vision)
    assert (
        str(signature)
        == "(clip_vision: '_ClipVisionEncoder', image: 'Any', "
        "crop: \"Literal['center', 'none']\" = 'center') -> 'Any'"
    )


def test_encode_clip_vision_calls_encode_image_with_expected_crop_flag() -> None:
    expected_output = object()

    class FakeClipVision:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, bool]] = []

        def encode_image(self, image: Any, crop: bool = True) -> Any:
            self.calls.append((image, crop))
            return expected_output

    clip_vision = FakeClipVision()
    image = object()

    center_output = encode_clip_vision(clip_vision, image, crop="center")
    none_output = encode_clip_vision(clip_vision, image, crop="none")

    assert center_output is expected_output
    assert none_output is expected_output
    assert clip_vision.calls == [(image, True), (image, False)]


def test_wan_image_to_video_signature_matches_contract() -> None:
    signature = inspect.signature(wan_image_to_video)
    assert (
        str(signature)
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 832, "
        "height: 'int' = 480, length: 'int' = 81, batch_size: 'int' = 1, "
        "start_image: 'Any | None' = None, clip_vision_output: 'Any | None' = None) "
        "-> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_wan_image_to_video_returns_positive_negative_and_latent_tuple(
    monkeypatch: Any,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(
        conditioning_module,
        "_get_wan_conditioning_dependencies",
        lambda: (fake_torch, _FakeModelManagement, _FakeComfyUtils, _FakeNodeHelpers),
    )

    positive = [["p", {"seed": 1}]]
    negative = [["n", {"seed": 2}]]

    output_positive, output_negative, latent = wan_image_to_video(
        positive=positive,
        negative=negative,
        vae=_FakeVaeForI2V(),
    )

    assert output_positive == positive
    assert output_negative == negative
    assert isinstance(latent, dict)
    assert "samples" in latent
    assert latent["samples"].shape == (1, 16, 21, 60, 104)
    assert fake_torch.zeros_calls == [([1, 16, 21, 60, 104], "fake-device")]


def test_wan_image_to_video_applies_concat_and_optional_clip_vision_output(
    monkeypatch: Any,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(
        conditioning_module,
        "_get_wan_conditioning_dependencies",
        lambda: (fake_torch, _FakeModelManagement, _FakeComfyUtils, _FakeNodeHelpers),
    )

    vae = _FakeVaeForI2V()
    start_image = _FakeTensor((2, 480, 832, 3), device="start-device", dtype="start-dtype")
    clip_vision_output = object()
    positive = [["p", {"source": "pos"}]]
    negative = [["n", {"source": "neg"}]]

    output_positive, output_negative, _ = wan_image_to_video(
        positive=positive,
        negative=negative,
        vae=vae,
        start_image=start_image,
        clip_vision_output=clip_vision_output,
    )

    assert len(vae.encode_calls) == 1
    assert "concat_latent_image" in output_positive[0][1]
    assert "concat_mask" in output_positive[0][1]
    assert output_positive[0][1]["clip_vision_output"] is clip_vision_output
    assert "concat_latent_image" in output_negative[0][1]
    assert "concat_mask" in output_negative[0][1]
    assert output_negative[0][1]["clip_vision_output"] is clip_vision_output
    assert "concat_latent_image" not in positive[0][1]
    assert "concat_latent_image" not in negative[0][1]


def test_wan_first_last_frame_to_video_signature_matches_contract() -> None:
    signature = inspect.signature(wan_first_last_frame_to_video)
    assert (
        str(signature)
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 832, "
        "height: 'int' = 480, length: 'int' = 81, batch_size: 'int' = 1, "
        "start_image: 'Any | None' = None, end_image: 'Any | None' = None, "
        "clip_vision_start_image: 'Any | None' = None, "
        "clip_vision_end_image: 'Any | None' = None) "
        "-> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_wan_first_last_frame_to_video_returns_structure_and_merges_clip_vision(
    monkeypatch: Any,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(
        conditioning_module,
        "_get_wan_conditioning_dependencies",
        lambda: (fake_torch, _FakeModelManagement, _FakeComfyUtils, _FakeNodeHelpers),
    )
    monkeypatch.setattr(
        conditioning_module,
        "_get_clip_vision_output_type",
        lambda: _FakeClipVisionOutput,
    )

    vae = _FakeVaeForFirstLast()
    start_image = _FakeTensor((1, 480, 832, 3), device="start-device", dtype="start-dtype")
    end_image = _FakeTensor((1, 480, 832, 3), device="end-device", dtype="end-dtype")
    start_clip_vision = _FakeClipVisionOutput(penultimate_hidden_states=_FakeTensor((1, 77, 768)))
    end_clip_vision = _FakeClipVisionOutput(penultimate_hidden_states=_FakeTensor((1, 77, 768)))
    positive = [["p", {"source": "pos"}]]
    negative = [["n", {"source": "neg"}]]

    output_positive, output_negative, latent = wan_first_last_frame_to_video(
        positive=positive,
        negative=negative,
        vae=vae,
        start_image=start_image,
        end_image=end_image,
        clip_vision_start_image=start_clip_vision,
        clip_vision_end_image=end_clip_vision,
    )

    assert len(vae.encode_calls) == 1
    assert latent["samples"].shape == (1, 16, 21, 60, 104)
    assert "concat_latent_image" in output_positive[0][1]
    assert "concat_mask" in output_positive[0][1]
    assert "concat_latent_image" in output_negative[0][1]
    assert "concat_mask" in output_negative[0][1]
    assert isinstance(output_positive[0][1]["clip_vision_output"], _FakeClipVisionOutput)
    assert output_positive[0][1]["clip_vision_output"].penultimate_hidden_states.shape == (1, 2, 3)
    assert fake_torch.cat_calls and fake_torch.cat_calls[0][1] == -2


def test_wan_functions_import_without_torch_or_comfy_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.conditioning import (\n"
        "  encode_clip_vision,\n"
        "  wan_first_last_frame_to_video,\n"
        "  wan_image_to_video,\n"
        ")\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'encode_clip_vision': encode_clip_vision.__name__,\n"
        "  'wan_image_to_video': wan_image_to_video.__name__,\n"
        "  'wan_first_last_frame_to_video': wan_first_last_frame_to_video.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'comfy_loaded': 'comfy' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["encode_clip_vision"] == "encode_clip_vision"
    assert payload["wan_image_to_video"] == "wan_image_to_video"
    assert payload["wan_first_last_frame_to_video"] == "wan_first_last_frame_to_video"
    assert payload["torch_loaded"] is False
    assert payload["comfy_loaded"] is False
    heavy = [
        module
        for module in payload["new_modules"]
        if module.startswith(("torch", "comfy.")) or module == "comfy"
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
