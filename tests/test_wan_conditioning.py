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
    wan_animate_to_video,
    wan_camera_embedding,
    wan_camera_image_to_video,
    wan_first_last_frame_to_video,
    wan_fun_control_to_video,
    wan_fun_inpaint_to_video,
    wan_humo_image_to_video,
    wan_image_to_video,
    wan_infinite_talk_to_video,
    wan_phantom_subject_to_video,
    wan_scail_to_video,
    wan_sound_image_to_video,
    wan_sound_image_to_video_extend,
    wan_track_to_video,
    wan22_fun_control_to_video,
    wan22_image_to_video_latent,
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

    def __mul__(self, _: Any) -> _FakeTensor:
        return self

    def __rmul__(self, _: Any) -> _FakeTensor:
        return self

    def __add__(self, _: Any) -> _FakeTensor:
        return self

    def __radd__(self, _: Any) -> _FakeTensor:
        return self

    def __sub__(self, _: Any) -> _FakeTensor:
        return self

    def __rsub__(self, _: Any) -> _FakeTensor:
        return self

    def repeat(self, *_: Any) -> _FakeTensor:
        return self

    def unsqueeze(self, _: int) -> _FakeTensor:
        return self

    def expand(self, *_: Any) -> _FakeTensor:
        return self

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    @property
    def ndim(self) -> int:
        return len(self.shape)


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


# ── Shared fakes for vace-based functions ────────────────────────────────────


class _FakeTorchVace(_FakeTorch):
    """Extended fake torch that also tracks zeros_like and cat calls."""

    def __init__(self) -> None:
        super().__init__()
        self.zeros_like_calls: list[Any] = []
        self.cat_batches: list[Any] = []

    def zeros_like(self, tensor: Any) -> _FakeTensor:
        self.zeros_like_calls.append(tensor)
        return _FakeTensor(tensor.shape, device="fake-device")

    def cat(self, tensors: Any, dim: int = 0) -> _FakeTensor:  # type: ignore[override]
        self.cat_batches.append((tensors, dim))
        return _FakeTensor((1, 16, 5, 60, 104), device="cat-device")

    def ones(  # type: ignore[override]
        self,
        shape: Any,
        *,
        device: str | None = None,
        dtype: str | None = None,
    ) -> _FakeTensor:
        return _FakeTensor(tuple(shape) if not isinstance(shape, tuple) else shape, device=device or "fake-device")

    def zeros(self, shape: Any, *, device: str | None = None, dtype: Any = None) -> _FakeTensor:  # type: ignore[override]
        self.zeros_calls.append((list(shape) if not isinstance(shape, list) else shape, device))
        return _FakeTensor(tuple(shape) if not isinstance(shape, tuple) else shape, device=device or "fake-device")


class _FakeComfyUtilsGeneric:
    @staticmethod
    def common_upscale(
        image: Any, width: int, height: int, upscale_method: str, crop_mode: str
    ) -> Any:
        return image


class _FakeNodeHelpersExtended:
    @staticmethod
    def conditioning_set_values(
        conditioning: Any, values: dict[str, Any], *, append: bool = False
    ) -> list[Any]:
        updated: list[Any] = []
        for token, metadata in conditioning:
            copied = metadata.copy()
            if append:
                for k, v in values.items():
                    existing = copied.get(k, [])
                    copied[k] = list(existing) + list(v) if isinstance(existing, list) else v
            else:
                copied.update(values)
            updated.append([token, copied])
        return updated

    @staticmethod
    def conditioning_set_values_with_timestep_range(
        conditioning: Any, values: dict[str, Any], start: float, end: float
    ) -> list[Any]:
        updated: list[Any] = []
        for token, metadata in conditioning:
            copied = metadata.copy()
            copied.update(values)
            updated.append([token, copied])
        return updated


class _FakeLatentFormats:
    class _FakeFormat:
        def process_out(self, tensor: Any) -> Any:
            return tensor

    class Wan21(_FakeFormat):
        pass

    class Wan22(_FakeFormat):
        pass


class _FakeVaeVace:
    latent_channels = 16

    def __init__(self) -> None:
        self.encode_calls: list[Any] = []

    def spacial_compression_encode(self) -> int:
        return 8

    def encode(self, image: Any) -> _FakeTensor:
        self.encode_calls.append(image)
        return _FakeTensor((1, 16, 5, 60, 104), device="latent-device")


def _make_vace_deps(fake_torch: _FakeTorchVace | None = None) -> Any:
    t = fake_torch or _FakeTorchVace()
    return lambda: (t, _FakeModelManagement, _FakeComfyUtilsGeneric, _FakeNodeHelpersExtended, _FakeLatentFormats)


# ── wan_fun_control_to_video ──────────────────────────────────────────────────


def test_wan_fun_control_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_fun_control_to_video))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 832, "
        "height: 'int' = 480, length: 'int' = 81, batch_size: 'int' = 1, *, "
        "clip_vision_output: 'Any | None' = None, start_image: 'Any | None' = None, "
        "control_video: 'Any | None' = None) -> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_wan_fun_control_to_video_returns_latent_with_correct_shape(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps())
    positive = [["p", {}]]
    negative = [["n", {}]]
    _, _, latent = wan_fun_control_to_video(positive=positive, negative=negative, vae=_FakeVaeVace())
    assert "samples" in latent
    assert latent["samples"].shape == (1, 16, 21, 60, 104)


def test_wan_fun_control_to_video_sets_concat_latent_image_on_conditioning(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps())
    positive = [["p", {}]]
    negative = [["n", {}]]
    out_pos, out_neg, _ = wan_fun_control_to_video(positive=positive, negative=negative, vae=_FakeVaeVace())
    assert "concat_latent_image" in out_pos[0][1]
    assert "concat_latent_image" in out_neg[0][1]


# ── wan22_fun_control_to_video ────────────────────────────────────────────────


def test_wan22_fun_control_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan22_fun_control_to_video))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 832, "
        "height: 'int' = 480, length: 'int' = 81, batch_size: 'int' = 1, *, "
        "ref_image: 'Any | None' = None, "
        "control_video: 'Any | None' = None) -> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_wan22_fun_control_to_video_returns_latent_with_samples(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps())
    positive = [["p", {}]]
    negative = [["n", {}]]
    _, _, latent = wan22_fun_control_to_video(positive=positive, negative=negative, vae=_FakeVaeVace())
    assert "samples" in latent


# ── wan_fun_inpaint_to_video ──────────────────────────────────────────────────


def test_wan_fun_inpaint_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_fun_inpaint_to_video))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 832, "
        "height: 'int' = 480, length: 'int' = 81, batch_size: 'int' = 1, *, "
        "clip_vision_output: 'Any | None' = None, start_image: 'Any | None' = None, "
        "end_image: 'Any | None' = None) -> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_wan_fun_inpaint_to_video_delegates_to_first_last_frame(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_first_last(**kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
        calls.append(kwargs)
        return kwargs["positive"], kwargs["negative"], {"samples": object()}

    monkeypatch.setattr(conditioning_module, "wan_first_last_frame_to_video", _fake_first_last)
    positive = [["p", {}]]
    negative = [["n", {}]]
    wan_fun_inpaint_to_video(positive=positive, negative=negative, vae=_FakeVaeVace())
    assert len(calls) == 1
    assert calls[0]["positive"] is positive
    assert calls[0]["negative"] is negative


# ── wan_camera_embedding ──────────────────────────────────────────────────────


def test_wan_camera_embedding_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_camera_embedding))
        == "(camera_pose: 'str', width: 'int' = 832, height: 'int' = 480, "
        "length: 'int' = 81, *, speed: 'float' = 1.0, fx: 'float' = 0.5, "
        "fy: 'float' = 0.5, cx: 'float' = 0.5, cy: 'float' = 0.5) "
        "-> 'tuple[Any, int, int, int]'"
    )


# ── wan_camera_image_to_video ─────────────────────────────────────────────────


def test_wan_camera_image_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_camera_image_to_video))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 832, "
        "height: 'int' = 480, length: 'int' = 81, batch_size: 'int' = 1, *, "
        "clip_vision_output: 'Any | None' = None, start_image: 'Any | None' = None, "
        "camera_conditions: 'Any | None' = None) -> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_wan_camera_image_to_video_returns_tuple_and_passes_camera_conditions(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps())
    positive = [["p", {}]]
    negative = [["n", {}]]
    cam_cond = object()
    out_pos, out_neg, latent = wan_camera_image_to_video(
        positive=positive, negative=negative, vae=_FakeVaeVace(), camera_conditions=cam_cond
    )
    assert "samples" in latent
    assert out_pos[0][1]["camera_conditions"] is cam_cond
    assert out_neg[0][1]["camera_conditions"] is cam_cond


# ── wan_phantom_subject_to_video ──────────────────────────────────────────────


def test_wan_phantom_subject_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_phantom_subject_to_video))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 832, "
        "height: 'int' = 480, length: 'int' = 81, batch_size: 'int' = 1, *, "
        "images: 'Any | None' = None) -> 'tuple[Any, Any, Any, dict[str, Any]]'"
    )


def test_wan_phantom_subject_to_video_returns_four_tuple(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps())
    positive = [["p", {}]]
    negative = [["n", {}]]
    result = wan_phantom_subject_to_video(positive=positive, negative=negative, vae=_FakeVaeVace())
    assert len(result) == 4
    assert "samples" in result[3]


# ── wan_track_to_video ────────────────────────────────────────────────────────


def test_wan_track_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_track_to_video))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', tracks: 'str', "
        "width: 'int' = 832, height: 'int' = 480, length: 'int' = 81, "
        "batch_size: 'int' = 1, *, temperature: 'float' = 220.0, topk: 'int' = 2, "
        "start_image: 'Any | None' = None, "
        "clip_vision_output: 'Any | None' = None) -> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_wan_track_to_video_falls_back_to_image_to_video_when_tracks_empty(
    monkeypatch: Any,
) -> None:
    """With empty/invalid tracks JSON, wan_track_to_video delegates to wan_image_to_video."""
    calls: list[dict[str, Any]] = []

    def _fake_i2v(**kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
        calls.append(kwargs)
        return kwargs["positive"], kwargs["negative"], {"samples": object()}

    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps())
    monkeypatch.setattr(conditioning_module, "wan_image_to_video", _fake_i2v)
    positive = [["p", {}]]
    negative = [["n", {}]]
    wan_track_to_video(positive=positive, negative=negative, vae=_FakeVaeVace(), tracks="")
    assert len(calls) == 1


# ── wan_sound_image_to_video ──────────────────────────────────────────────────


def test_wan_sound_image_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_sound_image_to_video))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 832, "
        "height: 'int' = 480, length: 'int' = 77, batch_size: 'int' = 1, *, "
        "audio_encoder_output: 'Any | None' = None, ref_image: 'Any | None' = None, "
        "control_video: 'Any | None' = None, "
        "ref_motion: 'Any | None' = None) -> 'tuple[Any, Any, dict[str, Any]]'"
    )


# ── wan_sound_image_to_video_extend ──────────────────────────────────────────


def test_wan_sound_image_to_video_extend_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_sound_image_to_video_extend))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', length: 'int', "
        "video_latent: 'dict[str, Any]', *, "
        "audio_encoder_output: 'Any | None' = None, ref_image: 'Any | None' = None, "
        "control_video: 'Any | None' = None) -> 'tuple[Any, Any, dict[str, Any]]'"
    )


# ── wan_humo_image_to_video ───────────────────────────────────────────────────


def test_wan_humo_image_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_humo_image_to_video))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 832, "
        "height: 'int' = 480, length: 'int' = 97, batch_size: 'int' = 1, *, "
        "audio_encoder_output: 'Any | None' = None, "
        "ref_image: 'Any | None' = None) -> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_wan_humo_image_to_video_returns_latent_and_sets_audio_embed(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps())
    positive = [["p", {}]]
    negative = [["n", {}]]
    out_pos, out_neg, latent = wan_humo_image_to_video(
        positive=positive, negative=negative, vae=_FakeVaeVace()
    )
    assert "samples" in latent
    assert "audio_embed" in out_pos[0][1]
    assert "audio_embed" in out_neg[0][1]


# ── wan_animate_to_video ──────────────────────────────────────────────────────


def test_wan_animate_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_animate_to_video))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 832, "
        "height: 'int' = 480, length: 'int' = 77, batch_size: 'int' = 1, "
        "continue_motion_max_frames: 'int' = 5, video_frame_offset: 'int' = 0, *, "
        "clip_vision_output: 'Any | None' = None, "
        "reference_image: 'Any | None' = None, "
        "face_video: 'Any | None' = None, pose_video: 'Any | None' = None, "
        "continue_motion: 'Any | None' = None, "
        "background_video: 'Any | None' = None, "
        "character_mask: 'Any | None' = None) "
        "-> 'tuple[Any, Any, dict[str, Any], int, int, int]'"
    )


def test_wan_animate_to_video_returns_six_tuple(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps())
    positive = [["p", {}]]
    negative = [["n", {}]]
    result = wan_animate_to_video(positive=positive, negative=negative, vae=_FakeVaeVace())
    assert len(result) == 6
    assert "samples" in result[2]


# ── wan_infinite_talk_to_video ────────────────────────────────────────────────


def test_wan_infinite_talk_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_infinite_talk_to_video))
        == "(model: 'Any', model_patch: 'Any', positive: 'Any', negative: 'Any', "
        "vae: '_VaeEncoder', width: 'int', height: 'int', length: 'int', "
        "audio_encoder_output_1: 'Any', *, mode: 'str' = 'single_speaker', "
        "start_image: 'Any | None' = None, "
        "clip_vision_output: 'Any | None' = None, "
        "audio_encoder_output_2: 'Any | None' = None, "
        "mask_1: 'Any | None' = None, mask_2: 'Any | None' = None, "
        "motion_frame_count: 'int' = 9, audio_scale: 'float' = 1.0, "
        "previous_frames: 'Any | None' = None) "
        "-> 'tuple[Any, Any, Any, dict[str, Any], int]'"
    )


# ── wan_scail_to_video ────────────────────────────────────────────────────────


def test_wan_scail_to_video_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan_scail_to_video))
        == "(positive: 'Any', negative: 'Any', vae: '_VaeEncoder', width: 'int' = 512, "
        "height: 'int' = 896, length: 'int' = 81, batch_size: 'int' = 1, "
        "pose_strength: 'float' = 1.0, pose_start: 'float' = 0.0, "
        "pose_end: 'float' = 1.0, *, "
        "clip_vision_output: 'Any | None' = None, "
        "reference_image: 'Any | None' = None, "
        "pose_video: 'Any | None' = None) -> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_wan_scail_to_video_returns_latent_dict(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps())
    positive = [["p", {}]]
    negative = [["n", {}]]
    _, _, latent = wan_scail_to_video(positive=positive, negative=negative, vae=_FakeVaeVace())
    assert "samples" in latent


# ── wan22_image_to_video_latent ───────────────────────────────────────────────


def test_wan22_image_to_video_latent_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(wan22_image_to_video_latent))
        == "(vae: '_VaeEncoder', width: 'int' = 1280, height: 'int' = 704, "
        "length: 'int' = 49, batch_size: 'int' = 1, *, "
        "start_image: 'Any | None' = None) -> 'dict[str, Any]'"
    )


def test_wan22_image_to_video_latent_returns_samples_without_start_image(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps())
    result = wan22_image_to_video_latent(vae=_FakeVaeVace())
    assert "samples" in result
    assert "noise_mask" not in result


def test_wan22_image_to_video_latent_returns_noise_mask_with_start_image(
    monkeypatch: Any,
) -> None:
    fake_torch = _FakeTorchVace()
    monkeypatch.setattr(conditioning_module, "_get_wan_vace_dependencies", _make_vace_deps(fake_torch))
    start_image = _FakeTensor((1, 704, 1280, 3))
    result = wan22_image_to_video_latent(vae=_FakeVaeVace(), start_image=start_image)
    assert "samples" in result
    assert "noise_mask" in result


# ── __all__ completeness ──────────────────────────────────────────────────────


def test_all_new_wan_functions_exported_in_dunder_all() -> None:
    import comfy_diffusion.conditioning as m

    new_fns = [
        "wan_fun_control_to_video",
        "wan22_fun_control_to_video",
        "wan_fun_inpaint_to_video",
        "wan_camera_embedding",
        "wan_camera_image_to_video",
        "wan_phantom_subject_to_video",
        "wan_track_to_video",
        "wan_sound_image_to_video",
        "wan_sound_image_to_video_extend",
        "wan_humo_image_to_video",
        "wan_animate_to_video",
        "wan_infinite_talk_to_video",
        "wan_scail_to_video",
        "wan22_image_to_video_latent",
    ]
    for fn in new_fns:
        assert fn in m.__all__, f"{fn!r} missing from conditioning.__all__"
