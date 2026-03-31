"""Tests for latent image helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import comfy_diffusion.latent as latent_module
import comfy_diffusion.sampling as sampling_module
from comfy_diffusion.conditioning import conditioning_combine
from comfy_diffusion.latent import (
    empty_latent_image,
    empty_wan_latent_video,
    inpaint_model_conditioning,
    latent_composite,
    latent_composite_masked,
    latent_concat,
    latent_crop,
    latent_cut_to_batch,
    latent_from_batch,
    latent_upscale,
    latent_upscale_by,
    ltxv_empty_latent_video,
    ltxv_latent_upsample,
    repeat_latent_batch,
    replace_video_latent_frames,
    set_latent_noise_mask,
    trim_video_latent,
)
from comfy_diffusion.sampling import sample


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


def test_latent_module_exports_empty_latent_image() -> None:
    assert latent_module.__all__ == [
        "empty_latent_image",
        "empty_wan_latent_video",
        "ltxv_empty_latent_video",
        "latent_from_batch",
        "latent_cut_to_batch",
        "repeat_latent_batch",
        "latent_concat",
        "replace_video_latent_frames",
        "latent_upscale",
        "latent_upscale_by",
        "latent_crop",
        "latent_composite",
        "latent_composite_masked",
        "set_latent_noise_mask",
        "inpaint_model_conditioning",
        "ltxv_latent_upsample",
        "trim_video_latent",
    ]


def test_empty_latent_image_signature_matches_contract() -> None:
    signature = inspect.signature(empty_latent_image)
    assert str(signature) == (
        "(width: 'int', height: 'int', batch_size: 'int' = 1) -> 'dict[str, Any]'"
    )


def test_latent_upscale_signature_matches_contract() -> None:
    signature = inspect.signature(latent_upscale)
    assert str(signature) == (
        "(latent: 'dict[str, Any]', method: 'str', width: 'int', "
        "height: 'int') -> 'dict[str, Any]'"
    )


def test_latent_upscale_by_signature_matches_contract() -> None:
    signature = inspect.signature(latent_upscale_by)
    assert str(signature) == (
        "(latent: 'dict[str, Any]', method: 'str', scale_by: 'float') -> 'dict[str, Any]'"
    )


def test_latent_crop_signature_matches_contract() -> None:
    signature = inspect.signature(latent_crop)
    assert str(signature) == (
        "(latent: 'dict[str, Any]', x: 'int', y: 'int', width: 'int', "
        "height: 'int') -> 'dict[str, Any]'"
    )


def test_latent_from_batch_signature_matches_contract() -> None:
    signature = inspect.signature(latent_from_batch)
    assert str(signature) == (
        "(latent: 'dict[str, Any]', batch_index: 'int', length: 'int' = 1) "
        "-> 'dict[str, Any]'"
    )


def test_repeat_latent_batch_signature_matches_contract() -> None:
    signature = inspect.signature(repeat_latent_batch)
    assert str(signature) == "(latent: 'dict[str, Any]', amount: 'int') -> 'dict[str, Any]'"


def test_latent_concat_signature_matches_contract() -> None:
    signature = inspect.signature(latent_concat)
    assert str(signature) == "(*latents: 'dict[str, Any]', dim: 'str' = 't') -> 'dict[str, Any]'"


def test_latent_cut_to_batch_signature_matches_contract() -> None:
    signature = inspect.signature(latent_cut_to_batch)
    assert str(signature) == (
        "(latent: 'dict[str, Any]', start: 'int', length: 'int') -> 'dict[str, Any]'"
    )


def test_replace_video_latent_frames_signature_matches_contract() -> None:
    signature = inspect.signature(replace_video_latent_frames)
    assert str(signature) == (
        "(latent: 'dict[str, Any]', replacement: 'dict[str, Any]', "
        "start_frame: 'int') -> 'dict[str, Any]'"
    )


def test_latent_composite_signature_matches_contract() -> None:
    signature = inspect.signature(latent_composite)
    assert str(signature) == (
        "(destination: 'dict[str, Any]', source: 'dict[str, Any]', x: 'int', "
        "y: 'int') -> 'dict[str, Any]'"
    )


def test_latent_composite_masked_signature_matches_contract() -> None:
    signature = inspect.signature(latent_composite_masked)
    assert str(signature) == (
        "(destination: 'dict[str, Any]', source: 'dict[str, Any]', mask: 'Any', "
        "x: 'int' = 0, y: 'int' = 0) -> 'dict[str, Any]'"
    )


def test_set_latent_noise_mask_signature_matches_contract() -> None:
    signature = inspect.signature(set_latent_noise_mask)
    assert str(signature) == "(latent: 'dict[str, Any]', mask: 'Any') -> 'dict[str, Any]'"


def test_inpaint_model_conditioning_signature_matches_contract() -> None:
    signature = inspect.signature(inpaint_model_conditioning)
    assert str(signature) == (
        "(model: 'Any', latent: 'dict[str, Any]', vae: 'Any', positive: 'Any', "
        "negative: 'Any') -> 'tuple[Any, Any, Any]'"
    )


def test_empty_latent_image_returns_latent_dict_compatible_with_sample(
    monkeypatch: Any,
) -> None:
    expected_samples = object()

    class FakeEmptyLatentImage:
        def generate(self, width: int, height: int, batch_size: int = 1) -> tuple[dict[str, Any]]:
            assert width == 512
            assert height == 640
            assert batch_size == 2
            return ({"samples": expected_samples, "downscale_ratio_spacial": 8},)

    def fake_common_ksampler(
        model: Any,
        seed: int,
        steps: Any,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent: Any,
        denoise: float = 1.0,
    ) -> tuple[dict[str, Any]]:
        return (latent,)

    monkeypatch.setattr(
        latent_module,
        "_get_empty_latent_image_type",
        lambda: FakeEmptyLatentImage,
    )
    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: fake_common_ksampler)

    latent = empty_latent_image(width=512, height=640, batch_size=2)
    denoised = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent=latent,
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=123,
    )

    assert isinstance(latent, dict)
    assert latent["samples"] is expected_samples
    assert denoised is latent


def test_empty_latent_image_divides_dimensions_by_8_and_uses_default_batch_size(
    monkeypatch: Any,
) -> None:
    generate_calls: list[tuple[int, int, int]] = []

    class FakeTensor:
        def __init__(self, shape: tuple[int, int, int, int]) -> None:
            self.shape = shape

    class FakeEmptyLatentImage:
        def generate(self, width: int, height: int, batch_size: int = 1) -> tuple[dict[str, Any]]:
            generate_calls.append((width, height, batch_size))
            return (
                {
                    "samples": FakeTensor((batch_size, 4, height // 8, width // 8)),
                    "downscale_ratio_spacial": 8,
                },
            )

    monkeypatch.setattr(
        latent_module,
        "_get_empty_latent_image_type",
        lambda: FakeEmptyLatentImage,
    )

    latent = empty_latent_image(width=1024, height=768)

    assert generate_calls == [(1024, 768, 1)]
    assert latent["samples"].shape == (1, 4, 96, 128)


def test_empty_latent_image_is_importable_from_comfy_diffusion_latent() -> None:
    result = _run_python(
        "from comfy_diffusion.latent import empty_latent_image; "
        "assert empty_latent_image.__name__ == 'empty_latent_image'; "
        "print('ok')"
    )

    assert result.stdout.strip() == "ok"


def test_empty_latent_image_import_contract() -> None:
    result = _run_python(
        "import json; "
        "from comfy_diffusion.latent import empty_latent_image\n"
        "payload = {'func_name': empty_latent_image.__name__}\n"
        "print(json.dumps(payload))"
    )
    payload = json.loads(result.stdout)
    assert payload["func_name"] == "empty_latent_image"


def test_ltxv_empty_latent_video_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_empty_latent_video)
    assert str(signature) == (
        "(width: 'int', height: 'int', length: 'int' = 97, batch_size: 'int' = 1, fps: 'int' = 24)"
        " -> 'dict[str, Any]'"
    )


def test_ltxv_empty_latent_video_returns_correct_shape(monkeypatch: Any) -> None:
    import types

    class FakeTensor:
        def __init__(self, shape: Any) -> None:
            self.shape = tuple(shape)

    fake_torch = types.SimpleNamespace(zeros=lambda s, device=None: FakeTensor(s))
    fake_comfy_mm = types.SimpleNamespace(intermediate_device=lambda: "cpu")
    fake_comfy = types.SimpleNamespace(model_management=fake_comfy_mm)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_comfy_mm)

    result = ltxv_empty_latent_video(width=768, height=512, length=97, batch_size=1)

    assert isinstance(result, dict)
    assert "samples" in result
    # ((97-1)//8)+1=13; 512//32=16; 768//32=24
    assert result["samples"].shape == (1, 128, 13, 16, 24)


def test_ltxv_empty_latent_video_default_length_and_batch_size(monkeypatch: Any) -> None:
    import types

    class FakeTensor:
        def __init__(self, shape: Any) -> None:
            self.shape = tuple(shape)

    fake_torch = types.SimpleNamespace(zeros=lambda s, device=None: FakeTensor(s))
    fake_comfy_mm = types.SimpleNamespace(intermediate_device=lambda: "cpu")
    fake_comfy = types.SimpleNamespace(model_management=fake_comfy_mm)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_comfy_mm)

    result = ltxv_empty_latent_video(width=512, height=512)

    # length=97 default → ((97-1)//8)+1 = 13; 512//32 = 16
    assert result["samples"].shape == (1, 128, 13, 16, 16)


def test_ltxv_empty_latent_video_batch_size(monkeypatch: Any) -> None:
    import types

    class FakeTensor:
        def __init__(self, shape: Any) -> None:
            self.shape = tuple(shape)

    fake_torch = types.SimpleNamespace(zeros=lambda s, device=None: FakeTensor(s))
    fake_comfy_mm = types.SimpleNamespace(intermediate_device=lambda: "cpu")
    fake_comfy = types.SimpleNamespace(model_management=fake_comfy_mm)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_comfy_mm)

    result = ltxv_empty_latent_video(width=256, height=256, length=25, batch_size=3)

    # ((25-1)//8)+1 = 4; 256//32 = 8
    assert result["samples"].shape == (3, 128, 4, 8, 8)


def test_ltxv_empty_latent_video_lazy_imports() -> None:
    result = _run_python(
        "import sys\n"
        "import comfy_diffusion.latent as m\n"
        "assert 'torch' not in sys.modules, 'torch imported at module level'\n"
        "assert 'comfy' not in sys.modules, 'comfy imported at module level'\n"
        "print('ok')"
    )
    assert result.stdout.strip() == "ok"


def test_ltxv_empty_latent_video_in_all() -> None:
    import comfy_diffusion.latent as m

    assert "ltxv_empty_latent_video" in m.__all__


def test_empty_wan_latent_video_signature_matches_contract() -> None:
    signature = inspect.signature(empty_wan_latent_video)
    assert str(signature) == (
        "(width: 'int', height: 'int', length: 'int' = 33, batch_size: 'int' = 1)"
        " -> 'dict[str, Any]'"
    )


def test_empty_wan_latent_video_returns_correct_shape(monkeypatch: Any) -> None:
    import types

    class FakeTensor:
        def __init__(self, shape: Any) -> None:
            self.shape = tuple(shape)

    fake_torch = types.SimpleNamespace(zeros=lambda s, device=None: FakeTensor(s))
    fake_comfy_mm = types.SimpleNamespace(intermediate_device=lambda: "cpu")
    fake_comfy = types.SimpleNamespace(model_management=fake_comfy_mm)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_comfy_mm)

    result = empty_wan_latent_video(width=832, height=480, length=33, batch_size=1)

    assert isinstance(result, dict)
    assert "samples" in result
    # ((33-1)//4)+1 = 9; 480//8 = 60; 832//8 = 104
    assert result["samples"].shape == (1, 16, 9, 60, 104)


def test_empty_wan_latent_video_default_length_and_batch_size(monkeypatch: Any) -> None:
    import types

    class FakeTensor:
        def __init__(self, shape: Any) -> None:
            self.shape = tuple(shape)

    fake_torch = types.SimpleNamespace(zeros=lambda s, device=None: FakeTensor(s))
    fake_comfy_mm = types.SimpleNamespace(intermediate_device=lambda: "cpu")
    fake_comfy = types.SimpleNamespace(model_management=fake_comfy_mm)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_comfy_mm)

    result = empty_wan_latent_video(width=512, height=512)

    # length=33 default → ((33-1)//4)+1 = 9; 512//8 = 64
    assert result["samples"].shape == (1, 16, 9, 64, 64)


def test_empty_wan_latent_video_batch_size(monkeypatch: Any) -> None:
    import types

    class FakeTensor:
        def __init__(self, shape: Any) -> None:
            self.shape = tuple(shape)

    fake_torch = types.SimpleNamespace(zeros=lambda s, device=None: FakeTensor(s))
    fake_comfy_mm = types.SimpleNamespace(intermediate_device=lambda: "cpu")
    fake_comfy = types.SimpleNamespace(model_management=fake_comfy_mm)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_comfy_mm)

    result = empty_wan_latent_video(width=256, height=256, length=17, batch_size=2)

    # ((17-1)//4)+1 = 5; 256//8 = 32
    assert result["samples"].shape == (2, 16, 5, 32, 32)


def test_empty_wan_latent_video_lazy_imports() -> None:
    result = _run_python(
        "import sys\n"
        "import comfy_diffusion.latent as m\n"
        "assert 'torch' not in sys.modules, 'torch imported at module level'\n"
        "assert 'comfy' not in sys.modules, 'comfy imported at module level'\n"
        "print('ok')"
    )
    assert result.stdout.strip() == "ok"


def test_empty_wan_latent_video_in_all() -> None:
    import comfy_diffusion.latent as m

    assert "empty_wan_latent_video" in m.__all__


def test_latent_upscale_supports_all_comfyui_methods_and_returns_sample_compatible_latent(
    monkeypatch: Any,
) -> None:
    methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    latent = {"samples": object(), "batch_index": [0]}

    class FakeLatentUpscale:
        calls: list[tuple[dict[str, Any], str, int, int, str]] = []

        def upscale(
            self,
            samples: dict[str, Any],
            upscale_method: str,
            width: int,
            height: int,
            crop: str,
        ) -> tuple[dict[str, Any]]:
            self.calls.append((samples, upscale_method, width, height, crop))
            return (samples,)

    def fake_common_ksampler(
        model: Any,
        seed: int,
        steps: Any,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent: Any,
        denoise: float = 1.0,
    ) -> tuple[dict[str, Any]]:
        return (latent,)

    monkeypatch.setattr(latent_module, "_get_latent_upscale_type", lambda: FakeLatentUpscale)
    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: fake_common_ksampler)

    for method in methods:
        upscaled = latent_upscale(latent=latent, method=method, width=1024, height=768)
        denoised = sample(
            model=object(),
            positive=object(),
            negative=object(),
            latent=upscaled,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            seed=123,
        )

        assert upscaled is latent
        assert denoised is latent

    assert FakeLatentUpscale.calls == [
        (latent, "nearest-exact", 1024, 768, "disabled"),
        (latent, "bilinear", 1024, 768, "disabled"),
        (latent, "area", 1024, 768, "disabled"),
        (latent, "bicubic", 1024, 768, "disabled"),
        (latent, "bislerp", 1024, 768, "disabled"),
    ]


def test_latent_upscale_by_returns_scaled_latent_and_is_sample_compatible(
    monkeypatch: Any,
) -> None:
    latent = {"samples": object(), "noise_mask": object()}

    class FakeLatentUpscaleBy:
        calls: list[tuple[dict[str, Any], str, float]] = []

        def upscale(
            self,
            samples: dict[str, Any],
            upscale_method: str,
            scale_by: float,
        ) -> tuple[dict[str, Any]]:
            self.calls.append((samples, upscale_method, scale_by))
            return (samples,)

    def fake_common_ksampler(
        model: Any,
        seed: int,
        steps: Any,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent: Any,
        denoise: float = 1.0,
    ) -> tuple[dict[str, Any]]:
        return (latent,)

    monkeypatch.setattr(latent_module, "_get_latent_upscale_by_type", lambda: FakeLatentUpscaleBy)
    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: fake_common_ksampler)

    upscaled = latent_upscale_by(latent=latent, method="nearest-exact", scale_by=1.75)
    denoised = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent=upscaled,
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=42,
    )

    assert upscaled is latent
    assert denoised is latent
    assert FakeLatentUpscaleBy.calls == [(latent, "nearest-exact", 1.75)]


def test_latent_upscale_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        latent_upscale(latent={"samples": object()}, method="lanczos", width=512, height=512)


def test_latent_upscale_by_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        latent_upscale_by(latent={"samples": object()}, method="lanczos", scale_by=2.0)


def test_latent_crop_returns_cropped_latent_dict_and_uses_pixel_space_inputs(
    monkeypatch: Any,
) -> None:
    latent = {"samples": object(), "batch_index": [0]}
    expected = {"samples": object(), "batch_index": [0]}
    calls: list[tuple[dict[str, Any], int, int, int, int]] = []

    class FakeLatentCrop:
        def crop(
            self,
            samples: dict[str, Any],
            width: int,
            height: int,
            x: int,
            y: int,
        ) -> tuple[dict[str, Any]]:
            calls.append((samples, width, height, x, y))
            return (expected,)

    monkeypatch.setattr(latent_module, "_get_latent_crop_type", lambda: FakeLatentCrop)

    output = latent_crop(latent=latent, x=24, y=40, width=256, height=320)

    assert output is expected
    assert calls == [(latent, 256, 320, 24, 40)]


def test_latent_from_batch_extracts_contiguous_slice_and_is_sample_compatible(
    monkeypatch: Any,
) -> None:
    latent = {"samples": ["f0", "f1", "f2", "f3"], "batch_index": [0, 1, 2, 3]}

    class FakeLatentFromBatch:
        calls: list[tuple[dict[str, Any], int, int]] = []

        def frombatch(
            self,
            samples: dict[str, Any],
            batch_index: int,
            length: int,
        ) -> tuple[dict[str, Any]]:
            self.calls.append((samples, batch_index, length))
            return (
                {
                    "samples": samples["samples"][batch_index : batch_index + length],
                    "batch_index": samples["batch_index"][batch_index : batch_index + length],
                },
            )

    def fake_common_ksampler(
        model: Any,
        seed: int,
        steps: Any,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent: Any,
        denoise: float = 1.0,
    ) -> tuple[dict[str, Any]]:
        return (latent,)

    monkeypatch.setattr(latent_module, "_get_latent_from_batch_type", lambda: FakeLatentFromBatch)
    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: fake_common_ksampler)

    sliced_latent = latent_from_batch(latent=latent, batch_index=1, length=2)
    denoised = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent=sliced_latent,
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=123,
    )

    assert sliced_latent["samples"] == ["f1", "f2"]
    assert sliced_latent["batch_index"] == [1, 2]
    assert denoised is sliced_latent
    assert FakeLatentFromBatch.calls == [(latent, 1, 2)]


def test_repeat_latent_batch_repeats_batch_and_is_sample_compatible(monkeypatch: Any) -> None:
    latent = {"samples": ["f0", "f1"], "batch_index": [0, 1]}

    class FakeRepeatLatentBatch:
        calls: list[tuple[dict[str, Any], int]] = []

        def repeat(self, samples: dict[str, Any], amount: int) -> tuple[dict[str, Any]]:
            self.calls.append((samples, amount))
            return (
                {
                    "samples": samples["samples"] * amount,
                    "batch_index": [0, 1, 2, 3, 4, 5],
                },
            )

    def fake_common_ksampler(
        model: Any,
        seed: int,
        steps: Any,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent: Any,
        denoise: float = 1.0,
    ) -> tuple[dict[str, Any]]:
        return (latent,)

    monkeypatch.setattr(
        latent_module,
        "_get_repeat_latent_batch_type",
        lambda: FakeRepeatLatentBatch,
    )
    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: fake_common_ksampler)

    repeated_latent = repeat_latent_batch(latent=latent, amount=3)
    denoised = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent=repeated_latent,
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=42,
    )

    assert repeated_latent["samples"] == ["f0", "f1", "f0", "f1", "f0", "f1"]
    assert repeated_latent["batch_index"] == [0, 1, 2, 3, 4, 5]
    assert denoised is repeated_latent
    assert FakeRepeatLatentBatch.calls == [(latent, 3)]


@pytest.mark.parametrize("dim", ["x", "-x", "y", "-y", "t", "-t"])
def test_latent_concat_supports_all_dimensions(dim: str, monkeypatch: Any) -> None:
    latent_a = {"samples": "a"}
    latent_b = {"samples": "b"}
    calls: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    class FakeLatentConcat:
        @classmethod
        def execute(
            cls, samples1: dict[str, Any], samples2: dict[str, Any], dim: str
        ) -> tuple[dict[str, Any]]:
            calls.append((samples1, samples2, dim))
            return ({"samples": f"{samples1['samples']}|{samples2['samples']}|{dim}"},)

    monkeypatch.setattr(latent_module, "_get_latent_concat_type", lambda: FakeLatentConcat)

    output = latent_concat(latent_a, latent_b, dim=dim)

    assert output["samples"] == f"a|b|{dim}"
    assert calls == [(latent_a, latent_b, dim)]


def test_latent_concat_accepts_variadic_latents_and_is_sample_compatible(monkeypatch: Any) -> None:
    latent_a = {"samples": "a", "batch_index": [0]}
    latent_b = {"samples": "b", "batch_index": [1]}
    latent_c = {"samples": "c", "batch_index": [2]}

    class FakeLatentConcat:
        @classmethod
        def execute(
            cls, samples1: dict[str, Any], samples2: dict[str, Any], dim: str
        ) -> tuple[dict[str, Any]]:
            return (
                {
                    "samples": f"{samples1['samples']}+{samples2['samples']}@{dim}",
                    "batch_index": [0],
                },
            )

    def fake_common_ksampler(
        model: Any,
        seed: int,
        steps: Any,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent: Any,
        denoise: float = 1.0,
    ) -> tuple[dict[str, Any]]:
        return (latent,)

    monkeypatch.setattr(latent_module, "_get_latent_concat_type", lambda: FakeLatentConcat)
    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: fake_common_ksampler)

    concatenated = latent_concat(latent_a, latent_b, latent_c, dim="t")
    denoised = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent=concatenated,
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=99,
    )

    assert concatenated["samples"] == "a+b@t+c@t"
    assert denoised is concatenated


def test_latent_concat_rejects_invalid_dim() -> None:
    with pytest.raises(ValueError, match="dim must be one of"):
        latent_concat({"samples": "a"}, {"samples": "b"}, dim="z")


def test_latent_concat_requires_at_least_two_latents() -> None:
    with pytest.raises(ValueError, match="at least two latents are required"):
        latent_concat({"samples": "a"})


def test_latent_cut_to_batch_extracts_segment_and_is_sample_compatible(monkeypatch: Any) -> None:
    latent = {"samples": ["f0", "f1", "f2", "f3"], "batch_index": [0, 1, 2, 3]}

    class FakeLatentFromBatch:
        calls: list[tuple[dict[str, Any], int, int]] = []

        def frombatch(
            self,
            samples: dict[str, Any],
            batch_index: int,
            length: int,
        ) -> tuple[dict[str, Any]]:
            self.calls.append((samples, batch_index, length))
            return (
                {
                    "samples": samples["samples"][batch_index : batch_index + length],
                    "batch_index": samples["batch_index"][batch_index : batch_index + length],
                },
            )

    def fake_common_ksampler(
        model: Any,
        seed: int,
        steps: Any,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent: Any,
        denoise: float = 1.0,
    ) -> tuple[dict[str, Any]]:
        return (latent,)

    monkeypatch.setattr(latent_module, "_get_latent_from_batch_type", lambda: FakeLatentFromBatch)
    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: fake_common_ksampler)

    sliced_latent = latent_cut_to_batch(latent=latent, start=1, length=2)
    denoised = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent=sliced_latent,
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=1337,
    )

    assert sliced_latent["samples"] == ["f1", "f2"]
    assert sliced_latent["batch_index"] == [1, 2]
    assert denoised is sliced_latent
    assert FakeLatentFromBatch.calls == [(latent, 1, 2)]


def test_replace_video_latent_frames_replaces_at_start_frame_and_is_sample_compatible(
    monkeypatch: Any,
) -> None:
    latent = {"samples": ["d0", "d1", "d2", "d3"], "batch_index": [0, 1, 2, 3]}
    replacement = {"samples": ["r0", "r1"], "batch_index": [10, 11]}
    calls: list[tuple[dict[str, Any], dict[str, Any], int]] = []

    class FakeReplaceVideoLatentFrames:
        @classmethod
        def execute(
            cls, destination: dict[str, Any], source: dict[str, Any], index: int
        ) -> tuple[dict[str, Any]]:
            calls.append((destination, source, index))
            output = dict(destination)
            merged = list(destination["samples"])
            merged[index : index + len(source["samples"])] = source["samples"]
            output["samples"] = merged
            return (output,)

    def fake_common_ksampler(
        model: Any,
        seed: int,
        steps: Any,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent: Any,
        denoise: float = 1.0,
    ) -> tuple[dict[str, Any]]:
        return (latent,)

    monkeypatch.setattr(
        latent_module,
        "_get_replace_video_latent_frames_type",
        lambda: FakeReplaceVideoLatentFrames,
    )
    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: fake_common_ksampler)

    replaced = replace_video_latent_frames(
        latent=latent,
        replacement=replacement,
        start_frame=1,
    )
    denoised = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent=replaced,
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=5,
    )

    assert replaced["samples"] == ["d0", "r0", "r1", "d3"]
    assert denoised is replaced
    assert calls == [(latent, replacement, 1)]


def test_latent_composite_returns_new_latent_with_source_positioned_using_pixel_coordinates(
    monkeypatch: Any,
) -> None:
    destination = {"samples": object()}
    source = {"samples": object()}
    expected = {"samples": object()}
    calls: list[tuple[dict[str, Any], dict[str, Any], int, int, int]] = []

    class FakeLatentComposite:
        def composite(
            self,
            samples_to: dict[str, Any],
            samples_from: dict[str, Any],
            x: int,
            y: int,
            feather: int,
        ) -> tuple[dict[str, Any]]:
            calls.append((samples_to, samples_from, x, y, feather))
            assert x // 8 == 3
            assert y // 8 == 5
            return (expected,)

    monkeypatch.setattr(latent_module, "_get_latent_composite_type", lambda: FakeLatentComposite)

    output = latent_composite(destination=destination, source=source, x=24, y=40)

    assert output is expected
    assert calls == [(destination, source, 24, 40, 0)]


def test_latent_composite_masked_returns_blended_latent_and_forwards_mask(
    monkeypatch: Any,
) -> None:
    destination = {"samples": object()}
    source = {"samples": object()}
    mask = object()
    expected = {"samples": object()}
    calls: list[tuple[dict[str, Any], dict[str, Any], int, int, bool, Any]] = []

    class FakeLatentCompositeMasked:
        def composite(
            self,
            destination: dict[str, Any],
            source: dict[str, Any],
            x: int,
            y: int,
            resize_source: bool,
            mask: Any = None,
        ) -> tuple[dict[str, Any]]:
            calls.append((destination, source, x, y, resize_source, mask))
            return (expected,)

    monkeypatch.setattr(
        latent_module,
        "_get_latent_composite_masked_type",
        lambda: FakeLatentCompositeMasked,
    )

    output = latent_composite_masked(destination=destination, source=source, mask=mask, x=24, y=40)

    assert output is expected
    assert calls == [(destination, source, 24, 40, False, mask)]


def test_latent_composite_masked_uses_default_coordinates(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeLatentCompositeMasked:
        def composite(
            self,
            destination: dict[str, Any],
            source: dict[str, Any],
            x: int,
            y: int,
            resize_source: bool,
            mask: Any = None,
        ) -> tuple[dict[str, Any]]:
            calls.append((x, y))
            return (destination,)

    monkeypatch.setattr(
        latent_module,
        "_get_latent_composite_masked_type",
        lambda: FakeLatentCompositeMasked,
    )

    output = latent_composite_masked(
        destination={"samples": object()},
        source={"samples": object()},
        mask=object(),
    )

    assert output["samples"] is not None
    assert calls == [(0, 0)]


def test_set_latent_noise_mask_returns_sample_compatible_latent_dict(
    monkeypatch: Any,
) -> None:
    class FakeTensor:
        pass

    latent = {"samples": object(), "batch_index": [0]}
    mask = FakeTensor()

    def fake_common_ksampler(
        model: Any,
        seed: int,
        steps: Any,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent: Any,
        denoise: float = 1.0,
    ) -> tuple[dict[str, Any]]:
        return (latent,)

    monkeypatch.setattr(latent_module, "_get_torch_tensor_type", lambda: FakeTensor)
    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: fake_common_ksampler)

    masked_latent = set_latent_noise_mask(latent=latent, mask=mask)
    denoised = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent=masked_latent,
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=123,
    )

    assert masked_latent["noise_mask"] is mask
    assert denoised is masked_latent
    assert "noise_mask" not in latent


def test_set_latent_noise_mask_rejects_non_tensor_mask(monkeypatch: Any) -> None:
    class FakeTensor:
        pass

    monkeypatch.setattr(latent_module, "_get_torch_tensor_type", lambda: FakeTensor)

    with pytest.raises(TypeError, match="mask must be a torch.Tensor"):
        set_latent_noise_mask(latent={"samples": object()}, mask=object())


def test_inpaint_model_conditioning_patches_model_and_conditionings(monkeypatch: Any) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    class FakeNodeHelpers:
        @staticmethod
        def conditioning_set_values(conditioning: Any, values: dict[str, Any]) -> Any:
            calls.append((conditioning, values))
            return [conditioning, values]

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def clone(self) -> FakeModel:
            return FakeModel(f"{self.name}-patched")

    monkeypatch.setattr(latent_module, "_get_node_helpers_module", lambda: FakeNodeHelpers)

    latent = {"samples": "latent-samples", "noise_mask": "latent-mask"}
    model = FakeModel("base-model")
    positive = [["pos", {}]]
    negative = [["neg", {}]]

    patched_model, patched_positive, patched_negative = inpaint_model_conditioning(
        model=model,
        latent=latent,
        vae=object(),
        positive=positive,
        negative=negative,
    )

    assert isinstance(patched_model, FakeModel)
    assert patched_model.name == "base-model-patched"
    assert patched_positive == [
        positive,
        {"concat_latent_image": "latent-samples", "concat_mask": "latent-mask"},
    ]
    assert patched_negative == [
        negative,
        {"concat_latent_image": "latent-samples", "concat_mask": "latent-mask"},
    ]
    assert calls == [
        (positive, {"concat_latent_image": "latent-samples", "concat_mask": "latent-mask"}),
        (negative, {"concat_latent_image": "latent-samples", "concat_mask": "latent-mask"}),
    ]


def test_inpaint_model_conditioning_outputs_are_compatible_with_sampling_and_conditioning(
    monkeypatch: Any,
) -> None:
    class FakeNodeHelpers:
        @staticmethod
        def conditioning_set_values(conditioning: Any, values: dict[str, Any]) -> list[Any]:
            output: list[Any] = []
            for token, metadata in conditioning:
                merged = dict(metadata)
                merged.update(values)
                output.append([token, merged])
            return output

    def fake_common_ksampler(
        model: Any,
        seed: int,
        steps: Any,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent: Any,
        denoise: float = 1.0,
    ) -> tuple[dict[str, Any]]:
        return (latent,)

    monkeypatch.setattr(latent_module, "_get_node_helpers_module", lambda: FakeNodeHelpers)
    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: fake_common_ksampler)

    class FakeModel:
        def clone(self) -> FakeModel:
            return self

    latent = {"samples": object(), "noise_mask": object()}
    positive = [["p", {"base": True}]]
    negative = [["n", {"base": False}]]

    patched_model, patched_positive, patched_negative = inpaint_model_conditioning(
        model=FakeModel(),
        latent=latent,
        vae=object(),
        positive=positive,
        negative=negative,
    )
    combined = conditioning_combine(patched_positive, [["p2", {"extra": True}]])
    denoised = sample(
        model=patched_model,
        positive=combined,
        negative=patched_negative,
        latent=latent,
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=777,
    )

    assert denoised is latent
    assert patched_positive[0][1]["concat_mask"] is latent["noise_mask"]
    assert patched_negative[0][1]["concat_latent_image"] is latent["samples"]
    assert len(combined) == 2


# ---------------------------------------------------------------------------
# ltxv_latent_upsample tests
# ---------------------------------------------------------------------------


def test_ltxv_latent_upsample_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_latent_upsample)
    assert str(signature) == (
        "(samples: 'dict[str, Any]', upscale_model: 'Any', vae: 'Any') -> 'dict[str, Any]'"
    )


def test_ltxv_latent_upsample_in_all() -> None:
    assert "ltxv_latent_upsample" in latent_module.__all__


def test_ltxv_latent_upsample_delegates_to_node(monkeypatch: Any) -> None:
    expected_output: dict[str, Any] = {"samples": object()}
    upscale_model = object()
    vae = object()
    input_samples: dict[str, Any] = {"samples": object()}

    class FakeLTXVLatentUpsampler:
        def upsample_latent(
            self,
            samples: dict[str, Any],
            upscale_model: Any,
            vae: Any,
        ) -> tuple[dict[str, Any]]:
            assert samples is input_samples
            return (expected_output,)

    monkeypatch.setattr(
        latent_module,
        "_load_latent_upscale_model",
        lambda: FakeLTXVLatentUpsampler,
    )

    result = ltxv_latent_upsample(
        samples=input_samples, upscale_model=upscale_model, vae=vae
    )
    assert result is expected_output


def test_ltxv_latent_upsample_removes_noise_mask(monkeypatch: Any) -> None:
    upsampled = {"samples": object()}
    noise_mask = object()
    input_samples: dict[str, Any] = {"samples": object(), "noise_mask": noise_mask}

    class FakeLTXVLatentUpsampler:
        def upsample_latent(
            self,
            samples: dict[str, Any],
            upscale_model: Any,
            vae: Any,
        ) -> tuple[dict[str, Any]]:
            result = dict(samples)
            result.pop("noise_mask", None)
            result["samples"] = upsampled["samples"]
            return (result,)

    monkeypatch.setattr(
        latent_module,
        "_load_latent_upscale_model",
        lambda: FakeLTXVLatentUpsampler,
    )

    result = ltxv_latent_upsample(
        samples=input_samples, upscale_model=object(), vae=object()
    )
    assert "noise_mask" not in result


def test_ltxv_latent_upsample_lazy_imports() -> None:
    result = _run_python(
        "import sys\n"
        "import comfy_diffusion.latent as m\n"
        "assert 'torch' not in sys.modules, 'torch imported at module level'\n"
        "assert 'comfy' not in sys.modules, 'comfy imported at module level'\n"
        "print('ok')"
    )
    assert result.stdout.strip() == "ok"


def test_trim_video_latent_signature_matches_contract() -> None:
    assert (
        str(inspect.signature(trim_video_latent))
        == "(latent: 'dict[str, Any]', n_latent_frames: 'int') -> 'dict[str, Any]'"
    )


def test_trim_video_latent_trims_time_dimension() -> None:
    import torch

    samples = torch.zeros(1, 16, 10, 8, 8)
    latent = {"samples": samples}
    result = trim_video_latent(latent, 3)
    assert result["samples"].shape == (1, 16, 7, 8, 8)
    assert "noise_mask" not in result


def test_trim_video_latent_also_trims_noise_mask() -> None:
    import torch

    samples = torch.zeros(1, 16, 10, 8, 8)
    mask = torch.ones(1, 1, 10, 8, 8)
    latent = {"samples": samples, "noise_mask": mask}
    result = trim_video_latent(latent, 2)
    assert result["samples"].shape == (1, 16, 8, 8, 8)
    assert result["noise_mask"].shape == (1, 1, 8, 8, 8)


def test_trim_video_latent_zero_trim_returns_full_latent() -> None:
    import torch

    samples = torch.zeros(1, 16, 5, 8, 8)
    latent = {"samples": samples}
    result = trim_video_latent(latent, 0)
    assert result["samples"].shape == (1, 16, 5, 8, 8)
