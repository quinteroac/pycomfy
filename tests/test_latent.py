"""Tests for latent image helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import comfy_diffusion.latent as latent_module
import comfy_diffusion.sampling as sampling_module
from comfy_diffusion.latent import empty_latent_image
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
    assert latent_module.__all__ == ["empty_latent_image"]


def test_empty_latent_image_signature_matches_contract() -> None:
    signature = inspect.signature(empty_latent_image)
    assert str(signature) == (
        "(width: 'int', height: 'int', batch_size: 'int' = 1) -> 'dict[str, Any]'"
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
