"""Tests for sampling helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import pycomfy.sampling as sampling_module
from pycomfy.sampling import sample


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


def test_sampling_public_api_exports_sample_only() -> None:
    assert sample.__name__ == "sample"
    assert sampling_module.__all__ == ["sample"]


def test_sample_signature_matches_contract() -> None:
    signature = inspect.signature(sample)

    assert str(signature) == (
        "(model: 'Any', positive: 'Any', negative: 'Any', latent: 'Any', "
        "steps: 'Any', cfg: 'Any', sampler_name: 'str', scheduler: 'str', "
        "seed: 'int', *, denoise: 'float' = 1.0) -> 'Any'"
    )


def test_sample_returns_raw_denoised_latent_without_transformation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_latent = {"samples": object(), "batch_index": [0]}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        return (raw_latent,)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    result = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=123,
    )

    assert result is raw_latent


def test_sample_uses_common_ksampler_call_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    positive = object()
    negative = object()
    latent = {"samples": object()}
    expected = object()
    recorded: dict[str, Any] = {}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return (expected,)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    result = sample(
        model=model,
        positive=positive,
        negative=negative,
        latent=latent,
        steps=30,
        cfg=5.5,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        seed=42,
        denoise=0.65,
    )

    assert result is expected
    assert recorded["args"] == (
        model,
        42,
        30,
        5.5,
        "dpmpp_2m",
        "karras",
        positive,
        negative,
        latent,
    )
    assert recorded["kwargs"] == {"denoise": 0.65}


def test_sample_passes_sampler_and_scheduler_strings_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}
    sampler_name = " Euler++ custom "
    scheduler = "Normal/alt schedule "

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["args"] = args
        return (object(),)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=10,
        cfg=4.0,
        sampler_name=sampler_name,
        scheduler=scheduler,
        seed=999,
    )

    assert recorded["args"][4] == sampler_name
    assert recorded["args"][5] == scheduler


def test_sample_passes_seed_to_common_ksampler_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}
    seed = 4_294_967_295

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["args"] = args
        return (object(),)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=20,
        cfg=6.5,
        sampler_name="euler",
        scheduler="normal",
        seed=seed,
    )

    assert recorded["args"][1] == seed


def test_uv_run_python_imports_sample_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from pycomfy.sampling import sample; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_import_pycomfy_sampling_has_no_additional_heavy_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import pycomfy\n"
        "baseline_path = list(sys.path)\n"
        "baseline_modules = set(sys.modules)\n"
        "from pycomfy.sampling import sample\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'func_name': sample.__name__,\n"
        "  'path_unchanged': baseline_path == list(sys.path),\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'nodes_loaded': 'nodes' in sys.modules,\n"
        "  'folder_paths_loaded': 'folder_paths' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["func_name"] == "sample"
    assert payload["path_unchanged"] is True
    assert payload["torch_loaded"] is False
    assert payload["nodes_loaded"] is False
    assert payload["folder_paths_loaded"] is False
    heavy = [
        module
        for module in payload["new_modules"]
        if module == "nodes"
        or module.startswith(("torch", "folder_paths", "comfy.", "numpy"))
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
