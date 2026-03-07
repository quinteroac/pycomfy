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
from pycomfy.sampling import basic_guider, cfg_guider, sample, sample_advanced


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


def test_sampling_public_api_exports_all_entrypoints() -> None:
    assert sample.__name__ == "sample"
    assert sample_advanced.__name__ == "sample_advanced"
    assert basic_guider.__name__ == "basic_guider"
    assert cfg_guider.__name__ == "cfg_guider"
    assert sampling_module.__all__ == [
        "sample",
        "sample_advanced",
        "basic_guider",
        "cfg_guider",
    ]


def test_sample_signature_matches_contract() -> None:
    signature = inspect.signature(sample)

    assert str(signature) == (
        "(model: 'Any', positive: 'Any', negative: 'Any', latent: 'Any', "
        "steps: 'Any', cfg: 'Any', sampler_name: 'str', scheduler: 'str', "
        "seed: 'int', *, denoise: 'float' = 1.0) -> 'Any'"
    )


def test_sample_advanced_signature_matches_contract() -> None:
    signature = inspect.signature(sample_advanced)

    assert str(signature) == (
        "(model: 'Any', positive: 'Any', negative: 'Any', latent: 'Any', "
        "steps: 'Any', cfg: 'Any', sampler_name: 'str', scheduler: 'str', "
        "noise_seed: 'int', *, add_noise: 'bool' = True, "
        "return_with_leftover_noise: 'bool' = False, denoise: 'float' = 1.0, "
        "start_at_step: 'int' = 0, end_at_step: 'int' = 10000) -> 'Any'"
    )


def test_basic_guider_signature_matches_contract() -> None:
    signature = inspect.signature(basic_guider)

    assert str(signature) == "(model: 'Any', conditioning: 'Any') -> 'Any'"


def test_cfg_guider_signature_matches_contract() -> None:
    signature = inspect.signature(cfg_guider)

    assert str(signature) == (
        "(model: 'Any', positive: 'Any', negative: 'Any', cfg: 'Any') -> 'Any'"
    )


def test_basic_guider_wraps_basic_guider_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    conditioning = object()

    class FakeBasicGuider:
        def __init__(self, received_model: Any) -> None:
            self.received_model = received_model
            self.received_conditioning: Any = None

        def set_conds(self, received_conditioning: Any) -> None:
            self.received_conditioning = received_conditioning

    monkeypatch.setattr(sampling_module, "_get_basic_guider_type", lambda: FakeBasicGuider)

    guider = basic_guider(model, conditioning)

    assert isinstance(guider, FakeBasicGuider)
    assert guider.received_model is model
    assert guider.received_conditioning is conditioning


def test_cfg_guider_wraps_cfg_guider_type(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object()
    positive = object()
    negative = object()
    cfg = 6.5

    class FakeCFGGuider:
        def __init__(self, received_model: Any) -> None:
            self.received_model = received_model
            self.received_positive: Any = None
            self.received_negative: Any = None
            self.received_cfg: Any = None

        def set_conds(self, received_positive: Any, received_negative: Any) -> None:
            self.received_positive = received_positive
            self.received_negative = received_negative

        def set_cfg(self, received_cfg: Any) -> None:
            self.received_cfg = received_cfg

    monkeypatch.setattr(sampling_module, "_get_cfg_guider_type", lambda: FakeCFGGuider)

    guider = cfg_guider(model, positive, negative, cfg)

    assert isinstance(guider, FakeCFGGuider)
    assert guider.received_model is model
    assert guider.received_positive is positive
    assert guider.received_negative is negative
    assert guider.received_cfg == cfg


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


def test_sample_advanced_returns_raw_denoised_latent_without_transformation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_latent = {"samples": object(), "batch_index": [0]}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        return (raw_latent,)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    result = sample_advanced(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        noise_seed=123,
    )

    assert result is raw_latent


def test_sample_advanced_uses_common_ksampler_call_pattern(
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

    result = sample_advanced(
        model=model,
        positive=positive,
        negative=negative,
        latent=latent,
        steps=30,
        cfg=5.5,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        noise_seed=42,
        denoise=0.65,
        start_at_step=3,
        end_at_step=20,
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
    assert recorded["kwargs"] == {
        "denoise": 0.65,
        "disable_noise": False,
        "start_step": 3,
        "last_step": 20,
        "force_full_denoise": True,
    }


def test_sample_advanced_disables_noise_when_add_noise_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["kwargs"] = kwargs
        return (object(),)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    sample_advanced(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=10,
        cfg=4.0,
        sampler_name="euler",
        scheduler="normal",
        noise_seed=999,
        add_noise=False,
    )

    assert recorded["kwargs"]["disable_noise"] is True


def test_sample_advanced_returns_with_leftover_noise_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["kwargs"] = kwargs
        return (object(),)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    sample_advanced(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=10,
        cfg=4.0,
        sampler_name="euler",
        scheduler="normal",
        noise_seed=999,
        return_with_leftover_noise=True,
    )

    assert recorded["kwargs"]["force_full_denoise"] is False


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


def test_uv_run_python_imports_sample_advanced_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from pycomfy.sampling import sample_advanced; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_basic_guider_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from pycomfy.sampling import basic_guider; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_cfg_guider_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from pycomfy.sampling import cfg_guider; print('ok')",
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
        "from pycomfy.sampling import basic_guider, cfg_guider, sample, sample_advanced\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'func_name': sample.__name__,\n"
        "  'advanced_name': sample_advanced.__name__,\n"
        "  'basic_name': basic_guider.__name__,\n"
        "  'cfg_name': cfg_guider.__name__,\n"
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
    assert payload["advanced_name"] == "sample_advanced"
    assert payload["basic_name"] == "basic_guider"
    assert payload["cfg_name"] == "cfg_guider"
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
