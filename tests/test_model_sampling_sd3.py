"""Tests for SD3 model sampling helper."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import comfy_diffusion.models as models_module
from comfy_diffusion.models import model_sampling_sd3


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


class _FakePatchedModel:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.patches: list[tuple[str, Any]] = []

    def add_object_patch(self, patch_name: str, patch_object: Any) -> None:
        self.patches.append((patch_name, patch_object))


class _FakeModel:
    def __init__(self, model_config: object) -> None:
        self.model = SimpleNamespace(model_config=model_config)
        self.clone_calls = 0

    def clone(self) -> _FakePatchedModel:
        self.clone_calls += 1
        return _FakePatchedModel(self.model)


def _install_fake_sd3_sampling_modules(
    monkeypatch: Any,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "init": [],
        "set_parameters": [],
    }

    comfy_model_sampling_module = ModuleType("comfy.model_sampling")

    class FakeModelSamplingDiscreteFlow:
        def __init__(self, model_config: object) -> None:
            calls["init"].append(model_config)

        def set_parameters(self, *, shift: float, multiplier: int) -> None:
            calls["set_parameters"].append(
                {
                    "shift": shift,
                    "multiplier": multiplier,
                }
            )

    class FakeConst:
        pass

    setattr(
        comfy_model_sampling_module,
        "ModelSamplingDiscreteFlow",
        FakeModelSamplingDiscreteFlow,
    )
    setattr(comfy_model_sampling_module, "CONST", FakeConst)

    comfy_module = ModuleType("comfy")
    setattr(comfy_module, "model_sampling", comfy_model_sampling_module)

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.model_sampling", comfy_model_sampling_module)
    return calls


def test_model_sampling_sd3_is_public_and_callable_from_models_module() -> None:
    assert callable(model_sampling_sd3)
    assert model_sampling_sd3.__name__ == "model_sampling_sd3"
    assert "model_sampling_sd3" in models_module.__all__


def test_model_sampling_sd3_signature_matches_contract() -> None:
    signature = inspect.signature(model_sampling_sd3)
    assert str(signature) == "(model: 'Any', shift: 'float') -> 'Any'"


def test_model_sampling_sd3_returns_patched_clone_and_applies_sd3_sampling(
    monkeypatch: Any,
) -> None:
    model_config = object()
    model = _FakeModel(model_config=model_config)
    calls = _install_fake_sd3_sampling_modules(monkeypatch)

    patched = model_sampling_sd3(model, shift=3.0)

    assert model.clone_calls == 1
    assert patched is not model
    assert isinstance(patched, _FakePatchedModel)
    assert len(patched.patches) == 1

    patch_name, patch_object = patched.patches[0]
    assert patch_name == "model_sampling"
    assert patch_object.__class__.__name__ == "ModelSamplingAdvanced"

    assert calls["init"] == [model_config]
    assert calls["set_parameters"] == [{"shift": 3.0, "multiplier": 1000}]


def test_model_sampling_sd3_shift_controls_continuous_schedule(monkeypatch: Any) -> None:
    model = _FakeModel(model_config=object())
    calls = _install_fake_sd3_sampling_modules(monkeypatch)

    model_sampling_sd3(model, shift=2.5)
    model_sampling_sd3(model, shift=4.25)

    assert calls["set_parameters"] == [
        {"shift": 2.5, "multiplier": 1000},
        {"shift": 4.25, "multiplier": 1000},
    ]
    assert calls["set_parameters"][0]["shift"] != calls["set_parameters"][1]["shift"]


def test_import_model_sampling_sd3_has_no_heavy_import_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.models import model_sampling_sd3\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'func_name': model_sampling_sd3.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'comfy_model_sampling_loaded': 'comfy.model_sampling' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["func_name"] == "model_sampling_sd3"
    assert payload["torch_loaded"] is False
    assert payload["comfy_model_sampling_loaded"] is False
    heavy = [
        module
        for module in payload["new_modules"]
        if module.startswith(("torch", "comfy.")) or module == "comfy"
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
