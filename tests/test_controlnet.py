"""Tests for ControlNet loading helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import comfy_diffusion.controlnet as controlnet_module
from comfy_diffusion.controlnet import apply_controlnet, load_controlnet, load_diff_controlnet


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


def _install_fake_comfy_controlnet(
    monkeypatch: Any,
    loader: Callable[[str], object | None],
) -> None:
    import types

    comfy_module: Any = types.ModuleType("comfy")
    comfy_controlnet_module: Any = types.ModuleType("comfy.controlnet")
    comfy_controlnet_module.load_controlnet = loader
    comfy_module.controlnet = comfy_controlnet_module

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.controlnet", comfy_controlnet_module)


def test_controlnet_module_exports_only_load_controlnet() -> None:
    assert controlnet_module.__all__ == [
        "load_controlnet",
        "load_diff_controlnet",
        "apply_controlnet",
    ]


def test_load_controlnet_signature_matches_contract() -> None:
    signature = inspect.signature(load_controlnet)
    assert str(signature) == "(path: 'str | Path') -> 'Any'"


def test_load_diff_controlnet_signature_matches_contract() -> None:
    signature = inspect.signature(load_diff_controlnet)
    assert str(signature) == "(model: 'Any', path: 'str | Path') -> 'Any'"


def test_apply_controlnet_signature_matches_contract() -> None:
    signature = inspect.signature(apply_controlnet)
    assert (
        str(signature)
        == "(positive: 'Any', negative: 'Any', control_net: 'Any', image: 'Any', "
        "strength: 'float' = 1.0, start_percent: 'float' = 0.0, "
        "end_percent: 'float' = 1.0, vae: 'Any' = None) -> 'tuple[Any, Any]'"
    )


def test_apply_controlnet_applies_to_positive_and_negative_with_defaults() -> None:
    moved_hint = object()
    previous_control = object()
    positive = [["p-token", {"tag": "pos"}], ["p-token-2", {"control": previous_control}]]
    negative = [["n-token", {"tag": "neg"}], ["n-token-2", {"control": previous_control}]]

    class FakeImage:
        def movedim(self, src: int, dest: int) -> object:
            assert (src, dest) == (-1, 1)
            return moved_hint

    class FakeControlNetInstance:
        def __init__(self) -> None:
            self.previous: Any = None

        def set_previous_controlnet(self, previous: Any) -> FakeControlNetInstance:
            self.previous = previous
            return self

    class FakeControlNet:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.instances: list[FakeControlNetInstance] = []

        def copy(self) -> FakeControlNet:
            return self

        def set_cond_hint(
            self,
            control_hint: Any,
            strength: float,
            bounds: tuple[float, float],
            vae: Any = None,
            extra_concat: Any = None,
        ) -> FakeControlNetInstance:
            self.calls.append(
                {
                    "control_hint": control_hint,
                    "strength": strength,
                    "bounds": bounds,
                    "vae": vae,
                    "extra_concat": extra_concat,
                }
            )
            instance = FakeControlNetInstance()
            self.instances.append(instance)
            return instance

    fake_control_net = FakeControlNet()
    out_positive, out_negative = apply_controlnet(
        positive=positive,
        negative=negative,
        control_net=fake_control_net,
        image=FakeImage(),
    )

    assert len(fake_control_net.calls) == 2
    for call in fake_control_net.calls:
        assert call["control_hint"] is moved_hint
        assert call["strength"] == 1.0
        assert call["bounds"] == (0.0, 1.0)
        assert call["vae"] is None
        assert call["extra_concat"] == []

    assert out_positive is not positive
    assert out_negative is not negative
    assert out_positive[0][1]["control_apply_to_uncond"] is False
    assert out_negative[0][1]["control_apply_to_uncond"] is False

    same_previous_pos = out_positive[1][1]["control"]
    same_previous_neg = out_negative[1][1]["control"]
    assert same_previous_pos is same_previous_neg
    assert same_previous_pos.previous is previous_control


def test_apply_controlnet_passes_custom_step_range_and_vae() -> None:
    vae = object()
    input_positive = [["p-token", {}]]
    input_negative = [["n-token", {}]]

    class FakeImage:
        def movedim(self, src: int, dest: int) -> object:
            assert (src, dest) == (-1, 1)
            return "hint"

    class FakeControlNetInstance:
        def set_previous_controlnet(self, previous: Any) -> FakeControlNetInstance:
            assert previous is None
            return self

    class FakeControlNet:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def copy(self) -> FakeControlNet:
            return self

        def set_cond_hint(
            self,
            control_hint: Any,
            strength: float,
            bounds: tuple[float, float],
            vae: Any = None,
            extra_concat: Any = None,
        ) -> FakeControlNetInstance:
            self.calls.append(
                {
                    "control_hint": control_hint,
                    "strength": strength,
                    "bounds": bounds,
                    "vae": vae,
                    "extra_concat": extra_concat,
                }
            )
            return FakeControlNetInstance()

    control = FakeControlNet()
    out_positive, out_negative = apply_controlnet(
        positive=input_positive,
        negative=input_negative,
        control_net=control,
        image=FakeImage(),
        strength=0.75,
        start_percent=0.2,
        end_percent=0.9,
        vae=vae,
    )

    assert len(control.calls) == 1
    call = control.calls[0]
    assert call["strength"] == 0.75
    assert call["bounds"] == (0.2, 0.9)
    assert call["vae"] is vae
    assert out_positive[0][1]["control_apply_to_uncond"] is False
    assert out_negative[0][1]["control_apply_to_uncond"] is False


def test_apply_controlnet_with_zero_strength_returns_original_conditioning() -> None:
    positive = [["p-token", {}]]
    negative = [["n-token", {}]]

    class FakeImage:
        def movedim(self, src: int, dest: int) -> object:
            raise AssertionError("movedim should not be called when strength is zero")

    class FakeControlNet:
        def copy(self) -> FakeControlNet:
            raise AssertionError("copy should not be called when strength is zero")

    out_positive, out_negative = apply_controlnet(
        positive=positive,
        negative=negative,
        control_net=FakeControlNet(),
        image=FakeImage(),
        strength=0.0,
    )

    assert out_positive is positive
    assert out_negative is negative


def test_load_controlnet_loads_controlnet_object_from_string_path(
    monkeypatch: Any, tmp_path: Path
) -> None:
    controlnet_file = tmp_path / "controlnet.safetensors"
    controlnet_file.write_text("stub controlnet")

    loaded_controlnet = object()
    calls: dict[str, Any] = {}

    def fake_load_controlnet(path: str) -> object:
        calls["path"] = path
        return loaded_controlnet

    _install_fake_comfy_controlnet(monkeypatch, fake_load_controlnet)

    result = load_controlnet(str(controlnet_file))

    assert result is loaded_controlnet
    assert calls["path"] == str(controlnet_file.resolve())


def test_load_controlnet_accepts_path_object(
    monkeypatch: Any, tmp_path: Path
) -> None:
    controlnet_file = tmp_path / "controlnet.safetensors"
    controlnet_file.write_text("stub controlnet")

    loaded_controlnet = object()
    calls: dict[str, Any] = {}

    def fake_load_controlnet(path: str) -> object:
        calls["path"] = path
        return loaded_controlnet

    _install_fake_comfy_controlnet(monkeypatch, fake_load_controlnet)

    result = load_controlnet(controlnet_file)

    assert result is loaded_controlnet
    assert calls["path"] == str(controlnet_file.resolve())


def test_load_controlnet_raises_file_not_found_error_for_missing_file(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: dict[str, int] = {"count": 0}

    def fake_load_controlnet(path: str) -> object:
        calls["count"] += 1
        return {"path": path}

    _install_fake_comfy_controlnet(monkeypatch, fake_load_controlnet)

    missing = tmp_path / "missing_controlnet.safetensors"

    with pytest.raises(FileNotFoundError, match="controlnet file not found") as exc_info:
        load_controlnet(missing)

    assert str(exc_info.value).endswith(str(missing.resolve()))
    assert calls["count"] == 0


def test_load_controlnet_raises_runtime_error_for_invalid_checkpoint(
    monkeypatch: Any, tmp_path: Path
) -> None:
    controlnet_file = tmp_path / "invalid_controlnet.safetensors"
    controlnet_file.write_text("invalid")

    def fake_load_controlnet(path: str) -> None:
        return None

    _install_fake_comfy_controlnet(monkeypatch, fake_load_controlnet)

    with pytest.raises(RuntimeError, match="controlnet file is invalid"):
        load_controlnet(controlnet_file)


def test_load_diff_controlnet_loads_controlnet_for_specific_base_model(
    monkeypatch: Any, tmp_path: Path
) -> None:
    controlnet_file = tmp_path / "controlnet.safetensors"
    controlnet_file.write_text("stub controlnet")

    loaded_controlnet = object()
    model = object()
    calls: dict[str, Any] = {}

    def fake_load_controlnet(path: str, model: Any | None = None) -> object:
        calls["path"] = path
        calls["model"] = model
        return loaded_controlnet

    _install_fake_comfy_controlnet(monkeypatch, fake_load_controlnet)

    result = load_diff_controlnet(model, str(controlnet_file))

    assert result is loaded_controlnet
    assert calls["path"] == str(controlnet_file.resolve())
    assert calls["model"] is model


def test_load_diff_controlnet_accepts_path_object(
    monkeypatch: Any, tmp_path: Path
) -> None:
    controlnet_file = tmp_path / "controlnet.safetensors"
    controlnet_file.write_text("stub controlnet")

    loaded_controlnet = object()

    def fake_load_controlnet(path: str, model: Any | None = None) -> object:
        assert model is not None
        assert path == str(controlnet_file.resolve())
        return loaded_controlnet

    _install_fake_comfy_controlnet(monkeypatch, fake_load_controlnet)

    result = load_diff_controlnet(object(), controlnet_file)

    assert result is loaded_controlnet


def test_load_diff_controlnet_raises_file_not_found_error_for_missing_file(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: dict[str, int] = {"count": 0}

    def fake_load_controlnet(path: str, model: Any | None = None) -> object:
        calls["count"] += 1
        return {"path": path, "model": model}

    _install_fake_comfy_controlnet(monkeypatch, fake_load_controlnet)

    missing = tmp_path / "missing_controlnet.safetensors"

    with pytest.raises(FileNotFoundError, match="controlnet file not found") as exc_info:
        load_diff_controlnet(model=object(), path=missing)

    assert str(exc_info.value).endswith(str(missing.resolve()))
    assert calls["count"] == 0


def test_load_diff_controlnet_raises_runtime_error_for_invalid_checkpoint(
    monkeypatch: Any, tmp_path: Path
) -> None:
    controlnet_file = tmp_path / "invalid_controlnet.safetensors"
    controlnet_file.write_text("invalid")

    def fake_load_controlnet(path: str, model: Any | None = None) -> None:
        return None

    _install_fake_comfy_controlnet(monkeypatch, fake_load_controlnet)

    with pytest.raises(RuntimeError, match="controlnet file is invalid"):
        load_diff_controlnet(model=object(), path=controlnet_file)


def test_import_comfy_diffusion_controlnet_has_no_heavy_import_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.controlnet import (\n"
        "  apply_controlnet,\n"
        "  load_controlnet,\n"
        "  load_diff_controlnet,\n"
        ")\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'apply_func_name': apply_controlnet.__name__,\n"
        "  'func_name': load_controlnet.__name__,\n"
        "  'diff_func_name': load_diff_controlnet.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'comfy_controlnet_loaded': 'comfy.controlnet' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["apply_func_name"] == "apply_controlnet"
    assert payload["func_name"] == "load_controlnet"
    assert payload["diff_func_name"] == "load_diff_controlnet"
    assert payload["torch_loaded"] is False
    assert payload["comfy_controlnet_loaded"] is False
    heavy = [
        module
        for module in payload["new_modules"]
        if module.startswith(("torch", "comfy.")) or module == "comfy"
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
