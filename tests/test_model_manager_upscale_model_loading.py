"""Tests for US-001 ModelManager.load_upscale_model()."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from comfy_diffusion.models import ModelManager


class _FakeImageModelDescriptor:
    """Minimal stand-in for spandrel.ImageModelDescriptor."""

    def eval(self) -> "_FakeImageModelDescriptor":
        return self


class _FakeNonImageModel:
    """Stands in for a non-ImageModelDescriptor spandrel result."""

    def eval(self) -> "_FakeNonImageModel":
        return self


class _FakeModelLoader:
    def __init__(self, result: Any) -> None:
        self._result = result

    def load_from_state_dict(self, sd: dict[str, Any]) -> Any:
        return self._result


def _install_fake_upscale_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state_dict: dict[str, Any],
    loader_result: Any,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "add_model_folder_path": [],
        "load_torch_file": [],
        "state_dict_prefix_replace": [],
    }

    folder_paths_module = ModuleType("folder_paths")

    def add_model_folder_path(
        folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        calls["add_model_folder_path"].append((folder_name, full_folder_path, is_default))

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)

    comfy_utils_module = ModuleType("comfy.utils")

    def load_torch_file(path: str, *, safe_load: bool) -> dict[str, Any]:
        calls["load_torch_file"].append((path, safe_load))
        return state_dict

    def state_dict_prefix_replace(sd: dict[str, Any], replace: dict[str, str]) -> dict[str, Any]:
        calls["state_dict_prefix_replace"].append((sd, replace))
        return {k.replace(list(replace.keys())[0], list(replace.values())[0]): v for k, v in sd.items()}

    setattr(comfy_utils_module, "load_torch_file", load_torch_file)
    setattr(comfy_utils_module, "state_dict_prefix_replace", state_dict_prefix_replace)

    comfy_module = ModuleType("comfy")
    setattr(comfy_module, "utils", comfy_utils_module)

    fake_descriptor_class = _FakeImageModelDescriptor

    spandrel_module = ModuleType("spandrel")
    setattr(spandrel_module, "ImageModelDescriptor", fake_descriptor_class)
    setattr(spandrel_module, "ModelLoader", lambda: _FakeModelLoader(loader_result))

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)
    monkeypatch.setitem(sys.modules, "spandrel", spandrel_module)

    return calls


def _make_models_dir(tmp_path: Path) -> Path:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)
    return models_dir


# ---------------------------------------------------------------------------
# AC01 — method exists on ModelManager
# ---------------------------------------------------------------------------

def test_load_upscale_model_method_exists_on_model_manager() -> None:
    assert callable(getattr(ModelManager, "load_upscale_model", None))


# ---------------------------------------------------------------------------
# AC02 — absolute path to existing file is loaded
# ---------------------------------------------------------------------------

def test_load_upscale_model_absolute_existing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)
    model_file = tmp_path / "weights" / "realesrgan_x4.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("stub upscale model")

    expected_descriptor = _FakeImageModelDescriptor()
    calls = _install_fake_upscale_loader_modules(
        monkeypatch,
        state_dict={"layers.0.weight": object()},
        loader_result=expected_descriptor,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_upscale_model(model_file.parent / "." / model_file.name)

    assert result is expected_descriptor
    assert calls["load_torch_file"] == [(str(model_file.resolve()), True)]


# ---------------------------------------------------------------------------
# AC03 — relative filename resolved against models_dir/upscale_models
# ---------------------------------------------------------------------------

def test_load_upscale_model_relative_path_resolved_against_upscale_models_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)
    upscale_dir = models_dir / "upscale_models"
    upscale_dir.mkdir(parents=True)
    model_file = upscale_dir / "realesrgan_x4.safetensors"
    model_file.write_text("stub upscale model")

    expected_descriptor = _FakeImageModelDescriptor()
    calls = _install_fake_upscale_loader_modules(
        monkeypatch,
        state_dict={"layers.0.weight": object()},
        loader_result=expected_descriptor,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_upscale_model("realesrgan_x4.safetensors")

    assert result is expected_descriptor
    assert calls["load_torch_file"] == [(str(model_file.resolve()), True)]


# ---------------------------------------------------------------------------
# AC04 — absolute path that does not exist raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_load_upscale_model_absolute_missing_path_raises_file_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)
    missing = tmp_path / "weights" / "nonexistent.safetensors"

    calls = _install_fake_upscale_loader_modules(
        monkeypatch,
        state_dict={},
        loader_result=_FakeImageModelDescriptor(),
    )

    with pytest.raises(FileNotFoundError, match="upscale model file not found"):
        ModelManager(models_dir=models_dir).load_upscale_model(missing)

    assert calls["load_torch_file"] == []


# ---------------------------------------------------------------------------
# AC05 — uses comfy.utils.load_torch_file(..., safe_load=True)
# ---------------------------------------------------------------------------

def test_load_upscale_model_uses_load_torch_file_with_safe_load_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)
    model_file = tmp_path / "weights" / "model.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("stub")

    calls = _install_fake_upscale_loader_modules(
        monkeypatch,
        state_dict={"weight": object()},
        loader_result=_FakeImageModelDescriptor(),
    )

    ModelManager(models_dir=models_dir).load_upscale_model(model_file)

    assert len(calls["load_torch_file"]) == 1
    _path, safe_load = calls["load_torch_file"][0]
    assert safe_load is True


# ---------------------------------------------------------------------------
# AC06 — non-ImageModelDescriptor raises TypeError
# ---------------------------------------------------------------------------

def test_load_upscale_model_non_image_descriptor_raises_type_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)
    model_file = tmp_path / "weights" / "video_model.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("stub")

    calls = _install_fake_upscale_loader_modules(
        monkeypatch,
        state_dict={"weight": object()},
        loader_result=_FakeNonImageModel(),
    )

    with pytest.raises(TypeError, match="ImageModelDescriptor"):
        ModelManager(models_dir=models_dir).load_upscale_model(model_file)


# ---------------------------------------------------------------------------
# AC07 — lazy imports: no heavy modules at import time
# ---------------------------------------------------------------------------

def test_load_upscale_model_lazy_imports_no_heavy_modules_at_import_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure spandrel and comfy are not imported just by importing ModelManager
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("spandrel") or mod_name.startswith("comfy.utils"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    import importlib
    import comfy_diffusion.models as models_module
    importlib.reload(models_module)

    assert "spandrel" not in sys.modules
    assert "comfy.utils" not in sys.modules


# ---------------------------------------------------------------------------
# AC08 — returned object can be passed to image_upscale_with_model
# ---------------------------------------------------------------------------

def test_load_upscale_model_result_passable_to_image_upscale_with_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)
    model_file = tmp_path / "weights" / "model.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("stub")

    expected_descriptor = _FakeImageModelDescriptor()
    _install_fake_upscale_loader_modules(
        monkeypatch,
        state_dict={"weight": object()},
        loader_result=expected_descriptor,
    )

    manager = ModelManager(models_dir=models_dir)
    upscale_model = manager.load_upscale_model(model_file)

    # Verify the returned object is an ImageModelDescriptor as declared by spandrel
    import spandrel
    assert isinstance(upscale_model, spandrel.ImageModelDescriptor)


# ---------------------------------------------------------------------------
# AC09 — ModelManager is in module __all__ (method is accessible via class)
# ---------------------------------------------------------------------------

def test_model_manager_is_in_module_all() -> None:
    from comfy_diffusion import models as models_module

    assert "ModelManager" in models_module.__all__
    assert hasattr(ModelManager, "load_upscale_model")
