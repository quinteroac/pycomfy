"""Tests for US-001 — ModelManager.load_audio_encoder."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from comfy_diffusion.models import ModelManager


def _install_fake_audio_encoder_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state_dict: dict[str, Any],
    audio_encoder_object: Any,
    resolved_model_path: str,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "add_model_folder_path": [],
        "get_full_path_or_raise": [],
        "load_torch_file": [],
        "load_audio_encoder_from_sd": [],
    }

    folder_paths_module = ModuleType("folder_paths")

    def add_model_folder_path(
        folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        calls["add_model_folder_path"].append((folder_name, full_folder_path, is_default))

    def get_full_path_or_raise(folder_name: str, filename: str) -> str:
        calls["get_full_path_or_raise"].append((folder_name, filename))
        if folder_name != "audio_encoders":
            raise AssertionError(f"unexpected folder name: {folder_name}")
        return resolved_model_path

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)
    setattr(folder_paths_module, "get_full_path_or_raise", get_full_path_or_raise)

    comfy_utils_module = ModuleType("comfy.utils")

    def load_torch_file(path: str, safe_load: bool = False) -> dict[str, Any]:
        calls["load_torch_file"].append((path, safe_load))
        return state_dict

    setattr(comfy_utils_module, "load_torch_file", load_torch_file)

    audio_encoders_module = ModuleType("comfy.audio_encoders.audio_encoders")

    def load_audio_encoder_from_sd(sd: dict[str, Any], prefix: str = "") -> Any:
        calls["load_audio_encoder_from_sd"].append((sd, prefix))
        return audio_encoder_object

    setattr(audio_encoders_module, "load_audio_encoder_from_sd", load_audio_encoder_from_sd)

    comfy_module = ModuleType("comfy")
    comfy_audio_encoders_pkg = ModuleType("comfy.audio_encoders")

    setattr(comfy_module, "utils", comfy_utils_module)
    setattr(comfy_audio_encoders_pkg, "audio_encoders", audio_encoders_module)

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)
    monkeypatch.setitem(sys.modules, "comfy.audio_encoders", comfy_audio_encoders_pkg)
    monkeypatch.setitem(sys.modules, "comfy.audio_encoders.audio_encoders", audio_encoders_module)

    return calls


# AC01 — method signature and return value
def test_load_audio_encoder_calls_loader_and_returns_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    encoder_file = tmp_path / "weights" / "wav2vec2.safetensors"
    encoder_file.parent.mkdir(parents=True)
    encoder_file.write_text("stub audio encoder")

    fake_sd: dict[str, Any] = {"encoder.layer_norm.bias": object()}
    fake_encoder = object()

    calls = _install_fake_audio_encoder_modules(
        monkeypatch,
        state_dict=fake_sd,
        audio_encoder_object=fake_encoder,
        resolved_model_path="/unused/path.safetensors",
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_audio_encoder(encoder_file)

    assert result is fake_encoder
    assert calls["get_full_path_or_raise"] == []
    assert calls["load_torch_file"] == [(str(encoder_file.resolve()), True)]
    assert calls["load_audio_encoder_from_sd"][0][0] is fake_sd


# AC02 — audio_encoders registered in __init__
def test_load_audio_encoder_registers_audio_encoders_folder_on_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    calls = _install_fake_audio_encoder_modules(
        monkeypatch,
        state_dict={},
        audio_encoder_object=object(),
        resolved_model_path="/unused",
    )

    ModelManager(models_dir=models_dir)

    registered = [c for c in calls["add_model_folder_path"] if c[0] == "audio_encoders"]
    assert len(registered) == 1
    folder_name, folder_path, is_default = registered[0]
    assert folder_name == "audio_encoders"
    assert folder_path == str(models_dir / "audio_encoders")
    assert is_default is True


# AC03 — uses load_torch_file + load_audio_encoder_from_sd
def test_load_audio_encoder_resolves_relative_name_via_folder_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    expected_resolved = str((models_dir / "audio_encoders" / "wav2vec2.safetensors").resolve())
    fake_encoder = object()
    calls = _install_fake_audio_encoder_modules(
        monkeypatch,
        state_dict={"encoder.layer_norm.bias": object()},
        audio_encoder_object=fake_encoder,
        resolved_model_path=expected_resolved,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_audio_encoder("wav2vec2.safetensors")

    assert result is fake_encoder
    assert calls["get_full_path_or_raise"] == [("audio_encoders", "wav2vec2.safetensors")]
    assert calls["load_torch_file"] == [(expected_resolved, True)]
    assert len(calls["load_audio_encoder_from_sd"]) == 1


# AC04 — lazy import: no comfy.* at module level (import safety)
def test_load_audio_encoder_method_exists_without_comfy_imports(tmp_path: Path) -> None:
    assert callable(getattr(ModelManager, "load_audio_encoder", None))


# AC01 — FileNotFoundError on missing absolute path (before any comfy call)
def test_load_audio_encoder_raises_file_not_found_for_missing_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    calls = _install_fake_audio_encoder_modules(
        monkeypatch,
        state_dict={},
        audio_encoder_object=object(),
        resolved_model_path="/unused",
    )

    missing = tmp_path / "weights" / "missing.safetensors"

    with pytest.raises(FileNotFoundError, match="audio encoder file not found"):
        ModelManager(models_dir=models_dir).load_audio_encoder(str(missing))

    assert calls["load_torch_file"] == []
    assert calls["load_audio_encoder_from_sd"] == []


# AC03 — ValueError when state dict is unrecognised (load_audio_encoder_from_sd returns None)
def test_load_audio_encoder_raises_value_error_for_unrecognised_sd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    encoder_file = tmp_path / "weights" / "unknown.safetensors"
    encoder_file.parent.mkdir(parents=True)
    encoder_file.write_text("stub")

    _install_fake_audio_encoder_modules(
        monkeypatch,
        state_dict={},
        audio_encoder_object=None,  # simulate unrecognised sd
        resolved_model_path="/unused",
    )

    with pytest.raises(ValueError, match="unrecognised audio encoder"):
        ModelManager(models_dir=models_dir).load_audio_encoder(encoder_file)
