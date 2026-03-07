"""Tests for US-003 standalone VAE loading behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pycomfy.models import ModelManager


def _install_fake_vae_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state_dict: dict[str, Any],
    metadata: dict[str, Any] | None,
    vae_object: Any,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "add_model_folder_path": [],
        "load_torch_file": [],
        "VAE": [],
        "throw_exception_if_invalid": [],
    }

    folder_paths_module = ModuleType("folder_paths")

    def add_model_folder_path(
        folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        calls["add_model_folder_path"].append((folder_name, full_folder_path, is_default))

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)

    comfy_utils_module = ModuleType("comfy.utils")

    def load_torch_file(path: str, *, return_metadata: bool) -> tuple[dict[str, Any], Any]:
        calls["load_torch_file"].append((path, return_metadata))
        return state_dict, metadata

    setattr(comfy_utils_module, "load_torch_file", load_torch_file)

    comfy_sd_module = ModuleType("comfy.sd")

    class _FakeVaeWrapper:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped

        def throw_exception_if_invalid(self) -> None:
            calls["throw_exception_if_invalid"].append(True)

        def __eq__(self, other: object) -> bool:
            return other is self._wrapped or other is self

    _fake_vae_instance = _FakeVaeWrapper(vae_object)
    calls["vae_instance"] = [_fake_vae_instance]

    def VAE(*, sd: dict[str, Any], metadata: dict[str, Any] | None) -> Any:
        calls["VAE"].append((sd, metadata))
        return _fake_vae_instance

    setattr(comfy_sd_module, "VAE", VAE)

    comfy_module = ModuleType("comfy")
    setattr(comfy_module, "sd", comfy_sd_module)
    setattr(comfy_module, "utils", comfy_utils_module)

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)
    return calls


def test_load_vae_calls_comfy_loader_with_resolved_path_and_returns_raw_vae(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    checkpoints_dir = models_dir / "checkpoints"
    embeddings_dir = models_dir / "embeddings"
    checkpoints_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    vae_file = tmp_path / "weights" / "anime_vae.safetensors"
    vae_file.parent.mkdir(parents=True)
    vae_file.write_text("stub vae")

    fake_state_dict = {"decoder.weight": object()}
    fake_metadata = {"format": "pt"}
    fake_vae = object()
    calls = _install_fake_vae_loader_modules(
        monkeypatch,
        state_dict=fake_state_dict,
        metadata=fake_metadata,
        vae_object=fake_vae,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_vae(vae_file.parent / "." / vae_file.name)

    assert result is calls["vae_instance"][0]
    assert calls["load_torch_file"] == [(str(vae_file.resolve()), True)]
    assert calls["VAE"] == [(fake_state_dict, fake_metadata)]
    assert calls["throw_exception_if_invalid"] == [True]
    assert calls["add_model_folder_path"] == [
        ("checkpoints", str(checkpoints_dir), True),
        ("embeddings", str(embeddings_dir), True),
        ("diffusion_models", str(models_dir / "unet"), True),
        ("diffusion_models", str(models_dir / "diffusion_models"), False),
        ("text_encoders", str(models_dir / "text_encoders"), True),
        ("text_encoders", str(models_dir / "clip"), False),
        ("vae", str(models_dir / "vae"), True),
    ]


def test_load_vae_raises_file_not_found_before_comfy_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    calls = _install_fake_vae_loader_modules(
        monkeypatch,
        state_dict={},
        metadata=None,
        vae_object=object(),
    )

    missing_vae_path = tmp_path / "weights" / "missing_vae.safetensors"
    expected_path = str(missing_vae_path.resolve())

    with pytest.raises(FileNotFoundError, match="vae file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_vae(str(missing_vae_path))

    assert expected_path in str(exc_info.value)
    assert calls["load_torch_file"] == []
    assert calls["VAE"] == []
