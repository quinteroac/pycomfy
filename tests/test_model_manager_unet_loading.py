"""Tests for US-005 standalone UNet loading behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from comfy_diffusion.models import ModelManager


def _install_fake_unet_loader_modules(
    monkeypatch: pytest.MonkeyPatch, *, unet_object: Any
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "add_model_folder_path": [],
        "load_diffusion_model": [],
        "load_unet": [],
    }

    folder_paths_module = ModuleType("folder_paths")

    def add_model_folder_path(
        folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        calls["add_model_folder_path"].append((folder_name, full_folder_path, is_default))

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)

    comfy_sd_module = ModuleType("comfy.sd")

    def load_diffusion_model(path: str) -> Any:
        calls["load_diffusion_model"].append(path)
        return unet_object

    def load_unet(path: str) -> Any:
        calls["load_unet"].append(path)
        raise AssertionError("deprecated comfy.sd.load_unet should not be used")

    setattr(comfy_sd_module, "load_diffusion_model", load_diffusion_model)
    setattr(comfy_sd_module, "load_unet", load_unet)

    comfy_module = ModuleType("comfy")
    setattr(comfy_module, "sd", comfy_sd_module)

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)
    return calls


def test_load_unet_calls_comfy_loader_with_resolved_path_and_returns_raw_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    checkpoints_dir = models_dir / "checkpoints"
    embeddings_dir = models_dir / "embeddings"
    checkpoints_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    unet_file = tmp_path / "diffusion_models" / "unet.safetensors"
    unet_file.parent.mkdir(parents=True)
    unet_file.write_text("stub unet")

    fake_unet = object()
    calls = _install_fake_unet_loader_modules(monkeypatch, unet_object=fake_unet)

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_unet(unet_file.parent / "." / unet_file.name)

    assert result is fake_unet
    assert calls["load_diffusion_model"] == [str(unet_file.resolve())]
    assert calls["load_unet"] == []
    assert calls["add_model_folder_path"] == [
        ("checkpoints", str(checkpoints_dir), True),
        ("embeddings", str(embeddings_dir), True),
        ("diffusion_models", str(models_dir / "unet"), True),
        ("diffusion_models", str(models_dir / "diffusion_models"), False),
        ("text_encoders", str(models_dir / "text_encoders"), True),
        ("text_encoders", str(models_dir / "clip"), False),
        ("vae", str(models_dir / "vae"), True),
        ("llm", str(models_dir / "llm"), True),
        ("upscale_models", str(models_dir / "upscale_models"), True),
        ("latent_upscale_models", str(models_dir / "upscale"), True),
    ]


def test_load_unet_raises_file_not_found_before_comfy_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    calls = _install_fake_unet_loader_modules(monkeypatch, unet_object=object())

    missing_unet_path = tmp_path / "diffusion_models" / "missing_unet.safetensors"
    expected_path = str(missing_unet_path.resolve())

    with pytest.raises(FileNotFoundError, match="unet file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_unet(str(missing_unet_path))

    assert expected_path in str(exc_info.value)
    assert calls["load_diffusion_model"] == []
    assert calls["load_unet"] == []

