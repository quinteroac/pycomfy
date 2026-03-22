"""Tests for CLIP vision loading behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from comfy_diffusion.models import ModelManager


def _install_fake_clip_vision_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    clip_vision_object: Any,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "add_model_folder_path": [],
        "load_clip_vision": [],
    }

    folder_paths_module = ModuleType("folder_paths")

    def add_model_folder_path(
        folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        calls["add_model_folder_path"].append((folder_name, full_folder_path, is_default))

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)

    clip_vision_module = ModuleType("comfy.clip_vision")

    def load(path: str) -> Any:
        calls["load_clip_vision"].append(path)
        return clip_vision_object

    setattr(clip_vision_module, "load", load)

    comfy_module = ModuleType("comfy")
    setattr(comfy_module, "clip_vision", clip_vision_module)

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.clip_vision", clip_vision_module)
    return calls


def test_load_clip_vision_calls_comfy_loader_with_resolved_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    checkpoints_dir = models_dir / "checkpoints"
    embeddings_dir = models_dir / "embeddings"
    checkpoints_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    clip_vision_file = tmp_path / "clip_vision" / "wan_clip_vision.safetensors"
    clip_vision_file.parent.mkdir(parents=True)
    clip_vision_file.write_text("stub clip vision")

    expected_clip_vision = object()
    calls = _install_fake_clip_vision_loader_modules(
        monkeypatch,
        clip_vision_object=expected_clip_vision,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_clip_vision(clip_vision_file.parent / "." / clip_vision_file.name)

    assert result is expected_clip_vision
    assert calls["load_clip_vision"] == [str(clip_vision_file.resolve())]
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


def test_load_clip_vision_resolves_relative_name_under_models_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)
    clip_vision_file = models_dir / "clip_vision" / "wan_clip_vision.safetensors"
    clip_vision_file.parent.mkdir(parents=True)
    clip_vision_file.write_text("stub clip vision")

    expected_clip_vision = object()
    calls = _install_fake_clip_vision_loader_modules(
        monkeypatch,
        clip_vision_object=expected_clip_vision,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_clip_vision("wan_clip_vision.safetensors")

    assert result is expected_clip_vision
    assert calls["load_clip_vision"] == [str(clip_vision_file.resolve())]


def test_load_clip_vision_raises_file_not_found_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    calls = _install_fake_clip_vision_loader_modules(
        monkeypatch,
        clip_vision_object=object(),
    )

    missing_clip_vision_path = tmp_path / "clip_vision" / "missing_clip_vision.safetensors"
    expected_path = str(missing_clip_vision_path.resolve())

    with pytest.raises(FileNotFoundError, match="clip vision file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_clip_vision(str(missing_clip_vision_path))

    assert expected_path in str(exc_info.value)
    assert calls["load_clip_vision"] == []
