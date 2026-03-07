"""Tests for US-004 standalone CLIP loading behavior."""

from __future__ import annotations

import enum
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pycomfy.models import ModelManager


class _FakeCLIPType(enum.Enum):
    STABLE_DIFFUSION = "stable_diffusion"


def _install_fake_clip_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embeddings_paths: list[str],
    clip_object: Any,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "add_model_folder_path": [],
        "get_folder_paths": [],
        "load_clip": [],
    }

    folder_paths_module = ModuleType("folder_paths")

    def add_model_folder_path(
        folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        calls["add_model_folder_path"].append((folder_name, full_folder_path, is_default))

    def get_folder_paths(folder_name: str) -> list[str]:
        calls["get_folder_paths"].append(folder_name)
        if folder_name != "embeddings":
            raise AssertionError(f"unexpected folder name: {folder_name}")
        return embeddings_paths

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)
    setattr(folder_paths_module, "get_folder_paths", get_folder_paths)

    comfy_sd_module = ModuleType("comfy.sd")
    setattr(comfy_sd_module, "CLIPType", _FakeCLIPType)

    def load_clip(
        *,
        ckpt_paths: list[str],
        clip_type: Any,
        embedding_directory: list[str],
    ) -> Any:
        calls["load_clip"].append((ckpt_paths, clip_type, embedding_directory))
        return clip_object

    setattr(comfy_sd_module, "load_clip", load_clip)

    comfy_module = ModuleType("comfy")
    setattr(comfy_module, "sd", comfy_sd_module)

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)
    return calls


def test_load_clip_calls_comfy_loader_with_resolved_path_and_returns_raw_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    checkpoints_dir = models_dir / "checkpoints"
    embeddings_dir = models_dir / "embeddings"
    checkpoints_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    clip_file = tmp_path / "text_encoders" / "clip_l.safetensors"
    clip_file.parent.mkdir(parents=True)
    clip_file.write_text("stub clip")

    embeddings_paths = [str(embeddings_dir)]
    fake_clip = object()
    calls = _install_fake_clip_loader_modules(
        monkeypatch, embeddings_paths=embeddings_paths, clip_object=fake_clip
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_clip(clip_file.parent / "." / clip_file.name)

    assert result is fake_clip
    assert calls["load_clip"] == [
        ([str(clip_file.resolve())], _FakeCLIPType.STABLE_DIFFUSION, embeddings_paths)
    ]
    assert calls["get_folder_paths"] == ["embeddings"]
    assert calls["add_model_folder_path"] == [
        ("checkpoints", str(checkpoints_dir), True),
        ("embeddings", str(embeddings_dir), True),
        ("diffusion_models", str(models_dir / "unet"), True),
        ("diffusion_models", str(models_dir / "diffusion_models"), False),
        ("text_encoders", str(models_dir / "text_encoders"), True),
        ("text_encoders", str(models_dir / "clip"), False),
        ("vae", str(models_dir / "vae"), True),
    ]


def test_load_clip_raises_file_not_found_before_comfy_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    calls = _install_fake_clip_loader_modules(
        monkeypatch,
        embeddings_paths=[str(models_dir / "embeddings")],
        clip_object=object(),
    )

    missing_clip_path = tmp_path / "text_encoders" / "missing_clip.safetensors"
    expected_path = str(missing_clip_path.resolve())

    with pytest.raises(FileNotFoundError, match="clip file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_clip(str(missing_clip_path))

    assert expected_path in str(exc_info.value)
    assert calls["get_folder_paths"] == []
    assert calls["load_clip"] == []
