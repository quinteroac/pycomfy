"""Tests for US-001 standalone LLM loading behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from comfy_diffusion.models import ModelManager


def _install_fake_llm_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embeddings_paths: list[str],
    llm_object: Any,
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

    def load_clip(*, ckpt_paths: list[str], embedding_directory: list[str]) -> Any:
        calls["load_clip"].append((ckpt_paths, embedding_directory))
        return llm_object

    setattr(comfy_sd_module, "load_clip", load_clip)

    comfy_module = ModuleType("comfy")
    setattr(comfy_module, "sd", comfy_sd_module)

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)
    return calls


def test_load_llm_calls_comfy_loader_with_resolved_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    checkpoints_dir = models_dir / "checkpoints"
    embeddings_dir = models_dir / "embeddings"
    checkpoints_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    llm_file = tmp_path / "llm" / "gemma_2_2b.safetensors"
    llm_file.parent.mkdir(parents=True)
    llm_file.write_text("stub llm")

    embeddings_paths = [str(embeddings_dir)]
    expected_llm = object()
    calls = _install_fake_llm_loader_modules(
        monkeypatch,
        embeddings_paths=embeddings_paths,
        llm_object=expected_llm,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_llm(llm_file.parent / "." / llm_file.name)

    assert result is expected_llm
    assert calls["load_clip"] == [([str(llm_file.resolve())], embeddings_paths)]
    assert calls["get_folder_paths"] == ["embeddings"]
    assert calls["add_model_folder_path"] == [
        ("checkpoints", str(checkpoints_dir), True),
        ("embeddings", str(embeddings_dir), True),
        ("diffusion_models", str(models_dir / "unet"), True),
        ("diffusion_models", str(models_dir / "diffusion_models"), False),
        ("text_encoders", str(models_dir / "text_encoders"), True),
        ("text_encoders", str(models_dir / "clip"), False),
        ("vae", str(models_dir / "vae"), True),
        ("llm", str(models_dir / "llm"), True),
    ]


def test_load_llm_resolves_relative_name_under_models_dir_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    embeddings_dir = models_dir / "embeddings"
    embeddings_dir.mkdir(parents=True)
    llm_file = models_dir / "llm" / "gemma_3_4b.safetensors"
    llm_file.parent.mkdir(parents=True)
    llm_file.write_text("stub llm")

    calls = _install_fake_llm_loader_modules(
        monkeypatch,
        embeddings_paths=[str(embeddings_dir)],
        llm_object=object(),
    )

    ModelManager(models_dir=models_dir).load_llm("gemma_3_4b.safetensors")

    assert calls["load_clip"] == [
        (
            [str(llm_file.resolve())],
            [str(embeddings_dir)],
        )
    ]
    assert calls["get_folder_paths"] == ["embeddings"]


def test_load_llm_raises_file_not_found_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    calls = _install_fake_llm_loader_modules(
        monkeypatch,
        embeddings_paths=[str(models_dir / "embeddings")],
        llm_object=object(),
    )

    missing_llm_path = tmp_path / "llm" / "missing_llm.safetensors"
    expected_path = str(missing_llm_path.resolve())

    with pytest.raises(FileNotFoundError, match="llm file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_llm(str(missing_llm_path))

    assert expected_path in str(exc_info.value)
    assert calls["get_folder_paths"] == []
    assert calls["load_clip"] == []
