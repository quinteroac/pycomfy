"""Tests for US-003 checkpoint loading behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import comfy_diffusion.models as models_module
from comfy_diffusion.models import CheckpointResult, ModelManager


def _install_fake_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolved_checkpoint_path: str,
    embeddings_paths: list[str],
    loader_result: tuple[Any, ...],
) -> dict[str, Any]:
    calls: dict[str, Any] = {
        "add_model_folder_path": [],
        "get_full_path_or_raise": [],
        "get_folder_paths": [],
        "load_checkpoint_guess_config": [],
    }

    folder_paths_module = ModuleType("folder_paths")

    def add_model_folder_path(
        folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        calls["add_model_folder_path"].append((folder_name, full_folder_path, is_default))

    def get_full_path_or_raise(folder_name: str, filename: str) -> str:
        calls["get_full_path_or_raise"].append((folder_name, filename))
        return resolved_checkpoint_path

    def get_folder_paths(folder_name: str) -> list[str]:
        calls["get_folder_paths"].append(folder_name)
        if folder_name != "embeddings":
            raise AssertionError(f"unexpected folder name: {folder_name}")
        return embeddings_paths

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)
    setattr(folder_paths_module, "get_full_path_or_raise", get_full_path_or_raise)
    setattr(folder_paths_module, "get_folder_paths", get_folder_paths)

    comfy_sd_module = ModuleType("comfy.sd")

    def load_checkpoint_guess_config(
        ckpt_path: str,
        *,
        output_vae: bool,
        output_clip: bool,
        embedding_directory: list[str],
    ) -> tuple[Any, ...]:
        calls["load_checkpoint_guess_config"].append(
            (ckpt_path, output_vae, output_clip, embedding_directory)
        )
        return loader_result

    setattr(comfy_sd_module, "load_checkpoint_guess_config", load_checkpoint_guess_config)

    comfy_module = ModuleType("comfy")
    setattr(comfy_module, "sd", comfy_sd_module)

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)
    return calls


def test_load_checkpoint_resolves_filename_and_returns_typed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    checkpoints_dir = models_dir / "checkpoints"
    embeddings_dir = models_dir / "embeddings"
    checkpoints_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    checkpoint_filename = "animagine-xl.safetensors"
    (checkpoints_dir / checkpoint_filename).write_text("stub checkpoint")
    resolved_path = str(checkpoints_dir / checkpoint_filename)
    embeddings_paths = [str(embeddings_dir)]

    model = object()
    clip = object()
    vae = object()
    calls = _install_fake_loader_modules(
        monkeypatch,
        resolved_checkpoint_path=resolved_path,
        embeddings_paths=embeddings_paths,
        loader_result=(model, clip, vae, object()),
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_checkpoint(checkpoint_filename)

    assert isinstance(result, CheckpointResult)
    assert result.model is model
    assert result.clip is clip
    assert result.vae is vae

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
    assert calls["get_full_path_or_raise"] == [("checkpoints", checkpoint_filename)]
    assert calls["get_folder_paths"] == ["embeddings"]
    assert calls["load_checkpoint_guess_config"] == [
        (resolved_path, True, True, embeddings_paths)
    ]


def test_load_checkpoint_allows_optional_clip_and_vae(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    model = object()
    (models_dir / "checkpoints" / "model.safetensors").write_text("stub checkpoint")
    _install_fake_loader_modules(
        monkeypatch,
        resolved_checkpoint_path="/resolved/model.safetensors",
        embeddings_paths=["/resolved/embeddings"],
        loader_result=(model, None, None),
    )

    result = ModelManager(models_dir=models_dir).load_checkpoint("model.safetensors")

    assert isinstance(result, CheckpointResult)
    assert result.model is model
    assert result.clip is None
    assert result.vae is None


def test_checkpoint_result_is_publicly_importable() -> None:
    assert CheckpointResult.__name__ == "CheckpointResult"
    assert "CheckpointResult" in models_module.__all__


def test_load_checkpoint_raises_file_not_found_before_comfy_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    calls = _install_fake_loader_modules(
        monkeypatch,
        resolved_checkpoint_path="/unused/path.safetensors",
        embeddings_paths=["/unused/embeddings"],
        loader_result=(object(), object(), object()),
    )

    missing_filename = "nonexistent.safetensors"
    expected_path = str((models_dir / "checkpoints" / missing_filename).resolve())

    with pytest.raises(FileNotFoundError, match="checkpoint file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_checkpoint(missing_filename)

    assert expected_path in str(exc_info.value)
    assert calls["get_full_path_or_raise"] == []
    assert calls["load_checkpoint_guess_config"] == []
