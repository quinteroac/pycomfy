"""Tests for US-005 LTXAV text encoder loading behavior."""

from __future__ import annotations

import enum
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pycomfy.models import ModelManager


class _FakeCLIPType(enum.Enum):
    LTXV = "ltxv"


def _install_fake_ltxav_text_encoder_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embeddings_paths: list[str],
    clip_object: Any,
    resolved_text_encoder_path: str,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "add_model_folder_path": [],
        "get_full_path_or_raise": [],
        "get_folder_paths": [],
        "load_clip": [],
    }

    folder_paths_module = ModuleType("folder_paths")

    def add_model_folder_path(
        folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        calls["add_model_folder_path"].append((folder_name, full_folder_path, is_default))

    def get_full_path_or_raise(folder_name: str, filename: str) -> str:
        calls["get_full_path_or_raise"].append((folder_name, filename))
        if folder_name != "text_encoders":
            raise AssertionError(f"unexpected folder name: {folder_name}")
        return resolved_text_encoder_path

    def get_folder_paths(folder_name: str) -> list[str]:
        calls["get_folder_paths"].append(folder_name)
        if folder_name != "embeddings":
            raise AssertionError(f"unexpected folder name: {folder_name}")
        return embeddings_paths

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)
    setattr(folder_paths_module, "get_full_path_or_raise", get_full_path_or_raise)
    setattr(folder_paths_module, "get_folder_paths", get_folder_paths)

    comfy_sd_module = ModuleType("comfy.sd")
    setattr(comfy_sd_module, "CLIPType", _FakeCLIPType)

    def load_clip(*, ckpt_paths: list[str], embedding_directory: list[str], clip_type: Any) -> Any:
        calls["load_clip"].append((ckpt_paths, embedding_directory, clip_type))
        return clip_object

    setattr(comfy_sd_module, "load_clip", load_clip)

    comfy_module = ModuleType("comfy")
    setattr(comfy_module, "sd", comfy_sd_module)

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)

    return calls


def test_load_ltxav_text_encoder_calls_loader_and_returns_raw_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    checkpoints_dir = models_dir / "checkpoints"
    embeddings_dir = models_dir / "embeddings"
    checkpoints_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    text_encoder_file = tmp_path / "text_encoders" / "ltxav_text_encoder.safetensors"
    text_encoder_file.parent.mkdir(parents=True)
    text_encoder_file.write_text("stub ltxav text encoder")

    embeddings_paths = [str(embeddings_dir)]
    fake_clip = object()
    calls = _install_fake_ltxav_text_encoder_loader_modules(
        monkeypatch,
        embeddings_paths=embeddings_paths,
        clip_object=fake_clip,
        resolved_text_encoder_path="/unused/for/absolute/path.safetensors",
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_ltxav_text_encoder(
        text_encoder_file.parent / "." / text_encoder_file.name
    )

    assert result is fake_clip
    assert calls["get_full_path_or_raise"] == []
    assert calls["get_folder_paths"] == ["embeddings"]
    assert calls["load_clip"] == [
        ([str(text_encoder_file.resolve())], embeddings_paths, _FakeCLIPType.LTXV)
    ]
    assert calls["add_model_folder_path"] == [
        ("checkpoints", str(checkpoints_dir), True),
        ("embeddings", str(embeddings_dir), True),
        ("diffusion_models", str(models_dir / "unet"), True),
        ("diffusion_models", str(models_dir / "diffusion_models"), False),
        ("text_encoders", str(models_dir / "text_encoders"), True),
        ("text_encoders", str(models_dir / "clip"), False),
        ("vae", str(models_dir / "vae"), True),
    ]


def test_load_ltxav_text_encoder_is_callable_from_models_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    expected_resolved = str(
        (models_dir / "text_encoders" / "ltxav_text_encoder.safetensors").resolve()
    )
    fake_clip = object()
    calls = _install_fake_ltxav_text_encoder_loader_modules(
        monkeypatch,
        embeddings_paths=[str(models_dir / "embeddings")],
        clip_object=fake_clip,
        resolved_text_encoder_path=expected_resolved,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_ltxav_text_encoder("ltxav_text_encoder.safetensors")

    assert result is fake_clip
    assert calls["get_full_path_or_raise"] == [("text_encoders", "ltxav_text_encoder.safetensors")]
    assert calls["load_clip"] == [
        ([expected_resolved], [str(models_dir / "embeddings")], _FakeCLIPType.LTXV)
    ]


def test_load_ltxav_text_encoder_raises_file_not_found_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    calls = _install_fake_ltxav_text_encoder_loader_modules(
        monkeypatch,
        embeddings_paths=[str(models_dir / "embeddings")],
        clip_object=object(),
        resolved_text_encoder_path="/unused/path.safetensors",
    )

    missing_text_encoder_path = (
        tmp_path / "text_encoders" / "missing_ltxav_text_encoder.safetensors"
    )
    expected_path = str(missing_text_encoder_path.resolve())

    with pytest.raises(FileNotFoundError, match="ltxav text encoder file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_ltxav_text_encoder(
            str(missing_text_encoder_path)
        )

    assert expected_path in str(exc_info.value)
    assert calls["get_full_path_or_raise"] == []
    assert calls["get_folder_paths"] == []
    assert calls["load_clip"] == []
