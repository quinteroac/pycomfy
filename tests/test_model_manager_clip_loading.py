"""Tests for US-004 standalone CLIP loading behavior."""

from __future__ import annotations

import enum
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from comfy_diffusion.models import ModelManager


class _FakeCLIPType(enum.Enum):
    STABLE_DIFFUSION = "stable_diffusion"
    SDXL = "sdxl"
    FLUX = "flux"


def _install_fake_clip_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text_encoders_dir: Path,
    embeddings_paths: list[str],
    clip_object: Any,
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

    def get_folder_paths(folder_name: str) -> list[str]:
        calls["get_folder_paths"].append(folder_name)
        if folder_name != "embeddings":
            raise AssertionError(f"unexpected folder name: {folder_name}")
        return embeddings_paths

    def get_full_path_or_raise(folder_name: str, filename: str) -> str:
        calls["get_full_path_or_raise"].append((folder_name, filename))
        if folder_name != "text_encoders":
            raise AssertionError(f"unexpected folder name: {folder_name}")
        return str(text_encoders_dir / filename)

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)
    setattr(folder_paths_module, "get_folder_paths", get_folder_paths)
    setattr(folder_paths_module, "get_full_path_or_raise", get_full_path_or_raise)

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
        monkeypatch,
        text_encoders_dir=tmp_path / "text_encoders",
        embeddings_paths=embeddings_paths,
        clip_object=fake_clip,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_clip(clip_file.parent / "." / clip_file.name)

    assert result is fake_clip
    assert calls["load_clip"] == [
        ([str(clip_file.resolve())], _FakeCLIPType.STABLE_DIFFUSION, embeddings_paths)
    ]
    assert calls["get_folder_paths"] == ["embeddings"]
    assert calls["get_full_path_or_raise"] == []
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
        ("audio_encoders", str(models_dir / "audio_encoders"), True),
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
        text_encoders_dir=tmp_path / "text_encoders",
        embeddings_paths=[str(models_dir / "embeddings")],
        clip_object=object(),
    )

    missing_clip_path = tmp_path / "text_encoders" / "missing_clip.safetensors"
    expected_path = str(missing_clip_path.resolve())

    with pytest.raises(FileNotFoundError, match="clip file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_clip(str(missing_clip_path))

    assert expected_path in str(exc_info.value)
    assert calls["get_folder_paths"] == []
    assert calls["get_full_path_or_raise"] == []
    assert calls["load_clip"] == []


def test_load_clip_resolves_two_relative_paths_and_uses_requested_clip_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    embeddings_dir = models_dir / "embeddings"
    (models_dir / "checkpoints").mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    clip_l = tmp_path / "text_encoders" / "clip_l.safetensors"
    t5 = tmp_path / "text_encoders" / "t5xxl_fp16.safetensors"
    clip_l.parent.mkdir(parents=True)
    clip_l.write_text("stub clip l")
    t5.write_text("stub t5")

    embeddings_paths = [str(embeddings_dir)]
    fake_clip = object()
    calls = _install_fake_clip_loader_modules(
        monkeypatch,
        text_encoders_dir=tmp_path / "text_encoders",
        embeddings_paths=embeddings_paths,
        clip_object=fake_clip,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_clip(
        "clip_l.safetensors",
        "t5xxl_fp16.safetensors",
        clip_type="flux",
    )

    assert result is fake_clip
    assert calls["get_full_path_or_raise"] == [
        ("text_encoders", "clip_l.safetensors"),
        ("text_encoders", "t5xxl_fp16.safetensors"),
    ]
    assert calls["load_clip"] == [
        (
            [str(clip_l), str(t5)],
            _FakeCLIPType.FLUX,
            embeddings_paths,
        )
    ]
    assert calls["get_folder_paths"] == ["embeddings"]


def test_load_clip_raises_value_error_when_called_without_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    calls = _install_fake_clip_loader_modules(
        monkeypatch,
        text_encoders_dir=tmp_path / "text_encoders",
        embeddings_paths=[str(models_dir / "embeddings")],
        clip_object=object(),
    )

    with pytest.raises(ValueError, match="load_clip requires at least one path"):
        ModelManager(models_dir=models_dir).load_clip()

    assert calls["get_full_path_or_raise"] == []
    assert calls["get_folder_paths"] == []
    assert calls["load_clip"] == []


def test_load_clip_resolves_relative_folder_paths_results_to_absolute_ckpt_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    embeddings_dir = models_dir / "embeddings"
    (models_dir / "checkpoints").mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    clip_l = tmp_path / "text_encoders" / "clip_l.safetensors"
    t5 = tmp_path / "text_encoders" / "t5xxl_fp16.safetensors"
    clip_l.parent.mkdir(parents=True)
    clip_l.write_text("stub clip l")
    t5.write_text("stub t5")

    monkeypatch.chdir(tmp_path)
    calls = _install_fake_clip_loader_modules(
        monkeypatch,
        text_encoders_dir=Path("text_encoders"),
        embeddings_paths=[str(embeddings_dir)],
        clip_object=object(),
    )

    ModelManager(models_dir=models_dir).load_clip(
        "clip_l.safetensors",
        "t5xxl_fp16.safetensors",
    )

    assert calls["load_clip"] == [
        (
            [str(clip_l.resolve()), str(t5.resolve())],
            _FakeCLIPType.STABLE_DIFFUSION,
            [str(embeddings_dir)],
        )
    ]


def test_load_clip_raises_file_not_found_when_second_path_is_missing_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    text_encoders_dir = tmp_path / "text_encoders"
    existing_clip = text_encoders_dir / "clip_l.safetensors"
    existing_clip.parent.mkdir(parents=True)
    existing_clip.write_text("stub clip")

    calls = _install_fake_clip_loader_modules(
        monkeypatch,
        text_encoders_dir=text_encoders_dir,
        embeddings_paths=[str(models_dir / "embeddings")],
        clip_object=object(),
    )

    with pytest.raises(
        FileNotFoundError, match=r"clip file not found: missing_second\.safetensors"
    ):
        ModelManager(models_dir=models_dir).load_clip(
            "clip_l.safetensors",
            "missing_second.safetensors",
        )

    assert calls["get_full_path_or_raise"] == [
        ("text_encoders", "clip_l.safetensors"),
        ("text_encoders", "missing_second.safetensors"),
    ]
    assert calls["get_folder_paths"] == []
    assert calls["load_clip"] == []


def test_load_clip_raises_file_not_found_when_relative_resolution_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    calls = _install_fake_clip_loader_modules(
        monkeypatch,
        text_encoders_dir=tmp_path / "text_encoders",
        embeddings_paths=[str(models_dir / "embeddings")],
        clip_object=object(),
    )

    with pytest.raises(
        FileNotFoundError,
        match=r"clip file not found: missing_relative\.safetensors",
    ):
        ModelManager(models_dir=models_dir).load_clip("missing_relative.safetensors")

    assert calls["get_full_path_or_raise"] == [
        ("text_encoders", "missing_relative.safetensors")
    ]
    assert calls["get_folder_paths"] == []
    assert calls["load_clip"] == []


def test_load_clip_raises_value_error_for_invalid_clip_type_with_valid_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    clip_file = tmp_path / "text_encoders" / "clip_l.safetensors"
    clip_file.parent.mkdir(parents=True)
    clip_file.write_text("stub")

    calls = _install_fake_clip_loader_modules(
        monkeypatch,
        text_encoders_dir=tmp_path / "text_encoders",
        embeddings_paths=[str(models_dir / "embeddings")],
        clip_object=object(),
    )

    with pytest.raises(ValueError, match="invalid clip_type 'not_real'") as exc_info:
        ModelManager(models_dir=models_dir).load_clip(clip_file, clip_type="not_real")

    message = str(exc_info.value)
    assert "stable_diffusion" in message
    assert "sdxl" in message
    assert "flux" in message
    assert calls["get_full_path_or_raise"] == []
    assert calls["get_folder_paths"] == []
    assert calls["load_clip"] == []
