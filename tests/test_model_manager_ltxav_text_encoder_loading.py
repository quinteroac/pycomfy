"""Tests for US-005 LTXAV text encoder loading behavior."""

from __future__ import annotations

import enum
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from comfy_diffusion.models import ModelManager


class _FakeCLIPType(enum.Enum):
    LTXV = "ltxv"


def _install_fake_ltxav_text_encoder_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embeddings_paths: list[str],
    clip_object: Any,
    resolved_text_encoder_path: str,
    resolved_checkpoint_path: str,
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
        if folder_name == "text_encoders":
            return resolved_text_encoder_path
        if folder_name == "checkpoints":
            return resolved_checkpoint_path
        raise AssertionError(f"unexpected folder name: {folder_name}")

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


def _install_fake_modules_with_model_options(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embeddings_paths: list[str],
    clip_object: Any,
    resolved_text_encoder_path: str,
    resolved_checkpoint_path: str,
) -> dict[str, list[Any]]:
    """Like the helper above but load_clip captures model_options too."""
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
        if folder_name == "text_encoders":
            return resolved_text_encoder_path
        if folder_name == "checkpoints":
            return resolved_checkpoint_path
        raise AssertionError(f"unexpected folder name: {folder_name}")

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

    def load_clip(
        *,
        ckpt_paths: list[str],
        embedding_directory: list[str],
        clip_type: Any,
        model_options: dict[str, Any] | None = None,
    ) -> Any:
        calls["load_clip"].append(
            {
                "ckpt_paths": ckpt_paths,
                "embedding_directory": embedding_directory,
                "clip_type": clip_type,
                "model_options": model_options,
            }
        )
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

    checkpoint_file = tmp_path / "weights" / "ltxav_checkpoint.safetensors"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_text("stub ltxav checkpoint")

    embeddings_paths = [str(embeddings_dir)]
    fake_clip = object()
    calls = _install_fake_ltxav_text_encoder_loader_modules(
        monkeypatch,
        embeddings_paths=embeddings_paths,
        clip_object=fake_clip,
        resolved_text_encoder_path="/unused/for/absolute/te.safetensors",
        resolved_checkpoint_path="/unused/for/absolute/ckpt.safetensors",
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_ltxav_text_encoder(
        text_encoder_file.parent / "." / text_encoder_file.name,
        checkpoint_file.parent / "." / checkpoint_file.name,
    )

    assert result is fake_clip
    assert calls["get_full_path_or_raise"] == []
    assert calls["get_folder_paths"] == ["embeddings"]
    assert calls["load_clip"] == [
        (
            [str(text_encoder_file.resolve()), str(checkpoint_file.resolve())],
            embeddings_paths,
            _FakeCLIPType.LTXV,
        )
    ]
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


def test_load_ltxav_text_encoder_is_callable_from_models_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    resolved_te = str(
        (models_dir / "text_encoders" / "ltxav_text_encoder.safetensors").resolve()
    )
    resolved_ckpt = str(
        (models_dir / "checkpoints" / "ltxav_checkpoint.safetensors").resolve()
    )
    fake_clip = object()
    calls = _install_fake_ltxav_text_encoder_loader_modules(
        monkeypatch,
        embeddings_paths=[str(models_dir / "embeddings")],
        clip_object=fake_clip,
        resolved_text_encoder_path=resolved_te,
        resolved_checkpoint_path=resolved_ckpt,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_ltxav_text_encoder(
        "ltxav_text_encoder.safetensors",
        "ltxav_checkpoint.safetensors",
    )

    assert result is fake_clip
    assert calls["get_full_path_or_raise"] == [
        ("text_encoders", "ltxav_text_encoder.safetensors"),
        ("checkpoints", "ltxav_checkpoint.safetensors"),
    ]
    assert calls["load_clip"] == [
        (
            [resolved_te, resolved_ckpt],
            [str(models_dir / "embeddings")],
            _FakeCLIPType.LTXV,
        )
    ]


def test_load_ltxav_text_encoder_raises_file_not_found_for_missing_text_encoder(
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
        resolved_checkpoint_path="/unused/ckpt.safetensors",
    )

    missing_te_path = tmp_path / "text_encoders" / "missing_ltxav_text_encoder.safetensors"
    existing_ckpt = tmp_path / "weights" / "ltxav_checkpoint.safetensors"
    existing_ckpt.parent.mkdir(parents=True)
    existing_ckpt.write_text("stub")

    with pytest.raises(FileNotFoundError, match="ltxav text encoder file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_ltxav_text_encoder(
            str(missing_te_path),
            str(existing_ckpt),
        )

    assert str(missing_te_path.resolve()) in str(exc_info.value)
    assert calls["get_full_path_or_raise"] == []
    assert calls["get_folder_paths"] == []
    assert calls["load_clip"] == []


def test_load_ltxav_text_encoder_raises_file_not_found_for_missing_checkpoint(
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
        resolved_checkpoint_path="/unused/ckpt.safetensors",
    )

    existing_te = tmp_path / "text_encoders" / "ltxav_text_encoder.safetensors"
    existing_te.parent.mkdir(parents=True)
    existing_te.write_text("stub")
    missing_ckpt_path = tmp_path / "weights" / "missing_checkpoint.safetensors"

    with pytest.raises(FileNotFoundError, match="ltxav checkpoint file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_ltxav_text_encoder(
            str(existing_te),
            str(missing_ckpt_path),
        )

    assert str(missing_ckpt_path.resolve()) in str(exc_info.value)
    assert calls["get_full_path_or_raise"] == []
    assert calls["get_folder_paths"] == []
    assert calls["load_clip"] == []


# ---------------------------------------------------------------------------
# US-004: device parameter tests
# ---------------------------------------------------------------------------


def test_load_ltxav_text_encoder_device_cpu_passes_model_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-02 / AC-05: device='cpu' passes model_options with cpu devices."""
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    text_encoder_file = tmp_path / "text_encoders" / "ltxav_te.safetensors"
    text_encoder_file.parent.mkdir(parents=True)
    text_encoder_file.write_text("stub")

    checkpoint_file = tmp_path / "weights" / "ltxav_ckpt.safetensors"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_text("stub")

    # Stub torch.device so the test runs without a real torch installation.
    fake_cpu_device = object()
    torch_module = ModuleType("torch")

    def fake_device(name: str) -> object:
        assert name == "cpu"
        return fake_cpu_device

    setattr(torch_module, "device", fake_device)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    embeddings_paths = [str(models_dir / "embeddings")]
    fake_clip = object()
    calls = _install_fake_modules_with_model_options(
        monkeypatch,
        embeddings_paths=embeddings_paths,
        clip_object=fake_clip,
        resolved_text_encoder_path="/unused/te.safetensors",
        resolved_checkpoint_path="/unused/ckpt.safetensors",
    )

    result = ModelManager(models_dir=models_dir).load_ltxav_text_encoder(
        text_encoder_file,
        checkpoint_file,
        device="cpu",
    )

    assert result is fake_clip
    assert len(calls["load_clip"]) == 1
    call = calls["load_clip"][0]
    assert call["model_options"] is not None
    assert call["model_options"]["load_device"] is fake_cpu_device
    assert call["model_options"]["offload_device"] is fake_cpu_device


def test_load_ltxav_text_encoder_device_default_omits_model_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-03: device='default' (or omitted) passes no model_options."""
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    text_encoder_file = tmp_path / "text_encoders" / "ltxav_te.safetensors"
    text_encoder_file.parent.mkdir(parents=True)
    text_encoder_file.write_text("stub")

    checkpoint_file = tmp_path / "weights" / "ltxav_ckpt.safetensors"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_text("stub")

    embeddings_paths = [str(models_dir / "embeddings")]
    fake_clip = object()
    calls = _install_fake_modules_with_model_options(
        monkeypatch,
        embeddings_paths=embeddings_paths,
        clip_object=fake_clip,
        resolved_text_encoder_path="/unused/te.safetensors",
        resolved_checkpoint_path="/unused/ckpt.safetensors",
    )

    result = ModelManager(models_dir=models_dir).load_ltxav_text_encoder(
        text_encoder_file,
        checkpoint_file,
        device="default",
    )

    assert result is fake_clip
    assert len(calls["load_clip"]) == 1
    call = calls["load_clip"][0]
    assert call["model_options"] is None


def test_load_ltxav_text_encoder_device_omitted_omits_model_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-03: omitting device is equivalent to device='default'."""
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    text_encoder_file = tmp_path / "text_encoders" / "ltxav_te.safetensors"
    text_encoder_file.parent.mkdir(parents=True)
    text_encoder_file.write_text("stub")

    checkpoint_file = tmp_path / "weights" / "ltxav_ckpt.safetensors"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_text("stub")

    embeddings_paths = [str(models_dir / "embeddings")]
    fake_clip = object()
    calls = _install_fake_modules_with_model_options(
        monkeypatch,
        embeddings_paths=embeddings_paths,
        clip_object=fake_clip,
        resolved_text_encoder_path="/unused/te.safetensors",
        resolved_checkpoint_path="/unused/ckpt.safetensors",
    )

    result = ModelManager(models_dir=models_dir).load_ltxav_text_encoder(
        text_encoder_file,
        checkpoint_file,
    )

    assert result is fake_clip
    assert len(calls["load_clip"]) == 1
    call = calls["load_clip"][0]
    assert call["model_options"] is None
