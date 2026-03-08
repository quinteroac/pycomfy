"""Tests for US-002 ModelManager constructor behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from comfy_diffusion.models import ModelManager


def test_model_manager_constructs_with_existing_models_dir(tmp_path: Path) -> None:
    manager = ModelManager(models_dir=tmp_path)

    assert isinstance(manager.models_dir, Path)
    assert manager.models_dir == tmp_path


def test_model_manager_accepts_string_models_dir(tmp_path: Path) -> None:
    manager = ModelManager(models_dir=str(tmp_path))

    assert isinstance(manager.models_dir, Path)
    assert manager.models_dir == tmp_path


def test_model_manager_raises_value_error_for_non_existing_models_dir(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(ValueError, match="models_dir does not exist") as exc_info:
        ModelManager(models_dir=missing_dir)

    assert str(missing_dir) in str(exc_info.value)
