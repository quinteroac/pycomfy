"""Tests for US-005 CPU-only import and constructor smoke behavior."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from pycomfy.models import ModelManager


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_with_repo_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    repo_root = str(_repo_root())
    env["PYTHONPATH"] = (
        repo_root if not existing else f"{repo_root}{os.pathsep}{existing}"
    )
    return env


def test_uv_run_python_imports_model_manager_on_cpu_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from pycomfy.models import ModelManager; print('ok')",
        ],
        cwd=_repo_root(),
        env=_env_with_repo_pythonpath(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_model_manager_is_importable_and_constructs_with_valid_models_dir(
    tmp_path: Path,
) -> None:
    manager = ModelManager(models_dir=tmp_path)
    assert manager.models_dir == tmp_path


def test_constructor_flow_does_not_call_load_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_call(_: ModelManager, filename: str) -> None:
        raise AssertionError(
            f"load_checkpoint should not be called during smoke test: {filename}"
        )

    monkeypatch.setattr(ModelManager, "load_checkpoint", _unexpected_call)

    manager = ModelManager(models_dir=tmp_path)
    assert manager.models_dir == tmp_path
