"""Tests for US-001 package structure and exports."""

from __future__ import annotations

from pathlib import Path

import pycomfy
from pycomfy import check_runtime


def test_package_root_exists_with_init() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "pycomfy"
    assert package_dir.is_dir()
    assert (package_dir / "__init__.py").is_file()


def test_check_runtime_is_public_symbol() -> None:
    assert callable(check_runtime)
    assert pycomfy.check_runtime is check_runtime
    assert "check_runtime" in pycomfy.__all__


def test_package_uses_src_less_layout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "pycomfy").is_dir()
    assert not (repo_root / "src" / "pycomfy").exists()
