"""Tests for US-004: `python -m parallax` entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# AC01: cli/__main__.py exists and calls app() from cli/main.py
# ---------------------------------------------------------------------------

def test_ac01_cli_main_py_exists():
    assert (REPO_ROOT / "cli" / "__main__.py").is_file()


def test_ac01_cli_main_py_imports_app_and_calls_it():
    source = (REPO_ROOT / "cli" / "__main__.py").read_text()
    assert "from cli.main import app" in source
    assert "app()" in source


# ---------------------------------------------------------------------------
# AC02: pyproject.toml registers parallax script pointing to cli.main:app
# ---------------------------------------------------------------------------

def test_ac02_pyproject_script_entry_point():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'parallax = "cli.main:app"' in pyproject


# ---------------------------------------------------------------------------
# AC03: uv run parallax --help works
# ---------------------------------------------------------------------------

def test_ac03_uv_run_parallax_help():
    result = subprocess.run(
        ["uv", "run", "parallax", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "create" in result.stdout


# ---------------------------------------------------------------------------
# AC04: uv run python -m parallax --help works
# ---------------------------------------------------------------------------

def test_ac04_python_m_parallax_help():
    result = subprocess.run(
        ["uv", "run", "python", "-m", "parallax", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "create" in result.stdout


def test_ac04_parallax_package_main_py_exists():
    assert (REPO_ROOT / "parallax" / "__main__.py").is_file()


def test_ac04_parallax_main_py_delegates_to_cli():
    source = (REPO_ROOT / "parallax" / "__main__.py").read_text()
    assert "from cli.main import app" in source
    assert "app()" in source
