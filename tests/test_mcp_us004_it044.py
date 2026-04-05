"""Tests for US-004: MCP server entry point."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# AC01: mcp/__main__.py exists and starts the fastmcp server in stdio mode
# ---------------------------------------------------------------------------

def test_ac01_mcp_main_py_exists():
    assert (REPO_ROOT / "mcp" / "__main__.py").is_file()


def test_ac01_mcp_main_py_imports_and_calls_main():
    source = (REPO_ROOT / "mcp" / "__main__.py").read_text()
    assert "from mcp.main import main" in source
    assert "main()" in source


def test_ac01_mcp_main_py_uses_fastmcp():
    source = (REPO_ROOT / "mcp" / "main.py").read_text()
    assert "fastmcp" in source
    assert "stdio" in source


# ---------------------------------------------------------------------------
# AC02: pyproject.toml registers parallax-mcp → mcp.main:main
# ---------------------------------------------------------------------------

def test_ac02_pyproject_script_entry_point():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'parallax-mcp = "mcp.main:main"' in pyproject


def test_ac02_pyproject_toml_parseable():
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    scripts = data["project"]["scripts"]
    assert scripts.get("parallax-mcp") == "mcp.main:main"


# ---------------------------------------------------------------------------
# AC03: uv run parallax-mcp starts without error
# ---------------------------------------------------------------------------

def test_ac03_uv_run_parallax_mcp_starts():
    """Server starts in stdio mode; send empty stdin so it exits cleanly."""
    result = subprocess.run(
        ["uv", "run", "parallax-mcp"],
        input="",
        capture_output=True,
        text=True,
        timeout=10,
        cwd=REPO_ROOT,
    )
    # The server exits cleanly when stdin closes (no data → EOF).
    assert result.returncode == 0


def test_ac03_mcp_main_importable():
    result = subprocess.run(
        ["uv", "run", "python", "-c", "from mcp.main import mcp, main; print('ok')"],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "ok" in result.stdout


# ---------------------------------------------------------------------------
# AC04: server name is "parallax-mcp" with version matching pyproject.toml
# ---------------------------------------------------------------------------

def test_ac04_server_name():
    result = subprocess.run(
        ["uv", "run", "python", "-c", "from mcp.main import mcp; print(mcp.name)"],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "parallax-mcp"


def test_ac04_server_version_matches_pyproject():
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    expected_version = data["project"]["version"]

    result = subprocess.run(
        ["uv", "run", "python", "-c", "from mcp.main import mcp; print(mcp.version)"],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == expected_version
