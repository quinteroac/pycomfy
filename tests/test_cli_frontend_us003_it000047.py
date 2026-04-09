"""Tests for US-003 it_000047 — parallax frontend version CLI command.

Covers:
  AC01 — prints installed version from ~/.parallax/frontend/version.txt
  AC02 — if no frontend installed, prints helpful message
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from cli.commands.frontend import app

runner = CliRunner()

# Wrap the frontend sub-app so ["frontend", "version"] routes correctly.
_cli = typer.Typer()
_cli.add_typer(app, name="frontend")


# ---------------------------------------------------------------------------
# AC01 — version.txt exists → prints its contents
# ---------------------------------------------------------------------------


def test_version_prints_installed_version(tmp_path: Path) -> None:
    """AC01: reads version.txt and prints the version string."""
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir(parents=True)
    (frontend_dir / "version.txt").write_text("1.5.0\n", encoding="utf-8")

    with patch("cli.commands.frontend.FRONTEND_DIR", frontend_dir):
        result = runner.invoke(_cli, ["frontend", "version"])

    assert result.exit_code == 0
    assert "1.5.0" in result.output


def test_version_strips_whitespace(tmp_path: Path) -> None:
    """AC01: trailing newlines/spaces in version.txt are stripped."""
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir(parents=True)
    (frontend_dir / "version.txt").write_text("  2.0.1  \n", encoding="utf-8")

    with patch("cli.commands.frontend.FRONTEND_DIR", frontend_dir):
        result = runner.invoke(_cli, ["frontend", "version"])

    assert result.exit_code == 0
    assert result.output.strip() == "2.0.1"


# ---------------------------------------------------------------------------
# AC02 — version.txt missing → helpful not-installed message
# ---------------------------------------------------------------------------


def test_version_not_installed(tmp_path: Path) -> None:
    """AC02: no frontend dir → prints helpful message."""
    frontend_dir = tmp_path / "frontend"  # deliberately does not exist

    with patch("cli.commands.frontend.FRONTEND_DIR", frontend_dir):
        result = runner.invoke(_cli, ["frontend", "version"])

    assert result.exit_code == 0
    assert "Frontend not installed" in result.output
    assert "parallax frontend install" in result.output


def test_version_dir_exists_but_no_version_file(tmp_path: Path) -> None:
    """AC02: dir exists but version.txt is absent → prints helpful message."""
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir(parents=True)
    # No version.txt created

    with patch("cli.commands.frontend.FRONTEND_DIR", frontend_dir):
        result = runner.invoke(_cli, ["frontend", "version"])

    assert result.exit_code == 0
    assert "Frontend not installed" in result.output
    assert "parallax frontend install" in result.output
