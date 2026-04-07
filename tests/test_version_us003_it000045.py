"""Tests for US-003: Binary is self-identified.

AC01 — ``parallax --version`` prints ``parallax 1.x.x`` matching pyproject.toml.
AC02 — Version is a build-time constant in ``cli._version``, not read from an
       installed package at runtime.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli._version import __version__
from cli.main import app


REPO_ROOT = Path(__file__).parent.parent
runner = CliRunner()


# ── AC01: --version output format ─────────────────────────────────────────

def test_version_flag_prints_parallax_prefix() -> None:
    """Output starts with 'parallax '."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("parallax ")


def test_version_flag_format_semver() -> None:
    """Output matches 'parallax MAJOR.MINOR.PATCH'."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert re.match(r"^parallax \d+\.\d+\.\d+", result.output)


def test_version_matches_pyproject_toml() -> None:
    """Printed version matches [project].version in pyproject.toml."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"parallax {expected}" in result.output


# ── AC02: version is a build-time constant ────────────────────────────────

def test_version_constant_exists_in_cli_version_module() -> None:
    """cli._version.__version__ is a non-empty string."""
    assert isinstance(__version__, str)
    assert __version__ != ""


def test_version_constant_is_semver() -> None:
    """cli._version.__version__ is MAJOR.MINOR.PATCH format."""
    assert re.match(r"^\d+\.\d+\.\d+$", __version__), (
        f"__version__ {__version__!r} is not semver"
    )


def test_version_constant_matches_pyproject_toml() -> None:
    """cli._version.__version__ equals [project].version in pyproject.toml."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    assert __version__ == expected


def test_version_not_from_importlib_metadata() -> None:
    """cli._version does not use importlib.metadata (binary-safe)."""
    version_source = (REPO_ROOT / "cli" / "_version.py").read_text()
    assert "importlib" not in version_source
    assert "pkg_resources" not in version_source


def test_version_module_file_exists() -> None:
    """cli/_version.py file exists in the repository."""
    assert (REPO_ROOT / "cli" / "_version.py").is_file()


# ── spec bakes the version ────────────────────────────────────────────────

def test_spec_reads_pyproject_toml_for_version() -> None:
    """parallax.spec contains code that reads pyproject.toml for the version."""
    spec_content = (REPO_ROOT / "parallax.spec").read_text()
    assert "pyproject.toml" in spec_content
    assert "_version.py" in spec_content or "_project_version" in spec_content


def test_spec_writes_version_to_cli_version_module() -> None:
    """parallax.spec writes the baked version into cli/_version.py."""
    spec_content = (REPO_ROOT / "parallax.spec").read_text()
    assert "_version.py" in spec_content
    assert "write_text" in spec_content or "write(" in spec_content
