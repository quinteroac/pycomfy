"""Tests for US-004 distributable agent skills."""

from __future__ import annotations

import subprocess
import tomllib
import zipfile
from importlib import resources
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_pyproject() -> dict[str, Any]:
    return tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))


def test_skills_directory_exists_with_markdown_files() -> None:
    skills_dir = _repo_root() / "comfy_diffusion" / "skills"

    assert skills_dir.is_dir()
    assert list(skills_dir.glob("*.md"))


def test_pyproject_declares_skills_as_package_data() -> None:
    pyproject = _read_pyproject()
    package_data = pyproject["tool"]["setuptools"]["package-data"]["comfy_diffusion"]

    assert "skills/**/*.md" in package_data


def test_skills_are_discoverable_via_importlib_resources() -> None:
    skills_root = resources.files("comfy_diffusion.skills")

    assert skills_root.is_dir()
    assert any(item.name.endswith(".md") for item in skills_root.iterdir())


def test_distributable_skills_are_distinct_from_internal_agent_skills() -> None:
    distributable_skills_dir = _repo_root() / "comfy_diffusion" / "skills"
    internal_skills_dir = _repo_root() / ".agents" / "skills"

    assert distributable_skills_dir.is_dir()
    assert internal_skills_dir.is_dir()
    assert distributable_skills_dir != internal_skills_dir
    assert ".agents" not in distributable_skills_dir.parts


def test_wheel_contains_distributable_skills_only_in_package_path(tmp_path: Path) -> None:
    build_result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
    )
    assert build_result.returncode == 0

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())

    assert "comfy_diffusion/skills/README.md" in names
    assert not any(name.startswith(".agents/skills/") for name in names)

