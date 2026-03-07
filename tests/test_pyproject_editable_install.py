"""Tests for US-003 pyproject metadata and editable install behavior."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_pyproject() -> dict:
    pyproject_path = _repo_root() / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def _run_command(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd or _repo_root(),
        check=True,
        text=True,
        capture_output=True,
    )


def _venv_python_path(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def test_pyproject_declares_required_project_metadata() -> None:
    pyproject = _read_pyproject()
    project = pyproject["project"]

    assert project["name"] == "pycomfy"
    assert project["version"]
    assert project["description"]
    assert project["requires-python"] == ">=3.12"


def test_torch_optional_dependencies_are_declared_as_cpu_and_cuda_extras() -> None:
    pyproject = _read_pyproject()
    optional = pyproject["project"]["optional-dependencies"]

    assert optional["cpu"] == ["torch"]
    assert optional["cuda"] == ["torch"]


def test_torch_optional_dependencies_are_not_version_pinned() -> None:
    pyproject = _read_pyproject()
    optional = pyproject["project"]["optional-dependencies"]

    for extra in ("cpu", "cuda"):
        for dependency in optional[extra]:
            assert dependency == "torch"
            assert "==" not in dependency
            assert ">=" not in dependency
            assert "<=" not in dependency


def test_uv_editable_install_succeeds_and_allows_importing_pycomfy(tmp_path: Path) -> None:
    submodule_path = _repo_root() / "vendor" / "ComfyUI"
    assert submodule_path.is_dir()

    venv_dir = tmp_path / ".venv"
    _run_command("uv", "venv", str(venv_dir))

    venv_python = _venv_python_path(venv_dir)
    assert venv_python.is_file()

    _run_command("uv", "pip", "install", "--python", str(venv_python), "-e", ".")
    import_result = _run_command(
        str(venv_python),
        "-c",
        (
            "import pycomfy; "
            "result = pycomfy.check_runtime(); "
            "assert isinstance(result, dict); "
            "assert 'python_version' in result"
        ),
    )
    assert import_result.returncode == 0
