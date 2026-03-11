"""Tests for US-003 typing metadata and public API annotations."""

from __future__ import annotations

import inspect
import subprocess
import tomllib
import zipfile
from pathlib import Path
from typing import Any

from comfy_diffusion import apply_lora, check_runtime, vae_decode, vae_encode
from comfy_diffusion.conditioning import encode_prompt
from comfy_diffusion.sampling import sample


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_pyproject() -> dict[str, Any]:
    return tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))


def test_py_typed_marker_file_exists() -> None:
    assert (_repo_root() / "comfy_diffusion" / "py.typed").is_file()


def test_pyproject_declares_py_typed_as_package_data() -> None:
    pyproject = _read_pyproject()
    package_data = pyproject["tool"]["setuptools"]["package-data"]["comfy_diffusion"]
    assert "py.typed" in package_data


def test_public_api_functions_have_inline_type_annotations() -> None:
    symbols = [
        check_runtime,
        vae_decode,
        vae_encode,
        apply_lora,
        encode_prompt,
        sample,
    ]

    for symbol in symbols:
        signature = inspect.signature(symbol)
        for parameter in signature.parameters.values():
            assert parameter.annotation is not inspect.Signature.empty
        assert signature.return_annotation is not inspect.Signature.empty


def test_wheel_contains_py_typed_marker(tmp_path: Path) -> None:
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

    assert "comfy_diffusion/py.typed" in names
