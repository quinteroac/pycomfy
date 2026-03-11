"""Post-install smoke tests for US-005 packaging regression coverage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def test_clean_venv_install_import_runtime_and_packaged_data_smoke(tmp_path: Path) -> None:
    venv_dir = tmp_path / ".venv"
    _run_command("uv", "venv", str(venv_dir))

    venv_python = _venv_python_path(venv_dir)
    assert venv_python.is_file()

    _run_command("uv", "pip", "install", "--python", str(venv_python), ".")
    smoke_result = _run_command(
        str(venv_python),
        "-c",
        (
            "import importlib.resources as ir; "
            "import comfy_diffusion; "
            "payload = comfy_diffusion.check_runtime(); "
            "assert isinstance(payload, dict); "
            "required = {'comfyui_version', 'device', 'vram_total_mb', "
            "'vram_free_mb', 'python_version'}; "
            "assert required.issubset(payload.keys()); "
            "skills = ir.files('comfy_diffusion.skills'); "
            "assert skills.is_dir(); "
            "assert any(item.name.endswith('.md') for item in skills.iterdir()); "
            "assert ir.files('comfy_diffusion').joinpath('py.typed').is_file()"
        ),
        cwd=tmp_path,
    )
    assert smoke_result.returncode == 0
