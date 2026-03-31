"""Tests for US-006 — example scripts for WAN 2.1 pipelines.

Covers:
  AC01: examples/video_wan21_t2v.py exists and follows the download+run pattern.
  AC02: examples/video_wan21_i2v.py exists and accepts --image (required).
  AC03: examples/video_wan21_flf2v.py exists and accepts --start-image and --end-image (required).
  AC04: Heavy imports deferred inside main(); missing required args print error and exit non-zero.
  AC05: All scripts follow the same structure as existing pipeline example scripts.
  AC06: Typecheck / lint passes (structural checks via AST).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples"

_T2V_SCRIPT = _EXAMPLES_DIR / "video_wan21_t2v.py"
_I2V_SCRIPT = _EXAMPLES_DIR / "video_wan21_i2v.py"
_FLF2V_SCRIPT = _EXAMPLES_DIR / "video_wan21_flf2v.py"

_ALL_SCRIPTS = [
    pytest.param(_T2V_SCRIPT, id="t2v"),
    pytest.param(_I2V_SCRIPT, id="i2v"),
    pytest.param(_FLF2V_SCRIPT, id="flf2v"),
]

# Heavy modules that must NOT be imported at the module top level.
_FORBIDDEN_TOP_LEVEL = {
    "torch",
    "comfy",
    "comfy_diffusion.downloader",
    "comfy_diffusion.pipelines",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(script: Path) -> ast.Module:
    return ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


def _top_level_import_names(tree: ast.Module) -> list[str]:
    """Return module names imported at the top level (not inside a function)."""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def _run_script(script: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(script)] + (extra_args or [])
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# AC01–AC03 — files exist and parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_script_file_exists(script: Path) -> None:
    assert script.is_file(), f"example script missing: {script}"


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_script_parses_without_syntax_errors(script: Path) -> None:
    tree = _parse(script)
    assert isinstance(tree, ast.Module)


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_script_has_module_docstring(script: Path) -> None:
    tree = _parse(script)
    docstring = ast.get_docstring(tree)
    assert docstring, f"{script.name} must have a module-level docstring"


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_script_has_future_annotations(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert "from __future__ import annotations" in source


# AC01 — t2v.py imports from the correct pipeline module
def test_t2v_imports_from_wan21_t2v_pipeline() -> None:
    source = _T2V_SCRIPT.read_text(encoding="utf-8")
    assert "from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest, run" in source
    assert "download_models(manifest()" in source
    assert "run(" in source


# AC02 — i2v.py imports from the correct pipeline module and requires --image
def test_i2v_imports_from_wan21_i2v_pipeline() -> None:
    source = _I2V_SCRIPT.read_text(encoding="utf-8")
    assert "from comfy_diffusion.pipelines.video.wan.wan21.i2v import manifest, run" in source
    assert "download_models(manifest()" in source
    assert "run(" in source


def test_i2v_has_required_image_flag() -> None:
    source = _I2V_SCRIPT.read_text(encoding="utf-8")
    assert "--image" in source


# AC03 — flf2v.py imports from the correct pipeline module and requires --start-image, --end-image
def test_flf2v_imports_from_wan21_flf2v_pipeline() -> None:
    source = _FLF2V_SCRIPT.read_text(encoding="utf-8")
    assert "from comfy_diffusion.pipelines.video.wan.wan21.flf2v import manifest, run" in source
    assert "download_models(manifest()" in source
    assert "run(" in source


def test_flf2v_has_required_image_flags() -> None:
    source = _FLF2V_SCRIPT.read_text(encoding="utf-8")
    assert "--start-image" in source
    assert "--end-image" in source


# ---------------------------------------------------------------------------
# AC04 (part 1) — common CLI flags present on all scripts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_script_has_common_cli_flags(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    for flag in ("--models-dir", "--prompt", "--width", "--height", "--length",
                 "--fps", "--steps", "--cfg", "--seed", "--output"):
        assert flag in source, f"{script.name} missing CLI flag: {flag}"


# ---------------------------------------------------------------------------
# AC04 (part 2) — missing required args → non-zero exit + error message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_script_exits_nonzero_with_no_args(script: Path) -> None:
    result = _run_script(script)
    assert result.returncode != 0, (
        f"{script.name} should exit non-zero when called with no arguments"
    )


def test_t2v_error_on_missing_models_dir() -> None:
    result = _run_script(_T2V_SCRIPT, ["--prompt", "test"])
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "error" in combined.lower() or "usage" in combined.lower()


def test_i2v_error_on_missing_models_dir(tmp_path: Path) -> None:
    dummy_img = tmp_path / "frame.png"
    dummy_img.touch()
    result = _run_script(_I2V_SCRIPT, ["--image", str(dummy_img), "--prompt", "test"])
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "error" in combined.lower() or "usage" in combined.lower()


def test_i2v_error_on_missing_image_flag() -> None:
    # --image is required; omitting it should error before even checking models_dir
    result = _run_script(_I2V_SCRIPT, ["--prompt", "test"])
    assert result.returncode != 0


def test_flf2v_error_on_missing_models_dir(tmp_path: Path) -> None:
    start = tmp_path / "start.png"
    end = tmp_path / "end.png"
    start.touch()
    end.touch()
    result = _run_script(
        _FLF2V_SCRIPT,
        ["--start-image", str(start), "--end-image", str(end), "--prompt", "test"],
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "error" in combined.lower() or "usage" in combined.lower()


def test_flf2v_error_on_missing_start_image_flag() -> None:
    result = _run_script(_FLF2V_SCRIPT, ["--prompt", "test"])
    assert result.returncode != 0


def test_flf2v_error_on_missing_end_image_flag(tmp_path: Path) -> None:
    start = tmp_path / "start.png"
    start.touch()
    result = _run_script(_FLF2V_SCRIPT, ["--start-image", str(start), "--prompt", "test"])
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# AC04 (part 3) / AC05 — no heavy imports at module top level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_no_heavy_top_level_imports(script: Path) -> None:
    tree = _parse(script)
    top_level = _top_level_import_names(tree)
    forbidden_found = [
        name for name in top_level
        if any(name == forbidden or name.startswith(forbidden + ".")
               for forbidden in _FORBIDDEN_TOP_LEVEL)
    ]
    assert not forbidden_found, (
        f"{script.name} has forbidden top-level imports: {forbidden_found}. "
        "All heavy imports must be inside main()."
    )


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_pipeline_imports_inside_main(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    lines_with_pipeline_import = [
        line for line in source.splitlines()
        if "comfy_diffusion.pipelines" in line and line.startswith("    ")
    ]
    assert lines_with_pipeline_import, (
        f"{script.name}: pipeline import must be inside main() (indented)"
    )


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_download_models_import_inside_main(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    lines_with_downloader_import = [
        line for line in source.splitlines()
        if "comfy_diffusion.downloader" in line and line.startswith("    ")
    ]
    assert lines_with_downloader_import, (
        f"{script.name}: downloader import must be inside main() (indented)"
    )


# ---------------------------------------------------------------------------
# AC05 — structural conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_main_function_defined(script: Path) -> None:
    tree = _parse(script)
    func_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    assert "main" in func_names, f"{script.name} must define a main() function"


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_if_name_main_guard(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source, (
        f"{script.name} must have an if __name__ == '__main__' guard"
    )


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_save_video_helper_defined(script: Path) -> None:
    tree = _parse(script)
    func_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    assert "_save_video" in func_names, f"{script.name} must define a _save_video() helper"


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_shebang_line(script: Path) -> None:
    first_line = script.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#!/usr/bin/env python"), (
        f"{script.name} must start with a shebang line"
    )
