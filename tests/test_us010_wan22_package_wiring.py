"""Tests for US-010 — WAN 2.2 package wiring and example scripts.

Covers:
  AC01: comfy_diffusion/pipelines/video/wan/wan22/__init__.py exists and exports
        t2v, i2v, flf2v, s2v, ti2v sub-modules.
  AC02: comfy_diffusion/pipelines/video/wan/__init__.py exposes wan22.
  AC03: examples/wan22_t2v.py, wan22_i2v.py, wan22_flf2v.py, wan22_s2v.py, wan22_ti2v.py
        are present, parse cleanly, document their models, and follow the
        download+run CLI pattern.
  AC04: Typecheck / lint passes (structural checks via AST; importability checks).
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINES_VIDEO = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video"
_EXAMPLES_DIR = _REPO_ROOT / "packages" / "parallax_cli" / "runtime"

_T2V_SCRIPT = _EXAMPLES_DIR / "video" / "wan" / "wan22" / "t2v.py"
_I2V_SCRIPT = _EXAMPLES_DIR / "video" / "wan" / "wan22" / "i2v.py"
_FLF2V_SCRIPT = _EXAMPLES_DIR / "video" / "wan" / "wan22" / "flf2v.py"
_S2V_SCRIPT = _EXAMPLES_DIR / "video" / "wan" / "wan22" / "s2v.py"
_TI2V_SCRIPT = _EXAMPLES_DIR / "video" / "wan" / "wan22" / "ti2v.py"

_ALL_SCRIPTS = [
    pytest.param(_T2V_SCRIPT, id="t2v"),
    pytest.param(_I2V_SCRIPT, id="i2v"),
    pytest.param(_FLF2V_SCRIPT, id="flf2v"),
    pytest.param(_S2V_SCRIPT, id="s2v"),
    pytest.param(_TI2V_SCRIPT, id="ti2v"),
]

# Heavy modules that must NOT be imported at the module top level.
_FORBIDDEN_TOP_LEVEL = {
    "torch",
    "comfy",
    "comfy_diffusion.downloader",
    "comfy_diffusion.pipelines",
    "comfy_diffusion.runtime",
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


def _run_script(script: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    cmd = [sys.executable, str(script)] + (extra_args or [])
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# AC01 — wan22/__init__.py exists and exports all 5 sub-modules
# ---------------------------------------------------------------------------


def test_wan22_init_exists() -> None:
    assert (_PIPELINES_VIDEO / "wan" / "wan22" / "__init__.py").is_file()


def test_wan22_init_parses() -> None:
    source = (_PIPELINES_VIDEO / "wan" / "wan22" / "__init__.py").read_text(encoding="utf-8")
    ast.parse(source)


def test_wan22_init_all_is_list() -> None:
    import comfy_diffusion.pipelines.video.wan.wan22 as wan22_pkg

    assert isinstance(wan22_pkg.__all__, list)


@pytest.mark.parametrize("name", ["t2v", "i2v", "flf2v", "s2v", "ti2v"])
def test_wan22_init_exports_submodule(name: str) -> None:
    import comfy_diffusion.pipelines.video.wan.wan22 as wan22_pkg

    assert name in wan22_pkg.__all__, (
        f"wan22/__init__.py __all__ must contain '{name}', got {wan22_pkg.__all__!r}"
    )


# ---------------------------------------------------------------------------
# AC02 — wan/__init__.py exposes wan22
# ---------------------------------------------------------------------------


def test_wan_init_exports_wan22() -> None:
    import comfy_diffusion.pipelines.video.wan as wan_pkg

    assert "wan22" in wan_pkg.__all__, (
        f"wan/__init__.py __all__ must contain 'wan22', got {wan_pkg.__all__!r}"
    )


def test_wan_init_exports_wan21_still_present() -> None:
    import comfy_diffusion.pipelines.video.wan as wan_pkg

    assert "wan21" in wan_pkg.__all__, (
        "wan/__init__.py __all__ must still contain 'wan21' alongside 'wan22'"
    )


# ---------------------------------------------------------------------------
# AC01 / AC04 — sub-modules are importable with manifest + run callables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_suffix", ["t2v", "i2v", "flf2v", "s2v", "ti2v"])
def test_wan22_submodule_importable(module_suffix: str) -> None:
    mod = importlib.import_module(
        f"comfy_diffusion.pipelines.video.wan.wan22.{module_suffix}"
    )
    assert mod is not None


@pytest.mark.parametrize("module_suffix", ["t2v", "i2v", "flf2v", "s2v", "ti2v"])
def test_wan22_submodule_exports_manifest_and_run(module_suffix: str) -> None:
    mod = importlib.import_module(
        f"comfy_diffusion.pipelines.video.wan.wan22.{module_suffix}"
    )
    assert callable(mod.manifest), f"wan22.{module_suffix}.manifest must be callable"
    assert callable(mod.run), f"wan22.{module_suffix}.run must be callable"


@pytest.mark.parametrize("module_suffix", ["t2v", "i2v", "flf2v", "s2v", "ti2v"])
def test_wan22_submodule_all_contains_manifest_and_run(module_suffix: str) -> None:
    mod = importlib.import_module(
        f"comfy_diffusion.pipelines.video.wan.wan22.{module_suffix}"
    )
    assert "manifest" in mod.__all__, f"wan22.{module_suffix}.__all__ must contain 'manifest'"
    assert "run" in mod.__all__, f"wan22.{module_suffix}.__all__ must contain 'run'"


# ---------------------------------------------------------------------------
# AC03 — example scripts exist and parse
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


# AC03 — each script imports from the correct wan22 pipeline submodule
def test_t2v_imports_from_wan22_t2v_pipeline() -> None:
    source = _T2V_SCRIPT.read_text(encoding="utf-8")
    assert "from comfy_diffusion.pipelines.video.wan.wan22.t2v import manifest, run" in source
    assert "download_models(manifest()" in source


def test_i2v_imports_from_wan22_i2v_pipeline() -> None:
    source = _I2V_SCRIPT.read_text(encoding="utf-8")
    assert "from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest, run" in source
    assert "download_models(manifest()" in source


def test_flf2v_imports_from_wan22_flf2v_pipeline() -> None:
    source = _FLF2V_SCRIPT.read_text(encoding="utf-8")
    assert "from comfy_diffusion.pipelines.video.wan.wan22.flf2v import manifest, run" in source
    assert "download_models(manifest()" in source


def test_s2v_imports_from_wan22_s2v_pipeline() -> None:
    source = _S2V_SCRIPT.read_text(encoding="utf-8")
    assert "from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest, run" in source
    assert "download_models(manifest()" in source


def test_ti2v_imports_from_wan22_ti2v_pipeline() -> None:
    source = _TI2V_SCRIPT.read_text(encoding="utf-8")
    assert "from comfy_diffusion.pipelines.video.wan.wan22.ti2v import manifest, run" in source
    assert "download_models(manifest()" in source


# AC03 — pipeline-specific required flags
def test_i2v_has_required_image_flag() -> None:
    source = _I2V_SCRIPT.read_text(encoding="utf-8")
    assert "--image" in source


def test_flf2v_has_required_image_flags() -> None:
    source = _FLF2V_SCRIPT.read_text(encoding="utf-8")
    assert "--start-image" in source
    assert "--end-image" in source


def test_s2v_has_required_audio_flag() -> None:
    source = _S2V_SCRIPT.read_text(encoding="utf-8")
    assert "--audio" in source


def test_ti2v_has_required_image_flag() -> None:
    source = _TI2V_SCRIPT.read_text(encoding="utf-8")
    assert "--image" in source


# ---------------------------------------------------------------------------
# AC03 — common CLI flags present on all scripts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_script_has_common_cli_flags(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    for flag in ("--models-dir", "--prompt", "--output"):
        assert flag in source, f"{script.name} missing CLI flag: {flag}"


# ---------------------------------------------------------------------------
# AC03 — missing required args → non-zero exit + error message
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


def test_i2v_error_on_missing_image_flag() -> None:
    result = _run_script(_I2V_SCRIPT, ["--prompt", "test"])
    assert result.returncode != 0


def test_flf2v_error_on_missing_start_image_flag() -> None:
    result = _run_script(_FLF2V_SCRIPT, ["--prompt", "test"])
    assert result.returncode != 0


def test_s2v_error_on_missing_audio_flag() -> None:
    result = _run_script(_S2V_SCRIPT, ["--prompt", "test"])
    assert result.returncode != 0


def test_ti2v_error_on_missing_image_flag() -> None:
    result = _run_script(_TI2V_SCRIPT, ["--prompt", "test"])
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# AC04 — no heavy top-level imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_no_heavy_top_level_imports(script: Path) -> None:
    tree = _parse(script)
    top_level = _top_level_import_names(tree)
    forbidden_found = [
        name for name in top_level
        if any(
            name == forbidden or name.startswith(forbidden + ".")
            for forbidden in _FORBIDDEN_TOP_LEVEL
        )
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
# AC04 — structural conventions
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
    assert (
        'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source
    ), f"{script.name} must have an if __name__ == '__main__' guard"


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_save_frames_helper_defined(script: Path) -> None:
    tree = _parse(script)
    func_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    assert "_save_frames_as_video" in func_names, (
        f"{script.name} must define a _save_frames_as_video() helper"
    )


@pytest.mark.parametrize("script", _ALL_SCRIPTS)
def test_shebang_line(script: Path) -> None:
    first_line = script.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#!/usr/bin/env python"), (
        f"{script.name} must start with a shebang line"
    )
