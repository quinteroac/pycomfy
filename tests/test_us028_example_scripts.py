"""Tests for US-008 — example scripts for each pipeline.

Covers:
  AC01: examples/video/ltx/ltx2/canny.py exists and follows the download+run pattern.
  AC02: examples/video/ltx/ltx2/depth.py exists and follows the download+run pattern.
  AC03: examples/video/ltx/ltx2/pose.py exists and follows the download+run pattern.
  AC04: examples/video/ltx/ltx23/ia2v.py exists and follows the download+run pattern.
  AC05: Each script accepts CLI flags --models-dir, --prompt, and pipeline-specific flags.
  AC06: Each script prints a clear error and usage hint if required arguments are missing.
  AC07: No script imports at module top level — all heavy imports inside main().
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
_EXAMPLES_DIR = _REPO_ROOT / "examples" / "video" / "ltx"

_CANNY_SCRIPT = _EXAMPLES_DIR / "ltx2" / "canny.py"
_DEPTH_SCRIPT = _EXAMPLES_DIR / "ltx2" / "depth.py"
_POSE_SCRIPT = _EXAMPLES_DIR / "ltx2" / "pose.py"
_IA2V_SCRIPT = _EXAMPLES_DIR / "ltx23" / "ia2v.py"

_ALL_SCRIPTS = [
    pytest.param(_CANNY_SCRIPT, id="canny"),
    pytest.param(_DEPTH_SCRIPT, id="depth"),
    pytest.param(_POSE_SCRIPT, id="pose"),
    pytest.param(_IA2V_SCRIPT, id="ia2v"),
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
# AC01–AC04 — files exist
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


# AC01 — canny.py calls download_models(manifest()) and run()
def test_canny_calls_download_models_and_run() -> None:
    source = _CANNY_SCRIPT.read_text(encoding="utf-8")
    assert "download_models(manifest()" in source
    assert "from comfy_diffusion.pipelines.video.ltx.ltx2.canny import manifest, run" in source
    assert "run(" in source


# AC02 — depth.py calls download_models(manifest()) and run()
def test_depth_calls_download_models_and_run() -> None:
    source = _DEPTH_SCRIPT.read_text(encoding="utf-8")
    assert "download_models(manifest()" in source
    assert "from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest, run" in source
    assert "run(" in source


# AC03 — pose.py calls download_models(manifest()) and run()
def test_pose_calls_download_models_and_run() -> None:
    source = _POSE_SCRIPT.read_text(encoding="utf-8")
    assert "download_models(manifest()" in source
    assert "from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest, run" in source
    assert "run(" in source


# AC04 — ia2v.py calls download_models(manifest()) and run() with image+audio
def test_ia2v_calls_download_models_and_run() -> None:
    source = _IA2V_SCRIPT.read_text(encoding="utf-8")
    assert "download_models(manifest()" in source
    assert "from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest, run" in source
    assert "run(" in source
    # ia2v is unique in requiring both image and audio_path args
    assert "audio_path=" in source
    assert "image=" in source


# ---------------------------------------------------------------------------
# AC05 — CLI flags: --models-dir, --prompt, pipeline-specific flags
# ---------------------------------------------------------------------------


def test_canny_has_required_cli_flags() -> None:
    source = _CANNY_SCRIPT.read_text(encoding="utf-8")
    for flag in ("--models-dir", "--prompt", "--video", "--cfg-pass1", "--cfg-pass2",
                 "--canny-lora-strength", "--seed", "--output"):
        assert flag in source, f"canny.py missing CLI flag: {flag}"


def test_depth_has_required_cli_flags() -> None:
    source = _DEPTH_SCRIPT.read_text(encoding="utf-8")
    for flag in ("--models-dir", "--prompt", "--video", "--cfg-pass1", "--cfg-pass2",
                 "--depth-lora-strength", "--seed", "--output",
                 "--lotus-model-filename", "--lotus-vae-filename"):
        assert flag in source, f"depth.py missing CLI flag: {flag}"


def test_pose_has_required_cli_flags() -> None:
    source = _POSE_SCRIPT.read_text(encoding="utf-8")
    for flag in ("--models-dir", "--prompt", "--video", "--first-frame",
                 "--cfg-pass1", "--cfg-pass2", "--pose-lora-strength", "--seed", "--output"):
        assert flag in source, f"pose.py missing CLI flag: {flag}"


def test_ia2v_has_required_cli_flags() -> None:
    source = _IA2V_SCRIPT.read_text(encoding="utf-8")
    for flag in ("--models-dir", "--prompt", "--image", "--audio",
                 "--cfg", "--audio-start-time", "--audio-duration",
                 "--guide-strength-pass1", "--guide-strength-pass2",
                 "--distilled-lora-strength", "--te-lora-strength", "--seed", "--output"):
        assert flag in source, f"ia2v.py missing CLI flag: {flag}"


# ---------------------------------------------------------------------------
# AC06 — error + usage hint when required args are missing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script,missing_args_label", [
    (_CANNY_SCRIPT, "no --models-dir and no --video"),
    (_DEPTH_SCRIPT, "no --models-dir and no --video"),
    (_POSE_SCRIPT, "no --models-dir and no --video"),
    (_IA2V_SCRIPT, "no --models-dir and no --image and no --audio"),
])
def test_script_exits_nonzero_with_no_args(
    script: Path, missing_args_label: str
) -> None:
    result = _run_script(script)
    assert result.returncode != 0, (
        f"{script.name} should exit non-zero when called with no arguments ({missing_args_label})"
    )


def test_canny_error_message_on_missing_models_dir(tmp_path: Path) -> None:
    # --video provided but no --models-dir
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()
    result = _run_script(_CANNY_SCRIPT, ["--video", str(dummy_video), "--prompt", "test"])
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "error" in combined.lower() or "usage" in combined.lower(), (
        "canny.py must print 'error' or 'usage' when --models-dir is absent"
    )


def test_depth_error_message_on_missing_models_dir(tmp_path: Path) -> None:
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()
    result = _run_script(_DEPTH_SCRIPT, ["--video", str(dummy_video), "--prompt", "test"])
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "error" in combined.lower() or "usage" in combined.lower()


def test_pose_error_message_on_missing_models_dir(tmp_path: Path) -> None:
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()
    result = _run_script(_POSE_SCRIPT, ["--video", str(dummy_video), "--prompt", "test"])
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "error" in combined.lower() or "usage" in combined.lower()


def test_ia2v_error_message_on_missing_models_dir(tmp_path: Path) -> None:
    dummy_img = tmp_path / "frame.png"
    dummy_audio = tmp_path / "audio.wav"
    dummy_img.touch()
    dummy_audio.touch()
    result = _run_script(
        _IA2V_SCRIPT,
        ["--image", str(dummy_img), "--audio", str(dummy_audio), "--prompt", "test"],
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "error" in combined.lower() or "usage" in combined.lower()


# ---------------------------------------------------------------------------
# AC07 — no heavy imports at module top level
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
    # The pipeline import must appear inside main(), not at top level.
    # We check that the import line contains leading whitespace (indented).
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
