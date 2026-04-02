"""Tests for examples/image/edit/qwen/edit_2511.py — Qwen Image Edit 2511 CLI script.

Covers:
  - AC01: script exists and the --download-only flag exits with code 0.
  - AC02: download_models(manifest()) is called so all four model files are
          downloaded (or skipped if present).
  - AC03: script prints "Models ready." and exits without performing inference
          when --download-only is given.
  - AC04: no heavy imports at module top level; script parses without errors.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "examples" / "image" / "edit" / "qwen" / "edit_2511.py"

# Heavy modules that must NOT be imported at the module top level.
_FORBIDDEN_TOP_LEVEL = {
    "torch",
    "comfy",
    "comfy_diffusion.downloader",
    "comfy_diffusion.pipelines",
    "PIL",
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


def _run_script(extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(_SCRIPT)] + (extra_args or [])
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# AC04 — structural checks (file exists, parses, no heavy top-level imports)
# ---------------------------------------------------------------------------


def test_script_file_exists() -> None:
    assert _SCRIPT.is_file(), f"example script missing: {_SCRIPT}"


def test_script_parses_without_syntax_errors() -> None:
    tree = _parse(_SCRIPT)
    assert isinstance(tree, ast.Module)


def test_script_has_module_docstring() -> None:
    tree = _parse(_SCRIPT)
    docstring = ast.get_docstring(tree)
    assert docstring, f"{_SCRIPT.name} must have a module-level docstring"


def test_script_has_future_annotations() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "from __future__ import annotations" in source


def test_no_heavy_top_level_imports() -> None:
    tree = _parse(_SCRIPT)
    top_level = _top_level_import_names(tree)
    forbidden_found = [
        name for name in top_level
        if any(
            name == forbidden or name.startswith(forbidden + ".")
            for forbidden in _FORBIDDEN_TOP_LEVEL
        )
    ]
    assert not forbidden_found, (
        f"{_SCRIPT.name} has forbidden top-level imports: {forbidden_found}. "
        "All heavy imports must be inside main()."
    )


def test_pipeline_import_inside_main() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    lines = [
        line for line in source.splitlines()
        if "comfy_diffusion.pipelines" in line and line.startswith("    ")
    ]
    assert lines, f"{_SCRIPT.name}: pipeline import must be inside main() (indented)"


def test_downloader_import_inside_main() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    lines = [
        line for line in source.splitlines()
        if "comfy_diffusion.downloader" in line and line.startswith("    ")
    ]
    assert lines, f"{_SCRIPT.name}: downloader import must be inside main() (indented)"


def test_main_function_defined() -> None:
    tree = _parse(_SCRIPT)
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "main" in func_names, f"{_SCRIPT.name} must define a main() function"


def test_if_name_main_guard() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source


# ---------------------------------------------------------------------------
# AC05 — required CLI flags present
# ---------------------------------------------------------------------------


def test_has_required_cli_flags() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    for flag in ("--models-dir", "--image", "--prompt", "--download-only", "--seed", "--output"):
        assert flag in source, f"{_SCRIPT.name} missing CLI flag: {flag}"


# ---------------------------------------------------------------------------
# AC01 — --download-only exits with code 0
# ---------------------------------------------------------------------------


def test_download_only_exits_zero(tmp_path: Path) -> None:
    """AC01: --download-only exits with code 0 given a valid models dir."""
    # Patch download_models so no real network call is made.
    fake_image = tmp_path / "placeholder.png"
    fake_image.write_bytes(b"")  # placeholder — download-only doesn't read the image

    patch_target_download = (
        "comfy_diffusion.downloader.download_models"
    )
    # We run a subprocess but inject a monkeypatch via import manipulation.
    # Since we can't easily monkeypatch inside a subprocess, we test through
    # direct main() invocation with patching instead.
    import importlib.util
    spec = importlib.util.spec_from_file_location("edit_2511_example", _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    with (
        patch("sys.argv", [
            str(_SCRIPT),
            "--models-dir", str(tmp_path),
            "--image", str(fake_image),
            "--prompt", "x",
            "--download-only",
        ]),
        patch("comfy_diffusion.downloader.download_models", return_value=None),
        patch(
            "comfy_diffusion.pipelines.image.qwen.edit_2511.manifest",
            return_value=[],
        ),
    ):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        exit_code = mod.main()

    assert exit_code == 0, f"--download-only must exit 0, got {exit_code}"


# ---------------------------------------------------------------------------
# AC02 — download_models(manifest()) is called
# ---------------------------------------------------------------------------


def test_download_only_calls_download_models(tmp_path: Path) -> None:
    """AC02: download_models is called with the result of manifest()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("edit_2511_example", _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    fake_entries = [MagicMock(name=f"entry_{i}") for i in range(4)]
    mock_download = MagicMock(return_value=None)

    with (
        patch("sys.argv", [
            str(_SCRIPT),
            "--models-dir", str(tmp_path),
            "--download-only",
        ]),
        patch("comfy_diffusion.downloader.download_models", mock_download),
        patch(
            "comfy_diffusion.pipelines.image.qwen.edit_2511.manifest",
            return_value=fake_entries,
        ),
    ):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.main()

    mock_download.assert_called_once()
    call_args = mock_download.call_args
    # First positional arg should be the manifest entries list.
    assert call_args.args[0] == fake_entries


# ---------------------------------------------------------------------------
# AC03 — "Models ready." is printed; no inference when --download-only
# ---------------------------------------------------------------------------


def test_download_only_prints_models_ready(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """AC03: 'Models ready.' is printed on stdout."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("edit_2511_example", _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    with (
        patch("sys.argv", [
            str(_SCRIPT),
            "--models-dir", str(tmp_path),
            "--download-only",
        ]),
        patch("comfy_diffusion.downloader.download_models", return_value=None),
        patch(
            "comfy_diffusion.pipelines.image.qwen.edit_2511.manifest",
            return_value=[],
        ),
    ):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        exit_code = mod.main()

    captured = capsys.readouterr()
    assert "Models ready." in captured.out, (
        f"'Models ready.' must be printed on stdout; got: {captured.out!r}"
    )
    assert exit_code == 0


def test_download_only_does_not_call_run(tmp_path: Path) -> None:
    """AC03: run() is NOT called when --download-only is set."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("edit_2511_example", _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    mock_run = MagicMock(return_value=[MagicMock()])

    with (
        patch("sys.argv", [
            str(_SCRIPT),
            "--models-dir", str(tmp_path),
            "--download-only",
        ]),
        patch("comfy_diffusion.downloader.download_models", return_value=None),
        patch("comfy_diffusion.pipelines.image.qwen.edit_2511.manifest", return_value=[]),
        patch("comfy_diffusion.pipelines.image.qwen.edit_2511.run", mock_run),
    ):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.main()

    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_models_dir_exits_nonzero(tmp_path: Path) -> None:
    """Script exits non-zero when --models-dir is missing or invalid."""
    result = _run_script(["--download-only"])
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "error" in combined.lower() or "usage" in combined.lower()


def test_missing_image_for_inference_exits_nonzero(tmp_path: Path) -> None:
    """Script exits non-zero when --image is absent during inference mode."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("edit_2511_example", _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    with (
        patch("sys.argv", [
            str(_SCRIPT),
            "--models-dir", str(tmp_path),
            "--prompt", "edit this",
            # no --image, no --download-only
        ]),
        patch("comfy_diffusion.downloader.download_models", return_value=None),
        patch("comfy_diffusion.pipelines.image.qwen.edit_2511.manifest", return_value=[]),
        patch("comfy_diffusion.runtime.check_runtime", return_value={"python_version": "3.12"}),
    ):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        exit_code = mod.main()

    assert exit_code != 0
