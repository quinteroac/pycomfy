"""Tests for US-005 — comfy_diffusion/pipelines/ltx2_t2v.py reference pipeline.

Covers:
  AC01: comfy_diffusion/pipelines/ package exists with __init__.py
  AC02: ltx2_t2v.py exists with manifest() and run()
  AC03: from comfy_diffusion.pipelines.ltx2_t2v import manifest, run works
  AC04: download_models(manifest()) with all files present completes idempotently
  AC05: module-level docstring explains the pattern
  AC06: file parses without syntax errors (typecheck / lint proxy)
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINES_PKG = _REPO_ROOT / "comfy_diffusion" / "pipelines"
_LTX2_T2V = _PIPELINES_PKG / "ltx2_t2v.py"


# ---------------------------------------------------------------------------
# AC01 — pipelines package structure
# ---------------------------------------------------------------------------


def test_pipelines_dir_exists() -> None:
    assert _PIPELINES_PKG.is_dir(), "comfy_diffusion/pipelines/ directory must exist"


def test_pipelines_init_exists() -> None:
    init = _PIPELINES_PKG / "__init__.py"
    assert init.is_file(), "comfy_diffusion/pipelines/__init__.py must exist"


def test_pipelines_init_importable() -> None:
    import comfy_diffusion.pipelines  # noqa: F401


# ---------------------------------------------------------------------------
# AC02 — ltx2_t2v.py exists with manifest() and run()
# ---------------------------------------------------------------------------


def test_ltx2_t2v_file_exists() -> None:
    assert _LTX2_T2V.is_file(), "comfy_diffusion/pipelines/ltx2_t2v.py must exist"


def test_ltx2_t2v_has_manifest_function() -> None:
    source = _LTX2_T2V.read_text(encoding="utf-8")
    assert "def manifest(" in source, "ltx2_t2v.py must define manifest()"


def test_ltx2_t2v_has_run_function() -> None:
    source = _LTX2_T2V.read_text(encoding="utf-8")
    assert "def run(" in source, "ltx2_t2v.py must define run()"


def test_ltx2_t2v_exports_manifest_and_run() -> None:
    source = _LTX2_T2V.read_text(encoding="utf-8")
    assert '"manifest"' in source or "'manifest'" in source, (
        "manifest must appear in __all__"
    )
    assert '"run"' in source or "'run'" in source, (
        "run must appear in __all__"
    )


# ---------------------------------------------------------------------------
# AC03 — import works without error
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.ltx2_t2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# AC04 — manifest() returns a non-empty list; download_models idempotent
# ---------------------------------------------------------------------------


def test_manifest_returns_list() -> None:
    from comfy_diffusion.pipelines.ltx2_t2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) > 0, "manifest() must return at least one entry"


def test_manifest_entries_are_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry, URLModelEntry, CivitAIModelEntry
    from comfy_diffusion.pipelines.ltx2_t2v import manifest

    valid_types = (HFModelEntry, URLModelEntry, CivitAIModelEntry)
    for entry in manifest():
        assert isinstance(entry, valid_types), (
            f"manifest() entry {entry!r} is not a ModelEntry subtype"
        )


def test_manifest_entries_have_dest() -> None:
    from comfy_diffusion.pipelines.ltx2_t2v import manifest

    for entry in manifest():
        assert hasattr(entry, "dest") and entry.dest, (
            f"manifest() entry {entry!r} must have a non-empty dest"
        )


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    """download_models(manifest()) completes without error when files are present."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.ltx2_t2v import manifest

    entries = manifest()

    # Pre-create all destination files so no download is attempted.
    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    # Should complete silently — all files already present.
    download_models(entries, models_dir=tmp_path, quiet=True)


def test_download_models_idempotent_runs_twice(tmp_path: Path) -> None:
    """Calling download_models twice with all files present is safe."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.ltx2_t2v import manifest

    entries = manifest()

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
    download_models(entries, models_dir=tmp_path, quiet=True)  # second call — no error


# ---------------------------------------------------------------------------
# AC05 — module-level docstring explains the pattern
# ---------------------------------------------------------------------------


def test_module_has_docstring() -> None:
    source = _LTX2_T2V.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_LTX2_T2V))
    docstring = ast.get_docstring(tree)
    assert docstring, "ltx2_t2v.py must have a module-level docstring"


def test_docstring_mentions_manifest_and_run_pattern() -> None:
    source = _LTX2_T2V.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_LTX2_T2V))
    docstring = ast.get_docstring(tree) or ""
    assert "manifest" in docstring.lower(), (
        "Module docstring must mention manifest()"
    )
    assert "run" in docstring.lower(), (
        "Module docstring must mention run()"
    )


# ---------------------------------------------------------------------------
# AC06 — file parses without syntax errors; no top-level comfy imports
# ---------------------------------------------------------------------------


def test_ltx2_t2v_parses_without_syntax_errors() -> None:
    source = _LTX2_T2V.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_LTX2_T2V))
    assert isinstance(tree, ast.Module)


def test_no_top_level_comfy_imports() -> None:
    """comfy.* (ComfyUI internals) must not be imported at module top level."""
    source = _LTX2_T2V.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        # Only flag bare ComfyUI internals (comfy.*), not comfy_diffusion.*
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), (
                f"Top-level comfy import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


def test_pipelines_init_parses_without_syntax_errors() -> None:
    source = (_PIPELINES_PKG / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="__init__.py")
    assert isinstance(tree, ast.Module)


def test_ltx2_t2v_has_future_annotations() -> None:
    source = _LTX2_T2V.read_text(encoding="utf-8")
    assert "from __future__ import annotations" in source
