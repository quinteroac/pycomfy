"""Tests for US-001 (PRD-002) — Build standalone binary locally.

These tests verify the PyInstaller spec file (parallax.spec) satisfies all
acceptance criteria WITHOUT actually running PyInstaller (which is not available
in CI and would be prohibitively slow).

AC01 — parallax.spec exists and is configured for single-file output.
AC02 — The spec entry-point is the CLI main module (verified structurally).
AC03 — torch, torchvision, torchaudio, transformers, comfy_diffusion are in excludes.
AC04 — (binary size) — verified indirectly: excluded packages are what make binaries large;
        static check confirms they are in the excludes list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_FILE = REPO_ROOT / "parallax.spec"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_content() -> str:
    return SPEC_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AC01 — spec file exists and is configured for single-file output
# ---------------------------------------------------------------------------


def test_spec_file_exists():
    """parallax.spec must exist at the repository root (AC01)."""
    assert SPEC_FILE.exists(), "parallax.spec not found at repo root"


def test_spec_is_valid_python():
    """parallax.spec must be syntactically valid Python (PyInstaller DSL)."""
    import ast

    source = _spec_content()
    # Should not raise SyntaxError
    ast.parse(source)


def test_spec_defines_analysis():
    """parallax.spec must define an Analysis object (required by PyInstaller)."""
    content = _spec_content()
    assert "Analysis(" in content, "parallax.spec must define an Analysis(...) block"


def test_spec_defines_exe():
    """parallax.spec must define an EXE object (required for executable output)."""
    content = _spec_content()
    assert "EXE(" in content, "parallax.spec must define an EXE(...) block"


def test_spec_no_collect_block():
    """Single-file mode: spec must NOT use a COLLECT step.

    In PyInstaller, the absence of COLLECT means binaries/datas are folded
    directly into the EXE, producing a single self-contained file (AC01).
    """
    content = _spec_content()
    assert "COLLECT(" not in content, (
        "parallax.spec must not use COLLECT — onefile mode passes a.binaries "
        "and a.datas directly to EXE"
    )


def test_spec_exe_named_parallax():
    """The EXE must be named 'parallax' so the output is dist/parallax (AC01)."""
    content = _spec_content()
    assert 'name="parallax"' in content or "name='parallax'" in content, (
        "EXE must have name='parallax' to produce dist/parallax"
    )


# ---------------------------------------------------------------------------
# AC02 — entry-point points to the CLI main module
# ---------------------------------------------------------------------------


def test_spec_entry_point_is_cli_main():
    """Analysis entry-point must reference the CLI __main__ module (AC02)."""
    content = _spec_content()
    # Accept either cli/__main__.py or cli/main.py as valid entry points
    assert "cli/__main__.py" in content or "cli/main.py" in content, (
        "parallax.spec Analysis must reference cli/__main__.py or cli/main.py "
        "as the entry-point script"
    )


def test_cli_entry_point_file_exists():
    """The entry-point script referenced in the spec must exist on disk (AC02)."""
    content = _spec_content()
    if "cli/__main__.py" in content:
        assert (REPO_ROOT / "cli" / "__main__.py").exists()
    elif "cli/main.py" in content:
        assert (REPO_ROOT / "cli" / "main.py").exists()


# ---------------------------------------------------------------------------
# AC03 — required excludes are present
# ---------------------------------------------------------------------------


REQUIRED_EXCLUDES = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "comfy_diffusion",
]


@pytest.mark.parametrize("package", REQUIRED_EXCLUDES)
def test_spec_excludes_heavy_package(package: str):
    """Each ML package must appear in the excludes list of parallax.spec (AC03)."""
    content = _spec_content()
    # The package name must appear inside the excludes=[...] section.
    # We check that it appears as a quoted string literal.
    assert f'"{package}"' in content or f"'{package}'" in content, (
        f"'{package}' must be listed in excludes in parallax.spec (AC03)"
    )


def test_spec_excludes_list_present():
    """parallax.spec must contain an explicit excludes=[...] argument (AC03)."""
    content = _spec_content()
    assert "excludes=[" in content or "excludes = [" in content, (
        "parallax.spec must define an excludes=[...] list in the Analysis block"
    )


# ---------------------------------------------------------------------------
# AC04 — binary size guardrails (static, structural check)
# ---------------------------------------------------------------------------


def test_spec_excludes_numpy_for_size():
    """numpy is excluded to help keep the binary under 50 MB (AC04)."""
    content = _spec_content()
    assert '"numpy"' in content or "'numpy'" in content, (
        "'numpy' should be in excludes to keep binary size under 50 MB (AC04)"
    )


def test_spec_excludes_scipy_for_size():
    """scipy is excluded to help keep the binary under 50 MB (AC04)."""
    content = _spec_content()
    assert '"scipy"' in content or "'scipy'" in content, (
        "'scipy' should be in excludes to keep binary size under 50 MB (AC04)"
    )


def test_spec_console_mode_enabled():
    """The EXE must be a console app (not windowed) for CLI use."""
    content = _spec_content()
    assert "console=True" in content, (
        "EXE must set console=True for CLI operation"
    )
