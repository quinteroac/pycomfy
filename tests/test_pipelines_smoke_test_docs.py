"""Tests for US-009 — GPU smoke test documentation.

Covers:
  AC01: Each pipeline's module docstring contains a "Usage" section with a
        runnable snippet (code block).
  AC02: The run() docstring lists all parameters with types and default values
        (numpy-style ``param : type`` annotations).
  AC03: SMOKE_TEST.md exists at the repo root and enumerates at least one GPU
        invocation per pipeline.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINES_DIR = _REPO_ROOT / "comfy_diffusion" / "pipelines"
_SMOKE_TEST_MD = _REPO_ROOT / "SMOKE_TEST.md"

_PIPELINE_MODULES = [
    "ltx2_t2v",
    "ltx2_t2v_distilled",
    "ltx2_i2v",
    "ltx2_i2v_distilled",
    "ltx2_i2v_lora",
    "ltx3_t2v",
    "ltx3_i2v",
]

# Maps legacy module name to relative path within _PIPELINES_DIR.
_PIPELINE_FILE_MAP: dict[str, str] = {
    "ltx2_t2v": "video/ltx/ltx2/t2v.py",
    "ltx2_t2v_distilled": "video/ltx/ltx2/t2v_distilled.py",
    "ltx2_i2v": "video/ltx/ltx2/i2v.py",
    "ltx2_i2v_distilled": "video/ltx/ltx2/i2v_distilled.py",
    "ltx2_i2v_lora": "video/ltx/ltx2/i2v_lora.py",
    "ltx3_t2v": "video/ltx/ltx3/t2v.py",
    "ltx3_i2v": "video/ltx/ltx3/i2v.py",
}


def _parse_pipeline(module_name: str) -> ast.Module:
    src = (_PIPELINES_DIR / _PIPELINE_FILE_MAP[module_name]).read_text(encoding="utf-8")
    return ast.parse(src, filename=f"{module_name}.py")


def _run_node(tree: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    raise AssertionError("run() not found in module")


def _run_params(run_node: ast.FunctionDef) -> list[tuple[str, str | None, bool]]:
    """Return list of (name, annotation_src, has_default) for all kwonly args."""
    results = []
    kwonly = run_node.args.kwonlyargs
    defaults = run_node.args.kw_defaults
    for i, arg in enumerate(kwonly):
        ann = ast.unparse(arg.annotation) if arg.annotation else None
        has_default = defaults[i] is not None
        results.append((arg.arg, ann, has_default))
    return results


# ---------------------------------------------------------------------------
# AC01 — module docstring contains a "Usage" section with a runnable snippet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", _PIPELINE_MODULES)
def test_module_docstring_has_usage_section(module_name: str) -> None:
    """AC01: module docstring must contain a 'Usage' heading."""
    tree = _parse_pipeline(module_name)
    docstring = ast.get_docstring(tree) or ""
    assert "Usage" in docstring, (
        f"{module_name}.py module docstring missing 'Usage' section"
    )


@pytest.mark.parametrize("module_name", _PIPELINE_MODULES)
def test_module_docstring_has_runnable_snippet(module_name: str) -> None:
    """AC01: module docstring must contain a fenced code block (``::`` or ```)."""
    src = (_PIPELINES_DIR / _PIPELINE_FILE_MAP[module_name]).read_text(encoding="utf-8")
    tree = ast.parse(src, filename=f"{module_name}.py")
    docstring = ast.get_docstring(tree) or ""
    # RST literal block marker OR markdown fenced block
    has_code = "::" in docstring or "```" in docstring
    assert has_code, (
        f"{module_name}.py module docstring has no runnable code snippet (:: or ```)"
    )


@pytest.mark.parametrize("module_name", _PIPELINE_MODULES)
def test_module_docstring_snippet_calls_run(module_name: str) -> None:
    """AC01: the Usage snippet must demonstrate calling run()."""
    tree = _parse_pipeline(module_name)
    docstring = ast.get_docstring(tree) or ""
    assert "run(" in docstring, (
        f"{module_name}.py Usage snippet does not demonstrate calling run()"
    )


# ---------------------------------------------------------------------------
# AC02 — run() docstring lists all parameters with types and default values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", _PIPELINE_MODULES)
def test_run_docstring_exists(module_name: str) -> None:
    """AC02: run() must have a docstring."""
    tree = _parse_pipeline(module_name)
    node = _run_node(tree)
    docstring = ast.get_docstring(node)
    assert docstring, f"{module_name}.py run() has no docstring"


@pytest.mark.parametrize("module_name", _PIPELINE_MODULES)
def test_run_docstring_has_parameters_section(module_name: str) -> None:
    """AC02: run() docstring must include a 'Parameters' section."""
    tree = _parse_pipeline(module_name)
    node = _run_node(tree)
    docstring = ast.get_docstring(node) or ""
    assert "Parameters" in docstring, (
        f"{module_name}.py run() docstring missing 'Parameters' section"
    )


@pytest.mark.parametrize("module_name", _PIPELINE_MODULES)
def test_run_docstring_lists_all_params(module_name: str) -> None:
    """AC02: every parameter must be mentioned in the run() docstring."""
    tree = _parse_pipeline(module_name)
    node = _run_node(tree)
    docstring = ast.get_docstring(node) or ""
    params = _run_params(node)
    missing = [name for name, _, _ in params if name not in docstring]
    assert not missing, (
        f"{module_name}.py run() docstring is missing parameters: {missing}"
    )


@pytest.mark.parametrize("module_name", _PIPELINE_MODULES)
def test_run_docstring_uses_typed_param_format(module_name: str) -> None:
    """AC02: run() docstring must use 'param : type' numpy format for at least
    the majority of parameters (verifies types are present)."""
    tree = _parse_pipeline(module_name)
    node = _run_node(tree)
    docstring = ast.get_docstring(node) or ""
    # Count numpy-style typed entries: "param_name : type"
    typed_count = len(re.findall(r"^\w+ : \S", docstring, re.MULTILINE))
    params = _run_params(node)
    # At least half of all parameters should have a typed entry
    assert typed_count >= len(params) // 2, (
        f"{module_name}.py run() docstring has only {typed_count} typed param entries "
        f"(expected ≥ {len(params) // 2} for {len(params)} params).  "
        "Use 'param : type' numpy format."
    )


@pytest.mark.parametrize("module_name", _PIPELINE_MODULES)
def test_run_docstring_mentions_defaults(module_name: str) -> None:
    """AC02: run() docstring must explicitly mention default values."""
    tree = _parse_pipeline(module_name)
    node = _run_node(tree)
    docstring = ast.get_docstring(node) or ""
    # Check that "Default" appears at least once (covers optional params)
    assert "Default" in docstring, (
        f"{module_name}.py run() docstring does not mention any default values.  "
        "Add 'Default ``value``.' for optional parameters."
    )


# ---------------------------------------------------------------------------
# AC03 — SMOKE_TEST.md exists and has per-pipeline GPU invocations
# ---------------------------------------------------------------------------


def test_smoke_test_md_exists() -> None:
    """AC03: SMOKE_TEST.md must exist at the repo root."""
    assert _SMOKE_TEST_MD.is_file(), (
        "SMOKE_TEST.md not found at repo root.  "
        "Create it to document manual GPU smoke tests."
    )


def test_smoke_test_md_not_empty() -> None:
    """AC03: SMOKE_TEST.md must not be empty."""
    content = _SMOKE_TEST_MD.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, "SMOKE_TEST.md is empty"


@pytest.mark.parametrize("module_name", _PIPELINE_MODULES)
def test_smoke_test_md_covers_pipeline(module_name: str) -> None:
    """AC03: SMOKE_TEST.md must reference each pipeline module."""
    content = _SMOKE_TEST_MD.read_text(encoding="utf-8")
    assert module_name in content, (
        f"SMOKE_TEST.md does not mention pipeline '{module_name}'.  "
        "Add a GPU invocation example for this pipeline."
    )


def test_smoke_test_md_has_code_blocks() -> None:
    """AC03: SMOKE_TEST.md must contain at least one fenced code block."""
    content = _SMOKE_TEST_MD.read_text(encoding="utf-8")
    assert "```python" in content, (
        "SMOKE_TEST.md must contain at least one ```python code block "
        "showing a runnable GPU invocation."
    )


def test_smoke_test_md_shows_run_calls() -> None:
    """AC03: SMOKE_TEST.md must call run() in its examples."""
    content = _SMOKE_TEST_MD.read_text(encoding="utf-8")
    assert "run(" in content, (
        "SMOKE_TEST.md must demonstrate calling run() from a pipeline."
    )
