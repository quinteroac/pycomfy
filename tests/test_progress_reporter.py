"""Tests for packages/parallax_cli/runtime/progress.py — US-005-AC01, AC02."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

# Make the runtime package importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "parallax_cli"))

from runtime.progress import progress


def _capture(func, *args, **kwargs) -> str:
    """Capture stdout output from a function call."""
    buf = StringIO()
    original = sys.stdout
    sys.stdout = buf
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = original
    return buf.getvalue()


def test_progress_exports_function():
    """AC01: progress() is callable and exported from runtime.progress."""
    assert callable(progress)


def test_progress_outputs_json_line(capsys):
    """AC02: progress() prints JSON with step and pct, followed by newline."""
    progress("download", 0.5)
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert data["step"] == "download"
    assert data["pct"] == 0.5


def test_progress_includes_kwargs(capsys):
    """AC02: extra kwargs are merged into the JSON object."""
    progress("done", 1.0, output="out.mp4", frames=42)
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert data["step"] == "done"
    assert data["pct"] == 1.0
    assert data["output"] == "out.mp4"
    assert data["frames"] == 42


def test_progress_ends_with_newline(capsys):
    """AC02: output ends with a newline (print default)."""
    progress("model_load", 0.0)
    captured = capsys.readouterr()
    assert captured.out.endswith("\n")


def test_progress_json_field_order(capsys):
    """AC02: step and pct appear first in the JSON object (dict ordering)."""
    progress("sampling_start", 0.1)
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    keys = list(data.keys())
    assert keys[0] == "step"
    assert keys[1] == "pct"


def test_progress_zero_pct(capsys):
    """AC02: pct=0.0 is serialized as a number."""
    progress("download", 0.0)
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert isinstance(data["pct"], float)
    assert data["pct"] == 0.0
