"""Tests for US-005: ProgressReporter helper."""

from __future__ import annotations

import io
import json
import sys

import pytest

from comfy_diffusion.progress import ProgressReporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_stdout(fn):
    """Run fn() and return all printed lines."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn()
    finally:
        sys.stdout = old
    return buf.getvalue()


# ---------------------------------------------------------------------------
# US-005-AC01: ProgressReporter exists with correct interface
# ---------------------------------------------------------------------------


def test_progress_reporter_importable():
    reporter = ProgressReporter()
    assert hasattr(reporter, "update")
    assert hasattr(reporter, "done")


def test_update_callable_with_step_and_pct():
    reporter = ProgressReporter()
    # Must not raise
    _capture_stdout(lambda: reporter.update("loading", 10.0))


def test_done_callable_with_output_path():
    reporter = ProgressReporter()
    _capture_stdout(lambda: reporter.done("/tmp/output.mp4"))


# ---------------------------------------------------------------------------
# US-005-AC02: update() prints a single valid JSON line matching PythonProgress
# ---------------------------------------------------------------------------


def test_update_prints_valid_json():
    reporter = ProgressReporter()
    output = _capture_stdout(lambda: reporter.update("sampling", 50.0))
    lines = [l for l in output.splitlines() if l.strip()]
    assert len(lines) == 1
    data = json.loads(lines[0])  # must not raise


def test_update_contains_step_and_pct():
    reporter = ProgressReporter()
    output = _capture_stdout(lambda: reporter.update("encoding", 25.0))
    data = json.loads(output.strip())
    assert data["step"] == "encoding"
    assert data["pct"] == 25.0


def test_update_optional_frame_kwarg():
    reporter = ProgressReporter()
    output = _capture_stdout(lambda: reporter.update("decoding", 75.0, frame=12))
    data = json.loads(output.strip())
    assert data["frame"] == 12


def test_update_optional_total_kwarg():
    reporter = ProgressReporter()
    output = _capture_stdout(lambda: reporter.update("decoding", 75.0, total=100))
    data = json.loads(output.strip())
    assert data["total"] == 100


def test_update_optional_error_kwarg():
    reporter = ProgressReporter()
    output = _capture_stdout(lambda: reporter.update("failed", 0.0, error="oops"))
    data = json.loads(output.strip())
    assert data["error"] == "oops"


def test_update_excludes_absent_optional_fields():
    reporter = ProgressReporter()
    output = _capture_stdout(lambda: reporter.update("start", 0.0))
    data = json.loads(output.strip())
    # Fields not passed must not appear
    for field in ("frame", "total", "output", "error"):
        assert field not in data


def test_update_single_line_only():
    reporter = ProgressReporter()
    output = _capture_stdout(lambda: reporter.update("step", 33.0))
    lines = [l for l in output.splitlines() if l.strip()]
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# US-005-AC03: done() prints final PythonProgress with pct=100.0 and output
# ---------------------------------------------------------------------------


def test_done_pct_is_100():
    reporter = ProgressReporter()
    output = _capture_stdout(lambda: reporter.done("/out/file.mp4"))
    data = json.loads(output.strip())
    assert data["pct"] == 100.0


def test_done_output_matches_path():
    reporter = ProgressReporter()
    path = "/some/path/result.png"
    output = _capture_stdout(lambda: reporter.done(path))
    data = json.loads(output.strip())
    assert data["output"] == path


def test_done_valid_json():
    reporter = ProgressReporter()
    output = _capture_stdout(lambda: reporter.done("/a/b/c.wav"))
    json.loads(output.strip())  # must not raise


def test_done_single_line():
    reporter = ProgressReporter()
    output = _capture_stdout(lambda: reporter.done("/x.png"))
    lines = [l for l in output.splitlines() if l.strip()]
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# US-005-AC04: Existing code unaffected
# ---------------------------------------------------------------------------


def test_comfy_diffusion_still_importable():
    import comfy_diffusion  # noqa: F401
    assert hasattr(comfy_diffusion, "check_runtime")


def test_progress_not_in_top_level_namespace():
    import comfy_diffusion
    # ProgressReporter is NOT re-exported at package level
    assert not hasattr(comfy_diffusion, "ProgressReporter")


def test_multiple_updates_each_one_line():
    reporter = ProgressReporter()
    lines_captured = []

    def run():
        for i in range(5):
            reporter.update(f"step{i}", float(i * 20))

    output = _capture_stdout(run)
    lines = [l for l in output.splitlines() if l.strip()]
    assert len(lines) == 5
    for line in lines:
        json.loads(line)  # each must be valid JSON
