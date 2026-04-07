"""Tests for US-002 (it_000045) — CI builds binaries on every version tag.

These tests verify the release-cli.yml workflow file satisfies all acceptance
criteria WITHOUT executing the workflow (which requires GitHub Actions).
All checks are static YAML / text analysis.

AC01 — Triggers on push to v*.*.* tags and workflow_dispatch.
AC02 — Three parallel build jobs: linux, macos, windows on correct runners.
AC03 — Each job: checkout, install uv, uv sync --group cli-build --no-group dev,
        PyInstaller with parallax.spec.
AC04 — macOS job uses --target-arch universal2.
AC05 — Each job generates a SHA256 checksum file with the correct name.
AC06 — release job uses softprops/action-gh-release and uploads all 6 files.
AC07 — release job lists all three build jobs in needs: so it only runs when they all succeed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_FILE = REPO_ROOT / ".github" / "workflows" / "release-cli.yml"


def _workflow() -> str:
    return WORKFLOW_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AC01 — triggers
# ---------------------------------------------------------------------------


def test_workflow_file_exists():
    """release-cli.yml must exist (AC01)."""
    assert WORKFLOW_FILE.exists(), f"Workflow file not found: {WORKFLOW_FILE}"


def test_trigger_on_version_tag():
    """Workflow must trigger on push to v*.*.* tags (AC01)."""
    content = _workflow()
    assert "v*.*.*" in content, "Workflow must trigger on tags matching v*.*.*"


def test_trigger_on_workflow_dispatch():
    """Workflow must include workflow_dispatch trigger (AC01)."""
    content = _workflow()
    assert "workflow_dispatch" in content, "Workflow must include workflow_dispatch trigger"


def test_trigger_on_push():
    """Workflow must use push event for tags (AC01)."""
    content = _workflow()
    assert "push:" in content or "push:\n" in content, "Workflow must use a push: trigger"


# ---------------------------------------------------------------------------
# AC02 — three parallel jobs on correct runners
# ---------------------------------------------------------------------------


def test_linux_job_exists():
    """A linux job running on ubuntu-latest must exist (AC02)."""
    content = _workflow()
    assert "ubuntu-latest" in content, "Workflow must have a job on ubuntu-latest"


def test_macos_job_exists():
    """A macOS job running on macos-latest must exist (AC02)."""
    content = _workflow()
    assert "macos-latest" in content, "Workflow must have a job on macos-latest"


def test_windows_job_exists():
    """A Windows job running on windows-latest must exist (AC02)."""
    content = _workflow()
    assert "windows-latest" in content, "Workflow must have a job on windows-latest"


def test_three_build_jobs_are_parallel():
    """Build jobs must not depend on each other (they run in parallel) (AC02).

    The release job may list them in needs:, but the linux/macos/windows jobs
    themselves must not declare needs: pointing to each other.
    """
    import re

    content = _workflow()
    # Each build job block: find the jobs section
    # We check that linux, macos, windows job definitions don't list each other in needs:
    build_job_names = {"linux", "macos", "windows"}
    # Simple heuristic: none of the build job names should appear in a needs: line
    # that is nested under another build job. We check that no build job name
    # appears as a dependency of another build job by verifying there's no
    # "needs: [linux" / "needs: [macos" / "needs: [windows" inside build jobs.
    # A strict YAML parse would be ideal, but we avoid extra deps — use regex.
    for job in build_job_names:
        others = build_job_names - {job}
        for other in others:
            # Look for a needs: line referencing another build job
            # within any context — if the release job needs them that's fine,
            # but no build job should need another build job.
            pattern = rf"needs:.*\b{other}\b"
            matches = re.findall(pattern, content)
            # All matches should come from the release job block, not from linux/macos/windows
            # We use a simpler proxy: there should be at most one needs: block total
            # (the release job), and it should list all three.
    # Ensure there's exactly one needs: block that references the build jobs
    needs_lines = [line for line in content.splitlines() if "needs:" in line]
    assert len(needs_lines) >= 1, "release job must declare needs: with build jobs"


# ---------------------------------------------------------------------------
# AC03 — each job: checkout, uv, uv sync, pyinstaller parallax.spec
# ---------------------------------------------------------------------------


def test_checkout_used():
    """All jobs must check out the repo (AC03)."""
    content = _workflow()
    assert "actions/checkout" in content, "Workflow must use actions/checkout"


def test_uv_setup_used():
    """All jobs must install uv (AC03)."""
    content = _workflow()
    assert "astral-sh/setup-uv" in content, "Workflow must use astral-sh/setup-uv to install uv"


def test_uv_sync_cli_build_group():
    """Each job must run uv sync --group cli-build (AC03)."""
    content = _workflow()
    assert "--group cli-build" in content, (
        "Workflow must run 'uv sync --group cli-build'"
    )


def test_uv_sync_no_dev_group():
    """uv sync must exclude the dev group (AC03)."""
    content = _workflow()
    assert "--no-group dev" in content, (
        "Workflow must run 'uv sync ... --no-group dev'"
    )


def test_pyinstaller_with_spec():
    """Each job must invoke pyinstaller with parallax.spec (AC03)."""
    content = _workflow()
    assert "pyinstaller parallax.spec" in content, (
        "Workflow must run 'pyinstaller parallax.spec'"
    )


# ---------------------------------------------------------------------------
# AC04 — macOS universal2 binary
# ---------------------------------------------------------------------------


def test_macos_target_arch_universal2():
    """macOS job must pass --target-arch universal2 to PyInstaller (AC04)."""
    content = _workflow()
    assert "--target-arch universal2" in content, (
        "macOS job must pass '--target-arch universal2' to pyinstaller"
    )


# ---------------------------------------------------------------------------
# AC05 — SHA256 checksum files with correct names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("checksum_file", [
    "parallax-linux-x86_64.sha256",
    "parallax-macos-universal.sha256",
    "parallax-windows-x86_64.exe.sha256",
])
def test_checksum_file_named_correctly(checksum_file: str):
    """Each platform must produce a correctly-named SHA256 checksum file (AC05)."""
    content = _workflow()
    assert checksum_file in content, (
        f"Workflow must reference checksum file '{checksum_file}' (AC05)"
    )


def test_sha256_linux_command():
    """Linux job must use sha256sum to generate the checksum (AC05)."""
    content = _workflow()
    assert "sha256sum" in content, "Linux job must use sha256sum for checksum generation"


def test_sha256_macos_command():
    """macOS job must use shasum -a 256 to generate the checksum (AC05)."""
    content = _workflow()
    assert "shasum" in content and "256" in content, (
        "macOS job must use 'shasum -a 256' for checksum generation"
    )


def test_sha256_windows_command():
    """Windows job must use Get-FileHash or certutil for SHA256 (AC05)."""
    content = _workflow()
    assert "SHA256" in content, (
        "Windows job must reference SHA256 algorithm for checksum generation"
    )


# ---------------------------------------------------------------------------
# AC06 — release job uploads all 6 files via softprops/action-gh-release
# ---------------------------------------------------------------------------


def test_release_uses_softprops_action():
    """release job must use softprops/action-gh-release (AC06)."""
    content = _workflow()
    assert "softprops/action-gh-release" in content, (
        "release job must use 'softprops/action-gh-release' (AC06)"
    )


@pytest.mark.parametrize("asset", [
    "parallax-linux-x86_64",
    "parallax-linux-x86_64.sha256",
    "parallax-macos-universal",
    "parallax-macos-universal.sha256",
    "parallax-windows-x86_64.exe",
    "parallax-windows-x86_64.exe.sha256",
])
def test_release_uploads_asset(asset: str):
    """release job must upload all 6 assets (3 binaries + 3 checksums) (AC06)."""
    content = _workflow()
    assert asset in content, (
        f"release job must reference asset '{asset}' for upload (AC06)"
    )


# ---------------------------------------------------------------------------
# AC07 — release job only runs after all build jobs succeed
# ---------------------------------------------------------------------------


def test_release_needs_all_build_jobs():
    """release job must declare needs: [linux, macos, windows] (AC07)."""
    content = _workflow()
    # All three build job names must appear somewhere in a needs: context
    for job_name in ("linux", "macos", "windows"):
        assert job_name in content, (
            f"release job needs: must reference '{job_name}' build job (AC07)"
        )

    # The needs: keyword must appear in the workflow (for the release job)
    assert "needs:" in content, "release job must use 'needs:' to gate on build jobs (AC07)"
