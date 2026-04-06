"""
Tests for install.ps1 — US-002 (it_000045)
All tests are static: they parse/inspect the PowerShell script without executing it.
This is appropriate for CI (no network, no side effects).
"""

import re
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "install.ps1"


def _src() -> str:
    return SCRIPT.read_text()


# ── AC01 — Always downloads x86_64 asset ──────────────────────────────────

class TestAC01Asset:
    def test_script_exists(self):
        assert SCRIPT.exists(), "install.ps1 does not exist"

    def test_asset_is_windows_x86_64(self):
        assert "parallax-windows-x86_64.exe" in _src()

    def test_no_arch_branching(self):
        src = _src()
        # Only x86_64 is supported — no arm/aarch64 branching
        assert "parallax-windows-arm" not in src
        assert "parallax-windows-aarch64" not in src


# ── AC02 — Version resolution ──────────────────────────────────────────────

class TestAC02VersionResolution:
    def test_github_api_url(self):
        src = _src()
        assert "api.github.com/repos" in src
        assert "releases/latest" in src

    def test_parallax_version_env_respected(self):
        assert "PARALLAX_VERSION" in _src()

    def test_env_var_takes_precedence(self):
        src = _src()
        # PARALLAX_VERSION must be checked before the API call
        idx_env = src.index("PARALLAX_VERSION")
        idx_api = src.index("releases/latest")
        assert idx_env < idx_api

    def test_repo_variable_defined(self):
        src = _src()
        assert "REPO" in src
        assert "quinteroac/comfy-diffusion" in src

    def test_tag_name_extracted(self):
        src = _src()
        assert "tag_name" in src


# ── AC03 — Download with Invoke-WebRequest ────────────────────────────────

class TestAC03Download:
    def test_invoke_web_request_used(self):
        assert "Invoke-WebRequest" in _src()

    def test_binary_url_constructed(self):
        src = _src()
        assert "BinaryUrl" in src or "ASSET" in src

    def test_sha256_checksum_downloaded(self):
        assert ".sha256" in _src()

    def test_outfile_used_for_binary(self):
        src = _src()
        assert re.search(r"Invoke-WebRequest.*-OutFile", src, re.DOTALL)

    def test_progress_bar_visible(self):
        src = _src()
        # ProgressPreference = "Continue" ensures the progress bar is shown
        assert "ProgressPreference" in src
        assert '"Continue"' in src or "'Continue'" in src


# ── AC04 — Checksum verification ──────────────────────────────────────────

class TestAC04Checksum:
    def test_get_file_hash_used(self):
        assert "Get-FileHash" in _src()

    def test_sha256_algorithm_specified(self):
        src = _src()
        assert re.search(r"Get-FileHash.*SHA256|SHA256.*Get-FileHash", src, re.DOTALL)

    def test_failure_message_exact(self):
        assert "Checksum verification failed. Aborting." in _src()

    def test_exits_with_code_1_on_failure(self):
        src = _src()
        # exit 1 must appear after the checksum mismatch check
        assert "exit 1" in src

    def test_deletes_file_on_failure(self):
        src = _src()
        assert "Remove-Item" in src

    def test_hash_comparison(self):
        src = _src()
        # The computed hash is compared to the expected hash
        assert re.search(r"ActualHash|ACTUAL_HASH|fileHash", src, re.IGNORECASE)
        assert re.search(r"ExpectedHash|EXPECTED_HASH", src, re.IGNORECASE)


# ── AC05 — Install to APPDATA\parallax\bin ────────────────────────────────

class TestAC05Install:
    def test_appdata_env_used(self):
        assert "APPDATA" in _src()

    def test_install_subdir_is_parallax_bin(self):
        src = _src()
        assert r"parallax\bin" in src or "parallax/bin" in src

    def test_install_path_is_parallax_exe(self):
        assert "parallax.exe" in _src()

    def test_creates_directory_if_needed(self):
        src = _src()
        assert "New-Item" in src
        assert "Directory" in src

    def test_copies_binary_to_install_path(self):
        src = _src()
        assert "Copy-Item" in src


# ── AC06 — PATH update ────────────────────────────────────────────────────

class TestAC06Path:
    def test_set_environment_variable_used(self):
        assert "SetEnvironmentVariable" in _src()

    def test_user_scope(self):
        src = _src()
        assert '"User"' in src or "'User'" in src

    def test_checks_path_before_adding(self):
        src = _src()
        # Guard against adding duplicate entry
        assert re.search(r"-notlike\s+|notcontains|-not.*PATH", src, re.IGNORECASE)

    def test_install_dir_variable_used(self):
        assert "INSTALL_DIR" in _src()

    def test_get_environment_variable_called(self):
        src = _src()
        assert "GetEnvironmentVariable" in src


# ── AC07 — Success message ────────────────────────────────────────────────

class TestAC07SuccessMessage:
    def test_installed_message_exact_text(self):
        assert "installed. Open a new terminal and run: parallax install" in _src()

    def test_bare_version_in_message(self):
        # BareVersion (stripped of leading 'v') is used in the success line
        assert "BareVersion" in _src()

    def test_version_stripping(self):
        src = _src()
        # Leading 'v' must be stripped (TrimStart)
        assert "TrimStart" in src or "Replace" in src


# ── General / structural ──────────────────────────────────────────────────

class TestGeneralStructure:
    def test_error_action_preference_stop(self):
        src = _src()
        assert "ErrorActionPreference" in src
        assert '"Stop"' in src or "'Stop'" in src

    def test_no_posix_shebang(self):
        assert "#!/bin/sh" not in _src()

    def test_no_set_e(self):
        assert "set -e" not in _src()

    def test_repo_variable_before_api_call(self):
        src = _src()
        idx_repo = src.index("REPO")
        idx_api = src.index("api.github.com")
        assert idx_repo < idx_api

    def test_install_dir_variable_before_use(self):
        src = _src()
        idx_def = src.index("INSTALL_DIR")
        idx_use = src.index("INSTALL_PATH")
        assert idx_def < idx_use
