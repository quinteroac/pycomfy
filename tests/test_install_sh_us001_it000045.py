"""
Tests for install.sh — US-001 (it_000045 PRD_003)
All tests are static: they parse/inspect the shell script without executing it.
This is appropriate for CI (no network, no side effects).
"""

import re
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "install.sh"


def _src() -> str:
    return SCRIPT.read_text()


# ── AC01 — OS/arch detection ───────────────────────────────────────────────

class TestAC01OsArchDetection:
    def test_uses_uname_s(self):
        assert 'uname -s' in _src()

    def test_uses_uname_m(self):
        assert 'uname -m' in _src()

    def test_linux_x86_64_asset(self):
        src = _src()
        assert 'parallax-linux-x86_64' in src

    def test_macos_universal_asset(self):
        src = _src()
        assert 'parallax-macos-universal' in src

    def test_linux_case_branch(self):
        src = _src()
        # Linux branch must assign parallax-linux-x86_64
        assert re.search(r'Linux\)', src)
        assert re.search(r'x86_64\)', src)

    def test_darwin_case_branch(self):
        src = _src()
        assert re.search(r'Darwin\)', src)


# ── AC02 — Version resolution ──────────────────────────────────────────────

class TestAC02VersionResolution:
    def test_github_api_url(self):
        assert 'api.github.com/repos' in _src()
        assert 'releases/latest' in _src()

    def test_parallax_version_env_respected(self):
        src = _src()
        assert 'PARALLAX_VERSION' in src

    def test_env_var_takes_precedence(self):
        src = _src()
        # Must check PARALLAX_VERSION before making the API call
        idx_env = src.index('PARALLAX_VERSION')
        idx_api = src.index('releases/latest')
        assert idx_env < idx_api

    def test_repo_variable_defined(self):
        src = _src()
        assert 'REPO=' in src
        assert 'quinteroac/comfy-diffusion' in src


# ── AC03 — Download with curl / wget ──────────────────────────────────────

class TestAC03Download:
    def test_curl_used_preferentially(self):
        src = _src()
        # curl is checked first
        assert re.search(r'command -v curl', src)
        # download uses redirect-following flag (-L or -fL or -fsSL)
        assert re.search(r'curl\s+-\S*[Ll]', src)

    def test_wget_fallback_present(self):
        src = _src()
        assert re.search(r'command -v wget', src)

    def test_both_binary_and_checksum_downloaded(self):
        src = _src()
        assert '.sha256' in src
        # Asset variable used for both
        assert src.count('ASSET}') >= 2 or src.count('$ASSET') >= 2

    def test_progress_indicator_curl(self):
        src = _src()
        # --progress-bar shows progress in curl
        assert '--progress-bar' in src

    def test_progress_indicator_wget(self):
        src = _src()
        # --show-progress shows progress in wget
        assert '--show-progress' in src


# ── AC04 — Checksum verification ──────────────────────────────────────────

class TestAC04Checksum:
    def test_sha256sum_on_linux(self):
        src = _src()
        assert 'sha256sum' in src

    def test_shasum_on_macos(self):
        src = _src()
        assert 'shasum -a 256' in src

    def test_failure_message(self):
        src = _src()
        assert 'Checksum verification failed. Aborting.' in src

    def test_exits_on_failure(self):
        src = _src()
        # error() helper calls exit 1
        assert 'exit 1' in src

    def test_deletes_file_on_failure(self):
        src = _src()
        assert re.search(r'rm -f.*TMP_BINARY|rm -f.*\$TMP_BINARY', src)


# ── AC05 — Install to ~/.local/bin ────────────────────────────────────────

class TestAC05Install:
    def test_default_install_dir(self):
        src = _src()
        assert '.local/bin' in src

    def test_mkdir_p(self):
        src = _src()
        assert 'mkdir -p' in src

    def test_chmod_x(self):
        src = _src()
        assert 'chmod +x' in src

    def test_binary_named_parallax(self):
        src = _src()
        assert "BINARY_NAME=\"parallax\"" in src or "BINARY_NAME='parallax'" in src

    def test_parallax_install_dir_override(self):
        src = _src()
        assert 'PARALLAX_INSTALL_DIR' in src


# ── AC06 — PATH guidance ──────────────────────────────────────────────────

class TestAC06PathGuidance:
    def test_path_check_present(self):
        src = _src()
        assert 'PATH' in src

    def test_export_line_shown(self):
        src = _src()
        assert 'export PATH' in src

    def test_bashrc_mentioned(self):
        src = _src()
        assert '.bashrc' in src

    def test_zshrc_mentioned(self):
        src = _src()
        assert '.zshrc' in src

    def test_profile_mentioned(self):
        src = _src()
        assert '.profile' in src

    def test_new_terminal_instruction(self):
        src = _src()
        assert 'new terminal' in src


# ── AC07 — Success message ────────────────────────────────────────────────

class TestAC07SuccessMessage:
    def test_installed_message_format(self):
        src = _src()
        assert 'installed. Run: parallax install' in src

    def test_version_in_message(self):
        src = _src()
        # The bare version variable is used in the success line
        assert 'BARE_VERSION' in src


# ── AC08 — Update detection ───────────────────────────────────────────────

class TestAC08UpdateDetection:
    def test_existing_binary_detected(self):
        src = _src()
        assert 'EXISTING_VERSION' in src

    def test_updating_message(self):
        src = _src()
        assert 'Updating parallax from' in src

    def test_already_installed_guard(self):
        src = _src()
        assert 'already installed' in src


# ── AC09 — API failure handling ───────────────────────────────────────────

class TestAC09ApiFailure:
    def test_error_message_on_api_failure(self):
        src = _src()
        assert 'Could not fetch latest release. Set PARALLAX_VERSION=vX.X.X to install a specific version.' in src

    def test_exits_on_api_failure(self):
        src = _src()
        # error() helper always calls exit 1
        assert 'exit 1' in src


# ── General / structural ──────────────────────────────────────────────────

class TestGeneralStructure:
    def test_script_exists(self):
        assert SCRIPT.exists(), "install.sh does not exist"

    def test_posix_shebang(self):
        src = _src()
        assert src.startswith('#!/bin/sh'), "Must use POSIX sh shebang"

    def test_no_bash_isms_double_bracket(self):
        src = _src()
        # [[ as a bash test construct — NOT POSIX character classes like [[:space:]]
        assert not re.search(r'\[\[\s', src), "No bash [[...]] test constructs allowed (POSIX sh)"

    def test_no_bash_version_var(self):
        src = _src()
        assert '$BASH_VERSION' not in src

    def test_no_source_command(self):
        src = _src()
        # 'source' is a bash-ism; POSIX uses '.'
        lines = [l for l in src.splitlines() if re.match(r'\s*source\s+', l)]
        assert not lines, "No 'source' command allowed (use '.' instead)"

    def test_set_e_present(self):
        src = _src()
        assert 'set -e' in src

    def test_repo_variable_at_top(self):
        src = _src()
        # REPO must be defined before any API call
        idx_repo = src.index('REPO=')
        idx_api = src.index('api.github.com')
        assert idx_repo < idx_api
