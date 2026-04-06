# Lessons Learned — Iteration 000045

## US-001 — Install parallax on Linux or macOS with one command

**Summary:** Created `install.sh` at the repo root — a POSIX `sh` installer that detects OS/arch, fetches the latest GitHub Release tag (or uses `PARALLAX_VERSION` env override), downloads the binary + `.sha256` checksum via curl (wget fallback), verifies integrity, installs to `~/.local/bin/parallax`, and prints PATH guidance if needed. Also updated README.md with the one-liner install command (FR-7). Tests are static content assertions in `tests/test_install_sh_us001_it000045.py` (45 tests, all passing).

**Key Decisions:**
- **POSIX sh only** (no bash-isms): used `case`/`esac` for OS and arch branching, `command -v` for tool detection, POSIX character classes in sed (`[[:space:]]`), and `. ` (dot) instead of `source`. No `[[`, no `$BASH_VERSION`.
- **sed-only JSON parsing**: `tag_name` is extracted from the GitHub API response using `sed` + newline splitting — no `jq` required. This keeps the installer dependency-free.
- **Temporary directory + trap for cleanup**: `mktemp -d` + `trap _cleanup EXIT` ensures the temp directory is always removed even if the script exits early.
- **Update detection via `--version` output**: the script runs the existing binary with `--version` to get the current version, then compares against the target. If equal, it exits early (already installed); if different, it prints the "Updating from … to …" message.
- **`PARALLAX_INSTALL_DIR` override** (FR-4): the install path defaults to `$HOME/.local/bin` but can be overridden via environment variable.

**Pitfalls Encountered:**
- **`[[:space:]]` in sed regex is not a bash `[[` construct**: a test that literally checks for `'[[' not in src` fails because POSIX character classes like `[[:space:]]` contain `[[` as a substring. The fix is to check for the bash test form with a trailing space: `re.search(r'\[\[\s', src)`.
- **`-fL` does not contain the substring `-L`**: curl flags combined as `-fL` have `-` followed by `f` then `L`. The string literal `-L` (hyphen + uppercase-L) is not a substring of `-fL`. Tests checking for `-L` must use a regex like `r'curl\s+-\S*[Ll]'` instead.

**Useful Context for Future Agents:**
- The script is intentionally dependency-free: it only requires `sh`, `curl` or `wget`, and either `sha256sum` (Linux) or `shasum` (macOS). No Python, no Node, no package manager.
- Static tests (no shell execution) are the right approach for CI: running the script in tests would require network access and GitHub Releases to exist. Content assertions are sufficient to verify every acceptance criterion.
- Test file naming convention: `test_install_sh_us001_it000045.py` (prefix `test_install_sh_`), distinct from `test_cli_`, `test_build_`, and `test_ci_` prefixes used in earlier iterations.
- The README update (FR-7) adds an "Install the parallax CLI" section immediately before "Quick Start" to be discoverable by new users.
