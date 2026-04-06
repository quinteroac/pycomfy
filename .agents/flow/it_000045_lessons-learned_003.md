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

## US-002 — Install parallax on Windows with one command

**Summary:** Created `install.ps1` at the repo root — a PowerShell installer that always targets `parallax-windows-x86_64.exe`, fetches the latest GitHub Release tag via the API (or uses `$env:PARALLAX_VERSION` override), downloads the binary + `.sha256` checksum with `Invoke-WebRequest`, verifies the checksum with `Get-FileHash -Algorithm SHA256`, installs to `$env:APPDATA\parallax\bin\parallax.exe`, and updates the user PATH via `[Environment]::SetEnvironmentVariable(..., "User")`. Tests are static content assertions in `tests/test_install_ps1_us002_it000045.py` (37 tests, all passing).

**Key Decisions:**
- **No arch branching**: Only `x86_64` is supported in this iteration — the asset name is hardcoded as `parallax-windows-x86_64.exe`, no `if` branching on CPU architecture.
- **`$ProgressPreference = "Continue"`**: Explicitly set before `Invoke-WebRequest` calls to ensure the progress bar is visible even if the caller's session has suppressed it.
- **`ConvertFrom-Json` for API response**: PowerShell's built-in JSON parser eliminates the need for string hacking — `$Response.Content | ConvertFrom-Json` extracts `tag_name` cleanly.
- **`$env:APPDATA` for install dir**: Standard Windows convention for per-user app data; `parallax\bin` subdirectory created with `New-Item -ItemType Directory -Force`.
- **`-notlike "*$INSTALL_DIR*"` guard**: PATH update is skipped if the directory is already present, preventing duplicates.
- **`TrimStart('v')`**: Used to strip the leading `v` from the version tag for the bare semver displayed in the success message.

**Pitfalls Encountered:**
- **`$ErrorActionPreference = "Stop"` replaces `set -e`**: PowerShell does not have `set -e`; the equivalent is `$ErrorActionPreference = "Stop"` at the top of the script.
- **`Invoke-WebRequest` progress bar is on by default** but can be suppressed by the caller's `$ProgressPreference`. Explicitly resetting it to `"Continue"` in the script is necessary for AC03.
- **Hash comparison requires case normalisation**: `Get-FileHash` returns uppercase hex; `.sha256` files may be lowercase. Both sides must be `.ToUpper()` before comparison.

**Useful Context for Future Agents:**
- Static content-assertion tests (no PowerShell execution) are the right approach for CI on Linux: running `install.ps1` in tests would require a Windows runner, network access, and live GitHub Releases. Content assertions are sufficient to verify every acceptance criterion.
- Test file naming convention: `test_install_ps1_us002_it000045.py` (prefix `test_install_ps1_`), parallel to `test_install_sh_` used for the Linux/macOS installer.
- The `ProgressPreference` test checks for `'"Continue"'` (double-quoted string in PowerShell source) — this is how PowerShell writes string literals, not Python string delimiters.
- `[Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")` writes to `HKCU\Environment` in the Windows Registry; the change is visible in new terminals without requiring admin rights.
