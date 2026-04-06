# Bootstrap Install Script: install.sh / install.ps1

## Context

Non-developer users need a one-command way to get the `parallax` binary onto their machine before they can run any `parallax install` subcommand. This PRD covers writing a shell script (`install.sh`) for Linux/macOS and a PowerShell script (`install.ps1`) for Windows. Each script detects the user's platform, downloads the correct binary from GitHub Releases (`https://raw.githubusercontent.com/quinteroac/comfy-diffusion`), verifies its SHA256 checksum, and installs it into the user's PATH. These scripts are the only thing a user needs to copy from the README.

## Goals

- Reduce onboarding to a single command: `curl -fsSL https://raw.githubusercontent.com/quinteroac/comfy-diffusion/master/install.sh | sh`.
- Handle platform/arch detection, download, checksum verification, and PATH setup automatically.
- Provide clear, human-readable output at each step for a non-developer audience.

## User Stories

### US-001: Install parallax on Linux or macOS with one command
**As a** non-developer on Linux or macOS, **I want** to run `curl -fsSL https://raw.githubusercontent.com/quinteroac/comfy-diffusion/master/install.sh | sh` **so that** `parallax` is available in my terminal without any manual steps.

**Acceptance Criteria:**
- [ ] The script detects OS (`uname -s`) and architecture (`uname -m`) and selects the correct asset name: `parallax-linux-x86_64` for Linux x86_64, `parallax-macos-universal` for macOS (any arch).
- [ ] The script queries `https://api.github.com/repos/quinteroac/comfy-diffusion/releases/latest` to obtain the latest tag; if `PARALLAX_VERSION` is set in the environment, it uses that version instead.
- [ ] The script downloads the binary and its `.sha256` file from the GitHub Release using `curl` (preferred) or `wget` (fallback), with a progress indicator.
- [ ] The script verifies the downloaded binary against the `.sha256` checksum using `sha256sum` (Linux) or `shasum -a 256` (macOS); if verification fails, it deletes the file, prints "Checksum verification failed. Aborting.", and exits with code 1.
- [ ] The script installs the binary to `~/.local/bin/parallax` (creating the directory if needed) and sets it executable with `chmod +x`.
- [ ] If `~/.local/bin` is not on `$PATH`, the script prints the exact export line to add to `~/.bashrc`, `~/.zshrc`, or `~/.profile`, and instructs the user to open a new terminal.
- [ ] On success, the script prints "parallax X.X.X installed. Run: parallax install" to guide the next step.
- [ ] Re-running the script when a binary already exists prints "Updating parallax from vX.X.X to vY.Y.Y." and replaces the binary.
- [ ] If the GitHub API request fails, the script prints "Could not fetch latest release. Set PARALLAX_VERSION=vX.X.X to install a specific version." and exits with code 1.

---

### US-002: Install parallax on Windows with one command
**As a** non-developer on Windows, **I want** to paste a single PowerShell command **so that** `parallax` is available in my terminal without downloading or configuring anything manually.

**Acceptance Criteria:**
- [ ] The `install.ps1` script always downloads `parallax-windows-x86_64.exe` (only x86_64 supported in this iteration).
- [ ] The script queries the GitHub Releases API to obtain the latest tag; if `$env:PARALLAX_VERSION` is set, it uses that version instead.
- [ ] The script downloads the binary and its `.sha256` file using `Invoke-WebRequest` with a visible progress bar.
- [ ] The script verifies the checksum using `Get-FileHash -Algorithm SHA256`; if it does not match, it deletes the file, prints "Checksum verification failed. Aborting.", and exits with code 1.
- [ ] The script installs the binary to `$env:APPDATA\parallax\bin\parallax.exe`, creating the directory if needed.
- [ ] The script adds `$env:APPDATA\parallax\bin` to the user's `PATH` via `[Environment]::SetEnvironmentVariable(..., "User")` if not already present.
- [ ] On success, the script prints "parallax X.X.X installed. Open a new terminal and run: parallax install".

---

## Functional Requirements

- FR-1: `install.sh` is written in POSIX `sh` (no bash-isms: no `[[`, no `$BASH_VERSION`, no `source`) so it runs on Alpine Linux, Debian, and macOS without modification.
- FR-2: `install.ps1` requires PowerShell 5.1+ (ships with Windows 10) and uses no external tools.
- FR-3: Both scripts define the GitHub repo as a single variable at the top (`REPO="quinteroac/comfy-diffusion"` / `$Repo = "quinteroac/comfy-diffusion"`) to simplify future rebranding.
- FR-4: Both scripts accept `PARALLAX_INSTALL_DIR` / `$env:PARALLAX_INSTALL_DIR` as an override for the install path.
- FR-5: Neither script requires elevated privileges — all paths are user-owned (`~/.local/bin`, `$APPDATA`).
- FR-6: The GitHub Releases API is queried at `https://api.github.com/repos/{REPO}/releases/latest`; the tag is extracted from the `tag_name` field.
- FR-7: The README is updated to show the one-liner install commands for Linux/macOS and Windows using the raw GitHub URLs.

## Non-Goals

- Custom domain or URL shortener (GitHub raw URLs are used directly).
- Uninstall script (deferred to a future iteration).
- Linux ARM support (deferred — no `parallax-linux-arm64` asset in this iteration).
- Installing system-level dependencies (GPU drivers, Python, etc.) — those are handled by `parallax install` in PRD-001.

## Open Questions

- None — GitHub raw URL confirmed; URL shortener deferred.
