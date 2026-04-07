# CLI Frontend Module — `parallax frontend install`

## Context

Following the pattern established by `parallax ms install` and `parallax mcp install`, the frontend should be installable as a module via the Parallax CLI. This PRD covers the `parallax frontend install` command, which downloads the pre-built frontend dist from GitHub Releases, places it at `~/.parallax/frontend/`, and sets `PARALLAX_FRONTEND_PATH` in the Parallax environment so the server picks it up automatically.

## Goals

- Let users install the frontend with a single command, consistent with the existing CLI module pattern.
- Store the frontend at a predictable, user-owned path (`~/.parallax/frontend/`).
- Persist the path so the server uses it without manual env var management.

## User Stories

### US-001: Install the frontend via CLI
**As a** user, **I want** to run `parallax frontend install` **so that** the chat UI is downloaded and ready to be served by the Parallax server without any manual steps.

**Acceptance Criteria:**
- [ ] Running `parallax frontend install` downloads the latest pre-built frontend archive from the project's GitHub Releases (asset name: `parallax-frontend-{version}.tar.gz`).
- [ ] The archive is extracted to `~/.parallax/frontend/`, replacing any previous installation.
- [ ] After installation, `~/.parallax/config.env` (or equivalent config file used by `parallax install`) contains `PARALLAX_FRONTEND_PATH=~/.parallax/frontend`.
- [ ] The command prints: `Frontend installed at ~/.parallax/frontend` on success.
- [ ] If the download fails (network error, 404), the command prints a human-readable error and exits with a non-zero code — it does not leave a partial installation.
- [ ] Running the command a second time re-downloads and overwrites the existing installation cleanly.

### US-002: Pin a specific frontend version
**As a** user, **I want** to install a specific version of the frontend with `parallax frontend install --version 1.2.3` **so that** I can pin the UI to a known-good release.

**Acceptance Criteria:**
- [ ] `--version` accepts a semver string (e.g. `1.2.3`).
- [ ] If the specified version does not exist as a GitHub Release asset, the command prints the error returned by GitHub and exits non-zero.
- [ ] If `--version` is omitted, the latest release is installed.

### US-003: Check installed frontend version
**As a** user, **I want** to run `parallax frontend version` **so that** I can see which version of the frontend is currently installed.

**Acceptance Criteria:**
- [ ] Running `parallax frontend version` prints the installed version string (read from `~/.parallax/frontend/version.txt`).
- [ ] If no frontend is installed, it prints: `Frontend not installed. Run \`parallax frontend install\` to install.`

---

## Functional Requirements

- FR-1: A new `frontend` command group is added to the Typer CLI in `cli/main.py`, with subcommands `install` and `version`.
- FR-2: `parallax frontend install` downloads the release archive via HTTPS using `httpx` (already available via FastAPI deps) — no new network deps.
- FR-3: The archive is a `.tar.gz` containing the `dist/` contents at its root; extraction target is `~/.parallax/frontend/`.
- FR-4: After extraction, the command writes `PARALLAX_FRONTEND_PATH` to the Parallax config file (same file written by `parallax install`).
- FR-5: A `version.txt` file is included in the release archive and contains the semver version string; `parallax frontend version` reads this file.
- FR-6: The GitHub Release asset URL pattern is: `https://github.com/{owner}/{repo}/releases/download/v{version}/parallax-frontend-{version}.tar.gz`.
- FR-7: The `.github/workflows/release-cli.yml` CI workflow is extended to build the frontend with Bun (`bun run build` in `frontend/`), package the `dist/` as `parallax-frontend-{version}.tar.gz`, and upload it as a release asset alongside the existing CLI binary.

## Non-Goals

- Auto-updating the frontend on server start.
- Frontend uninstall command.
- Installing from a local path or custom URL (beyond `--version`).
- Windows support for the install command in this iteration (Linux/macOS only, consistent with `ms install`).

## Open Questions

- Should the config file written by `parallax install` be `.env` format or a custom format? (Assume `.env` format consistent with existing `install.py` behavior — verify before implementing.)
