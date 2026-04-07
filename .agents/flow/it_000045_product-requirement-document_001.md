# Installer Subcommands: parallax install / mcp install / ms install

## Context

The `parallax` CLI is the single entry point for the Parallax ecosystem. End users (non-developers) install a standalone binary and use it to bootstrap the full stack. The stack is 100% Python: `comfy_diffusion` (core inference library), `server/` (FastAPI worker on :5000), and `mcp/` (FastMCP server for Claude Desktop). This PRD covers the CLI-side logic for three installer subcommands: `parallax install` (comfy-diffusion runtime + ComfyUI bootstrap), `parallax mcp install` (registers the MCP server with Claude Desktop), and `parallax ms install` (registers the FastAPI server as a system service via systemd/launchd). Binary compilation and distribution are handled separately in PRD-002 and PRD-003.

## Goals

- Allow a non-developer user to bootstrap the entire Parallax stack from a single CLI binary with no prior Python knowledge.
- Provide clear progress feedback and actionable error messages for each install step.
- Make installs idempotent — re-running any subcommand detects existing state and skips or updates gracefully.

## User Stories

### US-001: Install the comfy-diffusion runtime
**As a** non-developer user, **I want** to run `parallax install` **so that** the comfy-diffusion library, torch, and the ComfyUI engine are installed and ready on my machine.

**Acceptance Criteria:**
- [ ] `parallax install` detects whether `uv` is present; if not, downloads and installs it automatically using `urllib.request` (stdlib only, no curl/pip dependency) before proceeding.
- [ ] `parallax install` creates a dedicated virtual environment at `~/.parallax/env` using `uv venv` and installs `comfy-diffusion[cuda]` (or `[cpu]` via a `--cpu` flag) into it.
- [ ] After package installation, `parallax install` calls `check_runtime()` from within the installed environment to bootstrap the ComfyUI submodule; if it returns an error dict, the command prints the error and exits with code 1.
- [ ] On success, the command prints the installed version and the next step: "Run `parallax ms install` to set up the inference server."
- [ ] On failure at any step, the command prints the failing step, the subprocess error output, and a suggestion to re-run with `--verbose`.
- [ ] Re-running `parallax install` when already installed prints "Already installed (v1.x.x). Run `parallax install --upgrade` to update." and exits cleanly without performing any work.

---

### US-002: Install the MCP server
**As a** non-developer user, **I want** to run `parallax mcp install` **so that** the Parallax MCP server is registered in Claude Desktop and available as a tool immediately.

**Acceptance Criteria:**
- [ ] `parallax mcp install` checks that `parallax install` has been completed (i.e. `~/.parallax/env` exists and contains the `parallax-mcp` script); if not, it prints "Run `parallax install` first." and exits with code 1.
- [ ] `parallax mcp install` locates the Claude Desktop config file at the platform-appropriate path: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS), `%APPDATA%\Claude\claude_desktop_config.json` (Windows), `~/.config/claude/claude_desktop_config.json` (Linux).
- [ ] The command adds or updates the `parallax-mcp` entry under `mcpServers` in the config JSON, setting `command` to the absolute path of the `parallax-mcp` script inside `~/.parallax/env`.
- [ ] If the Claude Desktop config file does not exist, the command creates it with only the `mcpServers` key.
- [ ] Merging into the config file never overwrites keys other than `mcpServers.parallax-mcp`.
- [ ] On success, the command prints "MCP server registered. Restart Claude Desktop to apply."
- [ ] Re-running `parallax mcp install` when already registered prints "Already registered." and exits cleanly.

---

### US-003: Register the FastAPI server as a system service
**As a** non-developer user, **I want** to run `parallax ms install` **so that** the Parallax inference server starts automatically on boot without manual intervention.

**Acceptance Criteria:**
- [ ] `parallax ms install` checks that `parallax install` has been completed; if not, it prints "Run `parallax install` first." and exits with code 1.
- [ ] On Linux, the command writes a systemd user unit file to `~/.config/systemd/user/parallax-ms.service` and runs `systemctl --user enable --now parallax-ms` to start and enable the service.
- [ ] On macOS, the command writes a launchd plist to `~/Library/LaunchAgents/run.parallax.ms.plist` and runs `launchctl load` to register it.
- [ ] The service unit/plist configures the server to run `uvicorn server.main:app --host 0.0.0.0 --port 5000` using the Python interpreter from `~/.parallax/env`.
- [ ] On success, the command prints "Inference server running on http://localhost:5000" and confirms the service status with one line of output.
- [ ] Re-running `parallax ms install` when already registered prints "Already registered." and exits cleanly.

---

## Functional Requirements

- FR-1: All three subcommands (`install`, `mcp install`, `ms install`) are implemented as Typer sub-apps registered on the root `parallax` app in `cli/main.py`.
- FR-2: A new `cli/commands/install.py` module implements the `install` command and the `mcp` and `ms` sub-groups.
- FR-3: The install base directory defaults to `~/.parallax/` and is configurable via the `PARALLAX_HOME` environment variable.
- FR-4: All long-running subprocess calls display a spinner via `rich.progress` so the user knows the process is active.
- FR-5: All subcommands support a `--verbose` flag that streams subprocess stdout/stderr in real time instead of capturing it.
- FR-6: The uv auto-install uses `urllib.request` to download the official uv installer script from `https://astral.sh/uv/install.sh` and pipes it to `sh` via `subprocess` — no pip, curl, or wget dependency required.
- FR-7: Claude Desktop config manipulation reads the existing JSON, merges only the `mcpServers.parallax-mcp` key, and writes it back — never overwrites unrelated keys.
- FR-8: `check_runtime()` is invoked via `subprocess` against the installed environment's Python interpreter, not imported directly, so the frozen binary does not need comfy_diffusion in its bundled deps.

## Non-Goals

- Building or compiling the CLI binary (covered in PRD-002).
- Implementing `parallax server start` / `parallax ms start` runtime commands (future iteration).
- Managing GPU drivers or CUDA toolkit installation.
- Windows support for `parallax ms install` (systemd/launchd only; Windows service deferred).

## Open Questions

- None — all open questions from ideation resolved: `check_runtime()` is called during `parallax install`; all services are Python (no Bun); macOS uses universal binary; GitHub raw URL for scripts.
