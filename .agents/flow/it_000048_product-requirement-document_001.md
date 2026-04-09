# Requirement: `parallax comfyui` — Launch and Manage ComfyUI via CLI

## Context
Users who have the parallax runtime installed (`~/.parallax/env`) currently have no way to start ComfyUI's web UI through the CLI. They must locate the vendored ComfyUI directory and invoke it manually. This feature adds a `parallax comfyui` command group that handles start, stop, and status management cross-platform (Linux, macOS, Windows).

## Goals
- Allow any end user to launch the ComfyUI web interface with a single command.
- Provide lifecycle management (start, stop, status) consistent with the existing `parallax ms` service pattern.
- Support port customisation and optional browser auto-open out of the box.

## User Stories

### US-001: Start ComfyUI
**As a** developer, **I want** to run `parallax comfyui start` **so that** ComfyUI's web UI starts and is accessible in my browser without manual setup.

**Acceptance Criteria:**
- [ ] Running `parallax comfyui start` launches ComfyUI as a background process using the runtime at `~/.parallax/env`.
- [ ] The process listens on port `8188` by default.
- [ ] A PID file is written to `~/.config/parallax/comfyui.pid` so the process can be tracked.
- [ ] The command prints the URL (`http://localhost:<port>`) once the server is ready.
- [ ] If ComfyUI is already running, the command prints a warning and exits without starting a second instance.
- [ ] Typecheck / lint passes.
- [ ] Visually verified in terminal: output shows the URL and process starts cleanly.

### US-002: Stop ComfyUI
**As a** developer, **I want** to run `parallax comfyui stop` **so that** the running ComfyUI process is cleanly terminated.

**Acceptance Criteria:**
- [ ] Running `parallax comfyui stop` reads the PID from `~/.config/parallax/comfyui.pid` and terminates the process.
- [ ] The PID file is removed after a successful stop.
- [ ] If no instance is running, the command prints an informative message and exits with code 0.
- [ ] Works cross-platform: `SIGTERM` on Linux/macOS, `TerminateProcess` on Windows.
- [ ] Typecheck / lint passes.

### US-003: Override Port with `--port`
**As a** developer, **I want** to pass `--port <N>` to `parallax comfyui start` **so that** I can run ComfyUI on a port other than `8188` (e.g. when the default is occupied).

**Acceptance Criteria:**
- [ ] `parallax comfyui start --port 8189` starts ComfyUI on port `8189`.
- [ ] The printed URL reflects the custom port.
- [ ] Passing an invalid port (< 1 or > 65535) prints an error and exits with a non-zero code.
- [ ] Typecheck / lint passes.

### US-004: Auto-open Browser with `--open`
**As a** developer, **I want** to pass `--open` to `parallax comfyui start` **so that** my default browser opens the ComfyUI URL automatically after the server is ready.

**Acceptance Criteria:**
- [ ] `parallax comfyui start --open` opens `http://localhost:<port>` in the default browser once the server responds on the port.
- [ ] Uses `webbrowser.open()` (stdlib) — no additional dependency.
- [ ] Works on Linux, macOS, and Windows.
- [ ] Typecheck / lint passes.
- [ ] Visually verified in browser: ComfyUI interface loads.

### US-005: Check Status with `parallax comfyui status`
**As a** developer, **I want** to run `parallax comfyui status` **so that** I can confirm whether a ComfyUI instance is currently running and on which port.

**Acceptance Criteria:**
- [ ] If running: prints `ComfyUI is running (PID <pid>, port <port>)`.
- [ ] If not running: prints `ComfyUI is not running` and exits with code 0.
- [ ] Port is read from the PID file metadata (stored alongside the PID at start time).
- [ ] Typecheck / lint passes.

## Functional Requirements
- FR-1: A new Typer command group `comfyui` is registered in `cli/main.py` with subcommands `start`, `stop`, and `status`.
- FR-2: Implementation lives in `cli/commands/comfyui.py` (mirrors `cli/commands/ms.py` structure).
- FR-3: `start` launches ComfyUI via the runtime Python at `~/.parallax/env` using `subprocess.Popen` (non-blocking); the working directory is set to `vendor/ComfyUI` relative to the installed package.
- FR-4: The PID file at `~/.config/parallax/comfyui.pid` stores `pid:<N>\nport:<P>` so `status` and `stop` can read both values without re-scanning.
- FR-5: `start` polls the port (TCP connect) for up to 30 seconds before printing the ready URL; polling interval is 0.5 s.
- FR-6: All file paths use `pathlib.Path` and respect `PARALLAX_DB_PATH`-style conventions (config dir derived from `platformdirs` or `~/.config/parallax` fallback).
- FR-7: Cross-platform process termination: `os.kill(pid, signal.SIGTERM)` on POSIX; `subprocess.call(['taskkill', '/F', '/PID', str(pid)])` on Windows.
- FR-8: `start` redirects ComfyUI stdout and stderr to `~/.config/parallax/comfyui.log` (append mode). The ready message printed to the terminal includes the log path so the user knows where to find it.

## Non-Goals (Out of Scope)
- Registering ComfyUI as a persistent systemd/launchd/Windows service (that is `parallax ms` territory).
- Installing or updating ComfyUI models from this command.
- Exposing ComfyUI over a public network or HTTPS.
- A `--foreground` / attached mode (logs streamed to terminal) — may follow in a future iteration.
- Windows installer integration.

## Open Questions
- None.
