# Refactor Plan — Iteration 000048 — Pass 001

## Summary of changes

All recommended changes from the audit report were applied to `cli/commands/comfyui.py`:

### FR-8 (critical — does not comply → fixed)
- Replaced `stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL` with a real log file handle opened in append mode (`~/.config/parallax/comfyui.log`).
- Added `_log_file()` helper that returns `_config_dir() / "comfyui.log"`.
- The log file parent directory is created with `mkdir(parents=True, exist_ok=True)` before opening.
- The log fd is closed in the parent process after `Popen` returns so the child holds the only reference.
- The ready message (and the timeout warning) both print `Logs: <log_path>` so users know where to look.

### FR-5 (partially comply → fixed)
- Replaced `urllib.request.urlopen` HTTP polling with `socket.create_connection(("localhost", port), timeout=1)` — a raw TCP connect that does not depend on ComfyUI returning a valid HTTP response.
- Polling interval changed from `1s` to `0.5s` (`time.sleep(0.5)`).
- Default timeout constant `_READY_TIMEOUT` reduced from `60` to `30` seconds.
- Removed `urllib.error` and `urllib.request` imports (no longer needed).

### FR-3 (partially comply → fixed)
- `comfyui_main.parent` is captured as `comfyui_root` and passed as `cwd=comfyui_root` to `subprocess.Popen`.

### FR-4 (partially comply → fixed)
- Replaced JSON PID file format with the spec-mandated plain-text `pid:<N>\nport:<P>` format.
- Added `_write_pid_file(pid_file, pid, port)` helper that writes this format.
- Updated `_read_pid_file()` to parse the new format by splitting on `:` per line.
- Removed `import json` (no longer needed).

### FR-7 (partially comply → fixed)
- Windows branch in `_terminate_process()` now calls `subprocess.call(["taskkill", "/F", "/PID", str(pid)])` instead of the redundant `os.kill(pid, signal.SIGTERM)`.

### FR-6 (partially comply → improved)
- Added `_config_dir()` helper that honours a `PARALLAX_CONFIG_DIR` environment variable, mirroring the `PARALLAX_HOME` pattern from `_common.py`.
- `_pid_file()` and `_log_file()` both delegate to `_config_dir()`.

## Quality checks

| Check | Command | Result |
|-------|---------|--------|
| Type checking | `uv run mypy cli/commands/comfyui.py --ignore-missing-imports` | **Pass** — no issues found |
| Lint | `uv run ruff check cli/commands/comfyui.py` | **Pass** — all checks passed |

No test suite exists for this module (noted as a minor observation in the audit). Static type and lint checks both pass cleanly.

## Deviations from refactor plan

None. All six FR fixes recommended in the audit conclusions were fully applied.
