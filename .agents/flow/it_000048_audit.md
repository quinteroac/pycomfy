# Audit — it_000048 (PRD index 001)

## 1. Executive Summary

The implementation delivers all five user stories (US-001–US-005) with full functional coverage for the core ComfyUI lifecycle commands: `start`, `stop`, and `status`. The `comfyui` command group is correctly registered in `cli/main.py` and `cli/commands/comfyui.py` follows the expected structure. Four of eight functional requirements comply fully; the remaining four have minor deviations that do not break functionality but diverge from PRD specifications. The critical gap is **FR-8** (log file redirect): stdout/stderr are discarded to DEVNULL instead of being appended to `~/.config/parallax/comfyui.log`. FR-5 deviates in polling mechanism (HTTP vs TCP), interval (1 s vs 0.5 s), and default timeout (60 s vs 30 s). FR-4 uses JSON instead of the specified `pid:<N>\nport:<P>` format. FR-3 omits the `cwd` parameter in Popen. Overall quality is high and the feature is usable as-is.

---

## 2. Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | **comply** | `comfyui` Typer group registered in `cli/main.py:29`; `start`, `stop`, `status` subcommands present. |
| FR-2 | **comply** | Implementation in `cli/commands/comfyui.py`, mirrors `ms.py` structure. |
| FR-3 | **partially_comply** | `subprocess.Popen` used (non-blocking), runtime Python from `~/.parallax/env` resolved correctly. No `cwd` passed to Popen; ComfyUI root resolved dynamically via `_get_comfyui_main()` rather than `vendor/ComfyUI` relative path. |
| FR-4 | **partially_comply** | PID file stores both `pid` and `port` (satisfies intent) but uses JSON `{"pid": N, "port": N}` instead of the specified `pid:<N>\nport:<P>` plain-text format. |
| FR-5 | **partially_comply** | Polling implemented, but: uses HTTP (`urllib.request.urlopen`) not TCP connect; polling interval is 1 s (spec: 0.5 s); default timeout is 60 s (spec: 30 s). |
| FR-6 | **partially_comply** | `pathlib.Path` used consistently; `PARALLAX_HOME` env var respected for the base dir. Config dir for PID file (`~/.config/parallax`) is hardcoded — does not use `platformdirs`. |
| FR-7 | **partially_comply** | Both POSIX and Windows branches call `os.kill(pid, SIGTERM)` — branches are identical. Spec requires `subprocess.call(['taskkill', '/F', '/PID', str(pid)])` on Windows. `os.kill` does reach `TerminateProcess` via the CRT on Windows, so functionally acceptable. |
| FR-8 | **does_not_comply** | stdout/stderr from the ComfyUI subprocess are sent to `subprocess.DEVNULL`. No log file is created; no log path is printed in the ready message. |

---

## 3. Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | **partially_comply** | AC01–AC06 implemented correctly. AC07 (visual terminal verification) cannot be validated statically. The ready message does not include a log path (FR-8 not implemented). |
| US-002 | **comply** | All ACs satisfied: PID read, process terminated, PID file removed, no-instance path handles gracefully, cross-platform path present. |
| US-003 | **comply** | `--port` accepted, URL reflects custom port, invalid port rejected with non-zero exit. |
| US-004 | **partially_comply** | AC01–AC04 implemented. AC05 (visual browser verification) cannot be validated statically. |
| US-005 | **comply** | Running/not-running messages correct, port read from JSON metadata, exits with 0. |

---

## 4. Minor Observations

1. `_terminate_process` has identical bodies for `win32` and non-`win32` branches — the `if/else` is redundant. Either simplify to one `os.kill` call or implement the `taskkill` variant for true Windows parity.
2. `_wait_until_ready` uses `urllib.request.urlopen`, which raises on HTTP error responses (4xx/5xx) in addition to connection errors. If ComfyUI returns an HTTP error during startup, polling exits prematurely. A TCP-connect probe (`socket.create_connection`) would be more robust.
3. Default ready-wait timeout was increased to 60 s from the 30 s specified in FR-5. This is more forgiving but extends the maximum blocking time for the user.
4. No automated test files are included. Unit tests (mocking `subprocess.Popen`, PID file I/O) would improve reliability and enforce regression safety.
5. FR-8 absent means all ComfyUI subprocess output is silently discarded, making startup-failure diagnosis impossible without re-running in foreground.

---

## 5. Conclusions and Recommendations

The implementation is functionally complete. The one **does_not_comply** item (FR-8) is the priority fix: redirect ComfyUI stdout/stderr to `~/.config/parallax/comfyui.log` (append mode) and include the log path in the ready message.

Secondary fixes in priority order:

1. **FR-8** — Redirect stdout/stderr to `~/.config/parallax/comfyui.log`; include log path in the "running at" message.
2. **FR-5** — Switch `_wait_until_ready` to TCP connect via `socket.create_connection`; set polling interval to 0.5 s; reduce default timeout to 30 s.
3. **FR-3** — Add `cwd=str(comfyui_main.parent)` to the `Popen` call so ComfyUI runs in its own directory.
4. **FR-4** — Adopt the spec format `pid:<N>\nport:<P>` (or formally document the JSON deviation).
5. **FR-7** — Implement the `taskkill` Windows branch explicitly.
6. **FR-6** — Use `platformdirs.user_config_dir("parallax")` for config dir resolution.

---

## 6. Refactor Plan

### Step 1 — FR-8: Log file redirect (critical)

**File:** `cli/commands/comfyui.py`

- In `start()`, before `Popen`, open `~/.config/parallax/comfyui.log` in append mode.
- Pass the file handle as `stdout` and `stderr` in the `Popen` call (replacing `DEVNULL`).
- After printing the ready URL, append: `f"  Logs: {log_file}"`.
- Close the handle after `Popen` starts (the child inherits it).

### Step 2 — FR-5: TCP polling + interval + timeout

**File:** `cli/commands/comfyui.py`

- Replace `_wait_until_ready` body: use `socket.create_connection((\"localhost\", port), timeout=1)` inside a try/except.
- Change `time.sleep(1)` to `time.sleep(0.5)`.
- Change `_READY_TIMEOUT` default from `60` to `30`.

### Step 3 — FR-3: Set cwd in Popen

**File:** `cli/commands/comfyui.py`

- Add `cwd=str(comfyui_main.parent)` to the `subprocess.Popen(...)` call in `start()`.

### Step 4 — FR-7: Windows taskkill branch (optional but spec-aligned)

**File:** `cli/commands/comfyui.py`

- In `_terminate_process`, change the `win32` branch to:
  ```python
  subprocess.call(["taskkill", "/F", "/PID", str(pid)])
  ```

### Step 5 — FR-4: PID file format (low priority, or document deviation)

**File:** `cli/commands/comfyui.py`

- Either switch PID file format to `pid:<N>\nport:<P>` and update `_read_pid_file` accordingly,
  or add a comment formally documenting the JSON deviation and why it was chosen.

### Step 6 — FR-6: platformdirs config dir (low priority)

**File:** `cli/commands/comfyui.py`

- Replace hardcoded `Path.home() / ".config" / "parallax"` in `_pid_file()` with:
  ```python
  from platformdirs import user_config_dir
  Path(user_config_dir("parallax"))
  ```
- Add `platformdirs` to project dependencies if not already present.
