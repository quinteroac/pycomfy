# Lessons Learned — Iteration 000048

## US-001 — Start ComfyUI

**Summary:** Implemented `parallax comfyui start` as a new Typer sub-app (`cli/commands/comfyui.py`) registered in `cli/main.py`. The command launches ComfyUI as a detached background process using `~/.parallax/env/bin/python`, writes the process PID to `~/.config/parallax/comfyui.pid`, polls for readiness, and prints the URL. An already-running guard checks the PID file before spawning.

**Key Decisions:**
- Used `subprocess.Popen` with `start_new_session=True` and `stdout/stderr=DEVNULL` for a clean background launch that survives the parent process exiting.
- Resolved ComfyUI's `main.py` by invoking `comfy_diffusion._runtime._comfyui_root()` inside the env's Python interpreter — avoids hardcoding any path, works for both editable installs and installed packages.
- `_is_running()` uses `os.kill(pid, 0)` (the portable signal-0 trick) to check liveness without sending a real signal.
- Extracted all I/O-bound helpers (`_is_running`, `_get_comfyui_main`, `_wait_until_ready`, `_python_path`, `_pid_file`) as standalone functions so they can be individually mocked in tests.

**Pitfalls Encountered:**
- The `_run_start` test helper initially did not pre-create the PID file when `already_running=True`. The command still tried to `read_text()` the PID file to include the PID in its warning message, causing a `FileNotFoundError`. Fixed by writing a fake PID to the file in the helper when `already_running=True`.
- A pre-existing ruff lint error (`Optional[bool]` → `bool | None`) already existed in `cli/main.py:73` before this iteration; it is not introduced by this story and not fixed to stay within scope.

**Useful Context for Future Agents:**
- The `_comfyui_root()` function in `comfy_diffusion/_runtime.py` handles both repo (editable) and installed-package layouts. Always use it (via a subprocess call to the env's Python) rather than constructing paths manually.
- The PID file lives at `~/.config/parallax/comfyui.pid` — future `stop`/`restart` commands should use the same `_pid_file()` helper from `cli/commands/comfyui.py`.
- The `_DEFAULT_PORT = 8188` constant is canonical for ComfyUI; the inference server runs on port 5000. Do not mix them.
- When `_wait_until_ready` polls `http://localhost:<port>`, it swallows `urllib.error.URLError` and `OSError` — this is intentional since the server may refuse connections during startup.
