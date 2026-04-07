# Lessons Learned — Iteration 000045

## US-001 — Install the comfy-diffusion runtime

**Summary:** Implemented `parallax install` as a top-level Typer command in `cli/commands/install.py`, registered it via `app.command("install")(install)` in `cli/main.py`. The command handles uv detection/auto-install (stdlib urllib.request), venv creation at `~/.parallax/env`, package installation (`[cuda]`/`[cpu]`), ComfyUI bootstrapping via `check_runtime()`, and idempotent re-run guard.

**Key Decisions:**
- Registered as a plain function (not a Typer sub-app) to give a flat `parallax install` UX without nesting.
- All subprocess-touching logic was extracted into small, mockable helpers (`_find_uv`, `_download_and_install_uv`, `_ensure_uv`, `_installed_version`, `_run_step`, `_bootstrap_comfyui`) to keep tests fast and CI-friendly.
- `_installed_version()` uses the venv's own Python + `importlib.metadata` — avoids importing comfy_diffusion at CLI startup.
- The `--upgrade` flag bypasses the AC06 guard by simply not checking `_installed_version` first.

**Pitfalls Encountered:**
- `typer.testing.CliRunner` in the project's version of Typer does **not** support the `mix_stderr=False` constructor argument (unlike Click's CliRunner). Removing that kwarg fixed all 18 failures at once.
- When patching `subprocess.run` for multi-step flows, be careful about call order: the bootstrap step is the third `subprocess.run` call. Using a discriminating side-effect function (checking `cmd` contents) is more robust than a call-count approach.

**Useful Context for Future Agents:**
- The CLI entry-point pattern for single top-level commands (not sub-groups) is: define a plain function in `cli/commands/<name>.py`, import it in `cli/main.py`, and call `app.command("<name>")(<function>)`.
- `CliRunner()` from `typer.testing` merges stdout and stderr by default — tests should check `result.output` for both, or use `result.exit_code` to assert failure.
- The uv installer URL (`https://astral.sh/uv/install.sh`) is the official source; tests mock `urllib.request.urlopen` directly to avoid network calls.

## US-002 — Install the MCP server

**Summary:** Implemented `parallax mcp install` as a Typer sub-app (`mcp_app`) in `cli/commands/mcp.py`, registered via `app.add_typer(mcp_app, name="mcp")` in `cli/main.py`. The command checks `~/.parallax/env/bin/parallax-mcp` exists (AC01), resolves the Claude Desktop config path per platform using `platform.system()` (AC02), merges only `mcpServers.parallax-mcp` into the JSON (AC03–AC05), and handles both create (AC04) and idempotent re-run (AC07) cases.

**Key Decisions:**
- Registered as a Typer sub-app (not a plain function) so it supports `parallax mcp install` with future extensibility (`parallax mcp uninstall`, etc.).
- Three internal helpers (`_mcp_script_path`, `_claude_config_path`, `_read_config`, `_write_config`) extracted for clean unit testing without filesystem side-effects.
- "Already registered" check compares the stored command string to `str(script)` — if path changed (e.g. re-install to different location), it updates rather than short-circuits.
- `_write_config` creates parent directories with `mkdir(parents=True, exist_ok=True)` so the file can be created in a nested path that doesn't exist yet (AC04).

**Pitfalls Encountered:**
- `patch("cli.commands.mcp._ENV_DIR", env_dir)` patches the module-level constant directly; must be done before `_mcp_script_path` is called, which reads `_ENV_DIR`. Since `_mcp_script_path` is also patched in most tests, this works cleanly.
- Patching `Path` subclass instances for existence checks is awkward — using real `tmp_path` directories in tests (creating/not creating files) is simpler and more reliable than mocking Path.

**Useful Context for Future Agents:**
- The sub-app registration pattern for grouped commands is: `app = typer.Typer(...)` in `cli/commands/<name>.py`, `@app.command("<subcommand>")` decorator, then `app.add_typer(<name>_app, name="<name>")` in `cli/main.py`.
- All tests use real `tmp_path` directories (pytest fixture) rather than mocking filesystem calls — this gives confidence that the JSON read/write/create logic is correct.
- `platform.system()` returns `"Darwin"` (macOS), `"Windows"`, or `"Linux"` — patch it with `patch("platform.system", return_value=...)` in tests.

## US-003 — Register the FastAPI server as a system service

**Summary:** Implemented `parallax ms install` as a Typer sub-app (`ms_app`) in `cli/commands/ms.py`, registered via `app.add_typer(ms_app, name="ms")` in `cli/main.py`. The command checks `~/.parallax/env/bin/python` exists (AC01), writes a platform-appropriate service file (AC02/AC03), configures the service to run `python -m uvicorn server.main:app` from `~/.parallax/env` (AC04), prints the server URL + one status line (AC05), and handles idempotent re-runs by checking if the service file already exists (AC06).

**Key Decisions:**
- Used `python -m uvicorn` (not the uvicorn script) to be explicit that the Python interpreter from `~/.parallax/env` is used, satisfying AC04's wording.
- "Already registered" check tests for file existence (unit file on Linux, plist on macOS) — simple and robust without subprocess overhead.
- Extracted `_write_systemd_unit`, `_write_launchd_plist`, `_get_service_status_line` as testable helpers alongside path helpers `_systemd_unit_path`, `_launchd_plist_path`, `_python_path`.
- Status confirmation (AC05) calls `systemctl --user is-active` or `launchctl list` to obtain one status line after success.

**Pitfalls Encountered:**
- Using `runner.invoke(sub_app, ["install"])` on a Typer sub-app with a single `@app.command("install")` returns exit code 2 in Typer 0.24.x. Typer enters "single command mode" and treats `"install"` as a spurious argument. The fix is always to invoke via the **root app**: `runner.invoke(cli.main.app, ["ms", "install"])`.
- The `_app()` helper pattern (imported lazily inside a function to avoid circular imports at collection time) is the established pattern in this project's CLI tests.

**Useful Context for Future Agents:**
- **Critical:** CLI tests for sub-apps must use `cli.main.app` with the full command path (e.g. `["ms", "install"]`), not the sub-app directly. Using the sub-app with a named subcommand fails in Typer 0.24.x with exit code 2.
- Patching module-level constants like `_ENV_DIR` in `cli.commands.ms` works correctly as long as the patch is applied before the command function accesses it (which it does at call time, not import time).
- `subprocess.run` can be patched globally for the test; a discriminating `side_effect` function (checking `cmd` contents) is more robust than positional call counting when multiple subprocess calls happen in sequence.
