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
