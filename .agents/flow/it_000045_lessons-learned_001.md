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
