# Lessons Learned — Iteration 000044

## US-001 — Generation commands (sync mode)

**Summary:** Implemented a new Python Typer-based CLI under `cli/` at repo root with five commands: `parallax create image/video/audio`, `parallax edit image`, and `parallax upscale image`. All commands are in sync mode (blocking, calling `comfy_diffusion.pipelines` directly). Typer v0.24.1 was added as a core dependency.

**Key Decisions:**
- Created a clean `cli/` package at repo root, separate from the existing TypeScript `packages/parallax_cli/`.
- `cli/main.py` registers three sub-apps (`create`, `edit`, `upscale`); `cli/__main__.py` enables `python -m cli` invocation.
- `cli/_io.py` centralises save helpers (`save_image`, `save_video_frames`, `save_audio`, `resolve_models_dir`) — all lazily imported.
- `cli/commands/upscale.py` implements ESRGAN and latent-upscale inline using `comfy_diffusion` library primitives since no dedicated pipeline modules exist for these.
- Used `pyproject.toml` `[project.scripts]` `parallax = "cli.main:main"` for the `parallax` entry point, and added `cli*` to `[tool.setuptools.packages.find] include`.

**Pitfalls Encountered:**
1. **Typer `CliRunner` has no `mix_stderr` parameter** in v0.24.1. Tests using `mix_stderr=False` will raise `TypeError`. Use plain `CliRunner()`.
2. **Mock patching location**: When testing, always patch at the *import site* in the command module (`cli.commands.create.save_audio`), not at the source module (`cli._io.save_audio`), because `create.py` imports `save_audio` directly via `from cli._io import save_audio`.
3. **Pipeline call signature mismatches**: Several run() functions have non-obvious differences:
   - `ltx2.t2v.run()` uses `cfg_pass1` (not `cfg`); `ltx2.i2v.run()` uses `cfg`.
   - `wan22.t2v.run()` and `wan22.i2v.run()` use mixed positional + keyword-only signature with `*` separator before `models_dir` — they have no `fps` parameter.
   - `ace_step.checkpoint.run()` uses `tags=` (not `prompt=`) and `duration=` (not `length=`).
4. **Python parameter ordering**: Typer options with no default must come before those with defaults in the function signature — or use `= ...` (Typer's required sentinel) to work around this. `upscale.py` had `input` (required) after `prompt = ""` and needed `input = ...` to satisfy Python syntax.

**Useful Context for Future Agents:**
- `CliRunner` in Typer 0.24.1 mixes stdout and stderr into `result.output`; use that for all assertions.
- The existing TypeScript `packages/parallax_cli/` is a separate runtime; **do not modify it** when implementing the Python CLI.
- The `PYCOMFY_MODELS_DIR` env var is the fallback for `--models-dir`. `resolve_models_dir()` in `_io.py` handles both.
- `latent_upscale` upscale image command uses `sample()` from `comfy_diffusion.sampling` with `denoise=0.5` for the hi-res fix pass — this is intentional (50% denoise for refinement, not full generation).
- `typer.Option(...)` (with `...` as default) marks an option as required even when it appears after optional parameters in the function signature.

## US-002 — `--async` flag on generation commands

**Summary:** Replaced the "not yet available" stub in all five generation commands with a working `--async` implementation. Created `cli/_async.py` with `run_async()` and `_call_submit_job()`. The `--async` flag now calls `submit_job()` from `server/submit.py` and prints `Job <job_id> queued\n  → parallax jobs watch <job_id>`.

**Key Decisions:**
- Created `cli/_async.py` as a dedicated helper module to keep async logic DRY across all five commands. It contains `run_async()` (builds `JobData`, calls `submit_job()`, prints output) and `_call_submit_job()` (thin wrapper enabling test mocking).
- Kept `server.jobs.JobData` and `server.submit.submit_job` as lazy imports inside function bodies — `cli/_async.py` does NOT import them at module top level. This is critical because `pydantic` is not installed in the test environment, so top-level imports of `server.jobs` would break module loading.
- The `_call_submit_job()` wrapper is the patchable surface: `patch("cli._async._call_submit_job", return_value=job_id)`. Without it, tests would need to patch `server.submit.submit_job`, which fails because `server.submit` transitively imports `server.jobs` (pydantic), and pydantic isn't installed.
- `JobData` creation also requires pydantic; tests mock it via `sys.modules` injection of a fake `server.jobs` module with `JobData = MagicMock(...)` in an `autouse` fixture.
- Removed the now-obsolete `TestAsyncModeNotYetAvailable` class from `test_cli_us001_it044.py` since it tested the old "not yet available" stub.

**Pitfalls Encountered:**
1. **pydantic not installed in test env**: `server.jobs` requires pydantic which isn't available under `uv run pytest`. Any top-level import of `server.jobs` in `cli/_async.py` causes immediate `ModuleNotFoundError` at CLI module load time.
2. **`patch()` imports the target module**: When you call `patch("server.submit.submit_job")`, unittest.mock tries to import `server.submit`, which imports `server.jobs`, which needs pydantic → fails. Patching at `cli._async._call_submit_job` avoids this entirely since `cli._async` has no pydantic deps.
3. **`patch("cli._async.submit_job")` with top-level import**: If `submit_job` were imported at module level in `cli/_async.py`, patching `cli._async.submit_job` would work ONLY after the module is loaded. Adding `import cli._async` before `patch()` is required in that case, but the pydantic issue still bites at load time.
4. **Fake server.jobs module injection**: The `autouse` fixture uses `sys.modules` injection to provide `server.jobs.JobData` as a `MagicMock`. This must run before the command invokes `from server.jobs import JobData`.

**Useful Context for Future Agents:**
- The test environment does NOT have pydantic installed. Any code path in `cli/` that imports from `server.*` must use lazy imports (inside function bodies), never top-level.
- The pattern for mocking server-side dependencies in CLI tests is: inject fake modules into `sys.modules` via an `autouse` fixture, and patch `cli._async._call_submit_job` (not `server.submit.submit_job`) for controlling job IDs.
- `cli/_async.py` is the single place for async submission logic. If the JobData schema or submit_job signature changes, only `cli/_async.py` needs updating.
- The `_call_submit_job()` function is intentionally a thin wrapper (not inlined) — its sole purpose is to be patchable in tests.

## US-003 — `parallax jobs` subcommand group

**Summary:** The `parallax jobs` subcommand group (`list`, `status`, `watch`, `cancel`, `open`) was already fully implemented in `cli/commands/jobs.py` (registered in `cli/main.py`). The work for this story was writing comprehensive tests in `tests/test_cli_us003_it044.py` covering all five acceptance criteria.

**Key Decisions:**
- All tests patch the thin queue wrappers (`_call_list_jobs`, `_call_get_job`, `_call_cancel_job`) in `cli.commands.jobs` — these are isolated from the async `server.queue` so tests never need `aiosqlite` or a real SQLite DB.
- `subprocess.run` is patched directly for `open` command tests to avoid invoking `xdg-open`/`open` in CI.
- `sys.platform` is patched to test platform-specific opener logic (`xdg-open` on Linux, `open` on macOS).
- Rich table output truncates long strings (the `+00:00` timezone suffix in ISO timestamps may be ellipsised). Tests check date-portion substrings, not full timestamps.

**Pitfalls Encountered:**
1. **Rich truncates table cells**: `2026-04-05T12:00:00+00:00` is rendered as `2026-04-05T12:00:00…` in the terminal table. Asserting the full ISO string fails; checking `"2026-04-05"` passes.
2. **`time.sleep` must be patched for `watch` polling tests**: The `watch` command calls `time.sleep(1.0)` between polls; without patching it the test waits real seconds for each iteration.

**Useful Context for Future Agents:**
- `cli.commands.jobs` contains standalone thin wrappers (`_call_list_jobs`, `_call_get_job`, `_call_cancel_job`) that call `asyncio.run()` internally — they are the correct patch targets for any test of the jobs commands, avoiding all async complexity.
- The `cancel` command checks the job exists first via `_call_get_job` before calling `_call_cancel_job`; tests for "already terminal" must mock both (get returns a row, cancel returns False).
- `watch` polls until `status in {"completed", "failed", "cancelled"}`. It exits 0 for `completed` and `cancelled`, exits 1 for `failed`.

## US-004 — `python -m parallax` entry point

**Summary:** Created a `parallax/` package at repo root with `__init__.py` and `__main__.py` to enable `python -m parallax` invocation. Updated `pyproject.toml` to point the `parallax` script entry point to `cli.main:app` (was `cli.main:main`). Ran `uv sync` to reinstall the package with the new entry point.

**Key Decisions:**
- A thin `parallax/` package (two files: `__init__.py` + `__main__.py`) is the cleanest way to support `python -m parallax` while keeping the real CLI logic in `cli/`. It simply imports and calls `app` from `cli.main`.
- The entry point was changed from `cli.main:main` to `cli.main:app` as the Typer `app` instance is directly callable. The `main()` wrapper function is still present in `cli/main.py` but is no longer the entry point.
- Added `parallax*` to `[tool.setuptools.packages.find] include` so the package is picked up on install.

**Pitfalls Encountered:**
- None significant. The `pyproject.toml` entry point and `uv sync` flow worked cleanly. Both `uv run parallax --help` and `uv run python -m parallax --help` printed the Typer help tree without issues.

**Useful Context for Future Agents:**
- `python -m parallax` resolves through `parallax/__main__.py` → `cli.main:app`. Any change to the CLI app structure should keep this delegation intact.
- The `cli/__main__.py` (`python -m cli`) entry point is still present and also functional.
- Tests use `subprocess.run(["uv", "run", ...])` to verify the real installed entry points — this is the correct approach for integration-level AC verification without mocking.
