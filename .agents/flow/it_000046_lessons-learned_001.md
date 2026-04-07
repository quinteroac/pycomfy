# Lessons Learned — Iteration 000046

## US-001 — Add `--audio` option to `create video`

**Summary:** Added `--audio <path>` option to `parallax create video` that routes to the `ltx23/ia2v` pipeline when a model of `ltx23` is specified. Validation logic (unsupported model, missing file, missing `--input`) fires before async dispatch and before `ensure_env_on_path()` for fast-fail UX.

**Key Decisions:**
- Validation for `--audio` is performed before the `if async_mode:` branch so errors surface immediately, even in async mode, matching the existing `--input` file-check pattern.
- The `_ltx23` runner receives `audio` as an explicit keyword argument with default `None`; all other model runners absorb it via `**_`.
- `_AUDIO_SUPPORTED_MODELS` is a module-level `set` in `create.py`; extending support to new models is a one-line change.

**Pitfalls Encountered:**
- `CliRunner(mix_stderr=False)` is not supported by the pinned Typer version — use `result.output` directly (Typer mixes stdout/stderr by default in the test runner).
- Patching `cli._runners.video._ltx23` does **not** affect `RUNNERS["ltx23"]` because the dict holds a direct reference captured at module import time. Patch `cli._runners.video.RUNNERS` (the dict) instead.
- `resolve_models_dir` validates the directory exists on disk, so tests that rely on it must pass a real `tmp_path` subdirectory, not a fake string like `"/models"`.

**Useful Context for Future Agents:** When adding per-model CLI flags, follow the guard pattern: validate → exit early (with `err=True` + `raise typer.Exit(code=1)`) before `async_mode` dispatch. Pass the new kwarg through `RUNNERS[model](...)` using the existing `**_` catch-all in non-supporting runners.
