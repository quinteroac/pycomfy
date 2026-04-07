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

## US-002 — Route to `ia2v` pipeline in the `_ltx23` runner

**Summary:** Added `cfg` forwarding to the `ia2v` branch inside `_ltx23`. The runner already had the `audio` routing logic from US-001 but was missing `c` in its explicit parameter list (it fell into `**_`), so `cfg` was never passed to `ia2v.run()`. Adding `c` to the signature and passing `cfg=c` completed the wiring.

**Key Decisions:**
- Only the `ia2v` branch needed `c`/`cfg` — the `i2v` and `t2v` branches already worked without it (they use their own defaults). AC03/AC04 required those paths to be left unchanged.
- The fix was minimal: one parameter added to `_ltx23`'s signature (`c`) and one kwarg added to the `ia2v.run()` call (`cfg=c`).

**Pitfalls Encountered:**
- The US-001 implementation had already scaffolded the audio routing but inadvertently left `cfg` un-forwarded. Always verify every required arg is explicitly passed — don't assume `**_` absorbers are propagating needed values downstream.
- When patching `comfy_diffusion.pipelines.video.ltx.ltx23.ia2v.run` inside tests, the patch target must match the exact import path used at call time (the lazy `from ... import run` pattern). Patching the module-level attribute works correctly here.

**Useful Context for Future Agents:** The `_ltx23` runner uses short param aliases (`w`, `h`, `n`, `f`, `c`) matching the pattern of all other runners in `cli/_runners/video.py`. When extending the ia2v/i2v/t2v call in the future, always capture `c` explicitly in the signature rather than relying on `**_`, otherwise cfg will silently be ignored.
