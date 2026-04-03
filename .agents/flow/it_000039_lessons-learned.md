# Lessons Learned — Iteration 000039

## US-001 — sdxl image generation via CLI

**Summary:** Wired `parallax create image --model sdxl` to spawn `uv run python examples/image/generation/sdxl/t2i.py` as a subprocess. Added the same dispatch for `anima` and `z_image` since all three scripts existed. Added `--models-dir` option and two guard checks: `PYCOMFY_MODELS_DIR`/`--models-dir` and `PARALLAX_REPO_ROOT`. Unimplemented models (`flux_klein`, `qwen`) still fall through to `notImplemented()`.

**Key Decisions:**
- `spawnPipeline()` is a standalone async helper that calls `process.exit(await proc.exited)` — this cleanly propagates the subprocess exit code (AC03) and the `never`-returning `process.exit` avoids TypeScript unreachable-code issues.
- `IMAGE_SCRIPTS` is a `Partial<Record<string, string>>` so the model→script mapping is explicit and exhaustive; a missing key falls through to `notImplemented()` automatically.
- Validation order: `PYCOMFY_MODELS_DIR`/`--models-dir` first (per AC04), then `PARALLAX_REPO_ROOT` inside `spawnPipeline` (required to build the absolute script path).
- `z_image` turbo omits `--negative-prompt` and `--cfg` arguments per FR-7 — the flag-building branch checks `opts.model !== "z_image"`.

**Pitfalls Encountered:**
- The US-007 test `"create image with valid flags prints stub message and exits 0"` broke after replacing the stub handler. It was replaced with tests for the new validation behaviour and a separate test verifying `flux_klein` still hits `notImplemented()`.
- `runCLI` inherits the test process environment; if `PYCOMFY_MODELS_DIR` or `PARALLAX_REPO_ROOT` happen to be set by the developer's shell, validation tests would behave unexpectedly. Added `runCLIWithEnv` helper that explicitly overrides (or unsets) specific env vars for deterministic tests.

**Useful Context for Future Agents:**
- The same `spawnPipeline` helper and `IMAGE_SCRIPTS`-style dispatch pattern should be reused verbatim for `create video` and `create audio` (US-002 through US-004 in PRD 001 and future PRDs).
- The `--models-dir` option must appear on every `create` sub-command that dispatches to a Python script; it is passed as `--models-dir <path>` to the Python example.
- The `PARALLAX_REPO_ROOT` → script path resolution must happen inside `spawnPipeline`, not in the action handler, so all callers benefit from the same check.
- `bun test packages/parallax_cli/src/index.test.ts` is the correct command to run CLI tests.
- `bun tsc --noEmit` from `packages/parallax_cli/` verifies typecheck for the CLI package.

## US-002 — anima image generation via CLI

**Summary:** Added a `parallax CLI — anima image generation (US-002)` describe block to `packages/parallax_cli/src/index.test.ts` covering all acceptance criteria. The CLI dispatch (`IMAGE_SCRIPTS["anima"]`), the example script (`examples/image/generation/anima/t2i.py`), and the pipeline (`comfy_diffusion/pipelines/image/anima/t2i.py`) were already implemented by the US-001 agent — only test coverage was missing.

**Key Decisions:**
- Tests mirror the `US-001-it39` sdxl block exactly, replacing `sdxl` with `anima` and adjusting the prompt text.
- A dedicated test asserts that `--negative-prompt` IS forwarded for anima (flag-parsing succeeds and only `PARALLAX_REPO_ROOT` remains as the error), explicitly contrasting with `z_image` which omits it.
- All flag-forwarding tests are validated indirectly: if Commander accepted all flags without error, the only failure should be the missing `PARALLAX_REPO_ROOT` env var.

**Pitfalls Encountered:**
- None. The implementation was already in place; only test assertions needed to be added.

**Useful Context for Future Agents:**
- When US-001 was implemented, the agent pre-wired `anima` and `z_image` into `IMAGE_SCRIPTS` alongside `sdxl`, so US-002 and US-003 only needed test coverage — not production code changes.
- The pattern `runCLIWithEnv([...], { PARALLAX_REPO_ROOT: undefined })` is the canonical way to test flag-forwarding without actually spawning a real subprocess — it triggers the `PARALLAX_REPO_ROOT` guard after all flag parsing succeeds.

## US-003 — z_image image generation via CLI

**Summary:** Added 9 tests to `packages/parallax_cli/src/index.test.ts` covering all 5 acceptance criteria for the z_image turbo pipeline CLI integration. No production code changes were needed — the CLI already had z_image wired in `IMAGE_SCRIPTS` and the special-casing for dropping `--negative-prompt`/`--cfg`.

**Key Decisions:**
- Used temporary directories with fake Python scripts (created via `fs/promises` + `os.tmpdir()`) to test subprocess argument forwarding (AC02) and non-forwarding (AC03) without needing real model weights.
- For AC03, the fake script exits code 2 if it receives forbidden flags — the test asserts exit code 0, giving a clear signal if the CLI regresses and starts forwarding those flags.
- For AC04, the subprocess exit code propagation was tested both with a bad `PARALLAX_REPO_ROOT` (non-zero exit) and a fake script that exits with code 3 (exact propagation).
- Added `import { mkdtemp, mkdir, writeFile, rm } from "fs/promises"` and `import { tmpdir } from "os"` to the test file — both are Node-compatible APIs supported by Bun.

**Pitfalls Encountered:**
- The describe block name `(US-003)` conflicts with the existing `parallax CLI — edit subcommand help (US-003)` block. Used `(US-003-it39)` as a suffix to avoid confusion — Bun doesn't care about duplicate describe labels, but it is confusing for humans.
- The `makeFakeZImageRoot` helper must create the full relative path `examples/image/generation/z_image/turbo.py` under the temp dir, matching the `IMAGE_SCRIPTS` map in `index.ts`.

**Useful Context for Future Agents:**
- The fake-script pattern (create a Python script in a temp dir, point `PARALLAX_REPO_ROOT` at it) is the canonical way to test exact argument forwarding in this CLI test suite. Reuse `makeFakeZImageRoot` or an equivalent helper for other pipeline models.
- All temp dirs are cleaned up via `rm(tmpRoot, { recursive: true, force: true })` in a `finally` block — this pattern should be followed for any future subprocess tests.
- The test file now requires `fs/promises` and `os` imports — they are at the top of the file alongside the existing `bun:test` and `path` imports.

## US-004 — models-dir resolution

**Summary:** Added a dedicated `describe("parallax CLI — models-dir resolution (US-004)")` test block with 5 tests covering all acceptance criteria. No production code changes were needed — the implementation was already complete in `src/index.ts` (the US-001 agent pre-wired `--models-dir` and `PYCOMFY_MODELS_DIR` resolution).

**Key Decisions:**
- Added a `makeFakeSdxlRoot(scriptBody)` helper (mirroring `makeFakeZImageRoot`) so tests can use `sdxl` (the canonical, fully-implemented model) for subprocess argument inspection.
- AC04 tests verify the exact argument by splitting stdout on spaces and checking that the token after `--models-dir` matches the expected path.
- AC02 test asserts both that `--flag/models` is present AND that `/env/models` is absent, making the precedence rule explicit.

**Pitfalls Encountered:**
- None. The implementation was complete; only explicit test coverage under a US-004 describe block was missing.

**Useful Context for Future Agents:**
- When the user story says "AC03: If neither is set, the CLI prints…", the acceptance criteria text was truncated in the PRD. The implemented error message is `"Error: --models-dir or PYCOMFY_MODELS_DIR is required"` — use this exact string for future tests or error-message changes.
- `makeFakeSdxlRoot` is now a reusable helper in the test file alongside `makeFakeZImageRoot`. Use it whenever tests need to verify subprocess argument forwarding for SDXL.
- The `PYCOMFY_MODELS_DIR` env var is only checked on `create image`; other commands (`create video`, `create audio`, `edit image`, `edit video`) do not yet dispatch to a Python subprocess and therefore don't validate it.

## US-001 — ltx2 video generation via CLI

**Summary:** Implemented `parallax create video --model ltx2` to spawn `examples/video/ltx/ltx2/t2v.py` with forwarded flags. Added `VIDEO_SCRIPTS` map and `--models-dir` option to `create video`. The `--cfg` CLI flag is translated to `--cfg-pass1` for ltx2 (since the pipeline uses two-pass CFG and only exposes `--cfg-pass1`/`--cfg-pass2`).

**Key Decisions:**
- Added a separate `VIDEO_SCRIPTS` map (parallel to `IMAGE_SCRIPTS`) for clean extensibility when other video models are implemented.
- `--models-dir` was missing from the `create video` command definition — added it as an optional flag consistent with `create image`.
- The `--cfg` → `--cfg-pass1` remapping is model-specific, guarded by `if (opts.model === "ltx2")`. Future models may use different flag names.

**Pitfalls Encountered:**
- The AC03 test for "bare `--cfg` is not forwarded" requires checking `"--cfg " ` (with trailing space) in stdout to distinguish from `"--cfg-pass1"` which contains `--cfg` as a prefix. A Python `in sys.argv` check (exact element match) is cleaner and avoids this substring ambiguity.
- The stub test `US-007-AC01/02` used `wan21` (not `ltx2`), so it continued to hit `notImplemented()` after `ltx2` was wired up — no change needed to that test.

**Useful Context for Future Agents:**
- `makeFakeLtx2Root` is now a reusable test helper alongside `makeFakeSdxlRoot` and `makeFakeZImageRoot`. Add `makeFake<Model>Root` helpers for each new video model as it is implemented.
- The `PYCOMFY_MODELS_DIR` env var resolution for `create video` now follows the same pattern as `create image` — `opts.modelsDir ?? process.env.PYCOMFY_MODELS_DIR`.
- `VIDEO_SCRIPTS` is the single source of truth for which video models are live vs. stubbed. Add new entries there to enable dispatch without touching the action logic.

## US-002 — ltx23 video generation via CLI

**Summary:** Wired `parallax create video --model ltx23` to spawn `uv run python examples/video/ltx/ltx23/t2v.py`. Added `ltx23` to `VIDEO_SCRIPTS` and updated the args-building logic so `--steps` is skipped for ltx23 (distilled, no step count) while `--cfg` is forwarded as a bare `--cfg` flag (unlike ltx2 which remaps to `--cfg-pass1`).

**Key Decisions:**
- `VIDEO_SCRIPTS["ltx23"] = "examples/video/ltx/ltx23/t2v.py"` — one-liner addition, no new helpers.
- Args building uses `if (opts.model !== "ltx23") { args.push("--steps", ...) }` to skip `--steps` for the distilled model and `if (opts.model === "ltx2") { ... } else { args.push("--cfg", ...) }` for CFG routing. This is forward-compatible: future models automatically get `--steps` and `--cfg` by default.
- Added `makeFakeLtx23Root` test helper (mirrors `makeFakeLtx2Root`) and a new `describe` block with 9 tests covering all 5 ACs.

**Pitfalls Encountered:**
- None. The existing args-building logic was clean enough that only targeted additions were needed.

**Useful Context for Future Agents:**
- The `create video` action builds args in two phases: (1) common flags (`--models-dir`, `--prompt`, `--width`, `--height`, `--length`, `--output`); (2) model-specific flags (`--steps` conditional on model, `--cfg`/`--cfg-pass1` per model). Any new distilled model should add another `!== "<model>"` guard to the `--steps` skip, or refactor into a per-model config object.
- `examples/video/ltx/ltx23/t2v.py` already existed and accepts `--cfg`, `--prompt`, `--width`, `--height`, `--length`, `--seed`, `--output`, `--models-dir` — no Python changes were needed.
- The `--steps` AC test using a fake Python script that exits 2 if it receives `--steps` is the canonical pattern for "flag must NOT be forwarded" assertions in this test suite.

## US-003 — wan21 video generation via CLI

**Summary:** Wired `parallax create video --model wan21` to spawn `uv run python examples/video/wan/wan21/t2v.py`. Added `wan21` to `VIDEO_SCRIPTS` and added a `US-003` test suite with 8 tests covering all acceptance criteria. Updated the US-007 stub test which previously expected `wan21` to print "not yet implemented" — changed it to use `wan22` instead.

**Key Decisions:**
- `VIDEO_SCRIPTS["wan21"] = "examples/video/wan/wan21/t2v.py"` — one-liner addition. The existing `create video` action already handles `--steps` and `--cfg` for the default (non-ltx2, non-ltx23) branch, so `wan21` required zero logic changes in the action handler.
- The US-007 stub test targeted `wan21` specifically; it was updated to `wan22` (still unimplemented) to preserve coverage of the stub path.
- Added `makeFakeWan21Root` test helper following the established `makeFake<Model>Root` pattern.

**Pitfalls Encountered:**
- The US-007 stub test `"create video with valid flags prints stub message and exits 0"` would have failed after wiring `wan21` — it was updated proactively before running tests.
- The user story AC05 ("Typecheck / lint passes") was verified via `bun run typecheck` in `packages/parallax_cli/`.

**Useful Context for Future Agents:**
- `VIDEO_SCRIPTS` is the single source of truth for which video models dispatch to real Python scripts vs. fall through to `notImplemented()`. Adding a new model is a one-line change to this map.
- The default branch in the args-building logic forwards both `--steps` and `--cfg` — this is correct for `wan21`, `wan22`, and any future standard-sampler video model.
- The US-007 stub describe block should always test a model that is NOT yet in `VIDEO_SCRIPTS`. Update it to a different unimplemented model (currently `wan22`) when wiring new models.

## US-004 — wan22 video generation via CLI

**Summary:** Wired `parallax create video --model wan22` to spawn `uv run python examples/video/wan/wan22/t2v.py`. Added `wan22` to `VIDEO_SCRIPTS` (one-liner). Updated the US-007 stub test that was targeting `wan22` — moved it to `edit video --model wan22` (still unimplemented). Added a `US-004` test suite with `makeFakeWan22Root` helper following the exact wan21/US-003 pattern.

**Key Decisions:**
- The existing `create video` action handler already forwards `--steps` and `--cfg` by default (the only exceptions are `ltx23` which skips `--steps`, and `ltx2` which uses `--cfg-pass1`). `wan22` uses the standard path with no special-casing needed.
- `VIDEO_SCRIPTS` now has entries for all four models in `MODELS["create video"]`, leaving no "create video" stub. The US-007 stub test was redirected to `edit video --model wan22` to preserve stub-path coverage.
- Test helper `makeFakeWan22Root` matches the `makeFakeWan21Root` naming convention established in US-003.

**Pitfalls Encountered:**
- None. This was a pure additive change — one line in `VIDEO_SCRIPTS`, a one-line stub-test update, and a new test suite mirroring US-003.

**Useful Context for Future Agents:**
- After this story, all four `create video` models (`ltx2`, `ltx23`, `wan21`, `wan22`) are wired. Any future "create video" stub test must use a newly added model or a different command (e.g., `create audio`, `edit video`).
- The US-007 stub test for `edit video --model wan22` will need to be updated if/when `wan22` i2v or another `edit video` command is implemented.

## US-005 — models-dir and repo-root resolution

**Summary:** Added a dedicated `describe` block for US-005 covering `create video` models-dir and repo-root resolution. The production code in `index.ts` already implemented all required behaviour — `--models-dir` overrides `PYCOMFY_MODELS_DIR`, both-missing exits 1 with a clear error, missing `PARALLAX_REPO_ROOT` exits 1 with a clear error, and the resolved path is forwarded as `--models-dir <path>` to the subprocess. The story required only tests, not new production code.

**Key Decisions:** Used `wan21` as the representative video model for the US-005 test suite because it has no special-casing (no `--cfg-pass1`, no distilled `--steps` suppression), keeping the tests focused on resolution logic without noise from per-model arg transforms.

**Pitfalls Encountered:** None — production code was already complete. The acceptance criteria text was truncated (AC02/AC03 cut off mid-sentence), but the expected error messages were unambiguous from the existing test patterns.

**Useful Context for Future Agents:**
- The models-dir / repo-root resolution logic lives in the `create video` action handler (lines ~160–164 in `index.ts`) and in `spawnPipeline()` (~line 50–62). Both checks are shared across all `create image` and `create video` models.
- If a new `create video` or `create audio` model is added that changes models-dir resolution behaviour, update the US-005 describe block in `index.test.ts` accordingly.
- `makeFakeWan21Root()` helper (defined around line 1055 of `index.test.ts`) creates a minimal fake PARALLAX_REPO_ROOT suitable for subprocess forwarding tests.
