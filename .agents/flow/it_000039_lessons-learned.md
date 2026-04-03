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
