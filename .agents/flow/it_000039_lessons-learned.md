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
