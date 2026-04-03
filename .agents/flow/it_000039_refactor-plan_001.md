# Refactor Report — Iteration 000039 (Pass 001)

## Summary of changes

### RT-002 (low priority) — Doc comment for z_image exclusion

In `packages/parallax_cli/src/index.ts`, the comment above the z_image guard block was replaced with a more precise explanation:

**Before:**
```ts
// z_image turbo has no --negative-prompt or --cfg parameters
```

**After:**
```ts
// z_image turbo.py accepts neither --negative-prompt nor --cfg; omit both to
// avoid "unrecognized arguments" errors from the Python script.
```

This makes the rationale immediately clear to future readers (the flag omission is required by the script's argument parser, not merely optional).

### RT-001 (high priority) — Critical-path unit tests

In `packages/parallax_cli/src/index.test.ts`:

1. Added `makeFakeAnimaRoot(scriptBody)` helper — creates a temporary `PARALLAX_REPO_ROOT` tree with a fake `examples/image/generation/anima/t2i.py` script, following the same pattern already used by `makeFakeSdxlRoot` and `makeFakeZImageRoot`.

2. Added two new test cases inside the existing `"parallax CLI — anima image generation (US-002)"` describe block:
   - **RT-001: correct script path** — places the fake script at the exact path `IMAGE_SCRIPTS` maps to (`examples/image/generation/anima/t2i.py`) and asserts exit 0; if the CLI used the wrong path, Bun.spawn would fail to find the script and exit non-zero.
   - **RT-001: all flags forwarded** — uses a fake script that prints `sys.argv` and asserts all generation flags (`--models-dir`, `--prompt`, `--negative-prompt`, `--width`, `--height`, `--steps`, `--cfg`, `--seed`, `--output`) appear in the subprocess output.

The existing tests already covered PARALLAX_REPO_ROOT missing → exit 1, PYCOMFY_MODELS_DIR missing → exit 1, z_image flag exclusion, and sdxl script path; anima script-path invocation was the only gap.

Total tests: **190 → 192** (+2 new tests, +20 new `expect()` calls).

## Quality checks

| Check | Command | Result |
|-------|---------|--------|
| TypeScript typecheck | `bun run typecheck` (`tsc --noEmit`) | ✅ Pass — no errors |
| Unit tests | `bun test` | ✅ 192 pass, 0 fail (621 expect() calls) |

Both checks run inside `packages/parallax_cli/`.

## Deviations from refactor plan

None. Both RT-001 and RT-002 were applied in full. The existing test suite already covered three of the four RT-001 bullet points (PARALLAX_REPO_ROOT missing, PYCOMFY_MODELS_DIR missing, z_image flag exclusion); only the anima script-path invocation test was added as the missing piece.
