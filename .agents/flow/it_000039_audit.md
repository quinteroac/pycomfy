# Audit Report — Iteration 000039 / PRD-001

> Image Generation via parallax-cli (sdxl, anima, z_image)

---

## Executive Summary

The implementation in `packages/parallax_cli/src/index.ts` fully satisfies all functional requirements and user story acceptance criteria defined in PRD-001. The `create image` action handler correctly dispatches to the three Python scripts (`sdxl`, `anima`, `z_image`), resolves `PARALLAX_REPO_ROOT` for script path construction, spawns subprocesses via `uv run python` with inherited stdio, propagates the child's exit code, and handles models-dir resolution from either the `--models-dir` flag or `PYCOMFY_MODELS_DIR` env var with a clear error when neither is present. The z_image turbo special case (omitting `--negative-prompt` and `--cfg`) is correctly implemented. TypeScript type-checking passes cleanly.

---

## Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | ✅ comply | `IMAGE_SCRIPTS` map routes `sdxl` / `anima` / `z_image` to the correct `.py` paths. Unmapped models fall through to `notImplemented()`. |
| FR-2 | ✅ comply | `spawnPipeline()` checks `PARALLAX_REPO_ROOT`, prints `Error: PARALLAX_REPO_ROOT is required` and exits 1 when unset. |
| FR-3 | ✅ comply | Subprocess spawned as `['uv', 'run', 'python', <scriptPath>, ...args]`. |
| FR-4 | ✅ comply | `opts.modelsDir ?? process.env.PYCOMFY_MODELS_DIR`; explicit flag overrides env var. Resolved value passed as `--models-dir <path>` to child. |
| FR-5 | ✅ comply | `Bun.spawn` called with `{ stdin: 'inherit', stdout: 'inherit', stderr: 'inherit' }`. |
| FR-6 | ✅ comply | `process.exit(await proc.exited)` propagates child exit code. |
| FR-7 | ✅ comply | Guard `if (opts.model !== 'z_image')` prevents `--negative-prompt` and `--cfg` from being appended for z_image. |

---

## Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | ✅ comply | `sdxl` dispatches to `sdxl/t2i.py`. All flags forwarded. PYCOMFY_MODELS_DIR / `--models-dir` guard prints correct error and exits 1. `tsc --noEmit` passes. |
| US-002 | ✅ comply | `anima` dispatches to `anima/t2i.py`. Same flag forwarding as sdxl. Exit code propagated. `tsc --noEmit` passes. |
| US-003 | ✅ comply | `z_image` dispatches to `z_image/turbo.py`. Width/height/steps/seed/output forwarded; `--negative-prompt` and `--cfg` silently omitted. `tsc --noEmit` passes. |
| US-004 | ✅ comply | `PYCOMFY_MODELS_DIR` read from env; `--models-dir` flag overrides. Error message matches spec exactly. Resolved path passed to subprocess. |

---

## Minor Observations

1. The `--prompt` flag is forwarded to z_image turbo even though the US-003 acceptance criteria only lists `--width/--height/--steps/--seed/--output`. This is correct behaviour (`turbo.py` requires `--prompt`) but the AC omission may cause confusion during manual review.
2. The `--cfg` default value (`"7"`) is always present in `opts.cfg` via commander defaults. For non-z_image models, `--cfg` is always forwarded even if the user did not explicitly set it. This is intentional and acceptable.

---

## Conclusions and Recommendations

The prototype fully satisfies PRD-001 with no gaps. All 7 functional requirements and all 4 user story acceptance criteria are met. TypeScript type-checking passes with zero errors. Recommended action: **proceed to the Refactor phase**. No corrective changes are required before refactoring.

---

## Refactor Plan

No corrective refactoring is required for PRD-001 compliance. During the Refactor phase, the following quality improvements are recommended:

1. **Unit tests** — Add critical-path tests (per project convention) covering:
   - `PARALLAX_REPO_ROOT` missing → exit 1 with correct message.
   - `PYCOMFY_MODELS_DIR` missing and no `--models-dir` → exit 1 with correct message.
   - z_image model → `--cfg` and `--negative-prompt` absent from subprocess args.
   - Correct script path selected for each of the three models.
2. **Type safety** — `IMAGE_SCRIPTS` is typed `Partial<Record<string, string>>` which requires a null-check before use. The existing `if (!script)` guard is sufficient but could be strengthened with a non-null assertion after the guard.
3. **Documentation** — Consider adding a short doc comment above `IMAGE_SCRIPTS` explaining the z_image CFG/negative-prompt exclusion rationale for future maintainers.
