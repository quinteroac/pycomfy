# Refactor Completion Report — it_000044 pass 001

## Summary of changes

Two surgical fixes were applied based on the audit report:

1. **FR-5 — Pinned `bunqueue` version** (`packages/parallax_sdk/package.json`):
   Removed the `^` (caret) floating prefix from the `bunqueue` dependency version, changing it from `"^2.6.113"` to `"2.6.113"`. This ensures an exact, reproducible version is installed with no unintended minor/patch upgrades.

2. **US-005-AC04 — Bare output path in `wan22/t2v.py`** (`packages/parallax_cli/runtime/video/wan/wan22/t2v.py`):
   Changed the final stdout print from `print(f"Saved: {args.output}")` to `print(args.output)`. This aligns the wan22 pipeline's output with the AC-04 requirement and with `ltx2/t2v.py`'s established behaviour, ensuring `_run.ts` can reliably parse the last output line as a bare file path.

## Quality checks

| Check | Command | Outcome |
|-------|---------|---------|
| TypeScript typecheck (parallax_sdk) | `bun run typecheck` (in `packages/parallax_sdk`) | ✅ Passed — no errors |
| Python unit tests (progress reporter, CPU smoke, package structure) | `uv run pytest tests/test_progress_reporter.py tests/test_cpu_only_smoke.py tests/test_package_structure.py -q` | ✅ 12 passed |

> **Note:** `tests/test_ace_step_v1_5_pipelines.py::test_split_manifest_filenames_and_dests` fails on the unmodified branch (pre-existing, unrelated to this iteration's changes). All other tests that could be run in the CPU-only CI environment passed.

## Deviations from refactor plan

None. All two actionable fixes from the audit's `conclusionsAndRecommendations` section were applied exactly as specified. The optional `sampling_start` / per-step progress enhancement noted in the audit was explicitly marked optional and was not included in this refactor pass to keep changes surgical and risk-free.
