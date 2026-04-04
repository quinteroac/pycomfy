# Refactor Plan — Iteration 000041 (pass 001)

## Summary of changes

Three targeted fixes applied to close the two partial-compliance gaps and the locale inconsistency flagged by the audit:

1. **`packages/parallax_cli/src/commands/install.ts` — `BUNDLED_RUNTIME_DIR` binary-mode fix**
   Replaced the static `join(import.meta.dir, "../../runtime")` with a runtime detection pattern:
   ```ts
   const _devRuntimeDir = join(import.meta.dir, "../../runtime");
   const BUNDLED_RUNTIME_DIR = existsSync(_devRuntimeDir)
     ? _devRuntimeDir
     : join(dirname(process.execPath), "runtime");
   ```
   In dev mode, `import.meta.dir` resolves to the source directory where `runtime/` is two levels up — `existsSync` returns `true` and the dev path is used. In a compiled binary, `import.meta.dir` retains the compile-time source path which no longer exists on the target machine; `existsSync` returns `false` and `dirname(process.execPath)` (the directory holding the binary) is used instead, matching where the build scripts place `dist/runtime/`.

2. **`packages/parallax_cli/src/commands/install.ts` — outro locale fix (FR-English-only rule)**
   Changed the Spanish outro string `"Listo. Ejecuta: parallax create image --help"` to English: `"Done. Run: parallax create image --help"`.

3. **`examples/` directory removed**
   The `examples/image/edit/qwen/__pycache__/edit_2511.cpython-312.pyc` stale artifact and its parent directory tree were removed. The directory was not tracked in git (`.gitignore` already covers `__pycache__`), so removal was performed via `rm -rf`.

## Quality checks

| Check | Command | Result |
|-------|---------|--------|
| TypeScript typecheck | `bun run typecheck` (inside `packages/parallax_cli/`) | ✅ Pass — exit 0, no errors |
| Test suite | `bun test` (inside `packages/parallax_cli/`) | 179 pass / 112 fail — identical to pre-refactor baseline; no regressions introduced |

**Note on test failures:** The 112 failing tests are pre-existing failures from earlier iterations (ace_step model flags US-002, create audio US-003/US-004). Verified by running the test suite on the unmodified pre-refactor commit (stash round-trip) — same 112 failures. My changes touch only `install.ts`; none of the failing tests exercise that module.

## Deviations from refactor plan

None. All three items recommended in `conclusions_and_recommendations` of the audit report were applied as specified:
- examples/ stale directory removed ✅
- BUNDLED_RUNTIME_DIR fixed to support compiled binary mode ✅
- Outro locale corrected to English ✅
