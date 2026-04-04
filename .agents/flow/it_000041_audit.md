# Audit Report — Iteration 000041 (PRD Index 001)

## Executive Summary

All 5 user stories were implemented. The `runtime/` directory is correctly created under `packages/parallax_cli/` with the full Python script tree, `registry.ts` uses `runtime/` prefixes with no `examples/` references remaining in TypeScript sources, `runner.ts` correctly implements `runtimeDir`-first resolution with `repoRoot` fallback and error-exit when neither is set, `install.ts` copies `runtime/` in both interactive and non-interactive flows, and the build scripts copy `runtime/` into `dist/runtime/`. Two partial-compliance gaps remain: (1) `examples/` was not fully removed — its directory structure and a `__pycache__` file linger; (2) `install.ts` resolves the bundled runtime via `import.meta.dir` only, which works at dev time but does not switch to `dirname(process.execPath)` for the distributed binary as required by the PRD.

---

## Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | **partially_comply** | `packages/parallax_cli/runtime/` exists with the full tree and all `.py` files. However, `examples/` is not removed — `examples/image/` subtree persists with a stale `__pycache__/edit_2511.cpython-312.pyc`. |
| FR-2 | **comply** | `IMAGE_SCRIPTS`, `VIDEO_MODEL_CONFIG`, and `AUDIO_SCRIPTS` all use `runtime/` prefixes. Zero `examples/` references in any `.ts` source file. |
| FR-3 | **comply** | `ParallaxConfig` gains `runtimeDir?: string`. `runner.ts` resolves `scriptBase = runtimeDir ?? repoRoot`, prefers `runtimeDir`, falls back to `repoRoot`, prints error and exits 1 if neither is set. |
| FR-4 | **partially_comply** | Both install flows call `copyRuntime()` and store `runtimeDir` in config. Spinner/logging present. Gap: `BUNDLED_RUNTIME_DIR` uses `import.meta.dir` only; PRD requires switching to `dirname(process.execPath)` in compiled-binary mode. |
| FR-5 | **comply** | `build:linux` and `build:mac` copy `runtime/` to `dist/runtime/`. `dist/` in `.gitignore` covers `dist/runtime/`. |
| FR-6 | **comply** | `cpSync` called with `{ recursive: true, force: true }` — idempotent by design. |

---

## Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | **partially_comply** | AC01 ✅ AC02 ✅ AC03 ❌ (`examples/` not removed/empty) AC04 ✅ |
| US-002 | **comply** | AC01 ✅ AC02 ✅ AC03 ✅ |
| US-003 | **comply** | AC01–AC05 all ✅ |
| US-004 | **partially_comply** | AC01 partial (no `dirname(process.execPath)`) AC02–AC06 ✅ |
| US-005 | **comply** | AC01–AC05 all ✅ |

---

## Minor Observations

- `examples/` has a residual `__pycache__/edit_2511.cpython-312.pyc` — the directory is not empty and should be removed (`git rm -r examples/`).
- `PARALLAX_RUNTIME_DIR` env var already overrides `runtimeDir` in `config.ts`, providing a clean workaround for testing the compiled binary.
- `build:win` is included as a bonus target (not required by PRD) — no correctness issue.
- `runner.ts` passes `cwd: scriptBase` to `Bun.spawn` — callers should be aware that relative paths in Python scripts resolve from `scriptBase`.
- The `outro` message in `runInteractive` is in Spanish ("Listo. Ejecuta: parallax create image --help") — inconsistent with the English-only convention.

---

## Conclusions and Recommendations

The prototype delivers the core capability. Two fixes are needed before the feature is production-ready:

1. **Clean up `examples/`**: Run `git rm -r examples/` to fully remove the directory (or at minimum delete the `__pycache__` artifact and empty directories).
2. **Fix `BUNDLED_RUNTIME_DIR` for compiled binary**: Replace `import.meta.dir` with `dirname(process.execPath)` (which equals the source directory in dev mode and the binary directory at runtime), so the compiled binary correctly locates its sibling `runtime/` directory on end-user machines.
3. **Fix the Spanish `outro` string**: Change `"Listo. Ejecuta: parallax create image --help"` to English.

---

## Refactor Plan

### Task 1 — Clean up `examples/` (FR-1 / US-001 AC03)
- **File**: repo root
- **Action**: `git rm -r examples/` to remove the directory and its contents from tracking. Add `**/__pycache__/` to `.gitignore` if not already covered.
- **Effort**: trivial

### Task 2 — Fix `BUNDLED_RUNTIME_DIR` for compiled binary (FR-4 / US-004 AC01)
- **File**: `packages/parallax_cli/src/commands/install.ts`
- **Action**: Replace the constant with a runtime-safe resolution:
  ```ts
  // Works in both dev (Bun interpreter) and compiled binary modes
  const BUNDLED_RUNTIME_DIR = join(dirname(process.execPath), "runtime");
  ```
  Or use a guard:
  ```ts
  import { dirname } from "path";
  // import.meta.dir is the compile-time source path; process.execPath is the actual binary location
  const BUNDLED_RUNTIME_DIR = Bun.argv[0].endsWith("bun")
    ? join(import.meta.dir, "../../runtime")   // dev: bun interpreter
    : join(dirname(process.execPath), "runtime"); // prod: compiled binary
  ```
- **Effort**: small

### Task 3 — Fix English `outro` string (minor)
- **File**: `packages/parallax_cli/src/commands/install.ts`
- **Action**: Change `outro("Listo. Ejecuta: parallax create image --help")` to `outro("Done. Run: parallax create image --help")`.
- **Effort**: trivial
