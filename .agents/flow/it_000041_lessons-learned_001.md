# Lessons Learned — Iteration 000041

## US-001 — Move examples/ into packages/parallax_cli/runtime/

**Summary:** All Python scripts under `examples/` were moved to `packages/parallax_cli/runtime/` preserving the full subdirectory structure. The original `examples/` directory was removed. Four test files that hard-coded `examples/` paths were updated to the new location.

**Key Decisions:**
- Used `rsync -a --exclude='__pycache__' --exclude='*.pyc'` to copy only source files, then `rm -rf examples/` to remove the originals — clean and atomic.
- Test files `test_us028_example_scripts.py`, `test_us030_wan21_example_scripts.py`, `test_us010_wan22_package_wiring.py`, and `test_us037_qwen_edit_2511_example.py` all had hard-coded `_REPO_ROOT / "examples" / ...` path constants that needed updating to `_REPO_ROOT / "packages" / "parallax_cli" / "runtime" / ...`.
- `test_us030` and `test_us010` also had stale flat-file paths (e.g. `video_wan21_t2v.py`, `wan22_t2v.py`) that never matched the actual subdirectory structure — corrected them to the proper nested paths.

**Pitfalls Encountered:**
- Several test files referenced flat `examples/` paths (`wan22_t2v.py`, `video_wan21_t2v.py`) that never existed; these tests were already failing before this story. After the move they were corrected to the real nested paths.
- `test_us010_wan22_package_wiring.py` used `_PIPELINES_VIDEO` (pointing at the library's `comfy_diffusion/pipelines/video/`) as a separate variable from `_EXAMPLES_DIR`; care was needed not to delete that constant while updating the examples path.

**Useful Context for Future Agents:**
- The canonical home for pipeline example scripts is now `packages/parallax_cli/runtime/` (not `examples/`). All tests and documentation should reference this path.
- CHANGELOG.md still contains historical references to the old flat `examples/` paths — these are intentional historical records and should not be changed.
- The self-referential usage strings inside the scripts themselves (e.g. `uv run python examples/...`) are stale docstring examples; future iterations may want to update them to the new path.
- Baseline (before this iteration): 186 test failures. After this story: 72 failures (all pre-existing, none introduced by this change).

## US-002 — Update registry.ts paths to use runtime/ prefix

**Summary:** Updated all `examples/` path strings in `packages/parallax_cli/src/models/registry.ts` to use `runtime/` prefix. Updated the corresponding test assertions in `registry.test.ts` to match.

**Key Decisions:**
- Changed paths in `IMAGE_SCRIPTS`, `VIDEO_MODEL_CONFIG`, and `AUDIO_SCRIPTS` constants — the only three objects that contained `examples/` references in the TypeScript source.
- The test file `registry.test.ts` hard-coded the exact path strings in assertions, so it needed to be updated in lockstep with the source.

**Pitfalls Encountered:**
- None. The change was straightforward — a direct string substitution from `examples/` to `runtime/` in 3 constants and 5 test assertions.

**Useful Context for Future Agents:**
- After US-001+US-002, there are no remaining `examples/` references in any TypeScript source under `packages/`.
- All 20 registry tests pass after the update.

## US-003 — runner.ts resolves scripts from runtimeDir config key

**Summary:** Added `runtimeDir?: string` to `ParallaxConfig` in `config.ts`, updated `runner.ts` to prefer `runtimeDir` over `repoRoot` when resolving script paths, and added `PARALLAX_RUNTIME_DIR` env var support to `readConfig`. The error message was updated to be more descriptive when neither directory is configured.

**Key Decisions:**
- `runner.ts` uses `const scriptBase = runtimeDir ?? repoRoot` — single-line fallback chain, no nesting.
- Added `PARALLAX_RUNTIME_DIR` env var support to `readConfig` to enable integration testing without config file manipulation (consistent with `PARALLAX_REPO_ROOT` pattern).
- Error message changed from `"Error: PARALLAX_REPO_ROOT is required"` to `"Error: no script directory configured — run \`parallax install\` to set runtimeDir, or set PARALLAX_REPO_ROOT"`.
- `cwd` for `Bun.spawn` also uses `scriptBase`, so it works correctly in both dev (repoRoot) and installed (runtimeDir) modes.

**Pitfalls Encountered:**
- Bun includes source code context in error stack traces, so when `uv` is not in PATH, the stack trace output contains the source line with the error message string. Tests using `.not.toContain("no script directory configured")` were failing because the string appeared inside the stack trace's source listing. Fixed by using `.not.toMatch(/^Error: no script directory configured/)` to check the string doesn't appear at the **start** of stderr.
- The old `runner.test.ts` created a fake repo at `"examples/image/generation/sdxl/t2i.py"` but that path was already stale after US-001/US-002; the test was rewritten to use `"runtime/image/generation/sdxl/t2i.py"`.

**Useful Context for Future Agents:**
- `PARALLAX_RUNTIME_DIR` env var → `runtimeDir` config field (mirrors `PARALLAX_REPO_ROOT` → `repoRoot`).
- `spawnPipeline` precedence: `runtimeDir` (installed) > `repoRoot` (dev/CI via env). Neither set → exit 1.
- When writing integration tests for runner.ts via the CLI subprocess approach, always use `not.toMatch(/^Error: .../)` rather than `not.toContain(...)` to guard against Bun stack trace source context false positives.

## US-004 — parallax install copies runtime/ to ~/.config/parallax/runtime/

**Summary:** Extended `install.ts` to copy the bundled `packages/parallax_cli/runtime/` directory to `~/.config/parallax/runtime/` using `fs.cpSync` with `{ recursive: true, force: true }`. `applyConfig` now writes `runtimeDir` into the config JSON. Non-interactive path logs progress to stdout; interactive path uses a `@clack/prompts` spinner.

**Key Decisions:**
- Used `cpSync(src, dest, { recursive: true, force: true })` — available in Bun/Node 16.7+, single-call recursive overwrite with no extra dependencies.
- `BUNDLED_RUNTIME_DIR = join(import.meta.dir, "../../runtime")` resolves correctly from `src/commands/install.ts` to `packages/parallax_cli/runtime/` in both dev and test runs.
- `INSTALLED_RUNTIME_DIR` is a module-level constant used in both `copyRuntime()` and `applyConfig()` to ensure a single source of truth for the destination path.
- The spinner in the interactive flow uses a separate `rs` variable to avoid variable shadowing with the later `s` spinner for the Python environment step.

**Pitfalls Encountered:**
- None. The change was straightforward: one `cpSync` call, one constant, one extra `writeConfig` field, one log line. Tests verified all ACs.

**Useful Context for Future Agents:**
- `INSTALLED_RUNTIME_DIR` = `~/.config/parallax/runtime/` — this is now the canonical installed runtime path, populated by `parallax install`.
- After US-003+US-004, `runner.ts` prefers `runtimeDir` (set by install) over `repoRoot` for resolving script paths. The full chain is: `parallax install` copies scripts → sets `runtimeDir` in config → `runner.ts` reads it.
- Tests backup/restore `~/.config/parallax/config.json` in `beforeEach`/`afterEach`. The US-004 tests additionally clean up `INSTALLED_RUNTIME_DIR` if it was absent before the test.
