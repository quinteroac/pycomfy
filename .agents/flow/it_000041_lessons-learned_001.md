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
