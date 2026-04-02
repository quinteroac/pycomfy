# Iteration 000037 — Refactor Report

## Summary of changes

### RP-1 — `--output-prefix` alignment

Renamed the `--output` CLI argument in `examples/image/edit/qwen/edit_2511.py` to `--output-prefix`, aligning it with:

- The PRD specification (FR-5 / US-002-AC02) which describes `<output-prefix>.png` as the output pattern.
- The established convention in peer scripts (`examples/image/edit/qwen/layered_i2l.py`, `examples/image/generation/qwen/layered_t2l.py`).

Changes made:

- **`examples/image/edit/qwen/edit_2511.py`**:
  - Argument `--output` (default `"qwen_edit_2511_output.png"`) → `--output-prefix` (default `"qwen_edit_2511_output"`).
  - Output path construction changed from `Path(args.output)` → `Path(f"{args.output_prefix}.png")`.
  - Docstring CLI example updated to show `--output-prefix edited_output`.

- **`tests/test_us037_qwen_edit_2511_example.py`**:
  - All `"--output"` invocations in test patches updated to `"--output-prefix"` with prefix values (`.png` stripped).
  - `test_has_required_cli_flags`: updated required flag from `"--output"` to `"--output-prefix"` and reformatted to stay within the 99-character line limit.
  - `test_default_output_path_is_qwen_edit_2511_output_png`: updated assertion to validate that the source contains `"qwen_edit_2511_output"` (the prefix) and that `.png` is appended dynamically.
  - Removed now-unused `output_path` local variable in `test_inference_with_image_prompt_exits_zero`.
  - Fixed two pre-existing lint issues: removed unused `import argparse`; suppressed unused `patch_target_download` variable with `# noqa: F841`.

### RP-2 and RP-3 — Deferred

Pre-existing mypy debt in `comfy_diffusion/vae.py` and `comfy_diffusion/video.py`, and the missing `esrgan` example file referenced by `tests/test_esrgan_example_uses_public_api.py`, are explicitly deferred to a dedicated cleanup iteration as recommended by the audit.

---

## Quality checks

| Check | Scope | Outcome |
|---|---|---|
| `uv run ruff check examples/image/edit/qwen/edit_2511.py tests/test_us037_qwen_edit_2511_example.py` | Modified files | ✅ All checks passed |
| `uv run mypy examples/image/edit/qwen/edit_2511.py --ignore-missing-imports` | Modified file | ✅ No errors in `edit_2511.py`; 11 pre-existing errors in unrelated modules (`vae.py`, `video.py`, `sampling.py`, `conditioning.py`) — unchanged from pre-refactor baseline |
| `uv run pytest tests/test_us037_qwen_edit_2511_example.py -v` | Iteration test suite | ✅ 31/31 passed |

**Notes:**
- The pre-existing mypy errors in `comfy_diffusion/vae.py` and `comfy_diffusion/video.py` were acknowledged in the audit as unrelated to this iteration and remain unchanged.
- The pre-existing test failure in `tests/test_esrgan_example_uses_public_api.py` (missing example file) and `tests/test_ltxv2_t2sv_example.py` are unrelated to this iteration.

---

## Deviations from refactor plan

None. All actionable items from the refactor plan were applied:

- **RP-1** was fully implemented (`--output-prefix` rename with auto `.png` appending).
- **RP-2** and **RP-3** were intentionally deferred per the audit's recommendation ("address in a dedicated cleanup iteration") — this is not a deviation but a plan-prescribed deferral.
