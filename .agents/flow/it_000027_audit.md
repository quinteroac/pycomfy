# Iteration 000027 — Audit Report

## Executive summary

All 12 user stories are implemented and all 216 CPU-only tests pass. Ruff lint passes with no errors. Four wrapper functions (get_video_components, lotus_conditioning, ltxv_add_guide, text_generate_ltx2_prompt) omit the `ensure_comfyui_on_path()` call that the rest of the codebase uses before importing `comfy_extras` modules, creating a runtime fragility. Output unwrapping is inconsistent, and mypy reports two errors in new code. Overall the iteration is substantially complete with minor consistency and robustness gaps.

## Verification by FR

| FR | Assessment | Notes |
|----|------------|-------|
| FR-1 | Comply | `resize_image_mask` in image.py, lazy imports ResizeImageMaskNode |
| FR-2 | Comply | `resize_images_by_longer_edge` in image.py, lazy imports ResizeImagesByLongerEdgeNode |
| FR-3 | Comply | `empty_image` in image.py, lazy imports nodes.EmptyImage via ensure_comfyui_on_path |
| FR-4 | Comply | `math_expression` in image.py, lazy imports MathExpressionNode |
| FR-5 | Comply | `get_video_components` in video.py, lazy imports GetVideoComponents |
| FR-6 | Comply | `ltxv_add_guide` in controlnet.py, lazy imports LTXVAddGuide |
| FR-7 | Comply | `canny` in image.py, lazy imports Canny |
| FR-8 | Comply | `lotus_conditioning` in controlnet.py, lazy imports LotusConditioning |
| FR-9 | Comply | `set_first_sigma` in sampling.py, lazy imports SetFirstSigma |
| FR-10 | Comply | `image_invert` in image.py, lazy imports nodes.ImageInvert via ensure_comfyui_on_path |
| FR-11 | Comply | `text_generate_ltx2_prompt` in textgen.py, lazy imports TextGenerateLTX2Prompt |
| FR-12 | Comply | No path parameters in new functions |
| FR-13 | Comply | All comfy imports deferred to call time |
| FR-14 | Partially comply | `math_expression` triggers mypy error on return type annotation |
| FR-15 | Comply | DWPreprocessor not included |

## Verification by US

| US | Assessment | Notes |
|----|------------|-------|
| US-001 | Comply | All ACs met |
| US-002 | Comply | All ACs met |
| US-003 | Comply | All ACs met |
| US-004 | Partially comply | mypy error: Returning Any from int|float declared function (AC05) |
| US-005 | Partially comply | Missing ensure_comfyui_on_path(), fragile .args[] unwrapping |
| US-006 | Partially comply | Missing ensure_comfyui_on_path() |
| US-007 | Comply | All ACs met |
| US-008 | Partially comply | Missing ensure_comfyui_on_path(), unused model/image params |
| US-009 | Comply | All ACs met |
| US-010 | Comply | All ACs met |
| US-011 | Partially comply | Missing ensure_comfyui_on_path(), no V3 output unwrapping |
| US-012 | Comply | All 216 tests pass, lint clean |

## Minor observations

1. `get_video_components` (video.py:259), `lotus_conditioning` (controlnet.py:141), `ltxv_add_guide` (controlnet.py:171), and `text_generate_ltx2_prompt` (textgen.py:176) perform inline `from comfy_extras.*` imports without first calling `ensure_comfyui_on_path()`. Every other wrapper in the codebase uses this call.
2. `get_video_components` uses `output.args[0]`/`output.args[1]` for output extraction, while all other wrappers use the standard `getattr(result, "result", result)[n]` pattern.
3. `text_generate_ltx2_prompt` uses `result[0]` directly without the V3-safe `getattr(result, "result", result)` fallback.
4. `lotus_conditioning` accepts `model` and `image` parameters but passes neither to `LotusConditioning.execute()`.
5. mypy reports `Returning Any from function declared to return int | float` on image.py:344 (`math_expression`).
6. mypy reports `Unused type: ignore comment` on video.py:259 (`get_video_components` import line).
7. image.py and sampling.py wrappers use dedicated `_get_*_type()` helper functions for lazy imports, while controlnet.py, video.py, and textgen.py wrappers do inline imports.

## Conclusions and recommendations

The iteration delivers all 12 user stories with passing tests and clean lint. Five user stories (US-004, US-005, US-006, US-008, US-011) are partially compliant. Recommended refactors:

1. Add `ensure_comfyui_on_path()` + `_get_*_type()` helpers for `get_video_components`, `lotus_conditioning`, `ltxv_add_guide`, and `text_generate_ltx2_prompt`.
2. Standardize output unwrapping to `getattr(result, "result", result)[n]` across all wrappers.
3. Fix the mypy return-type annotation in `math_expression` (cast the return value).
4. Remove the unused `type: ignore` comment in video.py:259.
5. Reconsider whether `lotus_conditioning` should accept `model`/`image` if it never uses them.

## Refactor plan

1. **video.py — `get_video_components`**: Extract `_get_get_video_components_type()` helper with `ensure_comfyui_on_path()`. Replace `.args[]` unwrapping with standard `_unwrap_node_output`-style pattern returning two values.
2. **controlnet.py — `lotus_conditioning`**: Extract `_get_lotus_conditioning_type()` helper with `ensure_comfyui_on_path()`. Use standard `_unwrap_node_output` pattern.
3. **controlnet.py — `ltxv_add_guide`**: Extract `_get_ltxv_add_guide_type()` helper with `ensure_comfyui_on_path()`. Use standard `_unwrap_node_output` pattern.
4. **textgen.py — `text_generate_ltx2_prompt`**: Extract `_get_text_generate_ltx2_prompt_type()` helper with `ensure_comfyui_on_path()`. Use `getattr(result, "result", result)[0]` for V3-safe unwrapping.
5. **image.py — `math_expression`**: Cast return value with `cast(int | float, raw[0])` to satisfy mypy.
6. **video.py:259**: Remove unused `# type: ignore[import-untyped]` comment.
7. **Update tests**: Adjust test mocks to match new `_get_*_type()` helper patterns where needed.
