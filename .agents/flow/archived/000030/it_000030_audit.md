# Audit — it_000030 (WAN 2.1 Video Pipelines)

## 1. Executive Summary

All 10 functional requirements and 6 user stories for iteration 000030 are fully implemented and compliant. The `empty_wan_latent_video` helper, all three `wan21` pipeline modules (t2v, i2v, flf2v), the sub-package wiring, and the three example scripts are present with correct signatures, `__all__` exports, lazy-import patterns, and CLI argument definitions.

## 2. Verification by FR

| FR | Description | Assessment |
|----|-------------|------------|
| FR-1 | `latent.empty_wan_latent_video` | ✅ comply |
| FR-2 | `pipelines/video/wan/wan21/t2v.py` — `manifest()` (3 entries) + `run()` | ✅ comply |
| FR-3 | `pipelines/video/wan/wan21/i2v.py` — `manifest()` (4 entries) + `run()` | ✅ comply |
| FR-4 | `pipelines/video/wan/wan21/flf2v.py` — `manifest()` (4 entries) + `run()` | ✅ comply |
| FR-5 | `pipelines/video/wan/__init__.py` — `__all__ = ["wan21"]` | ✅ comply |
| FR-6 | `pipelines/video/wan/wan21/__init__.py` — `__all__ = ["t2v", "i2v", "flf2v"]` | ✅ comply |
| FR-7 | `pipelines/video/__init__.py` — `__all__` includes `"wan"` | ✅ comply |
| FR-8 | `examples/video_wan21_t2v.py` | ✅ comply |
| FR-9 | `examples/video_wan21_i2v.py` | ✅ comply |
| FR-10 | `examples/video_wan21_flf2v.py` | ✅ comply |

## 3. Verification by US

| US | Title | Assessment |
|----|-------|------------|
| US-001 | `empty_wan_latent_video` wrapper | ✅ comply |
| US-002 | WAN 2.1 text-to-video pipeline | ✅ comply |
| US-003 | WAN 2.1 image-to-video pipeline | ✅ comply |
| US-004 | WAN 2.1 first-last-frame-to-video pipeline | ✅ comply |
| US-005 | Pipeline sub-package wiring | ✅ comply |
| US-006 | Example scripts | ✅ comply |

## 4. Minor Observations

1. `i2v.py` and `flf2v.py` type-annotate image parameters as `str | Path | Any` rather than the more precise `str | Path | PIL.Image.Image`; functionally equivalent but less expressive in IDEs and type checkers.
2. No unit tests cover the new `empty_wan_latent_video` function or the `manifest()` calls — acceptable per project convention (critical paths only, GPU not available in CI), but worth adding a CPU-safe smoke test.
3. The `wan21` pipelines use `HFModelEntry` (Hugging Face) exclusively; inline documentation in `run()` could clarify that local-path overrides are not yet supported for these models.

## 5. Conclusions and Recommendations

The implementation is complete and correct. All acceptance criteria are satisfied. The three minor observations are non-blocking style/documentation items that should be addressed in the Refactor phase:

1. Tighten image parameter type annotations from `Any` to `PIL.Image.Image` in `i2v.py` and `flf2v.py`.
2. Add at least one CPU-safe `pytest` smoke-test for `empty_wan_latent_video`.
3. Add brief docstrings to each pipeline's `run()` explaining HF-only model resolution.

## 6. Refactor Plan

### R-1 — Type annotation cleanup (`i2v.py`, `flf2v.py`)
- **Files**: `comfy_diffusion/pipelines/video/wan/wan21/i2v.py`, `comfy_diffusion/pipelines/video/wan/wan21/flf2v.py`, `examples/video/wan/wan21/i2v.py`, `examples/video/wan/wan21/flf2v.py`
- **Change**: Replace `str | Path | Any` with `str | Path | PIL.Image.Image` for `image`, `start_image`, and `end_image` parameters. Add `from PIL import Image` import inside the function body (lazy) or use `TYPE_CHECKING` guard at module level.
- **Priority**: Low

### R-2 — CPU-safe smoke test for `empty_wan_latent_video`
- **File**: `tests/test_latent.py` (or equivalent)
- **Change**: Add a test that calls `empty_wan_latent_video(width=64, height=64, length=5)` and asserts the returned dict has `"samples"` with shape `[1, 16, 2, 8, 8]` (CPU tensor, no GPU needed).
- **Priority**: Medium

### R-3 — Docstrings for `run()` in all three wan21 pipelines
- **Files**: `comfy_diffusion/pipelines/video/wan/wan21/t2v.py`, `comfy_diffusion/pipelines/video/wan/wan21/i2v.py`, `comfy_diffusion/pipelines/video/wan/wan21/flf2v.py`
- **Change**: Add a one-paragraph docstring to each `run()` explaining parameters, HF-only model resolution, and return value (list of PIL Images or frames).
- **Priority**: Low
