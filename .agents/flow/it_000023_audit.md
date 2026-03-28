# Audit — Iteration 000023

## Executive summary

Iteration 000023 delivers 7 new public functions (ltxv_empty_latent_video, ltxv_concat_av_latent, ltxv_separate_av_latent, ltxv_crop_guides, ltxv_latent_upsample, ModelManager.load_latent_upscale_model, manual_sigmas), a CPU-safe pytest suite, and an end-to-end example script. Overall compliance is high: 8 of 9 US and 8 of 9 FR reached "comply" before refactor. One bug was identified and fixed: `"load_latent_upscale_model"` was listed in `models.py`'s `__all__` without a corresponding module-level function, which would cause `AttributeError` on `import *`. The fix removes the orphan `__all__` entry and updates the test to validate the method is accessible via `ModelManager`.

## Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | comply | `ltxv_empty_latent_video` — correct signature, shape, lazy imports, in `__all__` |
| FR-2 | comply | `ltxv_concat_av_latent` — NestedTensor output, noise_mask logic, lazy imports, in `__all__` |
| FR-3 | comply | `ltxv_separate_av_latent` — unbind, noise_mask propagation, import-safe, in `__all__` |
| FR-4 | comply | `ltxv_crop_guides` — keyframe metadata removal, latent crop, passthrough when no keyframes |
| FR-5 | comply | `ltxv_latent_upsample` — wraps `LTXVLatentUpsampler.upsample_latent`, private helpers present |
| FR-6 | comply | `load_latent_upscale_model` — path resolution, FileNotFoundError, `load_torch_file` params |
| FR-7 | comply | `manual_sigmas` — regex parsing, `torch.FloatTensor`, lazy torch import |
| FR-8 | comply (post-fix) | `folder_paths` registration correct in `__init__`; orphan `__all__` entry removed |
| FR-9 | comply | All 7 functions follow lazy import pattern |

## Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | comply | All ACs met |
| US-002 | comply | All ACs met |
| US-003 | comply | All ACs met |
| US-004 | comply | All ACs met |
| US-005 | comply | Memory management and noise_mask removal handled by delegated node |
| US-006 | comply (post-fix) | AC08 fixed: orphan `__all__` entry removed; method accessible via `ModelManager` |
| US-007 | comply | All ACs met |
| US-008 | comply | tests/test_ltxv2.py covers all 7 functions, CPU-safe, no heavy top-level imports |
| US-009 | comply | `examples/ltxv2_t2sv_example.py` covers all 15 pipeline steps, English, correct CLI args |

## Minor observations

- `ltxv_crop_guides` correctly avoids calling `_get_ltxv_conditioning_dependencies()` (and thus importing torch) when `keyframe_idxs is None` — the passthrough path is truly import-safe.
- `test_ltxv2.py` test `test_ltxv_latent_upsample_delegates_to_node` verifies delegation but does not explicitly assert `noise_mask` absence from the return value. Acceptable since the ComfyUI node contract guarantees it.
- `examples/ltxv2_t2sv_example.py` exposes `--checkpoint` as a reserved argument. Cosmetic; no functional impact.

## Conclusions and recommendations

All 9 user stories are now fully compliant after the single-line `__all__` fix in `models.py` and the corresponding test update in `tests/test_ltxv2.py`. The iteration is ready to proceed to approval.

## Refactor plan

| # | File | Change | Scope |
|---|------|--------|-------|
| 1 | `comfy_diffusion/models.py` | Remove `"load_latent_upscale_model"` from `__all__` (orphan entry — method is on `ModelManager`) | Done ✅ |
| 2 | `tests/test_ltxv2.py` | Replace `test_load_latent_upscale_model_in_module_all` with `test_load_latent_upscale_model_accessible_via_model_manager` that verifies the method exists on `ModelManager` and that `ModelManager` is in `__all__` | Done ✅ |
