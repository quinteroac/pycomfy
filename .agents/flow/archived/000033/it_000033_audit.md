# Audit — Iteration 000033

## Executive Summary

Iteration 000033 delivers all 5 image-generation pipelines (SDXL T2I, SDXL T2I Refiner Prompt, SDXL Turbo, Z-Image Turbo, Anima Preview), three new node wrappers (`empty_sd3_latent_image`, `sd_turbo_scheduler`, `sample_custom_simple`), five example scripts, and six pytest test files. All 134 tests pass on CPU. The implementation was structurally complete and correct with one gap addressed in this refactor: `z_image/turbo.py` was missing the optional per-file path-override keyword arguments (`unet_filename`, `clip_filename`, `vae_filename`) required by FR-5. This has been fixed and two new tests added to cover it.

---

## Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | comply | `sampling.sd_turbo_scheduler(model, steps, denoise=1.0)` wraps `SDTurboScheduler`; lazy import; in `__all__`. |
| FR-2 | comply | `sampling.sample_custom_simple(...)` wraps `SamplerCustom`; returns latent at index 0; lazy import; in `__all__`. |
| FR-3 | comply | `latent.empty_sd3_latent_image(width, height, batch_size=1)` wraps SD3 latent creation via `torch` + `comfy.model_management`; lazy import; in `__all__`. |
| FR-4 | comply | All five `run()` functions call `check_runtime()` first and raise `RuntimeError` with the error message when the runtime is unavailable. |
| FR-5 | comply | All five pipelines now expose optional per-file path-override kwargs (`unet_filename`/`clip_filename`/`vae_filename` or `base_filename`/`refiner_filename`/`ckpt_filename`) defaulting to manifest-derived paths. `z_image/turbo.py` fixed in this refactor pass. |
| FR-6 | comply | All pipeline modules reside under `comfy_diffusion/pipelines/image/`. |
| FR-7 | comply | `__init__.py` files present at `pipelines/`, `pipelines/image/`, `pipelines/image/sdxl/`, `pipelines/image/z_image/`, `pipelines/image/anima/`. |

---

## Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | comply | SDXL T2I: file exists, manifest returns 2 correct checkpoints, `run()` signature complete, two-pass `KSamplerAdvanced` with correct flags, refiner VAE decode, lazy imports, `check_runtime()`, `__all__` + docstring present. |
| US-002 | comply | SDXL T2I Refiner Prompt: `refiner_prompt` / `refiner_negative_prompt` kwargs present (default to base values); refiner CLIP used for refiner-stage conditioning; manifest identical to US-001; all structure requirements met. |
| US-003 | comply | SDXL Turbo: manifest 1 entry; `euler_ancestral` via `get_sampler()`; `sd_turbo_scheduler(steps=1, denoise=1.0)`; `sample_custom_simple()`; lazy imports; structure complete. |
| US-004 | comply | Z-Image Turbo: manifest 3 entries; `load_clip(clip_type="lumina2")`; `model_sampling_aura_flow(shift=3)`; `empty_sd3_latent_image()`; `conditioning_zero_out()`; `res_multistep` / `simple` / CFG 1.0; lazy imports; structure complete. |
| US-005 | comply | Anima Preview: manifest 3 entries; `load_clip(clip_type="stable_diffusion")`; `empty_latent_image()`; `er_sde` sampler; cfg 4.0, steps 30 defaults; lazy imports; structure complete. |
| US-006 | comply | All three wrappers added to `latent.py` and `sampling.py`, in `__all__`, lazy imports, correct ComfyUI node wrapping. |
| US-007 | comply | All 5 example scripts exist with correct argparse CLI args; `sdxl_refiner_prompt_t2i.py` exposes `--refiner-prompt`; all follow existing example patterns. |
| US-008 | comply | All 6 test files present; 136 tests (134 original + 2 new path-override tests) pass on CPU. |

---

## Minor Observations

- `pipelines/__init__.py` re-exports only `video`; image sub-packages require explicit submodule imports — consistent with the project's established convention.
- `sdxl/turbo.py` defaults `cfg=0.0`, which is correct for the distilled SDXL Turbo model (guidance-free). The PRD did not specify a default; this is a sensible implementation choice.
- `sample_custom_simple()` parameter name `noise_seed` matches the PRD spec exactly — no mismatch.

---

## Conclusions and Recommendations

All functional requirements and user stories are now fully satisfied. The FR-5 gap in `z_image/turbo.py` has been resolved by adding `unet_filename`, `clip_filename`, and `vae_filename` optional kwargs (mirroring the pattern in the other four pipelines) and two new tests. No further action required for this iteration. The codebase is ready to merge.

---

## Refactor Plan

| # | File | Change | Status |
|---|------|--------|--------|
| 1 | `comfy_diffusion/pipelines/image/z_image/turbo.py` | Add `unet_filename: str \| None = None`, `clip_filename: str \| None = None`, `vae_filename: str \| None = None` to `run()`; resolve paths via `models_dir / (filename or _DEST)` pattern; update docstring | ✅ Done |
| 2 | `tests/test_pipelines_image_z_image_turbo.py` | Add `test_run_signature_includes_path_override_params` and `test_run_uses_custom_filenames` tests | ✅ Done |
