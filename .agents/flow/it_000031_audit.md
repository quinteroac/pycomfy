# Audit — Iteration 000031

## Executive Summary

Iteration 000031 delivers WAN 2.2 audio/video pipeline support across 10 user stories. All 293 tests pass on CPU. All five pipelines (t2v, i2v, flf2v, s2v, ti2v) are implemented with `manifest()`/`run()` contracts, lazy-import patterns, example scripts, and unit tests.

Two minor deviations from the letter of the PRD were found:

1. **flf2v manifest count** — `manifest()` returns 4 `ModelEntry` items (2 UNets + text encoder + VAE) while US-007-AC01 specifies exactly 3. The implementation's rationale is sound (the two-pass `KSamplerAdvanced` flow requires two distinct UNet checkpoints), but the PRD count needs reconciliation.
2. **`audio_encoders` folder registration location** — FR-11 specifies `_runtime.py`, but registration happens in `ModelManager.__init__` (`models.py:77`) via `folder_paths.add_model_folder_path`. Functionally equivalent and architecturally cleaner in models.py, but deviates from the spec.

---

## Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | **comply** | `ModelManager.load_audio_encoder(path: str \| Path) -> Any` in `models.py:345`. Calls `comfy.utils.load_torch_file` + `comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd`. Lazy-import pattern followed. |
| FR-2 | **comply** | `audio_encoder_encode` in `audio.py:421`. Correctly calls `audio_encoder.encode_audio(audio["waveform"], audio["sample_rate"])`. Listed in `audio.__all__`. |
| FR-3 | **comply** | `wan_sound_image_to_video` wraps `_wan_sound_to_video()`, mirrors `WanSoundImageToVideo.execute()`, returns `(positive, negative, latent)`. Listed in `conditioning.__all__`. |
| FR-4 | **comply** | `wan_sound_image_to_video_extend` wraps `_wan_sound_to_video()` with `frame_offset` derived from `video_latent`, mirrors `WanSoundImageToVideoExtend.execute()`, returns `(positive, negative, latent)`. Listed in `conditioning.__all__`. |
| FR-5 | **comply** | `wan22_image_to_video_latent` in `latent.py:521`. Uses `comfy.latent_formats.Wan22()` for latent processing. Returns `{samples, noise_mask}` with correct shape when `start_image` is provided; `{samples}` only when `start_image is None`, matching node behaviour. Listed in `latent.__all__`. |
| FR-6 | **comply** | All 5 pipelines expose `manifest() -> list[ModelEntry]` and `run(...)` functions. Package wiring via `wan22/__init__.py` and `wan/__init__.py` is correct and consistent with the `ltx2_t2v.py` precedent. |
| FR-7 | **partially_comply** | t2v (6), i2v (6), s2v (5), ti2v (3) match PRD. flf2v returns 4 `ModelEntry` items; PRD US-007-AC01 specifies 3. |
| FR-8 | **comply** | Execution order mirrors workflow: model loading → conditioning → sampler passes → VAE decode. Dual-pass `KSamplerAdvanced` (high-noise → low-noise) preserved in t2v/i2v/flf2v. S2V iterates multi-pass extend loop. TI2V uses single-pass `KSampler` matching the 5B workflow. |
| FR-9 | **comply** | All 5 pipeline modules have zero top-level `torch`/`comfy.*` imports. All heavy imports are deferred inside function bodies. Verified by `test_no_heavy_top_level_imports` tests (all pass). |
| FR-10 | **comply** | 293/293 tests pass (`uv run pytest`). No GPU required — model weights are mocked. |
| FR-11 | **partially_comply** | `audio_encoders` registered via `folder_paths.add_model_folder_path("audio_encoders", ..., is_default=True)` in `ModelManager.__init__` (`models.py:77`), not in `_runtime.py` as specified. Functionally equivalent. |

---

## Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | **comply** | All 6 ACs met: method signature `str\|Path`, `audio_encoders` folder registered, calls `comfy.utils.load_torch_file` + `load_audio_encoder_from_sd`, lazy-import, docstring present. |
| US-002 | **comply** | All 5 ACs met: function signature `Any/dict`, calls `encode_audio` correctly, lazy-import, listed in `audio.__all__`. |
| US-003 | **comply** | All 5 ACs met: both functions implemented with correct signatures, both return `(positive, negative, latent)`, lazy-import, both in `conditioning.__all__`. |
| US-004 | **comply** | All 5 ACs met: signature matches spec, returns `{samples, noise_mask}` with correct shape, uses `comfy.latent_formats.Wan22()`, lazy-import, listed in `latent.__all__`. |
| US-005 | **comply** | All 7 ACs met: `manifest()` returns 6 items, `run()` implements dual two-pass `KSamplerAdvanced` with `LoraLoaderModelOnly` and `ModelSamplingSD3` shift, bypassed nodes excluded, returns `list[PIL.Image]`, unit test and `examples/wan22_t2v.py` present. |
| US-006 | **comply** | All 6 ACs met: `manifest()` returns 6 items, `run()` uses `WanImageToVideo` conditioning + two-pass `KSamplerAdvanced` + `ModelSamplingSD3` with switch defaults, bypassed nodes excluded, unit test and `examples/wan22_i2v.py` present. |
| US-007 | **partially_comply** | 5 of 7 ACs met. AC01 fails: `manifest()` returns 4 items, not 3. AC02 (UMT5-XXL included), AC03 (`WanFirstLastFrameToVideo` + two-pass), AC04 (bypassed nodes excluded), AC05 (unit test), AC06 (`examples/wan22_flf2v.py`) all pass. |
| US-008 | **comply** | All 6 ACs met: `manifest()` returns 5 items, `run()` calls `load_audio_encoder` + `audio_encoder_encode` + multi-pass `WanSoundImageToVideoExtend` + `KSampler` + `LatentConcat` loop, `audio` dict accepts `waveform`/`sample_rate`, unit test and `examples/wan22_s2v.py` present. |
| US-009 | **comply** | All 6 ACs met: `manifest()` returns 3 items, `run()` calls `wan22_image_to_video_latent` + `ModelSamplingSD3(shift=8)` + `KSampler(uni_pc, 20 steps, cfg=5)` + `vae_decode`, `start_image=None` uses empty latent without mask, unit test and `examples/wan22_ti2v.py` present. |
| US-010 | **comply** | All 4 ACs met: `wan22/__init__.py` exports `[t2v, i2v, flf2v, s2v, ti2v]`, `wan/__init__.py` exposes `wan22`, all 5 example scripts present with required CLI flags, lint passes. |

---

## Minor Observations

- `wan22_image_to_video_latent` is defined in two places: `latent.py:521` (public top-level function) and an internal helper inside `conditioning.py` used by the ti2v pipeline. Both are functionally equivalent. Consolidation into a single canonical implementation in `latent.py` would reduce duplication.
- The flf2v `manifest()` docstring states "Returns exactly 4 HFModelEntry items", contradicting US-007-AC01 which says 3. The docstring is internally consistent with the code but the PRD count needs reconciliation against the reference workflow.
- FR-11 specifies `_runtime.py` as the registration location for `audio_encoders`, but `models.py` is architecturally the correct home (model directories belong to the model manager). The PRD wording should be updated to match reality.
- `wan_first_last_frame_to_video` is listed in `conditioning.__all__` (line 1804) — verify it was pre-existing and not missing from a prior iteration's `__all__`.
- All 5 example scripts include a `save_frames()` helper and proper `if __name__ == "__main__"` guard. The 56-test wiring suite in `test_us010_wan22_package_wiring.py` provides thorough coverage of CLI flags, import patterns, and entrypoints.

---

## Conclusions and Recommendations

Iteration 000031 is substantially complete and production-ready. 9 of 10 user stories fully comply; US-007 (flf2v) partially complies due to a manifest count discrepancy (4 items vs 3 specified). FR-7 and FR-11 also partially comply for the same underlying reasons.

**Recommended actions (refactor targets):**

1. **Reconcile flf2v manifest count** — use the `workflow-reader` skill on `video_wan2_2_flf2v.json` (or equivalent) to confirm whether 3 or 4 active model loaders are present. Update either the PRD or the `flf2v.py` `manifest()` accordingly, and align the docstring.
2. **Update FR-11 wording** — change FR-11 to state that `audio_encoders` is registered in `ModelManager.__init__` rather than `_runtime.py`, reflecting the actual and better architectural location.
3. **Consolidate `wan22_image_to_video_latent`** — remove the internal duplicate in `conditioning.py` and have the ti2v pipeline import from `comfy_diffusion.latent` instead, reducing maintenance surface.

---

## Refactor Plan

### Task 1 — Reconcile flf2v manifest count
- **File:** `comfy_diffusion/pipelines/video/wan/wan22/flf2v.py`
- **Action:** Verify active model loader nodes in the reference workflow. If 3 is correct (one UNet shared between passes), refactor `manifest()` to return 3 entries and update `run()` to reuse the same checkpoint for both passes. If 4 is correct, update the PRD/US-007-AC01 to say "exactly 4".
- **Also update:** `flf2v.py` docstring, `tests/test_pipelines_wan22_flf2v.py` manifest count assertion.

### Task 2 — Update FR-11 PRD wording
- **File:** `.agents/flow/it_000031_PRD.json` or next iteration's PRD
- **Action:** Change FR-11 description to: "`audio_encoders/` directory is registered as a known folder path in `ModelManager.__init__` (`models.py`) via `folder_paths.add_model_folder_path`, consistent with how other model directories are registered."

### Task 3 — Consolidate `wan22_image_to_video_latent`
- **Files:** `comfy_diffusion/conditioning.py`, `comfy_diffusion/pipelines/video/wan/wan22/ti2v.py`
- **Action:** Remove the internal `wan22_image_to_video_latent` helper from `conditioning.py`. Update `ti2v.py` to import `wan22_image_to_video_latent` from `comfy_diffusion.latent` instead.
- **Verify:** `uv run pytest` continues to pass after the consolidation.
