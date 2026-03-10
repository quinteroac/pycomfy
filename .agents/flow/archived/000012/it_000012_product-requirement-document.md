# Requirement: Advanced Conditioning

## Context
comfy-diffusion currently exposes only basic prompt encoding (`encode_prompt`) via `CLIPTextEncode`. Users who need region-based prompting, prompt scheduling across timesteps, multi-prompt blending, or architecture-specific conditioning (Flux, WAN, LTXV) must fall back to running a ComfyUI server. This iteration adds those capabilities as composable Python functions in the `conditioning` module.

## Goals
- Expose conditioning combination, masking, and timestep-range operations as simple function calls
- Support Flux-specific text encoding and guidance scaling
- Support WAN image-to-video and first/last-frame conditioning
- Support LTXV image-to-video and LTXV-specific conditioning
- All new functions follow the existing lazy-import pattern and work on CPU for testing

## User Stories

### US-001: Combine multiple conditioning objects
**As a** developer using comfy-diffusion, **I want** to combine two or more conditioning objects into one **so that** I can blend multiple prompts (e.g. positive + style prompt) in a single sampling call.

**Acceptance Criteria:**
- [ ] `conditioning_combine(cond_a, cond_b)` returns a merged conditioning list
- [ ] Supports combining more than two by chaining or accepting a list
- [ ] Typecheck / lint passes
- [ ] Unit test verifies combined output contains entries from all inputs

### US-002: Apply a spatial mask to conditioning
**As a** developer, **I want** to apply a mask tensor to a conditioning object **so that** the prompt only affects a specific image region (region-based prompting / inpainting guidance).

**Acceptance Criteria:**
- [ ] `conditioning_set_mask(conditioning, mask, strength, set_cond_area)` returns conditioning with the mask applied
- [ ] `strength` defaults to `1.0`; `set_cond_area` defaults to `"default"` with `"mask bounds"` as alternative
- [ ] Typecheck / lint passes
- [ ] Unit test verifies mask and strength are attached to conditioning metadata

### US-003: Restrict conditioning to a timestep range
**As a** developer, **I want** to restrict a conditioning object to a specific start/end timestep range **so that** I can schedule different prompts for different denoising phases.

**Acceptance Criteria:**
- [ ] `conditioning_set_timestep_range(conditioning, start, end)` returns conditioning with timestep bounds
- [ ] `start` and `end` are floats in `[0.0, 1.0]` representing percentage of the denoising process
- [ ] Typecheck / lint passes
- [ ] Unit test verifies timestep range metadata is correctly set

### US-004: Flux-specific text encoding and guidance
**As a** developer working with Flux models, **I want** Flux-specific CLIP encoding (with separate clip_l guidance) and a guidance-scaling function **so that** I can use Flux models with their intended conditioning format.

**Acceptance Criteria:**
- [ ] `encode_prompt_flux(clip, text, clip_l_text, guidance)` encodes text using Flux's dual-encoder format
- [ ] `flux_guidance(conditioning, guidance)` applies Flux guidance scaling to conditioning
- [ ] `guidance` parameter defaults to `3.5`
- [ ] Typecheck / lint passes
- [ ] Unit tests verify both functions produce correctly structured output

### US-005: WAN image/video conditioning
**As a** developer working with WAN video models, **I want** to create WAN-specific conditioning from reference images **so that** I can drive image-to-video and first/last-frame-to-video generation.

**Acceptance Criteria:**
- [ ] `wan_image_to_video(positive, negative, vae, width=832, height=480, length=81, batch_size=1, start_image=None, clip_vision_output=None)` returns `(positive, negative, latent)` tuple
- [ ] `wan_first_last_frame_to_video(positive, negative, vae, width=832, height=480, length=81, batch_size=1, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None)` returns `(positive, negative, latent)` tuple
- [ ] `clip_vision_output` parameters are optional (WAN works without them)
- [ ] `encode_clip_vision(clip_vision, image, crop="center")` exposed as public helper for producing clip_vision_output
- [ ] `ModelManager.load_clip_vision(path)` loads a CLIP vision model via `comfy.clip_vision.load()`
- [ ] Both functions follow the lazy-import pattern (no top-level comfy imports)
- [ ] Typecheck / lint passes
- [ ] Unit tests verify conditioning structure

### US-006: LTXV conditioning
**As a** developer working with LTXV video models, **I want** LTXV-specific conditioning functions **so that** I can set up image-to-video and LTXV-specific conditioning parameters.

**Acceptance Criteria:**
- [ ] `ltxv_img_to_video(positive, negative, image, vae, width=768, height=512, length=97, batch_size=1, strength=1.0)` returns `(positive, negative, latent_with_noise_mask)` tuple
- [ ] `ltxv_conditioning(positive, negative, frame_rate=25.0)` returns `(positive, negative)` with frame_rate metadata injected
- [ ] Neither function uses clip_vision (confirmed not needed for LTXV)
- [ ] Both functions follow the lazy-import pattern
- [ ] Typecheck / lint passes
- [ ] Unit tests verify conditioning structure

## Functional Requirements
- FR-1: All new functions are defined in `comfy_diffusion/conditioning.py`, extending the existing module
- FR-2: All new public symbols are added to `__all__` in `conditioning.py`
- FR-3: All functions use lazy imports — no `comfy.*` or `torch` imports at module top level
- FR-4: Functions mirror the behavior of the corresponding ComfyUI nodes but expose a clean Python API (no node boilerplate)
- FR-5: Generic conditioning utilities (`conditioning_combine`, `conditioning_set_mask`, `conditioning_set_timestep_range`) are architecture-agnostic and work with any model type
- FR-6: Architecture-specific functions (`encode_prompt_flux`, `flux_guidance`, `wan_*`, `ltxv_*`) are grouped clearly but remain in the same module
- FR-7: Type hints on all public function signatures using the established `str | Path` and Protocol patterns
- FR-8: `encode_prompt_flux` re-exports or extends the existing CLIP Protocol as needed for Flux's dual-encoder interface
- FR-9: Expose `load_clip_vision(path)` on `ModelManager` and `encode_clip_vision(clip_vision, image, crop)` as public helpers — WAN nodes accept optional pre-encoded `CLIP_VISION_OUTPUT`, so callers need a way to load and encode clip_vision models independently
- FR-10: WAN functions return a tuple of `(positive, negative, latent)` matching the ComfyUI node output structure
- FR-11: LTXV functions do not use clip_vision at all — `LTXVImgToVideo` works purely via VAE encoding, `LTXVConditioning` is metadata-only (frame_rate injection)

## Non-Goals (Out of Scope)
- `ConditioningAverage`, `ConditioningConcat`, `ConditioningSetArea`, and other nice-to-have conditioning nodes
- `CLIPTextEncodeSD3`, `CLIPTextEncodeHunyuanDiT`, and other architecture-specific encoders not listed in roadmap item 12
- `StyleModelApply` as a standalone public function (out of scope for this iteration)
- High-level pipeline or workflow abstraction
- End-to-end example scripts
- `WanFunInpaintToVideo` (listed in roadmap but lower priority — can be added in a follow-up)

## Open Questions (Resolved)
- Q1: **Resolved — expose as public.** `encode_clip_vision(clip_vision, image, crop="center")` will be a public function in `conditioning.py`. WAN nodes accept optional pre-encoded `CLIP_VISION_OUTPUT` (not the raw model), so callers need a way to encode images. LTXV does not use clip_vision at all.
- Q2: **Resolved — add `load_clip_vision(path)` to `ModelManager`.** It wraps `comfy.clip_vision.load(path)`. Callers load clip_vision models via `ModelManager`, then encode with `encode_clip_vision()`, then pass the output to WAN functions. This follows the existing pattern (`load_vae`, `load_clip`, `load_unet`).
- Q3: **Resolved — exact signatures confirmed from source.** Key findings:
  - `WanImageToVideo`: params are `(positive, negative, vae, width=832, height=480, length=81, batch_size=1, start_image=None, clip_vision_output=None)`. Returns `(positive, negative, latent)`. clip_vision_output is **optional**.
  - `WanFirstLastFrameToVideo`: params are `(positive, negative, vae, width=832, height=480, length=81, batch_size=1, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None)`. Returns `(positive, negative, latent)`. All image/clip_vision params are **optional**.
  - `LTXVImgToVideo`: params are `(positive, negative, image, vae, width=768, height=512, length=97, batch_size=1, strength=1.0)`. Returns `(positive, negative, latent_with_noise_mask)`. **No clip_vision**.
  - `LTXVConditioning`: params are `(positive, negative, frame_rate=25.0)`. Returns `(positive, negative)`. Pure metadata injection — adds `frame_rate` to conditioning. **No clip_vision**.
