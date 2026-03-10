# Requirement: Latent Utilities Module

## Context
comfy-diffusion exposes ComfyUI's inference engine as importable Python modules. The library currently covers checkpoint loading, conditioning, sampling, VAE encode/decode (including tiled and batch), LoRA, audio, advanced sampling, advanced conditioning, and ControlNet. However, there is no way to create, resize, crop, compose, or batch-manipulate latent tensors without reaching into ComfyUI internals directly. A `comfy_diffusion.latent` module will fill this gap, enabling callers to perform all standard latent operations through the library's public API.

## Goals
- Expose all ROADMAP latent utility nodes as clean Python functions in `comfy_diffusion/latent.py`
- Follow established project patterns: lazy imports, `str | Path` annotations, no top-level torch/comfy imports
- All functions must be testable on CPU-only environments

## User Stories

### US-001: Create an empty latent image
**As a** Python developer, **I want** to create an empty latent image with specified width, height, and batch size **so that** I can use it as the starting point for txt2img generation.

**Acceptance Criteria:**
- [ ] `empty_latent_image(width, height, batch_size=1)` returns a latent dict compatible with `sample()`
- [ ] Width and height are divided by 8 internally (latent space downscale factor)
- [ ] Default batch size is 1
- [ ] Typecheck / lint passes

### US-002: Upscale a latent
**As a** Python developer, **I want** to upscale a latent by target dimensions or by a scale factor **so that** I can do multi-pass hi-res generation.

**Acceptance Criteria:**
- [ ] `latent_upscale(latent, method, width, height)` returns an upscaled latent dict
- [ ] Supported methods mirror ComfyUI: `nearest-exact`, `bilinear`, `area`, `bicubic`, `bislerp`
- [ ] `latent_upscale_by(latent, method, scale_by)` returns a latent scaled by a float factor
- [ ] Output latent is compatible with `sample()`
- [ ] Typecheck / lint passes

### US-003: Crop a region from a latent
**As a** Python developer, **I want** to crop a rectangular region from a latent **so that** I can isolate a section for targeted generation or compositing.

**Acceptance Criteria:**
- [ ] `latent_crop(latent, x, y, width, height)` returns a cropped latent dict
- [ ] Coordinates and dimensions are in pixel space (divided by 8 internally)
- [ ] Typecheck / lint passes

### US-004: Composite one latent onto another
**As a** Python developer, **I want** to composite a source latent onto a destination latent at a given position **so that** I can combine independently generated regions.

**Acceptance Criteria:**
- [ ] `latent_composite(destination, source, x, y)` returns a new latent with source placed at (x, y)
- [ ] Coordinates are in pixel space (divided by 8 internally)
- [ ] Typecheck / lint passes

### US-005: Composite a latent using a mask
**As a** Python developer, **I want** to composite a source latent onto a destination using a mask **so that** I can blend regions with soft boundaries.

**Acceptance Criteria:**
- [ ] `latent_composite_masked(destination, source, mask, x=0, y=0)` returns a blended latent
- [ ] Mask is a torch tensor controlling blend strength per pixel
- [ ] Coordinates default to (0, 0)
- [ ] Typecheck / lint passes

### US-006: Set a noise mask on a latent
**As a** Python developer, **I want** to set a noise mask on a latent **so that** the sampler only denoises the masked region (inpainting).

**Acceptance Criteria:**
- [ ] `set_latent_noise_mask(latent, mask)` returns a latent dict with `noise_mask` set
- [ ] Mask is a torch tensor
- [ ] Output is compatible with `sample()` for inpainting workflows
- [ ] Typecheck / lint passes

### US-007: Batch operations
**As a** Python developer, **I want** to extract a subset of frames from a latent batch and repeat a latent N times **so that** I can manipulate batches for video and multi-image workflows.

**Acceptance Criteria:**
- [ ] `latent_from_batch(latent, batch_index, length=1)` extracts a contiguous slice from the batch dimension
- [ ] `repeat_latent_batch(latent, amount)` repeats the latent along the batch dimension
- [ ] Both return latent dicts compatible with `sample()`
- [ ] Typecheck / lint passes

### US-008: Video latent operations
**As a** Python developer, **I want** to concatenate latents, cut a batch into segments, and replace frames in a video latent **so that** I can build video generation pipelines.

**Acceptance Criteria:**
- [ ] `latent_concat(*latents, dim="t")` concatenates multiple latents along the specified dimension (supports `"x"`, `"-x"`, `"y"`, `"-y"`, `"t"`, `"-t"`)
- [ ] `latent_cut_to_batch(latent, start, length)` extracts a segment from a batch
- [ ] `replace_video_latent_frames(latent, replacement, start_frame)` replaces frames in-place starting at a given index
- [ ] All return latent dicts compatible with `sample()`
- [ ] Typecheck / lint passes

### US-009: Inpaint conditioning helpers
**As a** Python developer, **I want** to encode an image for inpainting and set up inpaint model conditioning **so that** I can run inpainting pipelines end-to-end.

**Acceptance Criteria:**
- [ ] `vae_encode_for_inpaint(vae, image, mask, grow_mask_by=6)` encodes an image with an inpaint mask applied, returning a latent with noise mask set
- [ ] `inpaint_model_conditioning(model, latent, vae, positive, negative)` returns patched (model, positive, negative) tuple for inpaint-specific models
- [ ] Both are compatible with existing `sample()` and conditioning functions
- [ ] Typecheck / lint passes

## Functional Requirements
- FR-1: All functions live in `comfy_diffusion/latent.py`
- FR-2: All functions use lazy imports — no `torch` or `comfy.*` at module top level
- FR-3: Latent dicts use the `{"samples": tensor}` format established by ComfyUI, optionally with `"noise_mask"` and `"batch_index"` keys
- FR-4: Pixel-space coordinates/dimensions are converted to latent space (÷8) internally; callers always think in pixels
- FR-5: All functions must work on CPU tensors for testing purposes
- FR-6: Public API symbols are importable via `from comfy_diffusion.latent import <function>`
- FR-7: `vae_encode_for_inpaint` lives in `vae.py`; `inpaint_model_conditioning` lives in `latent.py`
- FR-8: `latent_concat` accepts variadic `*latents` and a `dim` parameter (`"x"`, `"-x"`, `"y"`, `"-y"`, `"t"`, `"-t"`) for concatenation along any spatial or temporal axis

## Non-Goals (Out of Scope)
- Nice-to-have latent nodes (`SaveLatent`, `LoadLatent`, `LatentFlip`, `LatentRotate`, `LatentBlend`, `LatentInterpolate`, arithmetic ops) — deferred to a future iteration
- Tiled VAE operations — already implemented in iteration 08
- High-level pipeline abstractions — the library remains modular building blocks
- Re-exporting latent functions from `comfy_diffusion/__init__.py` — callers use explicit submodule imports

## Resolved Questions
- `vae_encode_for_inpaint` lives in `vae.py` (alongside `vae_encode`). `inpaint_model_conditioning` lives in `latent.py`.
- `latent_concat` uses variadic `*latents` — Pythonic and flexible for 2+ latents.
