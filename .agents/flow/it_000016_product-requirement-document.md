# Requirement: Mask Module

## Context
comfy-diffusion lacks a dedicated mask module. Mask operations are essential for inpainting, regional conditioning, and compositing workflows. While `load_image()` in `image.py` already extracts an alpha-based mask, there is no way to load masks from specific channels, convert between image/mask tensors, or manipulate mask geometry (grow, feather). This iteration adds a `mask` module exposing these utilities.

## Goals
- Provide a standalone `comfy_diffusion/mask.py` module with mask creation and manipulation functions
- Support loading masks from image files with explicit channel selection (alpha, red, green, blue)
- Enable conversion between image tensors and mask tensors
- Provide mask geometry operations (grow, feather) needed for inpainting workflows
- Follow existing project patterns: lazy imports, external libraries (PIL, torch) over ComfyUI node wrappers where possible

## User Stories

### US-001: Load mask from image file
**As a** Python developer using comfy-diffusion, **I want** to load a mask from an image file choosing which channel to use (alpha, red, green, blue) **so that** I can obtain mask tensors from standard image files for inpainting or compositing.

**Acceptance Criteria:**
- [ ] `load_image_mask(path, channel)` accepts `str | Path` and a channel parameter
- [ ] Supported channels: `"alpha"`, `"red"`, `"green"`, `"blue"`
- [ ] Returns a float32 mask tensor with shape `(1, H, W)` and values in `[0.0, 1.0]`
- [ ] For alpha channel: transparent = 1.0 (masked), opaque = 0.0 (unmasked), matching ComfyUI convention
- [ ] For color channels: pixel value 255 → 1.0, pixel value 0 → 0.0
- [ ] Raises `FileNotFoundError` if the image path does not exist
- [ ] Typecheck / lint passes

### US-002: Convert image tensor to mask tensor
**As a** Python developer using comfy-diffusion, **I want** to convert a BHWC image tensor to a mask tensor by selecting a channel **so that** I can derive masks from existing image data without saving/loading files.

**Acceptance Criteria:**
- [ ] `image_to_mask(image, channel)` accepts a BHWC float32 image tensor and channel name
- [ ] Supported channels: `"red"`, `"green"`, `"blue"`
- [ ] Returns a float32 mask tensor with shape matching the batch and spatial dims
- [ ] Typecheck / lint passes

### US-003: Convert mask tensor to image tensor
**As a** Python developer using comfy-diffusion, **I want** to convert a mask tensor back to a PIL Image **so that** I can visualize or save masks for debugging and inspection.

**Acceptance Criteria:**
- [ ] `mask_to_image(mask)` accepts a float32 mask tensor
- [ ] Returns a BHWC float32 image tensor with the mask value replicated across RGB channels
- [ ] Typecheck / lint passes

### US-004: Grow mask
**As a** Python developer using comfy-diffusion, **I want** to expand or contract a mask by N pixels **so that** I can adjust mask boundaries for inpainting overlap or tighter crops.

**Acceptance Criteria:**
- [ ] `grow_mask(mask, expand, tapered_corners)` grows the mask by `expand` pixels
- [ ] Positive `expand` grows (dilates) the mask; negative `expand` shrinks (erodes) it
- [ ] `tapered_corners` parameter controls whether corners are rounded (True) or square (False)
- [ ] Works on CPU without GPU
- [ ] Typecheck / lint passes

### US-005: Feather mask
**As a** Python developer using comfy-diffusion, **I want** to blur the edges of a mask **so that** I get smooth transitions for compositing and inpainting.

**Acceptance Criteria:**
- [ ] `feather_mask(mask, left, top, right, bottom)` feathers mask edges by the specified pixel amounts per side
- [ ] Supports independent feathering amounts for each edge
- [ ] Output values remain in `[0.0, 1.0]`
- [ ] Works on CPU without GPU
- [ ] Typecheck / lint passes

## Functional Requirements
- FR-1: Create `comfy_diffusion/mask.py` module with all five public functions
- FR-2: Use lazy imports for `torch` and `PIL` (no top-level imports of heavy dependencies)
- FR-3: Use external libraries (PIL, torch) for image I/O and tensor ops; only wrap ComfyUI nodes when they provide non-trivial logic not easily replicated
- FR-4: Path parameters use `str | Path` type annotation, consistent with the rest of the codebase
- FR-5: Export all public functions in `__all__`
- FR-6: All functions must work on CPU-only environments
- FR-7: Follow the alpha convention from ComfyUI: for alpha masks, transparent = 1.0 (masked), opaque = 0.0
- FR-8: `grow_mask` wraps ComfyUI's `GrowMask` node (non-trivial scipy morphological ops with custom kernels)
- FR-9: `feather_mask` is implemented directly with torch ops (simple per-side linear gradient multiplication)

## Non-Goals (Out of Scope)
- `SolidMask`, `InvertMask`, `CropMask` — classified as "Discarded" in the roadmap (replaceable by torch/numpy ops)
- `MaskComposite`, `ThresholdMask` — classified as "Nice-to-have"
- Auto-import from `comfy_diffusion/__init__.py` — callers use `from comfy_diffusion.mask import ...`
- GPU-specific optimizations
- Integration with inpainting conditioning nodes (separate iteration)

## Open Questions
- ~~Should `grow_mask` and `feather_mask` use pure torch/scipy ops, or wrap the ComfyUI node implementations?~~ **Resolved:** `grow_mask` wraps the ComfyUI `GrowMask` node (uses `scipy.ndimage.grey_erosion`/`grey_dilation` with custom kernels — non-trivial). `feather_mask` is implemented directly with torch (simple per-side linear gradient multiplication).
