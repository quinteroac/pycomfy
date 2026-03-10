# Requirement: Image Utilities

## Context
comfy-diffusion exposes ComfyUI's inference engine as importable Python modules. Iterations 1–14 covered model loading, conditioning, sampling, VAE, LoRA, latent ops, audio, advanced conditioning, controlnet, and latent utilities. Image-to-video conditioning (WAN, LTXV) was already implemented in `conditioning.py` during iteration 12.

This iteration adds the missing **image-level utilities**: loading images from disk, padding for outpainting, upscaling with models, batch operations, masked compositing, LTXV preprocessing, and video I/O. These are the building blocks callers need between "I have a file on disk" and "I need a ComfyUI-compatible tensor."

## Goals
- Provide a complete `comfy_diffusion.image` module for image loading, transformation, and batch manipulation
- Provide video I/O helpers using `opencv-python` / `imageio` (behind `[video]` extra)
- Maintain BHWC tensor format and `{"samples": tensor}` dict conventions throughout
- All functions CPU-testable with no GPU requirement

## User Stories

### US-001: Load image from disk
**As a** Python developer, **I want** to load an image file into a ComfyUI-compatible BHWC float32 tensor **so that** I can feed it into conditioning, VAE encode, or other comfy-diffusion functions.

**Acceptance Criteria:**
- [ ] `load_image(path)` returns a tuple `(image_tensor, mask_tensor)` where image is BHWC float32 [0,1] and mask is HW float32 from alpha channel
- [ ] Supports common formats (PNG, JPEG, WEBP) via Pillow
- [ ] Images without alpha produce an all-zeros mask (fully opaque)
- [ ] EXIF orientation is applied automatically
- [ ] Typecheck / lint passes

### US-002: Pad image for outpainting
**As a** Python developer, **I want** to pad an image's edges with feathered borders **so that** I can use it for outpainting workflows.

**Acceptance Criteria:**
- [ ] `image_pad_for_outpaint(image, left, top, right, bottom, feathering)` returns `(padded_image, padded_mask)`
- [ ] Padded regions are filled and the mask indicates the original content area
- [ ] Feathering produces a gradient transition at the border
- [ ] Output tensors maintain BHWC format
- [ ] Typecheck / lint passes

### US-003: Upscale image with model
**As a** Python developer, **I want** to upscale an image using an upscale model (e.g. RealESRGAN) **so that** I can enhance resolution before further processing.

**Acceptance Criteria:**
- [ ] `image_upscale_with_model(upscale_model, image)` returns an upscaled BHWC tensor
- [ ] Wraps ComfyUI's `ImageUpscaleWithModel` node logic
- [ ] Works with any ComfyUI-compatible upscale model object
- [ ] Typecheck / lint passes

### US-004: Image batch operations
**As a** Python developer, **I want** to extract a frame from an image batch and repeat an image into a batch **so that** I can manipulate image sequences for video workflows.

**Acceptance Criteria:**
- [ ] `image_from_batch(image, batch_index, length)` extracts a contiguous subset from the batch dimension
- [ ] `repeat_image_batch(image, amount)` repeats the image tensor along the batch dimension
- [ ] Both preserve BHWC format and return tensors
- [ ] Typecheck / lint passes

### US-005: Video I/O
**As a** Python developer, **I want** to load video frames and save tensors as video files **so that** I can integrate video generation pipelines end-to-end.

**Acceptance Criteria:**
- [ ] `load_video(path)` returns a BHWC tensor of frames (or list of PIL images)
- [ ] `save_video(frames, path, fps)` writes frames to a video file
- [ ] `get_video_components(video_path)` returns metadata (frame count, fps, width, height)
- [ ] Uses `opencv-python` or `imageio` — imports are conditional on `[video]` extra
- [ ] Raises a clear error if opencv/imageio is not installed
- [ ] Typecheck / lint passes

### US-006: Composite image with mask
**As a** Python developer, **I want** to composite one image onto another using a mask **so that** I can blend inpainted regions or layer images.

**Acceptance Criteria:**
- [ ] `image_composite_masked(destination, source, mask, x, y)` composites source onto destination at given coordinates
- [ ] Mask controls blending (1.0 = full source, 0.0 = full destination)
- [ ] Output is BHWC tensor
- [ ] Typecheck / lint passes

### US-007: LTXV image preprocessing
**As a** Python developer, **I want** to preprocess an image for LTXV video generation **so that** the image is correctly formatted before passing to `ltxv_img_to_video`.

**Acceptance Criteria:**
- [ ] `ltxv_preprocess(image, width, height)` returns a resized/padded BHWC tensor suitable for LTXV
- [ ] Matches ComfyUI's `LTXVPreprocess` node behavior
- [ ] Typecheck / lint passes

## Functional Requirements
- FR-1: All public functions use lazy imports — no `torch`, `comfy.*`, `cv2`, or `imageio` at module top level
- FR-2: Image tensors follow ComfyUI's BHWC convention (batch, height, width, channels) with float32 values in [0, 1]
- FR-3: Video I/O functions depend on `opencv-python` / `imageio`, which are optional dependencies under the `[video]` extra
- FR-4: When optional dependencies are missing, functions raise `ImportError` with a message indicating which extra to install
- FR-5: All functions must be testable on CPU-only environments — tests use small synthetic images or lightweight fixtures
- FR-6: The `load_image` function must handle EXIF orientation metadata to match ComfyUI's `LoadImage` behavior
- FR-7: Module is exposed as `comfy_diffusion.image` — functions are not auto-imported from `__init__.py`
- FR-8: `save_video` accepts a file path and infers the codec from the extension (`.webm` → VP9, `.mp4` → H.264) — one function, no format-specific variants
- FR-9: `load_image` accepts `str | Path` only (file paths). A separate `image_to_tensor(image: Image) -> Tensor` helper is provided for callers who already have a PIL `Image` object
- FR-10: `ltxv_preprocess` lives in `comfy_diffusion.image`, not in `conditioning.py` — it is an image transform, not a conditioning operation

## Non-Goals (Out of Scope)
- High-level pipeline abstractions (e.g. "load → process → save" convenience wrappers)
- GPU-specific optimizations or CUDA kernels
- Image transforms already covered by Pillow/numpy (resize, crop, flip, rotate, blur, sharpen) — per roadmap "external libraries over node ports" principle
- Audio-related video operations (handled by `comfy_diffusion.audio`)
- WAN/LTXV img2vid conditioning — already implemented in `conditioning.py`
- Upscale model loading — model loading belongs in `comfy_diffusion.models`
- UI-specific nodes (`SaveImage`, `PreviewImage`)

## Open Questions
- None — all resolved (see FR-8, FR-9, FR-10)
