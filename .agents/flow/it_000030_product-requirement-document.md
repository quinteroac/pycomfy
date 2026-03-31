# Requirement: WAN 2.1 Pipelines — T2V, I2V, FLF2V

## Context

Iteration 000029 completed the full LTX pipeline family (Phase 6, audio-to-video).
This iteration delivers the first three WAN 2.1 pipelines (`t2v`, `i2v`, `flf2v`) as
importable Python modules, matching the existing pipeline contract (`manifest()` / `run()`).

One library node wrapper is missing and must be added first: `empty_wan_latent_video()` in
`comfy_diffusion/latent.py`. All conditioning helpers (`wan_image_to_video`,
`wan_first_last_frame_to_video`, `encode_clip_vision`, `model_sampling_sd3`) and model
loaders (`ModelManager.load_unet`, `load_clip`, `load_vae`, `load_clip_vision`) are already
implemented. All three pipelines mirror the workflow JSON files in
`comfyui_official_workflows/video/wan/wan2.1/` exactly.

---

## Goals

- Expose `empty_wan_latent_video()` in `comfy_diffusion.latent` so callers can create
  empty video latents for WAN models without importing raw `comfy.*` code.
- Implement three WAN 2.1 pipelines under `comfy_diffusion/pipelines/video/wan/wan21/`:
  `t2v`, `i2v`, `flf2v`.
- Provide one example script per pipeline under `examples/`.
- Wire the new `wan` sub-package into `comfy_diffusion/pipelines/video/__init__.py`.

---

## User Stories

### US-001: `empty_wan_latent_video` wrapper

**As a** Python developer, **I want** `empty_wan_latent_video(width, height, length,
batch_size)` in `comfy_diffusion.latent` **so that** I can create zeroed video latent
tensors for WAN models without calling ComfyUI nodes directly.

**Acceptance Criteria:**
- [ ] `empty_wan_latent_video(width: int, height: int, length: int = 33,
      batch_size: int = 1) -> dict[str, Any]` is implemented in `latent.py`.
- [ ] The returned dict has `"samples"` (shape `[batch, 16, ((length-1)//4)+1, height//8,
      width//8]`) and `"downscale_ratio_spacial": 8`, matching
      `EmptyHunyuanLatentVideo.execute()` in `comfy_extras/nodes_hunyuan.py`.
- [ ] The function follows the lazy-import pattern (no `torch` or `comfy.*` at module
      top level).
- [ ] `"empty_wan_latent_video"` is added to `latent.__all__`.
- [ ] Typecheck / lint passes.

---

### US-002: WAN 2.1 text-to-video pipeline

**As a** Python developer, **I want**
`comfy_diffusion.pipelines.video.wan.wan21.t2v` with `manifest()` and `run()`
**so that** I can generate a video from a text prompt using the WAN 2.1 T2V 1.3B model
without manually assembling node calls.

**Acceptance Criteria:**
- [ ] `manifest()` returns a `list[ModelEntry]` with exactly 3 entries matching the
      active nodes in `text_to_video_wan.json`:
      - UNet: `wan2.1_t2v_1.3B_fp16.safetensors` → `diffusion_models/`
      - CLIP: `umt5_xxl_fp8_e4m3fn_scaled.safetensors` → `text_encoders/`
      - VAE: `wan_2.1_vae.safetensors` → `vae/`
- [ ] `run(*, models_dir, prompt, negative_prompt, width=832, height=480, length=33,
      fps=16, steps=30, cfg=6.0, sampler="uni_pc", scheduler="simple", seed,
      shift=8.0)` executes the workflow in node order:
      1. Load UNet, CLIP, VAE
      2. `encode_prompt` (positive + negative)
      3. `model_sampling_sd3(model, shift=8.0)`
      4. `empty_wan_latent_video(width, height, length, batch_size=1)`
      5. `sample()`
      6. `vae_decode()`
      7. Return `{"frames": list[PIL.Image.Image]}`
- [ ] `__all__ = ["manifest", "run"]`.
- [ ] Typecheck / lint passes.

---

### US-003: WAN 2.1 image-to-video pipeline

**As a** Python developer, **I want**
`comfy_diffusion.pipelines.video.wan.wan21.i2v` with `manifest()` and `run()`
**so that** I can animate a single image using the WAN 2.1 I2V 480p 14B model.

**Acceptance Criteria:**
- [ ] `manifest()` returns a `list[ModelEntry]` with exactly 4 entries matching active
      nodes in `image_to_video_wan.json`:
      - UNet: `wan2.1_i2v_480p_14B_fp16.safetensors` → `diffusion_models/`
      - CLIP: `umt5_xxl_fp8_e4m3fn_scaled.safetensors` → `text_encoders/`
      - VAE: `wan_2.1_vae.safetensors` → `vae/`
      - CLIP Vision: `clip_vision_h.safetensors` → `clip_vision/`
- [ ] `run(*, models_dir, prompt, negative_prompt, image, width=512, height=512,
      length=33, fps=16, steps=20, cfg=6.0, sampler="uni_pc", scheduler="simple",
      seed, shift=8.0, clip_vision_crop="none")` executes the workflow in node order:
      1. Load UNet, CLIP, VAE, CLIP Vision
      2. `encode_prompt` (positive + negative)
      3. `encode_clip_vision(clip_vision, image, crop=clip_vision_crop)`
      4. `model_sampling_sd3(model, shift=8.0)`
      5. `wan_image_to_video(positive, negative, vae, clip_vision_output, image,
         width, height, length, batch_size=1)` → `(positive, negative, latent)`
      6. `sample()`
      7. `vae_decode()`
      8. Return `{"frames": list[PIL.Image.Image]}`
- [ ] `image` parameter accepts `str | Path | PIL.Image.Image`.
- [ ] `__all__ = ["manifest", "run"]`.
- [ ] Typecheck / lint passes.

---

### US-004: WAN 2.1 first-last-frame-to-video pipeline

**As a** Python developer, **I want**
`comfy_diffusion.pipelines.video.wan.wan21.flf2v` with `manifest()` and `run()`
**so that** I can generate a video interpolating between a start and end frame using the
WAN 2.1 FLF2V 720p 14B model.

**Acceptance Criteria:**
- [ ] `manifest()` returns a `list[ModelEntry]` with exactly 4 entries matching active
      nodes in `wan2.1_flf2v_720_f16.json`:
      - UNet: `wan2.1_flf2v_720p_14B_fp16.safetensors` → `diffusion_models/`
      - CLIP: `umt5_xxl_fp8_e4m3fn_scaled.safetensors` → `text_encoders/`
      - VAE: `wan_2.1_vae.safetensors` → `vae/`
      - CLIP Vision: `clip_vision_h.safetensors` → `clip_vision/`
- [ ] `run(*, models_dir, prompt, negative_prompt, start_image, end_image, width=720,
      height=1280, length=33, fps=16, steps=20, cfg=3.0, sampler="uni_pc",
      scheduler="simple", seed, shift=8.0, clip_vision_crop="none")` executes the
      workflow in node order:
      1. Load UNet, CLIP, VAE, CLIP Vision
      2. `encode_prompt` (positive + negative)
      3. `encode_clip_vision(clip_vision, start_image, crop=clip_vision_crop)` → `cv_start`
      4. `encode_clip_vision(clip_vision, end_image, crop=clip_vision_crop)` → `cv_end`
      5. `model_sampling_sd3(model, shift=8.0)`
      6. `wan_first_last_frame_to_video(positive, negative, vae, cv_start, cv_end,
         start_image, end_image, width, height, length, batch_size=1)`
         → `(positive, negative, latent)`
      7. `sample()`
      8. `vae_decode()`
      9. Return `{"frames": list[PIL.Image.Image]}`
- [ ] `start_image` and `end_image` accept `str | Path | PIL.Image.Image`.
- [ ] `__all__ = ["manifest", "run"]`.
- [ ] Typecheck / lint passes.

---

### US-005: Pipeline sub-package wiring

**As a** Python developer, **I want** the `wan` sub-package properly wired into the
`pipelines.video` namespace **so that** I can discover and import WAN pipelines
consistently with the LTX family.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/pipelines/video/wan/__init__.py` exports `"wan21"`.
- [ ] `comfy_diffusion/pipelines/video/wan/wan21/__init__.py` exports
      `["t2v", "i2v", "flf2v"]`.
- [ ] `comfy_diffusion/pipelines/video/__init__.py` `__all__` includes `"wan"`.
- [ ] Typecheck / lint passes.

---

### US-006: Example scripts

**As a** Python developer, **I want** CLI example scripts for each new pipeline
**so that** I can test end-to-end inference from the command line with minimal boilerplate.

**Acceptance Criteria:**
- [ ] `examples/video_wan21_t2v.py` — accepts `--models-dir`, `--prompt`,
      `--negative-prompt`, `--width`, `--height`, `--length`, `--fps`, `--steps`,
      `--cfg`, `--seed`, `--output`; calls `download_models` + `run`.
- [ ] `examples/video_wan21_i2v.py` — same as above, plus `--image` (required).
- [ ] `examples/video_wan21_flf2v.py` — same as above, plus `--start-image` and
      `--end-image` (both required).
- [ ] All scripts: heavy imports deferred inside `main()`; missing required args print
      usage error; output saved as MP4 (via PyAV) or PNG frames as fallback.
- [ ] All scripts follow the same structure as
      `examples/video_ltx2_audio_to_video.py`.
- [ ] Typecheck / lint passes.

---

## Functional Requirements

- **FR-1:** `latent.empty_wan_latent_video(width, height, length=33, batch_size=1)` —
  returns `{"samples": tensor[batch, 16, ((length-1)//4)+1, h//8, w//8],
  "downscale_ratio_spacial": 8}`; lazy-import; added to `latent.__all__`.
- **FR-2:** `pipelines/video/wan/wan21/t2v.py` — `manifest()` (3 entries) + `run()`
  following `text_to_video_wan.json` node order.
- **FR-3:** `pipelines/video/wan/wan21/i2v.py` — `manifest()` (4 entries) + `run()`
  following `image_to_video_wan.json` node order; `image` param accepts
  `str | Path | PIL.Image.Image`.
- **FR-4:** `pipelines/video/wan/wan21/flf2v.py` — `manifest()` (4 entries) + `run()`
  following `wan2.1_flf2v_720_f16.json` node order; `start_image`/`end_image` accept
  `str | Path | PIL.Image.Image`.
- **FR-5:** `pipelines/video/wan/__init__.py` — `__all__ = ["wan21"]`.
- **FR-6:** `pipelines/video/wan/wan21/__init__.py` — `__all__ = ["t2v", "i2v", "flf2v"]`.
- **FR-7:** `pipelines/video/__init__.py` — `__all__` updated to include `"wan"`.
- **FR-8:** `examples/video_wan21_t2v.py` — CLI script following existing example pattern.
- **FR-9:** `examples/video_wan21_i2v.py` — CLI script with `--image` argument.
- **FR-10:** `examples/video_wan21_flf2v.py` — CLI script with `--start-image` and
  `--end-image` arguments.

---

## Non-Goals (Out of Scope)

- WAN 2.2 pipelines — these follow in a future iteration.
- WAN Fun, VACE, ATI, Move, InfiniteTalk, SCAIL sub-family pipelines.
- HunyuanVideo pipelines.
- Audio pipelines (ACE Step, Chatterbox, Stable Audio).
- CPU-safe pytest tests for the new wrappers and pipelines (test plan is written during
  the Refactor phase, per project convention).
- GPU validation — CI is CPU-only.
- Modifying the ComfyUI submodule pin.

---

## Open Questions

- None.
