# Requirement: LTX2/LTX3 Base Pipelines (Phase 3)

## Context

Implement all remaining Phase 3 base pipelines from the roadmap as Python modules under
`comfy_diffusion/pipelines/`. The `ltx2_t2v` pipeline (implemented in it_000024) is the
canonical pattern: each pipeline file exports `manifest() -> list[ModelEntry]` and
`run(...) -> list[PIL.Image.Image]`.

Each pipeline must be a faithful replica of its corresponding official ComfyUI workflow in
`comfyui_official_workflows/video/ltx/`, translated to the `comfy_diffusion` public API.

**Already done (it_000024):** `ltx2_t2v` → `comfy_diffusion/pipelines/ltx2_t2v.py`

**Available building blocks (it_000025):**
- `ModelManager.load_ltxav_text_encoder(path, device)` — Gemma 3 / T5 encoder via
  `LTXAVTextEncoderLoader`
- `ModelManager.load_ltxav_audio_vae(path)` — Audio VAE via `LTXVAudioVAELoader`
- `ltxv_img_to_video_inplace(vae, image, latent, strength)` — image frame injection
- `ltxv_preprocess(image, width, height)` — LTX-specific image preprocessing
- `ltxv_latent_upsample(latent, upscale_model, vae)` — spatial upscaling (latent.py)

---

## Goals

- Implement 7 remaining Phase 3 pipeline files (one file per workflow)
- Each exposes `manifest()` and `run()` following the `ltx2_t2v.py` pattern
- `manifest()` is the single source of truth for model file paths used in `run()`
- CPU tests pass in CI (mocked weights); GPU smoke test documented as a manual step
- Pipelines replicate their corresponding workflow JSONs in
  `comfyui_official_workflows/video/ltx/ltx2/` and `ltx3/`

---

## User Stories

### US-001: `ltx2_t2v_distilled` pipeline
**As a** Python developer, **I want** to call `ltx2_t2v_distilled.run(models_dir, prompt)`
**so that** I generate a video using the LTX-Video 2 distilled model with the Gemma 3 text
encoder and spatial upscaler, matching `video_ltx2_t2v_distilled.json`.

**Reference workflow:** `comfyui_official_workflows/video/ltx/ltx2/video_ltx2_t2v_distilled.json`

**Models required:**
- `diffusion_models/ltx-2-19b-distilled.safetensors` (HF: `Lightricks/LTX-Video`)
- `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` (HF: `Lightricks/LTX-Video`)
- `upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors` (HF: `Lightricks/LTX-Video`)

**Key differences from `ltx2_t2v`:**
- UNet loaded via `mm.load_unet()` with the distilled checkpoint
- Text encoder loaded via `mm.load_ltxav_text_encoder()` (Gemma 3 instead of T5)
- Spatial upscale step via `ltxv_latent_upsample()` after sampling, before VAE decode
- Default `steps=8` (distilled model requires fewer steps)

**Acceptance Criteria:**
- [ ] `manifest()` returns 3 `HFModelEntry` items with correct `dest` paths
- [ ] `run()` accepts `models_dir`, `prompt`, `negative_prompt`, `width`, `height`,
  `length`, `steps`, `cfg`, `seed`, and per-model filename overrides
- [ ] Pipeline calls `ltxv_latent_upsample()` after sampling and before `vae_decode_batch_tiled()`
- [ ] CPU test passes with mocked `ModelManager` and no GPU
- [ ] Typecheck / lint passes

---

### US-002: `ltx2_i2v` pipeline
**As a** Python developer, **I want** to call `ltx2_i2v.run(models_dir, image, prompt)`
**so that** I generate a video from an input image using the LTX-Video 2 dev model,
matching `video_ltx2_i2v.json`.

**Reference workflow:** `comfyui_official_workflows/video/ltx/ltx2/video_ltx2_i2v.json`

**Models required:**
- `diffusion_models/ltx-2-19b-dev-fp8.safetensors` (HF: `Lightricks/LTX-Video`)
- `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` (HF: `Lightricks/LTX-Video`)
- `loras/ltx-2-19b-distilled-lora-384.safetensors` (HF: `Lightricks/LTX-Video`)
- `upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors` (HF: `Lightricks/LTX-Video`)

**Key differences from `ltx2_t2v`:**
- `run()` accepts an `image` parameter (`str | Path | PIL.Image`)
- Image is preprocessed via `ltxv_preprocess()` before being injected
- `ltxv_img_to_video_inplace(vae, image_tensor, latent)` injects the frame into the latent
- LoRA applied via `apply_lora()` after model load
- Spatial upscale step via `ltxv_latent_upsample()` after sampling

**Acceptance Criteria:**
- [ ] `manifest()` returns 4 `HFModelEntry` items
- [ ] `run()` accepts `image` (path or PIL) in addition to all t2v parameters
- [ ] Image is preprocessed with `ltxv_preprocess()` before `ltxv_img_to_video_inplace()`
- [ ] LoRA (`ltx-2-19b-distilled-lora-384`) is applied via `apply_lora()`
- [ ] CPU test passes with mocked inputs
- [ ] Typecheck / lint passes

---

### US-003: `ltx2_i2v_distilled` pipeline
**As a** Python developer, **I want** to call `ltx2_i2v_distilled.run(models_dir, image, prompt)`
**so that** I generate a video from an input image using the distilled LTX-Video 2 model,
matching `video_ltx2_i2v_distilled.json`.

**Reference workflow:** `comfyui_official_workflows/video/ltx/ltx2/video_ltx2_i2v_distilled.json`

**Models required:**
- `diffusion_models/ltx-2-19b-distilled.safetensors` (HF: `Lightricks/LTX-Video`)
- `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` (HF: `Lightricks/LTX-Video`)
- `upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors` (HF: `Lightricks/LTX-Video`)

**Key differences from `ltx2_i2v`:** Uses distilled checkpoint; no LoRA in base workflow.

**Acceptance Criteria:**
- [ ] `manifest()` returns 3 `HFModelEntry` items
- [ ] `run()` mirrors `ltx2_i2v.run()` signature (same parameters) but uses distilled checkpoint
- [ ] Default `steps=8`
- [ ] No `apply_lora()` call (no LoRA in this workflow)
- [ ] CPU test passes with mocked inputs
- [ ] Typecheck / lint passes

---

### US-004: `ltx2_i2v_lora` pipeline
**As a** Python developer, **I want** to call `ltx2_i2v_lora.run(models_dir, image, prompt, lora_path)`
**so that** I generate a video from an image with an additional style LoRA applied,
matching `video_ltx2_i2v_lora.json`.

**Reference workflow:** `comfyui_official_workflows/video/ltx/ltx2/video_ltx2_i2v_lora.json`

**Models required (base):**
- `diffusion_models/ltx-2-19b-dev.safetensors` (HF: `Lightricks/LTX-Video`)
- `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` (HF: `Lightricks/LTX-Video`)
- `loras/ltx-2-19b-distilled-lora-384.safetensors` (HF: `Lightricks/LTX-Video`)
- `upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors` (HF: `Lightricks/LTX-Video`)
- Style LoRA: caller-supplied path (not in `manifest()` — user provides their own)

**Key differences from `ltx2_i2v`:**
- No `ResizeImageMaskNode` in workflow; image loaded directly without resize preprocessing
- Two LoRAs stacked: base distilled LoRA + caller-supplied style LoRA
- `run()` accepts `lora_path` (path to style LoRA) and `lora_strength` (default `1.0`)
- Default resolution is square (1280×1280) per reference workflow

**Acceptance Criteria:**
- [ ] `manifest()` returns 4 `HFModelEntry` items (base models only; style LoRA excluded)
- [ ] `run()` accepts `lora_path: str | Path` and `lora_strength: float = 1.0`
- [ ] Both LoRAs are applied via `apply_lora()` (stacked, base LoRA first)
- [ ] Default `width=1280`, `height=1280` (square)
- [ ] CPU test passes with mocked inputs
- [ ] Typecheck / lint passes

---

### US-005: `ltx3_t2v` pipeline
**As a** Python developer, **I want** to call `ltx3_t2v.run(models_dir, prompt)`
**so that** I generate a video using the LTX-Video 2.3 model (22B distilled),
matching `video_ltx2_3_t2v.json`.

**Reference workflow:** `comfyui_official_workflows/video/ltx/ltx3/video_ltx2_3_t2v.json`

**Models required:**
- `diffusion_models/ltx-2.3-22b-distilled-fp8.safetensors` (HF: `Lightricks/LTX-Video`)
- `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` (HF: `Lightricks/LTX-Video`)
- `upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors` (HF: `Lightricks/LTX-Video`)

**Key differences from `ltx2_t2v`:**
- Uses 22B distilled LTX 2.3 checkpoint
- Gemma 3 text encoder via `load_ltxav_text_encoder()`
- Spatial upscale step after sampling via `ltxv_latent_upsample()`
- Default `steps=8` (distilled)

**Acceptance Criteria:**
- [ ] `manifest()` returns 3 `HFModelEntry` items
- [ ] `run()` matches same parameter signature as `ltx2_t2v_distilled.run()`
- [ ] CPU test passes with mocked inputs
- [ ] Typecheck / lint passes

---

### US-006: `ltx3_i2v` pipeline
**As a** Python developer, **I want** to call `ltx3_i2v.run(models_dir, image, prompt)`
**so that** I generate a video from an image using the LTX-Video 2.3 model,
matching `video_ltx2_3_i2v.json`.

**Reference workflow:** `comfyui_official_workflows/video/ltx/ltx3/video_ltx2_3_i2v.json`

**Models required:** Same 3 models as `ltx3_t2v`.

**Key differences from `ltx3_t2v`:**
- Accepts `image` parameter
- Accepts `fps: int = 24` (explicit frame-rate parameter exposed by this workflow)
- Image injected via `ltxv_img_to_video_inplace()`

**Acceptance Criteria:**
- [ ] `manifest()` returns same 3 entries as `ltx3_t2v`
- [ ] `run()` adds `image` and `fps` parameters vs `ltx3_t2v.run()`
- [ ] `fps` is passed to `ltxv_empty_latent_video()` if supported, else documented as reserved
- [ ] CPU test passes with mocked inputs
- [ ] Typecheck / lint passes

---

### US-007: CPU tests for all pipelines
**As a** CI system, **I want** all 6 new pipeline files to have CPU-only pytest tests
**so that** the test suite passes without a GPU on every commit.

**Acceptance Criteria:**
- [ ] Each pipeline has at least one test in `tests/` that:
  - Mocks `ModelManager` and all heavy imports (torch, comfy.*)
  - Calls `manifest()` and asserts it returns a non-empty list of `ModelEntry`
  - Calls `run()` with mock models and asserts the return value is a non-empty list
- [ ] `uv run pytest` passes on CI (CPU-only)
- [ ] Typecheck / lint passes

---

### US-009: GPU smoke test documentation
**As a** developer, **I want** each pipeline's docstring to include a manual GPU smoke test
snippet **so that** I can verify real inference before merging.

**Acceptance Criteria:**
- [ ] Each pipeline's module docstring contains a "Usage" section with a runnable
  `download_models(manifest(), models_dir=...)` + `run(...)` snippet
- [ ] The `run()` docstring lists all parameters with types and default values
- [ ] A `SMOKE_TEST.md` (or section in existing docs) enumerates one GPU invocation
  command per pipeline for manual pre-merge validation

---

## Functional Requirements

- FR-1: All pipeline files live under `comfy_diffusion/pipelines/` with snake_case names
  matching the workflow filename (e.g. `ltx2_t2v_distilled.py`).
- FR-2: `manifest()` must be the single source of truth — `run()` derives all default
  model paths from `manifest()`, never hardcodes paths independently.
- FR-3: All imports of `torch`, `comfy.*`, and `comfy_diffusion.*` heavy symbols inside
  pipeline `run()` functions must be lazy (inside the function body, not at module top level).
- FR-4: `ltxv_empty_latent_video()` in `latent.py` must be extended to accept an optional
  `fps: int | None = None` parameter (needed by US-006 / `ltx3_i2v`).
- FR-5: `path` type annotations must use `str | Path` (not `str | os.PathLike`).
- FR-6: All new pipeline files must be listed in `comfy_diffusion/pipelines/__init__.py`
  (or equivalent registry) if one exists.

---

## Non-Goals (Out of Scope)

- `ltx3_flf2v` pipeline — deferred to a later iteration; requires `LTXVAddGuide`,
  `LTXVCropGuides`, `LTXVConditioning`, `LTXVConcatAVLatent`, and `LTXVAudioVAEDecode`
  which will be exposed in Phase 4
- Controlnet / canny / depth / pose pipelines (Phase 5 of the roadmap)
- Audio-to-video pipeline (`video_ltx_2_audio_to_video`, Phase 7)
- WAN, HunyuanVideo, or any non-LTX pipeline
- Fine-tuning, training, or LoRA creation workflows
- Streaming or chunked inference
- Any UI or server layer
