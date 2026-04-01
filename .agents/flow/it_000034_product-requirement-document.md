# Requirement: Flux.2 Klein Pipelines + Qwen Image Layered Pipeline

## Context

comfy-diffusion has SDXL, SDXL Turbo, Z-Image Turbo, and Anima Preview pipelines from it_000033.
This iteration adds the Flux.2 Klein family (generation + editing) and the Qwen Image Layered editing
pipeline — all under `comfy_diffusion/pipelines/image/`.

Seven new ComfyUI node wrappers must be implemented first:
- `latent.empty_flux2_latent_image()` — wraps `EmptyFlux2LatentImage` (Flux.2 Klein generation + editing)
- `conditioning.reference_latent()` — wraps `ReferenceLatent` (all flux_klein editing + qwen layered)
- `sampling.flux_kv_cache()` — wraps `FluxKVCache` (flux_klein 9b KV editing)
- `latent.empty_qwen_image_layered_latent_image()` — wraps `EmptyQwenImageLayeredLatentImage` (qwen layered)
- `image.image_scale_to_total_pixels()` — wraps `ImageScaleToTotalPixels` (flux_klein editing)
- `image.image_scale_to_max_dimension()` — wraps `ImageScaleToMaxDimension` (qwen layered)
- `image.get_image_size()` — wraps `GetImageSize` (flux_klein editing + qwen layered)

Already-available wrappers: `flux2_scheduler`, `latent_cut_to_batch`, `sample_advanced`,
`SamplerCustomAdvanced` path via `sample_advanced`, `cfg_guider`, `get_sampler`, `random_noise`,
`vae_encode/decode`, `encode_prompt`, `conditioning_zero_out`, `model_sampling_aura_flow`,
`sample`, all `ModelManager` loaders.

## Goals

- Deliver 7 new library wrappers (lazy import, `__all__`, type-annotated, tested on CPU)
- Deliver 8 pipelines under `comfy_diffusion/pipelines/image/`: 7 flux_klein + 1 qwen layered
- Each pipeline mirrors its reference workflow exactly (node order, params, sampler settings)
- Each pipeline ships with a runnable example script and CPU pytest tests

## User Stories

### US-001: New library wrappers — latent module
**As a** Python developer, **I want** `latent.empty_flux2_latent_image()` and
`latent.empty_qwen_image_layered_latent_image()` **so that** I can create the latent tensors
expected by Flux.2 Klein and Qwen Image Layered models without touching ComfyUI internals.

**Acceptance Criteria:**
- [ ] `empty_flux2_latent_image(width: int, height: int, batch_size: int = 1) -> dict` wraps `EmptyFlux2LatentImage` from `comfy_extras.nodes_flux`; lazy import; returns latent dict
- [ ] `empty_qwen_image_layered_latent_image(width: int, height: int, layers: int, batch_size: int = 1) -> dict` wraps `EmptyQwenImageLayeredLatentImage` from `comfy_extras.nodes_qwen`; lazy import; returns latent dict
- [ ] Both functions appear in `latent.__all__`
- [ ] CPU tests verify both return a dict with key `"samples"` with the expected tensor shape
- [ ] Typecheck / lint passes

### US-002: New library wrappers — conditioning module
**As a** Python developer, **I want** `conditioning.reference_latent()` **so that** I can
inject reference-image conditioning as required by Flux.2 Klein editing and Qwen Image Layered.

**Acceptance Criteria:**
- [ ] `reference_latent(conditioning: list, latent: dict) -> list` wraps `ReferenceLatent` from `comfy_extras.nodes_edit_model`; lazy import; returns conditioning list
- [ ] Function appears in `conditioning.__all__`
- [ ] CPU test verifies output is a list (mock conditioning and latent inputs accepted)
- [ ] Typecheck / lint passes

### US-003: New library wrappers — sampling module
**As a** Python developer, **I want** `sampling.flux_kv_cache()` **so that** I can apply the
KV-cache model patch required by the Flux.2 Klein 9B KV editing pipeline.

**Acceptance Criteria:**
- [ ] `flux_kv_cache(model: Any) -> Any` wraps `FluxKVCache` from `comfy_extras.nodes_flux`; lazy import; returns patched model
- [ ] Function appears in `sampling.__all__`
- [ ] CPU test verifies the function accepts a mock model object and returns a value
- [ ] Typecheck / lint passes

### US-004: New library wrappers — image module
**As a** Python developer, **I want** `image.image_scale_to_total_pixels()`,
`image.image_scale_to_max_dimension()`, and `image.get_image_size()` **so that** I can resize
input images to the resolutions required by flux_klein and qwen layered pipelines.

**Acceptance Criteria:**
- [ ] `image_scale_to_total_pixels(image: Tensor, upscale_method: str, megapixels: float, smallest_side: int) -> Tensor` wraps `ImageScaleToTotalPixels` from `comfy_extras.nodes_post_processing`; lazy import
- [ ] `image_scale_to_max_dimension(image: Tensor, upscale_method: str, max_dimension: int) -> Tensor` wraps `ImageScaleToMaxDimension` from `comfy_extras.nodes_images`; lazy import
- [ ] `get_image_size(image: Tensor) -> tuple[int, int]` wraps `GetImageSize` from `comfy_extras.nodes_images`; lazy import; returns `(width, height)` as a 2-tuple of ints
- [ ] All three appear in `image.__all__`
- [ ] CPU tests verify each function returns the expected type (tensor or tuple) using a small synthetic image
- [ ] Typecheck / lint passes

### US-005: Flux.2 Klein Text-to-Image pipelines (4B base + 4B distilled)
**As a** Python developer, **I want** `comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base`
and `comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled` **so that** I can generate
images from text using the Flux.2 Klein 4B model.

Reference workflow: `comfyui_official_workflows/image/generation/flux_klein/image_flux2_klein_text_to_image.json`

**Acceptance Criteria:**
- [ ] `t2i_4b_base.manifest()` returns 3 entries: `flux-2-klein-base-4b.safetensors` (diffusion_models), `qwen_3_4b.safetensors` (text_encoders), `flux2-vae.safetensors` (vae)
- [ ] `t2i_4b_base.run(prompt, width, height, steps, cfg, seed, models_dir, *, unet_filename, clip_filename, vae_filename) -> list[PIL.Image]` follows node order: `UNETLoader → CLIPLoader → VAELoader → RandomNoise → CLIPTextEncode (positive) → CLIPTextEncode (negative empty) → EmptyFlux2LatentImage → KSamplerSelect (euler) → Flux2Scheduler → CFGGuider (cfg=5) → SamplerCustomAdvanced → VAEDecode`
- [ ] `t2i_4b_distilled.manifest()` returns 3 entries with `flux-2-klein-4b.safetensors` (diffusion_models) instead; same clip + vae
- [ ] `t2i_4b_distilled.run(...)` uses `cfg=1`, `conditioning_zero_out` for negative, and `steps=4` default
- [ ] Both call `check_runtime()` and raise `RuntimeError` on failure
- [ ] CPU tests: mock-based, verify `manifest()` length and field names, verify `run()` returns `list[PIL.Image]`
- [ ] Example script `examples/flux_klein_t2i_4b_base.py` and `examples/flux_klein_t2i_4b_distilled.py` with argparse `--prompt`, `--width`, `--height`, `--steps`, `--seed`, `--models-dir`
- [ ] Typecheck / lint passes

### US-006: Flux.2 Klein Image Edit pipelines (4B base, 4B distilled, 9B base, 9B distilled, 9B KV)
**As a** Python developer, **I want** five flux_klein editing pipelines **so that** I can edit
images using the Flux.2 Klein models at different sizes and variants.

Reference workflows:
- `image/editing/flux_klein/image_flux2_klein_image_edit_4b_base.json`
- `image/editing/flux_klein/image_flux2_klein_image_edit_4b_distilled.json`
- `image/editing/flux_klein/image_flux2_klein_image_edit_9b_base.json`
- `image/editing/flux_klein/image_flux2_klein_image_edit_9b_distilled.json`
- `image/editing/flux_klein/image_flux2_klein_9b_kv_image_edit.json`

**Acceptance Criteria:**
- [ ] Each pipeline file is at `comfy_diffusion/pipelines/image/flux_klein/edit_{variant}.py` where variant ∈ `{4b_base, 4b_distilled, 9b_base, 9b_distilled, 9b_kv}`
- [ ] `manifest()` for each returns exactly the 3 entries declared in its workflow (unet + clip + vae filenames match the JSON model downloads section)
- [ ] Each `run(prompt, image, width, height, steps, cfg, seed, models_dir, *, unet_filename, clip_filename, vae_filename) -> list[PIL.Image]` follows the node execution order from its reference workflow, including `vae_encode → reference_latent` for positive + negative conditioning, `EmptyFlux2LatentImage`, `Flux2Scheduler`, `CFGGuider`, `SamplerCustomAdvanced`, `VAEDecode`
- [ ] `edit_9b_kv.run()` additionally calls `flux_kv_cache(model)` after model load, and uses `image_scale_to_total_pixels` to resize the input image; accepts two input images (reference + subject)
- [ ] Base-variant pipelines default `cfg=5`, `steps=20`; distilled variants default `cfg=1`, `steps=4`
- [ ] All five call `check_runtime()` and raise `RuntimeError` on failure
- [ ] CPU tests: mock-based, one test file per pipeline (`tests/test_pipelines_image_flux_klein_edit_4b_base.py`, etc.) — verify `manifest()` count and `run()` returns `list[PIL.Image]`
- [ ] Five example scripts with argparse `--prompt`, `--image`, `--width`, `--height`, `--seed`, `--models-dir`
- [ ] Typecheck / lint passes

### US-007: Qwen Image Layered pipeline (text-to-layers + image-to-layers)
**As a** Python developer, **I want** `comfy_diffusion.pipelines.image.qwen.layered` **so that**
I can generate layered (multi-layer) images from a text prompt or an input image using the
Qwen Image Layered model.

Reference workflow: `comfyui_official_workflows/image/editing/qwen/qwen2512/image_qwen_image_layered.json`

**Acceptance Criteria:**
- [ ] `manifest()` returns 3 entries: `qwen_image_layered_bf16.safetensors` (diffusion_models), `qwen_2.5_vl_7b_fp8_scaled.safetensors` (text_encoders), `qwen_image_layered_vae.safetensors` (vae)
- [ ] `run_t2l(prompt, width, height, layers, steps, cfg, seed, models_dir, *, unet_filename, clip_filename, vae_filename) -> list[PIL.Image]` implements the text-to-layers subgraph: `UNETLoader → CLIPLoader → VAELoader → ModelSamplingAuraFlow(shift=1) → CLIPTextEncode (positive, with text) → CLIPTextEncode (negative, empty) → EmptyQwenImageLayeredLatentImage → ReferenceLatent → KSampler → VAEDecode → LatentCutToBatch`
- [ ] `run_i2l(prompt, image, layers, steps, cfg, seed, models_dir, *, unet_filename, clip_filename, vae_filename) -> list[PIL.Image]` implements the image-to-layers subgraph: same flow but adds `GetImageSize → EmptyQwenImageLayeredLatentImage`, `image_scale_to_max_dimension(image, "lanczos", 640)` → `CLIPTextEncode` receives the scaled image, `vae_encode` for reference latent
- [ ] Both `run_t2l` and `run_i2l` default `steps=20`, `cfg=2.5`, sampler `euler`, scheduler `simple`, `layers=2`
- [ ] Both call `check_runtime()` and raise `RuntimeError` on failure
- [ ] CPU tests: mock-based, verify `manifest()` count, verify `run_t2l()` and `run_i2l()` return `list[PIL.Image]`
- [ ] Example scripts `examples/qwen_layered_t2l.py` and `examples/qwen_layered_i2l.py` with argparse `--prompt`, `--width`, `--height`, `--layers`, `--seed`, `--models-dir` (i2l adds `--image`)
- [ ] Typecheck / lint passes

### US-008: Package structure and `__init__.py` files
**As a** Python developer, **I want** `__init__.py` files for the new pipeline sub-packages
**so that** all new pipeline modules are importable.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/pipelines/image/flux_klein/__init__.py` created (may be empty)
- [ ] `comfy_diffusion/pipelines/image/qwen/__init__.py` created (may be empty)
- [ ] `from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base` succeeds at import time (no GPU required)
- [ ] `from comfy_diffusion.pipelines.image.qwen import layered` succeeds at import time
- [ ] Typecheck / lint passes

## Functional Requirements

- FR-1: `latent.empty_flux2_latent_image(width, height, batch_size=1)` wraps `EmptyFlux2LatentImage` (lazy import from `comfy_extras.nodes_flux`); returns `{"samples": Tensor}`
- FR-2: `latent.empty_qwen_image_layered_latent_image(width, height, layers, batch_size=1)` wraps `EmptyQwenImageLayeredLatentImage` (lazy import from `comfy_extras.nodes_qwen`); returns `{"samples": Tensor}`
- FR-3: `conditioning.reference_latent(conditioning, latent)` wraps `ReferenceLatent` (lazy import from `comfy_extras.nodes_edit_model`); returns conditioning list
- FR-4: `sampling.flux_kv_cache(model)` wraps `FluxKVCache` (lazy import from `comfy_extras.nodes_flux`); returns patched model
- FR-5: `image.image_scale_to_total_pixels(image, upscale_method, megapixels, smallest_side)` wraps `ImageScaleToTotalPixels` (lazy import from `comfy_extras.nodes_post_processing`); returns tensor
- FR-6: `image.image_scale_to_max_dimension(image, upscale_method, max_dimension)` wraps `ImageScaleToMaxDimension` (lazy import from `comfy_extras.nodes_images`); returns tensor
- FR-7: `image.get_image_size(image)` wraps `GetImageSize` (lazy import from `comfy_extras.nodes_images`); returns `(width: int, height: int)`
- FR-8: All 8 pipelines reside under `comfy_diffusion/pipelines/image/` in their respective sub-packages (`flux_klein/`, `qwen/`)
- FR-9: Every pipeline exports `manifest() -> list[ModelEntry]` and one or more `run*(...)` functions returning `list[PIL.Image]`
- FR-10: Every pipeline `run*()` calls `check_runtime()` as its first statement and raises `RuntimeError(error["error"])` if `"error"` is present in the returned dict
- FR-11: All new wrappers use the lazy import pattern (no `torch` or `comfy.*` at module top level); all appear in their module's `__all__`
- FR-12: All tests pass on CPU (no GPU required); mocks/stubs used for model weights and ComfyUI runtime calls

## Non-Goals (Out of Scope)

- `image/editing/qwen/qwen2512/image_qwen_image_layered_control.json` — the ControlNet variant of Qwen layered is excluded from this iteration
- Flux.2 Klein ControlNet workflows (under `image/controlnet/`)
- Any other Qwen editing variants (qwen2508, qwen2509, qwen2511)
- Video pipelines (WAN, LTX remaining variants)
- GPU end-to-end validation (done locally by the developer before merging, not in CI)
- High-level pipeline abstraction or auto-import at `comfy_diffusion` package level

## Open Questions

- None
