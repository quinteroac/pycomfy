# Requirement: Image Generation Pipelines — SDXL, SDXL Turbo, Z-Image Turbo, Anima Preview

## Context

comfy-diffusion has mature video pipeline coverage (LTX, WAN) but no image generation pipelines yet.
This iteration adds the first batch of image pipelines, covering the official ComfyUI SDXL, SDXL Turbo,
Z-Image Turbo, and Anima Preview workflows.  Each pipeline follows the established `manifest()` + `run()`
contract and ships with a runnable example script and CPU smoke tests.

Two missing node wrappers must be implemented first:
- `latent.empty_sd3_latent_image()` — wraps `EmptySD3LatentImage` (used by Z-Image Turbo)
- `sampling.sd_turbo_scheduler()` — wraps `SDTurboScheduler` (used by SDXL Turbo)
- `sampling.sample_custom_simple()` — wraps `SamplerCustom` (distinct from `SamplerCustomAdvanced`; used by SDXL Turbo)

Already-available wrappers used by these pipelines:
`model_sampling_aura_flow`, `conditioning_zero_out`, `sample`, `sample_advanced`,
`empty_latent_image`, `vae_decode`, `encode_prompt`, `ModelManager.load_checkpoint`,
`ModelManager.load_unet`, `ModelManager.load_clip`, `ModelManager.load_vae`.

## Goals

- Deliver 5 image generation pipelines under `comfy_diffusion/pipelines/image/`
- Each pipeline mirrors its reference workflow exactly (node order, parameters, sampler settings)
- Expose two new node wrappers (`empty_sd3_latent_image`, `sd_turbo_scheduler`) and one new sampling function (`sample_custom_simple`) required by the new pipelines
- Ship one example script per pipeline under `examples/`
- All CPU smoke tests pass in CI; GPU validation done locally before merge

## User Stories

### US-001: SDXL base + refiner pipeline (shared prompt)
**As a** Python developer, **I want** to call `comfy_diffusion.pipelines.image.sdxl.t2i.run()` **so that** I can generate high-quality 1024×1024 images using the SDXL base + refiner two-pass workflow with a single shared prompt.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/pipelines/image/sdxl/t2i.py` exists with `manifest()` and `run()`
- [ ] `manifest()` returns exactly 2 `HFModelEntry` items: `checkpoints/sd_xl_base_1.0.safetensors` and `checkpoints/sd_xl_refiner_1.0.safetensors`
- [ ] `run()` accepts: `models_dir`, `prompt`, `negative_prompt`, `width`, `height`, `steps`, `base_end_step`, `cfg`, `seed` → returns `list[PIL.Image.Image]`
- [ ] Pass 1: `KSamplerAdvanced` on base model with `add_noise=enable`, `return_with_leftover_noise=enable`, `start_at_step=0`, `end_at_step=base_end_step` (default 20 out of 25)
- [ ] Pass 2: `KSamplerAdvanced` on refiner model with `add_noise=disable`, `return_with_leftover_noise=disable`, `start_at_step=base_end_step`, `end_at_step=10000`
- [ ] Refiner VAE used for final `vae_decode`
- [ ] `__all__ = ["manifest", "run"]`, `from __future__ import annotations`, module docstring present
- [ ] No top-level `comfy.*` or `torch` imports (lazy import pattern enforced)
- [ ] Typecheck / lint passes

### US-002: SDXL base + refiner pipeline (separate prompts per stage)
**As a** Python developer, **I want** to call `comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt.run()` **so that** I can provide a detailed base prompt and a refined prompt for the SDXL refiner stage separately.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/pipelines/image/sdxl/t2i_refiner_prompt.py` exists with `manifest()` and `run()`
- [ ] `manifest()` is identical to US-001 (same 2 checkpoints)
- [ ] `run()` accepts same args as US-001 plus `refiner_prompt: str | None` (defaults to `prompt` if `None`) and `refiner_negative_prompt: str | None` (defaults to `negative_prompt`)
- [ ] Base stage encodes `prompt`/`negative_prompt` with base CLIP; refiner stage encodes `refiner_prompt`/`refiner_negative_prompt` with refiner CLIP
- [ ] Two-pass `KSamplerAdvanced` flow identical to US-001
- [ ] `__all__ = ["manifest", "run"]`, lazy imports, module docstring present
- [ ] Typecheck / lint passes

### US-003: SDXL Turbo pipeline
**As a** Python developer, **I want** to call `comfy_diffusion.pipelines.image.sdxl.turbo.run()` **so that** I can generate 512×512 images in a single distilled step using SDXL Turbo.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/pipelines/image/sdxl/turbo.py` exists with `manifest()` and `run()`
- [ ] `manifest()` returns 1 `HFModelEntry`: `checkpoints/sd_xl_turbo_1.0_fp16.safetensors`
- [ ] `run()` accepts: `models_dir`, `prompt`, `negative_prompt`, `width`, `height`, `steps`, `cfg`, `seed` → returns `list[PIL.Image.Image]`
- [ ] Sampler: `euler_ancestral` via `get_sampler()`
- [ ] Sigmas: `sd_turbo_scheduler(model, steps=1, denoise=1.0)` (new wrapper — see FR-1)
- [ ] Sampling via `sample_custom_simple(model, add_noise=True, seed, cfg, positive, negative, sampler, sigmas, latent)` (new wrapper — see FR-2)
- [ ] `__all__ = ["manifest", "run"]`, lazy imports, module docstring present
- [ ] Typecheck / lint passes

### US-004: Z-Image Turbo pipeline
**As a** Python developer, **I want** to call `comfy_diffusion.pipelines.image.z_image.turbo.run()` **so that** I can generate images using the Z-Image Turbo distilled model with Qwen text encoder.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/pipelines/image/z_image/turbo.py` exists with `manifest()` and `run()`
- [ ] `manifest()` returns 3 `HFModelEntry` items: `diffusion_models/z_image_turbo_bf16.safetensors`, `text_encoders/qwen_3_4b.safetensors`, `vae/ae.safetensors`
- [ ] `run()` accepts: `models_dir`, `prompt`, `width`, `height`, `steps`, `seed` → returns `list[PIL.Image.Image]`
- [ ] `ModelManager.load_unet()` for the diffusion model; `ModelManager.load_clip()` with type `lumina2`; `ModelManager.load_vae()`
- [ ] `model_sampling_aura_flow(model, shift=3)` applied before sampling
- [ ] Latent created via `empty_sd3_latent_image(width, height, batch_size=1)` (new wrapper — see FR-3)
- [ ] Negative conditioning: `conditioning_zero_out(positive_cond)` (already in `conditioning.py`)
- [ ] Sampler: `res_multistep`, scheduler: `simple`, cfg: 1.0 (turbo — CFG-free effectively)
- [ ] `__all__ = ["manifest", "run"]`, lazy imports, module docstring present
- [ ] Typecheck / lint passes

### US-005: Anima Preview pipeline
**As a** Python developer, **I want** to call `comfy_diffusion.pipelines.image.anima.t2i.run()` **so that** I can generate anime-style images using the Anima Preview model with Qwen text encoder.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/pipelines/image/anima/t2i.py` exists with `manifest()` and `run()`
- [ ] `manifest()` returns 3 `HFModelEntry` items: `diffusion_models/anima-preview2.safetensors`, `text_encoders/qwen_3_06b_base.safetensors`, `vae/qwen_image_vae.safetensors`
- [ ] `run()` accepts: `models_dir`, `prompt`, `negative_prompt`, `width`, `height`, `steps`, `cfg`, `seed` → returns `list[PIL.Image.Image]`
- [ ] `ModelManager.load_unet()` for the diffusion model; `ModelManager.load_clip()` with type `stable_diffusion`; `ModelManager.load_vae()`
- [ ] Latent via `empty_latent_image(width, height, batch_size=1)` (existing wrapper)
- [ ] Sampler: `er_sde`, cfg: 4.0, 30 steps (workflow defaults)
- [ ] `__all__ = ["manifest", "run"]`, lazy imports, module docstring present
- [ ] Typecheck / lint passes

### US-006: New node wrappers for image pipelines
**As a** library developer, **I want** `latent.empty_sd3_latent_image()`, `sampling.sd_turbo_scheduler()`, and `sampling.sample_custom_simple()` exposed **so that** image pipelines can use them without inlining raw `comfy.*` calls.

**Acceptance Criteria:**
- [ ] `latent.empty_sd3_latent_image(width, height, batch_size=1)` added to `comfy_diffusion/latent.py`; wraps `comfy_extras.nodes_sd3.EmptySD3LatentImage`; added to `__all__`
- [ ] `sampling.sd_turbo_scheduler(model, steps, denoise=1.0)` added to `comfy_diffusion/sampling.py`; wraps `comfy_extras.nodes_custom_sampler.SDTurboScheduler`; added to `__all__`
- [ ] `sampling.sample_custom_simple(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)` added to `comfy_diffusion/sampling.py`; wraps `comfy_extras.nodes_custom_sampler.SamplerCustom`; returns `latent` dict; added to `__all__`
- [ ] All three wrappers follow the lazy-import pattern (no top-level `comfy.*`)
- [ ] Typecheck / lint passes

### US-007: Example scripts for each image pipeline
**As a** developer integrating comfy-diffusion, **I want** runnable example scripts in `examples/` for each image pipeline **so that** I can see working usage code I can copy and adapt.

**Acceptance Criteria:**
- [ ] `examples/sdxl_t2i.py` — covers SDXL base+refiner (shared prompt); CLI `--prompt`, `--models-dir`, `--output`, `--download-only`
- [ ] `examples/sdxl_refiner_prompt_t2i.py` — same as above but exposes `--refiner-prompt` CLI arg
- [ ] `examples/sdxl_turbo.py` — covers SDXL Turbo; CLI `--prompt`, `--models-dir`, `--output`, `--download-only`
- [ ] `examples/z_image_turbo.py` — covers Z-Image Turbo; CLI `--prompt`, `--models-dir`, `--output`, `--download-only`
- [ ] `examples/anima_preview.py` — covers Anima Preview; CLI `--prompt`, `--models-dir`, `--output`, `--download-only`
- [ ] All examples follow the pattern of existing example scripts (argparse, `download_models`, `run()`)
- [ ] Typecheck / lint passes

### US-008: CPU smoke tests for all new pipelines
**As a** maintainer, **I want** pytest tests that validate each pipeline's structure and mock-call `run()` on CPU **so that** CI passes without GPU.

**Acceptance Criteria:**
- [ ] Test files: `tests/test_pipelines_image_sdxl_t2i.py`, `tests/test_pipelines_image_sdxl_refiner_prompt.py`, `tests/test_pipelines_image_sdxl_turbo.py`, `tests/test_pipelines_image_z_image_turbo.py`, `tests/test_pipelines_image_anima_t2i.py`
- [ ] Each test file covers: file exists, parses, has future annotations, module docstring, `__all__ = ["manifest", "run"]`, no top-level comfy imports, `manifest()` returns correct number/type of entries, `run()` accepts expected signature, `run()` calls model loading + sampling stubs via `unittest.mock.patch`
- [ ] Tests for new wrappers (US-006): `tests/test_image_wrappers_it033.py` — validates `empty_sd3_latent_image`, `sd_turbo_scheduler`, `sample_custom_simple` exist and return correct shapes when mocked
- [ ] `uv run pytest` passes with no GPU

## Functional Requirements

- FR-1: `sampling.sd_turbo_scheduler(model, steps: int, denoise: float = 1.0) -> Any` — wraps `SDTurboScheduler.execute()` from `comfy_extras.nodes_custom_sampler`; lazy import only
- FR-2: `sampling.sample_custom_simple(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image) -> dict` — wraps `SamplerCustom.execute()` from `comfy_extras.nodes_custom_sampler`; returns the `output` latent (index 0); lazy import only
- FR-3: `latent.empty_sd3_latent_image(width: int, height: int, batch_size: int = 1) -> dict` — wraps `EmptySD3LatentImage.generate()` from `comfy_extras.nodes_sd3`; lazy import only
- FR-4: All pipeline `run()` functions must call `check_runtime()` first and raise `RuntimeError` with the error message if it returns an error dict
- FR-5: All pipeline `run()` functions derive model file paths from `manifest()` entries by default, with optional per-file overrides via keyword arguments (e.g. `unet_path`, `clip_path`, `vae_path`, `base_ckpt_path`, `refiner_ckpt_path`)
- FR-6: Pipeline modules live under `comfy_diffusion/pipelines/image/` with `__init__.py` files at each directory level
- FR-7: Each new sub-package (`image/`, `image/sdxl/`, `image/z_image/`, `image/anima/`) must have an `__init__.py`

## Non-Goals (Out of Scope)

- Z-Image base (non-turbo) pipeline — deferred to a future iteration
- SDXL ControlNet or img2img workflows
- Any editing or inpainting workflows
- Auto-downloading models in `run()` — callers must call `download_models(manifest())` first
- Adding a high-level `ImagePipeline` abstraction — modular `run()` functions only
- HuggingFace repo resolution for models where the official HF repo is unknown (placeholders acceptable in manifest with a comment)

## Open Questions

None — all model sources resolved from workflow MarkdownNote nodes and official ComfyUI examples pages.

### Resolved: HuggingFace repos per pipeline

| Pipeline | File | HF repo | HF path |
|---|---|---|---|
| SDXL base+refiner | `sd_xl_base_1.0.safetensors` | `stabilityai/stable-diffusion-xl-base-1.0` | `sd_xl_base_1.0.safetensors` |
| SDXL base+refiner | `sd_xl_refiner_1.0.safetensors` | `stabilityai/stable-diffusion-xl-refiner-1.0` | `sd_xl_refiner_1.0.safetensors` |
| SDXL Turbo | `sd_xl_turbo_1.0_fp16.safetensors` | `stabilityai/sdxl-turbo` | `sd_xl_turbo_1.0_fp16.safetensors` |
| Z-Image Turbo | `z_image_turbo_bf16.safetensors` | `Comfy-Org/z_image_turbo` | `split_files/diffusion_models/z_image_turbo_bf16.safetensors` |
| Z-Image Turbo | `qwen_3_4b.safetensors` | `Comfy-Org/z_image_turbo` | `split_files/text_encoders/qwen_3_4b.safetensors` |
| Z-Image Turbo | `ae.safetensors` | `Comfy-Org/z_image_turbo` | `split_files/vae/ae.safetensors` |
| Anima Preview | `anima-preview2.safetensors` | `circlestone-labs/Anima` | `split_files/diffusion_models/anima-preview2.safetensors` |
| Anima Preview | `qwen_3_06b_base.safetensors` | `circlestone-labs/Anima` | `split_files/text_encoders/qwen_3_06b_base.safetensors` |
| Anima Preview | `qwen_image_vae.safetensors` | `circlestone-labs/Anima` | `split_files/vae/qwen_image_vae.safetensors` |
