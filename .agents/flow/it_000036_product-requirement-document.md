# Requirement: Qwen Image Edit 2511 Pipeline

## Context

Implement `comfy_diffusion/pipelines/image/qwen/edit_2511.py` — a programmatic
Python interface for the official ComfyUI workflow
`comfyui_official_workflows/image/editing/qwen/qwen2511/image_qwen_image_edit_2511.json`.

The workflow performs multi-reference image editing using the Qwen Image 2511
model: it accepts a primary input image plus up to two optional reference images
and a text prompt, applies AuraFlow sampling with CFGNorm patching, and returns
an edited `PIL.Image`.

Four ComfyUI nodes used by this workflow are not yet wrapped in the
comfy-diffusion library and must be added before the pipeline can be
implemented:

| Node | Module | New function |
|---|---|---|
| `TextEncodeQwenImageEditPlus` | `conditioning.py` | `encode_qwen_image_edit_plus()` |
| `FluxKontextMultiReferenceLatentMethod` | `conditioning.py` | `apply_flux_kontext_multi_reference()` |
| `FluxKontextImageScale` | `image.py` | `flux_kontext_image_scale()` |
| `CFGNorm` | `video.py` | `apply_cfg_norm()` |

## Goals

- Expose Qwen Image Edit 2511 inference as a composable Python function usable
  with `from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest, run`.
- Add the four missing node wrappers to their respective library modules,
  following all established lazy-import and `__all__` conventions.
- Keep the pipeline CPU-safe: all tests pass without GPU using mocked weights.

## User Stories

### US-001: Add missing node wrappers to library modules

**As a** developer using comfy-diffusion, **I want** the four missing ComfyUI
nodes wrapped as library functions **so that** I (and the pipeline) can call
them without inlining raw `comfy.*` logic inside a pipeline file.

**Acceptance Criteria:**
- [ ] `conditioning.py` exports `encode_qwen_image_edit_plus(clip, vae, image1, image2=None, image3=None, prompt="") -> Any` — mirrors `TextEncodeQwenImageEditPlus.execute()` from `vendor/ComfyUI/comfy_extras/nodes_qwen.py`; lazy import; added to `__all__`
- [ ] `conditioning.py` exports `apply_flux_kontext_multi_reference(conditioning, reference_latents_method="index_timestep_zero") -> Any` — mirrors `FluxKontextMultiReferenceLatentMethod.execute()` from `vendor/ComfyUI/comfy_extras/nodes_flux.py`; lazy import; added to `__all__`
- [ ] `image.py` exports `flux_kontext_image_scale(image) -> Any` — mirrors `FluxKontextImageScale.execute()` from `vendor/ComfyUI/comfy_extras/nodes_flux.py`; lazy import; added to `__all__`
- [ ] `video.py` exports `apply_cfg_norm(model, strength=1.0) -> Any` — mirrors `CFGNorm.execute()` from `vendor/ComfyUI/comfy_extras/nodes_cfg.py`; lazy import; added to `__all__`
- [ ] Typecheck / lint passes

### US-002: Implement the `edit_2511` pipeline module

**As a** developer, **I want** a `run()` function that accepts a prompt and an
input image (plus optional reference images) and returns a `list[PIL.Image]`
**so that** I can integrate Qwen image editing into my application with a single
function call.

**Acceptance Criteria:**
- [ ] File `comfy_diffusion/pipelines/image/qwen/edit_2511.py` exists and is importable
- [ ] `manifest() -> list[ModelEntry]` returns exactly 4 entries matching the workflow model downloads:
  - `diffusion_models/qwen_image_edit_2511_bf16.safetensors`
  - `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`
  - `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
  - `vae/qwen_image_vae.safetensors`
- [ ] `run(prompt, image, image2=None, image3=None, models_dir, *, steps=40, cfg=3.0, use_lora=True, seed=0) -> list[PIL.Image]` executes the full node graph in the order specified by the workflow:
  1. `mm.load_vae()` → `mm.load_unet()` → `mm.load_clip()`
  2. `model_sampling_aura_flow(model, shift=3.1)`
  3. `apply_cfg_norm(model, strength=1)`
  4. `apply_lora(model, None, lora_path, 1.0, 0.0)` when `use_lora=True`; `ComfySwitchNode` logic is an inline Python conditional
  5. `flux_kontext_image_scale(image)` on the input image
  6. `encode_qwen_image_edit_plus(clip, vae, scaled_image, prompt="")` → negative conditioning
  7. `encode_qwen_image_edit_plus(clip, vae, scaled_image, image2, image3, prompt)` → positive conditioning
  8. `apply_flux_kontext_multi_reference(negative, "index_timestep_zero")`
  9. `apply_flux_kontext_multi_reference(positive, "index_timestep_zero")`
  10. `vae_encode(vae, scaled_image)` → latent
  11. `sample(model, positive, negative, latent, steps, cfg, "euler", "simple", seed)`
  12. `vae_decode(vae, latent)` → `list[PIL.Image]`
- [ ] Return value is a non-empty `list[PIL.Image]`
- [ ] `__all__ = ["manifest", "run"]` is present
- [ ] Typecheck / lint passes

### US-003: CPU-safe unit tests

**As a** CI system (CPU-only), **I want** tests that validate `manifest()` and
the `run()` call graph without loading real model weights **so that** the
pipeline is verified on every pull request.

**Acceptance Criteria:**
- [ ] `tests/test_qwen_edit_2511.py` exists
- [ ] Test verifies `manifest()` returns exactly 4 `ModelEntry` items with the correct filenames and destination directories
- [ ] Test for `run()` stubs all model loading and node functions with `unittest.mock`; asserts `vae_decode` is called and result is returned
- [ ] Tests also cover the four new node-wrapper functions with mocked `comfy.*` internals
- [ ] All tests pass under `uv run pytest tests/test_qwen_edit_2511.py` on CPU

## Functional Requirements

- FR-1: `encode_qwen_image_edit_plus` must accept `image2` and `image3` as optional keyword arguments (default `None`); when `None`, the underlying `TextEncodeQwenImageEditPlus` node receives no image for those slots.
- FR-2: `apply_flux_kontext_multi_reference` must accept a `reference_latents_method` string parameter defaulting to `"index_timestep_zero"`, matching the workflow widget value.
- FR-3: `flux_kontext_image_scale` must delegate entirely to `FluxKontextImageScale.execute()` without performing any custom scaling logic.
- FR-4: `apply_cfg_norm` must clone the input model before patching to avoid mutating a shared model object, consistent with other model-patch functions in the library.
- FR-5: `run()` must be wrapped in `torch.inference_mode()` via the established pattern (owned centrally in `sampling.py`'s `sample()` call — do not add a second `torch.inference_mode()` wrapper in `run()`).
- FR-6: `use_lora=True` default matches the workflow default (LoRA switch defaults to `True` — Lightning 4-step LoRA active).
- FR-7: All new functions must follow the lazy-import pattern: no `torch` or `comfy.*` at module top level.

## Non-Goals (Out of Scope)

- The LoRA inflation variant (`image-qwen_image_edit_2511_lora_inflation.json`) is not part of this iteration.
- No CLI or script entrypoint is added.
- No changes to `comfy_diffusion/__init__.py` re-exports (new symbols remain submodule-only per public API pattern).
- No new optional extras or top-level dependencies are introduced.
- GPU smoke-test script is out of scope; GPU validation is done locally before merging.

## Open Questions

- None
