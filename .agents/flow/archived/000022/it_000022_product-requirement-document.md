# Requirement: Upscale Model Loader

## Context
`comfy_diffusion` already exposes `image_upscale_with_model()` in `image.py` to run
inference with a loaded upscale model (e.g. RealESRGAN), but there is no public API to
load the model itself. The only loader lives as a private helper
(`_load_image_upscale_model`) inside the example script
`examples/simple_checkpoint_esrgan_upscale_example.py`. This iteration promotes that
logic into `ModelManager.load_upscale_model()` — consistent with `load_vae`,
`load_clip`, and `load_unet` — so any Python application or automated pipeline can
load upscale models through the standard library interface.

## Goals
- Expose `ModelManager.load_upscale_model(path)` as a first-class public API method.
- Register the `upscale_models` folder path with ComfyUI's `folder_paths` inside
  `ModelManager.__init__`, matching the convention already in place for checkpoints,
  VAE, CLIP, UNet, and embeddings.
- Remove the private loader helper from the example file and replace it with the
  public API call.
- Document the new method in the `comfy-diffusion-reference` skill so AI agents have
  accurate, up-to-date API signatures.

## User Stories

### US-001: `ModelManager.load_upscale_model(path)` method
**As a** Python developer or automated pipeline, **I want** to call
`manager.load_upscale_model(path)` **so that** I can obtain a spandrel
`ImageModelDescriptor` that is directly usable with `image_upscale_with_model()`.

**Acceptance Criteria:**
- [ ] `ModelManager.load_upscale_model(path: str | Path)` is implemented in
  `comfy_diffusion/models.py`.
- [ ] When `path` is an absolute path to an existing file, that file is loaded
  directly.
- [ ] When `path` is a relative filename, it is resolved against
  `models_dir/upscale_models/` first, then `models_dir/upscale/` as a fallback;
  `FileNotFoundError` is raised listing both candidate paths if neither exists.
- [ ] When `path` is an absolute path that does not exist, `FileNotFoundError` is
  raised immediately.
- [ ] The loader uses `comfy.utils.load_torch_file(..., safe_load=True)` to read the
  state dict, applies the `module.` prefix-strip for HAT/SwinIR models (same guard as
  the example), then passes the state dict to `spandrel.ModelLoader().load_from_state_dict().eval()`.
- [ ] If the loaded descriptor is not a `spandrel.ImageModelDescriptor`, `TypeError`
  is raised with a descriptive message.
- [ ] All ComfyUI and spandrel imports are deferred to function body (lazy import
  pattern); the method is import-safe in CPU-only environments.
- [ ] The returned object can be passed directly to `image_upscale_with_model()` in
  `comfy_diffusion/image.py` without error.
- [ ] `load_upscale_model` is added to `ModelManager`'s `__all__` (or the module
  `__all__`) if one exists.
- [ ] Typecheck / lint passes.

### US-002: `upscale_models` folder registered in `ModelManager.__init__`
**As a** developer, **I want** `folder_paths` to know about
`models_dir/upscale_models/` when a `ModelManager` is instantiated **so that**
relative filenames resolve consistently via ComfyUI's path registry.

**Acceptance Criteria:**
- [ ] `ModelManager.__init__` calls `folder_paths.add_model_folder_path("upscale_models", str(self.models_dir / "upscale_models"), is_default=True)`.
- [ ] The registration follows the same pattern as all other model-type registrations
  already in `__init__` (one `add_model_folder_path` call, `is_default=True`).
- [ ] Typecheck / lint passes.

### US-003: Example file updated to use public API
**As a** developer reading example code, **I want** the example script to call
`manager.load_upscale_model()` **so that** it demonstrates the canonical usage and
does not contain a private reimplementation.

**Acceptance Criteria:**
- [ ] `_load_image_upscale_model()` private helper function is removed from
  `examples/simple_checkpoint_esrgan_upscale_example.py`.
- [ ] A `ModelManager` instance is constructed (or reused from the checkpoint step)
  and `manager.load_upscale_model(args.esrgan_checkpoint.strip())` is called in its
  place.
- [ ] The example produces the same observable behaviour (loads model, runs upscale,
  saves image) as before the change.
- [ ] Typecheck / lint passes.

### US-004: Reference skill updated
**As an** AI agent using the `comfy-diffusion-reference` skill, **I want** the skill
to document `load_upscale_model` **so that** I can use the correct signature without
guessing.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/skills/comfy-diffusion-reference/SKILL.md` is updated to
  include the `load_upscale_model(path: str | Path) -> ImageModelDescriptor`
  signature under the `ModelManager` section.
- [ ] The entry documents the path resolution order
  (`upscale_models/` → `upscale/` fallback) and the `TypeError` raised for
  non-image models.
- [ ] The entry is consistent in style with the existing loader entries (`load_vae`,
  `load_clip`, `load_unet`).

## Functional Requirements
- **FR-1:** `ModelManager.load_upscale_model(path: str | Path) -> Any` — loads a
  spandrel `ImageModelDescriptor` from an ESRGAN/GAN-style upscale model file.
- **FR-2:** Path resolution order for relative filenames:
  `models_dir/upscale_models/<filename>` → `models_dir/upscale/<filename>`.
  Absolute paths are used directly.
- **FR-3:** State dict preprocessing: strip `module.` prefix when
  `module.layers.0.residual_group.blocks.0.norm1.weight` key is present (HAT/SwinIR
  compatibility).
- **FR-4:** Guard: raise `TypeError` if loaded descriptor is not a
  `spandrel.ImageModelDescriptor`.
- **FR-5:** `folder_paths` registration: `upscale_models` key added in
  `ModelManager.__init__` pointing to `models_dir/upscale_models/`.
- **FR-6:** All imports from `comfy.*` and `spandrel` are deferred (lazy import
  pattern — no top-level imports in `models.py`).

## Non-Goals (Out of Scope)
- Latent upscaling (e.g. `NNLatentUpscale`) — separate concern, no model file needed.
- Pixel-based upscaling without a model (e.g. bicubic via `comfy.utils.common_upscale`).
- Batch upscaling or tiled upscaling wrappers.
- Updating `comfy_diffusion/__init__.py` to re-export `load_upscale_model` at package level.
- Any change to `image_upscale_with_model()` in `image.py`.

## Open Questions
- None
