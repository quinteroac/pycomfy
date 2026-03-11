# Requirement: Model Sampling Patches & Video CFG Guidance

## Context
comfy-diffusion exposes ComfyUI's inference engine as importable Python modules. Several model architectures (Flux, SD3, AuraFlow) require model-level sampling patches to configure shift parameters and continuous noise schedules before sampling. Additionally, video generation workflows need per-frame CFG scaling (linear or triangle ramps) to produce temporally coherent output. Currently, callers must reach into ComfyUI internals to apply these patches. This iteration wraps them as clean public API functions.

## Goals
- Provide `model_sampling_flux`, `model_sampling_sd3`, and `model_sampling_aura_flow` functions in `comfy_diffusion/models.py` that apply model-specific sampling patches and return the patched model.
- Provide `video_linear_cfg_guidance` and `video_triangle_cfg_guidance` functions in `comfy_diffusion/sampling.py` that apply per-frame CFG ramps to a model and return the patched model.
- All functions follow existing conventions: lazy imports, duck-typed where possible, CPU-testable.

## User Stories

### US-001: ModelSamplingFlux
**Target file:** `comfy_diffusion/models.py` (add to existing module — do NOT create a new file)

**As a** Python developer using comfy-diffusion, **I want** to apply Flux shift parameters (max_shift, min_shift) to a loaded model **so that** I can configure the Flux continuous noise schedule without accessing ComfyUI internals.

**Acceptance Criteria:**
- [ ] `model_sampling_flux(model, max_shift, min_shift, width, height)` is callable from `comfy_diffusion.models`
- [ ] Returns a patched model clone with the Flux sampling configuration applied
- [ ] Parameters `max_shift` and `min_shift` control the shift schedule; `width` and `height` determine the latent resolution for shift interpolation
- [ ] Lazy imports: no `comfy.*` or `torch` at module top level
- [ ] Typecheck / lint passes
- [ ] Passing pytest test (CPU-only, mocking ComfyUI internals)

### US-002: ModelSamplingSD3
**Target file:** `comfy_diffusion/models.py` (add to existing module — do NOT create a new file)

**As a** Python developer using comfy-diffusion, **I want** to apply SD3 continuous noise scheduling to a loaded model **so that** I can set the shift parameter for SD3-family models.

**Acceptance Criteria:**
- [ ] `model_sampling_sd3(model, shift)` is callable from `comfy_diffusion.models`
- [ ] Returns a patched model clone with the SD3 sampling configuration applied
- [ ] The `shift` parameter controls the continuous EDM noise schedule shift
- [ ] Lazy imports: no `comfy.*` or `torch` at module top level
- [ ] Typecheck / lint passes
- [ ] Passing pytest test (CPU-only, mocking ComfyUI internals)

### US-003: ModelSamplingAuraFlow
**Target file:** `comfy_diffusion/models.py` (add to existing module — do NOT create a new file)

**As a** Python developer using comfy-diffusion, **I want** to apply AuraFlow continuous noise scheduling to a loaded model **so that** I can set the shift parameter for AuraFlow models.

**Acceptance Criteria:**
- [ ] `model_sampling_aura_flow(model, shift)` is callable from `comfy_diffusion.models`
- [ ] Returns a patched model clone with the AuraFlow sampling configuration applied
- [ ] The `shift` parameter controls the continuous V-prediction noise schedule shift
- [ ] Lazy imports: no `comfy.*` or `torch` at module top level
- [ ] Typecheck / lint passes
- [ ] Passing pytest test (CPU-only, mocking ComfyUI internals)

### US-004: VideoLinearCFGGuidance
**Target file:** `comfy_diffusion/sampling.py` (add to existing module — do NOT create a new file)

**As a** Python developer using comfy-diffusion, **I want** to apply a linear CFG scaling ramp across video frames **so that** earlier frames get full CFG and later frames are scaled down linearly, improving temporal coherence.

**Acceptance Criteria:**
- [ ] `video_linear_cfg_guidance(model, min_cfg)` is callable from `comfy_diffusion.sampling`
- [ ] Returns a patched model clone with the linear CFG guidance callback applied
- [ ] The `min_cfg` parameter sets the minimum CFG value at the last frame; CFG interpolates linearly from full strength to `min_cfg`
- [ ] Lazy imports: no `comfy.*` or `torch` at module top level
- [ ] Typecheck / lint passes
- [ ] Passing pytest test (CPU-only, mocking ComfyUI internals)

### US-005: VideoTriangleCFGGuidance
**Target file:** `comfy_diffusion/sampling.py` (add to existing module — do NOT create a new file)

**As a** Python developer using comfy-diffusion, **I want** to apply a triangle-shaped CFG scaling ramp across video frames **so that** the middle frames get full CFG and the first/last frames are scaled down, improving temporal coherence for looping or bidirectional video.

**Acceptance Criteria:**
- [ ] `video_triangle_cfg_guidance(model, min_cfg)` is callable from `comfy_diffusion.sampling`
- [ ] Returns a patched model clone with the triangle CFG guidance callback applied
- [ ] The `min_cfg` parameter sets the minimum CFG value at the endpoints; CFG peaks at the middle frame
- [ ] Lazy imports: no `comfy.*` or `torch` at module top level
- [ ] Typecheck / lint passes
- [ ] Passing pytest test (CPU-only, mocking ComfyUI internals)

## Functional Requirements
- FR-1: `model_sampling_flux(model, max_shift, min_shift, width, height)` patches the model's sampling object with Flux-specific shift logic and returns the patched model clone.
- FR-2: `model_sampling_sd3(model, shift)` patches the model's sampling object with SD3 continuous EDM noise scheduling and returns the patched model clone.
- FR-3: `model_sampling_aura_flow(model, shift)` patches the model's sampling object with AuraFlow continuous V-prediction noise scheduling and returns the patched model clone.
- FR-4: `video_linear_cfg_guidance(model, min_cfg)` lives in `sampling.py` and patches the model with a `sampler_cfg_function` callback that linearly interpolates CFG from full to `min_cfg` across the batch/frame dimension.
- FR-5: `video_triangle_cfg_guidance(model, min_cfg)` lives in `sampling.py` and patches the model with a `sampler_cfg_function` callback that applies a triangle-shaped CFG ramp peaking at the middle frame.
- FR-6: All functions use lazy imports (no `comfy.*` or `torch` at module top level).
- FR-7: All functions are testable on CPU-only environments with mocked ComfyUI internals.

## Non-Goals (Out of Scope)
- No high-level pipeline or orchestration layer — callers compose these building blocks manually.
- No `ModelSamplingDiscrete`, `ModelSamplingContinuousV`, `RescaleCFG`, or other nice-to-have model patches (deferred to future iterations).
- No re-export from `comfy_diffusion/__init__.py` — callers use explicit submodule imports.
- No GPU-specific tests in CI.

## Open Questions
- None at this time.
