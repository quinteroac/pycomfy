# Requirement: ControlNet Support

## Context
comfy-diffusion currently lacks ControlNet support. Users who need spatially guided generation (pose, depth, canny, etc.) cannot do so through the library. This iteration adds the ability to load ControlNet models and apply them to conditioning, mirroring the ComfyUI nodes `ControlNetLoader`, `DiffControlNetLoader`, `ControlNetApplyAdvanced`, and `SetUnionControlNetType`.

## Goals
- Expose ControlNet loading and application as simple Python functions in a new `comfy_diffusion/controlnet.py` module
- Support both standalone ControlNet loading and model-aware (diff) ControlNet loading
- Allow users to control strength, start percent, and end percent when applying ControlNet to conditioning
- Support union ControlNet type configuration for multi-type union models

## User Stories

### US-001: Load ControlNet from file path
**As a** Python developer, **I want** to load a ControlNet model from a local file path **so that** I can use it for spatially guided generation.

**Acceptance Criteria:**
- [ ] `load_controlnet(path)` loads a ControlNet model and returns a ControlNet object usable by `apply_controlnet`
- [ ] Accepts `str | Path` for the file path (consistent with existing API pattern)
- [ ] Raises or returns a clear error when the file does not exist or is not a valid ControlNet checkpoint
- [ ] Lazy imports — no `torch` or `comfy.*` at module top level
- [ ] Typecheck / lint passes

### US-002: Load diff ControlNet for a specific model
**As a** Python developer, **I want** to load a ControlNet that is paired with a specific diffusion model **so that** the ControlNet is compatible with my base model.

**Acceptance Criteria:**
- [ ] `load_diff_controlnet(model, path)` loads a diff ControlNet, taking the base model and a file path
- [ ] Returns a ControlNet object usable by `apply_controlnet`
- [ ] Accepts `str | Path` for the file path
- [ ] Lazy imports — no `torch` or `comfy.*` at module top level
- [ ] Typecheck / lint passes

### US-003: Apply ControlNet to conditioning
**As a** Python developer, **I want** to apply a loaded ControlNet to my positive/negative conditioning with configurable strength and step range **so that** I can control how strongly and when the ControlNet influences generation.

**Acceptance Criteria:**
- [ ] `apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent, vae=None)` returns updated `(positive, negative)` conditioning
- [ ] `strength` defaults to `1.0`, `start_percent` defaults to `0.0`, `end_percent` defaults to `1.0`
- [ ] `vae` is an optional parameter for pixel-space ControlNets (defaults to `None`)
- [ ] `image` is a torch Tensor (the control image / hint map)
- [ ] Mirrors `ControlNetApplyAdvanced` node behavior (applies to both positive and negative conditioning)
- [ ] Lazy imports — no `torch` or `comfy.*` at module top level
- [ ] Typecheck / lint passes

### US-004: Set union ControlNet type
**As a** Python developer, **I want** to configure a union ControlNet model with a specific control type (e.g. openpose, depth, canny) **so that** I can use multi-type union ControlNet models with the correct mode.

**Acceptance Criteria:**
- [ ] `set_union_controlnet_type(control_net, type)` returns a configured ControlNet object
- [ ] Supported types: `"auto"`, `"openpose"`, `"depth"`, `"hed/pidi/scribble/ted"`, `"canny/lineart/anime_lineart/mlsd"`, `"normal"`, `"segment"`, `"tile"`, `"repaint"` (sourced from `comfy.cldm.control_types.UNION_CONTROLNET_TYPES`)
- [ ] Returns a clear error for unsupported type strings
- [ ] Lazy imports — no `torch` or `comfy.*` at module top level
- [ ] Typecheck / lint passes

## Functional Requirements
- FR-1: All public functions live in `comfy_diffusion/controlnet.py`
- FR-2: All functions use lazy imports (no `torch` or `comfy.*` at module top level)
- FR-3: File path parameters use `str | Path` type annotation
- FR-4: Functions are not auto-imported from `comfy_diffusion/__init__.py` — consumers use `from comfy_diffusion.controlnet import ...`
- FR-5: All tests pass on CPU-only environments
- FR-6: An example script demonstrates a ControlNet-guided generation pipeline end-to-end

## Non-Goals (Out of Scope)
- Image preprocessing (canny edge detection, depth estimation, pose detection) — users provide pre-processed control images
- ControlNet training or fine-tuning
- `ControlNetApply` (deprecated simpler node) — only `ControlNetApplyAdvanced` is wrapped
- `ControlNetInpaintingAliMamaApply` — classified as nice-to-have in ROADMAP
- Auto-importing controlnet functions from the top-level `comfy_diffusion` package

## Open Questions
- None — all resolved.
