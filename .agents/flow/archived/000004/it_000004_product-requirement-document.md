# Requirement: Sampling Module — KSampler Wrapper

## Context

Iterations 01–03 established the package foundation (`_runtime`), model loading (`models`), and prompt conditioning (`conditioning`). The `sampling` module is the next vertical slice: it gives callers a clean Python function to run a full denoising loop — turning a latent tensor plus conditionings into a denoised latent ready for VAE decoding.

This module will be consumed directly by developers who want granular control over the denoising step, and later internally by `it_07` (`ImagePipeline`) as part of the high-level API.

## Goals

- Expose a single `sample()` function that wraps ComfyUI's KSampler with a clean, typed Python API.
- Delegate 100 % of denoising logic to `comfy.*` internals — no custom sampler implementations.
- Keep `from pycomfy.sampling import sample` CPU-safe (no torch import at module level).
- Allow sampler name, scheduler, CFG scale, step count, and seed to be fully controlled by the caller.

## User Stories

Each story is small enough to implement in one focused session.

### US-001: `sample()` returns a denoised latent

**As a** Python developer building an inference pipeline, **I want** to call `sample(model, positive, negative, latent, steps, cfg, sampler_name, scheduler, seed)` **so that** I receive a denoised latent tensor I can immediately pass to VAE decode.

**Acceptance Criteria:**
- [ ] `pycomfy/sampling.py` is created and exports `sample`.
- [ ] The function signature is: `sample(model, positive, negative, latent, steps, cfg, sampler_name, scheduler, seed, *, denoise: float = 1.0) -> Any`.
- [ ] The return value is the raw denoised latent object produced by `comfy` — no additional wrapping or transformation.
- [ ] The function calls `comfy`'s sampler using the same call pattern as the `KSampler` / `common_ksampler` node in `vendor/ComfyUI/nodes.py` (verified by grepping before implementation).
- [ ] Typecheck / lint passes (`ruff check . && mypy pycomfy/`).

### US-002: Sampler and scheduler are selectable by name

**As a** Python developer, **I want** to pass `sampler_name="euler"` and `scheduler="normal"` as plain strings **so that** I can switch between any ComfyUI-supported sampler/scheduler combination without modifying pycomfy.

**Acceptance Criteria:**
- [ ] `sampler_name` and `scheduler` are plain `str` parameters with no default that silently overrides the caller's choice (defaults are permitted only if they are documented and clearly labelled as such in the docstring).
- [ ] The strings are passed through to `comfy` unchanged — pycomfy does not validate, transform, or reject any sampler or scheduler name.
- [ ] A unit test using mocks verifies that `comfy`'s internal call receives exactly the `sampler_name` and `scheduler` values the caller passed.
- [ ] Typecheck / lint passes.

### US-003: Seed is controllable for reproducibility

**As a** Python developer building production pipelines, **I want** to pass `seed=42` **so that** I get a deterministic result for the same model, conditioning, and latent inputs.

**Acceptance Criteria:**
- [ ] `seed` is an `int` parameter.
- [ ] The value is passed through to `comfy`'s sampler call unchanged — reproducibility is comfy's responsibility.
- [ ] A unit test using mocks verifies that the seed value reaches the underlying comfy call.
- [ ] Typecheck / lint passes.

### US-004: CPU-only import never crashes

**As a** CI pipeline (CPU-only), **I want** `from pycomfy.sampling import sample` to succeed without a GPU, torch, or any model files **so that** the test suite stays green in all environments.

**Acceptance Criteria:**
- [ ] `uv run python -c "from pycomfy.sampling import sample; print('ok')"` exits with code 0 on a CPU-only machine.
- [ ] No `import torch`, no `import comfy.*`, and no ComfyUI path manipulation happens at module import time — these are deferred to call time only.
- [ ] The CI smoke test for this import is added to the pytest suite and passes.
- [ ] Typecheck / lint passes.

## Functional Requirements

- **FR-1:** `pycomfy/sampling.py` must exist and define `sample` as its sole public export (listed in `__all__`). The full signature is `sample(model, positive, negative, latent, steps, cfg, sampler_name, scheduler, seed, *, denoise: float = 1.0) -> Any`; `denoise` defaults to `1.0` (full denoising) and is passed through to comfy unchanged.
- **FR-2:** `sample()` must delegate denoising to `comfy` internals using the call pattern extracted from the `KSampler` / `common_ksampler` node in `vendor/ComfyUI/nodes.py`. The call pattern must be verified by inspection before implementation.
- **FR-3:** No `torch` import, no `comfy.*` import, and no `ensure_comfyui_on_path()` call may appear at module top level — only inside function bodies.
- **FR-4:** `sampler_name` and `scheduler` string values are passed through to comfy as-is; pycomfy does not validate or reject unknown values.
- **FR-5:** `seed` is passed through to comfy as-is; pycomfy does not manage RNG state beyond forwarding the value.
- **FR-6:** All public functions must carry type annotations; the module must pass `mypy pycomfy/` without new errors.
- **FR-7:** `ruff check .` must pass with no new violations.
- **FR-8:** The `pycomfy/__init__.py` public surface is updated to expose `sample` (re-export or documented import path, consistent with how `models` and `conditioning` were handled).

## Non-Goals (Out of Scope)

- **Empty latent creation** (`EmptyLatentImage`) — caller provides the latent; creating blank latents is out of scope for this iteration.
- **img2img / denoising strength** — noise injection and partial denoising deferred to a future iteration.
- **Batch size > 1** — single-image latents only.
- **Per-step progress callbacks** — deferred to `it_08` (async/queue module).
- **Listing available samplers/schedulers** — not required to validate the core contract.
- **Custom sampler implementations** — we wrap comfy, we never reimplement samplers.

## Open Questions

- **`latent` input format — RESOLVED (pending implementer verification):** The caller-facing shape of `latent` is intentionally left unspecified here. The implementer must inspect `vendor/ComfyUI/nodes.py` (`common_ksampler`) before writing code to confirm whether comfy expects a raw tensor or a `{"samples": tensor}` dict. The verified shape must be documented in `sample()`'s docstring.
- **`denoise` parameter — RESOLVED:** `sample()` must include `denoise: float = 1.0` as an optional keyword argument. This mirrors the ComfyUI `KSampler` node signature, keeps the API forward-compatible with future img2img support, and defaults to full denoising (no behaviour change for the MVP use case). FR-1 and US-001 are updated accordingly (see below).
