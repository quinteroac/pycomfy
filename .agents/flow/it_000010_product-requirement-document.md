# Requirement: Advanced Sampling — KSamplerAdvanced, SamplerCustomAdvanced, Schedulers & Sigma Tools

## Context

`pycomfy` currently exposes only a single sampling function (`sample()`) backed by `common_ksampler`, which
corresponds to ComfyUI's basic `KSampler` node.  Developers building pipelines for modern architectures
(Flux, LTXV, WAN) and for multi-pass / img2img workflows need finer control: the ability to toggle
noise injection, compose guiders, select schedulers independently, and manipulate sigma schedules.  This
iteration adds the full advanced sampling layer to `pycomfy.sampling`.

## Goals

- Expose `sample_advanced()` (maps to `KSamplerAdvanced`) with explicit `add_noise` and
  `return_with_leftover_noise` flags.
- Expose `sample_custom()` (maps to `SamplerCustomAdvanced`) accepting composable guiders, noise
  injectors, and sigma schedules — the entry point needed for Flux, LTXV, and WAN model pipelines.
- Expose scheduler factory functions (`basic_scheduler`, `karras_scheduler`, `ays_scheduler`,
  `flux2_scheduler`, `ltxv_scheduler`) that return sigma tensors ready to be passed into `sample_custom()`.
- Expose sigma utility functions (`split_sigmas`, `split_sigmas_denoise`) and a sampler selector
  (`get_sampler`) as thin wrappers around their ComfyUI equivalents.
- Ensure all new public symbols follow the lazy-import pattern and all tests pass on CPU-only CI.

## User Stories

### US-001: sample_advanced()
**As a** developer composing a multi-pass inference pipeline,
**I want** to call `sample_advanced(model, positive, negative, latent, ...)` with explicit `add_noise`
and `return_with_leftover_noise` boolean flags,
**so that** I can chain denoising passes without injecting noise between them, matching
`KSamplerAdvanced` semantics.

**Acceptance Criteria:**
- [ ] `pycomfy.sampling.sample_advanced(model, positive, negative, latent, steps, cfg, sampler_name, scheduler, noise_seed, *, add_noise=True, return_with_leftover_noise=False, denoise=1.0, start_at_step=0, end_at_step=10000)` exists and is importable.
- [ ] Returns a LATENT dict identical in structure to the output of `sample()`.
- [ ] Passing `add_noise=False` does not inject noise into the latent before sampling.
- [ ] `return_with_leftover_noise=True` returns the latent before the final noise removal step.
- [ ] `sample_advanced` is exported in `pycomfy/sampling.py`'s `__all__`.
- [ ] No `torch`, `comfy.*`, or `ensure_comfyui_on_path()` at module top level — all imports deferred to call time.
- [ ] Typecheck / lint passes.

> **Source (verified):** `KSamplerAdvanced` lives in `nodes.py` (old-style node API, no `io.ComfyNode`). Its `sample()` method signature is:
> ```python
> def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
>            positive, negative, latent_image, start_at_step, end_at_step,
>            return_with_leftover_noise, denoise=1.0)
> ```
> `add_noise` and `return_with_leftover_noise` are `"enable"/"disable"` strings internally. The pycomfy wrapper accepts Python `bool` values and converts them: `True → "enable"`, `False → "disable"`. Internally delegates to `common_ksampler` (same as `sample()`).

---

### US-002: Guider factory functions
**As a** developer building a Flux or LTXV pipeline,
**I want** factory functions `basic_guider(model, conditioning)` and `cfg_guider(model, positive, negative, cfg)`,
**so that** I can construct the guider object required by `sample_custom()` without accessing ComfyUI internals directly.

**Acceptance Criteria:**
- [ ] `pycomfy.sampling.basic_guider(model, conditioning) -> Any` exists and wraps `BasicGuider`.
- [ ] `pycomfy.sampling.cfg_guider(model, positive, negative, cfg) -> Any` exists and wraps `CFGGuider`.
- [ ] Both functions are exported in `__all__`.
- [ ] Lazy import pattern respected.
- [ ] Typecheck / lint passes.

---

### US-003: Noise injector factory functions
**As a** developer composing a custom denoising loop,
**I want** `random_noise(noise_seed)` and `disable_noise()` factory functions,
**so that** I can pass the correct noise injector to `sample_custom()`.

**Acceptance Criteria:**
- [ ] `pycomfy.sampling.random_noise(noise_seed: int) -> Any` exists and wraps `RandomNoise`.
- [ ] `pycomfy.sampling.disable_noise() -> Any` exists and wraps `DisableNoise`.
- [ ] Both exported in `__all__`.
- [ ] Lazy import pattern respected.
- [ ] Typecheck / lint passes.

> **Source:** Both classes are in `comfy_extras/nodes_custom_sampler.py` and use the `io.ComfyNode` + `execute()` classmethod pattern. Wrapper calls `RandomNoise.execute(noise_seed).result[0]` / `DisableNoise.execute().result[0]`.

---

### US-004: Scheduler factory functions
**As a** developer targeting SD, Flux, LTXV, or WAN model architectures,
**I want** scheduler functions that return sigma tensors for each schedule type,
**so that** I can pass them directly to `sample_custom()` without writing ComfyUI boilerplate.

**Acceptance Criteria:**
- [ ] `basic_scheduler(model, scheduler_name: str, steps: int, denoise: float = 1.0) -> Any` wraps `BasicScheduler.execute(model, scheduler, steps, denoise)`.
- [ ] `karras_scheduler(steps: int, sigma_max: float, sigma_min: float, rho: float = 7.0) -> Any` wraps `KarrasScheduler.execute(steps, sigma_max, sigma_min, rho)`. Note: takes no `model` argument.
- [ ] `ays_scheduler(model_type: str, steps: int, denoise: float = 1.0) -> Any` wraps `AlignYourStepsScheduler.execute(model_type, steps, denoise)`. `model_type` must be one of `"SD1"`, `"SDXL"`, or `"SVD"`. Note: takes no `model` object.
- [ ] `flux2_scheduler(steps: int, width: int, height: int) -> Any` wraps `Flux2Scheduler.execute(steps, width, height)`. Note: takes no `model` argument; image dimensions drive the sigma schedule.
- [ ] `ltxv_scheduler(steps: int, max_shift: float, base_shift: float, *, stretch: bool = True, terminal: float = 0.1, latent: Any = None) -> Any` wraps `LTXVScheduler.execute(steps, max_shift, base_shift, stretch, terminal, latent)`. `latent` is optional and used to derive token count.
- [ ] All scheduler functions return a SIGMAS tensor.
- [ ] All exported in `__all__`.
- [ ] Lazy import pattern respected.
- [ ] Typecheck / lint passes.

> **Source (verified):**
> - `BasicScheduler`, `KarrasScheduler`, `SplitSigmas`, `SplitSigmasDenoise`, `KSamplerSelect`, `BasicGuider`, `CFGGuider`, `RandomNoise`, `DisableNoise`, `SamplerCustomAdvanced` → all in `comfy_extras/nodes_custom_sampler.py`, `io.ComfyNode` API, call via `ClassName.execute(...)`.
> - `AlignYourStepsScheduler` → `comfy_extras/nodes_align_your_steps.py`, `io.ComfyNode` API. Accepts `model_type ∈ ["SD1", "SDXL", "SVD"]` — no model object.
> - `Flux2Scheduler` → `comfy_extras/nodes_flux.py`, signature `execute(steps, width, height)` — no model, uses image dims to compute sequence length.
> - `LTXVScheduler` → `comfy_extras/nodes_lt.py`, signature `execute(steps, max_shift, base_shift, stretch, terminal, latent=None)`.

---

### US-005: sample_custom()
**As a** developer needing full control over guiders and sigma schedules (Flux, LTXV, WAN),
**I want** `sample_custom(noise, guider, sampler, sigmas, latent_image)`,
**so that** I can compose advanced denoising workflows using any guider + scheduler combination.

> **Note:** Unlike `sample()` and `sample_advanced()`, this function does NOT take a `model` argument. The model is already embedded inside the `guider` object (constructed via `basic_guider()` or `cfg_guider()`). This is intentional and follows `SamplerCustomAdvanced` semantics.

**Acceptance Criteria:**
- [ ] `pycomfy.sampling.sample_custom(noise, guider, sampler, sigmas, latent_image) -> tuple[Any, Any]` exists and wraps `SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image)`.
- [ ] `noise` is the first positional argument (required); accepts any noise injector returned by `random_noise()` or `disable_noise()`.
- [ ] Accepts any guider returned by `basic_guider()` or `cfg_guider()`.
- [ ] Accepts any sampler returned by `get_sampler()`.
- [ ] Accepts any sigma tensor returned by the scheduler functions.
- [ ] Returns a tuple `(output_latent, denoised_latent)` — both LATENT dicts.
- [ ] `sample_custom` exported in `__all__`.
- [ ] Can be called end-to-end on a CPU dummy latent with `BasicGuider` + `BasicScheduler` without raising.
- [ ] Docstring explicitly states that `model` is embedded in `guider`, not a direct argument.
- [ ] Lazy import pattern respected.
- [ ] Typecheck / lint passes.

> **Source (verified):** `SamplerCustomAdvanced` in `comfy_extras/nodes_custom_sampler.py`. `execute()` aliased as `sample`. Signature: `execute(cls, noise, guider, sampler, sigmas, latent_image)`. Returns `io.NodeOutput(out, out_denoised)` — two LATENT dicts.

---

### US-006: Sigma tools and sampler selector
**As a** developer building multi-pass inference (e.g., hires-fix, img2img),
**I want** `split_sigmas`, `split_sigmas_denoise`, and `get_sampler` utilities,
**so that** I can manipulate sigma schedules and select named samplers without using ComfyUI node internals directly.

**Acceptance Criteria:**
- [ ] `split_sigmas(sigmas, step) -> tuple[Any, Any]` wraps `SplitSigmas` and returns `(sigmas_first, sigmas_second)`.
- [ ] `split_sigmas_denoise(sigmas, denoise) -> tuple[Any, Any]` wraps `SplitSigmasDenoise` and returns `(sigmas_first, sigmas_second)`.
- [ ] `get_sampler(sampler_name: str) -> Any` wraps `KSamplerSelect` and returns a SAMPLER object.
- [ ] All three exported in `__all__`.
- [ ] Lazy import pattern respected.
- [ ] Typecheck / lint passes.

---

## Functional Requirements

- FR-1: All new public functions must live in `pycomfy/sampling.py` and be added to `__all__`.
- FR-2: No `torch`, `comfy.*`, `comfy_extras.*`, or `ensure_comfyui_on_path()` calls at module top level — deferred to function body call time only.
- FR-3: All function signatures use `str | Path` for path arguments (none in this module) and `Any` for ComfyUI opaque objects, with type hints on every public API parameter.
- FR-4: The existing `sample()` function must remain unchanged and all existing tests must continue to pass.
- FR-5: New wrapper functions must call the corresponding ComfyUI node's `execute()` classmethod directly (or `sample()` for old-style nodes). Do not re-implement logic. For `KSamplerAdvanced` (old-style, `nodes.py`), call `.sample()` on an instance. For all `io.ComfyNode` subclasses (`comfy_extras/*`), call `ClassName.execute(...)` as a classmethod.
- FR-6: Import paths for extra schedulers: `AlignYourStepsScheduler` ← `comfy_extras.nodes_align_your_steps`; `Flux2Scheduler` ← `comfy_extras.nodes_flux`; `LTXVScheduler` ← `comfy_extras.nodes_lt`. All paths verified against the pinned `vendor/ComfyUI` submodule.
- FR-7: `sample_custom()` must call `SamplerCustomAdvanced.sample()` with the correct argument order as defined in `comfy_extras/nodes_custom_sampler.py`.
- FR-8: All new symbols must be documented with a one-line docstring describing input/output contract.

## Non-Goals (Out of Scope)

- No `FluxGuidance` conditioning node wrapper (belongs to `it_12` advanced conditioning).
- No `VideoLinearCFGGuidance` or `VideoTriangleCFGGuidance` model patch wrappers (belongs to `it_17`).
- No `DualCFGGuider`, `APG`, `TCFG`, `NAGuidance`, or other nice-to-have guiders.
- No sigma arithmetic nodes (`AddNoise`, `FlipSigmas`, `SetFirstSigma`, etc.).
- No `SamplerCustom` (the older non-advanced variant).
- No high-level pipeline abstraction — callers compose the primitives directly.
- No additional extras (`[video]`, `[audio]`, `[all]`) changes in this iteration.

## Open Questions

*(All open questions from the initial draft have been resolved by inspecting the vendored ComfyUI source.)*

**Resolved — class locations and import paths:**
- `KSamplerAdvanced` → `nodes.py` (old-style API). Import: `import nodes; nodes.KSamplerAdvanced`.
- `BasicScheduler`, `KarrasScheduler`, `SplitSigmas`, `SplitSigmasDenoise`, `KSamplerSelect`, `BasicGuider`, `CFGGuider`, `RandomNoise`, `DisableNoise`, `SamplerCustomAdvanced` → `comfy_extras/nodes_custom_sampler.py` (all `io.ComfyNode`).
- `AlignYourStepsScheduler` → `comfy_extras/nodes_align_your_steps.py`.
- `Flux2Scheduler` → `comfy_extras/nodes_flux.py`.
- `LTXVScheduler` → `comfy_extras/nodes_lt.py`.

**Resolved — AlignYourStepsScheduler `model_type` values:**
- Accepted values: `"SD1"`, `"SDXL"`, `"SVD"` (hardcoded `NOISE_LEVELS` dict in the module). No other values are valid.

**Resolved — `SamplerCustomAdvanced` noise argument contract:**
- `noise` is the **first, required positional argument**: `execute(cls, noise, guider, sampler, sigmas, latent_image)`. There is no optional noise — callers must always pass a noise object explicitly (use `disable_noise()` or `random_noise(seed)`).

**Resolved — `KSamplerAdvanced` `add_noise` / `return_with_leftover_noise` types:**
- Internally the node uses `"enable"/"disable"` strings. The pycomfy wrapper accepts Python `bool` and converts before delegating.

**Resolved — `KSamplerAdvanced` vs `SamplerCustomAdvanced` delegation style:**
- `KSamplerAdvanced` (old-style): instantiate and call `.sample(...)` — it wraps `common_ksampler` exactly like `sample()` does.
- `SamplerCustomAdvanced` (new `io.ComfyNode` style): call `SamplerCustomAdvanced.execute(...)` as classmethod.
