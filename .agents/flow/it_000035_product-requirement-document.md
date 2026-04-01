# Requirement: ACE Step 1.5 Audio Pipelines

## Context

`comfy_diffusion` exposes ComfyUI's inference engine as a library.  Audio
generation is already partially covered — `audio.py` has conditioning
(`encode_ace_step_15_audio`) and empty-latent helpers
(`empty_ace_step_15_latent_audio`) for ACE Step 1.5.  What is missing is a
working end-to-end inference pipeline that a developer can call with a text
prompt and get back a decoded audio waveform.

There are three official ACE Step 1.5 workflows in
`comfyui_official_workflows/audio/ace_step/v1.5/`:

| Workflow file | Model loading strategy |
|---|---|
| `audio_ace_step_1_5_checkpoint.json` | Single all-in-one checkpoint (`CheckpointLoaderSimple`) |
| `audio_ace_step_1_5_split.json` | Separate UNet + DualCLIP (0.6B + 1.7B) + VAE |
| `audio_ace_step_1_5_split_4b.json` | Separate UNet + DualCLIP (0.6B + 4B) + VAE |

This iteration implements all three as canonical pipelines under
`comfy_diffusion/pipelines/audio/ace_step/v1_5/`, exposes the missing
`vae_decode_audio()` primitive in `audio.py`, and creates the `audio`
sub-package of `comfy_diffusion/pipelines/`.

---

## Goals

- Expose `vae_decode_audio()` in `comfy_diffusion/audio.py` — the final
  missing building block for ACE Step decoding.
- Provide three pipeline modules (`checkpoint.py`, `split.py`, `split_4b.py`)
  each exporting `manifest() -> list[ModelEntry]` and
  `run(...) -> dict[str, Any]`.
- Allow developers to generate music from a text prompt with a single function
  call, using their preferred model variant.
- All tests pass on CPU (mocked weights).

---

## User Stories

### US-001: Add `vae_decode_audio` primitive

**As a** developer using comfy-diffusion, **I want** a
`vae_decode_audio(vae, latent)` function in `comfy_diffusion/audio.py` **so
that** I can decode ACE Step (and any compatible audio) latents into a
waveform dict without calling ComfyUI node internals directly.

**Acceptance Criteria:**
- [ ] `vae_decode_audio(vae, latent)` added to `comfy_diffusion/audio.py`.
- [ ] Implementation calls `vae.decode(latent["samples"]).movedim(-1, 1)`,
  applies the amplitude normalization (`std * 5.0`, floor 1.0), and returns
  `{"waveform": audio, "sample_rate": int}` — mirroring ComfyUI's
  `nodes_audio.vae_decode_audio`.
- [ ] Function added to `audio.py`'s `__all__` list.
- [ ] `vae_decode_audio` is a lazy-import function (no top-level `comfy.*`
  imports; all ComfyUI access deferred to call time or handled via duck typing
  like the existing `audio.py` pattern).
- [ ] Typecheck / lint passes.

---

### US-002: Checkpoint pipeline (`ace_step_1_5_t2a_checkpoint`)

**As a** developer, **I want**
`comfy_diffusion.pipelines.audio.ace_step.v1_5.checkpoint.run(...)` **so that**
I can generate audio using the single all-in-one checkpoint
(`ace_step_1.5_turbo_aio.safetensors`) without managing separate model files.

**Acceptance Criteria:**
- [ ] File created at
  `comfy_diffusion/pipelines/audio/ace_step/v1_5/checkpoint.py`.
- [ ] `manifest()` returns exactly one `HFModelEntry` for
  `ace_step_1.5_turbo_aio.safetensors` pointing to
  `Comfy-Org/ace_step_1.5_ComfyUI_files` / `checkpoints/ace_step_1.5_turbo_aio.safetensors`
  → destination `checkpoints/ace_step_1.5_turbo_aio.safetensors`.
- [ ] `run(...)` signature exposes at minimum: `models_dir`, `tags` (positive
  prompt / style description), `lyrics`, `seed`, `bpm`, `duration`,
  `time_signature`, `language`, `key_scale`, `steps`, `cfg`.
- [ ] Node execution order mirrors the workflow exactly:
  1. `ModelManager.load_checkpoint(path)` → model, clip, vae
  2. `model_sampling_aura_flow(model, shift=3)`
  3. `encode_ace_step_15_audio(clip, tags, lyrics, ...)` → positive
  4. `conditioning_zero_out(positive)` → negative
  5. `empty_ace_step_15_latent_audio(duration)` → latent
  6. `sample(model, positive, negative, latent, steps=8, cfg=1, sampler="euler", scheduler="simple", seed=seed)` (KSampler)
  7. `vae_decode_audio(vae, denoised_latent)` → audio dict
- [ ] `run(...)` returns `{"audio": {"waveform": tensor, "sample_rate": int}}`.
- [ ] Default sampler parameters match the workflow: `steps=8`, `cfg=1.0`,
  `sampler="euler"`, `scheduler="simple"`.
- [ ] Typecheck / lint passes.

---

### US-003: Split pipeline — 1.7B text encoder (`ace_step_1_5_t2a_split`)

**As a** developer, **I want**
`comfy_diffusion.pipelines.audio.ace_step.v1_5.split.run(...)` **so that** I
can use the memory-efficient split variant with separate UNet, 0.6B + 1.7B
text encoders, and VAE.

**Acceptance Criteria:**
- [ ] File created at
  `comfy_diffusion/pipelines/audio/ace_step/v1_5/split.py`.
- [ ] `manifest()` returns four `HFModelEntry` items from
  `Comfy-Org/ace_step_1.5_ComfyUI_files`:
  - `split_files/diffusion_models/acestep_v1.5_turbo.safetensors` → `diffusion_models/`
  - `split_files/text_encoders/qwen_0.6b_ace15.safetensors` → `text_encoders/`
  - `split_files/text_encoders/qwen_1.7b_ace15.safetensors` → `text_encoders/`
  - `split_files/vae/ace_1.5_vae.safetensors` → `vae/`
- [ ] `run(...)` loads models with:
  - `mm.load_unet(unet_path)`
  - `mm.load_clip("qwen_0.6b_ace15.safetensors", "qwen_1.7b_ace15.safetensors", clip_type="ace")`
  - `mm.load_vae(vae_path)`
- [ ] Sampling execution follows the same order as US-002 (steps 2–7).
- [ ] `run(...)` returns `{"audio": {"waveform": tensor, "sample_rate": int}}`.
- [ ] Typecheck / lint passes.

---

### US-004: Split pipeline — 4B text encoder (`ace_step_1_5_t2a_split_4b`)

**As a** developer, **I want**
`comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b.run(...)` **so that**
I can use the higher-capacity 4B text encoder variant for richer conditioning.

**Acceptance Criteria:**
- [ ] File created at
  `comfy_diffusion/pipelines/audio/ace_step/v1_5/split_4b.py`.
- [ ] `manifest()` returns four `HFModelEntry` items from
  `Comfy-Org/ace_step_1.5_ComfyUI_files`:
  - `split_files/diffusion_models/acestep_v1.5_turbo.safetensors` → `diffusion_models/`
  - `split_files/text_encoders/qwen_0.6b_ace15.safetensors` → `text_encoders/`
  - `split_files/text_encoders/qwen_4b_ace15.safetensors` → `text_encoders/`
  - `split_files/vae/ace_1.5_vae.safetensors` → `vae/`
- [ ] `run(...)` loads models with:
  - `mm.load_unet(unet_path)`
  - `mm.load_clip("qwen_0.6b_ace15.safetensors", "qwen_4b_ace15.safetensors", clip_type="ace")`
  - `mm.load_vae(vae_path)`
- [ ] Sampling execution follows the same order as US-002 (steps 2–7).
- [ ] `run(...)` returns `{"audio": {"waveform": tensor, "sample_rate": int}}`.
- [ ] Typecheck / lint passes.

---

### US-005: Package scaffolding for audio pipelines

**As a** developer, **I want** the `comfy_diffusion.pipelines.audio` sub-package
to exist and be importable **so that** imports like
`from comfy_diffusion.pipelines.audio.ace_step.v1_5.checkpoint import run`
work out of the box.

**Acceptance Criteria:**
- [ ] `__init__.py` files created for:
  - `comfy_diffusion/pipelines/audio/`
  - `comfy_diffusion/pipelines/audio/ace_step/`
  - `comfy_diffusion/pipelines/audio/ace_step/v1_5/`
- [ ] `comfy_diffusion/pipelines/__init__.py` updated to list `"audio"` in
  its `__all__` and docstring.
- [ ] Typecheck / lint passes.

---

### US-006: CPU-passing tests for all three pipelines

**As a** developer, **I want** pytest tests that validate the pipeline
structure (manifest shape, run signature, return type) on CPU without loading
real model weights **so that** CI passes with no GPU.

**Acceptance Criteria:**
- [ ] Test file `tests/test_ace_step_v1_5_pipelines.py` created.
- [ ] Tests verify `manifest()` returns a non-empty list of `ModelEntry` items
  for all three pipeline modules.
- [ ] Tests verify `manifest()` model filenames and destination paths match the
  workflow (checkpoint, split, split_4b).
- [ ] Tests for `run()` mock model loading and sampling; assert return dict
  contains key `"audio"` with sub-keys `"waveform"` and `"sample_rate"`.
- [ ] `uv run pytest tests/test_ace_step_v1_5_pipelines.py` passes on CPU.
- [ ] Typecheck / lint passes.

---

## Functional Requirements

- **FR-1:** `vae_decode_audio(vae, latent)` in `audio.py` applies the amplitude
  normalization (`std * 5.0`, min 1.0) consistent with ComfyUI's
  `nodes_audio.vae_decode_audio`.
- **FR-2:** All three pipeline modules export exactly `manifest()` and `run()`
  as their public API.
- **FR-3:** `manifest()` in each pipeline module is the single source of truth
  for model file paths; `run()` derives default paths from it.
- **FR-4:** Default sampler settings match the workflows: `steps=8`, `cfg=1.0`,
  `sampler_name="euler"`, `scheduler="simple"`, `denoise=1.0`.
- **FR-5:** `ModelSamplingAuraFlow` shift is `3` for all ACE Step 1.5 pipelines
  (as read from the workflow `widgets_values`).
- **FR-6:** All imports of `comfy.*` and `torch` inside pipeline `run()`
  functions are lazy (deferred to call time), consistent with the project's
  lazy-import convention.
- **FR-7:** No `torch` or `comfy.*` imports at the module top level in any new
  file.
- **FR-8:** `path` type annotations use `str | Path` (not `str | os.PathLike`).

---

## Non-Goals (Out of Scope)

- Audio-to-audio or continuation / inpainting pipelines.
- `SaveAudioMP3` wrapper — consumers use `torchaudio.save()` or similar.
- Streaming / chunked audio generation.
- Chatterbox, Stable Audio, or any other audio model family.
- Any UI or server-side endpoint.
- `vae_decode_audio_tiled` variant (can be added in a future iteration).
- Modifying `comfy_diffusion/__init__.py` to re-export audio pipeline symbols.

---

## Open Questions

- None.
