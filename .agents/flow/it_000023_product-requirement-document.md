# Requirement: LTX-2 Full Pipeline (T2V / I2V / T2SV)

## Context
`comfy_diffusion` already has partial LTXV support: `ltxv_img_to_video`, `ltxv_conditioning`,
`ltxv_preprocess`, `ltxv_scheduler`, and the LTXV audio VAE stack. However, the building blocks
needed for a complete LTX-2 Text-to-Video, Image-to-Video, and Text-to-Sound-to-Video pipeline
are missing:

- Empty video latent creation (`EmptyLTXVLatentVideo`)
- Audio/video latent merging and splitting (`LTXVConcatAVLatent`, `LTXVSeparateAVLatent`)
- Crop-guide conditioning (`LTXVCropGuides`)
- Latent-space upsampling (`LTXVLatentUpsampler`)
- Latent upscale model loader (`LatentUpscaleModelLoader`)
- Manual sigma schedule definition (`ManualSigmas`)

This iteration adds all seven missing public API functions, a CPU-safe unit test suite for
each, and an end-to-end LTX-2 T2SV (text-to-sound-to-video) example script.

## Goals
- Add `ltxv_empty_latent_video` to `latent.py` — create empty LTXV video latents.
- Add `ltxv_concat_av_latent` and `ltxv_separate_av_latent` to `audio.py` — merge/split AV latents.
- Add `ltxv_crop_guides` to `conditioning.py` — apply LTXV keyframe crop guides.
- Add `ltxv_latent_upsample` to `latent.py` — upsample a video latent with a latent upscale model.
- Add `ModelManager.load_latent_upscale_model(path)` to `models.py` — load `LATENT_UPSCALE_MODEL`.
- Add `manual_sigmas` to `sampling.py` — build a sigma tensor from a comma-separated string.
- Deliver `examples/ltxv2_t2sv_example.py` demonstrating the full LTX-2 T2SV pipeline.
- Provide a CPU-safe pytest suite covering all seven new functions.

## User Stories

### US-001: `ltxv_empty_latent_video` in `latent.py`
**As a** Python developer building an LTX-2 pipeline, **I want** to call
`ltxv_empty_latent_video(width, height, length, batch_size)` **so that** I get an empty
LTXV video latent with the correct shape for the LTXV model.

**Acceptance Criteria:**
- [ ] `ltxv_empty_latent_video(width: int, height: int, length: int = 97, batch_size: int = 1) -> dict[str, Any]`
  is implemented in `comfy_diffusion/latent.py`.
- [ ] Returns a dict `{"samples": tensor}` where tensor shape is
  `[batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32]`, created on
  `comfy.model_management.intermediate_device()`.
- [ ] All `comfy.*` and `torch` imports are deferred to the function body (lazy import pattern).
- [ ] `ltxv_empty_latent_video` is added to `latent.py`'s `__all__`.
- [ ] Typecheck / lint passes.

### US-002: `ltxv_concat_av_latent` in `audio.py`
**As a** Python developer composing an LTX-2 audio-visual pipeline, **I want** to call
`ltxv_concat_av_latent(video_latent, audio_latent)` **so that** I get a single combined
latent dict that the sampler can denoise jointly.

**Acceptance Criteria:**
- [ ] `ltxv_concat_av_latent(video_latent: dict[str, Any], audio_latent: dict[str, Any]) -> dict[str, Any]`
  is implemented in `comfy_diffusion/audio.py`.
- [ ] Output `samples` is a `comfy.nested_tensor.NestedTensor` wrapping `(video_samples, audio_samples)`.
- [ ] When either input has a `noise_mask`, both masks are filled with ones-tensors of matching
  shape if absent, and combined into a `NestedTensor` under `"noise_mask"`.
- [ ] All `comfy.*` and `torch` imports are deferred to the function body.
- [ ] `ltxv_concat_av_latent` is added to `audio.py`'s `__all__`.
- [ ] Typecheck / lint passes.

### US-003: `ltxv_separate_av_latent` in `audio.py`
**As a** Python developer post-processing LTX-2 output, **I want** to call
`ltxv_separate_av_latent(av_latent)` **so that** I can obtain the video and audio latents
separately for independent decoding.

**Acceptance Criteria:**
- [ ] `ltxv_separate_av_latent(av_latent: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]`
  is implemented in `comfy_diffusion/audio.py`.
- [ ] Returns `(video_latent, audio_latent)` dicts produced by unbinding the `NestedTensor` samples.
- [ ] When `"noise_mask"` is present in `av_latent`, it is also unbound and assigned to the
  respective output dicts.
- [ ] All `comfy.*` imports deferred; function is import-safe in CPU-only environments.
- [ ] `ltxv_separate_av_latent` is added to `audio.py`'s `__all__`.
- [ ] Typecheck / lint passes.

### US-004: `ltxv_crop_guides` in `conditioning.py`
**As a** Python developer running an LTX-2 pipeline with keyframe guides, **I want** to call
`ltxv_crop_guides(positive, negative, latent)` **so that** keyframe conditioning entries and
their corresponding latent frames are cropped out before the main sampling pass.

**Acceptance Criteria:**
- [ ] `ltxv_crop_guides(positive: Any, negative: Any, latent: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]`
  is implemented in `comfy_diffusion/conditioning.py`.
- [ ] Returns `(positive, negative, latent)` where conditioning has `keyframe_idxs` and
  `guide_attention_entries` cleared and `latent["samples"]` / `latent["noise_mask"]` are
  trimmed by `num_keyframes` frames along the time axis when keyframes are present.
- [ ] When `num_keyframes == 0`, the inputs are returned unchanged.
- [ ] All `comfy.*` imports deferred; function is import-safe in CPU-only environments.
- [ ] `ltxv_crop_guides` is added to `conditioning.py`'s `__all__`.
- [ ] Typecheck / lint passes.

### US-005: `ltxv_latent_upsample` in `latent.py`
**As a** Python developer running a multi-stage LTX-2 pipeline, **I want** to call
`ltxv_latent_upsample(samples, upscale_model, vae)` **so that** I can upsample a video latent
by factor 2 in the latent space without decoding to pixel space.

**Acceptance Criteria:**
- [ ] `ltxv_latent_upsample(samples: dict[str, Any], upscale_model: Any, vae: Any) -> dict[str, Any]`
  is implemented in `comfy_diffusion/latent.py`.
- [ ] Implementation wraps `LTXVLatentUpsampler.upsample_latent` from
  `comfy_extras.nodes_lt_upsampler` (or re-implements the same logic directly with deferred imports).
- [ ] Memory management follows the same pattern as the node: `free_memory` before moving
  model to device, `upscale_model.cpu()` in a `finally` block.
- [ ] `noise_mask` is removed from the returned dict (matching node behaviour).
- [ ] All `comfy.*` and `torch` imports deferred.
- [ ] `ltxv_latent_upsample` is added to `latent.py`'s `__all__`.
- [ ] The private `_load_latent_upscale_model` and `_upsample_latent` helpers in
  `examples/simple_checkpoint_latent_upscale_example.py` are replaced with calls to the
  new public API (`manager.load_latent_upscale_model(...)` and `ltxv_latent_upsample(...)`).
- [ ] Typecheck / lint passes.

### US-006: `ModelManager.load_latent_upscale_model` in `models.py`
**As a** Python developer, **I want** to call `manager.load_latent_upscale_model(path)` **so
that** I get a `LATENT_UPSCALE_MODEL` object compatible with `ltxv_latent_upsample`.

**Acceptance Criteria:**
- [ ] `ModelManager.load_latent_upscale_model(path: str | Path) -> Any` is implemented in
  `comfy_diffusion/models.py`.
- [ ] When `path` is an absolute path to an existing file, that file is loaded directly.
- [ ] When `path` is a relative filename, it is resolved against `models_dir/upscale/` first;
  `FileNotFoundError` is raised with a descriptive message listing the candidate path if
  the file is not found.
- [ ] When `path` is an absolute path that does not exist, `FileNotFoundError` is raised immediately.
- [ ] Loading uses `comfy.utils.load_torch_file(..., safe_load=True, return_metadata=True)` and
  passes `(sd, metadata)` through the same model-selection logic as `LatentUpscaleModelLoader`
  in `comfy_extras.nodes_hunyuan` (keys `"blocks.0.block.0.conv.weight"`,
  `"up.0.block.0.conv1.conv.weight"`, `"post_upsample_res_blocks.0.conv2.bias"`).
- [ ] `folder_paths.add_model_folder_path("latent_upscale_models", str(self.models_dir / "upscale"), is_default=True)`
  is called in `ModelManager.__init__` alongside the existing registrations.
- [ ] All `comfy.*` imports are deferred to the function body (lazy import pattern).
- [ ] `load_latent_upscale_model` is added to `models.py`'s `__all__`.
- [ ] Typecheck / lint passes.

### US-007: `manual_sigmas` in `sampling.py`
**As a** Python developer defining a custom noise schedule, **I want** to call
`manual_sigmas(sigmas_string)` **so that** I get a float tensor of sigma values I can pass
directly to `SamplerCustomAdvanced`.

**Acceptance Criteria:**
- [ ] `manual_sigmas(sigmas: str) -> Any` is implemented in `comfy_diffusion/sampling.py`.
- [ ] Parses all numeric tokens (including negative/decimal values) from `sigmas` using the
  same `re.findall(r"[-+]?(?:\d*\.*\d+)", ...)` pattern as `ManualSigmas.execute`.
- [ ] Returns a `torch.FloatTensor` of the parsed values.
- [ ] All `torch` imports are deferred to the function body.
- [ ] `manual_sigmas` is added to `sampling.py`'s `__all__`.
- [ ] Typecheck / lint passes.

### US-008: CPU-safe unit tests
**As a** developer merging this iteration, **I want** a pytest test file covering all 7 new
functions **so that** CI can verify correctness without a GPU.

**Acceptance Criteria:**
- [ ] `tests/test_ltxv2.py` (or equivalent) contains at least one test per function:
  `ltxv_empty_latent_video`, `ltxv_concat_av_latent`, `ltxv_separate_av_latent`,
  `ltxv_crop_guides`, `ltxv_latent_upsample`, `ModelManager.load_latent_upscale_model`,
  `manual_sigmas`.
- [ ] All tests run with `uv run pytest` on CPU (no GPU required); ComfyUI internals are
  stubbed/mocked where needed so no model weights are loaded.
- [ ] Tests do not import `torch`, `comfy.*`, or `comfy_diffusion.*` at module top level
  (deferred inside test functions or fixtures) — consistent with the lazy import pattern.
- [ ] All new tests pass; existing test suite continues to pass.
- [ ] Typecheck / lint passes.

### US-009: `examples/ltxv2_t2sv_example.py`
**As a** developer wanting to run an end-to-end LTX-2 Text-to-Sound-to-Video pipeline, **I
want** an example script demonstrating all new functions wired together **so that** I can
understand the canonical usage and run a real inference pass.

**Acceptance Criteria:**
- [ ] `examples/ltxv2_t2sv_example.py` is created following the same structure and style as
  existing examples (argparse CLI, `PYCOMFY_*` env-var defaults, `check_runtime()` first).
- [ ] The script exercises the following pipeline steps in order:
  1. `check_runtime()`
  2. `ModelManager` + `load_checkpoint` (LTXV UNet/CLIP/VAE)
  3. `ModelManager.load_latent_upscale_model(...)` for the LTXV latent upscaler
  4. `ltxv_empty_latent_audio(...)` + `ltxv_audio_vae_encode(...)` (audio latent creation)
  5. `ltxv_empty_latent_video(...)` (video latent creation)
  6. `ltxv_concat_av_latent(...)` (merge AV latents)
  7. Prompt encoding + `ltxv_conditioning(...)` + optional `ltxv_crop_guides(...)`
  8. `manual_sigmas(...)` or `ltxv_scheduler(...)` + `sample_custom_advanced(...)`
  9. `ltxv_separate_av_latent(...)` (split back)
  10. `ltxv_latent_upsample(...)` (optional upscale pass)
  11. `vae_decode_batch(...)` + save frames
- [ ] All CLI args include `--models-dir` / `PYCOMFY_MODELS_DIR`, `--checkpoint` /
  `PYCOMFY_CHECKPOINT`, `--latent-upscale-checkpoint` / `PYCOMFY_LATENT_UPSCALE_CHECKPOINT`,
  `--prompt`, `--output-dir`.
- [ ] Script is written entirely in English (comments, docstrings, error messages).
- [ ] Typecheck / lint passes.

## Functional Requirements
- **FR-1:** `ltxv_empty_latent_video(width, height, length, batch_size) -> dict` — creates LTXV
  video latent with shape `[B, 128, ((L-1)//8)+1, H//32, W//32]`.
- **FR-2:** `ltxv_concat_av_latent(video_latent, audio_latent) -> dict` — wraps video + audio
  samples in a `NestedTensor`; handles optional noise masks.
- **FR-3:** `ltxv_separate_av_latent(av_latent) -> tuple[dict, dict]` — unbinds `NestedTensor`
  samples and noise masks back into separate video/audio latent dicts.
- **FR-4:** `ltxv_crop_guides(positive, negative, latent) -> tuple` — strips keyframe metadata
  from conditioning and trims latent time dimension by `num_keyframes`; no-op when keyframes = 0.
- **FR-5:** `ltxv_latent_upsample(samples, upscale_model, vae) -> dict` — runs the latent
  through `upscale_model`, managing device placement and VAE channel statistics normalisation.
- **FR-6:** `ModelManager.load_latent_upscale_model(path) -> Any` — loads a `LATENT_UPSCALE_MODEL`
  via `comfy.utils.load_torch_file` + model-type detection logic matching `LatentUpscaleModelLoader`.
- **FR-7:** `manual_sigmas(sigmas: str) -> torch.FloatTensor` — parses comma/space-separated
  float values into a sigma tensor.
- **FR-8:** `folder_paths` registration: `"latent_upscale_models"` key pointing to
  `models_dir/upscale/` added in `ModelManager.__init__`.
- **FR-9:** All seven new public functions follow the lazy import pattern (no `comfy.*` or
  `torch` at module top level).

## Non-Goals (Out of Scope)
- `LTXVAddGuide` / `LTXVImgToVideoInplace` — not in the roadmap item.
- `ModelSamplingLTXV` patch — already covered or handled by existing `model_sampling_flux`.
- Tiled latent upsampling.
- Updating the `comfy-diffusion-reference` skill (deferred to a documentation iteration).
- Any change to `vae.py`, `image.py`, `mask.py`, or `controlnet.py`.
- Publishing / releasing a new package version.

## Open Questions
- None
