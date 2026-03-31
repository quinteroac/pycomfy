# Requirement: WAN 2.2 Pipelines — T2V, I2V, FLF2V, S2V, TI2V

## Context

Iteration 000030 completed the WAN 2.1 pipeline family (T2V, I2V, FLF2V).
This iteration delivers the five WAN 2.2 pipelines (`t2v`, `i2v`, `flf2v`, `s2v`, `ti2v`) as
importable Python modules following the existing pipeline contract (`manifest()` / `run()`).

Two new library node wrappers are required first:
- `load_audio_encoder(path)` in `ModelManager` (wraps `AudioEncoderLoader`)
- `audio_encoder_encode(audio_encoder, audio)` in `audio.py` (wraps `AudioEncoderEncode`)

Additionally, three new conditioning/latent helpers are needed:
- `wan_sound_image_to_video(...)` in `conditioning.py` (wraps `WanSoundImageToVideo`)
- `wan_sound_image_to_video_extend(...)` in `conditioning.py` (wraps `WanSoundImageToVideoExtend`)
- `wan22_image_to_video_latent(...)` in `latent.py` (wraps `Wan22ImageToVideoLatent`)

All five pipelines mirror the workflow JSON files in
`comfyui_official_workflows/video/wan/wan2.2/` exactly.

---

## Goals

- Expose `load_audio_encoder()` in `ModelManager` so callers can load an audio encoder without
  importing raw `comfy.*` code.
- Expose `audio_encoder_encode()` in `comfy_diffusion.audio` so callers can encode audio
  tensors into `AUDIO_ENCODER_OUTPUT` objects.
- Expose `wan_sound_image_to_video()` and `wan_sound_image_to_video_extend()` in
  `comfy_diffusion.conditioning` so callers can create S2V-style conditioning/latents.
- Expose `wan22_image_to_video_latent()` in `comfy_diffusion.latent` for TI2V latent creation.
- Implement five WAN 2.2 pipelines under `comfy_diffusion/pipelines/video/wan/wan22/`:
  `t2v`, `i2v`, `flf2v`, `s2v`, `ti2v`.
- Provide one example script per pipeline under `examples/`.
- Wire the new `wan22` sub-package into `comfy_diffusion/pipelines/video/wan/__init__.py`.

---

## User Stories

### US-001: `load_audio_encoder` in ModelManager

**As a** Python developer, **I want** `ModelManager.load_audio_encoder(path)` **so that** I
can load a wav2vec2 (or compatible) audio encoder without calling ComfyUI internals directly.

**Acceptance Criteria:**
- [ ] `ModelManager.load_audio_encoder(path: str | Path) -> Any` is implemented in `models.py`.
- [ ] The method registers `audio_encoders` as a known models directory (analogous to how
      `load_unet` registers `diffusion_models`).
- [ ] Internally it calls `comfy.utils.load_torch_file` and
      `comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd`, raising `RuntimeError`
      with a clear message if the result is `None` (invalid file).
- [ ] The method follows the lazy-import pattern (no `torch` or `comfy.*` at module top level).
- [ ] `load_audio_encoder` is listed in `ModelManager`'s public interface (docstring or type stub).
- [ ] Typecheck / lint passes.

---

### US-002: `audio_encoder_encode` in `audio.py`

**As a** Python developer, **I want** `audio_encoder_encode(audio_encoder, audio)` in
`comfy_diffusion.audio` **so that** I can encode a loaded audio dict into an
`AUDIO_ENCODER_OUTPUT` object suitable for S2V conditioning.

**Acceptance Criteria:**
- [ ] `audio_encoder_encode(audio_encoder: Any, audio: dict) -> Any` is implemented in `audio.py`.
- [ ] The function calls `audio_encoder.encode_audio(audio["waveform"], audio["sample_rate"])`,
      mirroring `AudioEncoderEncode.execute()`.
- [ ] The function follows the lazy-import pattern (no `torch` or `comfy.*` at module top level).
- [ ] `"audio_encoder_encode"` is added to `audio.__all__`.
- [ ] Typecheck / lint passes.

---

### US-003: `wan_sound_image_to_video` and `wan_sound_image_to_video_extend` in `conditioning.py`

**As a** Python developer, **I want** S2V conditioning helpers in `comfy_diffusion.conditioning`
**so that** I can build audio-driven video conditioning without calling raw ComfyUI nodes.

**Acceptance Criteria:**
- [ ] `wan_sound_image_to_video(positive, negative, vae, width, height, length, batch_size, *, audio_encoder_output=None, ref_image=None, control_video=None, ref_motion=None)` is implemented, mirrors `WanSoundImageToVideo.execute()`, returns `(positive, negative, latent)`.
- [ ] `wan_sound_image_to_video_extend(positive, negative, vae, length, video_latent, *, audio_encoder_output=None, ref_image=None, control_video=None)` is implemented, mirrors `WanSoundImageToVideoExtend.execute()`, returns `(positive, negative, latent)`.
- [ ] Both functions follow the lazy-import pattern.
- [ ] Both names are added to `conditioning.__all__`.
- [ ] Typecheck / lint passes.

---

### US-004: `wan22_image_to_video_latent` in `latent.py`

**As a** Python developer, **I want** `wan22_image_to_video_latent(vae, width, height, length, batch_size, *, start_image=None)` in `comfy_diffusion.latent` **so that** I can create TI2V latents for WAN 2.2 without calling ComfyUI nodes directly.

**Acceptance Criteria:**
- [ ] `wan22_image_to_video_latent(vae: Any, width: int, height: int, length: int = 49, batch_size: int = 1, *, start_image: Any = None) -> dict[str, Any]` is implemented in `latent.py`.
- [ ] The returned dict has `"samples"` (shape `[batch, 48, ((length-1)//4)+1, height//16, width//16]`) processed through `comfy.latent_formats.Wan22()`, and `"noise_mask"` when `start_image` is provided — matching `Wan22ImageToVideoLatent.execute()`.
- [ ] The function follows the lazy-import pattern.
- [ ] `"wan22_image_to_video_latent"` is added to `latent.__all__`.
- [ ] Typecheck / lint passes.

---

### US-005: WAN 2.2 text-to-video pipeline (`t2v`)

**As a** Python developer, **I want**
`comfy_diffusion.pipelines.video.wan.wan22.t2v` with `manifest()` and `run()`
**so that** I can generate video from a text prompt using the WAN 2.2 T2V 14B model.

**Acceptance Criteria:**
- [ ] `manifest()` returns exactly 6 `ModelEntry` items matching the active nodes in
      `video_wan2_2_14B_t2v.json`:
      - `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` → `diffusion_models/`
      - `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` → `diffusion_models/`
      - `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` → `loras/`
      - `wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` → `loras/`
      - `umt5_xxl_fp8_e4m3fn_scaled.safetensors` → `text_encoders/`
      - `wan_2.1_vae.safetensors` → `vae/`
- [ ] `run(prompt, negative_prompt, width, height, length, *, models_dir, seed, steps, cfg) -> list[PIL.Image]` executes the dual two-pass `KSamplerAdvanced` flow (high-noise pass first, then low-noise pass) with `LoraLoaderModelOnly` applied to each UNet and `ModelSamplingSD3` shift applied, exactly as ordered in the workflow.
- [ ] BYPASSED nodes are not included in `manifest()` or `run()`.
- [ ] Returns a list of `PIL.Image` frames decoded via `vae_decode`.
- [ ] A unit test verifies `manifest()` length and field names with mocked loaders.
- [ ] An example script `examples/wan22_t2v.py` demonstrates usage.
- [ ] Typecheck / lint passes.

---

### US-006: WAN 2.2 image-to-video pipeline (`i2v`)

**As a** Python developer, **I want**
`comfy_diffusion.pipelines.video.wan.wan22.i2v` with `manifest()` and `run()`
**so that** I can animate an input image using the WAN 2.2 I2V 14B model.

**Acceptance Criteria:**
- [ ] `manifest()` returns exactly 6 `ModelEntry` items matching active nodes in
      `video_wan2_2_14B_i2v.json`:
      - `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` → `diffusion_models/`
      - `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` → `diffusion_models/`
      - `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` → `loras/`
      - `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` → `loras/`
      - `umt5_xxl_fp8_e4m3fn_scaled.safetensors` → `text_encoders/`
      - `wan_2.1_vae.safetensors` → `vae/`
- [ ] `run(image, prompt, negative_prompt, width, height, length, *, models_dir, seed, steps, cfg) -> list[PIL.Image]` executes `WanImageToVideo` conditioning followed by two-pass `KSamplerAdvanced` with `ModelSamplingSD3` and the four `ComfySwitchNode` switches resolved at their default values (`False` = no LoRA branch active), matching the workflow's default configuration.
- [ ] BYPASSED nodes are not included.
- [ ] A unit test verifies `manifest()` length and field names.
- [ ] An example script `examples/wan22_i2v.py` demonstrates usage.
- [ ] Typecheck / lint passes.

---

### US-007: WAN 2.2 first-last-frame-to-video pipeline (`flf2v`)

**As a** Python developer, **I want**
`comfy_diffusion.pipelines.video.wan.wan22.flf2v` with `manifest()` and `run()`
**so that** I can interpolate between two frames using the WAN 2.2 FLF2V model.

**Acceptance Criteria:**
- [ ] `manifest()` returns exactly 3 `ModelEntry` items matching the **active** (non-bypassed)
      nodes in `video_wan2_2_14B_flf2v.json`:
      - `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` → `diffusion_models/`
      - `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` → `diffusion_models/`
      - `wan_2.1_vae.safetensors` → `vae/`
      *(LoRAs and the duplicate CLIPLoader are BYPASSED — excluded from manifest)*
- [ ] The CLIP loader active in the workflow (`umt5_xxl_fp8_e4m3fn_scaled.safetensors`) is
      included in `manifest()`, making the total 4 entries.
- [ ] `run(start_image, end_image, prompt, negative_prompt, width, height, length, *, models_dir, seed, steps, cfg) -> list[PIL.Image]` uses `WanFirstLastFrameToVideo` conditioning and two-pass `KSamplerAdvanced` (high-noise → low-noise), matching the workflow execution order.
- [ ] BYPASSED nodes are not included.
- [ ] A unit test verifies `manifest()` length and field names.
- [ ] An example script `examples/wan22_flf2v.py` demonstrates usage.
- [ ] Typecheck / lint passes.

---

### US-008: WAN 2.2 sound-to-video pipeline (`s2v`)

**As a** Python developer, **I want**
`comfy_diffusion.pipelines.video.wan.wan22.s2v` with `manifest()` and `run()`
**so that** I can generate audio-driven video using the WAN 2.2 S2V 14B model.

**Acceptance Criteria:**
- [ ] `manifest()` returns exactly 5 `ModelEntry` items matching active nodes in
      `video_wan2_2_14B_s2v.json`:
      - `wav2vec2_large_english_fp16.safetensors` → `audio_encoders/`
      - `wan2.2_s2v_14B_fp8_scaled.safetensors` → `diffusion_models/`
      - `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` → `loras/`
      - `umt5_xxl_fp8_e4m3fn_scaled.safetensors` → `text_encoders/`
      - `wan_2.1_vae.safetensors` → `vae/`
- [ ] `run(audio, ref_image, control_video, prompt, negative_prompt, *, models_dir, seed, steps, cfg) -> list[PIL.Image]` calls `load_audio_encoder`, `audio_encoder_encode`, then iterates the multi-pass `WanSoundImageToVideoExtend` + `KSampler` + `LatentConcat` loop as structured in the workflow's subgraph chain.
- [ ] The `audio` parameter accepts a `dict` with keys `"waveform"` (torch tensor) and
      `"sample_rate"` (int), matching ComfyUI's audio dict convention.
- [ ] A unit test verifies `manifest()` length and field names with mocked loaders.
- [ ] An example script `examples/wan22_s2v.py` demonstrates usage.
- [ ] Typecheck / lint passes.

---

### US-009: WAN 2.2 text-and-image-to-video pipeline (`ti2v`)

**As a** Python developer, **I want**
`comfy_diffusion.pipelines.video.wan.wan22.ti2v` with `manifest()` and `run()`
**so that** I can generate video conditioned on both a text prompt and an optional reference
image using the WAN 2.2 TI2V 5B model.

**Acceptance Criteria:**
- [ ] `manifest()` returns exactly 3 `ModelEntry` items matching active nodes in
      `video_wan2_2_5B_ti2v.json`:
      - `wan2.2_ti2v_5B_fp16.safetensors` → `diffusion_models/`
      - `umt5_xxl_fp8_e4m3fn_scaled.safetensors` → `text_encoders/`
      - `wan2.2_vae.safetensors` → `vae/`
- [ ] `run(prompt, negative_prompt, width, height, length, *, start_image=None, models_dir, seed, steps, cfg) -> list[PIL.Image]` calls `wan22_image_to_video_latent`, `ModelSamplingSD3` (shift=8), `KSampler` (uni_pc, 20 steps, cfg=5), and `vae_decode`, exactly matching `video_wan2_2_5B_ti2v.json`.
- [ ] When `start_image` is `None`, an empty latent is used (no mask injected).
- [ ] A unit test verifies `manifest()` length and field names with mocked loaders.
- [ ] An example script `examples/wan22_ti2v.py` demonstrates usage.
- [ ] Typecheck / lint passes.

---

### US-010: Package wiring and example scripts

**As a** Python developer, **I want** the WAN 2.2 pipelines discoverable via the standard
package structure **so that** I can import them with a predictable path.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/pipelines/video/wan/wan22/__init__.py` exists and exports
      `t2v`, `i2v`, `flf2v`, `s2v`, `ti2v` sub-modules.
- [ ] `comfy_diffusion/pipelines/video/wan/__init__.py` is updated to expose `wan22`.
- [ ] Each of `examples/wan22_t2v.py`, `examples/wan22_i2v.py`, `examples/wan22_flf2v.py`,
      `examples/wan22_s2v.py`, `examples/wan22_ti2v.py` is present, runnable, and documents
      its required model files in a leading docstring.
- [ ] Typecheck / lint passes.

---

## Functional Requirements

- **FR-1:** `ModelManager.load_audio_encoder(path: str | Path) -> Any` loads an audio encoder
  from a safetensors file, registering `audio_encoders` in `folder_paths` before loading.
- **FR-2:** `audio_encoder_encode(audio_encoder, audio: dict) -> Any` encodes a ComfyUI audio
  dict and returns an `AUDIO_ENCODER_OUTPUT` object.
- **FR-3:** `wan_sound_image_to_video(...)` wraps `WanSoundImageToVideo.execute()`, returning
  `(positive, negative, latent)` with all optional parameters keyword-only.
- **FR-4:** `wan_sound_image_to_video_extend(...)` wraps `WanSoundImageToVideoExtend.execute()`,
  returning `(positive, negative, latent)`.
- **FR-5:** `wan22_image_to_video_latent(vae, width, height, length, batch_size, *, start_image)` wraps `Wan22ImageToVideoLatent.execute()`, using `comfy.latent_formats.Wan22()` for output processing.
- **FR-6:** All five pipelines follow the `manifest() / run()` contract established in `ltx2_t2v.py`.
- **FR-7:** All pipeline `manifest()` functions must include only models from **active**
  (non-bypassed) nodes in the corresponding workflow JSON.
- **FR-8:** Pipeline `run()` execution order must mirror the workflow node execution order
  exactly (no reordering, no combined steps).
- **FR-9:** All new code follows the lazy-import pattern — no `torch` or `comfy.*` at module
  top level.
- **FR-10:** All tests must pass on CPU-only environments (mock model weights; no GPU needed).
- **FR-11:** `audio_encoders/` directory is registered as a known folder path in `_runtime.py`
  or inside `load_audio_encoder()` using `folder_paths.add_model_folder_path`.

---

## Non-Goals (Out of Scope)

- WAN 2.1 Fun/Control/InP pipelines (separate iteration).
- WAN 2.2 14B Animate pipeline (`video_wan2_2_14B_animate.json`) — separate iteration.
- Any GUI, REST API, or server layer.
- Streaming / incremental frame output.
- Multi-GPU or distributed inference.
- Fine-tuning or training workflows.

---

## Open Questions

- None.
