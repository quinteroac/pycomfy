# Requirement: Phase 5 — Remaining LTX Pipelines + Examples

## Context

Phase 4 (it_000027) exposed all secondary utility and ControlNet node wrappers required
by the remaining LTX workflows. Phase 5 delivers the four pipelines that were blocked by
those wrappers:

- **LTX2 ControlNet pipelines:** `canny_to_video`, `depth_to_video`, `pose_to_video`
- **LTX23 audio-conditioned pipeline:** `ia2v` (image + audio → video)

Three additional thin node wrappers are also missing and must be implemented before the
pipelines can be written:

| Missing wrapper | Node | Required by |
|---|---|---|
| `image.image_scale_by` | `ImageScaleBy` | canny pipeline |
| `image.dw_preprocessor` | `DWPreprocessor` | pose pipeline |
| `audio.load_audio` | `LoadAudio` (torchaudio-backed) | ia2v pipeline |

Each pipeline follows the canonical pattern: exports `manifest() -> list[ModelEntry]` and
`run(...) -> list[PIL.Image.Image]`, faithfully mirrors its official ComfyUI workflow JSON
in `comfyui_official_workflows/video/ltx/`, and is accompanied by a runnable example
script under `examples/video/ltx/`.

## Goals

- Expose the three missing node wrappers (`image_scale_by`, `dw_preprocessor`,
  `load_audio`) following the lazy-import pattern.
- Implement all four Phase 5 pipelines as faithful translations of their official
  workflows.
- Provide a runnable example script for each pipeline.
- Maintain CPU-only test coverage so CI remains green.
- Update `comfy_diffusion/pipelines/video/ltx/ltx23/__init__.py` to export `ia2v`.

## User Stories

### US-001: Scale an image by a factor (ImageScaleBy)

**As a** developer composing an LTX2 ControlNet pipeline, **I want** an
`image_scale_by(image, upscale_method, scale_by)` function **so that** I can resize
an image tensor by a float scale factor without specifying absolute dimensions.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.image.image_scale_by(image, upscale_method="lanczos", scale_by=0.5)` is callable and returns an IMAGE tensor.
- [ ] Result dimensions equal `floor(input_h * scale_by)` × `floor(input_w * scale_by)`.
- [ ] Function is added to `image.__all__`.
- [ ] Lazy-import pattern followed (no top-level `comfy.*` / `torch` imports).
- [ ] Typecheck / lint passes.

---

### US-002: Estimate pose from image (DWPreprocessor)

**As a** developer composing an LTX2 pose-to-video pipeline, **I want** a
`dw_preprocessor(image, detect_hand, detect_body, detect_face, resolution)` function
**so that** I can generate a pose map from an input image for ControlNet conditioning.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.image.dw_preprocessor(image, detect_hand=True, detect_body=True, detect_face=True, resolution=512)` is callable and returns an IMAGE tensor with the same batch dimension as the input.
- [ ] Function is added to `image.__all__`.
- [ ] Lazy-import pattern followed.
- [ ] Typecheck / lint passes.

---

### US-003: Load audio from disk (LoadAudio)

**As a** developer composing an LTX23 ia2v pipeline, **I want** a
`load_audio(path, start_time, duration)` function **so that** I can load an audio file
into the ComfyUI-compatible `AUDIO` dict format required by `ltxv_audio_vae_encode`.

**Acceptance Criteria:**
- [ ] `comfy_diffusion.audio.load_audio(path)` is callable with an optional `start_time: float = 0.0` and `duration: float | None = None`, and returns a `dict` with keys `"waveform"` (torch.Tensor, shape `[1, C, N]`) and `"sample_rate"` (int).
- [ ] Implementation uses `torchaudio` (consistent with the architecture rule: external libs over node ports for audio I/O).
- [ ] Function is added to `audio.__all__`.
- [ ] Typecheck / lint passes.

---

### US-004: LTX2 Canny-to-Video pipeline

**As a** developer, **I want** `comfy_diffusion.pipelines.video.ltx.ltx2.canny` with
`manifest()` and `run(...)` **so that** I can generate a video from a text prompt and a
reference video using Canny edge-detection ControlNet conditioning (LTX 2.0 dev fp8).

**Acceptance Criteria:**
- [ ] `manifest()` returns a list of `ModelEntry` objects matching exactly the
  **non-bypassed** model downloads declared in
  `comfyui_official_workflows/video/ltx/ltx2/video_ltx2_canny_to_video.json`:
  `ltx-2-19b-dev-fp8.safetensors`, `ltx-2-spatial-upscaler-x2-1.0.safetensors`,
  `ltx-2-19b-ic-lora-canny-control.safetensors`, `gemma_3_12B_it_fp4_mixed.safetensors`.
  (Distilled variant entries are excluded from the dev fp8 path.)
- [ ] `run(models_dir, prompt, video_path, ...)` executes end-to-end following the
  workflow node order: `GetVideoComponents` → `ImageFromBatch` → `ResizeImageMaskNode`
  → `ImageScaleBy(0.5)` → `Canny` → `LTXVAddGuide` → `LTXVImgToVideoInplace` →
  two-pass sampling (`CFGGuider` + `SamplerCustomAdvanced`) → `LTXVLatentUpsampler`
  → VAE decode → returns `list[PIL.Image.Image]`.
- [ ] Sampler, sigmas, and CFG values match the workflow (`euler_ancestral`, manual
  sigmas `1., 0.99375, ..., 0.0` for pass 1; `0.909375, ..., 0.0` for pass 2; CFG 3).
- [ ] `run()` accepts keyword arguments: `prompt`, `negative_prompt`, `video_path`,
  `first_frame_path`, `seed`, `width`, `height`, `num_frames`, `lora_strength`.
- [ ] Pipeline file is at `comfy_diffusion/pipelines/video/ltx/ltx2/canny.py`.
- [ ] Typecheck / lint passes.

---

### US-005: LTX2 Depth-to-Video pipeline

**As a** developer, **I want** `comfy_diffusion.pipelines.video.ltx.ltx2.depth` with
`manifest()` and `run(...)` **so that** I can generate a video from a reference video
using Lotus depth-map ControlNet conditioning (LTX 2.0 dev fp8).

**Acceptance Criteria:**
- [ ] `manifest()` returns entries for the **active** (non-bypassed) models from
  `video_ltx2_depth_to_video.json`:
  `ltx-2-19b-dev-fp8.safetensors`, `lotus-depth-d-v1-1.safetensors`,
  `ltx-2-19b-distilled-lora-384.safetensors`, `vae-ft-mse-840000-ema-pruned.safetensors`.
  (Bypassed entries — `ltx-2-19b-distilled.safetensors`, upscaler, depth LoRA,
  `gemma…`, distilled sampler nodes — are excluded.)
- [ ] `run(models_dir, prompt, video_path, ...)` follows the workflow order:
  `UNETLoader` (Lotus model) → `VAELoader` → `LotusConditioning` →
  `DisableNoise` → `BasicGuider` → `BasicScheduler` → `SamplerCustomAdvanced`
  (Lotus pass) → encode image → `LTXVAddGuide` → `SetFirstSigma` →
  two-pass LTX sampling → `LTXVLatentUpsampler` → VAE decode → returns
  `list[PIL.Image.Image]`.
- [ ] Sampler / scheduler values match the workflow: Lotus pass uses `euler` +
  `BasicScheduler(steps=1)`; LTX pass uses `euler_ancestral` + manual sigmas.
- [ ] `run()` accepts: `prompt`, `negative_prompt`, `video_path`, `seed`, `width`,
  `height`, `num_frames`, `lora_strength`.
- [ ] Pipeline file is at `comfy_diffusion/pipelines/video/ltx/ltx2/depth.py`.
- [ ] Typecheck / lint passes.

---

### US-006: LTX2 Pose-to-Video pipeline

**As a** developer, **I want** `comfy_diffusion.pipelines.video.ltx.ltx2.pose` with
`manifest()` and `run(...)` **so that** I can generate a video from a reference video
using DWPreprocessor pose estimation ControlNet conditioning (LTX 2.0 dev fp8).

**Acceptance Criteria:**
- [ ] `manifest()` returns entries for all active models from
  `video_ltx2_pose_to_video.json`:
  `ltx-2-19b-dev-fp8.safetensors`, `ltx-2-spatial-upscaler-x2-1.0.safetensors`,
  `ltx-2-19b-ic-lora-pose-control.safetensors`, `gemma_3_12B_it_fp4_mixed.safetensors`,
  `ltx-2-19b-distilled-lora-384.safetensors`.
- [ ] `run(models_dir, prompt, video_path, ...)` follows the workflow order:
  `GetVideoComponents` → `ImageFromBatch` → `ResizeImageMaskNode` →
  `DWPreprocessor` → `LTXVAddGuide` → `LTXVImgToVideoInplace` →
  two-pass sampling → `LTXVLatentUpsampler` → VAE decode → returns
  `list[PIL.Image.Image]`.
- [ ] Sampler / sigmas / CFG match the pose workflow (same manual sigma pattern as
  canny; CFG 3 for pass 1, CFG 1 for pass 2).
- [ ] `run()` accepts: `prompt`, `negative_prompt`, `video_path`, `first_frame_path`,
  `seed`, `width`, `height`, `num_frames`, `lora_strength`.
- [ ] Pipeline file is at `comfy_diffusion/pipelines/video/ltx/ltx2/pose.py`.
- [ ] Typecheck / lint passes.

---

### US-007: LTX23 Image+Audio-to-Video pipeline (ia2v)

**As a** developer, **I want** `comfy_diffusion.pipelines.video.ltx.ltx23.ia2v` with
`manifest()` and `run(...)` **so that** I can generate a video jointly conditioned on a
reference image and an audio file (LTX 2.3 22B dev fp8).

**Acceptance Criteria:**
- [ ] `manifest()` returns entries for all active models from
  `video_ltx2_3_ia2v.json`:
  `ltx-2.3-22b-dev-fp8.safetensors`, `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`,
  `gemma_3_12B_it_fp4_mixed.safetensors` (text encoder),
  `ltx-2.3-22b-distilled-lora-384.safetensors`,
  `gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors`.
- [ ] `run(models_dir, prompt, image_path, audio_path, ...)` follows the workflow order:
  `load_audio` → `LTXAVTextEncoderLoader` → `CheckpointLoaderSimple` →
  `LoraLoaderModelOnly` (distilled LoRA) → `LoraLoader` (Gemma abliterated LoRA) →
  `LTXVAudioVAELoader` → `ltxv_audio_vae_encode` → `ResizeImageMaskNode` →
  `ResizeImagesByLongerEdge` → `ComfyMathExpression(a/2)` → `EmptyLTXVLatentVideo` →
  `LTXVLatentUpsampler` → `LTXVImgToVideoInplace` → `LTXVConcatAVLatent` →
  pass-1 `CFGGuider` + `SamplerCustomAdvanced` → pass-2 (refinement) →
  `LTXVCropGuides` → VAE decode → returns `list[PIL.Image.Image]`.
- [ ] Samplers match the workflow: pass-1 uses `euler_cfg_pp` + manual sigmas
  `0.85, 0.7250, 0.4219, 0.0`; pass-2 uses `euler_ancestral_cfg_pp` + full sigma
  sequence `1.0, 0.99375, ..., 0.0`; CFG 1 throughout.
- [ ] `run()` accepts: `prompt`, `negative_prompt`, `image_path`, `audio_path`, `seed`,
  `width`, `height`, `num_frames`, `lora_strength_distilled`, `lora_strength_gemma`.
- [ ] Pipeline file is at `comfy_diffusion/pipelines/video/ltx/ltx23/ia2v.py`.
- [ ] `ltx23/__init__.py` updated to export `ia2v` and remove the "Not yet implemented"
  note.
- [ ] Typecheck / lint passes.

---

### US-008: Example scripts for each pipeline

**As a** developer, **I want** one runnable example script per pipeline **so that** I
can copy-paste a working invocation and adapt it to my own project.

**Acceptance Criteria:**
- [ ] `examples/video/ltx/ltx2/canny.py` — shows `download_models(manifest(), ...)`,
  `run(...)` with a sample video path, saves output frames to disk.
- [ ] `examples/video/ltx/ltx2/depth.py` — same pattern for depth pipeline.
- [ ] `examples/video/ltx/ltx2/pose.py` — same pattern for pose pipeline.
- [ ] `examples/video/ltx/ltx23/ia2v.py` — shows `download_models`, `run(...)` with
  both an image and an audio path.
- [ ] Each script accepts CLI flags `--models-dir`, `--prompt`, and pipeline-specific
  inputs (e.g. `--video`, `--image`, `--audio`) via `argparse`.
- [ ] Each script prints a clear error and usage hint if required arguments are missing.
- [ ] No script imports at module top level — all heavy imports inside `main()`.

---

### US-009: CPU-only test coverage

**As a** CI pipeline, **I want** CPU-only pytest tests for the new wrappers and
pipeline `manifest()` calls **so that** CI stays green without a GPU.

**Acceptance Criteria:**
- [ ] `tests/test_image.py` (or new file) contains unit tests for `image_scale_by`
  (verifies output shape) and `dw_preprocessor` (mocked ComfyUI node, verifies call
  signature) — all pass with `uv run pytest` on CPU.
- [ ] `tests/test_audio.py` (or new file) contains a unit test for `load_audio`
  (mocked `torchaudio.load`, verifies output dict keys and tensor shape) — passes on CPU.
- [ ] `tests/test_pipelines_ltx2.py` (or new file) verifies that `canny.manifest()`,
  `depth.manifest()`, and `pose.manifest()` return the correct file names without GPU.
- [ ] `tests/test_pipelines_ltx23.py` (or new file) verifies that `ia2v.manifest()`
  returns the correct file names without GPU.
- [ ] `uv run pytest` passes with no failures.

---

## Functional Requirements

- **FR-1:** `image_scale_by(image, upscale_method, scale_by)` wraps
  `comfy_extras.nodes_upscale_model.ImageScaleBy` (or equivalent comfy node) using the
  lazy-import pattern.
- **FR-2:** `dw_preprocessor(image, detect_hand, detect_body, detect_face, resolution)`
  wraps `comfy_extras.nodes_dwpose` (or equivalent) using the lazy-import pattern.
- **FR-3:** `load_audio(path, start_time, duration)` uses `torchaudio.load` and returns
  `{"waveform": tensor, "sample_rate": int}` — consistent with the ComfyUI AUDIO dict
  format expected by `ltxv_audio_vae_encode`.
- **FR-4:** All four pipeline files export exactly two public symbols: `manifest` and
  `run`. No other symbols are exported.
- **FR-5:** `manifest()` must list **only** models from active (non-bypassed) nodes as
  determined by reading the workflow JSON with the `workflow-reader` tool.
- **FR-6:** `run()` node execution order must mirror the `order` field from the workflow
  reader output. Do not reorder or skip steps.
- **FR-7:** Example scripts must be self-contained: they `import comfy_diffusion` and
  call the pipeline directly — no custom server or UI required.
- **FR-8:** `ltx23/__init__.py.__all__` must be updated to include `"ia2v"`.
- **FR-9:** New wrappers must be added to their respective module's `__all__`.

## Non-Goals (Out of Scope)

- The distilled variants of canny / depth / pose pipelines (those subgraphs are present
  in the workflow JSONs but are **bypassed** in the default active path).
- Phase 6 nodes (`LTXVAddGuideMulti`, `LTXVAudioVAEEncode`, `LTXVAudioVideoMask`, etc.)
  — those are for it_000029+.
- Phase 7 audio-to-video pipeline (`video_ltx_2_audio_to_video`).
- WAN, HunyuanVideo, or any non-LTX pipeline.
- A high-level `Pipeline` abstraction or auto-download on `run()`.

## Open Questions

- None.
