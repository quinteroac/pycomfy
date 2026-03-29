# Requirement: Phase 6 — LTX2 Audio-to-Video Pipeline

## Context

Phase 5 (it_000028) completed all remaining LTX2 ControlNet pipelines and the LTX2.3
image+audio-to-video pipeline. Phase 6 delivers the `video_ltx_2_audio_to_video` workflow
as an importable Python pipeline, completing the full LTX2 pipeline family.

This pipeline takes an image and an audio file as inputs and generates a video whose
visual content is driven by the image and audio track. Internally it uses:

- LTX2 distilled UNet (`transformer_only` variant)
- Gemma 3 12B fp8 text encoder + LTX2 embeddings connector
- Separate audio VAE and video VAE (loaded via `VAELoaderKJ`)
- `LTX2_NAG` model patch for improved audio-visual alignment
- `LTXVImgToVideoInplaceKJ` (KJ variant) for image-to-first-frame injection
- A multi-pass video extension strategy (4 extension passes after initial generation)
- `AudioCrop`, `AudioSeparation`, `TrimAudioDuration` for audio preprocessing
- `ImageResizeKJv2` for input image resizing
- `ImageBatchExtendWithOverlap` for seamless frame stitching
- `CreateVideo` to combine output frames and audio into a VIDEO object

Ten new node wrappers must be implemented before the pipeline can be written.

The KJ-prefixed nodes (`VAELoaderKJ`, `ImageResizeKJv2`, `LTXVImgToVideoInplaceKJ`,
`LTX2_NAG`, `LTX2SamplingPreviewOverride`) come from
[`kijai/ComfyUI-KJNodes`](https://github.com/kijai/ComfyUI-KJNodes) (`nodes/ltxv_nodes.py`,
`nodes/image_nodes.py`, `nodes/nodes.py`). Their logic must be re-implemented directly in
`comfy_diffusion` modules — **not** by importing KJNodes as a dependency.

---

## Goals

- Expose all 10 missing ComfyUI node wrappers as typed, lazy-import functions in the
  appropriate `comfy_diffusion` modules (`audio.py`, `video.py`, `image.py`, `models.py`).
- Implement `comfy_diffusion/pipelines/video/ltx/ltx2/audio_to_video.py` with the
  canonical `manifest()` / `run()` contract, mirroring the reference workflow exactly.
- Provide a self-contained example script under `examples/`.
- Cover all new wrappers and the pipeline with CPU-safe pytest tests.

---

## User Stories

### US-001: Audio preprocessing wrappers

**As a** Python developer, **I want** `audio_crop`, `audio_separation`, and
`trim_audio_duration` functions in `comfy_diffusion.audio` **so that** I can preprocess
audio segments before injecting them into LTX2 audio-to-video inference.

**Acceptance Criteria:**
- [ ] `audio_crop(audio, start_time, end_time)` wraps `AudioCrop`; `start_time` and
      `end_time` are string timestamps (e.g. `"1:05"`); returns an AUDIO dict.
- [ ] `audio_separation(audio, mode, fft_n, win_length)` wraps `AudioSeparation`;
      returns a dict with keys `bass`, `drums`, `other`, `vocals` (each an AUDIO dict).
- [ ] `trim_audio_duration(audio, start, duration)` wraps `TrimAudioDuration`; `start`
      is a float (seconds), `duration` is a float (seconds); returns an AUDIO dict.
- [ ] All three functions follow the lazy-import pattern (no top-level `comfy.*` imports).
- [ ] All three are added to `audio.__all__`.
- [ ] Typecheck / lint passes.

---

### US-002: Video / model wrappers

**As a** Python developer, **I want** `ltx2_nag`, `ltxv_img_to_video_inplace_kj`,
`ltx2_sampling_preview_override`, and `create_video` available in `comfy_diffusion.video`,
and `load_vae_kj` as a method on `ModelManager` in `comfy_diffusion.models`, **so that**
the audio-to-video pipeline can be implemented without raw `comfy.*` calls in the pipeline
file.

**Acceptance Criteria:**
- [ ] `ltx2_nag(model, nag_scale, nag_alpha, nag_tau, nag_cond_video=None,
      nag_cond_audio=None, inplace=True)` wraps `LTX2_NAG`; returns a patched MODEL.
      Source: `kijai/ComfyUI-KJNodes` → `nodes/ltxv_nodes.py`.
- [ ] `ltxv_img_to_video_inplace_kj(vae, latent, image, index=0, strength=1.0)` wraps
      `LTXVImgToVideoInplaceKJ`; the `DynamicCombo` UI pattern is flattened to explicit
      `image/index/strength` params for the Python API; returns a LATENT dict.
      Source: `kijai/ComfyUI-KJNodes` → `nodes/ltxv_nodes.py`.
- [ ] `ltx2_sampling_preview_override(model, preview_rate=8, latent_upscale_model=None,
      vae=None)` wraps `LTX2SamplingPreviewOverride`; returns a MODEL.
      Source: `kijai/ComfyUI-KJNodes` → `nodes/ltxv_nodes.py`.
- [ ] `create_video(images, audio, fps)` wraps `CreateVideo`; returns a VIDEO object.
- [ ] `ModelManager.load_vae_kj(path, device="main_device", dtype="bf16")` wraps
      `VAELoaderKJ` from `kijai/ComfyUI-KJNodes` (`nodes/nodes.py`); loads VAE with
      explicit device and dtype (`bf16`/`fp16`/`fp32`); returns a VAE object; path is
      resolved against `models_dir` and registered in `folder_paths["vae"]` before
      calling `comfy.sd.VAE`; follows the `str | Path` pattern.
- [ ] All five symbols follow the lazy-import pattern.
- [ ] `ltx2_nag`, `ltxv_img_to_video_inplace_kj`, `ltx2_sampling_preview_override`, and
      `create_video` are added to `video.__all__`.
- [ ] `load_vae_kj` is accessible via `ModelManager`.
- [ ] Typecheck / lint passes.

---

### US-003: Image wrappers

**As a** Python developer, **I want** `image_resize_kj` and
`image_batch_extend_with_overlap` in `comfy_diffusion.image` **so that** I can resize
input images (KJ-style) and stitch video extension frames with cross-fade overlap.

**Acceptance Criteria:**
- [ ] `image_resize_kj(image, width, height, upscale_method="lanczos",
      keep_proportion="crop", pad_color="0, 0, 0", crop_position="center",
      divisible_by=2, device="cpu")` wraps `ImageResizeKJv2`; returns
      `(IMAGE, width: int, height: int)` (the 4th `MASK` output is discarded).
      Source: `kijai/ComfyUI-KJNodes` → `nodes/image_nodes.py`.
- [ ] `image_batch_extend_with_overlap(source_images, new_images=None, overlap=13,
      overlap_side="source", overlap_mode="filmic_crossfade")` wraps
      `ImageBatchExtendWithOverlap`; returns the `extended_images` IMAGE tensor (3rd of
      3 outputs; the passthrough `source_images` and `start_images` outputs are discarded).
      Source: `kijai/ComfyUI-KJNodes` → `nodes/image_nodes.py`.
- [ ] Both functions follow the lazy-import pattern.
- [ ] Both are added to `image.__all__`.
- [ ] Typecheck / lint passes.

---

### US-004: `ltx2/audio_to_video` pipeline

**As a** Python developer, **I want** `comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video`
with `manifest()` and `run()` **so that** I can generate an audio-driven video from a
single image and audio file without manually assembling node calls.

**Acceptance Criteria:**
- [ ] `manifest()` returns a `list[ModelEntry]` containing exactly the model files
      required by active (non-bypassed) nodes in `video_ltx_2_audio_to_video.json`:
      UNet (`ltx-2-19b-distilled_transformer_only_bf16.safetensors`), two text encoders,
      audio VAE (`LTX2_audio_vae_bf16.safetensors`), and video VAE
      (`LTX2_video_vae_bf16.safetensors`). Bypassed `VHS_VideoCombine` nodes contribute
      no entries.
- [ ] `run(*, models_dir, prompt, image_path, audio_path, audio_start_time, audio_end_time,
      width, height, length, fps, steps, cfg, seed, ...)` executes the full workflow in
      node execution order: audio crop → audio separation → image resize → text encode →
      conditioning → LTX2_NAG patch → preview override → EmptyLTXVLatentVideo →
      img-to-video inplace (KJ) → audio VAE encode → latent concat → schedule → sample →
      separate AV latent → VAE decode → 4 × video extension passes
      (EmptyLTXVLatentVideo → img-to-video → audio encode → concat → sample → separate →
      decode) → ImageBatchExtendWithOverlap → CreateVideo → return.
- [ ] `run()` returns a dict with key `"video"` (VIDEO object) and `"frames"` (list of
      `PIL.Image.Image`).
- [ ] `__all__ = ["manifest", "run"]`.
- [ ] `ltx2/__init__.py` exports `"audio_to_video"`.
- [ ] Typecheck / lint passes.

---

### US-005: Example script

**As a** Python developer, **I want** `examples/video_ltx2_audio_to_video.py` **so that**
I can run the audio-to-video pipeline from the command line with minimal boilerplate.

**Acceptance Criteria:**
- [ ] Script accepts `--models-dir`, `--prompt`, `--image`, `--audio`,
      `--audio-start`, `--audio-end` CLI arguments; all required args print a usage
      error if missing.
- [ ] Script calls `download_models(manifest(), models_dir=...)` then `run(...)`.
- [ ] All heavy imports (`torch`, `torchaudio`, `comfy.*`, `comfy_diffusion.*`) are
      deferred inside `main()`, not at module top level.
- [ ] Typecheck / lint passes.

---

### US-006: Tests

**As a** developer, **I want** CPU-safe pytest tests for all new wrappers and the
pipeline **so that** CI stays green without GPU access.

**Acceptance Criteria:**
- [ ] `tests/test_audio_wrappers_phase6.py` covers `audio_crop`, `audio_separation`,
      `trim_audio_duration` — import paths, signatures, `__all__` membership, and lazy
      import behaviour (no `comfy.*` imported at module load time).
- [ ] `tests/test_video_wrappers_phase6.py` covers `ltx2_nag`,
      `ltxv_img_to_video_inplace_kj`, `ltx2_sampling_preview_override`, `create_video`,
      and `ModelManager.load_vae_kj` — same pattern.
- [ ] `tests/test_image_wrappers_phase6.py` covers `image_resize_kj` and
      `image_batch_extend_with_overlap`.
- [ ] `tests/test_pipelines_ltx2_audio_to_video.py` validates `manifest()` (entry count,
      dest paths, no duplicates) and `run()` signature via AST inspection.
- [ ] `uv run pytest` passes with 0 failures.

---

## Functional Requirements

- **FR-1:** `audio.audio_crop(audio, start_time, end_time)` — wraps `AudioCrop` with
  string-timestamp crop bounds; returns AUDIO dict; lazy imports; in `audio.__all__`.
- **FR-2:** `audio.audio_separation(audio, mode, fft_n, win_length)` — wraps
  `AudioSeparation`; returns dict with `bass`, `drums`, `other`, `vocals` keys; lazy
  imports; in `audio.__all__`.
- **FR-3:** `audio.trim_audio_duration(audio, start, duration)` — wraps
  `TrimAudioDuration`; returns AUDIO dict; lazy imports; in `audio.__all__`.
- **FR-4:** `video.ltx2_nag(model, nag_scale, nag_alpha, nag_tau, nag_cond_video=None,
  nag_cond_audio=None, inplace=True)` — wraps `LTX2_NAG` from
  `kijai/ComfyUI-KJNodes`; returns MODEL; lazy imports; in `video.__all__`.
- **FR-5:** `video.ltxv_img_to_video_inplace_kj(vae, latent, image, index=0, strength=1.0)`
  — wraps `LTXVImgToVideoInplaceKJ` from `kijai/ComfyUI-KJNodes`; flattens the
  DynamicCombo UI to explicit Python params; returns LATENT; lazy imports; in
  `video.__all__`.
- **FR-6:** `video.ltx2_sampling_preview_override(model, preview_rate=8,
  latent_upscale_model=None, vae=None)` — wraps `LTX2SamplingPreviewOverride` from
  `kijai/ComfyUI-KJNodes`; returns MODEL; lazy imports; in `video.__all__`.
- **FR-7:** `video.create_video(images, audio, fps)` — wraps `CreateVideo`; returns
  VIDEO object; lazy imports; in `video.__all__`.
- **FR-8:** `ModelManager.load_vae_kj(path, device="main_device", dtype="bf16")` — wraps
  `VAELoaderKJ`; returns VAE object; path follows `str | Path` pattern; lazy imports;
  `models_dir` is used for relative path resolution.
- **FR-9:** `image.image_resize_kj(image, width, height, upscale_method, keep_proportion,
  pad_color, crop_position, divisible_by, device)` — wraps `ImageResizeKJv2` from
  `kijai/ComfyUI-KJNodes`; returns `(IMAGE, int, int)`; discards the 4th MASK output;
  lazy imports; in `image.__all__`.
- **FR-10:** `image.image_batch_extend_with_overlap(source_images, new_images=None,
  overlap=13, overlap_side="source", overlap_mode="filmic_crossfade")` — wraps
  `ImageBatchExtendWithOverlap` from `kijai/ComfyUI-KJNodes`; returns `extended_images`
  IMAGE (3rd output); lazy imports; in `image.__all__`.
- **FR-11:** `pipelines/video/ltx/ltx2/audio_to_video.py` — exports `manifest()` and
  `run()` only; `manifest()` lists exactly 5 model entries (UNet + 2 text encoders +
  audio VAE + video VAE); `run()` executes all 104 active nodes in order, spanning
  initial generation + 4 video extension passes; returns `{"video": VIDEO, "frames":
  list[PIL.Image.Image]}`.
- **FR-12:** `ltx2/__init__.py` updated to include `"audio_to_video"` in `__all__`.
- **FR-13:** `examples/video_ltx2_audio_to_video.py` — self-contained CLI entry point
  following the same structure as existing example scripts.
- **FR-14:** All new wrappers and the pipeline module are covered by CPU-safe pytest tests.

---

## Non-Goals (Out of Scope)

- Implementing `LTXVAddGuideMulti`, `LTXVAudioVideoMask`, or `LTXVChunkFeedForward` —
  these nodes appear in the Phase 6 roadmap list but are NOT used in
  `video_ltx_2_audio_to_video.json` (not present in active nodes).
- Implementing a `SaveVideo` wrapper — the workflow's `SaveVideo` node is a UI save-to-disk
  utility; the pipeline returns a VIDEO object and leaves saving to the caller.
- Wrapping `VHS_VideoCombine` — all four instances are BYPASSED in the reference workflow.
- Wrapping `FloatConstant` — the constant value can be inlined directly in `run()`.
- GPU validation — the CI environment is CPU-only; GPU smoke-testing is out of scope.
- Modifying the ComfyUI submodule pin.

---

## Open Questions

- None. VAE source confirmed: both `LTX2_audio_vae_bf16.safetensors` and
  `LTX2_video_vae_bf16.safetensors` are hosted at `Kijai/LTXV2_comfy` on HuggingFace,
  under the `VAE/` subdirectory (e.g.
  `https://huggingface.co/Kijai/LTXV2_comfy/tree/main/VAE`). `manifest()` must include
  these as `HFModelEntry(repo_id="Kijai/LTXV2_comfy", filename="VAE/LTX2_audio_vae_bf16.safetensors", ...)`
  and similarly for the video VAE. Destination paths: `vae/LTX2_audio_vae_bf16.safetensors`
  and `vae/LTX2_video_vae_bf16.safetensors`.
