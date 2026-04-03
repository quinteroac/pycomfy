# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [000039] - 2026-04-03

### Added
- **PRD 001 — Image Generation via parallax-cli (sdxl, anima, z_image):** `parallax create image --model sdxl --prompt "..."` produces a PNG on disk.
- **PRD 001 — Image Generation via parallax-cli (sdxl, anima, z_image):** `parallax create image --model anima --prompt "..."` produces a PNG on disk.
- **PRD 001 — Image Generation via parallax-cli (sdxl, anima, z_image):** `parallax create image --model z_image --prompt "..."` produces a PNG on disk.
- **PRD 001 — Image Generation via parallax-cli (sdxl, anima, z_image):** The models directory is resolved from the `PYCOMFY_MODELS_DIR` environment variable
- **PRD 001 — Image Generation via parallax-cli (sdxl, anima, z_image):** The `--output` flag in the CLI maps directly to `--output` in the subprocess.
- **PRD 002 — Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** `parallax create video --model ltx2 --prompt "..."` produces an MP4 on disk.
- **PRD 002 — Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** `parallax create video --model ltx23 --prompt "..."` produces an MP4 on disk.
- **PRD 002 — Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** `parallax create video --model wan21 --prompt "..."` produces an MP4 on disk.
- **PRD 002 — Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** `parallax create video --model wan22 --prompt "..."` produces an MP4 on disk.
- **PRD 002 — Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** The models directory is resolved from `PYCOMFY_MODELS_DIR` / `--models-dir`
- **PRD 002 — Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** Env var and repo-root resolution follow the same pattern established in PRD 001
- **PRD 003 — Image-to-Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** `parallax create video --model ltx2 --input image.png --prompt "..."` produces an MP4.
- **PRD 003 — Image-to-Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** `parallax create video --model ltx23 --input image.png --prompt "..."` produces an MP4.
- **PRD 003 — Image-to-Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** `parallax create video --model wan21 --input image.png --prompt "..."` produces an MP4.
- **PRD 003 — Image-to-Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** `parallax create video --model wan22 --input image.png --prompt "..."` produces an MP4.
- **PRD 003 — Image-to-Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** The `--input` flag selects the i2v code path; omitting it preserves the existing t2v
- **PRD 003 — Image-to-Video Generation via parallax-cli (ltx2, ltx23, wan21, wan22):** Flag forwarding follows each script's exact interface (e.g. `--cfg` → `--cfg-pass1`
- **PRD 004 — Audio Generation via parallax-cli (ace_step):** `parallax create audio --model ace_step --prompt "electronic ambient" --output out.wav`
- **PRD 004 — Audio Generation via parallax-cli (ace_step):** Model component filenames (`--unet`, `--vae`, `--text-encoder-1`, `--text-encoder-2`)
- **PRD 004 — Audio Generation via parallax-cli (ace_step):** Extended generation parameters (`--cfg`, `--lyrics`, `--bpm`) are exposed as
- **PRD 004 — Audio Generation via parallax-cli (ace_step):** Env var and repo-root resolution follow the same pattern established in PRDs 001–003

## [000038] - 2026-04-03

### Added
- Define the full command/subcommand tree: `parallax create|edit image|video|audio`
- Declare all flags (`--model`, `--prompt`, `--input`, `--output`, and media-specific params) with types, defaults, and descriptions
- Validate required flags and known model values, exiting with code `1` and a clear error message on failure
- Ship as an installable global binary (`bun link` / `bun install -g`)
- Provide complete `--help` at every command level

## [000037] - 2026-04-02

### Added
- Provide a self-contained, executable CLI example for `edit_2511` that a developer can run
- Document the pipeline's public API (`manifest()` + `run()`) through concrete, annotated

## [000036] - 2026-04-02

### Added
- Expose Qwen Image Edit 2511 inference as a composable Python function usable
- Add the four missing node wrappers to their respective library modules,
- Keep the pipeline CPU-safe: all tests pass without GPU using mocked weights.

## [000035] - 2026-04-02

### Added
- Expose `vae_decode_audio()` in `comfy_diffusion/audio.py` — the final
- Provide three pipeline modules (`checkpoint.py`, `split.py`, `split_4b.py`)
- Allow developers to generate music from a text prompt with a single function
- All tests pass on CPU (mocked weights).

### Added

#### Library wrappers
- `latent.empty_flux2_latent_image(width, height, batch_size)` — creates 128-channel Flux.2 Klein latents (lazy import from `comfy_extras.nodes_flux`)
- `latent.empty_qwen_image_layered_latent_image(width, height, layers, batch_size)` — creates Qwen Image Layered latents (lazy import from `comfy_extras.nodes_qwen`)
- `conditioning.reference_latent(conditioning, latent)` — injects a reference-image latent into conditioning via `ReferenceLatent` from `comfy_extras.nodes_edit_model`
- `sampling.flux_kv_cache(model)` — applies FluxKVCache model patch from `comfy_extras.nodes_flux`
- `image.image_scale_to_total_pixels(image, upscale_method, megapixels, smallest_side)` — wraps `ImageScaleToTotalPixels` from `comfy_extras.nodes_post_processing`
- `image.image_scale_to_max_dimension(image, upscale_method, max_dimension)` — wraps `ImageScaleToMaxDimension` from `comfy_extras.nodes_images`
- `image.get_image_size(image)` — returns `(width, height)` tuple; wraps `GetImageSize` from `comfy_extras.nodes_images`

#### Pipelines — `comfy_diffusion.pipelines.image.flux_klein`
- `t2i_4b_base` — Flux.2 Klein 4B base text-to-image pipeline (euler, CFG 5, 20 steps)
- `t2i_4b_distilled` — Flux.2 Klein 4B distilled text-to-image pipeline (euler, CFG 1, 4 steps, `conditioning_zero_out` negative)
- `edit_4b_base` — Flux.2 Klein 4B base image-edit pipeline
- `edit_4b_distilled` — Flux.2 Klein 4B distilled image-edit pipeline
- `edit_9b_base` — Flux.2 Klein 9B base image-edit pipeline
- `edit_9b_distilled` — Flux.2 Klein 9B distilled image-edit pipeline
- `edit_9b_kv` — Flux.2 Klein 9B KV-cache image-edit pipeline; accepts `reference_image` + `subject_image`; applies `flux_kv_cache` and `image_scale_to_total_pixels`

#### Pipelines — `comfy_diffusion.pipelines.image.qwen`
- `layered` — Qwen Image Layered pipeline; exports `run_t2l()` (text-to-layers) and `run_i2l()` (image-to-layers); defaults: 20 steps, CFG 2.5, euler/simple, 2 layers

#### Example scripts
- `examples/flux_klein_t2i_4b_base.py`
- `examples/flux_klein_t2i_4b_distilled.py`
- `examples/flux_klein_edit_4b_base.py`
- `examples/flux_klein_edit_4b_distilled.py`
- `examples/flux_klein_edit_9b_base.py`
- `examples/flux_klein_edit_9b_distilled.py`
- `examples/flux_klein_edit_9b_kv.py`
- `examples/qwen_layered_t2l.py`
- `examples/qwen_layered_i2l.py`

## [1.3.0] - 2026-03-22

### Added
- `latent.ltxv_empty_latent_video(width, height, length, batch_size)` — empty latent for LTX-Video 2
- `latent.ltxv_latent_upsample(latent, upscale_model, ...)` — upsamples LTX-Video latents using a dedicated upscale model
- `conditioning.ltxv_crop_guides(...)` — crops guide frames for LTX-Video 2 conditioning
- `audio.ltxv_concat_av_latent(audio_latent, video_latent)` — concatenates audio and video latents for LTX-Video 2
- `audio.ltxv_separate_av_latent(av_latent)` — separates a combined audio-video latent
- `sampling.manual_sigmas(model, sigmas)` — injects custom sigma schedules into the sampler
- `ModelManager.load_latent_upscale_model(path)` — loads an LTX-Video 2 latent upscale model
- `pipelines.video.ltx2.audio_to_video` — LTX-Video 2 audio-to-video pipeline
- `pipelines.video.ltx2.t2sv` — LTX-Video 2 text-to-sound-and-video pipeline
- `pipelines.image.wan.i2v_21`, `t2v_21`, `flf2v_21` — WAN 2.1 image/text/first-last-frame-to-video pipelines
- `pipelines.image.wan.i2v`, `t2v`, `flf2v`, `s2v`, `ti2v` — WAN 2.2 pipelines with audio encoder support
- `audio.audio_encoder_encode(audio_encoder, audio)` — encodes audio features with a WAN audio encoder
- `ModelManager.load_audio_encoder(path)` — loads a WAN audio encoder model
- `conditioning.wan22_image_to_video_latent(...)` — creates WAN 2.2 image-to-video latent with optional start-image encoding

### Fixed
- Enhanced audio processing and corrected checkpoint path in `ModelManager`

## [1.2.0] - 2026-03-21

### Added
- `ModelManager.load_upscale_model(path)` — loads a standard upscale model
- `upscale_models` folder registered in `ModelManager.__init__`

### Changed
- Updated ComfyUI submodule to `v0.18.0`

### Fixed
- Updated `actions/upload-artifact` and `actions/download-artifact` to latest versions in CI
- Updated model registration to include `upscale_models`
- Removed stale `base-runtime-check.md` assertion from wheel test

## [1.1.0] - 2026-03-14

### Added
- `ModelManager.load_llm(path)` — loads an LLM (GGUF) model
- `generate_text(llm, prompt, ...)` — generates text with a loaded LLM
- `generate_ltx2_prompt(llm, image, prompt)` — generates an LTX-Video 2 enhanced prompt via LLM
- `load_clip(paths...)` — now accepts a variadic number of paths (backward-compatible)
- Auto-bootstrap: `check_runtime()` automatically downloads the ComfyUI submodule when it is missing rather than returning an error

## [1.0.0] - 2026-03-10

### Added
- Foundation: `comfy_diffusion` package with `_runtime.py` encapsulating all `sys.path` manipulation
- `check_runtime()` — returns error dict (never raises) when the ComfyUI submodule is not initialized; always populates `python_version`
- `vae_encode(vae, image)` / `vae_decode(vae, latent)` re-exported from package root
- `latent` module: `empty_latent_image`, `latent_upscale`, `latent_upscale_by`, `latent_crop`, `latent_from_batch`, `latent_cut`, `latent_cut_to_batch`, `latent_composite`, `latent_blend`, `latent_add`, `latent_multiply`, `latent_concat`
- `conditioning` module: `encode_prompt`, `encode_prompt_flux`, `encode_clip_vision`, `flux_guidance`, `wan_vace_to_video`, `wan_fun_control_to_video`, `wan22_fun_control_to_video`, `wan_fun_inpaint_to_video`, `wan_camera_embedding`, `wan_camera_image_to_video`, `wan_phantom_subject_to_video`, `wan_track_to_video`, `wan_sound_image_to_video`, `wan_sound_image_to_video_extend`, `wan_humo_image_to_video`, `wan_animate_to_video`, `wan_infinite_talk_to_video`, `wan_scail_to_video`
- `sampling` module: `ksampler`, `ksampler_select`, `random_noise`, `flux2_scheduler`, `cfg_guider`, `sampler_custom_advanced`, `conditioning_zero_out`, `model_sampling_flux`, `model_sampling_sd3`, `model_sampling_aura_flow`, `video_linear_cfg_guidance`, `video_triangle_cfg_guidance`
- `image` module: `load_image`, `save_image`, `image_to_tensor`, `tensor_to_image`, `image_scale`, `image_scale_by`, `image_crop`, `image_pad`, `image_composite`, `image_invert`, `empty_image`, `canny`, `dw_preprocessor`, `image_resize_kj`, `image_batch_extend_with_overlap`, `math_expression`
- `mask` module: `load_mask`, `image_to_mask`, `mask_to_image`, `grow_mask`, `feather_mask`
- `models` module: `ModelManager` with support for `checkpoints`, `unet`, `clip`, `vae`, `lora`, `controlnet`, `clip_vision`, `upscale_models`, `llm`, `audio_encoder`, `latent_upscale_model`
- `vae` module: `vae_encode`, `vae_decode`, `vae_encode_tiled`, `vae_decode_tiled`
- `apply_lora` re-exported from package root
- `py.typed` marker and type stubs shipped with the package
- `cuda` and `cpu` extras for PyTorch index selection
- Distributable agent skills bundled in `comfy_diffusion/skills/`
- Post-install smoke test

## [0.1.1-preview] - 2026-02-22

### Fixed
- Updated vendor path from `comfy_diffusion/vendor/ComfyUI` to `vendor/ComfyUI` in CI, MANIFEST.in, and tests
- Moved ComfyUI vendor inside package so PyPI wheel includes it

## [0.1.0-preview] - 2026-02-20

### Added
- Initial preview release with basic runtime bootstrap and `comfy.*` path injection

[1.3.0]: https://github.com/quinteroac/comfy-diffusion/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/quinteroac/comfy-diffusion/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/quinteroac/comfy-diffusion/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/quinteroac/comfy-diffusion/compare/v0.1.1-preview...v1.0.0
[0.1.1-preview]: https://github.com/quinteroac/comfy-diffusion/compare/v0.1.0-preview...v0.1.1-preview
[0.1.0-preview]: https://github.com/quinteroac/comfy-diffusion/releases/tag/v0.1.0-preview
