# Requirement: LTX Core Nodes — Phase 2

## Context

Phase 2 of the roadmap targets the three ComfyUI nodes that block all ltx2 and ltx3
pipelines. Two of the three (`LTXAVTextEncoderLoader` → `ModelManager.load_ltxav_text_encoder`,
`LTXVAudioVAELoader` → `ModelManager.load_ltxv_audio_vae`) are already implemented in
`comfy_diffusion/models.py` with tests. The remaining blocker is `LTXVImgToVideoInplace`,
which provides the image-conditioning latent injection used by every i2v, LoRA, ControlNet,
and ltx3 workflow. This iteration adds the Python wrapper for that node and supplies
CPU-only tests for all three exposed capabilities.

## Goals

- Expose `LTXVImgToVideoInplace` as a callable Python function following the lazy-import
  pattern used throughout the library.
- Ensure all three Phase 2 capabilities (both loaders + inplace i2v) are covered by
  CPU-only pytest tests that pass in CI.
- Add `device` parameter to `ModelManager.load_ltxav_text_encoder` to fully match the `LTXAVTextEncoderLoader` node signature.

## User Stories

### US-001: `ltxv_img_to_video_inplace()` function

**As a** Python developer composing an LTX2/LTX3 image-to-video pipeline,
**I want** a `ltxv_img_to_video_inplace(vae, image, latent, strength, bypass)` function
**so that** I can inject an image frame into a latent without operating a ComfyUI server.

**Acceptance Criteria:**
- [ ] Function is implemented in `comfy_diffusion/video.py` (or a similarly justified module).
- [ ] Signature: `ltxv_img_to_video_inplace(vae: Any, image: Any, latent: dict[str, Any], strength: float = 1.0, bypass: bool = False) -> dict[str, Any]`.
- [ ] Returns a latent dict containing `"samples"` and `"noise_mask"` keys (mirrors `LTXVImgToVideoInplace.execute()` output).
- [ ] When `bypass=True`, returns the input `latent` unchanged without touching `vae` or `image`.
- [ ] All ComfyUI/torch imports are deferred to call time (no top-level imports); follows the lazy-import pattern.
- [ ] Function is added to `__all__` in its module.
- [ ] Typecheck / lint passes.

### US-002: CPU-only tests for `ltxv_img_to_video_inplace()`

**As a** CI pipeline running without a GPU,
**I want** pytest tests for `ltxv_img_to_video_inplace()` that use mocked ComfyUI internals
**so that** correctness is verified without real model weights.

**Acceptance Criteria:**
- [ ] A new test file `tests/test_ltxv_img_to_video_inplace.py` exists.
- [ ] Tests cover: normal path (returns dict with `samples` + `noise_mask`), `bypass=True` path (returns unmodified latent), and image auto-resize path (image shape differs from latent spatial dims).
- [ ] All tests pass with `uv run pytest tests/test_ltxv_img_to_video_inplace.py` on a CPU-only machine (no real weights, no GPU).
- [ ] Typecheck / lint passes.

### US-003: Verify existing loader tests pass

**As a** developer completing Phase 2,
**I want** confirmation that `test_model_manager_ltxav_text_encoder_loading.py` and
`test_model_manager_ltxv_audio_vae_loading.py` still pass after this iteration's changes
**so that** the Phase 2 loader coverage is fully green.

**Acceptance Criteria:**
- [ ] `uv run pytest tests/test_model_manager_ltxav_text_encoder_loading.py tests/test_model_manager_ltxv_audio_vae_loading.py` exits 0 with no modifications to those test files.
- [ ] Typecheck / lint passes.

### US-004: `device` parameter on `ModelManager.load_ltxav_text_encoder`

**As a** developer running inference on CPU or offloading the text encoder to CPU to free VRAM,
**I want** `ModelManager.load_ltxav_text_encoder` to accept a `device` parameter (`"default"` or `"cpu"`)
**so that** the method fully matches the `LTXAVTextEncoderLoader` node signature.

**Acceptance Criteria:**
- [ ] `load_ltxav_text_encoder` gains a `device: str = "default"` parameter.
- [ ] When `device="cpu"`, `load_device` and `offload_device` in `model_options` are both set to `torch.device("cpu")`, matching the ComfyUI node behavior.
- [ ] When `device="default"` (or omitted), behavior is identical to the current implementation (no `model_options`).
- [ ] Existing tests in `test_model_manager_ltxav_text_encoder_loading.py` continue to pass without modification.
- [ ] New test cases cover the `device="cpu"` path (verify `model_options` is passed to `comfy.sd.load_clip`).
- [ ] Typecheck / lint passes.

## Functional Requirements

- FR-1: `ltxv_img_to_video_inplace(vae, image, latent, strength=1.0, bypass=False)` mirrors `LTXVImgToVideoInplace.execute()` from `vendor/ComfyUI/comfy_extras/nodes_lt.py` exactly — same logic, same output shape.
- FR-2: When `image` spatial dimensions differ from the latent spatial dimensions (derived via `vae.downscale_index_formula`), the function resizes the image with bilinear interpolation (using `comfy.utils.common_upscale`), matching the node's behavior.
- FR-3: The `noise_mask` tensor shape is `(batch, 1, latent_frames, 1, 1)`; frames `[:t.shape[2]]` are set to `1.0 - strength`.
- FR-4: All deferred imports follow the pattern established in `audio.py` and `conditioning.py` — no `torch` or `comfy.*` at module top level.
- FR-5: The function is callable without a GPU (tensors must remain on CPU when inputs are on CPU).
- FR-6: `ModelManager.load_ltxav_text_encoder(text_encoder_path, checkpoint_path, device="default")` — when `device="cpu"`, passes `model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}` to `comfy.sd.load_clip`.

## Non-Goals (Out of Scope)

- Exposing `LTXVImgToVideoInplace` as a `ModelManager` method — it is a conditioning operation, not a loader.
- Implementing any Phase 3–7 pipeline (ltx2_i2v, ltx3, etc.).
- Adding `ltxv_img_to_video_inplace` to the package-level re-exports in `__init__.py`.

## Open Questions

- None.
