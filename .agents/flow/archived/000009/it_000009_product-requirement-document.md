# Requirement: Batch VAE — `vae_decode_batch`, `vae_encode_batch` + tiled variants

## Context

Iterations 1–8 established single-image VAE encode/decode (including tiled variants).
Video generation models such as WAN and LTXV operate on sequences of frames packed into
a batched latent tensor. Callers currently have no clean API to decode all frames at once
or to encode a list of PIL images into a single video latent. This iteration adds four
batch-oriented helpers to `pycomfy.vae`.

## Goals

- Provide `vae_decode_batch` to convert any batched latent (4-D or 5-D) into a flat
  `list[PIL.Image]` — one image per frame, across all batch items.
- Provide `vae_encode_batch` to convert a `list[PIL.Image]` into a batched latent tensor.
- Provide tiled variants (`vae_decode_batch_tiled`, `vae_encode_batch_tiled`) for large
  frames where memory would otherwise OOM.
- All four functions must follow the established `vae.py` patterns: pure duck-typing
  protocols, lazy torch import, no comfy imports at module top level.
- All tests pass on CPU-only environments (CI constraint).

## User Stories

### US-001: Decode a video latent batch into PIL frames

**As a** Python developer building a video generation pipeline,
**I want** to call `vae_decode_batch(vae, latent)` with a ComfyUI LATENT dict whose
`"samples"` tensor is either 4-D `(B, C, H, W)` or 5-D `(B, C, T, H, W)`,
**so that** I get back a flat `list[PIL.Image]` with one image per frame (all batches
concatenated) without writing any reshape boilerplate myself.

**Acceptance Criteria:**
- [ ] `vae_decode_batch(vae, {"samples": tensor_4d})` returns a `list[PIL.Image]` of
      length `B` (one per batch item).
- [ ] `vae_decode_batch(vae, {"samples": tensor_5d})` returns a `list[PIL.Image]` of
      length `B * T` (all frames of all batch items, in order).
- [ ] Shape auto-detection: the function checks `len(tensor.shape)` — no explicit `mode`
      argument is required from the caller.
- [ ] The returned list is never empty when the input is valid.
- [ ] `is_nested` / unbind handling is applied (matching existing `vae_decode` logic).
- [ ] `detach()` / `cpu()` called on each image slice before conversion (GPU-safety).
- [ ] Typecheck / lint passes.

---

### US-002: Encode a list of PIL images into a video latent batch

**As a** Python developer building a video generation pipeline,
**I want** to call `vae_encode_batch(vae, images)` with a `list[PIL.Image]`,
**so that** I get back a ComfyUI LATENT dict `{"samples": tensor}` whose shape is
appropriate for passing to the sampler (batched 4-D or 5-D, depending on the VAE).

**Acceptance Criteria:**
- [ ] `vae_encode_batch(vae, [img1, img2, img3])` returns a dict with key `"samples"`.
- [ ] All images are converted to RGB and stacked into a single pixel tensor before
      encoding (matching the BHWC layout expected by ComfyUI's VAE).
- [ ] Works with `torch` available (returns real tensor) and without `torch` (returns
      `_ListTensor`-based dict) — consistent with the existing `vae_encode` fallback
      pattern.
- [ ] Raises `ValueError` when `images` is an empty list.
- [ ] Typecheck / lint passes.

---

### US-003: Tiled decode for large video frames

**As a** Python developer generating high-resolution video,
**I want** `vae_decode_batch_tiled(vae, latent, tile_size, overlap)`,
**so that** I can decode large batched latents frame-by-frame without running out of
GPU memory.

**Acceptance Criteria:**
- [ ] Signature: `vae_decode_batch_tiled(vae, latent, tile_size=512, overlap=64)` —
      matching the `vae_decode_tiled` defaults.
- [ ] Decodes every frame slice using `vae.decode_tiled(...)` and collects results into
      a flat `list[PIL.Image]`.
- [ ] Handles both 4-D and 5-D tensors (same auto-detection logic as US-001).
- [ ] `_VaeDecoderTiled` protocol is reused (no new protocol class needed).
- [ ] Typecheck / lint passes.

---

### US-004: Tiled encode for large video frames

**As a** Python developer generating high-resolution video,
**I want** `vae_encode_batch_tiled(vae, images, tile_size, overlap)`,
**so that** I can encode large PIL frames into a batched latent without OOM.

**Acceptance Criteria:**
- [ ] Signature: `vae_encode_batch_tiled(vae, images, tile_size=512, overlap=64)`.
- [ ] Each image is encoded individually via `vae.encode_tiled(...)` and results are
      concatenated along the batch dimension into a single `"samples"` tensor.
- [ ] Raises `ValueError` when `images` is an empty list.
- [ ] `_VaeEncoderTiled` protocol is reused (no new protocol class needed).
- [ ] Typecheck / lint passes.

---

### US-005: Public API export

**As a** Python developer,
**I want** to import all four batch functions from `pycomfy.vae`,
**so that** my import lines are consistent with the rest of pycomfy's API.

**Acceptance Criteria:**
- [ ] `from pycomfy.vae import vae_decode_batch, vae_encode_batch` works without error.
- [ ] `from pycomfy.vae import vae_decode_batch_tiled, vae_encode_batch_tiled` works
      without error.
- [ ] All four names are added to `__all__` in `pycomfy/vae.py`.
- [ ] All four names are imported in `pycomfy/__init__.py` and added to its `__all__`
      (matching the existing pattern for `vae_decode_tiled` / `vae_encode_tiled`).
- [ ] `from pycomfy import vae_decode_batch, vae_encode_batch, vae_decode_batch_tiled, vae_encode_batch_tiled`
      works without error.
- [ ] Existing `vae_decode`, `vae_decode_tiled`, `vae_encode`, `vae_encode_tiled` remain
      unchanged and continue to pass their tests.
- [ ] Typecheck / lint passes.

---

### US-006: Tests for batch functions

**As a** maintainer,
**I want** pytest tests covering the four new batch functions on CPU,
**so that** CI can validate correctness without a GPU.

**Acceptance Criteria:**
- [ ] `vae_decode_batch` tested with a mock VAE and a fake 4-D tensor → list of 1+ PIL
      images.
- [ ] `vae_decode_batch` tested with a fake 5-D tensor → correct frame count.
- [ ] `vae_encode_batch` tested with a list of small PIL images → dict with `"samples"`.
- [ ] `vae_encode_batch` raises `ValueError` on empty list.
- [ ] `vae_decode_batch_tiled` tested with a mock tiled VAE.
- [ ] `vae_encode_batch_tiled` tested with a mock tiled VAE.
- [ ] All existing VAE tests still pass (`uv run pytest` green).

## Functional Requirements

- FR-1: `vae_decode_batch(vae, latent)` — accepts a `_VaeDecoder` duck-type and a
  ComfyUI LATENT dict; auto-detects 4-D vs 5-D `"samples"` tensor; returns
  `list[PIL.Image]`.
- FR-2: `vae_encode_batch(vae, images)` — accepts a `_VaeEncoder` duck-type and
  `list[PIL.Image]`; raises `ValueError` on empty list; returns a ComfyUI LATENT dict.
- FR-3: `vae_decode_batch_tiled(vae, latent, tile_size=512, overlap=64)` — same as FR-1
  but uses `vae.decode_tiled(...)` per frame; accepts a `_VaeDecoderTiled` duck-type.
- FR-4: `vae_encode_batch_tiled(vae, images, tile_size=512, overlap=64)` — same as FR-2
  but uses `vae.encode_tiled(...)` per image; accepts a `_VaeEncoderTiled` duck-type.
- FR-5: All four functions must defer `import torch` to call time (lazy import pattern).
  No `comfy.*` imports at module top level or inside these functions (consistent with
  the current `vae.py` design).
- FR-6: All four names exported in `pycomfy/vae.py`'s `__all__`.
- FR-7: `vae_decode_batch` and `vae_decode_batch_tiled` must call `detach()` and `cpu()`
  on each decoded image slice before passing to `_tensor_like_to_pil` (GPU-safety,
  matching existing `vae_decode` / `vae_decode_tiled` logic).

## Non-Goals (Out of Scope)

- No `vae_decode_video` / `vae_encode_video` high-level abstractions — callers compose
  the batch functions themselves.
- No automatic chunking of very large batches (caller controls batch size).
- No changes to `ModelManager`, `conditioning.py`, `sampling.py`, `lora.py`, or any
  module other than `pycomfy/vae.py`, `pycomfy/__init__.py`, and the test file.

## Open Questions

- None. Shape auto-detection strategy (len(shape) == 5 ↔ video) confirmed; tiled
  variants included in scope.
