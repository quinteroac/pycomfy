# Requirement: Tiled VAE Encode and Decode

## Context
Running VAE decode or encode on large images requires loading the full image into GPU VRAM at once, causing out-of-memory (OOM) errors for high-resolution outputs. ComfyUI's VAE objects expose `decode_tiled` and `encode_tiled` methods that process the image in overlapping tiles to stay within VRAM budgets. This iteration wraps those methods as first-class pycomfy public functions — mirroring the signatures of `vae_decode` / `vae_encode` — so callers can opt into tiled processing with a single function swap.

## Goals
- Expose `vae_decode_tiled(vae, latent, tile_size=512, overlap=64) -> PIL.Image.Image` as a public pycomfy function.
- Expose `vae_encode_tiled(vae, image, tile_size=512, overlap=64) -> dict[str, Any]` as a public pycomfy function.
- Both functions re-exported from `pycomfy.__init__` alongside `vae_decode` and `vae_encode`.
- All tests pass on CPU-only environments (no GPU required in CI).

## User Stories

### US-001: Tiled VAE Decode
**As a** Python developer using pycomfy, **I want** to call `vae_decode_tiled(vae, latent, tile_size, overlap)` **so that** I can decode large latents into PIL images without running out of GPU memory.

**Acceptance Criteria:**
- [ ] `vae_decode_tiled` is importable directly from `pycomfy` (`from pycomfy import vae_decode_tiled`).
- [ ] `vae_decode_tiled` is importable from `pycomfy.vae` (`from pycomfy.vae import vae_decode_tiled`).
- [ ] Calling `vae_decode_tiled(vae, latent)` with default parameters returns a valid `PIL.Image.Image`.
- [ ] Calling `vae_decode_tiled(vae, latent, tile_size=64, overlap=8)` on a CPU-compatible mock VAE returns a `PIL.Image.Image` without error.
- [ ] The function delegates to `vae.decode_tiled(samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)`.
- [ ] PIL↔tensor conversion reuses existing internal helpers (`_tensor_like_to_pil`).
- [ ] No `torch` or `comfy.*` imports at module top level (lazy import pattern respected).
- [ ] Type hint signature: `def vae_decode_tiled(vae: _VaeDecoderTiled, latent: Mapping[str, Any], tile_size: int = 512, overlap: int = 64) -> Image.Image`.
- [ ] Typecheck / lint passes.

### US-002: Tiled VAE Encode
**As a** Python developer using pycomfy, **I want** to call `vae_encode_tiled(vae, image, tile_size, overlap)` **so that** I can encode large PIL images into latent dicts without running out of GPU memory.

**Acceptance Criteria:**
- [ ] `vae_encode_tiled` is importable directly from `pycomfy` (`from pycomfy import vae_encode_tiled`).
- [ ] `vae_encode_tiled` is importable from `pycomfy.vae` (`from pycomfy.vae import vae_encode_tiled`).
- [ ] Calling `vae_encode_tiled(vae, image, tile_size=64, overlap=8)` on a CPU-compatible mock VAE returns a `dict` with a `"samples"` key.
- [ ] The function delegates to `vae.encode_tiled(pixels, tile_x=tile_size, tile_y=tile_size, overlap=overlap)`.
- [ ] PIL→tensor conversion reuses existing internal helper (`_image_to_tensor_like`).
- [ ] No `torch` or `comfy.*` imports at module top level.
- [ ] Type hint signature: `def vae_encode_tiled(vae: _VaeEncoderTiled, image: Image.Image, tile_size: int = 512, overlap: int = 64) -> dict[str, Any]`.
- [ ] Typecheck / lint passes.

## Functional Requirements
- FR-1: `vae_decode_tiled` must accept the same `latent` dict shape as `vae_decode` (`{"samples": tensor}`).
- FR-2: For 2D VAE (standard still images — the only target of this iteration), `vae.decode_tiled` returns the same `(batch, H, W, C)` shape as `vae.decode` (confirmed in `comfy/sd.py` — both end with `.movedim(1, -1)`). `vae_decode_tiled` must therefore apply the same `detach`, `cpu`, and `[0]` batch-index extraction as `vae_decode`. The 5D reshape guard present in `vae_decode` (for 3D/video VAE) is **not required** in `vae_decode_tiled` for this iteration, as 3D VAE support is deferred to it_009.
- FR-3: `vae_encode_tiled` must return a dict `{"samples": ...}` identical in shape to `vae_encode`'s return value.
- FR-4: Both functions must be added to `pycomfy/vae.py`'s `__all__`.
- FR-5: Both functions must be added to `pycomfy/__init__.py`'s imports and `__all__`.
- FR-6: New Protocol classes `_VaeDecoderTiled` and `_VaeEncoderTiled` must be defined in `pycomfy/vae.py` with `decode_tiled` and `encode_tiled` method stubs respectively, following the duck-typing pattern of existing `_VaeDecoder` / `_VaeEncoder`.
- FR-7: Tests must use a reduced `tile_size` (e.g. `64`) and small test image so they run without GPU.
- FR-8: `tile_size` and `overlap` parameters must have documented defaults (`512` and `64` respectively) in the function docstring.

## Non-Goals (Out of Scope)
- Batch/video tiled VAE variants (`vae_decode_batch`, `vae_encode_batch`) — planned for iteration 009.
- TAESD-specific tiled decode — not part of this iteration.
- Inpaint tiled VAE (`VAEEncodeForInpaint`) — out of scope.
- Updating `PROJECT_CONTEXT.md` — deferred to the Refactor phase.
- Any changes outside `pycomfy/vae.py`, `pycomfy/__init__.py`, and `tests/`.

## Open Questions
- None — implementation path is clear from the existing `vae.py` patterns.
