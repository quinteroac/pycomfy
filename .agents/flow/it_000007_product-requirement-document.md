# Requirement: VAE Encode + Standalone Model Loaders

## Context

Iteration 007 completes the low-level building blocks needed before the high-level `ImagePipeline` (it_008) can support img2img and Flux workflows. Two capabilities are missing:

1. **`vae_encode`** — the inverse of `vae_decode`. Converts a PIL image into a ComfyUI-compatible `{"samples": tensor}` latent dict that `sample()` already consumes.
2. **Standalone model loaders** (`load_vae`, `load_clip`, `load_unet` on `ModelManager`) — currently the only way to obtain a VAE/CLIP/UNet is through `load_checkpoint`, which loads all three together. Flux and other architectures ship UNet, CLIP, and VAE as separate files and require individual loaders.

## Goals

- Allow developers to encode a PIL image into a latent tensor (enabling img2img pipelines).
- Allow developers to load VAE, CLIP, and UNet objects independently from separate checkpoint files.
- Maintain CPU-safe import behaviour — no torch/comfy side effects at module import time.
- Keep `vae_encode` accessible from both `pycomfy.vae` and `pycomfy` (top-level re-export), matching the pattern established by `vae_decode`.

## User Stories

### US-001: Encode PIL image to latent

**As a** Python developer (or internal pycomfy module), **I want** to call `vae_encode(vae, image)` with a PIL `Image` and receive a `{"samples": tensor}` dict **so that** I can feed the result directly into `sample()` for img2img inference.

**Acceptance Criteria:**
- [ ] `vae_encode(vae, image)` accepts a `PIL.Image.Image` and a VAE object; returns `dict` with key `"samples"` whose value is a tensor.
- [ ] The returned dict format is identical to the `latent` dict consumed by `sample()`.
- [ ] The function is defined in `pycomfy/vae.py` and added to `__all__`.
- [ ] `from pycomfy.vae import vae_encode` works on CPU-only with no torch/comfy import at module level.
- [ ] `from pycomfy import vae_encode` works (re-exported from `__init__.py`).
- [ ] Unit test (mock VAE): `vae_encode(mock_vae, pil_image)` returns `{"samples": mock_tensor}`.
- [ ] Typecheck / lint passes (`ruff check . && mypy pycomfy/` no new violations).

### US-002: Round-trip VAE smoke test

**As a** developer, **I want** `vae_decode(vae, vae_encode(vae, image))` to return a `PIL.Image.Image` without errors **so that** I can verify the encode→decode pipeline is coherent end-to-end on CPU.

**Acceptance Criteria:**
- [ ] A test exercises `vae_decode(mock_vae, vae_encode(mock_vae, pil_image))` on CPU (mocked VAE).
- [ ] The test completes without raising; the return value is an instance of `PIL.Image.Image`.
- [ ] `uv run pytest` passes on CPU-only CI.

### US-003: Load standalone VAE

**As a** Python developer, **I want** to call `manager.load_vae(path)` **so that** I can obtain a VAE object usable by both `vae_encode` and `vae_decode` without loading a full checkpoint.

**Acceptance Criteria:**
- [ ] `ModelManager.load_vae(path: str | Path)` is implemented and added to `ModelManager`.
- [ ] The method calls the appropriate ComfyUI internal loader (grep-verified against `comfy/sd.py` before implementation).
- [ ] Returns the raw comfy VAE object.
- [ ] Raises `FileNotFoundError` if `path` does not point to an existing file.
- [ ] Unit test (mock): confirms the correct comfy internal is called with the resolved path.
- [ ] Typecheck / lint passes.

### US-004: Load standalone CLIP

**As a** Python developer, **I want** to call `manager.load_clip(path)` **so that** I can obtain a CLIP object usable by `encode_prompt` when not loading a full checkpoint.

**Acceptance Criteria:**
- [ ] `ModelManager.load_clip(path: str | Path)` is implemented and added to `ModelManager`.
- [ ] The method calls the appropriate ComfyUI internal loader (grep-verified against `comfy/sd.py` before implementation).
- [ ] Returns the raw comfy CLIP object.
- [ ] Raises `FileNotFoundError` if `path` does not point to an existing file.
- [ ] Unit test (mock): confirms the correct comfy internal is called with the resolved path.
- [ ] Typecheck / lint passes.

### US-005: Load standalone UNet

**As a** Python developer, **I want** to call `manager.load_unet(path)` **so that** I can obtain a model/UNet object usable by `sample()` without loading a full checkpoint.

**Acceptance Criteria:**
- [ ] `ModelManager.load_unet(path: str | Path)` is implemented and added to `ModelManager`.
- [ ] The method calls `comfy.sd.load_diffusion_model(str(path))` directly — **not** the deprecated `comfy.sd.load_unet` (which emits a deprecation warning).
- [ ] Returns the raw comfy model object.
- [ ] Raises `FileNotFoundError` if `path` does not point to an existing file.
- [ ] Unit test (mock): confirms the correct comfy internal is called with the resolved path.
- [ ] Typecheck / lint passes.

### US-006: No regression on existing ModelManager

**As a** developer, **I want** `ModelManager` and `load_checkpoint` to continue working exactly as before **so that** existing code built on it_002–006 is not broken.

**Acceptance Criteria:**
- [ ] All pre-existing tests for `ModelManager.load_checkpoint` still pass.
- [ ] `from pycomfy.models import ModelManager` works without error on CPU-only.
- [ ] `uv run pytest` passes entirely.

## Functional Requirements

- **FR-1:** `vae_encode(vae, image: PIL.Image.Image) -> dict[str, Any]` — defined in `pycomfy/vae.py`, lazily imports torch/comfy inside the function body.
- **FR-2:** `vae_encode` converts the PIL image to a normalised float tensor `[0, 1]` in NHWC format before calling `vae.encode(...)`, matching ComfyUI's `VAEEncode` node internals (grep `VAEEncode` in `vendor/ComfyUI/nodes.py` for reference).
- **FR-3:** The returned dict has the shape `{"samples": tensor}` where tensor is the output of `vae.encode(pixels)` — same schema as the latent dicts produced by `sample()`.
- **FR-4:** `vae_encode` is added to `pycomfy/__init__.py` re-exports and `pycomfy/vae.py` `__all__`, following the same pattern as `vae_decode`.
- **FR-5:** `ModelManager.load_vae(path: str | Path) -> Any` — calls the appropriate `comfy.sd` loader; raises `FileNotFoundError` on missing file; lazy comfy import inside the method.
- **FR-6:** `ModelManager.load_clip(path: str | Path) -> Any` — calls the appropriate `comfy.sd` loader; raises `FileNotFoundError` on missing file; lazy comfy import inside the method.
- **FR-7:** `ModelManager.load_unet(path: str | Path) -> Any` — calls `comfy.sd.load_diffusion_model(str(path))` directly (NOT `comfy.sd.load_unet`, which is deprecated and emits a warning); raises `FileNotFoundError` on missing file; lazy comfy import inside the method.
- **FR-8:** All three standalone loaders accept `str | Path`; the path is resolved to absolute before passing to ComfyUI internals.
- **FR-9:** No `torch`, `comfy.*`, or `ensure_comfyui_on_path()` calls at module top level in `vae.py` or `models.py` — all deferred to function/method body call time.
- **FR-10:** Before implementing, grep `VAEEncode`, `load_vae`, `load_clip`, `load_unet` in `vendor/ComfyUI/nodes.py` and `vendor/ComfyUI/comfy/sd.py` to identify the exact internal APIs to call.

## Non-Goals (Out of Scope)

- Tiled VAE encode (no tiling support in this iteration).
- Batch encode (single image only).
- CLIP variant handling for Flux dual-encoder (deferred to it_008 or later).
- UNet variant detection / dtype casting (deferred).
- Any changes to `sampling.py`, `conditioning.py`, `lora.py`, or `_runtime.py`.
- High-level pipeline API (deferred to it_008).

## Open Questions

- ~~ComfyUI internal API selection~~ — **Resolved** (grep run 2026-03-07):
  - `vae_encode`: PIL → `np.array().astype(float32) / 255.0` → `torch.from_numpy()[None,]` → `vae.encode(pixels)` → `{"samples": t}` (mirrors `VAEEncode` + `LoadImage` in `nodes.py`).
  - `load_vae`: `comfy.utils.load_torch_file(path, return_metadata=True)` → `comfy.sd.VAE(sd=sd, metadata=metadata)` → `.throw_exception_if_invalid()`.
  - `load_clip`: `comfy.sd.load_clip(ckpt_paths=[str(path)], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)`.
  - `load_unet`: `comfy.sd.load_diffusion_model(str(path))` — `comfy.sd.load_unet` is deprecated.
