# Requirement: VAE Decode — Latent to PIL Image

## Context

After the sampling step (iteration 04), the output is a ComfyUI LATENT dict containing a
raw latent tensor. To produce a usable image, the latent must be decoded through the VAE
model. This module provides `vae_decode()` as the canonical bridge between the sampler
output and a `PIL.Image`, completing the txt2img path and serving as a building block for
the high-level `ImagePipeline` in iteration 07.

## Goals

- Expose a single, clean `vae_decode(vae, latent) -> PIL.Image` function in `pycomfy/vae.py`.
- Match the pattern established by `sampling.py`: lazy ComfyUI import, typed public API,
  minimal surface, `__all__` declared.
- Work correctly in CPU-only environments so CI passes without a GPU.
- Be composable: `it_07 (ImagePipeline)` must be able to call `vae_decode()` directly
  with no adapter or wrapper code.

## User Stories

### US-001: Decode a latent tensor to a PIL image

**As a** Python developer (or a higher-level pycomfy module),
**I want** to call `vae_decode(vae, latent)` with the VAE object from `CheckpointResult`
and the LATENT dict from `sample()`,
**so that** I receive a `PIL.Image.Image` I can save, display, or pass downstream.

**Acceptance Criteria:**

- [ ] `pycomfy/vae.py` exists and exports `vae_decode` via `__all__`.
- [ ] `vae_decode(vae, latent)` accepts the ComfyUI VAE object (from `CheckpointResult.vae`)
      and the LATENT dict (as returned by `pycomfy.sample()`).
- [ ] The function returns a `PIL.Image.Image` instance.
- [ ] Pixel values are in the expected 0–255 uint8 range (image is not washed-out or clipped).
- [ ] The function is importable as `from pycomfy import vae_decode` (re-exported from `__init__.py`).
- [ ] `pycomfy/vae.py` is import-safe at module level in a CPU-only environment (no GPU
      import side-effects at import time — lazy ComfyUI import pattern).
- [ ] `uv run pytest` passes in CI (CPU-only, no GPU required).
- [ ] Typecheck / lint passes (PEP 8, type hints on the public function signature).

---

### US-002: pytest coverage for vae_decode

**As a** developer maintaining pycomfy,
**I want** at least one pytest test that validates `vae_decode` produces a correct
`PIL.Image` from a synthetic or mocked latent,
**so that** regressions are caught in CI without needing a GPU or a real checkpoint.

**Acceptance Criteria:**

- [ ] `tests/test_vae.py` (or equivalent) exists.
- [ ] The test imports `vae_decode` from `pycomfy` (public surface, not internals).
- [ ] The test runs on CPU without loading a real checkpoint
      (mock/stub the VAE object or use a minimal synthetic tensor).
- [ ] All existing tests continue to pass (`uv run pytest`).

---

## Functional Requirements

- **FR-1:** `vae_decode(vae: Any, latent: Any) -> PIL.Image.Image` — public function in
  `pycomfy/vae.py`.
- **FR-2:** The function must use `ensure_comfyui_on_path()` (from `pycomfy._runtime`)
  and defer all ComfyUI imports to call time (not module import time).
- **FR-3:** The LATENT input follows the ComfyUI contract: a dict with a `"samples"` key
  holding a `torch.Tensor`. The canonical ComfyUI call pattern is
  `vae.decode(latent["samples"])` — verified at `vendor/ComfyUI/nodes.py:315`
  (`VAEDecode` node). Implementer must inspect that line and the surrounding class
  (lines 293–319) to confirm the exact tensor expectations before coding.
- **FR-4:** The PIL image output must be RGB mode with pixel values in the 0–255 uint8
  range (not normalised floats).
- **FR-5:** `pycomfy/__init__.py` **must** re-export `vae_decode`
  (`from pycomfy.vae import vae_decode`) so consumers can write
  `from pycomfy import vae_decode`. This differs from `encode_prompt` and `ModelManager`
  (it_02/it_03), which are **not** re-exported from `__init__.py` — the implementer must
  add this line explicitly.
- **FR-6:** The module must declare `__all__ = ["vae_decode"]`.

## Non-Goals (Out of Scope)

- **VAE encode** (`image → latent`) — deferred to a later iteration (img2img / inpainting).
- **VAE tiling** — large-image OOM mitigation is out of scope for this iteration.
- **Batch decoding** (multiple latents in one call) — single-image output only for MVP.
- **Format options** (RGBA, grayscale, saving to disk) — callers handle post-processing.
- **Custom VAE loading** — VAE is always obtained from `CheckpointResult.vae`; no
  standalone VAE file loading in this iteration.

## Open Questions

- ~~Should `vae_decode` accept raw `torch.Tensor` directly (in addition to the LATENT dict)?~~
  **Resolved:** LATENT dict only. Keeps the signature consistent with `sample()` output
  and avoids a second overload. Revisit if `ImagePipeline` needs a tensor shortcut in it_07.
