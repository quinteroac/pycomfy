# Requirement: LoRA Loading and Stacking

## Context

After sampling (it_04) and VAE decoding (it_05), the inference pipeline is functionally
complete for base checkpoints. However, production workflows almost universally rely on
LoRA weights to steer style, character, or concept without full fine-tuning. This iteration
adds `pycomfy/lora.py` which exposes `apply_lora()` — a thin, CPU-safe wrapper around
`comfy.sd.load_lora_for_models` that loads a LoRA file from disk and returns patched
model+CLIP copies without mutating the originals. Multiple LoRAs are composed by chaining
`apply_lora` calls. The module follows the exact same design pattern established by
`sampling.py` and `vae.py`: lazy ComfyUI import, typed public API, `__all__` declared,
re-exported from `pycomfy.__init__`.

## Goals

- Expose `apply_lora(model, clip, path, strength_model, strength_clip) -> tuple[Any, Any]`
  in `pycomfy/lora.py`.
- Return patched copies of model and CLIP — originals are never mutated.
- Match the lazy-import and CPU-safe patterns of every prior pycomfy module.
- Support LoRA stacking via simple call chaining (no extra API needed).
- Be directly consumable by `it_07 (ImagePipeline)` with no adapter code.

## User Stories

### US-001: Apply a single LoRA to a model+CLIP pair

**As a** Python developer (or a higher-level pycomfy module),
**I want** to call `apply_lora(model, clip, "/path/to/lora.safetensors", 0.8, 0.8)`,
**so that** I receive patched `(model, clip)` copies with the LoRA weights applied and the
originals unchanged.

**Acceptance Criteria:**

- [ ] `pycomfy/lora.py` exists and exports `apply_lora` via `__all__ = ["apply_lora"]`.
- [ ] `apply_lora(model, clip, path, strength_model, strength_clip)` accepts:
  - `model`: ComfyUI model patcher object (as returned by `ModelManager.load_checkpoint`)
  - `clip`: ComfyUI CLIP object (as returned by `ModelManager.load_checkpoint`)
  - `path`: `str | Path` — absolute path to the LoRA file on disk
  - `strength_model`: `float` — scale applied to model LoRA weights
  - `strength_clip`: `float` — scale applied to CLIP LoRA weights
- [ ] The function returns a `tuple[Any, Any]` of `(patched_model, patched_clip)`.
- [ ] The original `model` and `clip` objects are not mutated (ComfyUI's internal `.clone()`
  guarantees this — implementer must verify at `vendor/ComfyUI/comfy/sd.py:76`).
- [ ] `apply_lora` is importable as `from pycomfy import apply_lora`
  (re-exported from `pycomfy/__init__.py`).
- [ ] No `comfy.*` or `torch` import occurs at module level — all ComfyUI imports deferred
  to call time via `ensure_comfyui_on_path()` from `pycomfy._runtime`.
- [ ] Typecheck / lint passes (`ruff check .` and `mypy pycomfy/` produce no new violations).

---

### US-002: Stack multiple LoRAs via chaining

**As a** Python developer,
**I want** to chain multiple `apply_lora` calls using the output of one as the input to the
next,
**so that** I can compose any number of LoRA weights without a special multi-LoRA API.

**Acceptance Criteria:**

- [ ] The following pattern produces a correctly double-patched model+CLIP:
  ```python
  model, clip = apply_lora(model, clip, "style.safetensors", 0.8, 0.8)
  model, clip = apply_lora(model, clip, "character.safetensors", 0.6, 0.6)
  ```
- [ ] Each intermediate result is independent — applying the second LoRA does not
  affect the first patched copy in any observable way (verified by mock test).
- [ ] No additional API surface is required; the test demonstrates the chaining pattern.

---

### US-003: pytest coverage for apply_lora

**As a** developer maintaining pycomfy,
**I want** pytest tests that validate `apply_lora` behavior without GPU or real checkpoints,
**so that** regressions are caught in CPU-only CI.

**Acceptance Criteria:**

- [ ] `tests/test_lora.py` exists.
- [ ] The test imports `apply_lora` from `pycomfy` (public surface only, not internals).
- [ ] A mock test verifies that `comfy.sd.load_lora_for_models` is called with the correct
  arguments (model, clip, loaded state dict, strength_model, strength_clip).
- [ ] A mock test verifies that `comfy.utils.load_torch_file` is called with the given
  `path` and `safe_load=True`.
- [ ] A mock test verifies that chaining two `apply_lora` calls results in
  `comfy.sd.load_lora_for_models` being called twice with the correct intermediate objects.
- [ ] All existing tests continue to pass (`uv run pytest`).
- [ ] Tests run on CPU without loading any real checkpoint or LoRA file.

---

## Functional Requirements

- **FR-1:** `apply_lora(model: Any, clip: Any, path: str | os.PathLike, strength_model: float, strength_clip: float) -> tuple[Any, Any]` — public function in `pycomfy/lora.py`.
- **FR-2:** The function must call `ensure_comfyui_on_path()` (from `pycomfy._runtime`) and
  defer all ComfyUI imports (`comfy.utils`, `comfy.sd`) to call time.
- **FR-3:** The LoRA state dict is loaded via `comfy.utils.load_torch_file(path, safe_load=True)`,
  mirroring the pattern in `vendor/ComfyUI/nodes.py:709` (`LoraLoader.load_lora`).
  Implementer must inspect lines 696–713 to confirm the exact call contract.
- **FR-4:** The patching is performed via `comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)`
  (at `vendor/ComfyUI/comfy/sd.py:76`). This function clones model and CLIP internally —
  originals are not mutated. Implementer must inspect lines 76–100 to confirm.
- **FR-5:** The function returns `(model_lora, clip_lora)` — the tuple returned directly by
  `comfy.sd.load_lora_for_models`.
- **FR-6:** `pycomfy/__init__.py` must re-export `apply_lora`
  (`from pycomfy.lora import apply_lora`) so consumers can write
  `from pycomfy import apply_lora`.
- **FR-7:** The module must declare `__all__ = ["apply_lora"]`.

## Non-Goals (Out of Scope)

- No caching or memoisation of loaded LoRA state dicts (unlike the `LoraLoader` node which caches `self.loaded_lora`).
- No model-only LoRA application (`LoraLoaderModelOnly` pattern) — not needed for it_07.
- No unloading / reverting LoRA patches — not needed for this iteration.
- No validation of LoRA file format or key coverage warnings — ComfyUI handles this internally.
- No multi-LoRA convenience API (e.g. `apply_loras(model, clip, lora_list)`) — chaining is sufficient.

## Open Questions

- None — all design decisions confirmed during requirements interview.
