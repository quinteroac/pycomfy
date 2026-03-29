# Audit Report — Iteration 000026

## Executive Summary

All 6 new LTX-Video pipeline files (ltx2_t2v_distilled, ltx2_i2v, ltx2_i2v_distilled, ltx2_i2v_lora, ltx3_t2v, ltx3_i2v) are implemented and 165 CPU tests pass. Documentation is thorough: every pipeline has a module-level Usage section, a fully documented `run()` docstring, and SMOKE_TEST.md entries. Two FRs are non-compliant: **FR-4** (`ltxv_empty_latent_video()` lacks the optional `fps` parameter) and **FR-6** (`pipelines/__init__.py` contains no module listing). Additionally, the user has requested the pipelines be reorganised to mirror the `comfyui_official_workflows/video/ltx/{ltx2,ltx3}/` folder hierarchy.

---

## Verification by FR

| FR | Assessment | Evidence |
|---|---|---|
| FR-1 | ✅ comply | All 6 files in `comfy_diffusion/pipelines/` with correct snake_case names |
| FR-2 | ✅ comply | Module-level `_*_DEST` constants used by both `manifest()` and `run()` path resolution |
| FR-3 | ✅ comply | All `torch`, `comfy.*`, `comfy_diffusion.*` imports deferred inside function bodies; 6/6 no_top_level_comfy_imports tests pass |
| FR-4 | ❌ does_not_comply | `ltxv_empty_latent_video()` signature is `(width, height, length=97, batch_size=1)` — no `fps` parameter added |
| FR-5 | ✅ comply | All path annotations are `str \| Path`; no `os.PathLike` usage found |
| FR-6 | ❌ does_not_comply | `pipelines/__init__.py` is an empty docstring stub; no pipeline modules listed or registered |

---

## Verification by US

| US | Assessment | Notes |
|---|---|---|
| US-001 | ✅ comply | `manifest()` → 3 entries; `steps=8`; upsampler before VAE decode; CPU tests pass |
| US-002 | ✅ comply | `manifest()` → 4 entries; `ltxv_preprocess` before `ltxv_img_to_video_inplace`; `apply_lora` called; CPU tests pass |
| US-003 | ✅ comply | `manifest()` → 3 entries; mirrors `ltx2_i2v` signature; `steps=8`; no `apply_lora`; CPU tests pass |
| US-004 | ✅ comply | `manifest()` → 4 entries (style LoRA excluded); dual `apply_lora` (base first); `width=1280, height=1280`; CPU tests pass |
| US-005 | ✅ comply | `manifest()` → 3 entries; signature matches `ltx2_t2v_distilled`; `steps=8`; CPU tests pass |
| US-006 | ⚠️ partially_comply | `fps` accepted and documented as reserved; not forwarded (FR-4 gap); AC03 says "else documented as reserved" — satisfied documentarily |
| US-007 | ✅ comply | 165 tests pass; all mock `ModelManager` and `check_runtime`; no GPU required |
| US-009 | ✅ comply | All 6 pipelines have Usage docstring + full `run()` param docs + SMOKE_TEST.md entries |

---

## Minor Observations

1. `ltx2_i2v_lora.py`: base distilled LoRA strength hardcoded to `1.0` — not user-configurable. A `base_lora_strength` param could be exposed as a future improvement.
2. `ltx3_i2v.py`: `fps != 24` is silently ignored at runtime. A `warnings.warn` would improve DX until FR-4 is resolved.
3. `pipelines/__init__.py` design ("The package itself exports nothing") conflicts with FR-6. A registry (module-level `__all__` listing subpackages or a `PIPELINES` dict) is needed.
4. `ltx2_i2v.py` / `ltx2_i2v_distilled.py`: duck-typing heuristic `hasattr(image, 'mode')` could fail for non-standard image objects; a deferred `isinstance(image, PIL.Image.Image)` would be more robust.
5. All 6 pipelines call `check_runtime()` at the top of `run()` — correct pattern, but note that `ModelManager(models_dir)` is instantiated before any file-presence check.

---

## Conclusions and Recommendations

The iteration is **substantially complete** (6/6 pipelines, 165/165 tests). Two targeted fixes are required:

1. **FR-4** — Add `fps: int = 24` to `ltxv_empty_latent_video()` in `latent.py` and forward it from `ltx3_i2v.run()`. The parameter will be accepted but has no effect on latent shape (reserved for future scheduler/metadata use), consistent with the existing documentation.
2. **FR-6 + folder restructure** — Reorganise `comfy_diffusion/pipelines/` to mirror `comfyui_official_workflows/video/ltx/{ltx2,ltx3}/`, creating sub-packages `pipelines/video/ltx/ltx2/` and `pipelines/video/ltx/ltx3/`. Rename files to drop the model-family prefix (e.g. `ltx2_t2v.py` → `pipelines/video/ltx/ltx2/t2v.py`). Update `__init__.py` files to list submodules. Update all test files and SMOKE_TEST.md to use the new import paths.

---

## Refactor Plan

### 1. Extend `ltxv_empty_latent_video()` (FR-4)
- File: `comfy_diffusion/latent.py`
- Add `fps: int = 24` parameter. Accept and store/document; no latent shape change.
- Forward `fps` from `ltx3_i2v.run()`.

### 2. Restructure pipeline packages (FR-6 + user request)

**New layout:**
```
comfy_diffusion/pipelines/
  __init__.py                 ← list sub-packages
  video/
    __init__.py
    ltx/
      __init__.py
      ltx2/
        __init__.py           ← list ltx2 pipelines
        t2v.py                (was ltx2_t2v.py)
        t2v_distilled.py      (was ltx2_t2v_distilled.py)
        i2v.py                (was ltx2_i2v.py)
        i2v_distilled.py      (was ltx2_i2v_distilled.py)
        i2v_lora.py           (was ltx2_i2v_lora.py)
      ltx3/
        __init__.py           ← list ltx3 pipelines
        t2v.py                (was ltx3_t2v.py)
        i2v.py                (was ltx3_i2v.py)
```

**Steps:**
- `git mv` each pipeline file to new path.
- Create `__init__.py` for each new package (descriptive docstring + `__all__`).
- Update module docstrings (`from comfy_diffusion.pipelines.ltx2_t2v import` → `from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import`).
- Update Usage snippets in docstrings.
- Update all `tests/test_pipelines_*.py` import statements.
- Update `SMOKE_TEST.md` import paths.
- Top-level `pipelines/__init__.py`: add an `__all__` listing the video sub-packages.
