# Audit Report — Iteration 000035

## Executive Summary

All six user stories and all eight functional requirements comply. The three ACE Step 1.5 pipeline modules (`checkpoint`, `split`, `split_4b`), the `vae_decode_audio` primitive, package scaffolding, and the CPU-only pytest suite are all correctly implemented with no blocking gaps. Two minor observations around path-resolution asymmetry and the `vae_decode_audio` return contract are noted but do not block approval.

---

## Verification by FR

| FR | Assessment | Notes |
|---|---|---|
| FR-1 | comply | `vae_decode_audio` present in `audio.py`, calls `vae.decode(latent["samples"]).movedim(-1, 1)`, listed in `__all__` |
| FR-2 | comply | All three modules declare `__all__ = ["manifest", "run"]` only |
| FR-3 | comply | `split`/`split_4b` derive paths from `manifest()` inside `run()`; `checkpoint` shares `_CHECKPOINT_DEST` constant between `manifest()` and `run()` |
| FR-4 | comply | `steps=8`, `cfg=1.0`, `sampler_name="euler"`, `scheduler="simple"` confirmed as defaults in all three pipelines |
| FR-5 | comply | `model_sampling_aura_flow(model, shift=3)` applied in all three pipelines |
| FR-6 | comply | All `comfy.*` and `torch` imports inside `run()` are lazy local imports |
| FR-7 | comply | No top-level `torch` or `comfy.*` imports in any new pipeline file or `__init__.py` |
| FR-8 | comply | `models_dir: str \| Path` annotation used in all three `run()` signatures |

---

## Verification by US

| US | Assessment | Notes |
|---|---|---|
| US-001 | comply | `vae_decode_audio` added, correct implementation, in `__all__`, no top-level comfy imports |
| US-002 | comply | `checkpoint.py` created, one-entry manifest, correct `run()` signature, correct node order, correct return shape, correct defaults |
| US-003 | comply | `split.py` created, four-entry manifest (UNet + 0.6B + 1.7B + VAE), correct loading and sampling order |
| US-004 | comply | `split_4b.py` created, four-entry manifest (UNet + 0.6B + 4B + VAE), correct loading and sampling order |
| US-005 | comply | `__init__.py` files present for all three sub-package levels; `pipelines/__init__.py` lists `"audio"` in `__all__` |
| US-006 | comply | Test file created, covers manifest shape/filenames/dests, mocked `run()` return shape, default param values, runtime error handling |

---

## Minor Observations

1. **Path-resolution asymmetry**: `checkpoint.py` passes an absolute path (`models_dir / _CHECKPOINT_DEST`) to `mm.load_checkpoint()`, while `split.py` and `split_4b.py` pass only the filename stem (`dest.name`) to `mm.load_unet/load_clip/load_vae`. Presumably correct per the `ModelManager` API contract, but the inconsistency could surprise future pipeline authors.

2. **`vae_decode_audio` return contract**: Unlike `ltxv_audio_vae_decode` (which returns `{"waveform": ..., "sample_rate": ...}`), `vae_decode_audio` returns only the waveform tensor; callers must fetch `sample_rate` separately via `getattr(vae, "audio_sample_rate", 44100)`. Consistent across all three pipelines but diverges from the LTXV audio helper convention.

3. **Test patch strategy**: Patches target source module attributes (e.g. `comfy_diffusion.audio.vae_decode_audio`) rather than the pipeline's local namespace. This is correct because all imports are lazy, but should be documented to prevent breakage if any import is ever hoisted to module level.

---

## Conclusions and Recommendations

The iteration 000035 prototype is fully compliant with its PRD. All FRs and USs are assessed as **comply**. The implementation is clean, follows the established pipeline pattern, and the test coverage is thorough.

The two minor observations (path-resolution asymmetry and `vae_decode_audio` return contract) do not block approval. They are recorded for optional future standardisation.

**Decision: Approve as-is. Proceed to next phase.**

---

## Refactor Plan

No refactor required. All compliance checks passed and no blocking issues were found. The minor observations may be addressed opportunistically in a future iteration if the `ModelManager` API or audio helper conventions are revisited.
