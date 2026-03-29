# Audit Report — Iteration 000028

## Executive Summary

All 9 functional requirements and 9 user stories for iteration 000028 are fully implemented and verified. The codebase delivers three new node wrappers (`image_scale_by`, `dw_preprocessor`, `load_audio`), four complete pipeline modules (`ltx2/canny`, `ltx2/depth`, `ltx2/pose`, `ltx23/ia2v`), four corresponding example scripts, and CPU-only test coverage. The full test suite (1339 tests) passes with zero failures. All modules follow the established lazy-import pattern, export contracts, and architecture conventions.

---

## Verification by FR

| FR ID | Assessment | Notes |
|-------|------------|-------|
| **FR-1** | ✅ comply | `image_scale_by(image, upscale_method, scale_by)` defined in `comfy_diffusion/image.py` (lines 371–382). Floor-rounded dimensions, lazy `comfy.utils` import via `_get_comfy_utils()`, added to `__all__` at line 435. |
| **FR-2** | ✅ comply | `dw_preprocessor(image, detect_hand, detect_body, detect_face, resolution)` defined in `comfy_diffusion/image.py` (lines 384–418). Lazy import via `_get_dw_preprocessor_deps()`, added to `__all__` at line 436. Raises `ImportError` with clear message when `controlnet_aux` is not installed. |
| **FR-3** | ✅ comply | `load_audio(path, start_time, duration)` defined in `comfy_diffusion/audio.py` (lines 229–266). Uses `torchaudio.load()`, trims by `start_time`/`duration`, returns `{"waveform": Tensor[1,C,N], "sample_rate": int}`. Added to `__all__` at line 276. |
| **FR-4** | ✅ comply | All four pipeline files (`canny.py`, `depth.py`, `pose.py`, `ia2v.py`) export exactly `__all__ = ["manifest", "run"]`. No other public symbols. |
| **FR-5** | ✅ comply | `manifest()` in each pipeline lists only models from active (non-bypassed) workflow nodes: canny: 5 entries; depth: 7 entries (incl. Lotus model + SD VAE); pose: 5 entries; ia2v: 5 entries. Bypassed nodes correctly omitted. |
| **FR-6** | ✅ comply | `run()` node execution order mirrors workflow. Canny: `euler_ancestral` + `ManualSigmas` for both passes. Depth: `euler` + `ltxv_scheduler` for Lotus pass, `gradient_estimation` for LTX pass. Pose: `dw_preprocessor` → `euler` + `ltxv_scheduler` → `gradient_estimation`. ia2v: `euler_ancestral_cfg_pp` (pass 1, variable seed), `euler_cfg_pp` (pass 2, fixed seed 42) — all matching reference workflows. |
| **FR-7** | ✅ comply | All four example scripts are self-contained: import `comfy_diffusion` at top, all heavy imports (`torch`, `torchaudio`, `av`, `comfy.*`) deferred inside `main()`. Each calls `download_models(manifest(), ...)` then `run(...)` with parsed CLI args. |
| **FR-8** | ✅ comply | `comfy_diffusion/pipelines/video/ltx/ltx23/__init__.py` line 31: `__all__ = ["t2v", "i2v", "flf2v", "ia2v"]`. No "Not yet implemented" comment present. |
| **FR-9** | ✅ comply | `image_scale_by` and `dw_preprocessor` added to `image.__all__` (lines 435–436). `load_audio` added to `audio.__all__` (line 276). `ia2v` added to `ltx23.__all__` (line 31). |

---

## Verification by US

| US ID | Assessment | Notes |
|-------|------------|-------|
| **US-001** | ✅ comply | `image_scale_by` callable with correct signature, returns IMAGE tensor, dimensions are `floor(h*scale_by) × floor(w*scale_by)`, in `__all__`, lazy-import pattern verified. |
| **US-002** | ✅ comply | `dw_preprocessor` callable with correct signature, returns IMAGE tensor with same batch dimension as input, in `__all__`, lazy-import pattern verified. |
| **US-003** | ✅ comply | `load_audio` callable with `path`, `start_time=0.0`, `duration=None`. Uses `torchaudio`. Returns dict with `waveform [1,C,N]` and `sample_rate int`. In `__all__`. |
| **US-004** | ✅ comply | `comfy_diffusion/pipelines/video/ltx/ltx2/canny.py` present. `manifest()` returns 5 `ModelEntry` objects. `run()` accepts all required kwargs. Sampler `euler_ancestral`, `ManualSigmas`, CFG 3.0/1.0 per workflow spec. |
| **US-005** | ✅ comply | `comfy_diffusion/pipelines/video/ltx/ltx2/depth.py` present. `manifest()` returns 7 entries. `run()` accepts all required kwargs. Lotus pass: `euler` + `ltxv_scheduler`; LTX pass: `gradient_estimation` + `ManualSigmas`. |
| **US-006** | ✅ comply | `comfy_diffusion/pipelines/video/ltx/ltx2/pose.py` present. `manifest()` returns 5 entries. `run()` accepts all required kwargs including optional `first_frame_path`. DWPreprocessor integrated with `detect_body=True`, `detect_hand=True`, `detect_face=False`. |
| **US-007** | ✅ comply | `comfy_diffusion/pipelines/video/ltx/ltx23/ia2v.py` present. `manifest()` returns 5 entries. `run()` accepts all required kwargs including `audio_path`, `audio_start_time`, `audio_duration`. Samplers `euler_ancestral_cfg_pp` (pass 1) and `euler_cfg_pp` (pass 2, fixed seed 42). `ltx23/__init__.py` updated. |
| **US-008** | ✅ comply | All four example scripts present at expected paths. Each accepts `--models-dir`, `--prompt`, and pipeline-specific flags. Heavy imports inside `main()`. Error messages for missing required args. `download_models` + `run` pattern followed. |
| **US-009** | ✅ comply | All test files present and covering new wrappers, audio, all four pipelines, and example scripts. `uv run pytest`: 1339 passed, 0 failures. |

---

## Minor Observations

1. `tests/test_pipelines_ltx2_pose.py` uses `node.s` to access AST string constants. This attribute is deprecated since Python 3.12 and will be removed in Python 3.14 — replace with `node.value`.
2. `controlnet_aux` (required by `dw_preprocessor`) is not declared as an optional extra in `pyproject.toml`. Consider adding a `[pose]` extras group (consistent with the existing `[cuda]`/`[cpu]` pattern) to make the dependency discoverable.
3. The fixed seed `42` in `ia2v.py` pass 2 (line ~400) is a magic constant matching the reference workflow. An inline comment referencing the workflow source would help future maintainers.
4. Lotus-specific `manifest()` entries in `depth.py` (Lotus UNet + SD VAE) are interleaved with LTX entries. Grouping them with a comment would ease maintenance if the Lotus pass is refactored or made optional.

---

## Conclusions and Recommendations

Iteration 000028 is fully compliant with the PRD. All 9 FRs and 9 USs satisfy their acceptance criteria. The 1339-test suite passes cleanly on CPU with only two non-blocking deprecation warnings.

The iteration is ready to merge. Proceed with the following refactor actions:

1. Fix the Python 3.14 deprecation warning in `tests/test_pipelines_ltx2_pose.py`: replace `.s` with `.value` on AST string constant nodes.
2. Add an inline comment in `ia2v.py` next to the fixed seed `42` referencing the source workflow.
3. Add `controlnet_aux` as an optional `[pose]` extra in `pyproject.toml`.
4. Add grouping comments in `depth.py` `manifest()` to separate Lotus-specific entries from LTX-specific entries.

---

## Refactor Plan

### R-1 — Fix AST deprecation in test_pipelines_ltx2_pose.py
- **File:** `tests/test_pipelines_ltx2_pose.py`
- **Change:** Replace `elt.s` with `elt.value` on all AST string constant accesses.
- **Why:** Python 3.14 removes the `.s` attribute from `ast.Constant` nodes. This will cause test failures on the next Python version upgrade.

### R-2 — Document magic seed constant in ia2v.py
- **File:** `comfy_diffusion/pipelines/video/ltx/ltx23/ia2v.py`
- **Change:** Add inline comment `# Fixed seed from reference workflow (LTX 2.3 ia2v)` next to the `seed=42` constant in pass 2.
- **Why:** Prevents future maintainers from assuming this is an arbitrary default.

### R-3 — Declare controlnet_aux as [pose] optional extra
- **File:** `pyproject.toml`
- **Change:** Add `[pose]` optional-dependencies entry with `controlnet_aux>=0.0.20`.
- **Why:** Makes the dependency for `dw_preprocessor` discoverable and installable consistently with the project's existing extras pattern.

### R-4 — Group Lotus entries in depth.py manifest()
- **File:** `comfy_diffusion/pipelines/video/ltx/ltx2/depth.py`
- **Change:** Add `# --- Lotus depth estimation ---` and `# --- LTX 2.0 video generation ---` grouping comments inside `manifest()`.
- **Why:** Clarifies which model entries belong to which pipeline stage; eases future removal or refactoring of the Lotus pass.
