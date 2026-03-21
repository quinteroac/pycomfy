# Audit — Iteration 000021

## Executive Summary

All four user stories are implemented and all 11 dedicated tests pass on CPU-only CI. The bootstrap path (auto-download ComfyUI from GitHub when `vendor/ComfyUI` is absent) is fully functional: `urllib.request`/`zipfile`/`shutil` are used exclusively, the download is idempotent, error dicts are returned on failure with `python_version` always populated, and the README Quick Start section covers the required documentation. One minor naming deviation exists: FR-1 specified the constant name `COMFYUI_PINNED_TAG` but the implementation uses `COMFYUI_PINNED_REF` — semantically identical, but non-compliant with the exact identifier the PRD prescribed.

---

## Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | ⚠️ partially_comply | Constant exists but named `COMFYUI_PINNED_REF` instead of the PRD-specified `COMFYUI_PINNED_TAG`. |
| FR-2 | ✅ comply | `COMFYUI_PINNED_ARCHIVE_URL` is correctly derived from the pinned ref and points to the GitHub archive zip. |
| FR-3 | ✅ comply | Extraction strips the top-level `ComfyUI-{ref}/` directory via `glob('ComfyUI-*')` + `shutil.move`, placing contents directly under `vendor/ComfyUI/`. |
| FR-4 | ✅ comply | `ensure_comfyui_available()` calls `_has_comfyui_runtime()` and skips the download when `vendor/ComfyUI` already contains the `comfy/` subdirectory. |
| FR-5 | ✅ comply | Only `urllib.request`, `zipfile`, `shutil`, `pathlib`, and `tempfile` (all stdlib) are used for network and archive operations. |
| FR-6 | ✅ comply | `python_version` is populated in all return paths: healthy dict, `_runtime_not_found`, and `_runtime_not_responsive`. |

---

## Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | ✅ comply | AC01–AC05 all satisfied. `_has_comfyui_runtime` checks `comfy/` subdirectory; download uses urllib+zipfile; pinned ref constant present; healthy dict returned after bootstrap; tests pass. |
| US-002 | ✅ comply | AC01–AC04 all satisfied. Exceptions caught inside `check_runtime()`; error dict with `"error"` key returned; never raises; `python_version` always present; tests pass. |
| US-003 | ✅ comply | AC01–AC04 all satisfied. `test_check_runtime_bootstraps_from_absent_vendor_comfyui_dir` patches `urlretrieve` with a minimal zip fixture, asserts no `"error"` key, runs on CPU, passes. |
| US-004 | ✅ comply | AC01–AC03 all satisfied. README Quick Start shows `check_runtime()` as first call, explains automatic download, and shows error dict handling. All three README tests pass. |

---

## Minor Observations

1. **Constant name mismatch:** `COMFYUI_PINNED_REF` vs PRD-specified `COMFYUI_PINNED_TAG` — cosmetic rename needed.
2. **Stale error message:** `_runtime_not_found` still instructs the user to run `git submodule update --init`, which is misleading now that auto-download is implemented. Should reflect that auto-download was attempted and failed.
3. **`ensure_comfyui_on_path()` gap:** Does not call `ensure_comfyui_available()` — direct callers bypass the bootstrap. Low risk given current public API, but a latent inconsistency.

---

## Conclusions and Recommendations

The iteration is functionally complete and well-tested. The single spec deviation (`COMFYUI_PINNED_REF` vs `COMFYUI_PINNED_TAG`) is cosmetic but should be corrected for spec fidelity. The stale error message may confuse users who see it in a post-auto-download-failure scenario.

---

## Refactor Plan

### R-1 — Rename `COMFYUI_PINNED_REF` → `COMFYUI_PINNED_TAG`

- **File:** `comfy_diffusion/_runtime.py`
- **Action:** Rename constant `COMFYUI_PINNED_REF` to `COMFYUI_PINNED_TAG`; update `COMFYUI_PINNED_ARCHIVE_URL` f-string accordingly.
- **Files also affected:** `tests/test_runtime_autodownload.py` — all references to `_runtime.COMFYUI_PINNED_REF` must be updated to `_runtime.COMFYUI_PINNED_TAG`.

### R-2 — Fix stale `_runtime_not_found` error message

- **File:** `comfy_diffusion/runtime.py`
- **Action:** Update `_runtime_not_found` to produce a message that reflects the auto-download scenario: e.g. `"ComfyUI runtime bootstrap failed."` (detail appended from caught exception), removing the `git submodule` instruction that no longer applies as primary guidance.

### R-3 — Have `ensure_comfyui_on_path()` call `ensure_comfyui_available()` (optional)

- **File:** `comfy_diffusion/_runtime.py`
- **Action:** Call `ensure_comfyui_available()` at the start of `ensure_comfyui_on_path()` so that direct callers also trigger the bootstrap automatically.
- **Risk:** Low — existing tests mock `_comfyui_root` so the change will not break them.
