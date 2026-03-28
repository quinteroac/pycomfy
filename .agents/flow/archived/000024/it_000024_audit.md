# Audit — Iteration 000024

## Executive Summary

The implementation for iteration 000024 is largely compliant with the PRD. All 42 automated tests pass on CPU-only CI with no network calls. The three entry dataclasses, `download_models()`, the `ltx2_t2v.py` reference pipeline, SHA256 integrity verification, idempotency, and progress/quiet handling are all correctly implemented. The one notable gap is FR-5: `huggingface_hub` and `tqdm` are not declared as optional dependencies in their own extras group in `pyproject.toml` — `huggingface_hub` is absent entirely, while `tqdm` only appears under the unrelated `comfyui` extra. The CivitAI backend does not use the `civitai` library (which targets the Generator API, not model downloads) — this is a correct design decision and the urllib-based approach is functionally superior.

---

## Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | ✅ comply | `downloader.py` exposes only types + `download_models()`, no side effects on import |
| FR-2 | ✅ comply | `ltx2_t2v.py` owns its manifest; `pipelines/__init__.py` exports nothing |
| FR-3 | ⚠️ partially_comply | `huggingface_hub` and `tqdm` lazily imported; `civitai` library not used (CivitAI uses urllib directly — correct decision, REST API is the download mechanism) |
| FR-4 | ✅ comply | `_resolve_dest` correctly handles relative paths and raises `RuntimeError` when `models_dir=None` |
| FR-5 | ❌ does_not_comply | `huggingface_hub` absent from all optional-dependency groups; `tqdm` only in unrelated `comfyui` extra |
| FR-6 | ✅ comply | `hashlib.sha256` from stdlib, no external hashing library |
| FR-7 | ✅ comply | `urllib.request` only, no `requests` or `httpx` |
| FR-8 | ✅ comply | All public symbols have docstrings and are listed in `__all__` |
| FR-9 | ✅ comply | 42/42 tests pass via `uv run pytest`, all network calls patched |
| FR-10 | ✅ comply | Tokens read from env vars only, never logged/printed/embedded in paths |

---

## Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | ✅ comply | All three dataclasses with correct fields and defaults; `str \| Path` accepted for `dest` |
| US-002 | ✅ comply | Signature, dispatch, error messages, idempotency, directory creation all correct |
| US-003 | ⚠️ partially_comply | AC03: CivitAI progress uses `_stream_to_file` + tqdm, not the `civitai` library (acceptable — `civitai-py` targets Generator API, not downloads) |
| US-004 | ✅ comply | All five AC met: post-download hash, hash on existing, ValueError + delete on fresh mismatch, re-download on existing mismatch, None skips silently |
| US-005 | ✅ comply | `pipelines/` package created, `ltx2_t2v.py` with `manifest()`, `run()`, pattern docstring |
| US-006 | ✅ comply | 42 tests covering all required cases; pass on CPU-only CI with no network calls |

---

## Minor Observations

1. **FR-5 gap**: `huggingface_hub` is a runtime requirement for `HFModelEntry` downloads but is not declared as an optional extra. `tqdm` only appears in the unrelated `comfyui` extra.
2. **civitai library not used**: The `civitai-py` library targets the CivitAI Generator API (cloud inference), not model file downloads. Using `urllib.request` directly against the CivitAI REST API is the correct approach. The PRD's reference to the `civitai` library in US-003-AC03 and FR-3 was based on a mistaken assumption about what the library does.
3. **Gated-model detection is heuristic**: `_download_hf_entry` detects auth errors via keyword matching (`"gated"`, `"401"`, `"403"`, etc.) on the exception message string. This may miss future error patterns from `huggingface_hub`.
4. **`shutil.copy2` doubles disk usage**: HF downloads copy the cached file to the destination rather than symlinking/hardlinking. For large models (10–50 GB) this wastes significant disk space.
5. **`check_runtime()` return value ignored** in `ltx2_t2v.py`: The function returns an error dict but the pipeline discards it without checking, so a misconfigured runtime silently proceeds.

---

## Conclusions and Recommendations

The iteration is functionally complete. The only blocking item is FR-5. The refactor should:

1. **Add `[downloader]` optional extra** to `pyproject.toml` with `huggingface_hub>=0.20` and `tqdm>=4.67`.
2. **Accept the civitai-library deviation** as a deliberate and correct design decision; update the PROJECT_CONTEXT.md to record that CivitAI downloads use the REST API via `urllib.request` directly.
3. **Fix `check_runtime()` call** in `ltx2_t2v.py` to raise on error dict instead of silently discarding.
4. Defer gated-model heuristic improvement and copy2-vs-link optimisation to a future iteration.

---

## Refactor Plan

### Task 1 — Add `[downloader]` optional extra to `pyproject.toml` *(FR-5)*

**File:** `pyproject.toml`

Add a `downloader` group under `[project.optional-dependencies]`:

```toml
downloader = [
    "huggingface_hub>=0.20",
    "tqdm>=4.67",
]
```

Also add `huggingface_hub>=0.20` and `tqdm>=4.67` to the `all` extra so `pip install comfy-diffusion[all]` covers the downloader.

### Task 2 — Fix `check_runtime()` usage in `ltx2_t2v.py` *(minor)*

**File:** `comfy_diffusion/pipelines/ltx2_t2v.py`

Replace the bare `check_runtime()` call with a guard that raises on error:

```python
result = check_runtime()
if result.get("error"):
    raise RuntimeError(f"ComfyUI runtime not available: {result['error']}")
```

### Task 3 — Update PROJECT_CONTEXT.md *(documentation)*

Record the CivitAI design decision: downloads use the CivitAI REST API directly via `urllib.request`; the `civitai-py` library is not used because it targets the Generator API (cloud inference), not model file downloads.
