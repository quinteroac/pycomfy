# Requirement: Model Downloader Module

## Context

`comfy-diffusion` pipelines require model weights (checkpoints, VAEs, LoRAs, CLIP encoders, etc.)
to be present on disk before inference. Today, developers must manually download and place every
file, which creates friction when setting up new environments or sharing pipelines. Phase 1 of the
roadmap calls for an automatic model download module (`comfy_diffusion/downloader.py`) that
resolves and fetches all models required by a given pipeline before execution — removing the manual
step entirely and acting as a prerequisite for reliable pipeline execution across environments.

Models live on three sources: HuggingFace Hub (single-file downloads via `huggingface_hub`),
CivitAI (versioned model downloads via `civitai` library), and direct URLs (stdlib fallback).

Each pipeline owns its manifest. `downloader.py` provides only the types and `download_models()`.
`comfy_diffusion/pipelines/ltx2_t2v.py` is the reference implementation that establishes the
pattern for all future pipeline iterations.

## Goals

- Provide a manifest-based model download API (`downloader.py`) with polymorphic entry types for
  HuggingFace Hub, CivitAI, and direct URLs.
- Make downloads idempotent — already-present files are skipped without re-downloading.
- Verify file integrity via optional SHA256 hash checking.
- Establish `comfy_diffusion/pipelines/ltx2_t2v.py` as the canonical pipeline pattern: a module
  that exports both `manifest()` and a stub `run()`, serving as a template for future iterations.

## User Stories

### US-001: Model Entry Types

**As a** developer, **I want** typed manifest entry dataclasses for each download source
**so that** I can declare pipeline models with the exact fields each source requires — no
spurious `None` fields.

**Acceptance Criteria:**
- [ ] Three dataclasses defined in `comfy_diffusion/downloader.py`:

  ```python
  @dataclass
  class HFModelEntry:
      repo_id: str          # e.g. "Lightricks/LTX-Video"
      filename: str         # single file within the repo
      dest: str | Path      # destination path relative to models_dir
      revision: str | None = None
      sha256: str | None = None

  @dataclass
  class CivitAIModelEntry:
      model_id: int
      version_id: int
      dest: str | Path
      sha256: str | None = None

  @dataclass
  class URLModelEntry:
      url: str
      filename: str
      dest: str | Path
      sha256: str | None = None

  ModelEntry = HFModelEntry | CivitAIModelEntry | URLModelEntry
  ```

- [ ] `dest` is interpreted relative to `models_dir` when it is a relative path
- [ ] `from comfy_diffusion.downloader import HFModelEntry, CivitAIModelEntry, URLModelEntry,
  ModelEntry` works without error
- [ ] Typecheck / lint passes

---

### US-002: `download_models()` — Fetch All Models

**As a** developer or pipeline, **I want** `download_models(manifest, *, models_dir=None)`
**so that** all listed models are fetched to the correct local paths before inference begins.

**Acceptance Criteria:**
- [ ] `download_models(manifest: list[ModelEntry], *, models_dir: str | Path | None = None, quiet: bool = False) -> None`
  implemented in `comfy_diffusion/downloader.py`
- [ ] Dispatches by entry type:
  - `HFModelEntry` → `huggingface_hub.hf_hub_download(repo_id, filename, revision, local_dir=dest_dir)`; authenticates automatically via `HF_TOKEN` env var when present
  - `CivitAIModelEntry` → `civitai` library download by `model_id` / `version_id`; authenticates automatically via `CIVITAI_API_KEY` env var when present
  - `URLModelEntry` → `urllib.request` chunk loop (stdlib only); no token support
- [ ] If `HF_TOKEN` is not set and an `HFModelEntry` targets a gated model, raises a clear `RuntimeError` with instructions to set `HF_TOKEN`
- [ ] If `CIVITAI_API_KEY` is not set and a `CivitAIModelEntry` is requested, raises a clear `RuntimeError` with instructions to set `CIVITAI_API_KEY`
- [ ] Tokens are never logged, printed, or embedded in file paths
- [ ] Files already present at the resolved destination path are skipped (idempotent — no
  re-download, no error), **unless** SHA256 is set and the existing file fails verification —
  in that case the file is deleted and re-downloaded automatically
- [ ] Destination directories are created automatically (`parents=True, exist_ok=True`)
- [ ] On download failure, raises a clear `RuntimeError` with source identifier and reason
- [ ] `from comfy_diffusion.downloader import download_models` works without error
- [ ] Typecheck / lint passes

---

### US-003: Progress Reporting

**As a** developer, **I want** download progress shown per file **so that** I can monitor
long-running downloads without writing custom logging.

**Acceptance Criteria:**
- [ ] `HFModelEntry` downloads: progress is handled natively by `huggingface_hub`
- [ ] `URLModelEntry` downloads: when `tqdm` is installed shows a progress bar with filename and
  byte count; when not installed falls back to `print` statements — no `ImportError` raised
- [ ] `CivitAIModelEntry` downloads: uses whatever progress the `civitai` library provides
- [ ] All progress is suppressed when `quiet=True` is passed to `download_models()`
- [ ] Typecheck / lint passes

---

### US-004: SHA256 Integrity Verification

**As a** developer, **I want** downloaded files verified against an expected SHA256 hash
**so that** corrupted or mismatched weights are caught before they cause silent inference errors.

**Acceptance Criteria:**
- [ ] When `entry.sha256` is set, the file's SHA256 is computed after download and compared to
  the expected value (using `hashlib.sha256` from stdlib)
- [ ] Hash check also runs on files that were already present (skipped download)
- [ ] If the hash does not match on a **freshly downloaded** file, a `ValueError` is raised with
  filename, expected hash, and actual hash; the corrupted file is deleted before raising
- [ ] If the hash does not match on an **already-present** file, the file is deleted and
  re-downloaded once; if it fails again, `ValueError` is raised
- [ ] When `entry.sha256` is `None`, verification is skipped silently
- [ ] Typecheck / lint passes

---

### US-005: `comfy_diffusion/pipelines/ltx2_t2v.py` — Reference Pipeline Pattern

**As a** developer, **I want** a reference pipeline module **so that** I have a clear template
to follow when implementing future pipelines in later iterations.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/pipelines/` package created with `__init__.py`
- [ ] `comfy_diffusion/pipelines/ltx2_t2v.py` created with:
  - `manifest() -> list[ModelEntry]` — returns all `HFModelEntry` / `CivitAIModelEntry` /
    `URLModelEntry` instances required by the `ltx2/video_ltx2_t2v` workflow (checkpoint,
    VAE, CLIP/text encoder); `sha256` included where publicly known
  - `run(...)` — stub that raises `NotImplementedError("ltx2_t2v pipeline not yet implemented")`
    with a docstring describing the intended signature
- [ ] `from comfy_diffusion.pipelines.ltx2_t2v import manifest, run` works without error
- [ ] Calling `download_models(manifest())` with all files already present completes without
  error (idempotency integration test)
- [ ] A module-level docstring explains the pattern: "Each pipeline module exports `manifest()`
  and `run()`. Call `download_models(manifest())` before `run()`."
- [ ] Typecheck / lint passes

---

### US-006: Tests

**As a** developer, **I want** automated tests for the downloader **so that** CI catches
regressions without hitting the network or requiring GPU.

**Acceptance Criteria:**
- [ ] `tests/test_downloader.py` exists with at least the following test cases:
  - Idempotency: a file already present at the destination is not re-downloaded (mock all
    three backends)
  - SHA256 pass: existing file with matching hash is not re-downloaded
  - SHA256 fail on existing file: file is deleted, re-downloaded, verified again
  - SHA256 fail on fresh download: `ValueError` raised, file deleted
  - Progress fallback: `URLModelEntry` download runs without error when `tqdm` is not available
    (patch `tqdm` import to raise `ImportError`)
  - Dispatch: each entry type calls the correct backend
- [ ] All tests pass on CPU-only CI via `uv run pytest`; no real network calls in tests
- [ ] Typecheck / lint passes

---

## Functional Requirements

- FR-1: `comfy_diffusion/downloader.py` provides only types and `download_models()` — no
  manifest data. It is **not** re-exported at the `comfy_diffusion` package level.
- FR-2: Each pipeline module in `comfy_diffusion/pipelines/` owns its manifest. The `pipelines`
  package is **not** re-exported at the `comfy_diffusion` package level.
- FR-3: Lazy import pattern — `huggingface_hub`, `civitai`, and `tqdm` are imported inside
  function bodies; missing optional deps raise a clear `ImportError` with an install hint.
- FR-4: `download_models()` resolves relative `dest` paths against `models_dir`; when
  `models_dir` is `None`, defaults to `Path("./models")` (relative to CWD at call time).
- FR-5: `huggingface_hub`, `civitai`, and `tqdm` are declared as optional dependencies in
  `pyproject.toml` under `[project.optional-dependencies]` key `download`
  (e.g. `comfy-diffusion[download]`).
- FR-10: Authentication tokens are read exclusively from environment variables (`HF_TOKEN`,
  `CIVITAI_API_KEY`) — never hardcoded, never accepted as function parameters, never logged.
- FR-6: SHA256 is computed using `hashlib.sha256` from stdlib — no external hashing library.
- FR-7: `URLModelEntry` transport uses `urllib.request` only (stdlib) — no `requests` or
  `httpx`.
- FR-8: All new public symbols in `downloader.py` (`HFModelEntry`, `CivitAIModelEntry`,
  `URLModelEntry`, `ModelEntry`, `download_models`) listed in `__all__`.
- FR-9: All tests must pass on CPU-only CI (`uv run pytest`); no network calls in tests.

## Non-Goals (Out of Scope)

- Implementing the actual inference logic in `ltx2_t2v.run()` (deferred to pipeline iterations).
- Manifest builders for any pipeline other than `ltx2_t2v`.
- Re-exporting downloader or pipeline symbols at the `comfy_diffusion` package level.
- Resumable / chunked downloads with partial-file recovery.
- Parallel / concurrent multi-file downloads.
- A CLI entry point or standalone download script.
- Updating `ROADMAP.md` or bumping the package version (deferred to approve-prototype step).

## Open Questions

- None
