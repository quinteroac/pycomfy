# Requirement: Runtime — ComfyUI Auto-Bootstrap

## Context
Users who install `comfy-diffusion` via `pip install` (not from a git clone) end up with an empty `vendor/ComfyUI` directory because the git submodule is never initialized. This causes every `comfy.*` import to fail immediately. The fix is to make `check_runtime()` detect the missing submodule and download ComfyUI automatically using only Python stdlib, so the package works out of the box with no extra steps.

## Goals
- `pip install comfy-diffusion` produces a working installation without requiring `git submodule init`.
- `check_runtime()` remains the single entry point for runtime validation and bootstrap; callers never need to know about the download.
- Zero new runtime dependencies — only Python stdlib (`urllib.request`, `zipfile`, `shutil`, `pathlib`).

## User Stories

### US-001: Auto-detect missing ComfyUI and download it
**As a** developer who installed `comfy-diffusion` via `pip install`, **I want** `check_runtime()` to automatically download ComfyUI when `vendor/ComfyUI` is missing or empty **so that** I do not need to manually run `git submodule init && git submodule update`.

**Acceptance Criteria:**
- [ ] `check_runtime()` checks whether `vendor/ComfyUI` exists and contains at least one file (e.g. `comfy/` subdirectory present).
- [ ] If the directory is missing or empty, `check_runtime()` downloads the pinned ComfyUI release zip from GitHub using `urllib.request` and extracts it into `vendor/ComfyUI` using `zipfile`.
- [ ] The downloaded version matches the same tag/commit that the git submodule is pinned to (hardcoded constant in `_runtime.py`).
- [ ] After a successful download, `check_runtime()` returns the same healthy dict as if the submodule had been initialized manually.
- [ ] Typecheck / lint passes.

### US-002: Return error dict on bootstrap failure
**As a** developer, **I want** `check_runtime()` to return a structured error dict when the auto-download fails **so that** my application can surface a clear message without crashing.

**Acceptance Criteria:**
- [ ] If the download or extraction raises any exception (network error, permission denied, disk full, etc.), `check_runtime()` catches it and returns an error dict with key `"error"` containing a human-readable description of the failure.
- [ ] `check_runtime()` never raises an exception regardless of bootstrap outcome.
- [ ] The error dict includes `"python_version"` (already required by existing contract).
- [ ] Typecheck / lint passes.

### US-003: Test bootstrap with empty vendor directory
**As a** developer, **I want** a pytest test that simulates a missing `vendor/ComfyUI` and calls `check_runtime()` **so that** the bootstrap path is automatically validated on every CI run.

**Acceptance Criteria:**
- [ ] A test in `tests/` temporarily renames or mocks `vendor/ComfyUI` to simulate an empty/absent state, calls `check_runtime()`, and asserts the returned dict contains no `"error"` key after bootstrap succeeds.
- [ ] The test does not make real network calls — it patches `urllib.request.urlretrieve` (or equivalent) to provide a minimal valid zip fixture.
- [ ] Test passes on CPU-only CI environment without GPU.
- [ ] Typecheck / lint passes.

### US-004: Update documentation to instruct users to call `check_runtime()` first
**As a** developer reading the docs, **I want** the README (or equivalent entry-point documentation) to explicitly state that `check_runtime()` must be called before any other `comfy_diffusion` API **so that** I know the bootstrap will be triggered automatically.

**Acceptance Criteria:**
- [ ] `README.md` (or the primary usage doc) contains a "Quick Start" or "Usage" section showing `check_runtime()` as the first call before any model loading or inference.
- [ ] The section explains that calling `check_runtime()` on first use triggers an automatic download of ComfyUI if needed.
- [ ] The section shows how to handle the error dict returned on failure.

## Functional Requirements
- FR-1: `_runtime.py` must define a module-level constant `COMFYUI_PINNED_TAG` (e.g. `"v0.3.43"`) that identifies the ComfyUI release to download.
- FR-2: The download URL must be derived from `COMFYUI_PINNED_TAG` pointing to the GitHub archive zip (e.g. `https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/{tag}.zip`).
- FR-3: Extraction must place ComfyUI contents directly under `vendor/ComfyUI/` (strip the top-level archive directory created by GitHub's zip format).
- FR-4: Bootstrap must be idempotent — if `vendor/ComfyUI` is already populated, `check_runtime()` skips the download entirely.
- FR-5: Only Python stdlib modules may be used for network and archive operations (`urllib.request`, `zipfile`, `shutil`, `pathlib`, `tempfile`).
- FR-6: `check_runtime()` must always populate `"python_version"` in the returned dict, whether bootstrap succeeds, fails, or is skipped.

## Non-Goals (Out of Scope)
- Automatic updates of ComfyUI after initial bootstrap — pinned version only.
- Support for users cloning the repo (they continue to use `git submodule update --init`).
- Progress reporting or download progress callbacks.
- Checksum / signature verification of the downloaded archive.
- Support for offline/air-gapped environments.

## Open Questions
- None.
