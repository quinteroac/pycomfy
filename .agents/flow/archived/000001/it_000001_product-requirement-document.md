# Requirement: Project Foundation — Pure Python Library with ComfyUI Runtime Verification

## Context
pycomfy aims to expose ComfyUI's inference engine as an importable Python library — no server, no node graphs, no UI. Before any inference features can be built, the project needs a proper foundation: package structure, vendored ComfyUI dependency, correct path management, and a first importable function that proves the runtime is accessible. This iteration establishes that foundation so every subsequent iteration builds on a working, installable package.

## Goals
- Establish pycomfy as a standard, installable Python package consumable by downstream projects
- Vendor ComfyUI as a git submodule pinned to a stable release tag
- Provide a first public function (`check_runtime()`) that returns structured diagnostics proving the ComfyUI runtime is importable and responsive
- Ensure the project works on both CUDA and CPU-only environments (critical for CI)

## User Stories

### US-001: Python Package Structure
**As a** Python developer, **I want** pycomfy to be a proper Python package with `__init__.py` and standard layout **so that** I can `import pycomfy` like any other library.

**Acceptance Criteria:**
- [ ] `pycomfy/` directory exists as the package root with `__init__.py`
- [ ] `pycomfy/__init__.py` exports `check_runtime` as a public symbol
- [ ] Package follows standard src-less layout (package dir at repo root)
- [ ] Typecheck / lint passes

### US-002: ComfyUI as Git Submodule
**As a** developer cloning the repo, **I want** ComfyUI vendored as a git submodule at `vendor/ComfyUI` pinned to the latest stable release tag **so that** I get a reproducible, controlled dependency without floating on master.

**Acceptance Criteria:**
- [ ] `vendor/ComfyUI` is a git submodule pointing to a specific stable release tag of ComfyUI
- [ ] `.gitmodules` correctly references the ComfyUI repository
- [ ] `git submodule update --init` from a fresh clone brings ComfyUI in at the pinned tag — no manual steps
- [ ] The pinned tag is documented (in pyproject.toml metadata or a comment in `.gitmodules`)

### US-003: pyproject.toml and Editable Install
**As a** developer, **I want** `pyproject.toml` fully configured so that `uv pip install -e .` works from a fresh clone without manual path manipulation **so that** I can consume pycomfy as a real dependency in my own projects.

**Acceptance Criteria:**
- [ ] `pyproject.toml` declares package metadata (name, version, description, python `>=3.12`)
- [ ] `torch` is declared as an optional dependency with extras: `pycomfy[cuda]` and `pycomfy[cpu]`
- [ ] `uv pip install -e .` succeeds from a fresh clone (after submodule init)
- [ ] No hardcoded torch version — extras only declare the appropriate torch index/variant
- [ ] Typecheck / lint passes

### US-004: Runtime Path Management
**As a** developer importing pycomfy, **I want** the library to internally manage `sys.path` so that ComfyUI internals (e.g. `comfy.model_management`) are importable **so that** I never have to manually manipulate paths or set environment variables.

**Acceptance Criteria:**
- [ ] Importing `pycomfy` makes `vendor/ComfyUI` internals importable transparently
- [ ] Path manipulation is encapsulated inside pycomfy — no leaking into the consumer's `sys.path` beyond what is necessary
- [ ] Works regardless of the working directory the consumer runs from (uses absolute paths derived from the package location)
- [ ] Typecheck / lint passes

### US-005: `check_runtime()` — Structured Diagnostic Function
**As a** Python developer, **I want** to call `pycomfy.check_runtime()` and receive a structured dict with runtime diagnostics **so that** I can verify the ComfyUI engine is accessible and inspect the environment programmatically.

**Acceptance Criteria:**
- [ ] `check_runtime()` returns a dict with keys: `comfyui_version` (str), `device` (str, e.g. `"cuda:0"` or `"cpu"`), `vram_total_mb` (int), `vram_free_mb` (int), `python_version` (str)
- [ ] On CUDA environments, reports actual device and VRAM stats via `comfy.model_management`
- [ ] On CPU-only environments, returns `"device": "cpu"`, `vram_total_mb: 0`, `vram_free_mb: 0` without crashing
- [ ] Validates that `comfy.model_management` is importable and responsive — not just that the import didn't error
- [ ] `uv run python -c "from pycomfy import check_runtime; print(check_runtime())"` succeeds as a one-liner smoke test
- [ ] Typecheck / lint passes

## Functional Requirements
- FR-1: Package directory is `pycomfy/` at repo root with `__init__.py` exporting public API
- FR-2: ComfyUI is vendored at `vendor/ComfyUI` as a git submodule pinned to the latest stable release tag (not master)
- FR-3: `pyproject.toml` declares `requires-python = ">=3.12"`, package metadata, and optional `[cuda]`/`[cpu]` extras for torch
- FR-4: On import, pycomfy adds `vendor/ComfyUI` to the Python path using absolute paths derived from the package's own `__file__` location
- FR-5: `check_runtime()` returns `{"comfyui_version": str, "device": str, "vram_total_mb": int, "vram_free_mb": int, "python_version": str}`
- FR-6: `check_runtime()` gracefully handles CPU-only environments (no crash, reports `"cpu"` and zero VRAM)
- FR-7: `check_runtime()` returns an error dict (not raise) when ComfyUI runtime is unavailable (e.g. submodule not initialized), with `"error"` key describing the issue and `python_version` always populated
- FR-8: `uv pip install -e .` works from a fresh clone after `git submodule update --init`

## Non-Goals (Out of Scope)
- Any inference functionality (sampling, model loading, VAE, etc.) — that's for future iterations
- CLI or entry points — pycomfy is library-only for now
- Wrapping or abstracting ComfyUI nodes — this iteration only proves the runtime is reachable
- Supporting Python < 3.12
- Automatic torch installation — torch is an optional extra, consumers choose their variant
- Updating ComfyUI submodule to newer versions — deliberate, future-iteration decision

## Open Questions (Resolved)
- **OQ-1: Which ComfyUI stable release tag to pin?** Resolve at implementation time by checking https://github.com/comfyanonymous/ComfyUI/releases and taking the most recent tag matching `vX.Y.Z`. Do not decide now — resolve when running `git submodule add`.
- **OQ-2: Should `check_runtime()` raise or return error dict if submodule not initialized?** Return an error dict, not raise. It's a diagnostic function, not an execution function. The consumer needs to know *what* failed, not just *that* something failed. On failure: `{"error": "ComfyUI runtime not found. Run: git submodule update --init", "comfyui_version": None, "device": None, "vram_total_mb": None, "vram_free_mb": None, "python_version": "3.12.x"}` — `python_version` is always populated since it doesn't depend on the submodule.
- **OQ-3: Does `comfy.model_management` require initialization before reporting device/VRAM?** No — a bare import is sufficient. `comfy.model_management` exposes `get_torch_device()` and `get_total_memory()` without requiring any model to be loaded. Verify at implementation time with a direct `python -c` call against the submodule.
