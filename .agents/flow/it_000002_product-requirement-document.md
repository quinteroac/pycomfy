# Requirement: pycomfy.models — Checkpoint Loading Module

## Context

Iteration 002 introduces `pycomfy.models`, the first inference-facing module in the library.
It gives consumers a clean, typed API to load Stable Diffusion checkpoints (`.safetensors` /
`.ckpt`) from disk using ComfyUI's battle-tested internal loaders (`comfy.sd` / `comfy.utils`).
The returned objects are native comfy types, ready to be consumed directly by the future
`conditioning` (it-003) and `sampling` (it-004) modules — no wrapping, no conversion overhead.

## Goals

- Expose a minimal, ergonomic API (`ModelManager` + `CheckpointResult`) for loading checkpoints.
- Delegate all parsing and tensor work to `comfy.sd.load_checkpoint_guess_config()` — pycomfy
  never reimplements checkpoint logic.
- Guarantee that `from pycomfy.models import ModelManager` never crashes on a CPU-only
  environment; CI stays green without a GPU.
- Surface actionable Python exceptions (not raw comfy tracebacks) for the two most common failure
  modes: directory not found and file not found.
- Pass `ruff check .` and `mypy pycomfy/` without regressions.

## User Stories

### US-001: Import ModelManager with no side effects
**As a** Python developer, **I want** to import `ModelManager` from `pycomfy.models` **so that**
I can set up the manager before any model file is touched, without triggering unexpected errors or
GPU initialisation.

**Acceptance Criteria:**
- [ ] `from pycomfy.models import ModelManager` succeeds on a CPU-only machine with no model files
  present.
- [ ] The import does not load any model files, allocate GPU memory, or modify global state beyond
  what `import pycomfy` already does.
- [ ] `ruff check .` passes (no lint errors introduced by the new module).
- [ ] `mypy pycomfy/` passes in strict mode (no type errors introduced by the new module).

---

### US-002: Instantiate ModelManager with a valid models directory
**As a** Python developer, **I want** to construct `ModelManager(models_dir="/path/to/models")`
**so that** the manager knows where to look for checkpoint files without me having to pass the
path on every load call.

**Acceptance Criteria:**
- [ ] `ModelManager(models_dir=<existing_dir>)` constructs successfully and silently.
- [ ] `ModelManager(models_dir=<non_existing_dir>)` raises `ValueError` with a message that
  includes the offending path (e.g. `"models_dir does not exist: /bad/path"`).
- [ ] The raised exception is a `ValueError`, never a raw exception propagated from comfy internals.
- [ ] `models_dir` is stored as a `pathlib.Path` internally; the constructor accepts both `str`
  and `pathlib.Path`.

---

### US-003: Load a checkpoint and receive a typed CheckpointResult
**As a** Python developer (or a future pycomfy module), **I want** to call
`manager.load_checkpoint("animagine-xl.safetensors")` **so that** I receive a `CheckpointResult`
with `.model`, `.clip`, and `.vae` attributes that are the raw comfy objects — ready to pass
directly to sampling and conditioning APIs.

**Acceptance Criteria:**
- [ ] `load_checkpoint(filename)` accepts a filename (not a full path); the full path is resolved
  as `models_dir / "checkpoints" / filename`.
- [ ] Returns a `CheckpointResult` dataclass (or equivalent typed container) with three
  attributes: `.model`, `.clip`, `.vae`.
- [ ] The three attributes hold the native objects returned by
  `comfy.sd.load_checkpoint_guess_config()` — not wrapped in any additional abstraction.
- [ ] `.clip` and `.vae` are typed `Any | None`; the loader does not raise when comfy returns
  `None` (e.g. pure-unet checkpoints).
- [ ] `CheckpointResult` is importable directly: `from pycomfy.models import CheckpointResult`.
- [ ] `mypy` accepts the type annotations (comfy internals typed as `Any` via
  `ignore_missing_imports = true`).

---

### US-004: Graceful error when the checkpoint file is missing
**As a** Python developer, **I want** `load_checkpoint` to raise a clear `FileNotFoundError`
(not a raw comfy traceback) when the requested file does not exist **so that** I can quickly
identify and fix the issue.

**Acceptance Criteria:**
- [ ] `manager.load_checkpoint("nonexistent.safetensors")` raises `FileNotFoundError`.
- [ ] The exception message includes the full resolved path that was looked up.
- [ ] No partial comfy stack trace leaks through — the exception is raised by pycomfy before
  comfy is invoked.
- [ ] A pytest test covers this path and passes on CPU without a GPU.

---

### US-005: CPU-only smoke test — import never crashes without a GPU
**As a** CI pipeline, **I want** `from pycomfy.models import ModelManager` to succeed on a
CPU-only machine **so that** automated tests remain green without GPU hardware.

**Acceptance Criteria:**
- [ ] `uv run python -c "from pycomfy.models import ModelManager; print('ok')"` exits with code 0
  on a CPU-only machine.
- [ ] A pytest test asserts that `ModelManager` is importable and can be instantiated with a valid
  directory (using `tmp_path`) on CPU.
- [ ] The test does NOT attempt to call `load_checkpoint()` with a real model file (that would
  require downloading weights).

---

## Functional Requirements

- **FR-1** `pycomfy/models.py` (or `pycomfy/models/__init__.py`) exposes `ModelManager` and
  `CheckpointResult` as public names.
- **FR-2** `ModelManager.__init__(models_dir: str | pathlib.Path)` validates that the directory
  exists; raises `ValueError` if not. The constructor also registers the following subdirectories
  with ComfyUI's `folder_paths` system (creating them is **not** required — registration is
  unconditional):
  ```
  models_dir/
    checkpoints/   ← load_checkpoint resolves filenames here
    embeddings/    ← registered for embedding lookup during checkpoint load
    vae/           ← reserved for future it-02b (standalone VAE loading)
    loras/         ← reserved for future it-006 (LoRA loading)
  ```
  This mirrors the directory layout that ComfyUI itself uses and that ecosystem users expect.
- **FR-3** `ModelManager.load_checkpoint(filename: str) -> CheckpointResult` resolves the full
  path as `models_dir / "checkpoints" / filename`, raises `FileNotFoundError` before invoking
  comfy if the file does not exist.
- **FR-4** Internally, `load_checkpoint` calls `comfy.sd.load_checkpoint_guess_config()` (or the
  equivalent comfy loader) — pycomfy must not reimplement checkpoint parsing.
- **FR-5** `CheckpointResult` is a `dataclass` with attributes `.model: Any`, `.clip: Any | None`,
  `.vae: Any | None`. `clip` and `vae` may be `None` for pure-unet checkpoints; callers (e.g.
  it-003 conditioning, it-004 sampling) are responsible for asserting non-`None` before use.
- **FR-6** `pycomfy/__init__.py` does **not** automatically import `ModelManager` — consumers
  import from `pycomfy.models` explicitly. (Avoids eager comfy initialisation on bare `import
  pycomfy`.)
- **FR-7** All new code passes `ruff check .` (rules E, F, I, UP) and `mypy pycomfy/` in strict
  mode with `ignore_missing_imports = true`.

## Non-Goals (Out of Scope)

- **Standalone VAE loading** — loading separate `.vae.safetensors` files is deferred to a later
  iteration.
- **Standalone CLIP loading** — same; CLIP-only loading is out of scope for it-002.
- **Model caching** — no VRAM cache, no "skip reload if already loaded" logic.
- **Model discovery / listing** — no `ModelManager.list_available()` or similar API.
- **Custom comfy nodes or node graph** — pycomfy never loads `nodes.py`.
- **Automatic models directory detection** — no scanning of default ComfyUI model paths; the
  caller must always supply `models_dir` explicitly.

## Open Questions

- **Q1 (resolved):** `load_checkpoint` accepts **filename only** — `models_dir / "checkpoints" /
  filename` is the strict contract. This mirrors ComfyUI's own directory layout (`models/checkpoints/`)
  and is what ecosystem users expect. Absolute path overrides are not supported. If a caller needs
  a different base directory, they instantiate a new `ModelManager`. No special cases.
- **Q2 (resolved):** `CheckpointResult` allows `None` for `.clip` and `.vae`:
  ```python
  @dataclass
  class CheckpointResult:
      model: Any
      clip: Any | None
      vae: Any | None
  ```
  Pure-unet checkpoints may return `None` for either field. Callers (it-003 conditioning,
  it-004 sampling) are responsible for asserting non-`None` before use — the loader does not
  raise when comfy returns `None`.
