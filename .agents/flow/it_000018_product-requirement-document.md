# Requirement: Packaging, DX & Distributable Skills

## Context
`comfy-diffusion` has 16+ implemented modules but is not yet pip-installable as a proper Python package. Developers cannot install it as a dependency, IDEs lack type information for autocompletion, and agent skills intended for distribution are not bundled with the package. This iteration makes the library a first-class Python package with clean install experience, type support, and distributable skills.

## Goals
- Make `comfy-diffusion` installable via `pip install .` or `uv pip install .` with correct metadata and dependency resolution
- Provide functional extras groups (`[cuda]`, `[cpu]`, `[video]`, `[audio]`, `[all]`) so consumers pick only what they need
- Enable IDE autocompletion and type checking for the public API via `py.typed` marker and type stubs
- Bundle agent skills as package data in `comfy_diffusion/skills/` so they ship in the wheel and are discoverable at runtime via `importlib.resources`

## User Stories

### US-001: Base Installation
**As a** Python developer, **I want** to install `comfy-diffusion` from the repo root with `pip install .` or `uv pip install .` **so that** I can use it as a dependency in my own project.

**Acceptance Criteria:**
- [ ] `pyproject.toml` has complete PEP 621 metadata (name, version, description, license, requires-python >= 3.12, authors, URLs)
- [ ] Core dependencies are declared (excluding `torch`, which is optional-only)
- [ ] `uv pip install .` in a clean venv succeeds without errors
- [ ] `import comfy_diffusion` works after install
- [ ] `comfy_diffusion.check_runtime()` returns a valid diagnostics dict after install
- [ ] `torch` is NOT pulled in by a base install
- [ ] ComfyUI submodule is NOT declared as a pip dependency — it remains vendored

### US-002: Extras Groups
**As a** Python developer, **I want** to install specific extras like `comfy-diffusion[cuda]` or `comfy-diffusion[all]` **so that** I only install the heavy dependencies I actually need.

**Acceptance Criteria:**
- [ ] `[cuda]` extra installs torch with CUDA support
- [ ] `[cpu]` extra installs torch CPU-only
- [ ] `[video]` extra installs `opencv-python` and `imageio`
- [ ] `[audio]` extra installs `torchaudio`
- [ ] `[all]` extra installs the union of all above extras
- [ ] Installing an extra does not break the base install
- [ ] Typecheck / lint passes

### US-003: Type Stubs and py.typed
**As a** Python developer using an IDE, **I want** type information for `comfy-diffusion`'s public API **so that** I get autocompletion and type checking without additional configuration.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/py.typed` marker file exists and is included in the wheel
- [ ] Public API functions have inline type annotations (parameters and return types)
- [ ] Pylance / pyright resolves types for public symbols (`check_runtime`, `vae_decode`, `vae_encode`, `apply_lora`, `encode_prompt`, `sample`, etc.)
- [ ] `py.typed` is declared in `pyproject.toml` package data so it ships in the wheel
- [ ] Typecheck / lint passes

### US-004: Distributable Agent Skills
**As a** consumer of the library (human or AI agent), **I want** agent skills bundled in `comfy_diffusion/skills/` **so that** I can discover and load them at runtime from an installed package.

**Acceptance Criteria:**
- [ ] `comfy_diffusion/skills/` directory exists with skill files (`.md` or other formats)
- [ ] Skills are declared as `package_data` (or equivalent) in `pyproject.toml` so they are included in the built wheel
- [ ] Skills are discoverable at runtime via `importlib.resources` (e.g., `importlib.resources.files("comfy_diffusion.skills")`)
- [ ] Skills in `comfy_diffusion/skills/` are distinct from `.agents/skills/` (internal dev workflow) — no confusion between the two
- [ ] Typecheck / lint passes

### US-005: Smoke Test Post-Install
**As a** developer, **I want** a smoke test that validates the package installs and imports correctly **so that** regressions in packaging are caught early.

**Acceptance Criteria:**
- [ ] A test (or script) exists that: installs `comfy-diffusion` in a clean venv, imports it, and calls `check_runtime()`
- [ ] The test verifies that `comfy_diffusion/skills/` data files are accessible via `importlib.resources`
- [ ] The test verifies that `py.typed` is present in the installed package
- [ ] The test can run in CI (CPU-only, no GPU required)
- [ ] Typecheck / lint passes

## Functional Requirements
- FR-1: `pyproject.toml` must contain complete PEP 621 metadata with `requires-python = ">=3.12"`
- FR-2: `torch` must never appear in core `dependencies` — only in `[cuda]` and `[cpu]` optional-dependencies
- FR-3: ComfyUI vendored submodule must not be declared as a package dependency
- FR-4: `py.typed` marker file must be included in `package_data` so it ships in wheels and sdists
- FR-5: `comfy_diffusion/skills/*.md` (and any other skill files) must be declared in `package_data` so they ship in the wheel
- FR-6: A runtime utility must allow discovering bundled skills via `importlib.resources.files("comfy_diffusion.skills")`
- FR-7: All existing tests must continue to pass after `pyproject.toml` changes
- FR-8: The package must be installable with both `pip install .` and `uv pip install .`

## Non-Goals (Out of Scope)
- Publishing to PyPI (this iteration makes it installable from source only)
- Generating standalone `.pyi` stub files — inline annotations are sufficient
- Migrating `.agents/skills/` (internal dev workflow) into the distributable package
- Adding a CLI entry point or console scripts
- Creating a high-level pipeline abstraction
- Updating the ComfyUI submodule pin

## Open Questions
All resolved:

1. **What skill files to seed?** — A single `comfy_diffusion/skills/SKILL.md` containing: project conventions (lazy imports, `str | Path`, BHWC tensor layout, latent dict format), available modules with their public signatures, and the `get_skills_path()` helper. More granular skills (patterns, workflows) will be added iteratively post-release. **Important:** Before writing `SKILL.md`, read all existing module files (`models.py`, `conditioning.py`, `sampling.py`, `vae.py`, `lora.py`, `audio.py`, `controlnet.py`, `latent.py`, `image.py`, `video.py`, `mask.py`) and extract the actual public signatures from `__all__` in each. The `SKILL.md` must reflect the real implemented API, not assumptions.

2. **`__init__.py` or plain directory?** — Subpackage with an empty `__init__.py` so that `importlib.resources.files("comfy_diffusion.skills")` works correctly across all platforms, including wheels under zipimport.
