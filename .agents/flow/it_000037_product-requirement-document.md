# Requirement: Qwen Image Edit 2511 — CLI Example

## Context

The pipeline `comfy_diffusion/pipelines/image/qwen/edit_2511.py` is fully implemented but
has no runnable example in the `examples/` tree.  Developers integrating `comfy_diffusion`
into their own applications need a ready-to-run CLI script that demonstrates the complete
workflow: model download, optional Lightning LoRA toggle, single or multi-reference image
editing, and output saving.  This mirrors the established example pattern used by other
pipelines (e.g. `examples/image/edit/qwen/layered_i2l.py`).

## Goals

- Provide a self-contained, executable CLI example for `edit_2511` that a developer can run
  out of the box to validate the pipeline.
- Document the pipeline's public API (`manifest()` + `run()`) through concrete, annotated
  usage in the codebase.

## User Stories

### US-001: Download models via CLI

**As a** developer integrating `comfy_diffusion`, **I want** to run the example with
`--download-only` **so that** I can fetch all required model weights before running
inference, without modifying any code.

**Acceptance Criteria:**
- [ ] `uv run python examples/image/edit/qwen/edit_2511.py --models-dir /path/to/models --image placeholder.png --prompt "x" --download-only` exits with code `0`.
- [ ] All four model files listed in `manifest()` are downloaded (or skipped if present).
- [ ] The script prints a clear "Models ready." message and exits without performing inference.
- [ ] Typecheck / lint passes.

---

### US-002: Edit an image with the Lightning LoRA (default, 4 steps)

**As a** developer, **I want** to pass `--image`, `--prompt`, and (optionally) `--seed`
**so that** I get an edited output image saved to disk using the fast 4-step Lightning LoRA
path.

**Acceptance Criteria:**
- [ ] `uv run python examples/image/edit/qwen/edit_2511.py --models-dir /path/to/models --image input.png --prompt "Make the sofa look like it is covered in fur"` completes without error.
- [ ] The output image is saved as `<output-prefix>.png` (default prefix: `qwen_edit_2511_output`).
- [ ] Default `steps=4` and `use_lora=True` are used when `--no-lora` is not passed.
- [ ] Typecheck / lint passes.

---

### US-003: Edit without Lightning LoRA (40 steps)

**As a** developer, **I want** to pass `--no-lora` **so that** I can run the standard
40-step CFGNorm path when higher quality is preferred over speed.

**Acceptance Criteria:**
- [ ] Adding `--no-lora` sets `use_lora=False` and `steps=40` in the `run()` call.
- [ ] The script accepts an optional explicit `--steps` override that takes precedence over the LoRA-derived default.
- [ ] Typecheck / lint passes.

---

### US-004: Multi-reference image editing

**As a** developer, **I want** to pass `--image2` and/or `--image3` **so that** I can
provide additional reference images for the editing conditioning.

**Acceptance Criteria:**
- [ ] `--image2` and `--image3` are optional CLI arguments; both default to `None`.
- [ ] When provided, each path is validated (`is_file()`) before being loaded and passed to `run()`.
- [ ] When not provided, `None` is passed to `run()` (no error).
- [ ] Typecheck / lint passes.

---

## Functional Requirements

- **FR-1:** The example file is located at `examples/image/edit/qwen/edit_2511.py`.
- **FR-2:** The script uses `argparse` with the following arguments:
  - `--models-dir` (default: `PYCOMFY_MODELS_DIR` env var)
  - `--image` (required, primary input image path)
  - `--image2` (optional, second reference image path)
  - `--image3` (optional, third reference image path)
  - `--prompt` (default: `""`)
  - `--seed` (int, default: `42`)
  - `--steps` (int, optional; overrides LoRA-derived default when provided)
  - `--cfg` (float, default: `3.0`)
  - `--no-lora` (store_true flag; disables Lightning LoRA)
  - `--output-prefix` (default: `"qwen_edit_2511_output"`)
  - `--download-only` (store_true flag)
- **FR-3:** When `--models-dir` is not set and `PYCOMFY_MODELS_DIR` is not defined, the
  script prints an error to stderr and exits with code `1`.
- **FR-4:** Default steps: `4` when `use_lora=True`; `40` when `use_lora=False`.  An
  explicit `--steps` value always wins.
- **FR-5:** Output image is saved as `{output_prefix}.png` using `PIL.Image.save()`.
- **FR-6:** The module-level docstring mirrors the format used in `layered_i2l.py`:
  description, usage block with `uv run` commands, full-options example.
- **FR-7:** `check_runtime()` is called after model download and before inference; error
  causes exit with code `1` and a message to stderr.
- **FR-8:** All imports from `comfy_diffusion` are deferred to inside `main()` (after
  argument parsing), consistent with the lazy-import convention of the project.

## Non-Goals (Out of Scope)

- Implementing or modifying the `edit_2511` pipeline itself — it is already complete.
- Adding tests for the example script.
- Supporting batch inference (multiple images in one call).
- Providing a Jupyter notebook or any non-CLI interface.
- LoRA inflation variant (`image-qwen_image_edit_2511_lora_inflation.json`) — separate
  pipeline, separate iteration.

## Open Questions

- None.
