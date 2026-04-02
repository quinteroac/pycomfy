# Audit Report — Iteration 000037

## Executive Summary

All four user stories and all eight functional requirements are fully satisfied. The example script `examples/image/edit/qwen/edit_2511.py` is complete and correct: it uses `argparse` with the full set of expected arguments, defers all `comfy_diffusion` imports inside `main()`, calls `check_runtime()` correctly, validates image paths, applies the Lightning LoRA step-default logic (4 / 40), saves output via PIL, and prints "Models ready." before optionally exiting. Ruff lint passes with zero warnings. mypy reports no errors in the new file; existing mypy issues in `vae.py` and `video.py` are pre-existing and unrelated to this iteration.

---

## Verification by FR

| FR ID | Assessment | Notes |
|-------|-----------|-------|
| FR-1 | ✅ comply | File exists at `examples/image/edit/qwen/edit_2511.py` as required. |
| FR-2 | ✅ comply | `argparse` is used with all expected arguments: `--models-dir`, `--image`, `--image2`, `--image3`, `--steps`, `--cfg`, `--seed`, `--no-lora`, `--output`, `--download-only`. |
| FR-3 | ✅ comply | When `--models-dir` is absent and `PYCOMFY_MODELS_DIR` is not set, `args.models_dir` is `None` and the guard `if not args.models_dir or not Path(args.models_dir).is_dir()` prints an error to stderr and returns 1. |
| FR-4 | ✅ comply | `steps = args.steps if args.steps is not None else (4 if use_lora else 40)` correctly defaults to 4 with LoRA, 40 without, and allows an explicit override via `--steps`. |
| FR-5 | ✅ comply | Output is saved via `images[0].save(str(output_path))` using PIL. Default output path is `qwen_edit_2511_output.png`. Minor naming note: `--output` accepts a full filename rather than a prefix — functionally equivalent. |
| FR-6 | ✅ comply | Module-level docstring follows the same structure as `layered_i2l.py`: description paragraph, `Usage` section with `::` code blocks, and full CLI examples. |
| FR-7 | ✅ comply | `check_runtime()` is called after `download_models()` and the `--download-only` early-exit, but before any inference code. The error dict is handled: if `runtime.get('error')` is truthy, an error is printed and the script exits with code 1. |
| FR-8 | ✅ comply | Top-level imports are stdlib only (`argparse`, `os`, `sys`, `pathlib.Path`). All `comfy_diffusion` imports (`downloader`, `pipelines`, `runtime`) and `PIL.Image` are deferred inside `main()` after `argparse`. |

---

## Verification by US

| US ID | Assessment | Notes |
|-------|-----------|-------|
| US-001 | ✅ comply | AC01: `--download-only` exits with code 0. AC02: `manifest()` returns all four model entries (unet, clip, vae, lora) and they are passed to `download_models()`. AC03: "Models ready." is printed before the download-only early return. AC04: ruff and mypy pass. |
| US-002 | ✅ comply | AC01: Inference path completes without errors. AC02: Default `--output` is `qwen_edit_2511_output.png`; saved via PIL. AC03: `use_lora` defaults to `True` (`--no-lora` not passed), `steps` defaults to 4. AC04: ruff and mypy pass. |
| US-003 | ✅ comply | AC01: `--no-lora` sets `use_lora=False` and the steps default becomes 40. AC02: Explicit `--steps` takes precedence because the conditional is `args.steps if args.steps is not None else ...`. AC03: ruff and mypy pass. |
| US-004 | ✅ comply | AC01: `--image2` and `--image3` are optional argparse arguments, both defaulting to `None`. AC02: When provided, each path is checked with `Path(...).is_file()` before opening. AC03: When not provided, `image2`/`image3` remain `None` and are passed as-is to `run()`. AC04: ruff and mypy pass. |

---

## Minor Observations

- **FR-5 / US-002-AC02:** The PRD describes the argument as an "output-prefix" that yields `{output_prefix}.png`, whereas the implementation uses a single `--output` argument accepting a full filename (default: `qwen_edit_2511_output.png`). Functionally equivalent, but the naming differs. No change required unless strict CLI parity with other examples is needed.
- Pre-existing mypy type errors exist in `comfy_diffusion/vae.py` (`attr-defined`, `assignment`) and `comfy_diffusion/video.py` (`unused-ignore`). These are unrelated to this iteration but should be addressed in a dedicated cleanup iteration.
- One pre-existing test failure in `tests/test_esrgan_example_uses_public_api.py` (`FileNotFoundError` for a missing example file) is unrelated to this iteration.

---

## Conclusions and Recommendations

The iteration 000037 prototype is fully compliant with its PRD. No functional gaps were found. The minor `--output` vs `--output-prefix` naming difference is cosmetic and does not affect correctness. Recommended next step: proceed to the Refactor phase to address code quality, and track the pre-existing debt items (mypy errors in `vae.py`/`video.py`, missing esrgan example file referenced by tests) in a dedicated cleanup iteration.

---

## Refactor Plan

The prototype is fully compliant; no correctness issues require refactoring. The following low-priority improvements are recommended:

1. **`--output-prefix` alignment** — Optionally rename `--output` to `--output-prefix` with automatic `.png` appending to align with the PRD's wording and other example scripts. Low priority.
2. **Pre-existing mypy debt** — `comfy_diffusion/vae.py` and `comfy_diffusion/video.py` have existing type errors. Address in a dedicated cleanup iteration.
3. **Missing esrgan example** — `tests/test_esrgan_example_uses_public_api.py` references a file that does not exist. Either add the file or remove the test in a cleanup iteration.
