# Agents entry point

- **What this project is:** `pycomfy` is a standalone Python library that exposes ComfyUI's inference engine (`comfy.*` modules) as importable Python modules — no server, no node graph, no UI layer. It is consumed exactly like `diffusers` or `DiffSynth-Studio`: `import pycomfy` and run inference directly in your own code. ComfyUI is vendored as a git submodule at `vendor/ComfyUI` and its internal modules are made importable transparently on `import pycomfy`. The library is designed to be a single `pip`/`uv` dependency that any Python application (FastAPI backend, script, pipeline) can add without operating a separate ComfyUI server.

- **How to work here:** Use this file as the single entry point. Follow the process phases in order; read and update `.agents/state.json` for the current iteration and phase. Invoke the skills under `.agents/skills/` as indicated by each NVST command. All iteration artifacts live in `.agents/flow/` with the naming `it_` + 6-digit iteration number (e.g. `it_000001_product-requirement-document.md`). From the second iteration onward, adhere to [`.agents/PROJECT_CONTEXT.md`](.agents/PROJECT_CONTEXT.md). **Python:** use [uv](https://docs.astral.sh/uv/) for all install, run, and dependency commands (`uv sync`, `uv run`, `uv add`) — never use `pip` or `venv` directly. **NVST:** run all agent/workflow commands with [Bun](https://bun.sh) as `bun nvst <command>` (see `docs/nvst-flow/`).

- **Process:** Define → Prototype → Refactor (see `docs/nvst-flow/` or package documentation).

- **Project context:** [`.agents/PROJECT_CONTEXT.md`](.agents/PROJECT_CONTEXT.md) — conventions, architecture decisions, and modular structure; the agent adheres from the second iteration onward.

- **Key architecture decisions (do not revisit without explicit instruction):**
  - ComfyUI is vendored at `vendor/ComfyUI` as a git submodule pinned to a stable release tag — never floating on `master`. Update the pin deliberately between iterations only.
  - `sys.path` manipulation is encapsulated entirely inside `pycomfy/_runtime.py` — consumers never touch paths manually. Use absolute paths derived from `__file__`.
  - The node system (`nodes.py`, custom nodes) is never loaded — pycomfy imports `comfy.*` modules directly only.
  - `torch` is an optional dependency declared as extras (`pycomfy[cuda]` / `pycomfy[cpu]`) — never hardcode a torch version or index URL in core dependencies.
  - `check_runtime()` returns an error dict (never raises) when the ComfyUI submodule is not initialized. `python_version` is always populated regardless.
  - All tests must pass on CPU-only environments — CI has no GPU. GPU is validated locally before merging.
  - Test approach: critical paths only (pytest via `uv run pytest`). Test plans are written after prototyping, during the Refactor phase.
  - Git flow: feature branches per iteration (`feature/it-000001-foundation`), merged to `main` via PR.

- **Iteration plan (summary):**

  | # | Module | Goal |
  |---|--------|-------|
  | 01 | `_runtime` / `check_runtime()` | Package foundation + ComfyUI vendoring |
  | 02 | `models` | Checkpoint / VAE / CLIP loading |
  | 03 | `conditioning` | Prompt encoding, CLIP, weighting |
  | 04 | `sampling` | KSampler, schedulers, seeds |
  | 05 | `vae` | Encode image→latent, decode latent→PIL |
  | 06 | `lora` | LoRA loading and stacking |
  | 07 | `pipeline` | High-level `ImagePipeline` API |
  | 08 | `queue` | Async / asyncio / progress callbacks |
  | 09 | `plugins` | Optional capability plugin system |
  | 10 | packaging | pip-installable, type stubs, DX |

- **Rule:** All generated resources in this repo must be in English.