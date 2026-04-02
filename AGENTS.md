# Agents entry point

- **What this project is:** `comfy-diffusion` is a standalone Python library that exposes ComfyUI's inference engine (`comfy.*` modules) as importable Python modules — no server, no node graph, no UI layer. It is consumed exactly like `diffusers` or `DiffSynth-Studio`: `import comfy_diffusion` and run inference directly in your own code. ComfyUI is vendored as a git submodule at `vendor/ComfyUI` and its internal modules are made importable transparently on `import comfy_diffusion`. The library is designed to be a single `pip`/`uv` dependency that any Python application (FastAPI backend, script, pipeline) can add without operating a separate ComfyUI server.

- **How to work here:** Use this file as the single entry point. Follow the process phases in order; read and update `.agents/state.json` for the current iteration and phase. Invoke the skills under `.agents/skills/` as indicated by each NVST command. All iteration artifacts live in `.agents/flow/` with the naming `it_` + 6-digit iteration number (e.g. `it_000001_product-requirement-document.md`). From the second iteration onward, adhere to [`.agents/PROJECT_CONTEXT.md`](.agents/PROJECT_CONTEXT.md). **Python:** use [uv](https://docs.astral.sh/uv/) for all install, run, and dependency commands (`uv sync`, `uv run`, `uv add`) — never use `pip` or `venv` directly. **NVST:** run all agent/workflow commands with [Bun](https://bun.sh) as `bun nvst <command>` (see `docs/nvst-flow/`).

- **Process:** Define → Prototype → Refactor (see `docs/nvst-flow/` or package documentation).

- **Project context:** [`.agents/PROJECT_CONTEXT.md`](.agents/PROJECT_CONTEXT.md) — conventions, architecture decisions, and modular structure; the agent adheres from the second iteration onward.

- **Roadmap and node inventory:** [`ROADMAP.md`](ROADMAP.md) — full iteration plan, node classification (Roadmap / Nice-to-have / Discarded), and optional dependency schema.

- **Key architecture decisions (do not revisit without explicit instruction):**
  - ComfyUI is vendored at `vendor/ComfyUI` as a git submodule pinned to a stable release tag — never floating on `master`. Update the pin deliberately between iterations only.
  - `sys.path` manipulation is encapsulated entirely inside `comfy_diffusion/_runtime.py` — consumers never touch paths manually. Use absolute paths derived from `__file__`.
  - The node system (`nodes.py`, custom nodes) is never loaded — comfy-diffusion imports `comfy.*` modules directly only.
  - `torch` is an optional dependency declared as extras (`comfy-diffusion[cuda]` / `comfy-diffusion[cpu]`) — never hardcode a torch version or index URL in core dependencies.
  - `check_runtime()` returns an error dict (never raises) when the ComfyUI submodule is not initialized. `python_version` is always populated regardless.
  - All tests must pass on CPU-only environments — CI has no GPU. GPU is validated locally before merging.
  - Test approach: critical paths only (pytest via `uv run pytest`). Test plans are written after prototyping, during the Refactor phase.
  - Git flow: feature branches per iteration (`feature/it-000001-foundation`), merged to `main` via PR.
  - Public API pattern: modules are not auto-imported from `__init__.py` by default. Exceptions are `check_runtime`, `vae_decode`, `vae_encode`, and `apply_lora` which are re-exported for convenience. All other symbols use explicit submodule imports (e.g. `from comfy_diffusion.conditioning import encode_prompt`).
  - Lazy import pattern: no `torch`, `comfy.*`, or `ensure_comfyui_on_path()` at module top level — all deferred to call time inside function bodies. Exception: `vae.py` uses pure duck typing (no comfy import at all) — both patterns are valid.
  - Inference mode ownership: `torch.inference_mode()` is enforced centrally in core execution APIs (`sampling.py`, `vae.py`, and relevant `audio.py` wrappers). Pipeline authors must not duplicate inference-mode wrappers in each `run()` implementation.
  - `path` type annotation: `str | Path` is the established pattern across `ModelManager`, `load_checkpoint`, and `apply_lora`. Do not change to `str | os.PathLike` unless updating all occurrences simultaneously in a dedicated cleanup iteration.
  - No high-level pipeline abstraction: comfy-diffusion is a modular runtime library. There is no `ImagePipeline` or equivalent. Callers compose the building blocks directly. This is intentional — the modularity is the feature.
  - External libraries over node ports: prefer `Pillow`, `numpy`, `opencv-python`, `torchaudio` for image transforms, mask ops, video I/O, and audio I/O respectively. Only wrap comfy nodes when they provide non-trivial logic (VAE, samplers, model patches, conditioning). See ROADMAP.md for the full classification.

- **Rule:** All generated resources in this repo must be in English.
- For any file search or grep in the current git indexed directory use fff tools