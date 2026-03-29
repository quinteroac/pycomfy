# Project Context

<!-- Created or updated by `bun nvst create project-context`. Cap: 250 lines. -->

## Conventions
- Naming: snake_case for files, variables, functions; PascalCase for classes
- Formatting: no enforced formatter yet — follow PEP 8 conventions
- Git flow: feature branches per iteration (`feature/it_XXXXXX`), merge to `main` via PR
- Workflow: all agent commands via `bun nvst <command>`; all Python via `uv`
- Language: all generated resources must be in English

## Tech Stack
- Language: Python 3.12+
- Runtime: CPython
- Frameworks: none (pure library)
- Key libraries: torch (optional, via extras), ComfyUI (vendored submodule)
- Package manager: uv (no pip/venv)
- Build / tooling: pyproject.toml (PEP 621), uv for install/sync/run

## Code Standards
- Style: PEP 8, type hints on public API
- Error handling: `check_runtime()` returns error dicts (no exceptions for expected failures); `python_version` always populated
- Module organisation: src-less layout — `comfy_diffusion/` package at repo root, vendored deps in `vendor/`
- Forbidden patterns: no hardcoded torch versions; no manual `sys.path` manipulation outside `_runtime.py`; no pip/venv commands; no `torch`/`comfy.*` imports at module top level (lazy import pattern)
- `path` type annotation: `str | Path` across all loaders — do not change to `str | os.PathLike` without updating all occurrences

## Testing Strategy
- Approach: critical paths only
- Runner: pytest (via `uv run pytest`)
- Coverage targets: none enforced
- Test location: `tests/` at repo root
- Constraint: CI is CPU-only — all tests must pass without GPU; use mocks/stubs for model weights

## Product Architecture
- comfy-diffusion is a standalone Python library exposing ComfyUI's inference engine as importable modules
- No server, no UI, no application layer — import and run inference in your own code
- ComfyUI vendored as git submodule at `vendor/ComfyUI`, pinned to `COMFYUI_PINNED_TAG` in `_runtime.py`
- `check_runtime()` auto-downloads ComfyUI from GitHub if `vendor/ComfyUI` is missing/empty (idempotent; stdlib only: `urllib.request`, `zipfile`, `shutil`, `pathlib`, `tempfile`)

### Data Flow
1. Consumer does `import comfy_diffusion`
2. `comfy_diffusion/__init__.py` adds `vendor/ComfyUI` to `sys.path` (absolute paths from `__file__`)
3. Consumer calls `check_runtime()` — bootstraps ComfyUI if absent, returns diagnostics dict
4. ComfyUI internals (e.g. `comfy.model_management`) become importable; consumer uses library API

## Modular Structure
- `comfy_diffusion/_runtime.py`: path management, `COMFYUI_PINNED_TAG`, auto-bootstrap logic
- `comfy_diffusion/runtime.py`: public `check_runtime()` entry point
- `comfy_diffusion/models.py`: `ModelManager` — load_checkpoint, load_clip(*paths), load_vae, load_llm
- `comfy_diffusion/conditioning.py`: encode_prompt, advanced/regional/scheduled conditioning (Flux, WAN, LTXV)
- `comfy_diffusion/sampling.py`: sample(), advanced samplers, custom schedulers, sigma tools
- `comfy_diffusion/vae.py`: vae_decode/encode (single, tiled, batch variants)
- `comfy_diffusion/lora.py`: apply_lora, LoRA stacking
- `comfy_diffusion/controlnet.py`: load_controlnet, apply_controlnet (ControlNetApplyAdvanced, SetUnionControlNetType)
- `comfy_diffusion/latent.py`: latent create/resize/crop/compose/batch utilities
- `comfy_diffusion/image.py`: image load/save/transform utilities
- `comfy_diffusion/mask.py`: mask load/convert/grow/feather utilities
- `comfy_diffusion/audio.py`: LTXV Audio VAE + ACE Step 1.5 text-to-audio
- `comfy_diffusion/textgen.py`: generate_text(), generate_ltx2_prompt() (LLM/VLM inference)
- `comfy_diffusion/video.py`: video CFG guidance, model sampling patches (Flux, SD3, AuraFlow)
- `vendor/ComfyUI/`: vendored ComfyUI (submodule or auto-downloaded zip, not edited directly)
- `tests/`: pytest test files

## ComfyUI API Notes
- `clip.encode_from_tokens_scheduled(tokens)` — canonical for text conditioning (mirrors `CLIPTextEncode`)
- `clip.encode_from_tokens(tokens, return_pooled=True)` — GLIGEN only; do not use for standard encoding

## Public API Pattern
- Re-exported at package level: `check_runtime`, `vae_decode`, `vae_encode`, `apply_lora`
- All other symbols: explicit submodule imports (e.g. `from comfy_diffusion.conditioning import encode_prompt`)
- `textgen`, `audio`, `latent`, `image`, `mask`, `video`, `controlnet` — not re-exported at package level

## Pipeline Authoring Rules

When implementing a pipeline based on a ComfyUI workflow file, follow these rules strictly:

### 1. Always read the workflow first with the workflow-reader skill
Before writing any code, run:
```bash
python comfy_diffusion/skills/workflow-reader/tools/workflow.py <workflow.json>
```
This reveals the exact nodes, parameters, connections, execution order, and model downloads. Never assume — always read.

### 2. The workflow is the source of truth
- `manifest()` must list **exactly** the files declared in `get_model_downloads()` that belong to **active** (non-bypassed) nodes.
- HF repos, filenames, and destination directories must match the `url` and `directory` fields in the workflow metadata — not guessed or inferred from model names.
- `run()` must mirror the node execution order shown by `get_nodes()`. Do not reorder, skip, or combine steps unless the workflow itself does so.

### 3. Check node mode before including in manifest or run()
A node with `mode = "bypassed"` or `"muted"` is **not executed**. Do not add its models to `manifest()` and do not implement its logic in `run()`. This is the most common source of manifest drift.

### 4. Sampler, sigmas, and CFG values come from the workflow
- Sampler name: read from `KSamplerSelect` widget value.
- Sigmas string: read from `ManualSigmas` widget value.
- CFG: read from `CFGGuider` widget value.
- Never substitute defaults from other pipelines without verifying against the workflow.

### 5. Multi-pass pipelines must implement every pass
If the workflow contains multiple `SamplerCustomAdvanced` nodes, each is a distinct sampling pass. Implement all of them in order, including any upscaling or image re-injection between passes.

### 6. If a required node is not yet wrapped, implement it first
Before writing the pipeline, check whether every node used in the workflow has a corresponding function in `comfy_diffusion/`. If a node is missing:
- Implement it in the appropriate module (`conditioning.py`, `latent.py`, `audio.py`, `video.py`, etc.) following the standard lazy-import pattern.
- Add it to the module's `__all__`.
- Do **not** inline raw `comfy.*` calls inside a pipeline file — all ComfyUI node logic belongs in the library modules.
- Only then proceed to write the pipeline.

### 7. LoRA application order and strength matter
Check each `LoraLoader` / `LoraLoaderModelOnly` node:
- `LoraLoaderModelOnly` → `apply_lora(model, clip, path, strength, 0.0)` (clip strength = 0)
- `LoraLoader` → `apply_lora(model, clip, path, strength_model, strength_clip)`
- Apply in the order given by `order` field. Stack multiple calls when there are multiple LoRA nodes.

## Implemented Capabilities
<!-- Updated at the end of each iteration -->

### Summary — it_000001 through it_000017
- **001** Foundation: package structure, `check_runtime()`, ComfyUI path management
- **002** `models`: `ModelManager`, `load_checkpoint()`
- **003** `conditioning`: `encode_prompt()` via CLIP
- **004** `sampling`: `sample()` KSampler wrapper
- **005** `vae`: `vae_decode()` latent→PIL
- **006** `lora`: `apply_lora()`, LoRA stacking
- **007** `vae`: `vae_encode()` + standalone `load_clip()`, `load_vae()`
- **008** `vae`: tiled encode/decode (`vae_decode_tiled`, `vae_encode_tiled`)
- **009** `vae`: batch variants (`vae_decode_batch`, `vae_encode_batch`, tiled batch)
- **010** `sampling`: advanced samplers (KSamplerAdvanced, SamplerCustomAdvanced), schedulers, sigma tools
- **011** `audio`: LTXV Audio VAE, ACE Step 1.5 text-to-audio conditioning
- **012** `conditioning`: advanced/regional/scheduled conditioning; Flux, WAN, LTXV architecture support
- **013** `controlnet`: ControlNet load & apply (ControlNetApplyAdvanced, SetUnionControlNetType)
- **014** `latent`: latent create/resize/crop/compose/batch utilities
- **015** `image`: image load/save/transform utilities
- **016** `mask`: mask load/convert/grow/feather utilities
- **017** `video`: model sampling patches (Flux, SD3, AuraFlow), video CFG guidance (linear/triangle)

### it_000018 — Packaging & DX
- pip-installable package (`pyproject.toml` PEP 621), distributable skills, type stubs

### it_000019 — Variadic CLIP Loader
- `ModelManager.load_clip(*paths)` — variadic, wraps `comfy.sd.load_clip(ckpt_paths=[...])`

### it_000020 — LLM/VLM Text Generation (`textgen`)
- `ModelManager.load_llm(path)` — loads LLM via ComfyUI CLIP loader; registers `models_dir/llm`
- `generate_text(clip, prompt, ...)` — LLM inference, mirrors `TextGenerate.execute()`
- `generate_ltx2_prompt(clip, prompt, ...)` — LTX-Video 2 prompt enhancer with system prompt templates

### it_000021 — Auto-Bootstrap ComfyUI
- `COMFYUI_PINNED_TAG` constant in `_runtime.py` — pinned ComfyUI release tag
- `check_runtime()` auto-downloads ComfyUI zip from GitHub when `vendor/ComfyUI` is missing/empty
- Uses stdlib only: `urllib.request`, `zipfile`, `shutil`, `pathlib`, `tempfile`
- Idempotent: skips download if `vendor/ComfyUI` is already populated
- Returns error dict (with `python_version`) on bootstrap failure; never raises
- README documents `check_runtime()` as first call before any model loading

### it_000024 — Manifest-Based Downloader & Reference Pipeline
- `comfy_diffusion/downloader.py`: typed entry dataclasses (`HFModelEntry`, `CivitAIModelEntry`, `URLModelEntry`) + `download_models(manifest, *, models_dir, quiet)`
- SHA256 integrity verification via stdlib `hashlib`; idempotent (skips existing files, re-downloads on hash mismatch)
- `tqdm` progress for URL/CivitAI downloads; `huggingface_hub` native progress for HF downloads; all suppressed with `quiet=True`
- **CivitAI design decision**: `civitai-py` library is NOT used. It targets the Generator API (cloud inference), not model file downloads. CivitAI downloads use the CivitAI REST API directly via `urllib.request` with `Authorization: Bearer {CIVITAI_API_KEY}`. This is the same mechanism the library uses internally, avoids an extra dependency, and allows tqdm progress.
- `comfy_diffusion/pipelines/ltx2_t2v.py`: canonical pipeline pattern — exports `manifest() -> list[ModelEntry]` and `run(...) -> list[PIL.Image]`
- `[downloader]` optional extra: `huggingface_hub>=0.20`, `tqdm>=4.67` (install with `pip install comfy-diffusion[downloader]`)
