# Project Context

<!-- Created or updated by `bun nvst create project-context`. Cap: 250 lines. -->

## Conventions
- Naming: snake_case for files, variables, functions; PascalCase for classes
- Formatting: no enforced formatter yet — follow PEP 8 conventions
- Git flow: feature branches per iteration (`feature/it_XXXXXX`), merge to `main` via PR
- Workflow: all agent commands via `bun nvst <command>`; all Python via `uv`
- Language: all generated resources must be in English

## Monorepo Structure
This repo is a polyglot monorepo: a Python core library + TypeScript application packages.

```
comfy-diffusion/
├── comfy_diffusion/       # Python core library (PyPI: comfy-diffusion)
├── server/                # FastAPI worker — wraps comfy_diffusion, exposes HTTP on :5000
├── packages/
│   ├── parallax_sdk/      # @parallax/sdk — shared TypeScript types (request/response contracts)
│   ├── parallax_ms/       # @parallax/ms  — Elysia gateway on :3000, proxies to server/
│   ├── parallax_cli/      # @parallax/cli — Bun CLI, talks to parallax_ms
│   └── parallax_mcp/      # @parallax/mcp — MCP server for Claude, talks to parallax_ms
└── docs/                  # Per-package documentation
```

Request flow: `parallax_cli / parallax_mcp → parallax_ms (:3000) → server/FastAPI (:5000) → comfy_diffusion`

## Tech Stack
### Python (comfy_diffusion + server/)
- Language: Python 3.12+
- Runtime: CPython
- Frameworks: FastAPI + uvicorn (server/), none (core library)
- Key libraries: torch (optional, via extras), ComfyUI (vendored submodule)
- Package manager: uv (no pip/venv)
- Build / tooling: pyproject.toml (PEP 621), uv for install/sync/run

### TypeScript (packages/)
- Runtime: Bun
- Gateway framework: Elysia (parallax_ms)
- CLI framework: commander (parallax_cli)
- MCP framework: @modelcontextprotocol/sdk (parallax_mcp)
- Shared types: @parallax/sdk (workspace:*)
- Workspace manager: Bun workspaces (package.json root)

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
- `comfy_diffusion/models.py`: `ModelManager` — load_checkpoint, load_clip(*paths), load_vae, load_llm, load_upscale_model, load_audio_encoder, load_latent_upscale_model
- `comfy_diffusion/downloader.py`: `HFModelEntry`, `CivitAIModelEntry`, `URLModelEntry` + `download_models()` — manifest-based model downloader with SHA256 verification
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
- `comfy_diffusion/pipelines/`: hierarchical pipeline library — `image/` (sdxl, anima, z_image, flux_klein, qwen), `video/` (ltx/ltx2, ltx/ltx23, wan/wan21, wan/wan22), `audio/` (ace_step); each module exports `manifest() -> list[ModelEntry]` + `run()`
- `vendor/ComfyUI/`: vendored ComfyUI (submodule or auto-downloaded zip, not edited directly)
- `tests/`: pytest test files

## ComfyUI API Notes
- `clip.encode_from_tokens_scheduled(tokens)` — canonical for text conditioning (mirrors `CLIPTextEncode`)
- `clip.encode_from_tokens(tokens, return_pooled=True)` — GLIGEN only; do not use for standard encoding

## Public API Pattern
- Re-exported at package level: `check_runtime`, `apply_lora`; full `vae` surface: `vae_decode`, `vae_decode_tiled`, `vae_decode_batch`, `vae_decode_batch_tiled`, `vae_encode`, `vae_encode_tiled`, `vae_encode_batch`, `vae_encode_batch_tiled`, `vae_encode_for_inpaint`
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

### Summary — it_000001 through it_000024
- **001** Foundation: `check_runtime()`, `_runtime.py`, ComfyUI path management
- **002–007** `models`, `conditioning`, `sampling`, `vae` (encode/decode), `lora`
- **008–010** `vae` tiled + batch variants; advanced samplers, custom schedulers, sigma tools
- **011–013** `audio` (LTXV VAE + ACE Step conditioning); `conditioning` advanced/regional; `controlnet`
- **014–017** `latent`, `image`, `mask`, `video` (patches + CFG guidance)
- **018** Packaging: PEP 621, type stubs, distributable skills
- **019** `ModelManager.load_clip(*paths)` — variadic CLIP loader
- **020** `textgen`: `generate_text()`, `generate_ltx2_prompt()` (LLM/VLM)
- **021** Auto-bootstrap: `check_runtime()` downloads ComfyUI when `vendor/ComfyUI` is absent
- **022–023** Incremental refinements
- **024** `downloader.py`: manifest downloader (`HFModelEntry`, `CivitAIModelEntry`, `URLModelEntry`, SHA256, `tqdm`); CivitAI uses REST API directly (no `civitai-py`)

### it_000025–026 — Upscale Model + ComfyUI v0.18.0
- `ModelManager.load_upscale_model(path)`, `upscale_models` folder registered
- ComfyUI submodule updated to `v0.18.0`

### it_000027–034 — LTX-Video 2, WAN 2.1/2.2, Audio Encoder
- `latent`: `ltxv_empty_latent_video`, `ltxv_latent_upsample`, `ltxv_crop_guides`
- `audio`: `ltxv_concat_av_latent`, `ltxv_separate_av_latent`, `audio_encoder_encode`, `vae_decode_audio`
- `sampling`: `manual_sigmas`; `conditioning`: `wan22_image_to_video_latent`
- `ModelManager`: `load_audio_encoder`, `load_latent_upscale_model`
- Pipelines: `video/ltx/ltx2` (t2v, i2v, distilled, a2v, depth, canny, pose, lora), `video/ltx/ltx23` (t2v, i2v, flf2v, ia2v), `video/wan/wan21` (t2v, i2v, flf2v), `video/wan/wan22` (t2v, i2v, flf2v, s2v, ti2v), `image/sdxl` (t2i, turbo, refiner), `image/anima`, `image/z_image`

### it_000035 — ACE Step + Flux.2 Klein + new library wrappers
- `latent`: `empty_flux2_latent_image`, `empty_qwen_image_layered_latent_image`
- `conditioning`: `reference_latent`; `sampling`: `flux_kv_cache`
- `image`: `image_scale_to_total_pixels`, `image_scale_to_max_dimension`, `get_image_size`
- Pipelines: `image/flux_klein` (t2i_4b_base/distilled, edit_4b/9b base/distilled, edit_9b_kv), `image/qwen/layered`
- Pipelines: `audio/ace_step/v1_5` (checkpoint, split, split_4b)

### it_000036 — Qwen Image Edit 2511
- `image/qwen/edit_2511` pipeline — Qwen2.5-VL image editing
- 4 new library node wrappers across conditioning, latent, sampling, image modules

### it_000037 — Qwen edit_2511 CLI example
- `examples/qwen_edit_2511.py` — self-contained executable annotating pipeline public API (`manifest()` + `run()`)
