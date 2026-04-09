# Project Context

<!-- Created or updated by `bun nvst create project-context`. Cap: 250 lines. -->

## Conventions
- Naming: snake_case for files, variables, functions; PascalCase for classes
- Formatting: no enforced formatter yet — follow PEP 8 conventions; ruff for lint
- Git flow: feature branches per iteration (`feature/it_XXXXXX`), merge to `master` via PR
- Workflow: all agent commands via `bun nvst <command>`; all Python via `uv`
- Language: all generated resources must be in English

## Monorepo Structure
This repo is a Python monorepo: a core library + application layers, all in Python.

```
comfy-diffusion/
├── comfy_diffusion/       # Python core library (PyPI: comfy-diffusion)
├── server/                # FastAPI worker — job queue, SSE progress stream, :5000
├── cli/                   # Python CLI (Typer) — `parallax` entry point
├── mcp/                   # Python MCP server (FastMCP) — `parallax-mcp` entry point
├── parallax/              # Thin shim for `python -m parallax`
├── install.sh             # One-command Linux/macOS installer
├── install.ps1            # One-command Windows installer
├── parallax.spec          # PyInstaller spec — standalone binary (no Python required)
└── docs/                  # Per-package documentation
```

Request flow: `parallax CLI / parallax-mcp → server/ FastAPI (:5000) → comfy_diffusion`

## Tech Stack
### Python (comfy_diffusion + server/ + cli/ + mcp/)
- Language: Python 3.12+
- Runtime: CPython
- Frameworks: FastAPI + uvicorn (server/), Typer (cli/), FastMCP (mcp/), none (core library)
- Key libraries: torch (optional, via extras), ComfyUI (vendored submodule)
- Database: SQLite via aiosqlite — job persistence at `~/.config/parallax/jobs.db`
  (overridden by `PARALLAX_DB_PATH` env var)
- Package manager: uv (no pip/venv)
- Build / tooling: pyproject.toml (PEP 621), uv for install/sync/run,
  PyInstaller + parallax.spec for standalone binary
- Scripts: `parallax = "cli.main:app"`, `parallax-mcp = "mcp.main:main"`

### TypeScript (scripts only)
- Runtime: Bun — only for version-bump scripts (`scripts/`)
- No application code in TypeScript; `packages/` no longer exists

## Code Standards
- Style: PEP 8, type hints on public API, ruff for lint
- Error handling: `check_runtime()` returns error dicts (no exceptions for expected failures)
- Module organisation: src-less layout — `comfy_diffusion/` package at repo root, vendored deps in `vendor/`
- Forbidden patterns: no hardcoded torch versions; no manual `sys.path` manipulation outside `_runtime.py`; no pip/venv commands; no `torch`/`comfy.*` imports at module top level (lazy import pattern)
- `path` type annotation: `str | Path` across all loaders

## Testing Strategy
- Approach: critical paths only
- Runner: pytest (via `uv run pytest`)
- Coverage targets: none enforced
- Test location: `tests/` at repo root; test files named `test_<scope>_it<NNNNNN>.py` for iteration tests
- Constraint: CI is CPU-only — all tests must pass without GPU; use mocks/stubs for model weights

## Product Architecture
- comfy-diffusion is a standalone Python library exposing ComfyUI's inference engine as importable modules
- `server/` is a FastAPI application that wraps comfy_diffusion: job queue (aiosqlite), SSE progress stream, REST API for sync and async inference
- `cli/` is a Typer CLI that talks to `server/` — supports sync and `--async` mode, binary distributionas self-contained executable via PyInstaller
- `mcp/` is a FastMCP server that exposes inference tools to Claude (calls `server/` endpoints)
- ComfyUI vendored as git submodule at `vendor/ComfyUI`
- `parallax install` bootstraps the runtime to `~/.parallax/env`; `parallax ms install` registers server as systemd/launchd service

### Data Flow (inference)
1. `parallax create image|video|audio [--async]` → POST to `server/:5000`
2. `server/` enqueues the job in SQLite, spawns a subprocess via uv
3. Subprocess imports `comfy_diffusion` pipeline and runs inference
4. Progress reported via SSE (`/jobs/{id}/stream`); result available at `/jobs/{id}`

## Modular Structure
- `comfy_diffusion/_runtime.py`: path management, `COMFYUI_PINNED_TAG`, auto-bootstrap logic
- `comfy_diffusion/runtime.py`: public `check_runtime()` entry point
- `comfy_diffusion/models.py`: `ModelManager` — load_checkpoint, load_clip(*paths), load_vae, load_llm, load_upscale_model, load_audio_encoder, load_latent_upscale_model
- `comfy_diffusion/downloader.py`: `HFModelEntry`, `CivitAIModelEntry`, `URLModelEntry` + `download_models()`
- `comfy_diffusion/conditioning.py`: encode_prompt, advanced/regional/scheduled conditioning (Flux, WAN, LTXV)
- `comfy_diffusion/sampling.py`: sample(), advanced samplers, custom schedulers, sigma tools
- `comfy_diffusion/vae.py`: vae_decode/encode (single, tiled, batch variants)
- `comfy_diffusion/lora.py`: apply_lora, LoRA stacking
- `comfy_diffusion/controlnet.py`: load_controlnet, apply_controlnet
- `comfy_diffusion/latent.py`: latent create/resize/crop/compose/batch utilities
- `comfy_diffusion/image.py`: image load/save/transform utilities
- `comfy_diffusion/mask.py`: mask load/convert/grow/feather utilities
- `comfy_diffusion/audio.py`: LTXV Audio VAE + ACE Step 1.5 text-to-audio
- `comfy_diffusion/textgen.py`: generate_text(), generate_ltx2_prompt() (LLM/VLM inference)
- `comfy_diffusion/video.py`: video CFG guidance, model sampling patches (Flux, SD3, AuraFlow)
- `comfy_diffusion/pipelines/`: hierarchical pipeline library — `image/` (sdxl, anima, z_image, flux_klein, qwen), `video/` (ltx/ltx2, ltx/ltx23, wan/wan21, wan/wan22), `audio/` (ace_step)
- `server/gateway.py`: APIRouter — inference endpoints (POST /create/*, /jobs/*)
- `server/main.py`: FastAPI app + CORS middleware
- `server/job_queue.py`: aiosqlite job store; `PARALLAX_DB_PATH` env override
- `server/submit.py`: subprocess launcher (uv run) for inference jobs
- `server/schemas.py`: Pydantic request/response models
- `cli/main.py`: Typer app — create, edit, jobs, mcp, ms, upscale, install, async
- `cli/_runners/`: per-media type HTTP call helpers (image, video, audio, edit_image)
- `cli/commands/install.py`: `parallax install` — bootstraps `~/.parallax/env`
- `cli/commands/ms.py`: `parallax ms install` — registers server as systemd/launchd service
- `cli/commands/mcp.py`: `parallax mcp install` — registers MCP server
- `mcp/main.py`: FastMCP server; tools: create_image, create_video, create_audio, edit_image, upscale_image, get_job_status, wait_for_job
- `vendor/ComfyUI/`: vendored ComfyUI (submodule or auto-downloaded zip, not edited directly)
- `tests/`: pytest test files

## ComfyUI API Notes
- `clip.encode_from_tokens_scheduled(tokens)` — canonical for text conditioning (mirrors `CLIPTextEncode`)
- `clip.encode_from_tokens(tokens, return_pooled=True)` — GLINGEN only; do not use for standard encoding

## Public API Pattern
- Re-exported at package level: `check_runtime`, `apply_lora`; full `vae` surface
- All other symbols: explicit submodule imports (e.g. `from comfy_diffusion.conditioning import encode_prompt`)

## Pipeline Authoring Rules
### 1. Always read the workflow first with the workflow-reader skill
```bash
python comfy_diffusion/skills/workflow-reader/tools/workflow.py <workflow.json>
```
### 2. The workflow is the source of truth
- `manifest()` must list exactly the files from active (non-bypassed) nodes
- HF repos, filenames, and destination directories must match workflow metadata
- `run()` must mirror the node execution order
### 3. Check node mode before including in manifest or run()
A node with `mode = "bypassed"` or `"muted"` is not executed.
### 4. Sampler, sigmas, and CFG values come from the workflow
### 5. Multi-pass pipelines must implement every pass
### 6. If a required node is not yet wrapped, implement it first in the library
### 7. LoRA application order and strength matter
- `LoraLoaderModelOnly` → `apply_lora(model, clip, path, strength, 0.0)`
- `LoraLoader` → `apply_lora(model, clip, path, strength_model, strength_clip)`

## Implemented Capabilities

### it_000001–037 (archived summary)
- **001–021** Foundation, `check_runtime`, `models`, `conditioning`, `sampling`, `vae`, `lora`, `audio`, `controlnet`, `latent`, `image`, `mask`, `video`, `textgen`, `downloader`, auto-bootstrap
- **022–026** Upscale model, ComfyUI v0.18.0
- **027–034** LTX-Video 2, WAN 2.1/2.2, audio encoder; pipelines: ltx2, ltx23 (t2v/i2v/flf2v), wan21, wan22, sdxl, anima, z_image
- **035** ACE Step, Flux.2 Klein, Qwen layered; pipelines: flux_klein, qwen/layered, audio/ace_step/v1_5
- **036** `image/qwen/edit_2511` pipeline
- **037** `examples/qwen_edit_2511.py` CLI example

### it_000038–043 — Python Application Layer (replaced TypeScript)
- TypeScript `packages/parallax_cli`, `packages/parallax_ms`, `packages/parallax_mcp` — **removed in it_000044**
- Work in these iterations informed the Python rewrite architecture

### it_000044 — Python CLI + MCP + Server rewrite
- `cli/` — Typer CLI: `parallax create image|video|audio`, `edit image`, `upscale`, `jobs`, `async`
- `mcp/` — FastMCP server: `create_image`, `create_video`, `create_audio`, `edit_image`, `upscale_image`, `get_job_status`, `wait_for_job`
- `server/` — FastAPI with SQLite job queue, SSE progress stream (`/jobs/{id}/stream`), non-blocking inference via subprocess
- `parallax async <cmd>` — enqueues any command as async job, returns job ID

### it_000045 — Standalone Binary + Installer
- `parallax.spec` — PyInstaller spec; produces self-contained `parallax` binary (~50 MB, no Python required)
- `cli/commands/install.py` — `parallax install`: downloads comfy-diffusion runtime to `~/.parallax/env`
- `cli/commands/ms.py` — `parallax ms install`: registers FastAPI server as systemd (Linux) or launchd (macOS) service
- `cli/commands/mcp.py` — `parallax mcp install`: registers MCP server
- `install.sh` / `install.ps1` — one-command installers for Linux/macOS and Windows
- `.github/workflows/release-cli.yml` — CI builds and publishes binaries on version tags
- `cli/_version.py` — version constant baked in at PyInstaller build time

### it_000046 — LTX-23 ia2v + CLI --audio option
- `comfy_diffusion/pipelines/video/ltx/ltx23/ia2v.py` — image+audio-to-video pipeline (22B dev fp8, two-pass AV)
- `cli/commands/create.py` — `--audio` option on `parallax create video --model ltx23`
- Routes to `ia2v` pipeline when `--audio` is provided with `ltx23`
