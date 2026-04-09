# comfy-diffusion

[![PyPI version](https://badge.fury.io/py/comfy-diffusion.svg)](https://pypi.org/project/comfy-diffusion/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://pypi.org/project/comfy-diffusion/)
[![CI](https://github.com/quinteroac/comfy-diffusion/actions/workflows/publish.yml/badge.svg)](https://github.com/quinteroac/comfy-diffusion/actions/workflows/publish.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

A Python library that exposes ComfyUI's inference engine as importable modules — no server, no node graph, no UI.

```python
from comfy_diffusion import check_runtime

print(check_runtime())
# {"comfyui_version": "0.9.x", "device": "cuda:0", "vram_total_mb": 8192, ...}
```

---

## Monorepo packages

This repo contains the core Python library plus a set of Python application packages built on top of it.

| Package | Language | Description | Docs |
|---------|----------|-------------|------|
| `comfy_diffusion` | Python | Core inference library (this repo) | [docs/comfy_diffusion.md](docs/comfy_diffusion.md) |
| `server/` | Python | FastAPI worker — HTTP interface to the core on `:5000` | [docs/server.md](docs/server.md) |
| `cli/` | Python | Standalone binary CLI — `parallax install / comfyui / create / edit / …` | [docs/parallax_cli.md](docs/parallax_cli.md) |
| `mcp/` | Python | FastMCP server for Claude Desktop — registered via `parallax mcp install` | [docs/parallax_mcp.md](docs/parallax_mcp.md) |

Request flow:
```
parallax_cli / parallax_mcp  →  server/FastAPI (:5000)  →  comfy_diffusion
```

---

## Install the parallax CLI

The `parallax` CLI is distributed as a single self-contained binary — no Python installation required on the target machine.

**Linux / macOS (one command):**
```sh
curl -fsSL https://raw.githubusercontent.com/quinteroac/comfy-diffusion/master/install.sh | sh
```

Then run:
```sh
parallax install
```

**Windows (PowerShell, one command):**
```powershell
irm https://raw.githubusercontent.com/quinteroac/comfy-diffusion/master/install.ps1 | iex
```

Then open a new terminal and run:
```powershell
parallax install
```

### Full setup flow

After the binary is installed, run these installer subcommands in order to bootstrap the complete stack:

| Command | What it does |
|---------|-------------|
| `parallax install` | Creates `~/.parallax/env`, installs `comfy-diffusion` + torch via `uv`, and bootstraps the ComfyUI runtime. Pass `--cpu` for CPU-only environments. |
| `parallax mcp install` | Registers the Parallax MCP server in Claude Desktop (`~/Library/Application Support/Claude/…` on macOS, `%APPDATA%\Claude\…` on Windows, `~/.config/claude/…` on Linux). Restart Claude Desktop to apply. |
| `parallax ms install` | Registers the inference server as a systemd user service (Linux) or launchd agent (macOS) so it starts automatically on boot. Prints the service URL on success. |
| `parallax frontend install` | Downloads and installs the pre-built web UI to `~/.parallax/frontend/`. |

All commands are **idempotent** — re-running them detects existing state and skips or updates gracefully. Add `--verbose` to any command to stream subprocess output in real time.

The install base directory defaults to `~/.parallax/`. Override with `PARALLAX_HOME=/custom/path parallax install`.

### Pre-built binaries

Standalone binaries for all supported platforms are published as release assets on every [GitHub Release](https://github.com/quinteroac/comfy-diffusion/releases):

| Platform | Asset |
|----------|-------|
| Linux x86_64 | `parallax-linux-x86_64` |
| macOS (Apple Silicon + Intel) | `parallax-macos-universal` |
| Windows x86_64 | `parallax-windows-x86_64.exe` |

Each asset ships with a `.sha256` checksum file. The `install.sh` / `install.ps1` scripts above download and verify checksums automatically.

---

## Quick Start

Call `check_runtime()` before any other `comfy_diffusion` API (model loading, prompt encoding, sampling, etc.).

On first use, `check_runtime()` bootstraps the runtime and triggers an automatic download of the pinned ComfyUI release if `vendor/ComfyUI` is not available yet.

```python
from comfy_diffusion import check_runtime

runtime = check_runtime()
if "error" in runtime:
    # check_runtime() returns an error dict instead of raising
    print(f"Runtime bootstrap failed: {runtime['error']}")
    raise SystemExit(1)

from comfy_diffusion.models import ModelManager

manager = ModelManager(models_dir="/path/to/models")
checkpoint = manager.load_checkpoint("model.safetensors")
```

After this succeeds, continue with the rest of your inference flow.

---

## Why I built this

I've been building creative AI applications — tools that generate music, visuals, and video for streaming platforms. For a while I used `diffusers` and `DiffSynth-Studio` as my inference backends. They're great libraries, well-documented, easy to import. But I kept hitting the same wall: the best models, the best fine-tunes, the ones that actually produce good results, are all built for ComfyUI.

The LoRAs on Civitai, the checkpoints people spend months training, the workflows the community shares — they're tested on ComfyUI. When I used them through diffusers I'd get inconsistent results, or they just wouldn't work the way they were intended. ComfyUI's sampler implementations, its VRAM management, its model loading logic — these aren't just UI conveniences, they're the reason the outputs look the way they do.

The problem is ComfyUI wasn't built to be a library. It's an application. The only way to use it programmatically is to run it as a server and talk to it over HTTP — which means every project I build needs to depend on a full ComfyUI backend running somewhere. That's a separate process to manage, a separate service to deploy, and a monolith that loads every node and capability whether my app needs them or not.

`comfy-diffusion` is my answer to that. ComfyUI's inference engine — `comfy.model_management`, `comfy.samplers`, `comfy.sd`, all of it — is perfectly importable Python code. It just was never packaged as a library. So I'm packaging it as one.

I built this for myself, to use in my own projects. But I'm building it in the open because I suspect I'm not the only one who wants to write `import comfy_diffusion` instead of running a server.

---

## What it is

`comfy-diffusion` imports ComfyUI's internal modules directly — no server, no HTTP, no node system. ComfyUI is vendored as a git submodule and its internals are made transparently importable when you `import comfy_diffusion`.

The API exposes ComfyUI's building blocks as plain Python functions. You compose them directly — the same way you'd wire nodes in ComfyUI, but in code:

```python
from comfy_diffusion.models import ModelManager
from comfy_diffusion.conditioning import encode_prompt
from comfy_diffusion.sampling import sample
from comfy_diffusion import vae_decode, vae_encode, apply_lora

manager = ModelManager(models_dir="/path/to/models")
checkpoint = manager.load_checkpoint("animagine-xl.safetensors")

# Apply a LoRA
model, clip = apply_lora(checkpoint.model, checkpoint.clip, "style.safetensors", 0.8, 0.8)

# Encode prompts
positive = encode_prompt(clip, "a portrait of a woman, studio lighting")
negative = encode_prompt(clip, "blurry, low quality")

# txt2img
import torch
latent = {"samples": torch.zeros(1, 4, 64, 64)}
denoised = sample(
    model, positive, negative, latent,
    steps=20, cfg=7.0, sampler_name="euler",
    scheduler="normal", seed=42,
)
image = vae_decode(checkpoint.vae, denoised)
image.save("output.png")

# img2img
source = Image.open("input.png")
latent = vae_encode(checkpoint.vae, source)
denoised = sample(
    model, positive, negative, latent,
    steps=20, cfg=7.0, sampler_name="euler",
    scheduler="normal", seed=42, denoise=0.75,
)
image = vae_decode(checkpoint.vae, denoised)
image.save("output_img2img.png")
```

The modularity is the point. Every building block is explicit — you see exactly what's happening at each step, and you can swap any piece without fighting a pipeline abstraction.

---

## How ComfyUI is embedded

`comfy-diffusion` ships ComfyUI's source as a **git submodule** vendored at
`vendor/ComfyUI`. This means the full ComfyUI source tree is included in every
PyPI wheel — no separate ComfyUI installation, no running server, no `git clone`.

When you `import comfy_diffusion` the package calls `ensure_comfyui_on_path()`, which inserts
the vendored ComfyUI directory into `sys.path`. After that single call the entire `comfy.*`
namespace is importable directly from your code:

```python
import comfy_diffusion  # bootstraps the path

import comfy.model_management  # ComfyUI internals, directly importable
import comfy.samplers
import comfy.sd
```

The `[comfyui]` extra (e.g. `pip install "comfy-diffusion[cpu,comfyui]"`) installs all of
ComfyUI's Python runtime dependencies — the same packages listed in ComfyUI's own
`requirements.txt`. Without this extra the `comfy.*` modules will be missing their deps and fail
to import.

**Summary of the three moving parts:**

| Part | What it does |
|------|-------------|
| `vendor/ComfyUI` | ComfyUI source, vendored as a git submodule and shipped in the wheel |
| `comfy_diffusion/_runtime.py` | Inserts the vendor path into `sys.path` on first import |
| `[comfyui]` extra | Installs ComfyUI's Python runtime dependencies from PyPI |

---

## What it is not

- Not a ComfyUI wrapper that talks to a running server
- Not a node system or workflow runner
- Not a replacement for ComfyUI — it depends on it
- Not a general-purpose diffusion library — it's opinionated toward ComfyUI's engine
- Optional pipeline — You can use the existing pipelines or compose the blocks yourself.

---

## Installation

Requires Python 3.12+. ComfyUI is vendored inside the package — no separate ComfyUI installation needed.

### Extras (what to install)

`comfy-diffusion` keeps heavy deps optional via extras:

| Extra | Includes | When to use |
|------:|----------|-------------|
| `[cpu]` | `torch` (CPU build when installed via `uv`) | CPU-only inference and CI |
| `[cuda]` | `torch` (CUDA build when installed via `uv`) | NVIDIA GPU inference |
| `[comfyui]` | ComfyUI runtime deps (from ComfyUI `requirements.txt`) | Required for importing `comfy.*` internals |
| `[video]` | `opencv-python`, `imageio` | Video I/O helpers |
| `[audio]` | `torchaudio` | Audio pipelines |
| `[all]` | union of `[cuda]`, `[video]`, `[audio]` | Convenience bundle |

In most cases you want **one of**:

- CPU: `comfy-diffusion[cpu,comfyui]`
- CUDA: `comfy-diffusion[cuda,comfyui]`

### From PyPI

```bash
# CPU
pip install "comfy-diffusion[cpu,comfyui]"

# CUDA (RTX 30xx / 40xx — cu124)
pip install "comfy-diffusion[cuda,comfyui]"

# CUDA (RTX 50xx / Blackwell — cu128)
pip install "comfy-diffusion[cuda,comfyui]" --extra-index-url https://download.pytorch.org/whl/cu128
```

### In your own project (uv)

When using `uv` in your own project you need to declare the PyTorch package index in your
`pyproject.toml` — uv does not inherit `[tool.uv.sources]` from dependencies.

Add the following to your `pyproject.toml` before running `uv add`:

**CPU:**

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]
torchaudio = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

**CUDA cu124 (RTX 30xx / 40xx):**

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cuda" }]
torchvision = [{ index = "pytorch-cuda" }]
torchaudio = [{ index = "pytorch-cuda" }]

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

For CUDA cu128 (RTX 50xx / Blackwell) use `https://download.pytorch.org/whl/cu128` instead.

Then add the dependency and verify:

```bash
# CPU
uv add "comfy-diffusion[cpu,comfyui]"

# CUDA
uv add "comfy-diffusion[cuda,comfyui]"

# Verify
uv run python -c "import comfy_diffusion; print(comfy_diffusion.check_runtime())"
```

### From source

```bash
# 1. Clone with submodule
git clone --recurse-submodules https://github.com/quinteroac/comfy-diffusion.git
cd comfy-diffusion

# 2. Install (CPU torch — works on all machines including CI)
uv sync --extra cpu --extra comfyui

# 3. GPU only: replace torch with CUDA build (run after every uv sync)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
# RTX 50xx (Blackwell): use cu128
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
# Verify: uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **Note:** `uv.lock` pins the CPU variant of torch so that CI (no GPU) can run `uv sync` reproducibly. GPU users replace torch after syncing with the command above.

---

## parallax CLI reference

The `parallax` binary exposes the full stack as a unified CLI. All subcommands support `--help`.

### `parallax install`

Bootstraps the runtime to `~/.parallax/env`.

```bash
parallax install [--cpu] [--upgrade] [--verbose]
```

| Flag | Description |
|------|-------------|
| `--cpu` | Install CPU-only torch (no CUDA) |
| `--upgrade` | Upgrade an existing installation |
| `--verbose` | Stream full subprocess output |

---

### `parallax comfyui`

Manages the ComfyUI web UI as a background process. Use this when you want the full ComfyUI node-graph interface alongside the CLI.

```bash
parallax comfyui start   [--port N] [--timeout N] [--open] [--models-dir PATH]
parallax comfyui stop
parallax comfyui status
```

| Subcommand | Description |
|-----------|-------------|
| `start` | Launch ComfyUI in the background, write PID to `~/.config/parallax/comfyui.pid` |
| `stop` | Terminate the running ComfyUI process and remove the PID file |
| `status` | Print whether ComfyUI is running, the PID, and the port |

**`start` options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8188` | Port for ComfyUI to listen on |
| `--timeout` | `30` | Seconds to wait for the server to become ready |
| `--open` | `false` | Auto-open the URL in the default browser once ready |
| `--models-dir` | `$PYCOMFY_MODELS_DIR` | Custom models directory |

```bash
# Start on default port
parallax comfyui start

# Start on a custom port and open the browser automatically
parallax comfyui start --port 8189 --open

# Check if it's running
parallax comfyui status
# ComfyUI is running (PID 12345, port 8189)

# Stop it
parallax comfyui stop
# ComfyUI (PID 12345) stopped.
```

Logs are written to `~/.config/parallax/comfyui.log`.

---

### `parallax create`

Generates images, videos, and audio.

```bash
parallax create image  --model MODEL --prompt TEXT [options]
parallax create video  --model MODEL --prompt TEXT [options]
parallax create audio  --model MODEL --prompt TEXT [options]
```

**Image models:** `sdxl`, `anima`, `z_image`, `flux_klein`, `qwen`

**Video models:** `ltx2`, `ltx23`, `wan21`, `wan22`

**Audio models:** `ace_step`

**Common options** (all `create` subcommands):

| Flag | Description |
|------|-------------|
| `--prompt` | Text description (required) |
| `--negative-prompt` | Negative guidance |
| `--steps` | Sampling steps |
| `--cfg` | CFG guidance scale |
| `--seed` | Random seed (default: 0) |
| `--output` | Output file path |
| `--models-dir` | Custom models directory |
| `--async` | Queue job and return immediately with a job ID |

**Video-specific options:**

| Flag | Description |
|------|-------------|
| `--input` | Input image for image-to-video |
| `--audio` | Audio file for audio-conditioned generation (`ltx23` only) |
| `--width`, `--height`, `--length`, `--fps` | Dimensions and duration |

```bash
# Text-to-image
parallax create image --model sdxl --prompt "a portrait of a woman, studio lighting" --output out.png

# Text-to-video
parallax create video --model ltx2 --prompt "ocean waves at sunset" --output clip.mp4

# Image-to-video with audio conditioning
parallax create video --model ltx23 --prompt "dancer on stage" --input frame.png --audio music.wav

# Text-to-audio (queued, non-blocking)
parallax create audio --model ace_step --prompt "lo-fi hip hop beat" --async
# Job ID: 550e8400-e29b-41d4-a716-446655440000
```

---

### `parallax edit`

Edits existing images with an instruction prompt.

```bash
parallax edit image --model MODEL --prompt TEXT --input PATH [options]
```

**Models:** `flux_9b_kv`, `qwen`

| Flag | Description |
|------|-------------|
| `--input` | Input image path (required) |
| `--subject-image` | Reference subject image (`flux_9b_kv` only) |
| `--image2`, `--image3` | Additional reference images (`qwen` only) |
| `--no-lora` | Disable Lightning LoRA (`qwen` only) |

---

### `parallax upscale`

Super-resolution upscaling.

```bash
parallax upscale image --model MODEL --input PATH [options]
```

**Models:** `esrgan`, `latent_upscale`

---

### `parallax jobs`

Monitors and manages async inference jobs.

```bash
parallax jobs list   [--limit N]
parallax jobs status <job_id>
parallax jobs watch  <job_id>
parallax jobs cancel <job_id>
parallax jobs open   <job_id>
```

| Subcommand | Description |
|-----------|-------------|
| `list` | List the N most recent jobs (default: 20) |
| `status` | Get status of a specific job |
| `watch` | Stream job progress in real-time via SSE |
| `cancel` | Cancel a pending or running job |
| `open` | Open the job result in the default browser |

Job states: `pending` → `running` → `completed` / `failed` / `cancelled`

---

### `parallax async`

Queues any subcommand as a non-blocking job.

```bash
parallax async create image --model sdxl --prompt "portrait"
# Job ID: 550e8400-e29b-41d4-a716-446655440000

parallax jobs watch 550e8400-e29b-41d4-a716-446655440000
```

---

### `parallax mcp install`

Registers the Parallax MCP server in Claude Desktop.

```bash
parallax mcp install [--verbose]
```

Config is written to the platform-appropriate path:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/claude/claude_desktop_config.json`

Restart Claude Desktop to apply. Idempotent — safe to re-run.

---

### `parallax ms install`

Registers the inference server as a persistent system service.

```bash
parallax ms install [--verbose]
```

- **Linux:** systemd user unit at `~/.config/systemd/user/parallax-ms.service`
- **macOS:** launchd agent at `~/Library/LaunchAgents/run.parallax.ms.plist`

The service runs the FastAPI server at `http://localhost:5000`. Idempotent.

---

### `parallax frontend install`

Downloads and installs the pre-built web UI.

```bash
parallax frontend install [--version X.Y.Z] [--verbose]
parallax frontend version
```

Installs to `~/.parallax/frontend/` and mounts at `/ui` on the FastAPI server.

---

## Type checking (stubs)

`comfy-diffusion` is a typed package (PEP 561) and ships `py.typed`, so editors and type checkers
will automatically pick up inline type annotations after installation.

### Mypy

If your project type-checks strictly, you will typically want one (or both) of:

- Install `comfy-diffusion[comfyui]` so ComfyUI runtime imports resolve.
- Keep `ignore_missing_imports = true` for ComfyUI internals and GPU-only packages.

Example `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
```

### Pyright / Pylance

No special config is required. If you see missing-import diagnostics for `comfy.*`, install
`comfy-diffusion[comfyui]` (or relax missing-import reporting in your editor).

---

## Developer experience (DX)

This repo is optimized for `uv`:

```bash
# Install dev deps + selected extras (example: CPU + comfyui)
uv sync --extra cpu --extra comfyui

# Run tests
uv run python -m pytest

# Lint and format checks
uv run ruff check .

# Type check
uv run mypy comfy_diffusion
```

---

## Bundled skills

The package includes distributable agent skill documents under `comfy_diffusion/skills/` and
ships them as package data (so they are available after `pip` / `uv` install).

Discover them at runtime:

```python
from comfy_diffusion.skills import get_skills_path

skills_root = get_skills_path()
print([p.name for p in skills_root.iterdir()])  # e.g. ["README.md", "SKILL.md", "base-runtime-check.md", ...]
```

These are intentionally separate from any repo-local agent workflow assets under `.agents/`.

### Install skills via npx

```bash
npx skills add https://github.com/quinteroac/comfy-diffusion/tree/master/comfy_diffusion/skills
```

### Agent discovery (copy into AGENTS.md)

If you are running an AI agent workflow that uses `AGENTS.md` as an entry point, add a note like
this so the agent knows how to discover the bundled, install-time skills:

```md
## Bundled skills (discoverable at runtime)

`comfy_diffusion` ships distributable skill documents inside the installed package under
`comfy_diffusion/skills/` (this is separate from repo-local `.agents/skills/`).

To discover them at runtime:

```python
from comfy_diffusion.skills import get_skills_path

skills_root = get_skills_path()
```
```

---

## License

GPL-3.0 — same as ComfyUI, which this project depends on.
