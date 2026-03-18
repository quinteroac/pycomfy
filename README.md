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
- Not an opinionated pipeline — there is no `ImagePipeline`. You compose the blocks yourself.

---

## Status

Early development. Built iteratively, one capability block at a time. The full node inventory and iteration plan is in [`ROADMAP.md`](ROADMAP.md).

| # | Module | Goal | Status |
|---|--------|------|--------|
| 01 | `_runtime` / `check_runtime()` | Package foundation + ComfyUI vendoring | ✅ Done |
| 02 | `models` | Checkpoint loading (`ModelManager`, `CheckpointResult`) | ✅ Done |
| 03 | `conditioning` | Prompt encoding via `encode_prompt` | ✅ Done |
| 04 | `sampling` | KSampler wrapper via `sample()` | ✅ Done |
| 05 | `vae` | VAE decode latent→PIL via `vae_decode()` | ✅ Done |
| 06 | `lora` | LoRA loading and stacking via `apply_lora()` | ✅ Done |
| 07 | `vae` + `models` | VAE encode + standalone loaders (`load_vae`, `load_clip`, `load_unet`) | ✅ Done |
| 08 | `vae` — tiled | Tiled VAE encode/decode for large images without OOM | ✅ Done |
| 09 | `vae` — batch/video | Batch VAE encode/decode for video frame sequences | ✅ Done |
| 10 | `sampling` — advanced | `SamplerCustomAdvanced`, schedulers, sigma manipulation | ✅ Done |
| 11 | `audio` | Stable Audio, WAN sound-to-video, LTXV audio, ACE Step | ✅ Done |
| — | **`v0.1.1-preview`** | **Preview release milestone** | ✅ Done |
| 12–19 | conditioning, controlnet, latent, image, mask, model patches, packaging, skills | ✅ Done | ✅ Done |

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
