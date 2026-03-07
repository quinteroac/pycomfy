# pycomfy

A Python library that exposes ComfyUI's inference engine as importable modules — no server, no node graph, no UI.

```python
from pycomfy import check_runtime

print(check_runtime())
# {"comfyui_version": "0.9.x", "device": "cuda:0", "vram_total_mb": 8192, ...}
```

---

## Why I built this

I've been building creative AI applications — tools that generate music, visuals, and video for streaming platforms. For a while I used `diffusers` and `DiffSynth-Studio` as my inference backends. They're great libraries, well-documented, easy to import. But I kept hitting the same wall: the best models, the best fine-tunes, the ones that actually produce good results, are all built for ComfyUI.

The LoRAs on Civitai, the checkpoints people spend months training, the workflows the community shares — they're tested on ComfyUI. When I used them through diffusers I'd get inconsistent results, or they just wouldn't work the way they were intended. ComfyUI's sampler implementations, its VRAM management, its model loading logic — these aren't just UI conveniences, they're the reason the outputs look the way they do.

The problem is ComfyUI wasn't built to be a library. It's an application. The only way to use it programmatically is to run it as a server and talk to it over HTTP — which means every project I build needs to depend on a full ComfyUI backend running somewhere. That's a separate process to manage, a separate service to deploy, and a monolith that loads every node and capability whether my app needs them or not.

`pycomfy` is my answer to that. ComfyUI's inference engine — `comfy.model_management`, `comfy.samplers`, `comfy.sd`, all of it — is perfectly importable Python code. It just was never packaged as a library. So I'm packaging it as one.

I built this for myself, to use in my own projects. But I'm building it in the open because I suspect I'm not the only one who wants to write `import pycomfy` instead of running a server.

---

## What it is

`pycomfy` imports ComfyUI's internal modules directly — no server, no HTTP, no node system. ComfyUI is vendored as a git submodule and its internals are made transparently importable when you `import pycomfy`.

The API exposes ComfyUI's building blocks as plain Python functions. You compose them directly — the same way you'd wire nodes in ComfyUI, but in code:

```python
from pycomfy.models import ModelManager
from pycomfy.conditioning import encode_prompt
from pycomfy.sampling import sample
from pycomfy import vae_decode, vae_encode, apply_lora

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
| 08 | `vae` — tiled | Tiled VAE encode/decode for large images without OOM | ⬜ Next |
| 09 | `vae` — batch/video | Batch VAE encode/decode for video frame sequences | ⬜ |
| 10 | `sampling` — advanced | `SamplerCustomAdvanced`, schedulers, sigma manipulation | ⬜ |
| 11 | `audio` | Stable Audio, WAN sound-to-video, LTXV audio, ACE Step | ⬜ |
| — | **`v0.1.0-preview`** | **Preview release milestone** | ⬜ |
| 12–18 | conditioning, controlnet, latent, image, mask, model patches, packaging | Post-preview | ⬜ |

---

## Installation

The package is **not published on PyPI yet**. Install from the repo (clone + submodule + uv).

ComfyUI deps come from `vendor/ComfyUI/requirements.txt` (extra `comfyui`).

**Note:** `uv.lock` is kept with the CPU variant of torch so CI (no GPU) can run `uv sync` and get reproducible tests. One sync installs CPU torch for everyone; GPU users replace torch with the step below.

```bash
# 1. ComfyUI submodule (required after clone)
git submodule update --init

# 2. Same for everyone (installs CPU torch)
uv sync --extra comfyui

# 3. GPU only: replace torch with CUDA build (required after every uv sync)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
# RTX 50xx (Blackwell): use cu128
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
# Verify: uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> Requires Python 3.12+. ComfyUI is vendored — no separate installation needed. Once the package is on PyPI you can use `pip install pycomfy[cuda]` or `uv add pycomfy[cuda]`.

---

## License

GPL-3.0 — same as ComfyUI, which this project depends on.