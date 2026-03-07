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

The goal is an API that feels like `diffusers` but runs on ComfyUI's engine:

```python
from pycomfy.models import ModelManager
from pycomfy.pipeline import ImagePipeline

manager = ModelManager(models_dir="/path/to/models")
checkpoint = manager.load_checkpoint("animagine-xl.safetensors")

pipeline = ImagePipeline(checkpoint)
image = await pipeline.run(
    prompt="a portrait of a woman, studio lighting",
    negative_prompt="blurry, low quality",
    steps=20,
    cfg=7.0,
    seed=42,
)
image.save("output.png")
```

---

## What it is not

- Not a ComfyUI wrapper that talks to a running server
- Not a node system or workflow runner
- Not a replacement for ComfyUI — it depends on it
- Not a general-purpose diffusion library — it's opinionated toward ComfyUI's engine

---

## Status

Early development. Built iteratively, one capability block at a time.

| Iteration | Module | Status |
|-----------|--------|--------|
| 01 | Package foundation + `check_runtime()` | 🔨 In progress |
| 02 | Model loading (checkpoint, VAE, CLIP) | ⬜ Planned |
| 03 | Conditioning (CLIP encode, prompt weighting) | ⬜ Planned |
| 04 | Sampling (KSampler, schedulers) | ⬜ Planned |
| 05 | VAE encode / decode | ⬜ Planned |
| 06 | LoRA loading and stacking | ⬜ Planned |
| 07 | High-level `ImagePipeline` API | ⬜ Planned |
| 08 | Async / asyncio / progress callbacks | ⬜ Planned |
| 09 | Plugin system (video, audio, vision) | ⬜ Planned |
| 10 | Packaging, type stubs, DX | ⬜ Planned |

---

## Installation

```bash
# CPU
pip install "pycomfy[cpu]"

# CUDA
pip install "pycomfy[cuda]"
```

> Requires Python 3.12+. ComfyUI is vendored — no separate installation needed.

---

## License

GPL-3.0 — same as ComfyUI, which this project depends on.