# GPU Smoke Test

Manual GPU verification checklist — run **locally** before merging any
pipeline change.  CI is CPU-only; these invocations require a CUDA device
and downloaded model weights.

---

## Prerequisites

```bash
# Install with downloader extras
uv pip install -e ".[downloader,cuda]"

# Set your models directory
export MODELS_DIR="$HOME/models"
```

---

## ltx2_t2v — LTX-Video 2 Text-to-Video

```python
from comfy_diffusion import check_runtime
from comfy_diffusion.downloader import download_models
from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest, run

check_runtime()
download_models(manifest(), models_dir="/path/to/models")

frames = run(
    models_dir="/path/to/models",
    prompt="a golden retriever running through a sunlit park",
    width=768,
    height=512,
    length=97,
    steps=30,
    seed=42,
)
frames[0].save("ltx2_t2v_smoke.png")
print(f"OK — {len(frames)} frames")
```

**Expected:** no exception; `ltx2_t2v_smoke.png` contains a recognisable image.

---

## ltx2_t2v_distilled — LTX-Video 2 Distilled Text-to-Video

```python
from comfy_diffusion import check_runtime
from comfy_diffusion.downloader import download_models
from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest, run

check_runtime()
download_models(manifest(), models_dir="/path/to/models")

frames = run(
    models_dir="/path/to/models",
    prompt="a golden retriever running through a sunlit park",
    width=768,
    height=512,
    length=97,
    steps=8,
    seed=42,
)
frames[0].save("ltx2_t2v_distilled_smoke.png")
print(f"OK — {len(frames)} frames")
```

**Expected:** no exception; result is visually coherent.

---

## ltx2_i2v — LTX-Video 2 Image-to-Video

```python
from comfy_diffusion import check_runtime
from comfy_diffusion.downloader import download_models
from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest, run

check_runtime()
download_models(manifest(), models_dir="/path/to/models")

frames = run(
    models_dir="/path/to/models",
    image="/path/to/input.png",
    prompt="the subject slowly turns their head",
    width=1280,
    height=720,
    length=121,
    steps=8,
    seed=42,
)
frames[0].save("ltx2_i2v_smoke.png")
print(f"OK — {len(frames)} frames")
```

**Expected:** first frame resembles the input image; motion is smooth.

---

## ltx2_i2v_distilled — LTX-Video 2 Distilled Image-to-Video

```python
from comfy_diffusion import check_runtime
from comfy_diffusion.downloader import download_models
from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_distilled import manifest, run

check_runtime()
download_models(manifest(), models_dir="/path/to/models")

frames = run(
    models_dir="/path/to/models",
    image="/path/to/input.png",
    prompt="the subject slowly turns their head",
    width=1280,
    height=720,
    length=121,
    steps=8,
    seed=42,
)
frames[0].save("ltx2_i2v_distilled_smoke.png")
print(f"OK — {len(frames)} frames")
```

**Expected:** no exception; first frame resembles input image.

---

## ltx2_i2v_lora — LTX-Video 2 Image-to-Video with Style LoRA

```python
from comfy_diffusion import check_runtime
from comfy_diffusion.downloader import download_models
from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest, run

check_runtime()
download_models(manifest(), models_dir="/path/to/models")

frames = run(
    models_dir="/path/to/models",
    image="/path/to/input.png",
    prompt="the subject smiles and waves",
    lora_path="/path/to/style.safetensors",
    width=1280,
    height=1280,
    length=121,
    steps=8,
    seed=42,
)
frames[0].save("ltx2_i2v_lora_smoke.png")
print(f"OK — {len(frames)} frames")
```

**Expected:** style LoRA visually affects output; no exception.

---

## ltx23_t2v — LTX-Video 2.3 (22B) Text-to-Video

```python
from comfy_diffusion import check_runtime
from comfy_diffusion.downloader import download_models
from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest, run

check_runtime()
download_models(manifest(), models_dir="/path/to/models")

frames = run(
    models_dir="/path/to/models",
    prompt="a golden retriever running through a sunlit park",
    width=768,
    height=512,
    length=97,
    steps=8,
    seed=42,
)
frames[0].save("ltx23_t2v_smoke.png")
print(f"OK — {len(frames)} frames")
```

**Expected:** no exception; higher-quality output vs. 2B models.

---

## ltx23_i2v — LTX-Video 2.3 (22B) Image-to-Video

```python
from comfy_diffusion import check_runtime
from comfy_diffusion.downloader import download_models
from comfy_diffusion.pipelines.video.ltx.ltx23.i2v import manifest, run

check_runtime()
download_models(manifest(), models_dir="/path/to/models")

frames = run(
    models_dir="/path/to/models",
    image="/path/to/first_frame.png",
    prompt="the queen turns her head slowly towards the camera",
    width=768,
    height=512,
    length=97,
    fps=24,
    steps=8,
    seed=42,
)
frames[0].save("ltx23_i2v_smoke.png")
print(f"OK — {len(frames)} frames")
```

**Expected:** first frame matches input image; smooth, high-quality motion.
