---
name: comfy-diffusion-reference
description: Project conventions and complete public API reference for comfy_diffusion — covers coding patterns, data formats, and every module signature.
user-invocable: false
---

# comfy_diffusion — Project Conventions and Public API Reference

This document is the canonical skill reference for AI agents working with the
`comfy_diffusion` package. It covers coding conventions, data format contracts,
and complete public signatures for every module.

---

## Runtime Bootstrap (mandatory first step)

**Always call `check_runtime()` before any other `comfy_diffusion` API.** It bootstraps
the ComfyUI runtime (vendored at `vendor/ComfyUI`), validates device availability, and
returns an error dict instead of raising — so you can detect failure explicitly.

```python
from comfy_diffusion import check_runtime

runtime = check_runtime()
if "error" in runtime:
    print(f"Runtime bootstrap failed: {runtime['error']}")
    raise SystemExit(1)

# Safe to call model loading, sampling, VAE, etc.
```

Skipping this step will cause import errors or undefined behaviour in any module that
depends on `comfy.*` internals being on `sys.path`.

---

## Project Conventions

### 1. Lazy ComfyUI Imports
All ComfyUI-dependent imports (`comfy.*`, `nodes`, `node_helpers`, `comfy_extras.*`,
`comfyui_version`) are **always deferred into function bodies**, never at module top
level. This keeps every module import-safe in CPU-only or no-ComfyUI environments.

```python
# Correct: deferred inside the function
def my_helper(model: Any) -> Any:
    from ._runtime import ensure_comfyui_on_path
    ensure_comfyui_on_path()
    import comfy.sd
    ...

# Wrong: top-level import will crash without ComfyUI on sys.path
import comfy.sd  # DO NOT do this at module level
```

### 2. Path Parameters: `str | Path`
All file-path parameters accept `str | Path`. Internally paths are always resolved
via `pathlib.Path`. Never pass bare strings with `os.path`; use `Path(x)` first.

### 3. Image Tensor Layout: BHWC float32 in [0, 1]
ComfyUI image tensors use `(Batch, Height, Width, Channels)` — not `BCHW`.
Values are `float32` in the range `[0.0, 1.0]`.

```python
# shape: (1, 512, 512, 3)  dtype: torch.float32  range: [0, 1]
image_tensor = load_image("photo.png")[0]
```

### 4. LATENT Dict Format
Latents are plain Python `dict[str, Any]` with a mandatory `"samples"` tensor key.
Optional metadata keys include `"noise_mask"`, `"type"` (e.g. `"audio"`).

```python
latent = {"samples": torch.zeros([1, 4, 64, 64])}                      # image latent
latent_with_mask = {"samples": ..., "noise_mask": mask_tensor}          # inpaint
audio_latent = {"samples": ..., "sample_rate": 44100, "type": "audio"}  # audio
```

### 5. Discovering Bundled Skills
Use `get_skills_path()` (or `importlib.resources.files` directly) to locate skill
files distributed with the installed package:

```python
from comfy_diffusion.skills import get_skills_path

skills_root = get_skills_path()
skill_text = (skills_root / "comfy-diffusion-reference" / "SKILL.md").read_text(encoding="utf-8")
```

---

## Module Reference

### `comfy_diffusion` (top-level re-exports)

```python
from comfy_diffusion import (
    check_runtime,
    vae_decode, vae_decode_tiled, vae_decode_batch, vae_decode_batch_tiled,
    vae_encode, vae_encode_tiled, vae_encode_batch, vae_encode_batch_tiled,
    vae_encode_for_inpaint,
    apply_lora,
)
```

---

### `comfy_diffusion.runtime`

```python
def check_runtime() -> dict[str, Any]:
    """Return structured runtime diagnostics for the current Python process.

    Keys always present:
      - "python_version": str  (e.g. "3.12.3")
      - "comfyui_version": str | None
      - "device": str | None   (e.g. "cpu", "cuda:0")
      - "vram_total_mb": int | None
      - "vram_free_mb": int | None
    On error, also contains:
      - "error": str  (human-readable message)
    """
```

---

### `comfy_diffusion.models`

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class CheckpointResult:
    model: Any
    clip: Any | None
    vae: Any | None

class ModelManager:
    def __init__(self, models_dir: str | Path) -> None:
        """Validate and register models_dir with ComfyUI folder_paths.

        Registers subdirectories: checkpoints, embeddings, diffusion_models/unet,
        text_encoders/clip, vae, llm.
        """

    def load_checkpoint(self, filename: str) -> CheckpointResult:
        """Load a .safetensors/.ckpt checkpoint from models_dir/checkpoints/."""

    def load_vae(self, path: str | Path) -> Any:
        """Load a standalone VAE (absolute path or filename under models_dir/vae/)."""

    def load_clip(self, *paths: str | Path, clip_type: str = "stable_diffusion") -> Any:
        """Load a standalone text encoder from one or more paths/filenames.

        Accepts variadic paths for multi-file encoders (e.g. dual text encoders).
        clip_type: "stable_diffusion" | "wan" | "sd3" | "flux" | "ltxv" (any CLIPType name).
        Paths are resolved under models_dir/text_encoders/ or models_dir/clip/.
        """

    def load_clip_vision(self, path: str | Path) -> Any:
        """Load a CLIP vision model from models_dir/clip_vision/ or absolute path."""

    def load_unet(self, path: str | Path) -> Any:
        """Load a standalone diffusion model (UNet/transformer).

        Resolved under models_dir/unet/ or models_dir/diffusion_models/.
        """

    def load_ltxv_audio_vae(self, path: str | Path) -> object:
        """Load an LTXV audio VAE checkpoint from models_dir/checkpoints/ or absolute path."""

    def load_ltxav_text_encoder(
        self,
        text_encoder_path: str | Path,
        checkpoint_path: str | Path,
    ) -> object:
        """Load an LTXAV text encoder from two files (text encoder + companion checkpoint)."""

    def load_llm(self, path: str | Path) -> Any:
        """Load a standalone LLM/VLM text model.

        Search order for relative paths: models_dir/llm → models_dir/text_encoders → models_dir/clip.
        """

# Module-level model patching helpers

def model_sampling_flux(
    model: Any,
    max_shift: float,
    min_shift: float,
    width: int,
    height: int,
) -> Any:
    """Patch a model clone with Flux sampling shift settings.

    The shift is interpolated between min_shift and max_shift based on latent token count
    derived from width and height.
    """

def model_sampling_sd3(model: Any, shift: float) -> Any:
    """Patch a model clone with SD3 discrete-flow sampling shift settings."""

def model_sampling_aura_flow(model: Any, shift: float) -> Any:
    """Patch a model clone with AuraFlow continuous V-prediction shift settings."""
```

---

### `comfy_diffusion.vae`

All encode functions return `dict[str, Any]` (a LATENT dict with `"samples"`).
All decode functions return PIL `Image` or `list[Image]`.

```python
def vae_decode(vae: Any, latent: Mapping[str, Any]) -> Image.Image: ...
def vae_decode_tiled(
    vae: Any, latent: Mapping[str, Any],
    tile_size: int = 512, overlap: int = 64,
) -> Image.Image: ...
def vae_decode_batch(vae: Any, latent: Mapping[str, Any]) -> list[Image.Image]: ...
def vae_decode_batch_tiled(
    vae: Any, latent: Mapping[str, Any],
    tile_size: int = 512, overlap: int = 64,
) -> list[Image.Image]: ...

def vae_encode(vae: Any, image: Image.Image) -> dict[str, Any]: ...
def vae_encode_tiled(
    vae: Any, image: Image.Image,
    tile_size: int = 512, overlap: int = 64,
) -> dict[str, Any]: ...
def vae_encode_batch(vae: Any, images: list[Image.Image]) -> dict[str, Any]: ...
def vae_encode_batch_tiled(
    vae: Any, images: list[Image.Image],
    tile_size: int = 512, overlap: int = 64,
) -> dict[str, Any]: ...
def vae_encode_for_inpaint(
    vae: Any, image: Image.Image, mask: Any, grow_mask_by: int = 6,
) -> dict[str, Any]:
    """Returns LATENT with "samples" and "noise_mask". Requires torch."""
```

---

### `comfy_diffusion.lora`

```python
def apply_lora(
    model: Any,
    clip: Any,
    path: str | Path,
    strength_model: float,
    strength_clip: float,
) -> tuple[Any, Any]:
    """Apply a LoRA file to a model/CLIP pair; returns (patched_model, patched_clip).
    Chain multiple calls to stack LoRAs."""
```

---

### `comfy_diffusion.sampling`

```python
def sample(
    model: Any, positive: Any, negative: Any,
    latent: Any, steps: Any, cfg: Any,
    sampler_name: str, scheduler: str, seed: int,
    *, denoise: float = 1.0,
) -> Any:
    """Run KSampler; returns denoised LATENT dict."""

def sample_advanced(
    model: Any, positive: Any, negative: Any,
    latent: Any, steps: Any, cfg: Any,
    sampler_name: str, scheduler: str, noise_seed: int,
    *, add_noise: bool = True, return_with_leftover_noise: bool = False,
    denoise: float = 1.0, start_at_step: int = 0, end_at_step: int = 10000,
) -> Any:
    """Run KSamplerAdvanced; returns denoised LATENT dict."""

def sample_custom(
    noise: Any, guider: Any, sampler: Any, sigmas: Any, latent_image: Any,
) -> tuple[Any, Any]:
    """Run SamplerCustomAdvanced; returns (output_latent, denoised_output).
    The model is embedded in the guider — use basic_guider() or cfg_guider()."""

# Guider builders (for sample_custom)
def basic_guider(model: Any, conditioning: Any) -> Any: ...
def cfg_guider(model: Any, positive: Any, negative: Any, cfg: Any) -> Any: ...

# Video CFG guidance model patches
def video_linear_cfg_guidance(model: Any, min_cfg: float) -> Any:
    """Patch a model clone with a frame-wise linear CFG ramp from cfg_scale down to min_cfg."""

def video_triangle_cfg_guidance(model: Any, min_cfg: float) -> Any:
    """Patch a model clone with a frame-wise triangle CFG ramp oscillating between min_cfg and cfg_scale."""

# Noise builders
def random_noise(noise_seed: int) -> Any: ...
def disable_noise() -> Any: ...

# Scheduler builders (return SIGMAS tensor)
def basic_scheduler(model: Any, scheduler_name: str, steps: int, denoise: float = 1.0) -> Any: ...
def karras_scheduler(steps: int, sigma_max: float, sigma_min: float, rho: float = 7.0) -> Any: ...
def ays_scheduler(model_type: str, steps: int, denoise: float = 1.0) -> Any:
    """model_type: "SD1" | "SDXL" | "SVD"."""
def flux2_scheduler(steps: int, width: int, height: int) -> Any: ...
def ltxv_scheduler(
    steps: int, max_shift: float, base_shift: float,
    *, stretch: bool = True, terminal: float = 0.1, latent: Any = None,
) -> Any: ...

def split_sigmas(sigmas: Any, step: int) -> tuple[Any, Any]: ...
def split_sigmas_denoise(sigmas: Any, denoise: float) -> tuple[Any, Any]: ...
def get_sampler(sampler_name: str) -> Any: ...
```

---

### `comfy_diffusion.conditioning`

```python
def encode_prompt(clip: Any, text: str) -> Any:
    """Encode a single prompt string; empty string is normalized to a space."""

def encode_prompt_flux(
    clip: Any, text: str, clip_l_text: str, guidance: float = 3.5,
) -> Any:
    """Flux dual-encoder encoding: t5xxl text + clip_l text + guidance."""

def encode_clip_vision(
    clip_vision: Any, image: Any,
    crop: Literal["center", "none"] = "center",
) -> Any: ...

def wan_image_to_video(
    positive: Any, negative: Any, vae: Any,
    width: int = 832, height: int = 480, length: int = 81, batch_size: int = 1,
    start_image: Any | None = None, clip_vision_output: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Returns (positive, negative, latent) for WAN i2v."""

def wan_first_last_frame_to_video(
    positive: Any, negative: Any, vae: Any,
    width: int = 832, height: int = 480, length: int = 81, batch_size: int = 1,
    start_image: Any | None = None, end_image: Any | None = None,
    clip_vision_start_image: Any | None = None,
    clip_vision_end_image: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]: ...

def ltxv_img_to_video(
    positive: Any, negative: Any, image: Any, vae: Any,
    width: int = 768, height: int = 512, length: int = 97,
    batch_size: int = 1, strength: float = 1.0,
) -> tuple[Any, Any, dict[str, Any]]:
    """Returns (positive, negative, latent_with_noise_mask) for LTXV i2v."""

def ltxv_conditioning(
    positive: Any, negative: Any, frame_rate: float = 25.0,
) -> tuple[Any, Any]:
    """Inject frame_rate metadata into LTXV conditioning."""

def conditioning_combine(
    cond_a: Any, cond_b: Any | None = None, *additional: Any,
) -> list[Any]:
    """Merge two or more conditioning objects; also accepts a single list."""

def conditioning_set_mask(
    conditioning: Any, mask: Any,
    strength: float = 1.0,
    set_cond_area: Literal["default", "mask bounds"] = "default",
) -> list[Any]: ...

def conditioning_set_timestep_range(
    conditioning: Any, start: float, end: float,
) -> list[Any]:
    """Attach timestep bounds (0.0–1.0) to conditioning."""

def flux_guidance(conditioning: Any, guidance: float = 3.5) -> list[Any]: ...
```

---

### `comfy_diffusion.image`

```python
def load_image(path: str | Path) -> tuple[Any, Any]:
    """Returns (image_tensor BHWC float32, mask_tensor HW float32).
    Mask is all-zeros (opaque) unless the source has an alpha channel."""

def image_to_tensor(image: PIL.Image.Image) -> Any:
    """Convert PIL Image to BHWC float32 tensor, shape (1, H, W, 3)."""

def image_pad_for_outpaint(
    image: Any, left: int, top: int, right: int, bottom: int, feathering: int,
) -> tuple[Any, Any]:
    """Returns (padded_image BHWC, outpaint_mask BHW). All args non-negative."""

def image_upscale_with_model(upscale_model: Any, image: Any) -> Any: ...
def image_from_batch(image: Any, batch_index: int, length: int = 1) -> Any: ...
def repeat_image_batch(image: Any, amount: int) -> Any: ...
def image_composite_masked(
    destination: Any, source: Any, mask: Any, x: int, y: int,
) -> Any: ...
def ltxv_preprocess(image: Any, width: int, height: int) -> Any:
    """Center-resize + LTXV node compression (img_compression=35)."""
```

---

### `comfy_diffusion.latent`

```python
def empty_latent_image(width: int, height: int, batch_size: int = 1) -> dict[str, Any]: ...
def latent_upscale(latent: dict[str, Any], method: str, width: int, height: int) -> dict[str, Any]:
    """method: "nearest-exact" | "bilinear" | "area" | "bicubic" | "bislerp"."""
def latent_upscale_by(latent: dict[str, Any], method: str, scale_by: float) -> dict[str, Any]: ...
def latent_crop(latent: dict[str, Any], x: int, y: int, width: int, height: int) -> dict[str, Any]: ...
def latent_from_batch(latent: dict[str, Any], batch_index: int, length: int = 1) -> dict[str, Any]: ...
def latent_cut_to_batch(latent: dict[str, Any], start: int, length: int) -> dict[str, Any]:
    """Alias for latent_from_batch with positional start."""
def repeat_latent_batch(latent: dict[str, Any], amount: int) -> dict[str, Any]: ...
def latent_concat(*latents: dict[str, Any], dim: str = "t") -> dict[str, Any]:
    """dim: "x" | "-x" | "y" | "-y" | "t" | "-t". Requires ≥ 2 latents."""
def replace_video_latent_frames(
    latent: dict[str, Any], replacement: dict[str, Any], start_frame: int,
) -> dict[str, Any]: ...
def latent_composite(
    destination: dict[str, Any], source: dict[str, Any], x: int, y: int,
) -> dict[str, Any]: ...
def latent_composite_masked(
    destination: dict[str, Any], source: dict[str, Any], mask: Any,
    x: int = 0, y: int = 0,
) -> dict[str, Any]: ...
def set_latent_noise_mask(latent: dict[str, Any], mask: Any) -> dict[str, Any]:
    """mask must be a torch.Tensor."""
def inpaint_model_conditioning(
    model: Any, latent: dict[str, Any], vae: Any, positive: Any, negative: Any,
) -> tuple[Any, Any, Any]:
    """Returns (patched_model, patched_positive, patched_negative)."""
```

---

### `comfy_diffusion.mask`

```python
def load_image_mask(path: str | Path, channel: str) -> Any:
    """channel: "alpha" | "red" | "green" | "blue". Returns MASK (1, H, W) float32."""
def image_to_mask(image: Any, channel: str) -> Any:
    """channel: "red" | "green" | "blue". image must be BHWC. Returns BHW float32."""
def mask_to_image(mask: Any) -> Any:
    """mask must be BHW. Returns BHWC float32 (grayscale broadcast to 3 channels)."""
def grow_mask(mask: Any, expand: int, tapered_corners: bool) -> Any:
    """expand > 0 grows, expand < 0 shrinks."""
def feather_mask(mask: Any, left: int, top: int, right: int, bottom: int) -> Any:
    """mask must be BHW. All side values non-negative pixels."""
```

---

### `comfy_diffusion.controlnet`

```python
def load_controlnet(path: str | Path) -> Any: ...
def load_diff_controlnet(model: Any, path: str | Path) -> Any:
    """Diff ControlNet requires the paired base model."""
def apply_controlnet(
    positive: Any, negative: Any, control_net: Any, image: Any,
    strength: float = 1.0, start_percent: float = 0.0, end_percent: float = 1.0,
    vae: Any = None,
) -> tuple[Any, Any]:
    """image is a BHWC tensor hint map. Returns (positive, negative)."""
def set_union_controlnet_type(control_net: Any, type: str) -> Any:
    """type: "auto" | any key in comfy.cldm.control_types.UNION_CONTROLNET_TYPES."""
```

---

### `comfy_diffusion.video`

> Requires the `[video]` extra: `pip install comfy-diffusion[video]`

```python
def load_video(path: str | Path) -> Any:
    """Returns BHWC float32 tensor (torch) or list[PIL.Image] when torch absent."""
def save_video(frames: Any, path: str | Path, fps: float) -> None:
    """frames: BHWC tensor, numpy BHWC/HWC, or iterable of frames."""
def get_video_components(video_path: str | Path) -> dict[str, int | float]:
    """Returns {"frame_count": int, "fps": float, "width": int, "height": int}."""
```

---

### `comfy_diffusion.audio`

> LTXV audio requires the `[audio]` extra.

```python
def ltxv_audio_vae_encode(vae: Any, audio: Any) -> dict[str, Any]:
    """Returns {"samples": ..., "sample_rate": int, "type": "audio"}."""
def ltxv_audio_vae_decode(vae: Any, latent: Any) -> dict[str, Any]:
    """Returns {"waveform": ..., "sample_rate": int}."""
def ltxv_empty_latent_audio(
    audio_vae: Any, frames_number: int, frame_rate: int = 25, batch_size: int = 1,
) -> dict[str, Any]: ...
def encode_ace_step_15_audio(
    clip: Any,
    tags: str,
    lyrics: str = "",
    seed: int = 0,
    bpm: int = 120,
    duration: float = 120.0,
    timesignature: str = "4",
    language: str = "en",
    keyscale: str = "C major",
    generate_audio_codes: bool = True,
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
) -> Any:
    """ACE Step 1.5 text/audio metadata conditioning."""
def empty_ace_step_15_latent_audio(seconds: float, batch_size: int = 1) -> dict[str, Any]:
    """Returns {"samples": zeros tensor, "type": "audio"}."""
```

---

### `comfy_diffusion.textgen`

```python
def generate_text(
    clip: Any,
    prompt: str,
    *,
    image: Any | None = None,
    max_length: int = 256,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_k: int = 64,
    top_p: float = 0.95,
    min_p: float = 0.05,
    repetition_penalty: float = 1.05,
    seed: int = 0,
) -> str:
    """Generate text with a ComfyUI-compatible text encoder (LLM/VLM clip object).

    Pass image (BHWC tensor) for vision-language models.
    Returns the decoded string output.
    """

def generate_ltx2_prompt(
    clip: Any,
    prompt: str,
    *,
    image: Any | None = None,
    max_length: int = 256,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_k: int = 64,
    top_p: float = 0.95,
    min_p: float = 0.05,
    repetition_penalty: float = 1.05,
    seed: int = 0,
) -> str:
    """Expand a short user prompt into a rich LTX-Video 2 generation prompt.

    Automatically selects the T2V or I2V system prompt based on whether image is provided.
    image: BHWC tensor of the first frame (I2V mode), or None (T2V mode).
    """
```
