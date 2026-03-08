#!/usr/bin/env python3
"""
Example using comfy_diffusion with diffusion model, CLIP, and VAE loaded separately.

Runs txt2img or img2img (same as simple_checkpoint_example) but loads the three
components from distinct files via ModelManager.load_unet(), load_clip(),
and load_vae().

- Without --image: txt2img (empty latent, denoise=1.0).
- With --image: img2img (vae_encode input image, then sample with --denoise).

Use this when you have:
  - A standalone diffusion model (UNet) in diffusion_models/ or unet/
  - A standalone text encoder (CLIP) in text_encoders/ or clip/
  - A standalone VAE in vae/

Setup (from repo root):
  1. ComfyUI submodule: git submodule update --init
  2. Python deps: uv sync --extra comfyui
     GPU: uv pip install torch torchvision torchaudio --index-url ... (see AGENTS.md)

Usage:
  export PYCOMFY_MODELS_DIR=/path/to/models
  uv run python examples/separate_components_example.py --unet unet.safetensors --clip clip.safetensors --vae vae.safetensors
  uv run python examples/separate_components_example.py --unet unet.safetensors --clip clip.safetensors --vae vae.safetensors --image input.png --denoise 0.85
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image


def _resolve_path(models_dir: Path, value: str, subdirs: list[str]) -> Path:
    """Resolve path: if value is an existing path use it, else look under models_dir/subdir."""
    p = Path(value)
    if p.is_absolute() and p.is_file():
        return p
    if p.is_file():
        return p.resolve()
    for sub in subdirs:
        candidate = models_dir / sub / value
        if candidate.is_file():
            return candidate
    return Path(value).resolve()


def _empty_latent(width: int, height: int, batch_size: int = 1) -> dict:
    """Build an empty LATENT dict for txt2img (ComfyUI contract)."""
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import torch
    import comfy.model_management

    device = comfy.model_management.intermediate_device()
    latent = torch.zeros(
        [batch_size, 4, height // 8, width // 8],
        device=device,
    )
    return {"samples": latent, "downscale_ratio_spacial": 8}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Txt2img or img2img with separate UNet/CLIP/VAE (use --image for img2img).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root. Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--unet",
        required=True,
        help="Diffusion model (UNet) file: path or filename under diffusion_models/ or unet/.",
    )
    parser.add_argument(
        "--clip",
        required=True,
        help="CLIP text encoder file: path or filename under text_encoders/ or clip/.",
    )
    parser.add_argument(
        "--vae",
        required=True,
        help="VAE file: path or filename under vae/.",
    )
    parser.add_argument(
        "--prompt",
        default="a portrait of a woman, studio lighting, detailed",
        help="Positive prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, distorted",
        help="Negative prompt.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (multiple of 8).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (multiple of 8).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Sampling steps.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--sampler",
        default="euler",
        help="Sampler name (e.g. euler, dpm_2, ddim).",
    )
    parser.add_argument(
        "--scheduler",
        default="normal",
        help="Scheduler name (e.g. normal, simple, karras).",
    )
    parser.add_argument(
        "--output",
        default="output_separate.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--image",
        default="",
        help="Input image for img2img. If omitted, runs txt2img (empty latent).",
    )
    parser.add_argument(
        "--denoise",
        type=float,
        default=0.85,
        help="Denoise strength for img2img (0–1). Only used when --image is set (default 0.85).",
    )
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1

    models_dir = Path(args.models_dir)

    # Resolve component paths (filename → models_dir/subdir/filename)
    unet_path = _resolve_path(
        models_dir,
        args.unet.strip(),
        ["diffusion_models", "unet"],
    )
    clip_path = _resolve_path(
        models_dir,
        args.clip.strip(),
        ["text_encoders", "clip"],
    )
    vae_path = _resolve_path(
        models_dir,
        args.vae.strip(),
        ["vae"],
    )

    if not unet_path.is_file():
        print("error: diffusion model file not found:", unet_path, file=sys.stderr)
        return 1
    if not clip_path.is_file():
        print("error: CLIP file not found:", clip_path, file=sys.stderr)
        return 1
    if not vae_path.is_file():
        print("error: VAE file not found:", vae_path, file=sys.stderr)
        return 1

    # Resolve optional input image (img2img)
    input_image_path: Path | None = None
    if args.image.strip():
        p = Path(args.image.strip())
        if p.is_absolute() and p.is_file():
            input_image_path = p
        elif p.is_file():
            input_image_path = p.resolve()
        else:
            print("error: input image not found:", args.image, file=sys.stderr)
            return 1

    # 1) Runtime check
    from comfy_diffusion import check_runtime, vae_decode, vae_encode
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.sampling import sample

    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1
    print("runtime:", runtime.get("comfyui_version", "?"), runtime.get("device", "?"))

    # 2) Load components separately
    manager = ModelManager(args.models_dir)
    model = manager.load_unet(unet_path)
    clip = manager.load_clip(clip_path)
    vae = manager.load_vae(vae_path)
    print("loaded diffusion model:", unet_path.name)
    print("loaded CLIP:", clip_path.name)
    print("loaded VAE:", vae_path.name)

    # 3) Encode prompts
    positive = encode_prompt(clip, args.prompt)
    negative = encode_prompt(clip, args.negative_prompt)

    # 4) Latent: img2img (vae_encode input image) or txt2img (empty latent)
    if input_image_path is not None:
        input_pil = Image.open(input_image_path).convert("RGB")
        latent = vae_encode(vae, input_pil)
        denoise = args.denoise
        print("mode: img2img, denoise:", denoise)
    else:
        latent = _empty_latent(args.width, args.height, batch_size=1)
        denoise = 1.0
        print("mode: txt2img")

    # 5) Sample
    denoised = sample(
        model,
        positive,
        negative,
        latent,
        steps=args.steps,
        cfg=args.cfg,
        sampler_name=args.sampler,
        scheduler=args.scheduler,
        seed=args.seed,
        denoise=denoise,
    )

    # 6) VAE decode → PIL
    image = vae_decode(vae, denoised)
    image.save(args.output)
    print("saved:", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
