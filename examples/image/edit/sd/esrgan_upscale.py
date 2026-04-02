#!/usr/bin/env python3
"""
Simple checkpoint example with ESRGAN image upscaling using comfy_diffusion.

Flow:
  1. check_runtime()
  2. load base checkpoint
  3. txt2img → generate a base latent
  4. VAE decode → save base image
  5. load ESRGAN / GAN upscaler model from <models-dir>/upscale/
  6. run ImageUpscaleWithModel → save upscaled image

Setup (from repo root):
  1. ComfyUI submodule:
       git submodule update --init
  2. Python deps (CPU by default):
       uv sync --extra comfyui
     GPU (optional, recommended for production):
       uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

Upscale model:
  - Uses a GAN/ESRGAN-style IMAGE upscaler model compatible with ComfyUI's
    ImageUpscaleWithModel node (e.g. RealESRGAN_x4plus.safetensors).
  - The model should live in <models-dir>/upscale/ or be provided as an
    absolute path via --esrgan-checkpoint.

Example usage:
  export PYCOMFY_MODELS_DIR=/path/to/models
  export PYCOMFY_CHECKPOINT=your_base_checkpoint.safetensors
  # Assumes the ESRGAN model lives at <models-dir>/upscale/RealESRGAN_x4plus.safetensors
  export PYCOMFY_ESRGAN_CHECKPOINT=RealESRGAN_x4plus.safetensors

  uv run python examples/simple_checkpoint_esrgan_upscale_example.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image


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


def _bhwc_tensor_to_pil(image_tensor: Any) -> Image.Image:
    """
    Convert a single BHWC float32 tensor in [0, 1] range into a RGB PIL image.

    Expects a tensor-like object with shape (1, H, W, 3).
    """
    image = image_tensor
    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()

    values = image[0].tolist()
    if not values or not values[0]:
        raise ValueError("upscaled image tensor is empty")

    height = len(values)
    width = len(values[0])

    def _clip_to_uint8(value: float) -> int:
        scaled = int(value * 255.0)
        if scaled < 0:
            return 0
        if scaled > 255:
            return 255
        return scaled

    pixels: list[tuple[int, int, int]] = []
    for row in values:
        for pixel in row:
            r = _clip_to_uint8(pixel[0])
            g = _clip_to_uint8(pixel[1])
            b = _clip_to_uint8(pixel[2])
            pixels.append((r, g, b))

    result = Image.new("RGB", (width, height))
    result.putdata(pixels)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple checkpoint example: txt2img + ESRGAN image upscale.",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root (must contain checkpoints/). Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PYCOMFY_CHECKPOINT", ""),
        help="Checkpoint filename in checkpoints/. Default: PYCOMFY_CHECKPOINT.",
    )
    parser.add_argument(
        "--esrgan-checkpoint",
        default=os.environ.get("PYCOMFY_ESRGAN_CHECKPOINT", ""),
        help=(
            "ESRGAN / GAN upscaler checkpoint filename in <models-dir>/upscale/ "
            "or absolute path. Default: PYCOMFY_ESRGAN_CHECKPOINT."
        ),
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
        default=768,
        help="Base image width (multiple of 8).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Base image height (multiple of 8).",
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
        "--output-base",
        default="output.png",
        help="Output path for the base image (before ESRGAN).",
    )
    parser.add_argument(
        "--output-upscaled",
        default="output_esrgan.png",
        help="Output path for the ESRGAN upscaled image.",
    )
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1
    if not args.checkpoint.strip():
        print(
            "error: --checkpoint (or PYCOMFY_CHECKPOINT) is required",
            file=sys.stderr,
        )
        return 1
    if not args.esrgan_checkpoint.strip():
        print(
            "error: --esrgan-checkpoint (or PYCOMFY_ESRGAN_CHECKPOINT) "
            "is required for this example",
            file=sys.stderr,
        )
        return 1

    # 1) Runtime check
    from comfy_diffusion import check_runtime, vae_decode_tiled
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.image import load_image, image_upscale_with_model
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.sampling import sample

    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1
    print("runtime:", runtime.get("comfyui_version", "?"), runtime.get("device", "?"))

    # 2) Load base checkpoint
    manager = ModelManager(args.models_dir)
    checkpoint = manager.load_checkpoint(args.checkpoint.strip())
    print("loaded checkpoint:", args.checkpoint)

    if checkpoint.clip is None:
        print("error: checkpoint has no CLIP (cannot encode prompts)", file=sys.stderr)
        return 1
    if checkpoint.vae is None:
        print("error: checkpoint has no VAE (cannot decode latent)", file=sys.stderr)
        return 1

    model = checkpoint.model
    clip = checkpoint.clip
    vae = checkpoint.vae

    # 3) Encode prompts
    positive = encode_prompt(clip, args.prompt)
    negative = encode_prompt(clip, args.negative_prompt)

    # 4) Base latent (txt2img)
    latent = _empty_latent(args.width, args.height, batch_size=1)
    print("mode: txt2img (base size:", args.width, "x", args.height, ")")

    # 5) Sample base latent
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
        denoise=1.0,
    )

    # 6) Decode base image
    image_base = vae_decode_tiled(vae, denoised)
    image_base.save(args.output_base)
    print("saved base image:", args.output_base)

    # 7) Load ESRGAN / GAN image upscaler model
    try:
        esrgan_model = manager.load_upscale_model(args.esrgan_checkpoint.strip())
    except Exception as exc:  # noqa: BLE001
        print("error: could not load ESRGAN upscaler model:", exc, file=sys.stderr)
        return 1
    print("loaded ESRGAN upscaler model:", args.esrgan_checkpoint)

    # 8) Reload base image as ComfyUI IMAGE tensor and run ESRGAN
    image_tensor, _mask = load_image(args.output_base)
    upscaled_tensor = image_upscale_with_model(esrgan_model, image_tensor)

    # 9) Convert upscaled tensor back to PIL and save
    image_upscaled = _bhwc_tensor_to_pil(upscaled_tensor)
    image_upscaled.save(args.output_upscaled)
    print("saved ESRGAN upscaled image:", args.output_upscaled)

    return 0


if __name__ == "__main__":
    sys.exit(main())

