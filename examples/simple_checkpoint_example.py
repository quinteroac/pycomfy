#!/usr/bin/env python3
"""
Simple checkpoint example using pycomfy.

Runs the complete flow: runtime check → load checkpoint → encode prompts →
create empty latent → sample → VAE decode → save image.

Setup (from repo root):
  1. ComfyUI submodule:
       git submodule update --init
  2. Python deps: uv sync --extra comfyui (installs CPU torch; same for everyone).
       GPU: then uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
            (RTX 50xx: cu128). Without --force-reinstall, torch stays CPU.

Usage:
  export PYCOMFY_MODELS_DIR=/path/to/models
  export PYCOMFY_CHECKPOINT=your_checkpoint.safetensors
  uv run python examples/simple_checkpoint_example.py
  uv run python examples/simple_checkpoint_example.py --lora path/to/lora.safetensors --lora-strength-model 0.8

If you see "ComfyUI runtime not found", the message will include the missing
module (e.g. psutil, torch, safetensors); install ComfyUI's requirements as above.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _empty_latent(width: int, height: int, batch_size: int = 1) -> dict:
    """Build an empty LATENT dict for txt2img (ComfyUI contract)."""
    from pycomfy._runtime import ensure_comfyui_on_path

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
        description="Simple checkpoint example: load one checkpoint and run txt2img with pycomfy.",
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
        default="output.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--lora",
        default="",
        help="Path to LoRA file (.safetensors). If relative and not found, tries <models-dir>/loras/<value>. Omit to skip LoRA.",
    )
    parser.add_argument(
        "--lora-strength-model",
        type=float,
        default=1.0,
        help="LoRA strength for model (default 1.0).",
    )
    parser.add_argument(
        "--lora-strength-clip",
        type=float,
        default=1.0,
        help="LoRA strength for CLIP (default 1.0).",
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

    # Resolve optional LoRA path
    lora_path: str | None = None
    if args.lora.strip():
        p = Path(args.lora.strip())
        if p.is_absolute() and p.is_file():
            lora_path = str(p)
        elif p.is_file():
            lora_path = str(p.resolve())
        elif args.models_dir:
            fallback = Path(args.models_dir) / "loras" / p.name
            if fallback.is_file():
                lora_path = str(fallback)
        if lora_path is None:
            print(
                "error: LoRA file not found:",
                args.lora,
                "(tried cwd and <models-dir>/loras/)",
                file=sys.stderr,
            )
            return 1

    # 1) Runtime check
    from pycomfy import check_runtime, vae_decode, apply_lora
    from pycomfy.conditioning import encode_prompt
    from pycomfy.models import ModelManager
    from pycomfy.sampling import sample

    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1
    print("runtime:", runtime.get("comfyui_version", "?"), runtime.get("device", "?"))

    # 2) Load checkpoint
    manager = ModelManager(args.models_dir)
    checkpoint = manager.load_checkpoint(args.checkpoint.strip())
    print("loaded checkpoint:", args.checkpoint)

    if checkpoint.clip is None:
        print("error: checkpoint has no CLIP (cannot encode prompts)", file=sys.stderr)
        return 1
    if checkpoint.vae is None:
        print("error: checkpoint has no VAE (cannot decode latent)", file=sys.stderr)
        return 1

    # 2b) Apply LoRA if requested
    model = checkpoint.model
    clip = checkpoint.clip
    if lora_path is not None:
        model, clip = apply_lora(
            checkpoint.model,
            checkpoint.clip,
            lora_path,
            args.lora_strength_model,
            args.lora_strength_clip,
        )
        print("applied LoRA:", lora_path, f"(model={args.lora_strength_model}, clip={args.lora_strength_clip})")
    if clip is None:
        clip = checkpoint.clip

    # 3) Encode prompts
    positive = encode_prompt(clip, args.prompt)
    negative = encode_prompt(clip, args.negative_prompt)

    # 4) Empty latent
    latent = _empty_latent(args.width, args.height, batch_size=1)

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
        denoise=1.0,
    )

    # 6) VAE decode → PIL
    image = vae_decode(checkpoint.vae, denoised)
    image.save(args.output)
    print("saved:", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
