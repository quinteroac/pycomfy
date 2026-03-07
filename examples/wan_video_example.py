#!/usr/bin/env python3
"""
Wan text-to-video example using pycomfy.

Runs the full flow: runtime check → load Wan model, CLIP, and VAE separately →
encode prompts → build empty video latent (5-D) → sample → VAE decode batch →
save as MP4.

Wan is a video diffusion model (Wan 2.1). Model weights are usually split:
  - Diffusion model (UNet) in diffusion_models/ or unet/
  - Text encoder (e.g. UMT5-XXL for Wan) in text_encoders/ or clip/
  - VAE in vae/

Setup (from repo root):
  1. ComfyUI submodule: git submodule update --init
  2. Python deps: uv sync --extra comfyui
  3. GPU (recommended for video): uv pip install torch torchvision torchaudio
        --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
  4. Download Wan 2.1 model, text encoder (umt5-xxl), and VAE; place under
     PYCOMFY_MODELS_DIR in diffusion_models/ (or unet/), text_encoders/ (or clip/),
     and vae/ respectively.

Usage:
  export PYCOMFY_MODELS_DIR=/path/to/models
  export PYCOMFY_WAN_UNET=wan2.1.safetensors
  export PYCOMFY_WAN_CLIP=umt5_xxl.safetensors
  export PYCOMFY_WAN_VAE=wan_vae.safetensors
  uv run python examples/wan_video_example.py
  uv run python examples/wan_video_example.py --unet wan.safetensors --clip umt5_xxl.safetensors --vae wan_vae.safetensors --output out.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path

from PIL import Image


def _empty_wan_latent(
    width: int,
    height: int,
    length: int,
    batch_size: int = 1,
) -> dict:
    """Build an empty 5-D LATENT dict for Wan text-to-video (ComfyUI contract).

    Wan latent shape: (batch, 16, temporal_steps, height//8, width//8).
    Temporal steps = ((length - 1) // 4) + 1 (length = number of output frames).
    """
    from pycomfy._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import torch
    import comfy.model_management

    device = comfy.model_management.intermediate_device()
    latent_t = ((length - 1) // 4) + 1
    latent = torch.zeros(
        [batch_size, 16, latent_t, height // 8, width // 8],
        device=device,
    )
    return {"samples": latent}


def _save_frames_as_video(
    frames: list[Image.Image],
    path: str | Path,
    fps: float = 16.0,
) -> None:
    """Write a list of PIL images to an MP4 file using PyAV (av)."""
    try:
        import av
    except ImportError:
        # Fallback: save frames as PNGs in a directory
        out_dir = Path(path).with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(frames):
            img.save(out_dir / f"frame_{i:04d}.png")
        print(f"PyAV not available; saved {len(frames)} frames to {out_dir}/")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("frames must not be empty")

    w, h = frames[0].size
    container = av.open(str(path), "w")
    rate = Fraction(int(fps), 1) if fps == int(fps) else Fraction(round(fps * 1000), 1000)
    stream = container.add_stream("libx264", rate=rate)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    for i, pil_img in enumerate(frames):
        frame = av.VideoFrame.from_image(pil_img)
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wan text-to-video example (pycomfy).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root (diffusion_models/, text_encoders/, vae/). Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--unet",
        default=os.environ.get("PYCOMFY_WAN_UNET", ""),
        help="Wan diffusion model filename in diffusion_models/ or unet/. Default: PYCOMFY_WAN_UNET.",
    )
    parser.add_argument(
        "--clip",
        default=os.environ.get("PYCOMFY_WAN_CLIP", ""),
        help="Text encoder filename (e.g. umt5_xxl) in text_encoders/ or clip/. Default: PYCOMFY_WAN_CLIP.",
    )
    parser.add_argument(
        "--vae",
        default=os.environ.get("PYCOMFY_WAN_VAE", ""),
        help="VAE filename in vae/. Default: PYCOMFY_WAN_VAE.",
    )
    parser.add_argument(
        "--prompt",
        default="A serene landscape with gentle wind moving the grass, cinematic",
        help="Positive prompt for the video.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, distorted, static",
        help="Negative prompt.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (multiple of 8).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (multiple of 8).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=81,
        help="Number of frames (e.g. 81 ≈ 5s at 16fps). Must be (4*n)+1 for full latent.",
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
        default="wan_output.mp4",
        help="Output video path (MP4).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="Output video FPS (default 16, typical for Wan).",
    )
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1
    if not args.unet.strip():
        print(
            "error: --unet (or PYCOMFY_WAN_UNET) is required",
            file=sys.stderr,
        )
        return 1
    if not args.clip.strip():
        print(
            "error: --clip (or PYCOMFY_WAN_CLIP) is required",
            file=sys.stderr,
        )
        return 1
    if not args.vae.strip():
        print(
            "error: --vae (or PYCOMFY_WAN_VAE) is required",
            file=sys.stderr,
        )
        return 1

    # 1) Runtime check
    from pycomfy import check_runtime, vae_decode_batch
    from pycomfy.conditioning import encode_prompt
    from pycomfy.models import ModelManager
    from pycomfy.sampling import sample

    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1
    print("runtime:", runtime.get("comfyui_version", "?"), runtime.get("device", "?"))

    # 2) Load Wan model, CLIP, and VAE separately (from diffusion_models, text_encoders, vae)
    manager = ModelManager(args.models_dir)
    model = manager.load_unet(args.unet.strip())
    print("loaded unet:", args.unet)
    clip = manager.load_clip(args.clip.strip(), clip_type="wan")
    print("loaded clip:", args.clip, "(type=wan)")
    vae = manager.load_vae(args.vae.strip())
    print("loaded vae:", args.vae)

    # 3) Encode prompts
    positive = encode_prompt(clip, args.prompt)
    negative = encode_prompt(clip, args.negative_prompt)

    # 4) Empty Wan video latent (5-D)
    latent = _empty_wan_latent(
        width=args.width,
        height=args.height,
        length=args.length,
        batch_size=1,
    )
    print(
        "latent shape:",
        latent["samples"].shape,
        f"(frames ~{args.length})",
    )

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

    # 6) VAE decode batch → list of PIL images
    frames = vae_decode_batch(vae, denoised)
    print("decoded frames:", len(frames))

    # 7) Save as MP4 (or directory of PNGs if PyAV missing)
    _save_frames_as_video(frames, args.output, fps=args.fps)
    print("saved:", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
