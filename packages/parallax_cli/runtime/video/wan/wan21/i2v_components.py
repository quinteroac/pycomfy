#!/usr/bin/env python3
"""
Wan image-to-video (I2V) example using comfy_diffusion.

Uses comfy_diffusion.conditioning.wan_image_to_video for I2V conditioning (empty
16ch @ 1/8 latent + concat_latent_image/concat_mask when a start image is given).
Two diffusion models (--unet-high, --unet-low), two-stage sampling. Matches the
official ComfyUI I2V workflow (no LoRA; models can be merged).

  uv sync --extra comfyui
  uv run python examples/wan_video_example.py --unet-high ... --unet-low ... --clip ... --vae wan_2.1_vae.safetensors --image first_frame.png [--width 832 --height 480 --length 81]
"""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image


def _wan22_latent_to_wan21_for_decode(latent: dict, width: int, height: int) -> dict:
    """Convert Wan 2.2 latent to 16ch @ 1/8 for Wan 2.1 VAE decode.

    Wan 2.2 is decoded with Wan 2.1 VAE. The 2.1 VAE expects 16ch @ 1/8 spatial.
    - If the sampler returns 48ch @ 1/16: take first 16ch and 2x spatial upsample.
    - If the sampler returns 16ch @ 1/16: only 2x spatial upsample to 1/8.
    Without this, decode would treat 1/16 spatial as 1/8 and output half resolution.
    """
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import torch

    samples = latent["samples"]
    b, c, t, h, w = samples.shape
    target_h, target_w = height // 8, width // 8
    # Need conversion when spatial is 1/16 (Wan 2.2 scale) but VAE expects 1/8.
    is_spatial_1_16 = (h, w) == (height // 16, width // 16)

    if samples.shape[1] == 48 and is_spatial_1_16:
        # 48ch @ 1/16 -> 16ch @ 1/8: first 16 ch + 2x spatial
        s16 = samples[:, :16, :, :, :].contiguous()
    elif samples.shape[1] == 16 and is_spatial_1_16:
        # 16ch @ 1/16 -> 16ch @ 1/8: 2x spatial only
        s16 = samples
    else:
        return latent

    s16 = torch.nn.functional.interpolate(
        s16.reshape(b * t, 16, h, w),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    s16 = s16.reshape(b, 16, t, target_h, target_w).to(samples.device, dtype=samples.dtype)
    out = latent.copy()
    out["samples"] = s16
    return out


def _apply_model_sampling_shift(model: Any, shift: float = 5.0, multiplier: float = 1000.0) -> Any:
    """Apply ModelSamplingSD3-style patch: set model_sampling shift (and multiplier) for flow models.

    Same logic as ComfyUI node ModelSamplingSD3: clone model, create ModelSamplingDiscreteFlow+CONST,
    set_parameters(shift=shift, multiplier=multiplier), add_object_patch. Wan uses FLOW sampling.
    """
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_sampling

    m = model.clone()
    sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
    sampling_type = comfy.model_sampling.CONST

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=shift, multiplier=multiplier)
    m.add_object_patch("model_sampling", model_sampling)
    return m


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
    # CRF 18 = high quality (lower = sharper, default 23 is often too blocky for AI frames)
    stream = container.add_stream("libx264", rate=rate, options={"crf": "18"})
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
        description="Wan 2.2 image-to-video (I2V) example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root (diffusion_models/, text_encoders/, vae/). Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--unet-high",
        default=os.environ.get("PYCOMFY_WAN_UNET_HIGH", ""),
        help="Wan 2.2 high-noise model (required).",
    )
    parser.add_argument(
        "--unet-low",
        default=os.environ.get("PYCOMFY_WAN_UNET_LOW", ""),
        help="Wan 2.2 low-noise model (required).",
    )
    parser.add_argument(
        "--high-steps",
        type=int,
        default=10,
        help="Wan 2.2: steps run with high-noise model (default 10). Remaining steps use low-noise.",
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
        "--sampling-shift",
        type=float,
        default=5.0,
        help="Model sampling shift (ModelSamplingSD3-style). Applied to high and low when two-stage (default 5).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="Output video FPS (default 16, typical for Wan).",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to start image (first frame) for image-to-video.",
    )
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1

    if not args.unet_high.strip() or not args.unet_low.strip():
        print(
            "error: both --unet-high and --unet-low are required",
            file=sys.stderr,
        )
        return 1
    if args.high_steps >= args.steps:
        print(
            "error: --high-steps must be less than --steps for Wan 2.2 two-stage",
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

    image_path = Path(args.image.strip())
    if not image_path.is_file():
        image_path = image_path.resolve()
    if not image_path.is_file():
        print("error: --image file not found:", args.image, file=sys.stderr)
        return 1

    # 1) Runtime check
    from comfy_diffusion import check_runtime, vae_decode_batch
    from comfy_diffusion.conditioning import encode_prompt, wan_image_to_video
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.sampling import sample_advanced

    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1

    manager = ModelManager(args.models_dir)
    model_high = manager.load_unet(args.unet_high.strip())
    model_low = manager.load_unet(args.unet_low.strip())
    clip = manager.load_clip(args.clip.strip(), clip_type="wan")
    vae = manager.load_vae(args.vae.strip())

    if args.sampling_shift != 1.0:
        model_high = _apply_model_sampling_shift(model_high, shift=args.sampling_shift)
        model_low = _apply_model_sampling_shift(model_low, shift=args.sampling_shift)

    # 3) Encode prompts
    positive = encode_prompt(clip, args.prompt)
    negative = encode_prompt(clip, args.negative_prompt)

    # 4) WAN image-to-video conditioning + empty latent (comfy_diffusion.conditioning)
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import numpy as np
    import torch
    import comfy.model_management

    pil_img = Image.open(image_path).convert("RGB")
    arr = np.array(pil_img)
    device = comfy.model_management.intermediate_device()
    # start_image must be a tensor (batch, height, width, channels), float in [0, 1]
    start_image_tensor = (
        torch.from_numpy(arr).float().to(device=device) / 255.0
    ).unsqueeze(0)

    positive, negative, latent = wan_image_to_video(
        positive,
        negative,
        vae,
        args.width,
        args.height,
        args.length,
        batch_size=1,
        start_image=start_image_tensor,
        clip_vision_output=None,
    )

    # 5) Two-stage sampling: high-noise model then low-noise model (match ComfyUI workflow)
    # First KSampler: return_with_leftover_noise so second stage continues from same noise schedule
    denoised = sample_advanced(
        model_high,
        positive,
        negative,
        latent,
        steps=args.steps,
        cfg=args.cfg,
        sampler_name=args.sampler,
        scheduler=args.scheduler,
        noise_seed=args.seed,
        add_noise=True,
        start_at_step=0,
        end_at_step=args.high_steps,
        denoise=1.0,
        return_with_leftover_noise=True,
    )
    denoised = sample_advanced(
        model_low,
        positive,
        negative,
        denoised,
        steps=args.steps,
        cfg=args.cfg,
        sampler_name=args.sampler,
        scheduler=args.scheduler,
        noise_seed=args.seed,
        add_noise=False,
        start_at_step=args.high_steps,
        end_at_step=args.steps,
        denoise=1.0,
        return_with_leftover_noise=False,
    )

    # 6) VAE decode: if sampler output is 48ch/16ch @ 1/16, convert to 16ch @ 1/8 for Wan 2.1 VAE
    to_decode = _wan22_latent_to_wan21_for_decode(denoised, args.width, args.height)
    frames = vae_decode_batch(vae, to_decode)

    _save_frames_as_video(frames, args.output, fps=args.fps)
    print(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
