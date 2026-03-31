#!/usr/bin/env python3
"""
WAN 2.1 text-to-video (T2V) example using comfy_diffusion pipelines.

Generates a video from a text prompt using the WAN 2.1 T2V 1.3B model.
Downloads model weights automatically on the first run (idempotent).

  uv sync --extra comfyui
  uv run python examples/video_wan21_t2v.py \\
      --models-dir /path/to/models \\
      --prompt "a fox moving quickly in a beautiful winter scenery"

  # Download models first (idempotent):
  # from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest
  # from comfy_diffusion.downloader import download_models
  # download_models(manifest(), models_dir="/path/to/models")
"""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


def _save_video(frames: list[Any], output_path: str, fps: float = 16.0) -> None:
    """Write a list of PIL images to an MP4 file using PyAV (fallback: PNG frames)."""
    try:
        import av
    except ImportError:
        out_dir = Path(output_path).with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(frames):
            img.save(out_dir / f"frame_{i:04d}.png")
        print(f"PyAV not available; saved {len(frames)} frames to {out_dir}/")
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = frames[0].size
    container = av.open(str(path), "w")
    rate = Fraction(int(fps), 1) if fps == int(fps) else Fraction(round(fps * 1000), 1000)
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
        description="WAN 2.1 text-to-video (T2V) example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory. Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--prompt",
        default="a fox moving quickly in a beautiful winter scenery",
        help="Positive text prompt describing the desired video content.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative text prompt (defaults to built-in WAN 2.1 negative prompt).",
    )
    parser.add_argument("--width", type=int, default=832, help="Output width in px (default 832).")
    parser.add_argument("--height", type=int, default=480, help="Output height in px (default 480).")
    parser.add_argument("--length", type=int, default=33, help="Number of frames (default 33).")
    parser.add_argument("--fps", type=int, default=16, help="Output frame rate (default 16).")
    parser.add_argument("--steps", type=int, default=30, help="Sampling steps (default 30).")
    parser.add_argument("--cfg", type=float, default=6.0, help="CFG guidance scale (default 6.0).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    parser.add_argument("--output", default="wan21_t2v_output.mp4", help="Output video path (default wan21_t2v_output.mp4).")
    parser.add_argument("--unet-filename", default=None, help="Override UNet filename in diffusion_models/.")
    parser.add_argument("--text-encoder-filename", default=None, help="Override text-encoder filename in text_encoders/.")
    parser.add_argument("--vae-filename", default=None, help="Override VAE filename in vae/.")
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest, run

    download_models(manifest(), models_dir=args.models_dir)

    kwargs: dict[str, Any] = dict(
        models_dir=args.models_dir,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        length=args.length,
        fps=args.fps,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        unet_filename=args.unet_filename,
        text_encoder_filename=args.text_encoder_filename,
        vae_filename=args.vae_filename,
    )
    if args.negative_prompt is not None:
        kwargs["negative_prompt"] = args.negative_prompt

    frames = run(**kwargs)
    print(f"decoded {len(frames)} frames")

    _save_video(frames, args.output, fps=args.fps)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
