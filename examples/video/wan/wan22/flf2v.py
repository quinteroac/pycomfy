#!/usr/bin/env python3
"""WAN 2.2 first-last-frame-to-video example using comfy_diffusion.

Demonstrates usage of the ``comfy_diffusion.pipelines.video.wan.wan22.flf2v``
pipeline: download models once, then call ``run()`` to interpolate between a
start frame and an end frame using the WAN 2.2 14B FLF2V model.

The pipeline uses the WAN 2.2 14B dual-model architecture with a dual
two-pass KSamplerAdvanced flow and ModelSamplingSD3 (shift=8) applied to
both models:

- Pass 1: high-noise UNet, add_noise=True, steps 0→steps//2,
  return_with_leftover_noise=True
- Pass 2: low-noise UNet, add_noise=False, steps//2→steps,
  return_with_leftover_noise=False

No LoRA is applied — all LoRA nodes in the reference workflow are bypassed.

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models (idempotent — skips files already present)
    uv run python examples/wan22_flf2v.py --download-only

    # Interpolate between two frames
    uv run python examples/wan22_flf2v.py \\
        --start-image start.jpg \\
        --end-image end.jpg \\
        --prompt "a crystal cat waking up and transforming into a giant beast"

    # Full options
    uv run python examples/wan22_flf2v.py \\
        --models-dir /path/to/models \\
        --start-image start.jpg \\
        --end-image end.jpg \\
        --prompt "a crystal cat waking up and transforming into a giant beast" \\
        --width 640 --height 640 --length 81 \\
        --steps 20 --cfg 4.0 --seed 42 \\
        --output output.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path


def _save_frames_as_video(
    frames: list,
    path: str | Path,
    fps: float = 16.0,
) -> None:
    """Write PIL frames to an MP4 file using PyAV, or fall back to PNG frames."""
    try:
        import av
    except ImportError:
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
        description="WAN 2.2 first-last-frame-to-video example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory. Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--start-image",
        required=False,
        default=None,
        help="Path to the starting frame image.",
    )
    parser.add_argument(
        "--end-image",
        required=False,
        default=None,
        help="Path to the ending frame image.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "a crystal cat waking up and transforming into a giant beast, "
            "cinematic motion, smooth transition"
        ),
        help="Positive text prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
            "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
            "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        ),
        help="Negative text prompt.",
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width in pixels.")
    parser.add_argument("--height", type=int, default=640, help="Frame height in pixels.")
    parser.add_argument(
        "--length",
        type=int,
        default=81,
        help="Number of frames (81 ≈ 5 s at 16 fps).",
    )
    parser.add_argument("--steps", type=int, default=20, help="Total denoising steps.")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fps", type=float, default=16.0, help="Output video FPS.")
    parser.add_argument(
        "--output", default="wan22_flf2v_output.mp4", help="Output video path."
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download models then exit without running inference.",
    )
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1

    models_dir = Path(args.models_dir)

    # Import pipeline.
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.wan.wan22.flf2v import manifest, run

    # Download models (idempotent — skips files already present).
    print("Checking / downloading models …")
    download_models(manifest(), models_dir=models_dir)
    print("Models ready.")

    if args.download_only:
        print("--download-only: exiting without inference.")
        return 0

    if not args.start_image or not args.end_image:
        print(
            "error: --start-image and --end-image are required for inference "
            "(or use --download-only)",
            file=sys.stderr,
        )
        return 1

    start_path = Path(args.start_image)
    end_path = Path(args.end_image)

    if not start_path.is_file():
        print(f"error: start image not found: {start_path}", file=sys.stderr)
        return 1

    if not end_path.is_file():
        print(f"error: end image not found: {end_path}", file=sys.stderr)
        return 1

    # Runtime check.
    from comfy_diffusion.runtime import check_runtime

    runtime = check_runtime()
    if runtime.get("error"):
        print(f"error: runtime check failed: {runtime['error']}", file=sys.stderr)
        return 1

    # Load input images.
    from PIL import Image

    start_image = Image.open(start_path).convert("RGB")
    end_image = Image.open(end_path).convert("RGB")

    # Generate frames.
    print(
        f"Interpolating {start_path.name} → {end_path.name}: "
        f"{args.length} frames ({args.width}x{args.height}) …"
    )
    frames = run(
        start_image,
        end_image,
        args.prompt,
        args.negative_prompt,
        args.width,
        args.height,
        args.length,
        models_dir=models_dir,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
    )
    print(f"Generated {len(frames)} frames.")

    # Save output.
    _save_frames_as_video(frames, args.output, fps=args.fps)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
