#!/usr/bin/env python3
"""WAN 2.2 text-to-video example using comfy_diffusion.

Demonstrates usage of the ``comfy_diffusion.pipelines.video.wan.wan22.t2v``
pipeline: download models once, then call ``run()`` to generate video frames
from a text prompt.

The pipeline uses the WAN 2.2 14B dual-model architecture with LightX2V 4-step
LoRA and a dual two-pass KSamplerAdvanced flow (high-noise first, low-noise
second) for high-quality video generation.

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models (idempotent — skips files already present)
    uv run python examples/wan22_t2v.py --download-only

    # Generate video
    uv run python examples/wan22_t2v.py --prompt "a serene mountain landscape with gentle wind"

    # Full options
    uv run python examples/wan22_t2v.py \\
        --models-dir /path/to/models \\
        --prompt "a fox running through a snowy forest" \\
        --width 832 --height 480 --length 81 \\
        --steps 4 --cfg 1.0 --seed 42 \\
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
        description="WAN 2.2 text-to-video example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory. Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--prompt",
        default="a serene mountain landscape with gentle wind moving the grass, cinematic",
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
    parser.add_argument("--width", type=int, default=832, help="Frame width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Frame height in pixels.")
    parser.add_argument(
        "--length",
        type=int,
        default=81,
        help="Number of frames (81 ≈ 5 s at 16 fps).",
    )
    parser.add_argument("--steps", type=int, default=4, help="Total denoising steps.")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fps", type=float, default=16.0, help="Output video FPS.")
    parser.add_argument("--output", default="wan22_t2v_output.mp4", help="Output video path.")
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
    from comfy_diffusion.pipelines.video.wan.wan22.t2v import manifest, run

    # Download models (idempotent — skips files already present).
    print("Checking / downloading models …")
    download_models(manifest(), models_dir=models_dir)
    print("Models ready.")

    if args.download_only:
        print("--download-only: exiting without inference.")
        return 0

    # Runtime check.
    from comfy_diffusion.runtime import check_runtime

    runtime = check_runtime()
    if runtime.get("error"):
        print(f"error: runtime check failed: {runtime['error']}", file=sys.stderr)
        return 1

    # Generate frames.
    print(f"Generating {args.length} frames ({args.width}x{args.height}) …")
    frames = run(
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
