#!/usr/bin/env python3
"""WAN 2.2 sound-to-video example using comfy_diffusion.

Demonstrates usage of the ``comfy_diffusion.pipelines.video.wan.wan22.s2v``
pipeline: download models once, then call ``run()`` to generate audio-driven
video from an audio file and a reference image using the WAN 2.2 14B S2V model.

The pipeline uses:
- WAN 2.2 S2V 14B UNet with LightX2V LoRA
- ModelSamplingSD3(shift=8)
- AudioEncoderEncode (wav2vec2) → WanSoundImageToVideo (initial pass)
- Multi-pass WanSoundImageToVideoExtend + KSampler + LatentConcat loop
- VAEDecode to PIL frames

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models (idempotent — skips files already present)
    uv run python examples/wan22_s2v.py --download-only

    # Generate a video from audio + reference image
    uv run python examples/wan22_s2v.py \\
        --audio input.mp3 \\
        --image reference.jpg \\
        --prompt "The man is playing the guitar"

    # Full options
    uv run python examples/wan22_s2v.py \\
        --models-dir /path/to/models \\
        --audio input.mp3 \\
        --image reference.jpg \\
        --prompt "The man is playing the guitar" \\
        --steps 10 --cfg 6.0 --seed 42 \\
        --length 77 --extend-passes 2 \\
        --output wan22_s2v_output.mp4
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
        description="WAN 2.2 sound-to-video example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory. Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--audio",
        required=False,
        default=None,
        help="Path to the input audio file (MP3, WAV, etc.).",
    )
    parser.add_argument(
        "--image",
        required=False,
        default=None,
        help="Path to the reference image that anchors the video identity.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "The man is playing the guitar. He looks down at his hands "
            "playing the guitar and sings affectionately and gently."
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
    parser.add_argument("--steps", type=int, default=10, help="Denoising steps per pass.")
    parser.add_argument("--cfg", type=float, default=6.0, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--width", type=int, default=640, help="Frame width in pixels.")
    parser.add_argument("--height", type=int, default=640, help="Frame height in pixels.")
    parser.add_argument(
        "--length",
        type=int,
        default=77,
        help="Frames per segment (initial + each extend pass). Default 77 ≈ 5 s at 16 fps.",
    )
    parser.add_argument(
        "--extend-passes",
        type=int,
        default=2,
        help="Number of WanSoundImageToVideoExtend passes after the initial segment.",
    )
    parser.add_argument("--fps", type=float, default=16.0, help="Output video FPS.")
    parser.add_argument(
        "--output",
        default="wan22_s2v_output.mp4",
        help="Output video path.",
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
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest, run

    # Download models (idempotent — skips files already present).
    print("Checking / downloading models …")
    download_models(manifest(), models_dir=models_dir)
    print("Models ready.")

    if args.download_only:
        print("--download-only: exiting without inference.")
        return 0

    if not args.audio:
        print(
            "error: --audio is required for inference (or use --download-only)",
            file=sys.stderr,
        )
        return 1

    if not args.image:
        print(
            "error: --image is required for inference (or use --download-only)",
            file=sys.stderr,
        )
        return 1

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"error: image file not found: {image_path}", file=sys.stderr)
        return 1

    # Runtime check.
    from comfy_diffusion.runtime import check_runtime

    runtime = check_runtime()
    if runtime.get("error"):
        print(f"error: runtime check failed: {runtime['error']}", file=sys.stderr)
        return 1

    # Load inputs.
    from comfy_diffusion.audio import load_audio
    from PIL import Image

    print(f"Loading audio from {audio_path.name} …")
    audio = load_audio(str(audio_path))

    print(f"Loading reference image from {image_path.name} …")
    ref_image = Image.open(image_path).convert("RGB")

    total_frames = args.length * (1 + args.extend_passes)
    print(
        f"Generating ~{total_frames} frames "
        f"({args.width}x{args.height}, {args.extend_passes} extend passes) …"
    )

    frames = run(
        audio,
        ref_image,
        None,
        args.prompt,
        args.negative_prompt,
        models_dir=models_dir,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        width=args.width,
        height=args.height,
        length=args.length,
        num_extend_passes=args.extend_passes,
    )
    print(f"Generated {len(frames)} frames.")

    # Save output.
    _save_frames_as_video(frames, args.output, fps=args.fps)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
