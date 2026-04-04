#!/usr/bin/env python3
"""
LTX-Video 2 audio-to-video example using comfy_diffusion pipelines.

Generates a video from a single input image and an audio file.  The pipeline
animates the image so that the motion is driven by the audio content.

Setup (from repo root):
  uv sync --extra comfyui

Usage:
  uv run python examples/video_ltx2_audio_to_video.py \\
      --models-dir /path/to/models \\
      --image first_frame.png \\
      --audio track.mp3 \\
      --audio-start 0.0 \\
      --audio-end 10.0 \\
      --prompt "the singer performs passionately on stage, dramatic lighting"

  # Models are downloaded automatically on first run (idempotent).
  # Set PYCOMFY_MODELS_DIR to avoid repeating --models-dir every time.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


def _save_video(frames: list[Any], output_path: str, fps: float = 24.0) -> None:
    """Save a list of PIL frames to an MP4 file via PyAV, or PNG fallback."""
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


def _mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Mux a silent video file with an audio file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LTX-Video 2 audio-to-video example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory. Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image (first frame to animate).",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the input audio file (MP3, WAV, etc.).",
    )
    parser.add_argument(
        "--audio-start",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Start time in seconds to crop the audio. Default: 0.0.",
    )
    parser.add_argument(
        "--audio-end",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="End time in seconds to crop the audio. Default: 10.0.",
    )
    parser.add_argument(
        "--prompt",
        default="the subject moves naturally with the music, cinematic motion",
        help="Positive text prompt describing the desired video content.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=704,
        help="Output frame width in pixels (default 704).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Output frame height in pixels (default 704).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=241,
        help="Number of frames per sampling segment (default 241).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output frame rate (default 24).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Denoising steps per segment (default 8).",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=1.0,
        help="CFG scale (default 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42).",
    )
    parser.add_argument(
        "--num-extensions",
        type=int,
        default=4,
        help="Number of video extension passes (default 4).",
    )
    parser.add_argument(
        "--output",
        default="ltx2_a2v_output.mp4",
        help="Output video path (default ltx2_a2v_output.mp4).",
    )
    parser.add_argument(
        "--unet-filename",
        default=None,
        help="Override UNet filename in diffusion_models/.",
    )
    parser.add_argument(
        "--text-encoder-filename",
        default=None,
        help="Override text encoder 1 filename in text_encoders/.",
    )
    parser.add_argument(
        "--text-encoder-2-filename",
        default=None,
        help="Override text encoder 2 filename in text_encoders/.",
    )
    parser.add_argument(
        "--audio-vae-filename",
        default=None,
        help="Override audio VAE filename in vae/.",
    )
    parser.add_argument(
        "--video-vae-filename",
        default=None,
        help="Override video VAE filename in vae/.",
    )
    args = parser.parse_args()

    # --- Validate required arguments ---
    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"error: --image file not found: {args.image}", file=sys.stderr)
        return 1

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"error: --audio file not found: {args.audio}", file=sys.stderr)
        return 1

    if args.audio_start >= args.audio_end:
        print(
            f"error: --audio-start ({args.audio_start}) must be less than "
            f"--audio-end ({args.audio_end})",
            file=sys.stderr,
        )
        return 1

    # --- Lazy imports: all comfy_diffusion symbols deferred to after arg validation ---
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import manifest, run

    # Download models (idempotent — skips files already present).
    print("checking / downloading models...")
    download_models(manifest(), models_dir=args.models_dir)

    # Run the pipeline.
    print("running audio-to-video pipeline...")
    result = run(
        models_dir=args.models_dir,
        prompt=args.prompt,
        image_path=image_path,
        audio_path=audio_path,
        audio_start_time=args.audio_start,
        audio_end_time=args.audio_end,
        width=args.width,
        height=args.height,
        length=args.length,
        fps=args.fps,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        num_extensions=args.num_extensions,
        unet_filename=args.unet_filename,
        text_encoder_filename=args.text_encoder_filename,
        text_encoder_2_filename=args.text_encoder_2_filename,
        audio_vae_filename=args.audio_vae_filename,
        video_vae_filename=args.video_vae_filename,
    )

    frames = result["frames"]
    print(f"decoded {len(frames)} frames")

    # Save frames as a silent video then mux with the original audio.
    output_path = args.output
    tmp_silent = str(Path(output_path).with_suffix(".silent.mp4"))
    _save_video(frames, tmp_silent, fps=float(args.fps))

    try:
        _mux_audio(tmp_silent, str(audio_path), output_path)
        Path(tmp_silent).unlink(missing_ok=True)
        print(output_path)
    except Exception as exc:
        print(f"warning: audio mux failed ({exc}); silent video saved to {tmp_silent}", file=sys.stderr)
        Path(tmp_silent).rename(output_path)
        print(output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
