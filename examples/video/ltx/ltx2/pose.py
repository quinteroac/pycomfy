#!/usr/bin/env python3
"""
LTX-Video 2 Pose-to-Video example using comfy_diffusion pipelines.

Extracts frames from an input video, runs DWPreprocessor pose estimation to
produce skeleton overlays, and generates a new video that follows the detected
poses — including a matching audio track. Uses the 19B dev fp8 checkpoint with
a pose-control LoRA.

  uv sync --extra comfyui
  uv run python examples/video/ltx/ltx2/pose.py \\
      --models-dir /path/to/models \\
      --video dance.mp4 \\
      --first-frame first_frame.png \\
      --prompt "a woman moves gracefully through a sunlit garden, birdsong"

  # Download models first (idempotent):
  # from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest
  # from comfy_diffusion.downloader import download_models
  # download_models(manifest(), models_dir="/path/to/models")
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


def _save_audio(waveform: Any, sample_rate: int, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torchaudio
        torchaudio.save(str(out), waveform.cpu(), sample_rate)
    except ImportError:
        import numpy as np
        from scipy.io import wavfile
        wav_data = waveform.cpu().numpy() if hasattr(waveform, "numpy") else np.array(waveform)
        wav_data = (np.clip(wav_data, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(str(out), sample_rate, wav_data.T)


def _save_video(frames: list[Any], output_path: str, fps: float = 25.0) -> None:
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
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LTX-Video 2 Pose-to-Video example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory. Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file (pose source).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Positive text prompt describing the output video and audio.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative text prompt.",
    )
    parser.add_argument(
        "--first-frame",
        default=None,
        dest="first_frame",
        help=(
            "Optional path to an image used as the temporal anchor frame "
            "(loaded separately from the pose video). "
            "If omitted the pipeline uses no separate first-frame image."
        ),
    )
    parser.add_argument("--width", type=int, default=1280, help="Output width in px (default 1280).")
    parser.add_argument("--height", type=int, default=720, help="Output height in px (default 720).")
    parser.add_argument("--length", type=int, default=121, help="Number of frames (default 121).")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate (default 25).")
    parser.add_argument("--cfg-pass1", type=float, default=3.0, help="CFG scale for pass 1 (default 3.0).")
    parser.add_argument("--cfg-pass2", type=float, default=1.0, help="CFG scale for pass 2 (default 1.0).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    parser.add_argument("--pose-lora-strength", type=float, default=1.0, help="Pose control LoRA strength (default 1.0).")
    parser.add_argument("--lora-strength", type=float, default=1.0, help="Distilled LoRA strength (default 1.0).")
    parser.add_argument("--output", default="ltx2_pose_output.mp4", help="Output video path.")
    parser.add_argument("--text-encoder-filename", default=None, help="Override text encoder filename in text_encoders/.")
    parser.add_argument("--ckpt-filename", default=None, help="Override checkpoint filename in checkpoints/.")
    parser.add_argument("--vae-filename", default=None, help="Override video VAE filename.")
    parser.add_argument("--audio-vae-filename", default=None, help="Override audio VAE filename.")
    parser.add_argument("--pose-lora-filename", default=None, help="Override pose control LoRA filename in loras/.")
    parser.add_argument("--lora-filename", default=None, help="Override distilled LoRA filename in loras/.")
    parser.add_argument("--upscaler-filename", default=None, help="Override upscaler filename in upscale_models/.")
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory\n"
            "usage: python examples/video/ltx/ltx2/pose.py --models-dir /path/to/models "
            "--video dance.mp4 --prompt '...'",
            file=sys.stderr,
        )
        return 1

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"error: --video file not found: {args.video}", file=sys.stderr)
        return 1

    first_frame_path: Path | None = None
    if args.first_frame:
        first_frame_path = Path(args.first_frame)
        if not first_frame_path.is_file():
            print(f"error: --first-frame file not found: {args.first_frame}", file=sys.stderr)
            return 1

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest, run

    download_models(manifest(), models_dir=args.models_dir)

    result = run(
        models_dir=args.models_dir,
        video_path=video_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        first_frame_path=first_frame_path,
        width=args.width,
        height=args.height,
        length=args.length,
        fps=args.fps,
        cfg_pass1=args.cfg_pass1,
        cfg_pass2=args.cfg_pass2,
        seed=args.seed,
        pose_lora_strength=args.pose_lora_strength,
        lora_strength=args.lora_strength,
        text_encoder_filename=args.text_encoder_filename,
        ckpt_filename=args.ckpt_filename,
        vae_filename=args.vae_filename,
        audio_vae_filename=args.audio_vae_filename,
        pose_lora_filename=args.pose_lora_filename,
        lora_filename=args.lora_filename,
        upscaler_filename=args.upscaler_filename,
    )

    frames = result["frames"]
    audio = result["audio"]
    print(f"decoded {len(frames)} frames")

    tmp_video = Path(args.output).with_suffix(".noaudio.mp4")
    _save_video(frames, str(tmp_video), fps=args.fps)

    tmp_audio = Path(args.output).with_suffix(".wav")
    try:
        _save_audio(audio["waveform"], audio["sample_rate"], str(tmp_audio))
        _mux_audio(str(tmp_video), str(tmp_audio), args.output)
        tmp_video.unlink(missing_ok=True)
        tmp_audio.unlink(missing_ok=True)
    except Exception as exc:
        print(f"warning: audio mux failed ({exc}); video saved to {tmp_video}", file=sys.stderr)
        tmp_video.rename(args.output)

    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
