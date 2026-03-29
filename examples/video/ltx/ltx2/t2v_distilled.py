#!/usr/bin/env python3
"""
LTX-Video 2 distilled text-to-video (T2V) example using comfy_diffusion pipelines.

Uses the 19B distilled checkpoint directly (no LoRA) — faster inference with
fewer steps. Generates video and audio simultaneously via the AV sampling chain.

  uv sync --extra comfyui
  uv run python examples/ltx2_t2v_distilled_example.py \\
      --models-dir /path/to/models \\
      --prompt "a golden retriever running through a sunlit park, birds singing"

  # Download models first (idempotent):
  # from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest
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
        description="LTX-Video 2 distilled text-to-video example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory. Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--prompt",
        default="a golden retriever running through a sunlit park, birds singing",
        help="Positive text prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative text prompt.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Output width in px (default 1280). Latent sampled at half, upscaler restores full size.")
    parser.add_argument("--height", type=int, default=720, help="Output height in px (default 720). Latent sampled at half, upscaler restores full size.")
    parser.add_argument("--length", type=int, default=121, help="Number of frames (default 121).")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate (default 25).")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale for both passes (default 1.0).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    parser.add_argument("--output", default="ltx2_t2v_distilled_output.mp4", help="Output video path.")
    parser.add_argument("--text-encoder-filename", default=None, help="Override text encoder filename in text_encoders/.")
    parser.add_argument("--unet-filename", default=None, help="Override UNet/checkpoint filename in diffusion_models/.")
    parser.add_argument("--vae-filename", default=None, help="Override video VAE filename.")
    parser.add_argument("--audio-vae-filename", default=None, help="Override audio VAE filename.")
    parser.add_argument("--upscaler-filename", default=None, help="Override upscaler filename in upscale_models/.")
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print("error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory", file=sys.stderr)
        return 1

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import manifest, run

    download_models(manifest(), models_dir=args.models_dir)

    result = run(
        models_dir=args.models_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        length=args.length,
        fps=args.fps,
        cfg=args.cfg,
        seed=args.seed,
        text_encoder_filename=args.text_encoder_filename,
        unet_filename=args.unet_filename,
        vae_filename=args.vae_filename,
        audio_vae_filename=args.audio_vae_filename,
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
