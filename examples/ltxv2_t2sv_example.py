#!/usr/bin/env python3
"""
LTX-Video 2 Text-to-Sound-to-Video (T2SV) example using comfy_diffusion.

Demonstrates joint audio+video generation with the LTX-Video 2 architecture:
  1. check_runtime()
  2. Load diffusion UNet, video VAE, audio VAE, and LTXAV text encoder
  3. [Optional] Enhance prompt with generate_ltx2_prompt() using a local LLM
  4. Encode positive and negative text conditioning with encode_prompt()
  5. Inject frame-rate metadata via ltxv_conditioning()
  6. Create empty video latent with ltxv_empty_latent_video()
  7. Crop keyframe guides with ltxv_crop_guides() (no-op when no guides are set)
  8. Create empty audio latent with ltxv_empty_latent_audio()
  9. Concatenate into a joint AV NestedTensor latent with ltxv_concat_av_latent()
 10. Run joint audio+video denoising with sample()
 11. Separate video and audio latents with ltxv_separate_av_latent()
 12. [Optional] Upsample video latent 2x with ltxv_latent_upsample()
 13. Decode video latent frames with vae_decode_batch_tiled()
 14. Decode audio latent to waveform with ltxv_audio_vae_decode()
 15. Save video frames as PNG files and audio as WAV

Setup (from repo root):
  git submodule update --init
  uv sync --extra comfyui

Usage:
  export PYCOMFY_MODELS_DIR=/path/to/models
  export PYCOMFY_LTXV2_UNET=ltxv2_unet.safetensors
  export PYCOMFY_LTXV2_VAE=ltxv2_vae.safetensors
  export PYCOMFY_LTXV2_AUDIO_VAE=ltxv2_audio_vae.safetensors
  export PYCOMFY_LTXV2_TEXT_ENCODER=ltxv2_t5.safetensors
  export PYCOMFY_LTXV2_LTXAV_CHECKPOINT=ltxv2_ltxav.safetensors

  uv run python examples/ltxv2_t2sv_example.py \\
      --prompt "a dog runs through a sunlit meadow, wind in fur, birds singing" \\
      --output-dir ./ltxv2_output

  # With optional LLM prompt enhancement and latent upscaling:
  uv run python examples/ltxv2_t2sv_example.py \\
      --prompt "a serene mountain lake at dawn" \\
      --llm Qwen2.5-3B-Instruct-Q4_K_M.gguf \\
      --latent-upscale-model ltxv2_latent_upscaler.safetensors \\
      --output-dir ./ltxv2_output
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _save_audio(waveform: Any, sample_rate: int, output_path: str) -> None:
    """Save a waveform tensor to a WAV file (torchaudio preferred, scipy fallback)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torchaudio

        torchaudio.save(str(out), waveform.cpu(), sample_rate)
    except ImportError:
        import numpy as np

        wav_data = waveform.cpu().numpy() if hasattr(waveform, "numpy") else np.array(waveform)
        wav_data = (np.clip(wav_data, -1.0, 1.0) * 32767).astype(np.int16)
        try:
            from scipy.io import wavfile

            wavfile.write(str(out), sample_rate, wav_data.T)
        except ImportError:
            print(
                "error: install torchaudio or scipy to save WAV output",
                file=sys.stderr,
            )
            raise


def _combine_video(frame_dir: str, audio_path: str, output_path: str, frame_rate: float) -> None:
    """Combine PNG frames and a WAV file into an MP4 using ffmpeg."""
    import subprocess

    frame_pattern = str(Path(frame_dir) / "frame_%05d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(frame_rate),
        "-i", frame_pattern,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print(f"ffmpeg: {result.stderr}", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def _save_video_frames(frames: list[Any], output_dir: str) -> list[str]:
    """Save decoded video frames as numbered PNG files; return the saved paths."""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for i, frame in enumerate(frames):
        frame_path = str(dir_path / f"frame_{i:05d}.png")
        frame.save(frame_path)
        saved.append(frame_path)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LTX-Video 2 Text-to-Sound-to-Video generation (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help=(
            "Models root directory (checkpoints/, vae/, text_encoders/, etc.). "
            "Default: PYCOMFY_MODELS_DIR."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PYCOMFY_CHECKPOINT", ""),
        help="[Reserved] Checkpoint filename in checkpoints/. Default: PYCOMFY_CHECKPOINT.",
    )
    parser.add_argument(
        "--unet",
        default=os.environ.get("PYCOMFY_LTXV2_UNET", ""),
        help="LTX-Video 2 diffusion model filename in diffusion_models/ (or unet/). Default: PYCOMFY_LTXV2_UNET.",
    )
    parser.add_argument(
        "--vae",
        default=os.environ.get("PYCOMFY_LTXV2_VAE", ""),
        help="LTX-Video 2 video VAE filename in vae/. Default: PYCOMFY_LTXV2_VAE.",
    )
    parser.add_argument(
        "--audio-vae",
        default=os.environ.get("PYCOMFY_LTXV2_AUDIO_VAE", ""),
        help="LTX-Video 2 audio VAE filename in vae/. Default: PYCOMFY_LTXV2_AUDIO_VAE.",
    )
    parser.add_argument(
        "--text-encoder",
        default=os.environ.get("PYCOMFY_LTXV2_TEXT_ENCODER", ""),
        help=(
            "LTXAV text encoder filename in text_encoders/. "
            "Default: PYCOMFY_LTXV2_TEXT_ENCODER."
        ),
    )
    parser.add_argument(
        "--ltxav-checkpoint",
        default=os.environ.get("PYCOMFY_LTXV2_LTXAV_CHECKPOINT", ""),
        help=(
            "LTXAV companion checkpoint filename in checkpoints/ (paired with --text-encoder). "
            "Default: PYCOMFY_LTXV2_LTXAV_CHECKPOINT."
        ),
    )
    parser.add_argument(
        "--llm",
        default=os.environ.get("PYCOMFY_LLM_MODEL", ""),
        help=(
            "[Optional] LLM filename in llm/ used for LTX2 prompt enhancement via "
            "generate_ltx2_prompt(). Default: PYCOMFY_LLM_MODEL."
        ),
    )
    parser.add_argument(
        "--latent-upscale-model",
        default=os.environ.get("PYCOMFY_LTXV2_LATENT_UPSCALE_MODEL", ""),
        help=(
            "[Optional] Latent upscale model filename in upscale/ for 2x video upsampling. "
            "Default: PYCOMFY_LTXV2_LATENT_UPSCALE_MODEL."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="a dog runs through a sunlit meadow, flowers, wind in fur, cinematic",
        help="Positive prompt describing the desired video and audio content.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative prompt.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Video width in pixels (must be divisible by 32; default 768).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height in pixels (must be divisible by 32; default 512).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=97,
        help="Number of video frames (default 97; must satisfy (length-1)%%8==0).",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Video frame rate in fps injected into LTXV conditioning (default 25).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Diffusion sampling steps (default 30).",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale (default 3.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42).",
    )
    parser.add_argument(
        "--sampler",
        default="euler",
        help="Sampler name (e.g. euler, dpm_2, ddim; default euler).",
    )
    parser.add_argument(
        "--scheduler",
        default="normal",
        help="Scheduler name (e.g. normal, simple, karras; default normal).",
    )
    parser.add_argument(
        "--output-dir",
        default="ltxv2_output",
        help=(
            "Output directory for video frames (PNG) and audio (WAV). "
            "Default: ltxv2_output."
        ),
    )
    parser.add_argument(
        "--reserve-vram",
        type=float,
        default=None,
        metavar="GB",
        help="Amount of VRAM in GB to reserve for other applications (e.g. 1.0).",
    )
    args = parser.parse_args()

    # --- Validate required arguments ---
    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1
    if not args.unet.strip():
        print("error: --unet (or PYCOMFY_LTXV2_UNET) is required", file=sys.stderr)
        return 1
    if not args.vae.strip():
        print("error: --vae (or PYCOMFY_LTXV2_VAE) is required", file=sys.stderr)
        return 1
    if not args.audio_vae.strip():
        print("error: --audio-vae (or PYCOMFY_LTXV2_AUDIO_VAE) is required", file=sys.stderr)
        return 1
    if not args.text_encoder.strip():
        print(
            "error: --text-encoder (or PYCOMFY_LTXV2_TEXT_ENCODER) is required",
            file=sys.stderr,
        )
        return 1
    if not args.ltxav_checkpoint.strip():
        print(
            "error: --ltxav-checkpoint (or PYCOMFY_LTXV2_LTXAV_CHECKPOINT) is required",
            file=sys.stderr,
        )
        return 1

    # --- Lazy imports: all comfy_diffusion symbols are imported after check_runtime() ---
    from comfy_diffusion import check_runtime
    from comfy_diffusion.audio import (
        ltxv_audio_vae_decode,
        ltxv_concat_av_latent,
        ltxv_empty_latent_audio,
        ltxv_separate_av_latent,
    )
    from comfy_diffusion.conditioning import encode_prompt, ltxv_conditioning, ltxv_crop_guides
    from comfy_diffusion.latent import ltxv_empty_latent_video
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode_batch_tiled

    # 1) Verify ComfyUI runtime is present and functional
    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1
    print("runtime:", runtime.get("comfyui_version", "?"), runtime.get("device", "?"))

    if args.reserve_vram is not None:
        from comfy_diffusion._runtime import ensure_comfyui_on_path
        ensure_comfyui_on_path()
        import comfy.model_management
        comfy.model_management.EXTRA_RESERVED_VRAM = args.reserve_vram * 1024 * 1024 * 1024
        print(f"reserved {args.reserve_vram} GB VRAM")

    mm = ModelManager(args.models_dir)

    # 2) Load LTX-Video 2 diffusion UNet
    model = mm.load_unet(args.unet.strip())
    print("loaded unet:", args.unet)

    # 3) Load video VAE (used to decode video latent to frames)
    vae = mm.load_vae(args.vae.strip())
    print("loaded video vae:", args.vae)

    # 4) Load audio VAE (used to create and decode audio latents)
    audio_vae = mm.load_ltxv_audio_vae(args.audio_vae.strip())
    print("loaded audio vae:", args.audio_vae)

    # 5) Load LTXAV text encoder (requires both text encoder weights and companion checkpoint)
    clip = mm.load_ltxav_text_encoder(
        text_encoder_path=args.text_encoder.strip(),
        checkpoint_path=args.ltxav_checkpoint.strip(),
    )
    print("loaded ltxav text encoder:", args.text_encoder, "+", args.ltxav_checkpoint)

    # 6) [Optional] Enhance prompt with a local LLM using generate_ltx2_prompt()
    prompt = args.prompt
    if args.llm.strip():
        from comfy_diffusion.textgen import generate_ltx2_prompt

        llm_clip = mm.load_llm(args.llm.strip())
        print("loaded llm:", args.llm)
        prompt = generate_ltx2_prompt(clip=llm_clip, prompt=prompt)
        preview = prompt[:120] + ("..." if len(prompt) > 120 else "")
        print("enhanced prompt:", preview)

    # 7) [Optional] Load latent upscale model for 2x video latent upsampling after sampling
    latent_upscale_model = None
    if args.latent_upscale_model.strip():
        latent_upscale_model = mm.load_latent_upscale_model(args.latent_upscale_model.strip())
        print("loaded latent upscale model:", args.latent_upscale_model)

    # 8) Encode positive and negative text conditioning
    positive = encode_prompt(clip, prompt)
    negative = encode_prompt(clip, args.negative_prompt)

    # 9) Inject frame-rate metadata required by the LTXV architecture
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=args.frame_rate)

    # 10) Create empty video latent (batch=1; shape depends on width/height/length)
    video_latent = ltxv_empty_latent_video(
        width=args.width,
        height=args.height,
        length=args.length,
        batch_size=1,
    )
    print(
        f"video latent shape: {list(video_latent['samples'].shape)} "
        f"({args.width}x{args.height}, {args.length} frames)"
    )

    # 11) Crop keyframe guide frames from conditioning + latent (no-op when no guides are set)
    positive, negative, video_latent = ltxv_crop_guides(positive, negative, video_latent)

    # 12) Create empty audio latent (frame count and rate must match the video latent)
    audio_latent = ltxv_empty_latent_audio(
        audio_vae=audio_vae,
        frames_number=args.length,
        frame_rate=int(args.frame_rate),
        batch_size=1,
    )
    print(f"audio latent shape: {list(audio_latent['samples'].shape)}")

    # 13) Concatenate video and audio latents into a single joint AV NestedTensor latent
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    # 14) Run joint audio+video denoising in a single sampling pass
    denoised_av = sample(
        model,
        positive,
        negative,
        av_latent,
        steps=args.steps,
        cfg=args.cfg,
        sampler_name=args.sampler,
        scheduler=args.scheduler,
        seed=args.seed,
        denoise=1.0,
    )

    # 15) Separate the denoised joint latent back into individual video and audio tensors
    video_denoised, audio_denoised = ltxv_separate_av_latent(denoised_av)

    # 16) [Optional] Upsample the video latent 2x in latent space before decoding
    if latent_upscale_model is not None:
        from comfy_diffusion.latent import ltxv_latent_upsample

        video_denoised = ltxv_latent_upsample(
            samples=video_denoised,
            upscale_model=latent_upscale_model,
            vae=vae,
        )
        print("applied latent upscale (2x spatial)")

    # 17) Decode video latent to a list of PIL frames (tiled to manage VRAM)
    video_frames = vae_decode_batch_tiled(vae, video_denoised)
    print(f"decoded {len(video_frames)} video frames")

    # 18) Decode audio latent to a waveform tensor
    audio_out = ltxv_audio_vae_decode(audio_vae, audio_denoised)
    waveform = audio_out["waveform"]
    audio_sample_rate = audio_out["sample_rate"]

    # 19) Save video frames to output directory and audio to WAV
    frame_paths = _save_video_frames(video_frames, args.output_dir)
    print(f"saved {len(frame_paths)} video frames to {args.output_dir}/")

    audio_path = str(Path(args.output_dir) / "audio.wav")
    audio_saved = False
    try:
        _save_audio(waveform, audio_sample_rate, audio_path)
        print(f"saved audio to {audio_path}")
        audio_saved = True
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not save audio: {exc}", file=sys.stderr)

    video_path = str(Path(args.output_dir) / "output.mp4")
    try:
        if audio_saved:
            _combine_video(args.output_dir, audio_path, video_path, args.frame_rate)
        else:
            import subprocess
            frame_pattern = str(Path(args.output_dir) / "frame_%05d.png")
            result = subprocess.run(
                ["ffmpeg", "-y", "-framerate", str(args.frame_rate), "-i", frame_pattern,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
        print(f"saved video to {video_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not combine video: {exc}", file=sys.stderr)
    finally:
        for fp in frame_paths:
            Path(fp).unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
