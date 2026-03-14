#!/usr/bin/env python3
"""
ACE Step 1.5 text-to-audio example using comfy_diffusion.

Uses split components:
    - UNet from diffusion_models/
    - VAE from vae/
    - 2 text encoders from text_encoders/ (ACE Step 1.5 uses DualCLIP)

Encodes conditioning with encode_ace_step_15_audio, creates empty latent with
empty_ace_step_15_latent_audio, runs sampling, and decodes to WAV.

  uv run python examples/ace_step_15_example.py --models-dir /path/to/models \\
    --unet ace_step_15_unet.safetensors --vae ace_step_15_vae.safetensors \\
        --text-encoder-1 qwen_0.6b_ace15.safetensors --text-encoder-2 qwen_4b_ace15.safetensors \\
        --tags "electronic" --duration 30 --output out.wav
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _negative_conditioning_ace(clip: Any, duration: float) -> Any:
    """Return negative conditioning for ACE (empty tags, minimal duration)."""
    from comfy_diffusion.audio import encode_ace_step_15_audio

    return encode_ace_step_15_audio(
        clip,
        tags="",
        lyrics="",
        seed=0,
        bpm=120,
        duration=min(1.0, duration),
        timesignature="4",
        language="en",
        keyscale="C major",
        generate_audio_codes=False,
        cfg_scale=2.0,
    )


def _ace_step_15_conditioning(
    clip: Any,
    tags: str,
    lyrics: str,
    seed: int,
    bpm: int,
    duration: float,
    cfg: float,
) -> tuple[Any, Any]:
    """Build positive/negative ACE Step 1.5 conditioning pair."""
    from comfy_diffusion.audio import encode_ace_step_15_audio

    positive = encode_ace_step_15_audio(
        clip,
        tags=tags,
        lyrics=lyrics,
        seed=seed,
        bpm=bpm,
        duration=duration,
        timesignature="4",
        language="en",
        keyscale="C major",
        generate_audio_codes=True,
        cfg_scale=cfg,
    )
    negative = _negative_conditioning_ace(clip, duration)
    return positive, negative


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ACE Step 1.5 text-to-audio generation (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root (checkpoints/, text_encoders/, etc.). Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--unet",
        default=os.environ.get("PYCOMFY_ACE_UNET", ""),
        help="ACE diffusion model filename in diffusion_models/ (or unet/).",
    )
    parser.add_argument(
        "--vae",
        default=os.environ.get("PYCOMFY_ACE_VAE", ""),
        help="ACE VAE filename in vae/.",
    )
    parser.add_argument(
        "--text-encoder-1",
        default=os.environ.get("PYCOMFY_ACE_TEXT_ENCODER_1", ""),
        help="First ACE Step 1.5 text encoder filename in text_encoders/ (e.g. qwen_0.6b_ace15.safetensors).",
    )
    parser.add_argument(
        "--text-encoder-2",
        default=os.environ.get("PYCOMFY_ACE_TEXT_ENCODER_2", ""),
        help="Second ACE Step 1.5 text encoder filename in text_encoders/ (e.g. qwen_4b_ace15.safetensors).",
    )
    parser.add_argument(
        "--tags",
        default="electronic, ambient, synth",
        help="Genre/style tags for the generated audio.",
    )
    parser.add_argument(
        "--lyrics",
        default="",
        help="Optional lyrics (empty for instrumental).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds (default 30).",
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=120,
        help="BPM (default 120).",
    )
    parser.add_argument(
        "--output",
        default="ace_output.wav",
        help="Output WAV path.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Sampling steps.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.0,
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
        "--trim-end",
        type=float,
        default=5.0,
        help="Seconds to trim from the end of the output (ACE often has trailing silence). Set 0 to disable (default 5.0).",
    )
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1

    if not args.unet.strip():
        print("error: --unet (or PYCOMFY_ACE_UNET) is required", file=sys.stderr)
        return 1
    if not args.vae.strip():
        print("error: --vae (or PYCOMFY_ACE_VAE) is required", file=sys.stderr)
        return 1
    if not args.text_encoder_1.strip() or not args.text_encoder_2.strip():
        print(
            "error: --text-encoder-1 and --text-encoder-2 are required for ACE Step 1.5",
            file=sys.stderr,
        )
        return 1

    from comfy_diffusion import check_runtime
    from comfy_diffusion.audio import empty_ace_step_15_latent_audio
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.sampling import sample

    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1

    mm = ModelManager(args.models_dir)

    # 1) Load ACE 1.5 split components (UNet + VAE + DualCLIP text encoders)
    model = mm.load_unet(args.unet.strip())
    vae = mm.load_vae(args.vae.strip())
    clip = mm.load_clip(
        args.text_encoder_1.strip(),
        args.text_encoder_2.strip(),
        clip_type="ace",
    )
    if vae is None:
        print("error: no VAE (required for ACE 1.5 decode).", file=sys.stderr)
        return 1

    # 3) ACE Step 1.5 conditioning (positive + negative)
    positive, negative = _ace_step_15_conditioning(
        clip=clip,
        tags=args.tags,
        lyrics=args.lyrics,
        seed=args.seed,
        bpm=args.bpm,
        duration=args.duration,
        cfg=args.cfg,
    )

    # 4) Empty ACE Step 1.5 latent
    latent = empty_ace_step_15_latent_audio(seconds=args.duration, batch_size=1)

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

    # 6) Decode latent to audio (ACE uses MusicDCAE, 44100 Hz)
    samples_tensor = denoised["samples"]
    waveform = vae.decode(samples_tensor)
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    sample_rate = 44100
    if waveform.dim() == 3:
        waveform = waveform[0]

    # Trim trailing silence (ACE often generates ~5 s at the end). Never trim so much that we leave < 1 s.
    if args.trim_end > 0:
        total_samples = waveform.shape[-1]
        min_keep_samples = int(sample_rate * 1.0)
        max_trim = max(0, total_samples - min_keep_samples)
        trim_samples = min(int(sample_rate * args.trim_end), max_trim)
        if trim_samples > 0:
            waveform = waveform[..., :-trim_samples].contiguous()

    # 7) Save WAV (torchaudio preferred per project convention)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torchaudio
        torchaudio.save(str(out_path), waveform, sample_rate)
    except ImportError:
        import numpy as np
        wav = waveform.numpy() if hasattr(waveform, "numpy") else np.array(waveform)
        wav = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
        try:
            from scipy.io import wavfile
            wavfile.write(str(out_path), sample_rate, wav)
        except ImportError:
            print("error: install torchaudio or scipy to save WAV", file=sys.stderr)
            return 1

    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
