#!/usr/bin/env python3
"""
ACE Step 1.5 text-to-audio example using pycomfy.

Supports two loading modes:

  • Checkpoint: one file with model + VAE + CLIP (--checkpoint). The text encoder (CLIP) comes
    embedded in the checkpoint (e.g. ace_step_1.5_turbo_aio.safetensors); no --text-encoder needed.
  • Separado: model from diffusion_models/ (--unet), VAE from vae/ or checkpoint. Text encoder
    must be loaded separately (--text-encoder + --ckpt-ace-te).

Encodes conditioning with encode_ace_step_15_audio, creates empty latent with
empty_ace_step_15_latent_audio, runs sampling, and decodes to WAV.

  # From full checkpoint (model + VAE + CLIP in one file)
  uv run python examples/ace_step_15_example.py --models-dir /path/to/models \\
    --checkpoint ace_step_1.5_turbo_aio.safetensors --tags "synthwave, retro" --duration 30 --output out.wav

  # From separate components (unet + vae; text encoder required)
  uv run python examples/ace_step_15_example.py --models-dir /path/to/models \\
    --unet ace_step_15_unet.safetensors --vae ace_step_15_vae.safetensors \\
    --text-encoder <name> --ckpt-ace-te <ckpt> --tags "electronic" --duration 30 --output out.wav
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _load_ltxav_text_encoder_two_paths(
    text_encoder_name: str,
    checkpoint_name: str,
) -> Any:
    """Load LTXAV text encoder from two files (text_encoders + checkpoints).

    ACE Step 1.5 requires both a text encoder file and a checkpoint file.
    ModelManager must have been constructed first so folder_paths is configured.
    """
    from pycomfy._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import folder_paths
    from comfy import sd as comfy_sd

    clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder_name)
    clip_path2 = folder_paths.get_full_path_or_raise("checkpoints", checkpoint_name)
    return comfy_sd.load_clip(
        ckpt_paths=[clip_path1, clip_path2],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy_sd.CLIPType.LTXV,
    )


def _load_vae_only_from_checkpoint(checkpoint_name: str) -> Any:
    """Load only the VAE from an ACE checkpoint (model not loaded)."""
    from pycomfy._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import folder_paths
    from comfy import sd as comfy_sd

    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint_name)
    _, _, vae, _ = comfy_sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=False,
        output_clipvision=False,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        output_model=False,
    )
    return vae


def _negative_conditioning_ace(clip: Any, duration: float) -> Any:
    """Return negative conditioning for ACE (empty tags, minimal duration)."""
    from pycomfy.audio import encode_ace_step_15_audio

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ACE Step 1.5 text-to-audio generation (pycomfy).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root (checkpoints/, text_encoders/, etc.). Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PYCOMFY_ACE_CHECKPOINT", ""),
        help="ACE 1.5 checkpoint in checkpoints/ (model + VAE). Required if --unet not set; if --unet set, optional to load VAE only.",
    )
    parser.add_argument(
        "--unet",
        default=os.environ.get("PYCOMFY_ACE_UNET", ""),
        help="ACE diffusion model filename in diffusion_models/ (unet/). Use for separate loading; with --vae or --checkpoint for VAE.",
    )
    parser.add_argument(
        "--vae",
        default=os.environ.get("PYCOMFY_ACE_VAE", ""),
        help="ACE VAE filename in vae/ (standalone). Use with --unet for fully separate loading.",
    )
    parser.add_argument(
        "--text-encoder",
        default=os.environ.get("PYCOMFY_ACE_TEXT_ENCODER", ""),
        help="LTXAV text encoder filename in text_encoders/. Required only when using --unet (separate loading).",
    )
    parser.add_argument(
        "--ckpt-ace-te",
        default=os.environ.get("PYCOMFY_ACE_CKPT_TE", ""),
        help="Checkpoint for LTXAV text encoder (in checkpoints/). Required only when using --unet (separate loading).",
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

    use_separate = bool(args.unet.strip())
    if use_separate:
        if not args.vae.strip() and not args.checkpoint.strip():
            print(
                "error: with --unet you must provide --vae and/or --checkpoint (for VAE)",
                file=sys.stderr,
            )
            return 1
        if not args.text_encoder.strip() or not args.ckpt_ace_te.strip():
            print(
                "error: with --unet you must provide --text-encoder and --ckpt-ace-te for the LTXAV text encoder",
                file=sys.stderr,
            )
            return 1
    else:
        if not args.checkpoint.strip():
            print(
                "error: --checkpoint (or PYCOMFY_ACE_CHECKPOINT) is required when not using --unet",
                file=sys.stderr,
            )
            return 1

    from pycomfy import check_runtime
    from pycomfy.audio import encode_ace_step_15_audio, empty_ace_step_15_latent_audio
    from pycomfy.models import ModelManager
    from pycomfy.sampling import sample

    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1

    mm = ModelManager(args.models_dir)

    # 1) Load ACE 1.5 model, VAE, and optionally CLIP (checkpoint or separate)
    clip = None
    if use_separate:
        model = mm.load_unet(args.unet.strip())
        if args.vae.strip():
            vae = mm.load_vae(args.vae.strip())
        else:
            vae = _load_vae_only_from_checkpoint(args.checkpoint.strip())
        clip = _load_ltxav_text_encoder_two_paths(
            args.text_encoder.strip(),
            args.ckpt_ace_te.strip(),
        )
    else:
        result = mm.load_checkpoint(args.checkpoint.strip())
        model, clip, vae = result.model, result.clip, result.vae
        if clip is None and (args.text_encoder.strip() and args.ckpt_ace_te.strip()):
            clip = _load_ltxav_text_encoder_two_paths(
                args.text_encoder.strip(),
                args.ckpt_ace_te.strip(),
            )
        elif clip is None:
            print(
                "error: checkpoint has no CLIP and no --text-encoder/--ckpt-ace-te provided. Use a full checkpoint (e.g. ace_step_1.5_turbo_aio.safetensors) or pass --text-encoder and --ckpt-ace-te.",
                file=sys.stderr,
            )
            return 1
    if vae is None:
        print("error: no VAE (required for ACE 1.5 decode). Use --checkpoint or --vae.", file=sys.stderr)
        return 1

    # 3) Encode conditioning (positive)
    positive = encode_ace_step_15_audio(
        clip,
        tags=args.tags,
        lyrics=args.lyrics,
        seed=args.seed,
        bpm=args.bpm,
        duration=args.duration,
        timesignature="4",
        language="en",
        keyscale="C major",
        generate_audio_codes=True,
        cfg_scale=args.cfg,
    )
    negative = _negative_conditioning_ace(clip, args.duration)

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

    # 6) Decode latent to audio (VAE from checkpoint; ACE uses MusicDCAE, 44100 Hz)
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
