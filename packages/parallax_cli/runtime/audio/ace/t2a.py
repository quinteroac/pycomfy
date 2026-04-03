#!/usr/bin/env python3
"""ACE Step 1.5 text-to-audio example using comfy_diffusion.

Uses the high-level ``manifest()`` + ``run()`` API from
``comfy_diffusion.pipelines.audio.ace_step.v1_5.split`` so model downloads
are handled automatically (idempotent, SHA-256 verified).

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models then generate audio
    uv run python examples/audio/ace/t2a.py \\
        --tags "electronic ambient, synth, atmospheric" \\
        --duration 30 \\
        --output out.wav

    # Full options
    uv run python examples/audio/ace/t2a.py \\
        --models-dir /path/to/models \\
        --tags "neo-soul, warm groove, live drums" \\
        --duration 60 \\
        --steps 60 \\
        --cfg 2.0 \\
        --bpm 95 \\
        --seed 42 \\
        --output my_track.wav
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ACE Step 1.5 text-to-audio generation (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory.  Default: PYCOMFY_MODELS_DIR env var.",
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
        "--steps",
        type=int,
        default=60,
        help="Sampling steps (default 60).",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale (default 2.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default 0).",
    )
    parser.add_argument(
        "--sampler",
        default="euler",
        help="Sampler name (default euler).",
    )
    parser.add_argument(
        "--scheduler",
        default="normal",
        help="Scheduler name (default normal).",
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV file path (default output.wav).",
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

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split import manifest, run

    print("Checking / downloading models …")
    download_models(manifest(), models_dir=args.models_dir)
    print("Models ready.")

    if args.download_only:
        print("--download-only: exiting without inference.")
        return 0

    result = run(
        models_dir=args.models_dir,
        tags=args.tags,
        lyrics=args.lyrics,
        duration=args.duration,
        bpm=args.bpm,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        sampler_name=args.sampler,
        scheduler=args.scheduler,
    )

    audio = result["audio"]
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()

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
