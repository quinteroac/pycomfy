#!/usr/bin/env python3
"""Anima Preview text-to-image example using comfy_diffusion.

Demonstrates usage of the
``comfy_diffusion.pipelines.image.anima.t2i`` pipeline: download models once,
then call ``run()`` to generate an anime-style image from a text prompt.

The pipeline uses the Anima Preview diffusion model with the Qwen 3 0.6B text
encoder, the ``er_sde`` sampler, and ``normal`` scheduler.  It produces
high-quality anime-style images at 1024×1024 resolution in 30 steps.

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models (idempotent — skips files already present)
    uv run python examples/anima_preview.py --download-only

    # Generate an image
    uv run python examples/anima_preview.py --prompt "1girl, anime style, masterpiece, cherry blossoms"

    # Full options
    uv run python examples/anima_preview.py \\
        --models-dir /path/to/models \\
        --prompt "1girl, anime style, masterpiece, cherry blossoms" \\
        --negative-prompt "lowres, bad anatomy, watermark" \\
        --width 1024 --height 1024 \\
        --steps 30 --cfg 4.0 --seed 42 \\
        --output anima_preview_output.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anima Preview text-to-image example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory.  Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--prompt",
        default="1girl, anime style, masterpiece, cherry blossoms, detailed eyes",
        help="Positive text prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="lowres, bad anatomy, bad hands, watermark, signature",
        help="Negative text prompt.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=1024, help="Output height in pixels.")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps.")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default="anima_preview_output.png",
        help="Output image path.",
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

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.anima.t2i import manifest, run

    print("Checking / downloading models …")
    download_models(manifest(), models_dir=models_dir)
    print("Models ready.")

    if args.download_only:
        print("--download-only: exiting without inference.")
        return 0

    from comfy_diffusion.runtime import check_runtime

    runtime = check_runtime()
    if runtime.get("error"):
        print(f"error: runtime check failed: {runtime['error']}", file=sys.stderr)
        return 1

    print(f"Generating image ({args.width}x{args.height}, {args.steps} steps) …")
    images = run(
        models_dir=models_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(str(output_path))
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
