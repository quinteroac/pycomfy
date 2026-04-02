#!/usr/bin/env python3
"""SDXL Turbo text-to-image example using comfy_diffusion.

Demonstrates usage of the
``comfy_diffusion.pipelines.image.sdxl.turbo`` pipeline: download models once,
then call ``run()`` to generate an image from a text prompt in one or a few
steps.

The pipeline uses the distilled SDXL Turbo checkpoint with ``SDTurboScheduler``
and the ``euler_ancestral`` sampler.  A single denoising step (``--steps 1``)
is enough for reasonable quality; CFG defaults to 0.0 (guidance-free) as
recommended for distilled models.

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models (idempotent — skips files already present)
    uv run python examples/sdxl_turbo.py --download-only

    # Generate an image
    uv run python examples/sdxl_turbo.py --prompt "a golden retriever puppy on a summer beach"

    # Full options
    uv run python examples/sdxl_turbo.py \\
        --models-dir /path/to/models \\
        --prompt "a golden retriever puppy on a summer beach" \\
        --width 512 --height 512 \\
        --steps 1 --cfg 0.0 --seed 42 \\
        --output sdxl_turbo_output.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SDXL Turbo text-to-image example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory.  Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--prompt",
        default="a golden retriever puppy on a summer beach, vibrant colors",
        help="Positive text prompt.",
    )
    parser.add_argument("--width", type=int, default=512, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=512, help="Output height in pixels.")
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of denoising steps (1–10).  Default 1.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=0.0,
        help="CFG scale.  Default 0.0 (guidance-free, recommended for SDXL Turbo).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", default="sdxl_turbo_output.png", help="Output image path.")
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
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest, run

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

    print(f"Generating image ({args.width}x{args.height}, {args.steps} step(s)) …")
    images = run(
        models_dir=models_dir,
        prompt=args.prompt,
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
