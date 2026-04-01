#!/usr/bin/env python3
"""Flux.2 Klein 9B distilled image-edit example using comfy_diffusion.

Demonstrates usage of the
``comfy_diffusion.pipelines.image.flux_klein.edit_9b_distilled`` pipeline:
download models once, then call ``run()`` to edit an image with a text prompt.

The 9B distilled variant uses the larger Qwen 3 8B text encoder, ``cfg=1``
with ``conditioning_zero_out`` for the negative, and requires only 4 steps.

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models (idempotent — skips files already present)
    uv run python examples/flux_klein_edit_9b_distilled.py --download-only

    # Edit an image
    uv run python examples/flux_klein_edit_9b_distilled.py \\
        --prompt "apply the yellow logo to the steering wheel hub" \\
        --image car_interior.jpg

    # Full options
    uv run python examples/flux_klein_edit_9b_distilled.py \\
        --models-dir /path/to/models \\
        --prompt "apply the yellow logo to the steering wheel hub" \\
        --image car_interior.jpg \\
        --width 1024 --height 1024 \\
        --steps 4 --seed 42 \\
        --output edited_output.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flux.2 Klein 9B distilled image-edit example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory.  Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--prompt",
        default="change the clothing style while keeping the person's pose and expression",
        help="Edit instruction prompt.",
    )
    parser.add_argument(
        "--image",
        required=False,
        default=None,
        help="Path to the reference image to edit.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=1024, help="Output height in pixels.")
    parser.add_argument("--steps", type=int, default=4, help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default="flux_klein_edit_9b_distilled_output.png",
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
    from comfy_diffusion.pipelines.image.flux_klein.edit_9b_distilled import manifest, run

    print("Checking / downloading models …")
    download_models(manifest(), models_dir=models_dir)
    print("Models ready.")

    if args.download_only:
        print("--download-only: exiting without inference.")
        return 0

    if not args.image:
        print("error: --image is required for inference", file=sys.stderr)
        return 1

    from PIL import Image

    from comfy_diffusion.runtime import check_runtime

    runtime = check_runtime()
    if runtime.get("error"):
        print(f"error: runtime check failed: {runtime['error']}", file=sys.stderr)
        return 1

    ref_image = Image.open(args.image).convert("RGB")

    print(f"Editing image ({args.width}x{args.height}, {args.steps} steps) …")
    images = run(
        models_dir=models_dir,
        prompt=args.prompt,
        image=ref_image,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(str(output_path))
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
