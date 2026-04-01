#!/usr/bin/env python3
"""Qwen Image Layered image-to-layers example using comfy_diffusion.

Demonstrates usage of the
``comfy_diffusion.pipelines.image.qwen.layered`` pipeline: download
models once, then call ``run_i2l()`` to decompose an input image into
multiple generated layers.

The pipeline scales the input image to a maximum of 640 px on the longer
edge, encodes it as a reference latent via the VAE, and uses ``ReferenceLatent``
conditioning to guide generation.  The ``euler`` sampler with ``simple``
scheduler and CFG guidance of 2.5 is used throughout.

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models (idempotent — skips files already present)
    uv run python examples/qwen_layered_i2l.py --download-only

    # Generate layers from an input image
    uv run python examples/qwen_layered_i2l.py \\
        --image input.png \\
        --prompt "a woman standing on a coastline"

    # Full options
    uv run python examples/qwen_layered_i2l.py \\
        --models-dir /path/to/models \\
        --image input.png \\
        --prompt "a woman standing on a coastline" \\
        --layers 2 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen Image Layered image-to-layers example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory.  Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional text prompt describing the image content.",
    )
    parser.add_argument("--layers", type=int, default=2, help="Number of output layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-prefix",
        default="qwen_layered_i2l",
        help="Output image filename prefix (layer index appended).",
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
    from comfy_diffusion.pipelines.image.qwen.layered import manifest, run_i2l

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

    from PIL import Image

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"error: image file not found: {image_path}", file=sys.stderr)
        return 1

    input_image = Image.open(str(image_path)).convert("RGB")
    print(f"Loaded input image: {image_path} ({input_image.width}x{input_image.height})")
    print(f"Generating {args.layers} layers (seed={args.seed}) …")

    images = run_i2l(
        prompt=args.prompt,
        image=input_image,
        layers=args.layers,
        seed=args.seed,
        models_dir=models_dir,
    )

    for i, img in enumerate(images):
        output_path = Path(f"{args.output_prefix}_layer{i}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))
        print(f"Saved layer {i}: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
