#!/usr/bin/env python3
"""Qwen Image Layered text-to-layers example using comfy_diffusion.

Demonstrates usage of the
``comfy_diffusion.pipelines.image.qwen.layered`` pipeline: download
models once, then call ``run_t2l()`` to generate layered images from a
text prompt.

The pipeline uses the Qwen Image Layered diffusion model with the Qwen 2.5
VL 7B FP8 text encoder, the ``euler`` sampler with ``simple`` scheduler, and
CFG guidance of 2.5.  Each call returns ``layers`` PIL images corresponding
to the generated layers.

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models (idempotent — skips files already present)
    uv run python examples/qwen_layered_t2l.py --download-only

    # Generate layered images
    uv run python examples/qwen_layered_t2l.py \\
        --prompt "a beautiful mountain landscape with snow-capped peaks"

    # Full options
    uv run python examples/qwen_layered_t2l.py \\
        --models-dir /path/to/models \\
        --prompt "a photo of a cat on a chair" \\
        --width 640 --height 640 \\
        --layers 2 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen Image Layered text-to-layers example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory.  Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "A cinematic medium shot of a beautiful young woman with fair skin "
            "and a joyful smile, standing on a rugged coastline overlooking the ocean."
        ),
        help="Positive text prompt.",
    )
    parser.add_argument("--width", type=int, default=640, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=640, help="Output height in pixels.")
    parser.add_argument("--layers", type=int, default=2, help="Number of output layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-prefix",
        default="qwen_layered_t2l",
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
    from comfy_diffusion.pipelines.image.qwen.layered import manifest, run_t2l

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

    print(
        f"Generating {args.layers} layers "
        f"({args.width}x{args.height}, seed={args.seed}) …"
    )
    images = run_t2l(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
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
