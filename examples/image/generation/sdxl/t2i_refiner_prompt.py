#!/usr/bin/env python3
"""SDXL base + refiner text-to-image example with separate prompts per stage.

Demonstrates usage of the
``comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt`` pipeline: download
models once, then call ``run()`` to generate an image using separate prompts
for the base and refiner stages.

The pipeline uses the SDXL base and refiner checkpoints in a two-pass
``KSamplerAdvanced`` flow.  ``--prompt`` drives the base stage; the optional
``--refiner-prompt`` drives the refiner stage (defaults to ``--prompt`` when
omitted).  This allows crafting a broad scene description for the base pass and
a detail-focused prompt for the refiner pass.

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models (idempotent — skips files already present)
    uv run python examples/sdxl_refiner_prompt_t2i.py --download-only

    # Generate with separate prompts
    uv run python examples/sdxl_refiner_prompt_t2i.py \\
        --prompt "a majestic eagle soaring over mountains, wide shot" \\
        --refiner-prompt "photorealistic, sharp focus, award-winning photography"

    # Full options
    uv run python examples/sdxl_refiner_prompt_t2i.py \\
        --models-dir /path/to/models \\
        --prompt "a majestic eagle soaring over mountains, wide shot" \\
        --negative-prompt "blurry, low quality" \\
        --refiner-prompt "photorealistic, sharp focus, award-winning photography" \\
        --width 1024 --height 1024 \\
        --steps 25 --cfg 7.5 --seed 42 \\
        --output sdxl_refiner_prompt_output.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "SDXL base + refiner text-to-image example with separate prompts per stage"
            " (comfy_diffusion)."
        ),
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory.  Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--prompt",
        default="a majestic eagle soaring over snow-capped mountains, wide shot",
        help="Positive text prompt for the base stage.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, watermark",
        help="Negative text prompt (shared by both stages unless overridden).",
    )
    parser.add_argument(
        "--refiner-prompt",
        default=None,
        help=(
            "Positive text prompt for the refiner stage.  "
            "Defaults to --prompt when not provided."
        ),
    )
    parser.add_argument("--width", type=int, default=1024, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=1024, help="Output height in pixels.")
    parser.add_argument("--steps", type=int, default=25, help="Total denoising steps.")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default="sdxl_refiner_prompt_output.png",
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
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest, run

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

    print(f"Generating image ({args.width}x{args.height}) …")
    images = run(
        models_dir=models_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        refiner_prompt=args.refiner_prompt,
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
