#!/usr/bin/env python3
"""Qwen Image Edit 2511 example using comfy_diffusion.

Demonstrates usage of the
``comfy_diffusion.pipelines.image.qwen.edit_2511`` pipeline: download
models once, then call ``run()`` to edit an image with a text instruction.

The pipeline uses the Qwen Image Edit 2511 diffusion model with the Qwen 2.5
VL 7B fp8 text encoder, the ``euler`` sampler with ``simple`` scheduler, and
an optional Lightning LoRA for 4-step turbo inference.

Usage
-----
::

    # Set models directory via environment variable (or pass via --models-dir)
    export PYCOMFY_MODELS_DIR=/path/to/models

    # Download models (idempotent — skips files already present)
    uv run python examples/image/edit/qwen/edit_2511.py --download-only

    # Edit an image
    uv run python examples/image/edit/qwen/edit_2511.py \\
        --image input.png \\
        --prompt "Make the sofa look like it is covered in fur"

    # Full options
    uv run python examples/image/edit/qwen/edit_2511.py \\
        --models-dir /path/to/models \\
        --image input.png \\
        --prompt "Make the sofa look like it is covered in fur" \\
        --steps 4 --seed 42 --no-lora \\
        --output edited_output.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen Image Edit 2511 example (comfy_diffusion).",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root directory.  Default: PYCOMFY_MODELS_DIR env var.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to the input image to edit.  Required for inference.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Text editing instruction (e.g. 'Make the sofa look like it is covered in fur').",
    )
    parser.add_argument(
        "--image2",
        default=None,
        help="Optional second reference image path.",
    )
    parser.add_argument(
        "--image3",
        default=None,
        help="Optional third reference image path.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of denoising steps (default=4 with LoRA, 40 without).",
    )
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable the Lightning LoRA (use standard 40-step inference instead).",
    )
    parser.add_argument(
        "--output",
        default="qwen_edit_2511_output.png",
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
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest, run

    print("Checking / downloading models …")
    download_models(manifest(), models_dir=models_dir)
    print("Models ready.")

    if args.download_only:
        return 0

    if not args.image:
        print("error: --image is required for inference", file=sys.stderr)
        return 1

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

    image2 = None
    if args.image2:
        image2_path = Path(args.image2)
        if not image2_path.is_file():
            print(f"error: image2 file not found: {image2_path}", file=sys.stderr)
            return 1
        image2 = Image.open(str(image2_path)).convert("RGB")

    image3 = None
    if args.image3:
        image3_path = Path(args.image3)
        if not image3_path.is_file():
            print(f"error: image3 file not found: {image3_path}", file=sys.stderr)
            return 1
        image3 = Image.open(str(image3_path)).convert("RGB")

    use_lora = not args.no_lora
    steps = args.steps if args.steps is not None else (4 if use_lora else 40)
    print(
        f"Editing image ({steps} steps, cfg={args.cfg}, "
        f"lora={'on' if use_lora else 'off'}, seed={args.seed}) …"
    )

    images = run(
        prompt=args.prompt,
        image=input_image,
        image2=image2,
        image3=image3,
        models_dir=models_dir,
        steps=steps,
        cfg=args.cfg,
        use_lora=use_lora,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(str(output_path))
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
