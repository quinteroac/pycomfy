#!/usr/bin/env python3
"""
ControlNet txt2img example using comfy-diffusion.

This runs: runtime check → load checkpoint → encode prompts → load ControlNet →
load image → preprocess to control hint (or load hint directly) → apply ControlNet →
sample → VAE decode.

Setup (from repo root):
  1. ComfyUI submodule:
       git submodule update --init
  2. Python deps:
       uv sync --extra comfyui

Usage:
  export PYCOMFY_MODELS_DIR=/path/to/models
  export PYCOMFY_CHECKPOINT=your_checkpoint.safetensors
  uv run python examples/controlnet_txt2img_example.py \
    --controlnet path/to/controlnet.safetensors \
    --image path/to/input.png \
    --preprocess canny \
    --output out.png

Notes:
  - You can either provide a ready-made hint map via --control-image, OR provide a
    regular RGB image via --image and pick a preprocess (default: canny).
  - The control image/hint is converted to a tensor of shape (B,H,W,C), float in [0,1].
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image


def _empty_latent(width: int, height: int, batch_size: int = 1) -> dict:
    """Build an empty LATENT dict for txt2img (ComfyUI contract)."""
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import torch
    import comfy.model_management

    device = comfy.model_management.intermediate_device()
    latent = torch.zeros(
        [batch_size, 4, height // 8, width // 8],
        device=device,
    )
    return {"samples": latent, "downscale_ratio_spacial": 8}


def _resolve_file(
    value: str,
    *,
    models_dir: str | None,
    fallback_subdir: str,
) -> Path | None:
    """Resolve a user-provided file path with an optional models-dir fallback."""
    v = value.strip()
    if not v:
        return None

    p = Path(v)
    if p.is_absolute() and p.is_file():
        return p
    if p.is_file():
        return p.resolve()

    if models_dir:
        fallback = Path(models_dir) / fallback_subdir / p.name
        if fallback.is_file():
            return fallback.resolve()

    return None


def _load_control_image_tensor(
    control_image_path: Path,
    *,
    width: int,
    height: int,
) -> object:
    """Load an RGB control image and return a (1,H,W,3) float tensor on comfy device."""
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import numpy as np
    import torch
    import comfy.model_management

    pil_img = Image.open(control_image_path).convert("RGB")
    if pil_img.size != (width, height):
        pil_img = pil_img.resize((width, height), resample=Image.BICUBIC)

    arr = np.array(pil_img)
    device = comfy.model_management.intermediate_device()
    return (torch.from_numpy(arr).float().to(device=device) / 255.0).unsqueeze(0)


def _preprocess_hint_tensor(
    image_path: Path,
    *,
    preprocess: str,
    width: int,
    height: int,
) -> object:
    """Load an image and turn it into a ControlNet hint tensor.

    Returns a torch tensor with shape (1, H, W, 3), float in [0, 1].
    """
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import numpy as np
    import torch
    import comfy.model_management

    pil_img = Image.open(image_path).convert("RGB")
    if pil_img.size != (width, height):
        pil_img = pil_img.resize((width, height), resample=Image.BICUBIC)

    arr_rgb = np.array(pil_img)

    if preprocess == "none":
        hint = arr_rgb
    elif preprocess == "canny":
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "canny preprocess requires opencv-python. "
                "Install it (for example via comfy-diffusion[video]) or pass --preprocess none / --control-image."
            ) from exc

        gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        hint = np.stack([edges, edges, edges], axis=-1)
    else:
        raise ValueError(f"unsupported preprocess {preprocess!r}")

    device = comfy.model_management.intermediate_device()
    return (torch.from_numpy(hint).float().to(device=device) / 255.0).unsqueeze(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="ControlNet txt2img example.")
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root (must contain checkpoints/). Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PYCOMFY_CHECKPOINT", ""),
        help="Checkpoint filename in checkpoints/. Default: PYCOMFY_CHECKPOINT.",
    )
    parser.add_argument(
        "--controlnet",
        default=os.environ.get("PYCOMFY_CONTROLNET", ""),
        help=(
            "ControlNet checkpoint path (.safetensors). If relative and not found, "
            "tries <models-dir>/controlnet/<value>. Default: PYCOMFY_CONTROLNET."
        ),
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("PYCOMFY_IMAGE", ""),
        help=(
            "Input image to preprocess into a ControlNet hint. Default: PYCOMFY_IMAGE. "
            "Ignored if --control-image is set."
        ),
    )
    parser.add_argument(
        "--preprocess",
        default=os.environ.get("PYCOMFY_PREPROCESS", "canny"),
        choices=["canny", "none"],
        help="Preprocess to generate hint from --image (default: canny).",
    )
    parser.add_argument(
        "--control-image",
        default=os.environ.get("PYCOMFY_CONTROL_IMAGE", ""),
        help=(
            "Optional: already-preprocessed control hint image path (e.g. a canny/depth/pose map). "
            "If set, skips preprocessing. Default: PYCOMFY_CONTROL_IMAGE."
        ),
    )
    parser.add_argument(
        "--union-type",
        default=os.environ.get("PYCOMFY_UNION_CONTROLNET_TYPE", ""),
        help=(
            "Optional union ControlNet type (e.g. auto, openpose, depth, canny/lineart/anime_lineart/mlsd). "
            "Only needed for union ControlNet models. Default: PYCOMFY_UNION_CONTROLNET_TYPE."
        ),
    )
    parser.add_argument(
        "--control-strength",
        type=float,
        default=1.0,
        help="ControlNet strength (default 1.0). Use 0 to disable.",
    )
    parser.add_argument(
        "--control-start",
        type=float,
        default=0.0,
        help="ControlNet start percent (default 0.0).",
    )
    parser.add_argument(
        "--control-end",
        type=float,
        default=1.0,
        help="ControlNet end percent (default 1.0).",
    )
    parser.add_argument(
        "--control-use-vae",
        action="store_true",
        help="Pass VAE into apply_controlnet (needed for some pixel-space controlnets).",
    )
    parser.add_argument(
        "--prompt",
        default="a cinematic photo, ultra detailed, 35mm, soft light",
        help="Positive prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, distorted",
        help="Negative prompt.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Image width (multiple of 8).")
    parser.add_argument("--height", type=int, default=1024, help="Image height (multiple of 8).")
    parser.add_argument("--steps", type=int, default=20, help="Sampling steps.")
    parser.add_argument("--cfg", type=float, default=7.0, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--sampler", default="euler", help="Sampler name.")
    parser.add_argument("--scheduler", default="normal", help="Scheduler name.")
    parser.add_argument("--output", default="output.png", help="Output image path.")
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1
    if not args.checkpoint.strip():
        print("error: --checkpoint (or PYCOMFY_CHECKPOINT) is required", file=sys.stderr)
        return 1

    controlnet_path = _resolve_file(args.controlnet, models_dir=args.models_dir, fallback_subdir="controlnet")
    if controlnet_path is None:
        print(
            "error: --controlnet (or PYCOMFY_CONTROLNET) is required and must exist",
            "(tried cwd and <models-dir>/controlnet/)",
            file=sys.stderr,
        )
        return 1

    control_image_path = (
        _resolve_file(args.control_image, models_dir=None, fallback_subdir="")
        if args.control_image.strip()
        else None
    )
    input_image_path = (
        _resolve_file(args.image, models_dir=None, fallback_subdir="") if args.image.strip() else None
    )
    if control_image_path is None and input_image_path is None:
        print(
            "error: provide --control-image (hint) OR --image (to preprocess into a hint)",
            file=sys.stderr,
        )
        return 1

    from comfy_diffusion import check_runtime, vae_decode_tiled
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.controlnet import (
        apply_controlnet,
        load_diff_controlnet,
        set_union_controlnet_type,
    )
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.sampling import sample

    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1
    print("runtime:", runtime.get("comfyui_version", "?"), runtime.get("device", "?"))

    manager = ModelManager(args.models_dir)
    checkpoint = manager.load_checkpoint(args.checkpoint.strip())
    print("loaded checkpoint:", args.checkpoint)

    if checkpoint.clip is None:
        print("error: checkpoint has no CLIP (cannot encode prompts)", file=sys.stderr)
        return 1
    if checkpoint.vae is None:
        print("error: checkpoint has no VAE (cannot decode latent)", file=sys.stderr)
        return 1

    positive = encode_prompt(checkpoint.clip, args.prompt)
    negative = encode_prompt(checkpoint.clip, args.negative_prompt)

    if control_image_path is not None:
        control_image = _load_control_image_tensor(
            control_image_path,
            width=args.width,
            height=args.height,
        )
    else:
        assert input_image_path is not None
        control_image = _preprocess_hint_tensor(
            input_image_path,
            preprocess=args.preprocess,
            width=args.width,
            height=args.height,
        )

    # Use diff controlnet loader so the ControlNet is paired with the base model.
    control_net = load_diff_controlnet(checkpoint.model, controlnet_path)
    if args.union_type.strip():
        control_net = set_union_controlnet_type(control_net, args.union_type.strip())

    positive, negative = apply_controlnet(
        positive=positive,
        negative=negative,
        control_net=control_net,
        image=control_image,
        strength=args.control_strength,
        start_percent=args.control_start,
        end_percent=args.control_end,
        vae=checkpoint.vae if args.control_use_vae else None,
    )

    latent = _empty_latent(args.width, args.height, batch_size=1)
    denoised = sample(
        checkpoint.model,
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

    image = vae_decode_tiled(checkpoint.vae, denoised)
    image.save(args.output)
    print("saved:", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

