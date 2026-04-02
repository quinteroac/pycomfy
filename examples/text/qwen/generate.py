#!/usr/bin/env python3
"""
Minimal LLM/VLM text generation example using comfy_diffusion.textgen.generate_text.

This script loads a text model from models_dir/llm and runs text generation through
`generate_text()` (or `generate_ltx2_prompt()` when requested).

Setup (from repo root):
  uv sync --extra comfyui --extra cpu

Usage:
  export PYCOMFY_MODELS_DIR=/path/to/models
  export PYCOMFY_LLM_MODEL=Qwen2.5-3B-Instruct-Q4_K_M.gguf

  # Basic text generation
  uv run python examples/textgen_generate_text_example.py

  # Deterministic generation (no sampling)
  uv run python examples/textgen_generate_text_example.py --do-sample off

  # VLM mode (optional image)
  uv run python examples/textgen_generate_text_example.py --image input.png

  # LTX2 prompt-enhancer wrapper
  uv run python examples/textgen_generate_text_example.py --ltx2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from comfy_diffusion import check_runtime
from comfy_diffusion.models import ModelManager
from comfy_diffusion.textgen import generate_ltx2_prompt, generate_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with ComfyUI-compatible LLM/VLM via comfy_diffusion.textgen.",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Models root containing llm/. Default: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--llm",
        default=os.environ.get("PYCOMFY_LLM_MODEL", ""),
        help="LLM/VLM filename under llm/ (or absolute path). Default: PYCOMFY_LLM_MODEL.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a concise one-paragraph description of a futuristic city at sunset.",
        help="Input prompt.",
    )
    parser.add_argument(
        "--image",
        default="",
        help="Optional image path for VLM-capable models.",
    )
    parser.add_argument(
        "--ltx2",
        action="store_true",
        help="Use generate_ltx2_prompt() instead of generate_text().",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum generated length.",
    )
    parser.add_argument(
        "--do-sample",
        choices=["on", "off"],
        default="on",
        help="Sampling mode: on/off.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _resolve_optional_image(path_value: str) -> Any | None:
    if not path_value.strip():
        return None

    p = Path(path_value.strip())
    if not p.is_file():
        raise FileNotFoundError(f"image file not found: {path_value}")

    return Image.open(p).convert("RGB")


def main() -> int:
    args = _parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (or PYCOMFY_MODELS_DIR) must point to an existing directory",
            file=sys.stderr,
        )
        return 1

    if not args.llm.strip():
        print("error: --llm (or PYCOMFY_LLM_MODEL) is required", file=sys.stderr)
        return 1

    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1

    print("runtime:", runtime.get("comfyui_version", "?"), runtime.get("device", "?"))

    manager = ModelManager(args.models_dir)
    clip = manager.load_llm(args.llm.strip())
    print("loaded llm:", args.llm)

    try:
        image = _resolve_optional_image(args.image)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    do_sample = args.do_sample == "on"

    if args.ltx2:
        generated = generate_ltx2_prompt(
            clip=clip,
            prompt=args.prompt,
            image=image,
            max_length=args.max_length,
            do_sample=do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
    else:
        generated = generate_text(
            clip=clip,
            prompt=args.prompt,
            image=image,
            max_length=args.max_length,
            do_sample=do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )

    print("\n--- generated text ---\n")
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
