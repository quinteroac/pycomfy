#!/usr/bin/env python3
"""
Simple checkpoint example with latent upscaling using comfy_diffusion.

Flujo:
  1. check_runtime()
  2. cargar checkpoint base
  3. txt2img → genera un LATENT base
  4. cargar modelo de upscale latente
  5. aplicar upscaler en el espacio latente
  6. VAE decode → guardar imagen upscaled

Requisitos (desde la raíz del repo):
  1. Submódulo ComfyUI:
       git submodule update --init
  2. Dependencias Python (CPU por defecto):
       uv sync --extra comfyui
     GPU (opcional, recomendado para producción):
       uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

Modelo de upscale:
  - Usa un modelo de tipo LATENT_UPSCALE_MODEL compatible con el nodo
    LTXVLatentUpsampler de ComfyUI (por ejemplo, modelos LTXV).
  - Debe estar en <models-dir>/upscale/ o se puede pasar la ruta absoluta con
    --latent-upscale-checkpoint.

Ejemplos de uso:
  export PYCOMFY_MODELS_DIR=/path/to/models
  export PYCOMFY_CHECKPOINT=your_base_checkpoint.safetensors
  # Supone que el modelo de upscale está en <models-dir>/upscale/lt_upscale.safetensors
  export PYCOMFY_LATENT_UPSCALE_CHECKPOINT=lt_upscale.safetensors

  # txt2img base + latent upscale
  uv run python examples/simple_checkpoint_latent_upscale_example.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image


def _empty_latent(width: int, height: int, batch_size: int = 1) -> dict:
    """Crea un LATENT vacío para txt2img (contrato de ComfyUI)."""
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


def _load_latent_upscale_model(models_dir: Path, filename: str) -> object:
    """
    Carga un modelo de tipo LATENT_UPSCALE_MODEL desde <models-dir>/upscale/<filename>
    o desde una ruta absoluta si filename es una ruta existente.
    """
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management

    path = Path(filename)
    if not path.is_absolute():
        path = models_dir / "upscale" / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"latent upscale checkpoint not found: {path} "
            "(expected in <models-dir>/upscale/ or provided as absolute path)"
        )

    # Usa la utilidad de carga estándar de ComfyUI para modelos de upscale latente.
    # En ComfyUI, los modelos LATENT_UPSCALE_MODEL se cargan como módulos torch estándar.
    device = comfy.model_management.get_torch_device()
    latent_upscale_model = comfy.model_management.load_torch_file(str(path), device=device)
    return latent_upscale_model


def _upsample_latent(
    samples: dict,
    upscale_model: object,
    vae: object,
) -> dict:
    """
    Aplica un upscale x2 en espacio latente usando la lógica de LTXVLatentUpsampler.

    Reimplementa la parte interna relevante del nodo LTXVLatentUpsampler como
    función pura, sin usar el sistema de nodos de ComfyUI.
    """
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import math
    from comfy import model_management

    device = model_management.get_torch_device()
    memory_required = model_management.module_size(upscale_model)

    latents = samples["samples"]
    input_dtype = latents.dtype
    model_dtype = next(upscale_model.parameters()).dtype

    memory_required += math.prod(latents.shape) * 3000.0
    model_management.free_memory(memory_required, device)

    try:
        upscale_model.to(device)
        latents = latents.to(dtype=model_dtype, device=device)

        # Normalizar / des-normalizar igual que el nodo original
        latents = vae.first_stage_model.per_channel_statistics.un_normalize(latents)
        upsampled_latents = upscale_model(latents)
    finally:
        upscale_model.cpu()

    upsampled_latents = vae.first_stage_model.per_channel_statistics.normalize(
        upsampled_latents
    )
    upsampled_latents = upsampled_latents.to(
        dtype=input_dtype, device=model_management.intermediate_device()
    )
    result = samples.copy()
    result["samples"] = upsampled_latents
    result.pop("noise_mask", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Simple checkpoint example: txt2img + latent upscale (factor ~2x en resolución)."
        ),
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("PYCOMFY_MODELS_DIR"),
        help="Directorio de modelos (debe contener checkpoints/). Por defecto: PYCOMFY_MODELS_DIR.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PYCOMFY_CHECKPOINT", ""),
        help="Nombre del checkpoint en checkpoints/. Por defecto: PYCOMFY_CHECKPOINT.",
    )
    parser.add_argument(
        "--latent-upscale-checkpoint",
        default=os.environ.get("PYCOMFY_LATENT_UPSCALE_CHECKPOINT", ""),
        help=(
            "Nombre del checkpoint de upscale latente (archivo .safetensors) en "
            "<models-dir>/upscale/ o ruta absoluta. Por defecto: "
            "PYCOMFY_LATENT_UPSCALE_CHECKPOINT."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="a portrait of a woman, studio lighting, detailed",
        help="Prompt positivo.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, distorted",
        help="Prompt negativo.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Ancho inicial de la imagen (múltiplo de 8).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Alto inicial de la imagen (múltiplo de 8).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Número de pasos de muestreo.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed aleatorio.",
    )
    parser.add_argument(
        "--sampler",
        default="euler",
        help="Nombre del sampler (e.g. euler, dpm_2, ddim).",
    )
    parser.add_argument(
        "--scheduler",
        default="normal",
        help="Nombre del scheduler (e.g. normal, simple, karras).",
    )
    parser.add_argument(
        "--output",
        default="output_separate.png",
        help="Ruta de salida para la imagen upscaled.",
    )
    parser.add_argument(
        "--output-base",
        default="output.png",
        help="Ruta de salida para la imagen base (sin upscale).",
    )
    args = parser.parse_args()

    if not args.models_dir or not Path(args.models_dir).is_dir():
        print(
            "error: --models-dir (o PYCOMFY_MODELS_DIR) debe apuntar a un directorio existente",
            file=sys.stderr,
        )
        return 1
    if not args.checkpoint.strip():
        print(
            "error: --checkpoint (o PYCOMFY_CHECKPOINT) es obligatorio",
            file=sys.stderr,
        )
        return 1
    if not args.latent_upscale_checkpoint.strip():
        print(
            "error: --latent-upscale-checkpoint (o PYCOMFY_LATENT_UPSCALE_CHECKPOINT) "
            "es obligatorio para este ejemplo",
            file=sys.stderr,
        )
        return 1

    # 1) Runtime check
    from comfy_diffusion import check_runtime, vae_decode_tiled
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.sampling import sample

    runtime = check_runtime()
    if runtime.get("error"):
        print("error: runtime check failed:", runtime["error"], file=sys.stderr)
        return 1
    print("runtime:", runtime.get("comfyui_version", "?"), runtime.get("device", "?"))

    # 2) Cargar checkpoint base
    manager = ModelManager(args.models_dir)
    checkpoint = manager.load_checkpoint(args.checkpoint.strip())
    print("loaded checkpoint:", args.checkpoint)

    if checkpoint.clip is None:
        print("error: checkpoint no tiene CLIP (no se pueden codificar prompts)", file=sys.stderr)
        return 1
    if checkpoint.vae is None:
        print("error: checkpoint no tiene VAE (no se puede decodificar latent)", file=sys.stderr)
        return 1

    model = checkpoint.model
    clip = checkpoint.clip
    vae = checkpoint.vae

    # 3) Encode prompts
    positive = encode_prompt(clip, args.prompt)
    negative = encode_prompt(clip, args.negative_prompt)

    # 4) Latent base (txt2img)
    latent = _empty_latent(args.width, args.height, batch_size=1)
    print("mode: txt2img (latent base size:", args.width, "x", args.height, ")")

    # 5) Muestreo base
    denoised = sample(
        model,
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

    # 6) Decodificar imagen base
    image_base = vae_decode_tiled(vae, denoised)
    image_base.save(args.output_base)
    print("saved base image:", args.output_base)

    # 7) Cargar modelo de latent upscale
    models_dir = Path(args.models_dir)
    try:
        latent_upscale_model = _load_latent_upscale_model(
            models_dir,
            args.latent_upscale_checkpoint.strip(),
        )
    except Exception as exc:  # noqa: BLE001
        print("error: no se pudo cargar el modelo de latent upscale:", exc, file=sys.stderr)
        return 1
    print("loaded latent upscale model:", args.latent_upscale_checkpoint)

    # 8) Aplicar upscale en el espacio latente (aprox. 2x en resolución)
    upsampled_latent = _upsample_latent(denoised, latent_upscale_model, vae)

    # 9) Decodificar imagen upscaled
    image_upscaled = vae_decode_tiled(vae, upsampled_latent)
    image_upscaled.save(args.output)
    print("saved upscaled image:", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())

