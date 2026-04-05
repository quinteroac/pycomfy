"""``parallax upscale`` subcommand group — image super-resolution.

All pipeline imports are deferred to call time (lazy imports).

Supports two models:
- ``esrgan``: ESRGAN super-resolution via ``image_upscale_with_model``.
- ``latent_upscale``: SD latent hi-res fix via encode → upscale → sample → decode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer

from cli._io import resolve_models_dir, save_image

app = typer.Typer(help="Upscale an image using a super-resolution model.")

UPSCALE_IMAGE_MODELS = ["esrgan", "latent_upscale"]


def _tensor_to_pil_list(tensor: object) -> list[object]:
    """Convert a BHWC float32 tensor (B, H, W, C) to a list of PIL images."""
    import numpy as np
    from PIL import Image as _PIL

    arr = getattr(tensor, "cpu", lambda: tensor)().numpy()  # (B, H, W, C)
    return [_PIL.fromarray((arr[i] * 255.0).clip(0, 255).astype(np.uint8)) for i in range(arr.shape[0])]


@app.command("image")
def upscale_image(
    model: Annotated[str, typer.Option(
        "--model", help=f"Model to use. Choices: {', '.join(UPSCALE_IMAGE_MODELS)}.",
    )],
    prompt: Annotated[str, typer.Option("--prompt", help="Text prompt (used for latent_upscale only).")] = "",
    input: Annotated[str, typer.Option("--input", help="Path to the input image file.")] = ...,  # required
    output: Annotated[str, typer.Option("--output", help="Output file path.")] = "output.png",
    output_base: Annotated[str, typer.Option(
        "--output-base", help="Intermediate base image before upscaling (latent_upscale only).",
    )] = "output_base.png",
    checkpoint: Annotated[Optional[str], typer.Option(
        "--checkpoint",
        envvar="PYCOMFY_CHECKPOINT",
        help="Base checkpoint filename (latent_upscale; overrides PYCOMFY_CHECKPOINT).",
    )] = None,
    esrgan_checkpoint: Annotated[Optional[str], typer.Option(
        "--esrgan-checkpoint", help="ESRGAN model filename (required for esrgan).",
    )] = None,
    latent_upscale_checkpoint: Annotated[Optional[str], typer.Option(
        "--latent-upscale-checkpoint",
        help="Upscale checkpoint filename (latent_upscale; required if using upscale model).",
    )] = None,
    negative_prompt: Annotated[str, typer.Option(
        "--negative-prompt", help="Negative prompt (latent_upscale only).",
    )] = "",
    width: Annotated[Optional[int], typer.Option("--width", help="Target width in pixels.")] = None,
    height: Annotated[Optional[int], typer.Option("--height", help="Target height in pixels.")] = None,
    steps: Annotated[int, typer.Option("--steps", help="Sampling steps (latent_upscale).")] = 20,
    cfg: Annotated[float, typer.Option("--cfg", help="CFG scale (latent_upscale).")] = 7.5,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility.")] = 0,
    models_dir: Annotated[Optional[str], typer.Option(
        "--models-dir", help="Models directory (overrides PYCOMFY_MODELS_DIR).",
    )] = None,
    async_mode: Annotated[bool, typer.Option(
        "--async", help="Queue job and return job ID (non-blocking).",
    )] = False,
) -> None:
    """Upscale an image using a super-resolution model."""
    if model not in UPSCALE_IMAGE_MODELS:
        typer.echo(
            f"Error: unknown model '{model}'. Choices: {', '.join(UPSCALE_IMAGE_MODELS)}.",
            err=True,
        )
        raise typer.Exit(code=1)

    if async_mode:
        typer.echo("Error: --async mode requires a job queue (not yet available).", err=True)
        raise typer.Exit(code=1)

    if not Path(input).is_file():
        typer.echo(f"Error: input file not found: {input}", err=True)
        raise typer.Exit(code=1)

    mdir = resolve_models_dir(models_dir)

    try:
        if model == "esrgan":
            _run_esrgan(
                input=input,
                output=output,
                mdir=mdir,
                esrgan_checkpoint=esrgan_checkpoint,
            )

        elif model == "latent_upscale":
            _run_latent_upscale(
                input=input,
                output=output,
                output_base=output_base,
                mdir=mdir,
                checkpoint=checkpoint,
                latent_upscale_checkpoint=latent_upscale_checkpoint,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
            )

        else:
            typer.echo(f"Error: model '{model}' not yet implemented.", err=True)
            raise typer.Exit(code=1)

    except (typer.Exit, SystemExit):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# ESRGAN implementation
# ---------------------------------------------------------------------------

def _run_esrgan(*, input: str, output: str, mdir: Path, esrgan_checkpoint: str | None) -> None:
    if not esrgan_checkpoint:
        typer.echo(
            "Error: --esrgan-checkpoint is required for esrgan (e.g. RealESRGAN_x4plus.pth).",
            err=True,
        )
        raise typer.Exit(code=1)

    from comfy_diffusion.image import load_image, image_upscale_with_model
    from comfy_diffusion.models import ModelManager

    mm = ModelManager(mdir)
    upscale_model = mm.load_upscale_model(esrgan_checkpoint)
    image_tensor, _mask = load_image(input)
    upscaled = image_upscale_with_model(upscale_model, image_tensor)
    pil_images = _tensor_to_pil_list(upscaled)

    out_path = save_image(pil_images, output)
    typer.echo(out_path)


# ---------------------------------------------------------------------------
# Latent upscale (hi-res fix) implementation
# ---------------------------------------------------------------------------

def _run_latent_upscale(
    *,
    input: str,
    output: str,
    output_base: str,
    mdir: Path,
    checkpoint: str | None,
    latent_upscale_checkpoint: str | None,
    prompt: str,
    negative_prompt: str,
    width: int | None,
    height: int | None,
    steps: int,
    cfg: float,
    seed: int,
) -> None:
    if not checkpoint:
        typer.echo(
            "Error: --checkpoint (or PYCOMFY_CHECKPOINT env) is required for latent_upscale.",
            err=True,
        )
        raise typer.Exit(code=1)

    from PIL import Image as _PIL
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.image import load_image, image_upscale_with_model
    from comfy_diffusion.vae import vae_encode, vae_decode
    from comfy_diffusion.latent import latent_upscale
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.sampling import sample

    mm = ModelManager(mdir)
    ckpt = mm.load_checkpoint(checkpoint)
    model_obj = ckpt["model"]
    clip = ckpt["clip"]
    vae = ckpt["vae"]

    image_tensor, _mask = load_image(input)
    src_img = _PIL.open(input).convert("RGB")

    target_w = width  or src_img.width  * 2
    target_h = height or src_img.height * 2

    positive = encode_prompt(clip, prompt)
    negative = encode_prompt(clip, negative_prompt)

    latent = vae_encode(vae, image_tensor)

    # Optional: first pass upscale model before latent re-sample
    if latent_upscale_checkpoint:
        upscale_model = mm.load_upscale_model(latent_upscale_checkpoint)
        upscaled_tensor = image_upscale_with_model(upscale_model, image_tensor)
        pil_base = _tensor_to_pil_list(upscaled_tensor)
        _save_base(pil_base, output_base)
        latent = vae_encode(vae, upscaled_tensor)
    else:
        latent = latent_upscale(latent, "nearest-exact", target_w, target_h)

    latent = sample(
        model=model_obj,
        latent=latent,
        positive=positive,
        negative=negative,
        steps=steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler="normal",
        denoise=0.5,
        seed=seed,
    )

    output_tensor = vae_decode(vae, latent)
    pil_images = _tensor_to_pil_list(output_tensor)

    out_path = save_image(pil_images, output)
    typer.echo(out_path)


def _save_base(images: list, path: str) -> None:
    """Save intermediate base image (best-effort; non-fatal)."""
    try:
        if images:
            images[0].save(path)
    except Exception:
        pass
