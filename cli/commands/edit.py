"""``parallax edit`` subcommand group — image editing.

All pipeline imports are deferred to call time (lazy imports).
"""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from cli._io import resolve_models_dir, save_image

app = typer.Typer(help="Edit existing media using a model and prompt.")

EDIT_IMAGE_MODELS = [
    "flux_4b_base",
    "flux_4b_distilled",
    "flux_9b_base",
    "flux_9b_distilled",
    "flux_9b_kv",
    "qwen",
]

_EDIT_IMAGE_DEFAULTS: dict[str, dict] = {
    "flux_4b_base":      {"width": 1024, "height": 1024, "steps": 30,  "cfg": 3.5},
    "flux_4b_distilled": {"width": 1024, "height": 1024, "steps": 4,   "cfg": 3.5},
    "flux_9b_base":      {"width": 1024, "height": 1024, "steps": 30,  "cfg": 3.5},
    "flux_9b_distilled": {"width": 1024, "height": 1024, "steps": 4,   "cfg": 3.5},
    "flux_9b_kv":        {"width": 1024, "height": 1024, "steps": 30,  "cfg": 3.5},
    "qwen":              {"width": 1024, "height": 1024, "steps": 40,  "cfg": 3.0},
}


@app.command("image")
def edit_image(
    model: Annotated[str, typer.Option(
        "--model",
        help=f"Model to use. Choices: {', '.join(EDIT_IMAGE_MODELS)}.",
    )],
    prompt: Annotated[str, typer.Option("--prompt", help="Editing instruction or text prompt.")],
    input: Annotated[str, typer.Option("--input", help="Path to the input image file.")],
    width: Annotated[Optional[int], typer.Option("--width", help="Output width in pixels.")] = None,
    height: Annotated[Optional[int], typer.Option("--height", help="Output height in pixels.")] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", help="Number of sampling steps.")] = None,
    cfg: Annotated[Optional[float], typer.Option("--cfg", help="CFG guidance scale.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility.")] = 0,
    output: Annotated[str, typer.Option("--output", help="Output file path.")] = "output.png",
    models_dir: Annotated[Optional[str], typer.Option(
        "--models-dir", help="Models directory (overrides PYCOMFY_MODELS_DIR).",
    )] = None,
    # flux_9b_kv only
    subject_image: Annotated[Optional[str], typer.Option(
        "--subject-image", help="Subject reference image (flux_9b_kv only).",
    )] = None,
    # qwen only
    image2: Annotated[Optional[str], typer.Option(
        "--image2", help="Optional second reference image (qwen only).",
    )] = None,
    image3: Annotated[Optional[str], typer.Option(
        "--image3", help="Optional third reference image (qwen only).",
    )] = None,
    no_lora: Annotated[bool, typer.Option(
        "--no-lora", help="Disable Lightning LoRA (qwen only).",
    )] = False,
    async_mode: Annotated[bool, typer.Option(
        "--async", help="Queue job and return job ID (non-blocking).",
    )] = False,
) -> None:
    """Edit an image using an instruction prompt and a reference image."""
    if model not in EDIT_IMAGE_MODELS:
        typer.echo(f"Error: unknown model '{model}'. Choices: {', '.join(EDIT_IMAGE_MODELS)}.", err=True)
        raise typer.Exit(code=1)

    if async_mode:
        typer.echo("Error: --async mode requires a job queue (not yet available).", err=True)
        raise typer.Exit(code=1)

    from pathlib import Path as _Path
    if not _Path(input).is_file():
        typer.echo(f"Error: input file not found: {input}", err=True)
        raise typer.Exit(code=1)

    mdir = resolve_models_dir(models_dir)
    defaults = _EDIT_IMAGE_DEFAULTS.get(model, {})
    w = width  or int(defaults.get("width",  1024))
    h = height or int(defaults.get("height", 1024))
    s = steps  or int(defaults.get("steps",  30))
    c = cfg    if cfg is not None else float(defaults.get("cfg", 3.5))

    try:
        from PIL import Image as _PIL
        img = _PIL.open(input).convert("RGB")

        if model == "flux_4b_base":
            from comfy_diffusion.pipelines.image.flux_klein.edit_4b_base import run
            images = run(
                models_dir=mdir,
                prompt=prompt,
                image=img,
                width=w,
                height=h,
                steps=s,
                cfg=c,
                seed=seed,
            )

        elif model == "flux_4b_distilled":
            from comfy_diffusion.pipelines.image.flux_klein.edit_4b_distilled import run
            images = run(
                models_dir=mdir,
                prompt=prompt,
                image=img,
                width=w,
                height=h,
                steps=s,
                cfg=c,
                seed=seed,
            )

        elif model == "flux_9b_base":
            from comfy_diffusion.pipelines.image.flux_klein.edit_9b_base import run
            images = run(
                models_dir=mdir,
                prompt=prompt,
                image=img,
                width=w,
                height=h,
                steps=s,
                cfg=c,
                seed=seed,
            )

        elif model == "flux_9b_distilled":
            from comfy_diffusion.pipelines.image.flux_klein.edit_9b_distilled import run
            images = run(
                models_dir=mdir,
                prompt=prompt,
                image=img,
                width=w,
                height=h,
                steps=s,
                cfg=c,
                seed=seed,
            )

        elif model == "flux_9b_kv":
            if subject_image is None:
                typer.echo(
                    "Error: --subject-image is required for flux_9b_kv.", err=True
                )
                raise typer.Exit(code=1)
            if not _Path(subject_image).is_file():
                typer.echo(f"Error: subject image not found: {subject_image}", err=True)
                raise typer.Exit(code=1)
            subj = _PIL.open(subject_image).convert("RGB")
            from comfy_diffusion.pipelines.image.flux_klein.edit_9b_kv import run
            images = run(
                models_dir=mdir,
                prompt=prompt,
                reference_image=img,
                subject_image=subj,
                width=w,
                height=h,
                steps=s,
                cfg=c,
                seed=seed,
            )

        elif model == "qwen":
            img2 = _PIL.open(image2).convert("RGB") if image2 else None
            img3 = _PIL.open(image3).convert("RGB") if image3 else None
            from comfy_diffusion.pipelines.image.qwen.edit_2511 import run
            images = run(
                prompt=prompt,
                image=img,
                image2=img2,
                image3=img3,
                models_dir=mdir,
                steps=s,
                cfg=c,
                use_lora=not no_lora,
                seed=seed,
            )

        else:
            typer.echo(f"Error: model '{model}' not yet implemented.", err=True)
            raise typer.Exit(code=1)

        out_path = save_image(images, output)
        typer.echo(out_path)

    except (typer.Exit, SystemExit):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
