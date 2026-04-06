"""``parallax edit`` subcommand group — image editing."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from cli._io import resolve_models_dir, save_image
from cli._runners import edit_image

app = typer.Typer(help="Edit existing media using a model and prompt.")


@app.command("image")
def edit_image_cmd(
    model: Annotated[str, typer.Option("--model", help=f"Model. Choices: {', '.join(edit_image.MODELS)}.")],
    prompt: Annotated[str, typer.Option("--prompt", help="Editing instruction or text prompt.")],
    input: Annotated[str, typer.Option("--input", help="Path to the input image file.")],
    width: Annotated[Optional[int], typer.Option("--width", help="Output width in pixels.")] = None,
    height: Annotated[Optional[int], typer.Option("--height", help="Output height in pixels.")] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", help="Sampling steps.")] = None,
    cfg: Annotated[Optional[float], typer.Option("--cfg", help="CFG guidance scale.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    output: Annotated[str, typer.Option("--output", help="Output file path.")] = "output.png",
    models_dir: Annotated[Optional[str], typer.Option("--models-dir", help="Models directory.")] = None,
    subject_image: Annotated[Optional[str], typer.Option("--subject-image", help="Subject reference image (flux_9b_kv only).")] = None,
    image2: Annotated[Optional[str], typer.Option("--image2", help="Optional second reference image (qwen only).")] = None,
    image3: Annotated[Optional[str], typer.Option("--image3", help="Optional third reference image (qwen only).")] = None,
    no_lora: Annotated[bool, typer.Option("--no-lora", help="Disable Lightning LoRA (qwen only).")] = False,
    async_mode: Annotated[bool, typer.Option("--async", help="Queue job, return ID immediately.")] = False,
) -> None:
    """Edit an image using an instruction prompt and a reference image."""
    if model not in edit_image.MODELS:
        typer.echo(f"Error: unknown model '{model}'. Choices: {', '.join(edit_image.MODELS)}.", err=True)
        raise typer.Exit(code=1)
    if async_mode:
        from cli._async import run_async
        run_async(action="edit", media="image", model=model,
                  args={"prompt": prompt, "input": input, "width": width, "height": height,
                        "steps": steps, "cfg": cfg, "seed": seed, "output": output,
                        "models_dir": models_dir, "subject_image": subject_image,
                        "image2": image2, "image3": image3, "no_lora": no_lora})
        return
    from pathlib import Path as _Path
    if not _Path(input).is_file():
        typer.echo(f"Error: input file not found: {input}", err=True)
        raise typer.Exit(code=1)
    if model == "flux_9b_kv":
        if subject_image is None:
            typer.echo("Error: --subject-image is required for flux_9b_kv.", err=True)
            raise typer.Exit(code=1)
        if not _Path(subject_image).is_file():
            typer.echo(f"Error: subject image not found: {subject_image}", err=True)
            raise typer.Exit(code=1)
    mdir = resolve_models_dir(models_dir)
    w = width  or int(edit_image.default(model, "width",  1024))
    h = height or int(edit_image.default(model, "height", 1024))
    s = steps  or int(edit_image.default(model, "steps",  30))
    c = cfg    if cfg is not None else float(edit_image.default(model, "cfg", 3.5))
    try:
        from PIL import Image as _PIL
        img      = _PIL.open(input).convert("RGB")
        subj     = _PIL.open(subject_image).convert("RGB") if subject_image else None
        img2_pil = _PIL.open(image2).convert("RGB") if image2 else None
        img3_pil = _PIL.open(image3).convert("RGB") if image3 else None
        images = edit_image.RUNNERS[model](
            mdir=mdir, img=img, prompt=prompt, w=w, h=h, s=s, c=c, seed=seed,
            subject_img=subj, img2=img2_pil, img3=img3_pil, no_lora=no_lora,
        )
        typer.echo(save_image(images, output))
    except (typer.Exit, SystemExit):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
