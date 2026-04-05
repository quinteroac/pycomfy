"""``parallax create`` subcommand group — image, video, audio generation.

All pipeline imports are deferred to call time (lazy imports) so that the CLI
module is importable without torch or comfy being installed.
"""

from __future__ import annotations

import sys
from typing import Annotated, Optional

import typer

from cli._io import resolve_models_dir, save_audio, save_image, save_video_frames

app = typer.Typer(help="Generate media from a text prompt.")

# ---------------------------------------------------------------------------
# Model choices
# ---------------------------------------------------------------------------
IMAGE_MODELS = ["sdxl", "anima", "z_image", "flux_klein", "qwen"]
VIDEO_MODELS = ["ltx2", "ltx23", "wan21", "wan22"]
AUDIO_MODELS = ["ace_step"]

# Per-model defaults (mirrors packages/parallax_cli/src/models/registry.ts).
_IMAGE_DEFAULTS: dict[str, dict] = {
    "sdxl":      {"width": 1024, "height": 1024, "steps": 25,  "cfg": 7.5},
    "anima":     {"width": 1024, "height": 1024, "steps": 30,  "cfg": 4.0},
    "z_image":   {"width": 1024, "height": 1024, "steps": 8,   "cfg": 7.0},
    "flux_klein": {"width": 1024, "height": 1024, "steps": 4,   "cfg": 1.0},
    "qwen":      {"width": 640,  "height": 640,  "steps": 20,  "cfg": 2.5},
}

_VIDEO_DEFAULTS: dict[str, dict] = {
    "ltx2":  {"width": 1280, "height": 720,  "length": 97, "fps": 24, "steps": 20, "cfg": 4.0},
    "ltx23": {"width": 768,  "height": 512,  "length": 97, "fps": 25, "steps": 20, "cfg": 1.0},
    "wan21": {"width": 832,  "height": 480,  "length": 33, "fps": 16, "steps": 30, "cfg": 6.0},
    "wan22": {"width": 832,  "height": 480,  "length": 81, "fps": 16, "steps": 4,  "cfg": 1.0},
}

_AUDIO_DEFAULTS: dict[str, dict] = {
    "ace_step": {"length": 120, "steps": 8, "cfg": 1.0, "bpm": 120},
}


def _d(media: str, model: str, key: str, fallback: object) -> object:
    """Look up a per-model default value; return *fallback* when absent."""
    table = {"image": _IMAGE_DEFAULTS, "video": _VIDEO_DEFAULTS, "audio": _AUDIO_DEFAULTS}.get(
        media, {}
    )
    return table.get(model, {}).get(key, fallback)


# ---------------------------------------------------------------------------
# create image
# ---------------------------------------------------------------------------

@app.command("image")
def create_image(
    model: Annotated[str, typer.Option(
        "--model", help=f"Model to use. Choices: {', '.join(IMAGE_MODELS)}.",
    )],
    prompt: Annotated[str, typer.Option("--prompt", help="Text prompt describing the image.")],
    negative_prompt: Annotated[Optional[str], typer.Option(
        "--negative-prompt", help="Negative prompt (what to avoid).",
    )] = None,
    width: Annotated[Optional[int], typer.Option("--width", help="Image width in pixels.")] = None,
    height: Annotated[Optional[int], typer.Option("--height", help="Image height in pixels.")] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", help="Number of sampling steps.")] = None,
    cfg: Annotated[Optional[float], typer.Option("--cfg", help="CFG guidance scale.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility.")] = 0,
    output: Annotated[str, typer.Option("--output", help="Output file path.")] = "output.png",
    models_dir: Annotated[Optional[str], typer.Option(
        "--models-dir", help="Models directory (overrides PYCOMFY_MODELS_DIR).",
    )] = None,
    async_mode: Annotated[bool, typer.Option("--async", help="Queue job and return job ID (non-blocking).")] = False,
) -> None:
    """Generate an image from a text prompt."""
    if model not in IMAGE_MODELS:
        typer.echo(f"Error: unknown model '{model}'. Choices: {', '.join(IMAGE_MODELS)}.", err=True)
        raise typer.Exit(code=1)

    if async_mode:
        from cli._async import run_async
        run_async(
            action="create",
            media="image",
            model=model,
            args={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "output": output,
                "models_dir": models_dir,
            },
        )
        return

    mdir = resolve_models_dir(models_dir)

    w = width or int(_d("image", model, "width", 1024))
    h = height or int(_d("image", model, "height", 1024))
    s = steps or int(_d("image", model, "steps", 20))
    c = cfg if cfg is not None else float(_d("image", model, "cfg", 7.0))

    try:
        if model == "sdxl":
            from comfy_diffusion.pipelines.image.sdxl.t2i import run
            images = run(
                models_dir=mdir,
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=w,
                height=h,
                steps=s,
                cfg=c,
                seed=seed,
            )
        elif model == "anima":
            from comfy_diffusion.pipelines.image.anima.t2i import run
            images = run(
                models_dir=mdir,
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=w,
                height=h,
                steps=s,
                cfg=c,
                seed=seed,
            )
        elif model == "z_image":
            from comfy_diffusion.pipelines.image.z_image.turbo import run
            images = run(
                models_dir=mdir,
                prompt=prompt,
                width=w,
                height=h,
                steps=s,
                seed=seed,
            )
        elif model == "flux_klein":
            from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import run
            images = run(
                models_dir=mdir,
                prompt=prompt,
                width=w,
                height=h,
                steps=s,
                seed=seed,
            )
        elif model == "qwen":
            from comfy_diffusion.pipelines.image.qwen.layered import run_t2l
            images = run_t2l(
                prompt=prompt,
                width=w,
                height=h,
                steps=s,
                cfg=c,
                seed=seed,
                models_dir=mdir,
            )
        else:
            typer.echo(f"Error: model '{model}' not yet implemented.", err=True)
            raise typer.Exit(code=1)

        out_path = save_image(images, output)
        typer.echo(out_path)

    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# create video
# ---------------------------------------------------------------------------

@app.command("video")
def create_video(
    model: Annotated[str, typer.Option(
        "--model", help=f"Model to use. Choices: {', '.join(VIDEO_MODELS)}.",
    )],
    prompt: Annotated[str, typer.Option("--prompt", help="Text prompt describing the video.")],
    input: Annotated[Optional[str], typer.Option(
        "--input", help="Input image path for image-to-video (ltx2, ltx23, wan21, wan22).",
    )] = None,
    width: Annotated[Optional[int], typer.Option("--width", help="Frame width in pixels.")] = None,
    height: Annotated[Optional[int], typer.Option("--height", help="Frame height in pixels.")] = None,
    length: Annotated[Optional[int], typer.Option("--length", help="Number of frames to generate.")] = None,
    fps: Annotated[Optional[int], typer.Option("--fps", help="Output frame rate.")] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", help="Number of sampling steps.")] = None,
    cfg: Annotated[Optional[float], typer.Option("--cfg", help="CFG guidance scale.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility.")] = 0,
    output: Annotated[str, typer.Option("--output", help="Output file path.")] = "output.mp4",
    models_dir: Annotated[Optional[str], typer.Option(
        "--models-dir", help="Models directory (overrides PYCOMFY_MODELS_DIR).",
    )] = None,
    async_mode: Annotated[bool, typer.Option("--async", help="Queue job and return job ID (non-blocking).")] = False,
) -> None:
    """Generate a video from a text prompt."""
    if model not in VIDEO_MODELS:
        typer.echo(f"Error: unknown model '{model}'. Choices: {', '.join(VIDEO_MODELS)}.", err=True)
        raise typer.Exit(code=1)

    if async_mode:
        from cli._async import run_async
        run_async(
            action="create",
            media="video",
            model=model,
            args={
                "prompt": prompt,
                "input": input,
                "width": width,
                "height": height,
                "length": length,
                "fps": fps,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "output": output,
                "models_dir": models_dir,
            },
        )
        return

    if input is not None:
        from pathlib import Path as _Path
        if not _Path(input).is_file():
            typer.echo(f"Error: input file not found: {input}", err=True)
            raise typer.Exit(code=1)

    mdir = resolve_models_dir(models_dir)

    w = width  or int(_d("video", model, "width",  832))
    h = height or int(_d("video", model, "height", 480))
    n = length or int(_d("video", model, "length", 33))
    f = fps    or int(_d("video", model, "fps",    16))
    s = steps  or int(_d("video", model, "steps",  20))
    c = cfg if cfg is not None else float(_d("video", model, "cfg", 6.0))

    try:
        if model == "ltx2":
            if input is not None:
                from PIL import Image as _PIL
                from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import run
                ref = _PIL.open(input).convert("RGB")
                result = run(
                    models_dir=mdir,
                    prompt=prompt,
                    image=ref,
                    width=w,
                    height=h,
                    length=n,
                    fps=f,
                    steps=s,
                    cfg=c,
                    seed=seed,
                )
            else:
                from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import run
                result = run(
                    models_dir=mdir,
                    prompt=prompt,
                    width=w,
                    height=h,
                    length=n,
                    fps=f,
                    steps=s,
                    cfg_pass1=c,
                    seed=seed,
                )
            frames = result["frames"]
            out_path = save_video_frames(frames, output, fps=float(f))

        elif model == "ltx23":
            if input is not None:
                from PIL import Image as _PIL
                from comfy_diffusion.pipelines.video.ltx.ltx23.i2v import run
                ref = _PIL.open(input).convert("RGB")
                result = run(
                    models_dir=mdir,
                    prompt=prompt,
                    image=ref,
                    width=w,
                    height=h,
                    length=n,
                    fps=f,
                    seed=seed,
                )
            else:
                from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import run
                result = run(
                    models_dir=mdir,
                    prompt=prompt,
                    width=w,
                    height=h,
                    length=n,
                    fps=f,
                    seed=seed,
                )
            frames = result["frames"]
            out_path = save_video_frames(frames, output, fps=float(f))

        elif model == "wan21":
            if input is not None:
                from PIL import Image as _PIL
                from comfy_diffusion.pipelines.video.wan.wan21.i2v import run
                ref = _PIL.open(input).convert("RGB")
                frames = run(
                    models_dir=mdir,
                    prompt=prompt,
                    image=ref,
                    width=w,
                    height=h,
                    length=n,
                    fps=f,
                    steps=s,
                    cfg=c,
                    seed=seed,
                )
            else:
                from comfy_diffusion.pipelines.video.wan.wan21.t2v import run
                frames = run(
                    models_dir=mdir,
                    prompt=prompt,
                    width=w,
                    height=h,
                    length=n,
                    fps=f,
                    steps=s,
                    cfg=c,
                    seed=seed,
                )
            out_path = save_video_frames(frames, output, fps=float(f))

        elif model == "wan22":
            if input is not None:
                from PIL import Image as _PIL
                from comfy_diffusion.pipelines.video.wan.wan22.i2v import run
                ref = _PIL.open(input).convert("RGB")
                frames = run(
                    image=ref,
                    prompt=prompt,
                    width=w,
                    height=h,
                    length=n,
                    models_dir=mdir,
                    steps=s,
                    cfg=c,
                    seed=seed,
                )
            else:
                from comfy_diffusion.pipelines.video.wan.wan22.t2v import run
                frames = run(
                    prompt=prompt,
                    width=w,
                    height=h,
                    length=n,
                    models_dir=mdir,
                    steps=s,
                    cfg=c,
                    seed=seed,
                )
            out_path = save_video_frames(frames, output, fps=float(f))

        else:
            typer.echo(f"Error: model '{model}' not yet implemented.", err=True)
            raise typer.Exit(code=1)

        typer.echo(out_path)

    except (typer.Exit, SystemExit):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# create audio
# ---------------------------------------------------------------------------

@app.command("audio")
def create_audio(
    model: Annotated[str, typer.Option(
        "--model", help=f"Model to use. Choices: {', '.join(AUDIO_MODELS)}.",
    )],
    prompt: Annotated[str, typer.Option("--prompt", help="Text prompt (genre tags, instruments, mood).")],
    length: Annotated[Optional[float], typer.Option("--length", help="Duration in seconds.")] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", help="Number of sampling steps.")] = None,
    cfg: Annotated[Optional[float], typer.Option("--cfg", help="CFG guidance scale.")] = None,
    bpm: Annotated[Optional[int], typer.Option("--bpm", help="Beats per minute.")] = None,
    lyrics: Annotated[str, typer.Option("--lyrics", help="Lyrics text.")] = "",
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility.")] = 0,
    output: Annotated[str, typer.Option("--output", help="Output file path.")] = "output.wav",
    models_dir: Annotated[Optional[str], typer.Option(
        "--models-dir", help="Models directory (overrides PYCOMFY_MODELS_DIR).",
    )] = None,
    async_mode: Annotated[bool, typer.Option("--async", help="Queue job and return job ID (non-blocking).")] = False,
) -> None:
    """Generate audio from a text prompt."""
    if model not in AUDIO_MODELS:
        typer.echo(f"Error: unknown model '{model}'. Choices: {', '.join(AUDIO_MODELS)}.", err=True)
        raise typer.Exit(code=1)

    if async_mode:
        from cli._async import run_async
        run_async(
            action="create",
            media="audio",
            model=model,
            args={
                "prompt": prompt,
                "length": length,
                "steps": steps,
                "cfg": cfg,
                "bpm": bpm,
                "lyrics": lyrics,
                "seed": seed,
                "output": output,
                "models_dir": models_dir,
            },
        )
        return

    mdir = resolve_models_dir(models_dir)
    dur = length if length is not None else float(_d("audio", model, "length", 120))
    s   = steps  or int(_d("audio", model, "steps", 8))
    c   = cfg    if cfg is not None else float(_d("audio", model, "cfg", 1.0))
    b   = bpm    or int(_d("audio", model, "bpm", 120))

    try:
        if model == "ace_step":
            from comfy_diffusion.pipelines.audio.ace_step.v1_5.checkpoint import run
            result = run(
                models_dir=mdir,
                tags=prompt,
                lyrics=lyrics,
                duration=dur,
                bpm=b,
                seed=seed,
                steps=s,
                cfg=c,
            )
            out_path = save_audio(result["audio"], output)
        else:
            typer.echo(f"Error: model '{model}' not yet implemented.", err=True)
            raise typer.Exit(code=1)

        typer.echo(out_path)

    except (typer.Exit, SystemExit):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
