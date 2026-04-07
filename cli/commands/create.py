"""``parallax create`` subcommand group — image, video, audio generation."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from cli._io import resolve_models_dir, save_audio, save_image, save_video_frames
from cli._runners import audio, image, video
from cli.commands._common import ensure_env_on_path

app = typer.Typer(help="Generate media from a text prompt.")


@app.command("image")
def create_image(
    model: Annotated[str, typer.Option("--model", help=f"Model. Choices: {', '.join(image.MODELS)}.")],
    prompt: Annotated[str, typer.Option("--prompt", help="Text prompt describing the image.")],
    negative_prompt: Annotated[Optional[str], typer.Option("--negative-prompt", help="Negative prompt.")] = None,
    width: Annotated[Optional[int], typer.Option("--width", help="Image width in pixels.")] = None,
    height: Annotated[Optional[int], typer.Option("--height", help="Image height in pixels.")] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", help="Sampling steps.")] = None,
    cfg: Annotated[Optional[float], typer.Option("--cfg", help="CFG guidance scale.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    output: Annotated[str, typer.Option("--output", help="Output file path.")] = "output.png",
    models_dir: Annotated[Optional[str], typer.Option("--models-dir", help="Models directory.")] = None,
    async_mode: Annotated[bool, typer.Option("--async", help="Queue job, return ID immediately.")] = False,
) -> None:
    """Generate an image from a text prompt."""
    if model not in image.MODELS:
        typer.echo(f"Error: unknown model '{model}'. Choices: {', '.join(image.MODELS)}.", err=True)
        raise typer.Exit(code=1)
    if async_mode:
        from cli._async import run_async
        run_async(action="create", media="image", model=model,
                  args={"prompt": prompt, "negative_prompt": negative_prompt, "width": width,
                        "height": height, "steps": steps, "cfg": cfg, "seed": seed,
                        "output": output, "models_dir": models_dir})
        return
    ensure_env_on_path()
    mdir = resolve_models_dir(models_dir)
    w = width  or int(image.default(model, "width",  1024))
    h = height or int(image.default(model, "height", 1024))
    s = steps  or int(image.default(model, "steps",  20))
    c = cfg if cfg is not None else float(image.default(model, "cfg", 7.0))
    try:
        images = image.RUNNERS[model](mdir=mdir, prompt=prompt, neg=negative_prompt, w=w, h=h, s=s, c=c, seed=seed)
        typer.echo(save_image(images, output))
    except (typer.Exit, SystemExit):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


_AUDIO_SUPPORTED_MODELS = {"ltx23"}


@app.command("video")
def create_video(
    model: Annotated[str, typer.Option("--model", help=f"Model. Choices: {', '.join(video.MODELS)}.")],
    prompt: Annotated[str, typer.Option("--prompt", help="Text prompt describing the video.")],
    input: Annotated[Optional[str], typer.Option("--input", help="Input image for i2v.")]= None,
    audio: Annotated[Optional[str], typer.Option("--audio", help="Input audio file for audio-conditioned generation (ltx23 only).")] = None,
    width: Annotated[Optional[int], typer.Option("--width", help="Frame width in pixels.")] = None,
    height: Annotated[Optional[int], typer.Option("--height", help="Frame height in pixels.")] = None,
    length: Annotated[Optional[int], typer.Option("--length", help="Number of frames.")] = None,
    fps: Annotated[Optional[int], typer.Option("--fps", help="Output frame rate.")] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", help="Sampling steps.")] = None,
    cfg: Annotated[Optional[float], typer.Option("--cfg", help="CFG guidance scale.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    output: Annotated[str, typer.Option("--output", help="Output file path.")] = "output.mp4",
    models_dir: Annotated[Optional[str], typer.Option("--models-dir", help="Models directory.")] = None,
    async_mode: Annotated[bool, typer.Option("--async", help="Queue job, return ID immediately.")] = False,
) -> None:
    """Generate a video from a text prompt."""
    from pathlib import Path as _Path

    if model not in video.MODELS:
        typer.echo(f"Error: unknown model '{model}'. Choices: {', '.join(video.MODELS)}.", err=True)
        raise typer.Exit(code=1)

    # --audio validation (before async dispatch so errors surface immediately).
    if audio is not None:
        if model not in _AUDIO_SUPPORTED_MODELS:
            typer.echo("Error: --audio is only supported for model 'ltx23'.", err=True)
            raise typer.Exit(code=1)
        if not _Path(audio).is_file():
            typer.echo(f"Error: audio file not found: {audio}", err=True)
            raise typer.Exit(code=1)
        if input is None:
            typer.echo("Error: --audio requires --input (image).", err=True)
            raise typer.Exit(code=1)

    if async_mode:
        from cli._async import run_async
        run_async(action="create", media="video", model=model,
                  args={"prompt": prompt, "input": input, "audio": audio, "width": width,
                        "height": height, "length": length, "fps": fps, "steps": steps,
                        "cfg": cfg, "seed": seed, "output": output, "models_dir": models_dir})
        return
    if input is not None:
        if not _Path(input).is_file():
            typer.echo(f"Error: input file not found: {input}", err=True)
            raise typer.Exit(code=1)
    ensure_env_on_path()
    mdir = resolve_models_dir(models_dir)
    w = width  or int(video.default(model, "width",  832))
    h = height or int(video.default(model, "height", 480))
    n = length or int(video.default(model, "length", 33))
    f = fps    or int(video.default(model, "fps",    16))
    s = steps  or int(video.default(model, "steps",  20))
    c = cfg if cfg is not None else float(video.default(model, "cfg", 6.0))
    image_obj = None
    if input is not None:
        from PIL import Image as _PIL
        image_obj = _PIL.open(input).convert("RGB")
    try:
        frames = video.RUNNERS[model](mdir=mdir, prompt=prompt, image=image_obj, audio=audio, w=w, h=h, n=n, f=f, s=s, c=c, seed=seed)
        typer.echo(save_video_frames(frames, output, fps=float(f)))
    except (typer.Exit, SystemExit):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("audio")
def create_audio(
    model: Annotated[str, typer.Option("--model", help=f"Model. Choices: {', '.join(audio.MODELS)}.")],
    prompt: Annotated[str, typer.Option("--prompt", help="Text prompt (genre tags, mood).")],
    length: Annotated[Optional[float], typer.Option("--length", help="Duration in seconds.")] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", help="Sampling steps.")] = None,
    cfg: Annotated[Optional[float], typer.Option("--cfg", help="CFG guidance scale.")] = None,
    bpm: Annotated[Optional[int], typer.Option("--bpm", help="Beats per minute.")] = None,
    lyrics: Annotated[str, typer.Option("--lyrics", help="Lyrics text.")] = "",
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    output: Annotated[str, typer.Option("--output", help="Output file path.")] = "output.wav",
    models_dir: Annotated[Optional[str], typer.Option("--models-dir", help="Models directory.")] = None,
    async_mode: Annotated[bool, typer.Option("--async", help="Queue job, return ID immediately.")] = False,
) -> None:
    """Generate audio from a text prompt."""
    if model not in audio.MODELS:
        typer.echo(f"Error: unknown model '{model}'. Choices: {', '.join(audio.MODELS)}.", err=True)
        raise typer.Exit(code=1)
    if async_mode:
        from cli._async import run_async
        run_async(action="create", media="audio", model=model,
                  args={"prompt": prompt, "length": length, "steps": steps, "cfg": cfg, "bpm": bpm,
                        "lyrics": lyrics, "seed": seed, "output": output, "models_dir": models_dir})
        return
    ensure_env_on_path()
    mdir = resolve_models_dir(models_dir)
    dur = length if length is not None else float(audio.default(model, "length", 120))
    s   = steps  or int(audio.default(model, "steps", 8))
    c   = cfg    if cfg is not None else float(audio.default(model, "cfg", 1.0))
    b   = bpm    or int(audio.default(model, "bpm", 120))
    try:
        result = audio.RUNNERS[model](mdir=mdir, prompt=prompt, lyrics=lyrics, dur=dur, b=b, s=s, c=c, seed=seed)
        typer.echo(save_audio(result["audio"], output))
    except (typer.Exit, SystemExit):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
