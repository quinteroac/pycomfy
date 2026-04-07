"""Video creation runners — maps model names to pipeline callables."""

from __future__ import annotations

from typing import Callable

MODELS = ["ltx2", "ltx23", "wan21", "wan22"]

DEFAULTS: dict[str, dict] = {
    "ltx2":  {"width": 1280, "height": 720,  "length": 97, "fps": 24, "steps": 20, "cfg": 4.0},
    "ltx23": {"width": 768,  "height": 512,  "length": 97, "fps": 25, "steps": 20, "cfg": 1.0},
    "wan21": {"width": 832,  "height": 480,  "length": 33, "fps": 16, "steps": 30, "cfg": 6.0},
    "wan22": {"width": 832,  "height": 480,  "length": 81, "fps": 16, "steps": 4,  "cfg": 1.0},
}


def default(model: str, key: str, fallback: object) -> object:
    return DEFAULTS.get(model, {}).get(key, fallback)


def _ltx2(*, mdir, prompt, image, w, h, n, f, s, c, seed, **_):  # type: ignore[no-untyped-def]
    if image is not None:
        from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import run
        result = run(models_dir=mdir, prompt=prompt, image=image, width=w, height=h, length=n, fps=f, steps=s, cfg=c, seed=seed)
    else:
        from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import run
        result = run(models_dir=mdir, prompt=prompt, width=w, height=h, length=n, fps=f, steps=s, cfg_pass1=c, seed=seed)
    return result["frames"]


def _ltx23(*, mdir, prompt, image, w, h, n, f, c, seed, audio=None, **_):  # type: ignore[no-untyped-def]
    if audio is not None:
        from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import run
        result = run(models_dir=mdir, prompt=prompt, image=image, audio_path=audio, width=w, height=h, length=n, fps=f, cfg=c, seed=seed)
        return result["frames"]
    if image is not None:
        from comfy_diffusion.pipelines.video.ltx.ltx23.i2v import run
        result = run(models_dir=mdir, prompt=prompt, image=image, width=w, height=h, length=n, fps=f, seed=seed)
    else:
        from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import run
        result = run(models_dir=mdir, prompt=prompt, width=w, height=h, length=n, fps=f, seed=seed)
    return result["frames"]


def _wan21(*, mdir, prompt, image, w, h, n, f, s, c, seed, **_):  # type: ignore[no-untyped-def]
    if image is not None:
        from comfy_diffusion.pipelines.video.wan.wan21.i2v import run
        return run(models_dir=mdir, prompt=prompt, image=image, width=w, height=h, length=n, fps=f, steps=s, cfg=c, seed=seed)
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import run
    return run(models_dir=mdir, prompt=prompt, width=w, height=h, length=n, fps=f, steps=s, cfg=c, seed=seed)


def _wan22(*, mdir, prompt, image, w, h, n, s, c, seed, **_):  # type: ignore[no-untyped-def]
    if image is not None:
        from comfy_diffusion.pipelines.video.wan.wan22.i2v import run
        return run(image=image, prompt=prompt, width=w, height=h, length=n, models_dir=mdir, steps=s, cfg=c, seed=seed)
    from comfy_diffusion.pipelines.video.wan.wan22.t2v import run
    return run(prompt=prompt, width=w, height=h, length=n, models_dir=mdir, steps=s, cfg=c, seed=seed)


RUNNERS: dict[str, Callable] = {
    "ltx2":  _ltx2,
    "ltx23": _ltx23,
    "wan21": _wan21,
    "wan22": _wan22,
}
