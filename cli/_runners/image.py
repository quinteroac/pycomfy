"""Image creation runners — maps model names to pipeline callables."""

from __future__ import annotations

from typing import Callable

MODELS = ["sdxl", "anima", "z_image", "flux_klein", "qwen"]

DEFAULTS: dict[str, dict] = {
    "sdxl":       {"width": 1024, "height": 1024, "steps": 25, "cfg": 7.5},
    "anima":      {"width": 1024, "height": 1024, "steps": 30, "cfg": 4.0},
    "z_image":    {"width": 1024, "height": 1024, "steps": 8,  "cfg": 1.0},
    "flux_klein": {"width": 1024, "height": 1024, "steps": 4,  "cfg": 1.0},
    "qwen":       {"width": 640,  "height": 640,  "steps": 20, "cfg": 2.5},
}


def default(model: str, key: str, fallback: object) -> object:
    return DEFAULTS.get(model, {}).get(key, fallback)


def _sdxl(*, mdir, prompt, neg, w, h, s, c, seed, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.sdxl.t2i import run
    return run(models_dir=mdir, prompt=prompt, negative_prompt=neg or "", width=w, height=h, steps=s, cfg=c, seed=seed)


def _anima(*, mdir, prompt, neg, w, h, s, c, seed, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.anima.t2i import run
    return run(models_dir=mdir, prompt=prompt, negative_prompt=neg or "", width=w, height=h, steps=s, cfg=c, seed=seed)


def _z_image(*, mdir, prompt, w, h, s, seed, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.z_image.turbo import run
    return run(models_dir=mdir, prompt=prompt, width=w, height=h, steps=s, seed=seed)


def _flux_klein(*, mdir, prompt, w, h, s, seed, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import run
    return run(models_dir=mdir, prompt=prompt, width=w, height=h, steps=s, seed=seed)


def _qwen(*, mdir, prompt, w, h, s, c, seed, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.qwen.layered import run_t2l
    return run_t2l(prompt=prompt, width=w, height=h, steps=s, cfg=c, seed=seed, models_dir=mdir)


RUNNERS: dict[str, Callable] = {
    "sdxl":       _sdxl,
    "anima":      _anima,
    "z_image":    _z_image,
    "flux_klein": _flux_klein,
    "qwen":       _qwen,
}
