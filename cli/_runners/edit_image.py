"""Image editing runners — maps model names to pipeline callables."""

from __future__ import annotations

from typing import Callable

MODELS = [
    "flux_4b_base",
    "flux_4b_distilled",
    "flux_9b_base",
    "flux_9b_distilled",
    "flux_9b_kv",
    "qwen",
]

DEFAULTS: dict[str, dict] = {
    "flux_4b_base":      {"width": 1024, "height": 1024, "steps": 30, "cfg": 3.5},
    "flux_4b_distilled": {"width": 1024, "height": 1024, "steps": 4,  "cfg": 3.5},
    "flux_9b_base":      {"width": 1024, "height": 1024, "steps": 30, "cfg": 3.5},
    "flux_9b_distilled": {"width": 1024, "height": 1024, "steps": 4,  "cfg": 3.5},
    "flux_9b_kv":        {"width": 1024, "height": 1024, "steps": 30, "cfg": 3.5},
    "qwen":              {"width": 1024, "height": 1024, "steps": 40, "cfg": 3.0},
}


def default(model: str, key: str, fallback: object) -> object:
    return DEFAULTS.get(model, {}).get(key, fallback)


def _flux_4b_base(*, mdir, img, prompt, w, h, s, c, seed, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.flux_klein.edit_4b_base import run
    return run(models_dir=mdir, prompt=prompt, image=img, width=w, height=h, steps=s, cfg=c, seed=seed)


def _flux_4b_distilled(*, mdir, img, prompt, w, h, s, c, seed, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.flux_klein.edit_4b_distilled import run
    return run(models_dir=mdir, prompt=prompt, image=img, width=w, height=h, steps=s, cfg=c, seed=seed)


def _flux_9b_base(*, mdir, img, prompt, w, h, s, c, seed, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.flux_klein.edit_9b_base import run
    return run(models_dir=mdir, prompt=prompt, image=img, width=w, height=h, steps=s, cfg=c, seed=seed)


def _flux_9b_distilled(*, mdir, img, prompt, w, h, s, c, seed, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.flux_klein.edit_9b_distilled import run
    return run(models_dir=mdir, prompt=prompt, image=img, width=w, height=h, steps=s, cfg=c, seed=seed)


def _flux_9b_kv(*, mdir, img, prompt, w, h, s, c, seed, subject_img, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.flux_klein.edit_9b_kv import run
    return run(models_dir=mdir, prompt=prompt, reference_image=img, subject_image=subject_img,
               width=w, height=h, steps=s, cfg=c, seed=seed)


def _qwen(*, mdir, img, prompt, w, h, s, c, seed, img2, img3, no_lora, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import run
    return run(prompt=prompt, image=img, image2=img2, image3=img3,
               models_dir=mdir, steps=s, cfg=c, use_lora=not no_lora, seed=seed)


RUNNERS: dict[str, Callable] = {
    "flux_4b_base":      _flux_4b_base,
    "flux_4b_distilled": _flux_4b_distilled,
    "flux_9b_base":      _flux_9b_base,
    "flux_9b_distilled": _flux_9b_distilled,
    "flux_9b_kv":        _flux_9b_kv,
    "qwen":              _qwen,
}
