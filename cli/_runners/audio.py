"""Audio creation runners — maps model names to pipeline callables."""

from __future__ import annotations

from typing import Callable

MODELS = ["ace_step"]

DEFAULTS: dict[str, dict] = {
    "ace_step": {"length": 120, "steps": 8, "cfg": 1.0, "bpm": 120},
}


def default(model: str, key: str, fallback: object) -> object:
    return DEFAULTS.get(model, {}).get(key, fallback)


def _ace_step(*, mdir, prompt, lyrics, dur, b, s, c, seed, **_):  # type: ignore[no-untyped-def]
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split import run
    return run(models_dir=mdir, tags=prompt, lyrics=lyrics, duration=dur, bpm=b, seed=seed, steps=s, cfg=c)


RUNNERS: dict[str, Callable] = {
    "ace_step": _ace_step,
}
