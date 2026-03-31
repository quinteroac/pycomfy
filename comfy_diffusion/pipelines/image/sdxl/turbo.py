"""SDXL Turbo single-step text-to-image pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes single-step (or few-step) distilled inference end-to-end:
  model loading, conditioning, turbo-scheduled sampling, and VAE decoding.

This pipeline mirrors the SDXL Turbo workflow: ``SDTurboScheduler`` generates
sigmas for distilled single-step generation, ``euler_ancestral`` is the sampler,
and ``SamplerCustomAdvanced`` drives inference via ``sample_custom_simple``.

Pattern
-------
Every pipeline module in ``comfy_diffusion/pipelines/`` follows this contract:

1. Define a ``manifest() -> list[ModelEntry]`` function that declares all
   required model files as typed ``ModelEntry`` dataclass instances.
2. Define a ``run(...)`` function that implements the full pipeline using the
   ``comfy_diffusion`` public API (lazy imports only — no top-level
   ``comfy.*`` or ``comfy_diffusion.*`` imports).
3. Keep the manifest as the single source of truth for model file paths —
   ``run()`` should derive its default paths from ``manifest()`` so the two
   stay in sync automatically.

Usage
-----
::

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    images = run(
        models_dir="/path/to/models",
        prompt="a golden retriever puppy on a summer beach",
    )
    image = images[0]  # PIL.Image.Image (512×512)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repository for SDXL Turbo
# ---------------------------------------------------------------------------

_HF_REPO = "stabilityai/sdxl-turbo"

# Relative destination path (resolved against models_dir by download_models).
_CKPT_DEST = Path("checkpoints") / "sd_xl_turbo_1.0_fp16.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the SDXL Turbo pipeline.

    Returns exactly 1 :class:`~comfy_diffusion.downloader.HFModelEntry`:
    the SDXL Turbo fp16 checkpoint.
    Pass the result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="sd_xl_turbo_1.0_fp16.safetensors",
            dest=_CKPT_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 1,
    cfg: float = 0.0,
    seed: int = 0,
    ckpt_filename: str | None = None,
) -> list[Any]:
    """Run the SDXL Turbo text-to-image pipeline end-to-end.

    Single-pass distilled sampling using ``SDTurboScheduler`` and the
    ``euler_ancestral`` sampler.  For distilled models, ``cfg=0.0`` is the
    recommended default; non-zero values are accepted but may degrade quality.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    prompt : str
        Positive text prompt describing the desired image content.
    negative_prompt : str, optional
        Negative text prompt.  Default ``""``.
    width : int, optional
        Output image width in pixels.  Default ``512``.
    height : int, optional
        Output image height in pixels.  Default ``512``.
    steps : int, optional
        Number of denoising steps (1–10).  Default ``1``.
    cfg : float, optional
        CFG scale.  For SDXL Turbo, ``0.0`` (guidance-free) is recommended.
        Default ``0.0``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    ckpt_filename : str | None, optional
        Override the default checkpoint filename relative to ``models_dir``.
        Default ``None``.

    Returns
    -------
    list[PIL.Image.Image]
        A list containing the generated image (one element for batch size 1).
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.latent import empty_latent_image
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import get_sampler, sample_custom_simple, sd_turbo_scheduler
    from comfy_diffusion.vae import vae_decode

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    # Resolve checkpoint path (allow caller override, fall back to manifest path).
    ckpt_path = models_dir / (ckpt_filename or _CKPT_DEST)

    # Load checkpoint.
    result = mm.load_checkpoint(str(ckpt_path))
    model = result.model
    clip = result.clip
    vae = result.vae

    # Encode prompt.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    # Create the starting latent.
    latent = empty_latent_image(width, height)

    # Build sampler and sigmas.
    sampler = get_sampler("euler_ancestral")
    sigmas = sd_turbo_scheduler(model, steps=steps, denoise=1.0)

    # Run distilled single-step sampling.
    latent_out = sample_custom_simple(
        model,
        positive,
        negative,
        latent,
        sampler,
        sigmas,
        cfg,
        seed,
        add_noise=True,
    )

    # Decode final latent.
    image = vae_decode(vae, latent_out)

    return [image]
