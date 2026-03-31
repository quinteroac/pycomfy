"""SDXL base + refiner text-to-image pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full two-pass inference pipeline end-to-end: model
  loading, conditioning, base sampling, refiner sampling, and VAE decoding.

This pipeline mirrors the standard SDXL base + refiner workflow: pass 1 runs
``KSamplerAdvanced`` on the base model with leftover noise enabled, and pass 2
continues denoising with the refiner model starting from where pass 1 stopped.

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
    from comfy_diffusion.pipelines.image.sdxl.t2i import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    images = run(
        models_dir="/path/to/models",
        prompt="a majestic eagle soaring over snow-capped mountains",
    )
    image = images[0]  # PIL.Image.Image (1024×1024)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repositories for SDXL
# ---------------------------------------------------------------------------

_HF_REPO_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
_HF_REPO_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Relative destination paths (resolved against models_dir by download_models).
_BASE_DEST = Path("checkpoints") / "sd_xl_base_1.0.safetensors"
_REFINER_DEST = Path("checkpoints") / "sd_xl_refiner_1.0.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the SDXL T2I pipeline.

    Returns exactly 2 :class:`~comfy_diffusion.downloader.HFModelEntry`
    instances: the SDXL base checkpoint and the SDXL refiner checkpoint.
    Pass the result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_BASE,
            filename="sd_xl_base_1.0.safetensors",
            dest=_BASE_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_REFINER,
            filename="sd_xl_refiner_1.0.safetensors",
            dest=_REFINER_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    base_end_step: int = 20,
    cfg: float = 7.5,
    seed: int = 0,
    sampler_name: str = "euler",
    scheduler: str = "karras",
    base_filename: str | None = None,
    refiner_filename: str | None = None,
) -> list[Any]:
    """Run the SDXL base + refiner text-to-image pipeline end-to-end.

    Two-pass sampling: pass 1 runs ``KSamplerAdvanced`` on the base model with
    ``add_noise=True`` and ``return_with_leftover_noise=True``, stopping at
    ``base_end_step``; pass 2 continues on the refiner with ``add_noise=False``
    and ``return_with_leftover_noise=False`` from ``base_end_step`` to
    ``end_at_step=10000``.  The final latent is decoded with the refiner VAE.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    prompt : str
        Positive text prompt describing the desired image content.
    negative_prompt : str, optional
        Negative text prompt.  Default ``""``.
    width : int, optional
        Output image width in pixels.  Default ``1024``.
    height : int, optional
        Output image height in pixels.  Default ``1024``.
    steps : int, optional
        Total number of denoising steps shared across both passes.  Default ``25``.
    base_end_step : int, optional
        Step at which the base pass ends and the refiner pass begins.
        Default ``20`` (i.e., base handles steps 0–20, refiner handles 20–25).
    cfg : float, optional
        CFG scale applied in both passes.  Default ``7.5``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    sampler_name : str, optional
        Sampler algorithm name.  Default ``"euler"``.
    scheduler : str, optional
        Noise schedule name.  Default ``"karras"``.
    base_filename : str | None, optional
        Override the default base checkpoint filename.  Default ``None``.
    refiner_filename : str | None, optional
        Override the default refiner checkpoint filename.  Default ``None``.

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
    from comfy_diffusion.sampling import sample_advanced
    from comfy_diffusion.vae import vae_decode

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    # Resolve checkpoint paths (allow caller overrides, fall back to manifest paths).
    base_path = models_dir / (base_filename or _BASE_DEST)
    refiner_path = models_dir / (refiner_filename or _REFINER_DEST)

    # Load base and refiner checkpoints.
    base_result = mm.load_checkpoint(str(base_path))
    refiner_result = mm.load_checkpoint(str(refiner_path))

    base_model = base_result.model
    base_clip = base_result.clip

    refiner_model = refiner_result.model
    refiner_clip = refiner_result.clip
    refiner_vae = refiner_result.vae

    # Encode prompt with both base and refiner CLIP encoders.
    base_positive, base_negative = encode_prompt(base_clip, prompt, negative_prompt)
    refiner_positive, refiner_negative = encode_prompt(refiner_clip, prompt, negative_prompt)

    # Create the starting latent.
    latent = empty_latent_image(width, height)

    # Pass 1: base model — add noise, keep leftover noise for refiner.
    latent_pass1 = sample_advanced(
        base_model,
        base_positive,
        base_negative,
        latent,
        steps,
        cfg,
        sampler_name,
        scheduler,
        seed,
        add_noise=True,
        return_with_leftover_noise=True,
        start_at_step=0,
        end_at_step=base_end_step,
    )

    # Pass 2: refiner model — no new noise, full denoise to completion.
    latent_pass2 = sample_advanced(
        refiner_model,
        refiner_positive,
        refiner_negative,
        latent_pass1,
        steps,
        cfg,
        sampler_name,
        scheduler,
        seed,
        add_noise=False,
        return_with_leftover_noise=False,
        start_at_step=base_end_step,
        end_at_step=10000,
    )

    # Decode final latent with the refiner VAE.
    image = vae_decode(refiner_vae, latent_pass2)

    return [image]
