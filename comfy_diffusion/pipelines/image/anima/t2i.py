"""Anima Preview text-to-image pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  text conditioning, sampling with the ``er_sde`` sampler, and VAE decoding.

This pipeline uses the Anima Preview diffusion model with a Qwen 3 0.6B text
encoder, producing anime-style images.

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
    from comfy_diffusion.pipelines.image.anima.t2i import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    images = run(
        models_dir="/path/to/models",
        prompt="1girl, anime style, masterpiece",
    )
    image = images[0]  # PIL.Image.Image (1024×1024)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repository for Anima Preview
# ---------------------------------------------------------------------------

_HF_REPO = "Comfy-Org/Anima-Preview"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "anima-preview2.safetensors"
_CLIP_DEST = Path("text_encoders") / "qwen_3_06b_base.safetensors"
_VAE_DEST = Path("vae") / "qwen_image_vae.safetensors"

# Workflow defaults.
_DEFAULT_SAMPLER = "er_sde"
_DEFAULT_SCHEDULER = "normal"
_DEFAULT_CFG = 4.0
_DEFAULT_STEPS = 30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the Anima Preview T2I pipeline.

    Returns exactly 3 :class:`~comfy_diffusion.downloader.HFModelEntry`
    instances: the Anima diffusion model, the Qwen text encoder, and the
    Qwen image VAE.  Pass the result directly to
    :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="anima-preview2.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="text_encoders/qwen_3_06b_base.safetensors",
            dest=_CLIP_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="vae/qwen_image_vae.safetensors",
            dest=_VAE_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = _DEFAULT_STEPS,
    cfg: float = _DEFAULT_CFG,
    seed: int = 0,
    unet_filename: str | None = None,
    clip_filename: str | None = None,
    vae_filename: str | None = None,
) -> list[Any]:
    """Run the Anima Preview text-to-image pipeline end-to-end.

    Loads the Anima diffusion model, Qwen text encoder, and Qwen image VAE,
    encodes the prompt, creates an empty latent, runs the ``er_sde`` sampler,
    and decodes the result to a PIL image.

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
        Number of denoising steps.  Default ``30``.
    cfg : float, optional
        CFG guidance scale.  Default ``4.0``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    unet_filename : str | None, optional
        Override the default diffusion model filename.  Default ``None``.
    clip_filename : str | None, optional
        Override the default text encoder filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the default VAE filename.  Default ``None``.

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
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    # Resolve model paths (allow caller overrides, fall back to manifest paths).
    unet_path = models_dir / (unet_filename or _UNET_DEST)
    clip_path = models_dir / (clip_filename or _CLIP_DEST)
    vae_path = models_dir / (vae_filename or _VAE_DEST)

    # Load models.
    model = mm.load_unet(str(unet_path))
    clip = mm.load_clip(str(clip_path), clip_type="stable_diffusion")
    vae = mm.load_vae(str(vae_path))

    # Encode text conditioning.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    # Create starting latent.
    latent = empty_latent_image(width, height, batch_size=1)

    # Denoise with the er_sde sampler.
    latent_out = sample(
        model,
        positive,
        negative,
        latent,
        steps,
        cfg,
        _DEFAULT_SAMPLER,
        _DEFAULT_SCHEDULER,
        seed,
    )

    # Decode latent to image.
    image = vae_decode(vae, latent_out)

    return [image]
