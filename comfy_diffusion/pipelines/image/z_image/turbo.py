"""Z-Image Turbo distilled text-to-image pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes distilled text-to-image inference end-to-end: model
  loading, conditioning with the Qwen3-4B text encoder, AuraFlow model
  sampling patch, and VAE decoding.

This pipeline mirrors the Z-Image Turbo workflow: the ``ModelSamplingAuraFlow``
patch (shift=3) is applied before sampling, ``res_multistep`` is the sampler,
``simple`` is the scheduler, and CFG is 1.0 (turbo — effectively guidance-free).
Negative conditioning is produced by zeroing out the positive conditioning via
``conditioning_zero_out``.

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
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    images = run(
        models_dir="/path/to/models",
        prompt="Latina female with thick wavy hair, harbor boats behind",
    )
    image = images[0]  # PIL.Image.Image (1024×1024)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repository for Z-Image Turbo
# ---------------------------------------------------------------------------

_HF_REPO = "Comfy-Org/z_image_turbo"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "z_image_turbo_bf16.safetensors"
_CLIP_DEST = Path("text_encoders") / "qwen_3_4b.safetensors"
_VAE_DEST = Path("vae") / "ae.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the Z-Image Turbo pipeline.

    Returns exactly 3 :class:`~comfy_diffusion.downloader.HFModelEntry` items:

    - ``diffusion_models/z_image_turbo_bf16.safetensors`` — UNet diffusion model
    - ``text_encoders/qwen_3_4b.safetensors`` — Qwen3-4B text encoder (lumina2)
    - ``vae/ae.safetensors`` — AE variational autoencoder

    All files are sourced from the ``Comfy-Org/z_image_turbo`` HF repo.
    Pass the result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="split_files/diffusion_models/z_image_turbo_bf16.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="split_files/text_encoders/qwen_3_4b.safetensors",
            dest=_CLIP_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="split_files/vae/ae.safetensors",
            dest=_VAE_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    seed: int = 0,
) -> list[Any]:
    """Run the Z-Image Turbo text-to-image pipeline end-to-end.

    Distilled single-pass sampling using the ``res_multistep`` sampler and
    ``simple`` scheduler with CFG 1.0.  The Qwen3-4B text encoder is loaded
    with the ``lumina2`` clip type.  Negative conditioning is produced by
    zeroing out the positive conditioning (``conditioning_zero_out``).

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    prompt : str
        Positive text prompt describing the desired image content.
    width : int, optional
        Output image width in pixels.  Default ``1024``.
    height : int, optional
        Output image height in pixels.  Default ``1024``.
    steps : int, optional
        Number of denoising steps.  Default ``4``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.

    Returns
    -------
    list[PIL.Image.Image]
        A list containing the generated image (one element for batch size 1).
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.conditioning import conditioning_zero_out, encode_prompt
    from comfy_diffusion.latent import empty_sd3_latent_image
    from comfy_diffusion.models import ModelManager, model_sampling_aura_flow
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

    # Load all three models independently (no combined checkpoint).
    model = mm.load_unet(_UNET_DEST.name)
    clip = mm.load_clip(_CLIP_DEST.name, clip_type="lumina2")
    vae = mm.load_vae(_VAE_DEST.name)

    # Apply AuraFlow model sampling patch (shift=3) before sampling.
    model = model_sampling_aura_flow(model, shift=3)

    # Encode positive prompt; derive negative by zeroing out the conditioning.
    positive, _ = encode_prompt(clip, prompt, "")
    negative = conditioning_zero_out(positive)

    # Create SD3-family latent (16-channel, spatial factor 8).
    latent = empty_sd3_latent_image(width, height, batch_size=1)

    # Run distilled sampling — CFG 1.0 (turbo / guidance-free effectively).
    latent_out = sample(
        model,
        positive,
        negative,
        latent,
        steps,
        1.0,
        "res_multistep",
        "simple",
        seed,
    )

    # Decode final latent.
    image = vae_decode(vae, latent_out)

    return [image]
