"""Flux.2 Klein 4B base text-to-image pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  text conditioning, latent creation, custom sampling with the ``euler``
  sampler and ``Flux2Scheduler``, and VAE decoding.

This pipeline mirrors the *Text to Image (Flux.2 Klein 4B)* subgraph in
``comfyui_official_workflows/image/generation/flux_klein/
image_flux2_klein_text_to_image.json``:

``UNETLoader → CLIPLoader → VAELoader → RandomNoise →
CLIPTextEncode (positive) → CLIPTextEncode (negative empty) →
EmptyFlux2LatentImage → KSamplerSelect (euler) → Flux2Scheduler →
CFGGuider (cfg=5) → SamplerCustomAdvanced → VAEDecode``

Usage
-----
::

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    images = run(
        models_dir="/path/to/models",
        prompt="a photo of a cat sitting on a chair",
    )
    image = images[0]  # PIL.Image.Image (1024×1024)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repositories for Flux.2 Klein
# ---------------------------------------------------------------------------

_HF_REPO_KLEIN = "Comfy-Org/flux2-klein"
_HF_REPO_DEV = "Comfy-Org/flux2-dev"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "flux-2-klein-base-4b.safetensors"
_CLIP_DEST = Path("text_encoders") / "qwen_3_4b.safetensors"
_VAE_DEST = Path("vae") / "flux2-vae.safetensors"

# Workflow defaults.
_DEFAULT_STEPS = 20
_DEFAULT_CFG = 5.0
_DEFAULT_SAMPLER = "euler"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the Flux.2 Klein 4B base T2I pipeline.

    Returns exactly 3 :class:`~comfy_diffusion.downloader.HFModelEntry`
    instances:

    - ``diffusion_models/flux-2-klein-base-4b.safetensors`` — Flux.2 Klein 4B base UNet
    - ``text_encoders/qwen_3_4b.safetensors`` — Qwen 3 4B text encoder
    - ``vae/flux2-vae.safetensors`` — Flux 2 VAE

    Pass the result directly to
    :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_KLEIN,
            filename="split_files/diffusion_models/flux-2-klein-base-4b.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_KLEIN,
            filename="split_files/text_encoders/qwen_3_4b.safetensors",
            dest=_CLIP_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_DEV,
            filename="split_files/vae/flux2-vae.safetensors",
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
    """Run the Flux.2 Klein 4B base text-to-image pipeline end-to-end.

    Loads the Flux.2 Klein 4B base UNet, Qwen 3 4B text encoder (flux2 type),
    and Flux 2 VAE, encodes the prompt, creates an empty Flux 2 latent, runs
    the ``euler`` sampler with ``Flux2Scheduler`` and ``CFGGuider``, and
    decodes the result to a PIL image.

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
        Number of denoising steps.  Default ``20``.
    cfg : float, optional
        CFG guidance scale.  Default ``5.0``.
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
    from comfy_diffusion.latent import empty_flux2_latent_image
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import (
        cfg_guider,
        flux2_scheduler,
        get_sampler,
        random_noise,
        sample_custom,
    )
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

    # Load all three models independently.
    model = mm.load_unet(str(unet_path))
    clip = mm.load_clip(str(clip_path), clip_type="flux2")
    vae = mm.load_vae(str(vae_path))

    # Encode positive and negative text conditioning.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    # Create empty Flux 2 latent (16-channel, spatial factor 8).
    latent = empty_flux2_latent_image(width, height, batch_size=1)

    # Build sampling primitives following the workflow node order.
    noise = random_noise(seed)
    sampler = get_sampler(_DEFAULT_SAMPLER)
    sigmas = flux2_scheduler(steps, width, height)
    guider = cfg_guider(model, positive, negative, cfg)

    # Run SamplerCustomAdvanced.
    latent_out, _ = sample_custom(noise, guider, sampler, sigmas, latent)

    # Decode latent to image.
    image = vae_decode(vae, latent_out)

    return [image]
