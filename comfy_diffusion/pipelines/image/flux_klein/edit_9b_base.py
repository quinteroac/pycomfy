"""Flux.2 Klein 9B base image-edit pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  text conditioning, reference-image encoding, latent creation, custom sampling
  with the ``euler`` sampler and ``Flux2Scheduler``, and VAE decoding.

This pipeline mirrors the *Image Edit (Flux.2 Klein 9B)* subgraph in
``comfyui_official_workflows/image/editing/flux_klein/
image_flux2_klein_image_edit_9b_base.json``:

``UNETLoader → CLIPLoader → VAELoader →
CLIPTextEncode (positive) → CLIPTextEncode (negative empty) →
VAEEncode (reference image) →
ReferenceLatent (positive + ref) → ReferenceLatent (negative + ref) →
EmptyFlux2LatentImage → RandomNoise → KSamplerSelect (euler) →
Flux2Scheduler → CFGGuider (cfg=5) → SamplerCustomAdvanced → VAEDecode``

Usage
-----
::

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.flux_klein.edit_9b_base import manifest, run
    from PIL import Image

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    ref = Image.open("photo.jpg")
    images = run(
        models_dir="/path/to/models",
        prompt="change the background to a sunny beach",
        image=ref,
    )
    result = images[0]  # PIL.Image.Image
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry, URLModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# Model source constants
# ---------------------------------------------------------------------------

_HF_REPO_CLIP_9B = "Comfy-Org/flux2-klein-9B"
_HF_REPO_VAE = "Comfy-Org/flux2-dev"

_UNET_URL = (
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8"
    "/resolve/main/flux-2-klein-base-9b-fp8.safetensors"
)

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "flux-2-klein-base-9b-fp8.safetensors"
_CLIP_DEST = Path("text_encoders") / "qwen_3_8b_fp8mixed.safetensors"
_VAE_DEST = Path("vae") / "flux2-vae.safetensors"

# Workflow defaults.
_DEFAULT_STEPS = 20
_DEFAULT_CFG = 5.0
_DEFAULT_SAMPLER = "euler"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the Flux.2 Klein 9B base edit pipeline.

    Returns exactly 3 model entries:

    - ``diffusion_models/flux-2-klein-base-9b-fp8.safetensors`` — UNet
    - ``text_encoders/qwen_3_8b_fp8mixed.safetensors`` — Qwen 3 8B text encoder
    - ``vae/flux2-vae.safetensors`` — Flux 2 VAE

    Pass the result directly to
    :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        URLModelEntry(
            url=_UNET_URL,
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_CLIP_9B,
            filename="split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
            dest=_CLIP_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_VAE,
            filename="split_files/vae/flux2-vae.safetensors",
            dest=_VAE_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    prompt: str,
    image: Any,
    width: int = 1024,
    height: int = 1024,
    steps: int = _DEFAULT_STEPS,
    cfg: float = _DEFAULT_CFG,
    seed: int = 0,
    unet_filename: str | None = None,
    clip_filename: str | None = None,
    vae_filename: str | None = None,
) -> list[Any]:
    """Run the Flux.2 Klein 9B base image-edit pipeline end-to-end.

    Loads the Flux.2 Klein 9B base UNet, Qwen 3 8B text encoder (flux2 type),
    and Flux 2 VAE, encodes the reference image and text prompt, builds
    reference conditioning, creates an empty Flux 2 latent, runs the ``euler``
    sampler with ``Flux2Scheduler`` and ``CFGGuider``, and decodes the result.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    prompt : str
        Positive text prompt describing the desired image edit.
    image : PIL.Image.Image
        Reference image to edit.
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
        A list containing the edited image (one element for batch size 1).
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.conditioning import encode_prompt, reference_latent
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
    from comfy_diffusion.vae import vae_decode, vae_encode

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

    # Load all three models.
    model = mm.load_unet(str(unet_path))
    clip = mm.load_clip(str(clip_path), clip_type="flux2")
    vae = mm.load_vae(str(vae_path))

    # Encode positive prompt and empty negative (CLIPTextEncode '').
    positive, _ = encode_prompt(clip, prompt, "")
    negative, _ = encode_prompt(clip, "", "")

    # VAEEncode reference image → reference latent.
    ref_latent = vae_encode(vae, image)

    # Attach reference latent to both conditionings (ReferenceLatent nodes).
    positive = reference_latent(positive, ref_latent)
    negative = reference_latent(negative, ref_latent)

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
    result = vae_decode(vae, latent_out)

    return [result]
