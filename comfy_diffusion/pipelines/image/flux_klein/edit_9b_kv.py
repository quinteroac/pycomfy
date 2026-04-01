"""Flux.2 Klein 9B KV image-edit pipeline (dual-image reference conditioning).

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  KV-cache patching, dual-image scaling, text conditioning, reference-image
  encoding, latent creation, custom sampling with the ``euler`` sampler and
  ``Flux2Scheduler``, and VAE decoding.

This pipeline mirrors ``comfyui_official_workflows/image/editing/flux_klein/
image_flux2_klein_9b_kv_image_edit.json``:

``UNETLoader → FluxKVCache → CLIPLoader → VAELoader →
CLIPTextEncode (positive) → ConditioningZeroOut (negative) →
ImageScaleToTotalPixels (reference image, lanczos, 1MP) →
ImageScaleToTotalPixels (subject image, lanczos, 1MP) →
GetImageSize → EmptyFlux2LatentImage →
VAEEncode (reference) → VAEEncode (subject) →
ReferenceLatent (positive + ref) → ReferenceLatent (positive + subj) →
ReferenceLatent (negative + ref) → ReferenceLatent (negative + subj) →
RandomNoise → KSamplerSelect (euler) → Flux2Scheduler →
CFGGuider (cfg=1) → SamplerCustomAdvanced → VAEDecode``

The 9B KV variant applies ``FluxKVCache`` after model load for accelerated
KV-cache inference, accepts two input images (a reference person/scene and a
subject/style), scales both to 1 megapixel, and attaches both as reference
latents to the conditioning.

Usage
-----
::

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.flux_klein.edit_9b_kv import manifest, run
    from PIL import Image

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    reference = Image.open("person.jpg")
    subject = Image.open("outfit.jpg")
    images = run(
        models_dir="/path/to/models",
        prompt="Have the person wear the outfit from the subject image",
        reference_image=reference,
        subject_image=subject,
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
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8"
    "/resolve/main/flux-2-klein-9b-kv-fp8.safetensors"
)

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "flux-2-klein-9b-kv-fp8.safetensors"
_CLIP_DEST = Path("text_encoders") / "qwen_3_8b_fp8mixed.safetensors"
_VAE_DEST = Path("vae") / "flux2-vae.safetensors"

# Workflow defaults (from Flux2Scheduler params [4, ...] and CFGGuider [1]).
_DEFAULT_STEPS = 4
_DEFAULT_CFG = 1.0
_DEFAULT_SAMPLER = "euler"

# Image scaling: 1 megapixel (lanczos), resolution steps = 1 (from workflow).
_MEGAPIXELS = 1.0
_RESOLUTION_STEPS = 1
_UPSCALE_METHOD = "lanczos"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the Flux.2 Klein 9B KV edit pipeline.

    Returns exactly 3 model entries:

    - ``diffusion_models/flux-2-klein-9b-kv-fp8.safetensors`` — UNet (KV-cache)
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
    reference_image: Any,
    subject_image: Any,
    width: int = 1024,
    height: int = 1024,
    steps: int = _DEFAULT_STEPS,
    cfg: float = _DEFAULT_CFG,
    seed: int = 0,
    unet_filename: str | None = None,
    clip_filename: str | None = None,
    vae_filename: str | None = None,
) -> list[Any]:
    """Run the Flux.2 Klein 9B KV image-edit pipeline end-to-end.

    Loads the Flux.2 Klein 9B KV UNet (with KV-cache patch applied via
    ``flux_kv_cache``), Qwen 3 8B text encoder (flux2 type), and Flux 2 VAE.
    Both input images are scaled to 1 megapixel with ``image_scale_to_total_pixels``,
    encoded into latents, and attached as reference conditionings.  The pipeline
    uses ``conditioning_zero_out`` for the negative, runs the ``euler`` sampler
    with ``Flux2Scheduler`` at ``cfg=1``, and decodes the result.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    prompt : str
        Positive text prompt describing the desired image edit.
    reference_image : PIL.Image.Image
        Primary reference image (e.g. the person/scene to edit).
    subject_image : PIL.Image.Image
        Subject/style image (e.g. the outfit or object to apply).
    width : int, optional
        Target output width hint used to compute megapixels for scaling.
        Default ``1024``.  Actual output dimensions come from
        ``image_scale_to_total_pixels`` applied to ``reference_image``.
    height : int, optional
        Target output height hint.  Default ``1024``.
    steps : int, optional
        Number of denoising steps.  Default ``4``.
    cfg : float, optional
        CFG guidance scale.  Default ``1.0``.
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
    from comfy_diffusion.conditioning import (
        conditioning_zero_out,
        encode_prompt,
        reference_latent,
    )
    from comfy_diffusion.image import (
        get_image_size,
        image_scale_to_total_pixels,
        image_to_tensor,
    )
    from comfy_diffusion.latent import empty_flux2_latent_image
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import (
        cfg_guider,
        flux2_scheduler,
        flux_kv_cache,
        get_sampler,
        random_noise,
        sample_custom,
    )
    from comfy_diffusion.vae import vae_decode, vae_encode_tensor

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

    # Load all three models; apply FluxKVCache patch immediately after UNet load.
    model = mm.load_unet(str(unet_path))
    model = flux_kv_cache(model)
    clip = mm.load_clip(str(clip_path), clip_type="flux2")
    vae = mm.load_vae(str(vae_path))

    # Encode positive prompt; derive negative by zeroing out conditioning.
    positive, _ = encode_prompt(clip, prompt, "")
    negative = conditioning_zero_out(positive)

    # Convert PIL images to ComfyUI IMAGE tensors.
    ref_tensor = image_to_tensor(reference_image)
    subj_tensor = image_to_tensor(subject_image)

    # Scale both images to 1 megapixel (lanczos, resolution_steps=1).
    scaled_ref = image_scale_to_total_pixels(
        ref_tensor, _UPSCALE_METHOD, _MEGAPIXELS, _RESOLUTION_STEPS
    )
    scaled_subj = image_scale_to_total_pixels(
        subj_tensor, _UPSCALE_METHOD, _MEGAPIXELS, _RESOLUTION_STEPS
    )

    # GetImageSize: derive actual output dimensions from the scaled reference.
    actual_width, actual_height = get_image_size(scaled_ref)

    # EmptyFlux2LatentImage at the actual scaled dimensions.
    latent = empty_flux2_latent_image(actual_width, actual_height, batch_size=1)

    # VAEEncode reference and subject image tensors.
    ref_latent = vae_encode_tensor(vae, scaled_ref)
    subj_latent = vae_encode_tensor(vae, scaled_subj)

    # Attach both reference latents to positive conditioning (ReferenceLatent × 2).
    positive = reference_latent(positive, ref_latent)
    positive = reference_latent(positive, subj_latent)

    # Attach both reference latents to negative conditioning (ReferenceLatent × 2).
    negative = reference_latent(negative, ref_latent)
    negative = reference_latent(negative, subj_latent)

    # Build sampling primitives following the workflow node order.
    noise = random_noise(seed)
    sampler = get_sampler(_DEFAULT_SAMPLER)
    sigmas = flux2_scheduler(steps, actual_width, actual_height)
    guider = cfg_guider(model, positive, negative, cfg)

    # Run SamplerCustomAdvanced.
    latent_out, _ = sample_custom(noise, guider, sampler, sigmas, latent)

    # Decode latent to image.
    result = vae_decode(vae, latent_out)

    return [result]
