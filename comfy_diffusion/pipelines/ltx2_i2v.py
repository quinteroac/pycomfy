"""LTX-Video 2 image-to-video pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  LoRA application, image preprocessing, conditioning, sampling, latent
  upsampling, and VAE decoding.

The image-to-video variant uses:

- The dev fp8 UNet checkpoint (``ltx-2-19b-dev-fp8.safetensors``) with a
  distilled LoRA (``ltx-2-19b-distilled-lora-384.safetensors``) applied after
  loading for fast inference.
- A Gemma 3 12B text encoder loaded via
  :meth:`~comfy_diffusion.models.ModelManager.load_ltxav_text_encoder`.
- An input image preprocessed with
  :func:`~comfy_diffusion.image.ltxv_preprocess` and injected into the latent
  via :func:`~comfy_diffusion.video.ltxv_img_to_video_inplace`.
- A spatial latent upsampler (``ltx-2-spatial-upscaler-x2-1.0.safetensors``)
  applied after sampling and before VAE decoding.

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
    from comfy_diffusion.pipelines.ltx2_i2v import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    frames = run(
        models_dir="/path/to/models",
        image="/path/to/input.png",
        prompt="the waitress smiles and turns her head",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# Canonical HuggingFace repository for LTX-Video 2
# ---------------------------------------------------------------------------

_HF_REPO = "Lightricks/LTX-Video"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "ltx-2-19b-dev-fp8.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "gemma_3_12B_it_fp4_mixed.safetensors"
_LORA_DEST = Path("loras") / "ltx-2-19b-distilled-lora-384.safetensors"
_UPSCALER_DEST = Path("upscale_models") / "ltx-2-spatial-upscaler-x2-1.0.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2 I2V pipeline.

    Each entry is an :class:`~comfy_diffusion.downloader.HFModelEntry` that
    resolves to a deterministic relative path under ``models_dir``.  Pass the
    result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="ltx-2-19b-dev-fp8.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="gemma_3_12B_it_fp4_mixed.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="ltx-2-19b-distilled-lora-384.safetensors",
            dest=_LORA_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            dest=_UPSCALER_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    image: Any,
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int = 1280,
    height: int = 720,
    length: int = 121,
    steps: int = 8,
    cfg: float = 3.0,
    seed: int = 0,
    sampler: str = "euler",
    scheduler: str = "beta",
    lora_strength: float = 1.0,
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
    lora_filename: str | None = None,
    upscaler_filename: str | None = None,
) -> list[Any]:
    """Run the LTX-Video 2 image-to-video pipeline end-to-end.

    Parameters
    ----------
    models_dir:
        Root directory where model weights are stored.
    image:
        Input image.  Accepts a file path (``str`` or :class:`~pathlib.Path`)
        or a :class:`~PIL.Image.Image` instance.
    prompt:
        Positive text prompt describing the desired video content.
    negative_prompt:
        Negative text prompt.  Defaults to a standard quality-rejection string.
    width:
        Output frame width in pixels (default 1280; must be divisible by 32).
    height:
        Output frame height in pixels (default 720; must be divisible by 32).
    length:
        Number of video frames to generate (default 121 ≈ ~5 s at 24 fps;
        must be divisible by 8 + 1).
    steps:
        Number of denoising steps.  Default is 8 (distilled LoRA).
    cfg:
        Classifier-free guidance scale.
    seed:
        Random seed for reproducibility.
    sampler:
        Sampler name passed to :func:`~comfy_diffusion.sampling.sample`.
    scheduler:
        Noise scheduler name.
    lora_strength:
        Strength applied to both the model and CLIP components of the LoRA.
        Default is 1.0.
    unet_filename:
        Override the default UNet filename (relative to ``models_dir`` or
        absolute).  When ``None`` the path from :func:`manifest` is used.
    vae_filename:
        Override the VAE filename.  When ``None`` the VAE is loaded from the
        UNet checkpoint path (the dev fp8 checkpoint bundles both UNet and
        VAE weights).
    text_encoder_filename:
        Override the default text-encoder filename.
    lora_filename:
        Override the default LoRA filename.
    upscaler_filename:
        Override the default spatial upscaler filename.

    Returns
    -------
    list[PIL.Image.Image]
        Decoded video frames as PIL images, one per generated frame.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.image import image_to_tensor, load_image, ltxv_preprocess
    from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode_batch_tiled
    from comfy_diffusion.video import ltxv_img_to_video_inplace

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    # Resolve model paths (allow caller overrides, fall back to manifest paths).
    unet_path = Path(unet_filename) if unet_filename else models_dir / _UNET_DEST
    te_path = (
        Path(text_encoder_filename)
        if text_encoder_filename
        else models_dir / _TEXT_ENCODER_DEST
    )
    lora_path = Path(lora_filename) if lora_filename else models_dir / _LORA_DEST
    upscaler_path = (
        Path(upscaler_filename) if upscaler_filename else models_dir / _UPSCALER_DEST
    )
    # The dev fp8 checkpoint bundles VAE weights; default to the same file.
    vae_path = Path(vae_filename) if vae_filename else unet_path

    # Load models.
    model = mm.load_unet(unet_path)
    vae = mm.load_vae(vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)
    upscale_model = mm.load_latent_upscale_model(upscaler_path)

    # Apply distilled LoRA after model load.
    model, clip = apply_lora(model, clip, lora_path, lora_strength, lora_strength)

    # Load input image and convert to BHWC tensor.
    if hasattr(image, "mode"):
        image_tensor = image_to_tensor(image)
    else:
        image_tensor, _ = load_image(image)

    # Preprocess image for LTXV (center resize + compression).
    preprocessed = ltxv_preprocess(image_tensor, width, height)

    # Text conditioning.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    # Create empty latent.
    latent = ltxv_empty_latent_video(width=width, height=height, length=length)

    # Inject the preprocessed image frame into the latent.
    latent = ltxv_img_to_video_inplace(vae, preprocessed, latent)

    # Sample.
    samples = sample(
        model=model,
        positive=positive,
        negative=negative,
        latent_image=latent,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler=scheduler,
        seed=seed,
    )

    # Spatial upscale in latent space before VAE decode.
    samples = ltxv_latent_upsample(samples, upscale_model=upscale_model, vae=vae)

    # Decode latent → PIL frames.
    frames = vae_decode_batch_tiled(vae, samples)
    return frames
