"""LTX-Video 2 image-to-video pipeline with a caller-supplied style LoRA.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  dual LoRA application, conditioning, sampling, latent upsampling, and VAE
  decoding.

The LoRA image-to-video variant uses:

- The dev UNet checkpoint (``ltx-2-19b-dev.safetensors``) with two LoRAs
  applied after loading — the base distilled LoRA first, then the
  caller-supplied style LoRA.
- A Gemma 3 12B text encoder loaded via
  :meth:`~comfy_diffusion.models.ModelManager.load_ltxav_text_encoder`.
- The input image loaded directly (no ``ltxv_preprocess`` resize step) and
  injected into the latent via
  :func:`~comfy_diffusion.video.ltxv_img_to_video_inplace`.
- A spatial latent upsampler (``ltx-2-spatial-upscaler-x2-1.0.safetensors``)
  applied after sampling and before VAE decoding.
- Default output resolution of 1280×1280 (square) matching the reference
  workflow.

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
    from comfy_diffusion.pipelines.ltx2_i2v_lora import manifest, run

    # 1. Download base models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference with a style LoRA.
    frames = run(
        models_dir="/path/to/models",
        image="/path/to/input.png",
        prompt="the waitress smiles and turns her head",
        lora_path="/path/to/style.safetensors",
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
_UNET_DEST = Path("diffusion_models") / "ltx-2-19b-dev.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "gemma_3_12B_it_fp4_mixed.safetensors"
_LORA_DEST = Path("loras") / "ltx-2-19b-distilled-lora-384.safetensors"
_UPSCALER_DEST = Path("upscale_models") / "ltx-2-spatial-upscaler-x2-1.0.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of base model files required by the LTX-Video 2 I2V LoRA pipeline.

    The style LoRA is caller-supplied and is not included here.  Each entry is
    an :class:`~comfy_diffusion.downloader.HFModelEntry` that resolves to a
    deterministic relative path under ``models_dir``.  Pass the result
    directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="ltx-2-19b-dev.safetensors",
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
    lora_path: str | Path,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int = 1280,
    height: int = 1280,
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
    base_lora_filename: str | None = None,
    upscaler_filename: str | None = None,
) -> list[Any]:
    """Run the LTX-Video 2 image-to-video pipeline with a caller-supplied style LoRA.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    image : str | Path | PIL.Image.Image
        Input image.  Accepts a file path (``str`` or :class:`~pathlib.Path`)
        or a :class:`~PIL.Image.Image` instance.
    prompt : str
        Positive text prompt describing the desired video content.
    lora_path : str | Path
        Path to the caller-supplied style LoRA weights file.
    negative_prompt : str, optional
        Negative text prompt.
        Default ``"worst quality, inconsistent motion, blurry, jittery, distorted"``.
    width : int, optional
        Output frame width in pixels (must be divisible by 32).  Default ``1280``.
    height : int, optional
        Output frame height in pixels (must be divisible by 32).  Default ``1280``.
    length : int, optional
        Number of video frames to generate (≈ ~5 s at 24 fps; must be
        divisible by 8 + 1).  Default ``121``.
    steps : int, optional
        Number of denoising steps.  Default ``8`` (distilled LoRA).
    cfg : float, optional
        Classifier-free guidance scale.  Default ``3.0``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    sampler : str, optional
        Sampler name passed to :func:`~comfy_diffusion.sampling.sample`.
        Default ``"euler"``.
    scheduler : str, optional
        Noise scheduler name.  Default ``"beta"``.
    lora_strength : float, optional
        Strength applied to both the model and CLIP components of the
        caller-supplied style LoRA.  Default ``1.0``.
    unet_filename : str | None, optional
        Override the default UNet filename (relative to ``models_dir`` or
        absolute).  When ``None`` the path from :func:`manifest` is used.
        Default ``None``.
    vae_filename : str | None, optional
        Override the VAE filename.  When ``None`` the VAE is loaded from the
        UNet checkpoint path (the dev checkpoint bundles both UNet and VAE
        weights).  Default ``None``.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.  Default ``None``.
    base_lora_filename : str | None, optional
        Override the default base distilled LoRA filename.  Default ``None``.
    upscaler_filename : str | None, optional
        Override the default spatial upscaler filename.  Default ``None``.

    Returns
    -------
    list[PIL.Image.Image]
        Decoded video frames as PIL images, one per generated frame.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.image import image_to_tensor, load_image
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
    base_lora_path = (
        Path(base_lora_filename) if base_lora_filename else models_dir / _LORA_DEST
    )
    upscaler_path = (
        Path(upscaler_filename) if upscaler_filename else models_dir / _UPSCALER_DEST
    )
    # The dev checkpoint bundles VAE weights; default to the same file.
    vae_path = Path(vae_filename) if vae_filename else unet_path
    style_lora_path = Path(lora_path)

    # Load models.
    model = mm.load_unet(unet_path)
    vae = mm.load_vae(vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)
    upscale_model = mm.load_latent_upscale_model(upscaler_path)

    # Apply base distilled LoRA first, then caller-supplied style LoRA (stacked).
    model, clip = apply_lora(model, clip, base_lora_path, 1.0, 1.0)
    model, clip = apply_lora(model, clip, style_lora_path, lora_strength, lora_strength)

    # Load input image and convert to BHWC tensor (no ltxv_preprocess resize).
    if hasattr(image, "mode"):
        image_tensor = image_to_tensor(image)
    else:
        image_tensor, _ = load_image(image)

    # Text conditioning.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    # Create empty latent.
    latent = ltxv_empty_latent_video(width=width, height=height, length=length)

    # Inject the image frame into the latent.
    latent = ltxv_img_to_video_inplace(vae, image_tensor, latent)

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
