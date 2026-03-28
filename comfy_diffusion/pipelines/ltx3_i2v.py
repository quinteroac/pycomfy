"""LTX-Video 2.3 (22B distilled) image-to-video pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  image preprocessing, conditioning, sampling, latent upsampling, and VAE
  decoding.

Key differences from ``ltx3_t2v``:

- Accepts an ``image`` parameter (file path, ``str``, or
  :class:`~PIL.Image.Image`) as the first conditioning frame.
- Accepts an explicit ``fps`` parameter (default 24).  This value is reserved
  for future use — :func:`~comfy_diffusion.latent.ltxv_empty_latent_video`
  does not currently expose an fps argument; the parameter is accepted so
  callers can document their intent and the API will honour it automatically
  once upstream support is added.
- The preprocessed image is injected into the latent via
  :func:`~comfy_diffusion.video.ltxv_img_to_video_inplace` (same mechanism as
  ``ltx2_i2v``).

Models required are identical to ``ltx3_t2v`` — no extra LoRA is needed
because the 22B distilled checkpoint is already optimised for fast inference.

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
    from comfy_diffusion.pipelines.ltx3_i2v import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    frames = run(
        models_dir="/path/to/models",
        image="/path/to/first_frame.png",
        prompt="the queen turns her head slowly towards the camera",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# Canonical HuggingFace repository for LTX-Video
# ---------------------------------------------------------------------------

_HF_REPO = "Lightricks/LTX-Video"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "ltx-2.3-22b-distilled-fp8.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "gemma_3_12B_it_fp4_mixed.safetensors"
_UPSCALER_DEST = Path("upscale_models") / "ltx-2-spatial-upscaler-x2-1.0.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2.3 I2V pipeline.

    Returns the same three entries as :func:`comfy_diffusion.pipelines.ltx3_t2v.manifest`:
    the 22B distilled UNet checkpoint, the Gemma 3 12B text encoder, and the
    spatial latent upscaler.

    Each entry is an :class:`~comfy_diffusion.downloader.HFModelEntry` that
    resolves to a deterministic relative path under ``models_dir``.  Pass the
    result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="ltx-2.3-22b-distilled-fp8.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="gemma_3_12B_it_fp4_mixed.safetensors",
            dest=_TEXT_ENCODER_DEST,
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
    width: int = 768,
    height: int = 512,
    length: int = 97,
    fps: int = 24,
    steps: int = 8,
    cfg: float = 3.0,
    seed: int = 0,
    sampler: str = "euler",
    scheduler: str = "beta",
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
    upscaler_filename: str | None = None,
) -> list[Any]:
    """Run the LTX-Video 2.3 (22B distilled) image-to-video pipeline end-to-end.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    image : str | Path | PIL.Image.Image
        Input image for the first conditioning frame.  Accepts a file path
        (``str`` or :class:`~pathlib.Path`) or a :class:`~PIL.Image.Image`
        instance.
    prompt : str
        Positive text prompt describing the desired video content.
    negative_prompt : str, optional
        Negative text prompt.
        Default ``"worst quality, inconsistent motion, blurry, jittery, distorted"``.
    width : int, optional
        Output frame width in pixels (must be divisible by 32).  Default ``768``.
    height : int, optional
        Output frame height in pixels (must be divisible by 32).  Default ``512``.
    length : int, optional
        Number of video frames to generate (≈ ~4 s at 24 fps).  Default ``97``.
    fps : int, optional
        Target frame rate of the generated video.  Default ``24``.  Currently
        reserved — :func:`~comfy_diffusion.latent.ltxv_empty_latent_video`
        does not yet expose an fps argument; it will be forwarded automatically
        once upstream support is added.
    steps : int, optional
        Number of denoising steps.  Default ``8`` (distilled model).
    cfg : float, optional
        Classifier-free guidance scale.  Default ``3.0``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    sampler : str, optional
        Sampler name passed to :func:`~comfy_diffusion.sampling.sample`.
        Default ``"euler"``.
    scheduler : str, optional
        Noise scheduler name.  Default ``"beta"``.
    unet_filename : str | None, optional
        Override the default UNet filename (relative to ``models_dir`` or
        absolute).  When ``None`` the path from :func:`manifest` is used.
        Default ``None``.
    vae_filename : str | None, optional
        Override the VAE filename.  When ``None`` the VAE is loaded from the
        UNet checkpoint path (the distilled checkpoint bundles both UNet and
        VAE weights).  Default ``None``.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.  Default ``None``.
    upscaler_filename : str | None, optional
        Override the default spatial upscaler filename.  Default ``None``.

    Returns
    -------
    list[PIL.Image.Image]
        Decoded video frames as PIL images, one per generated frame.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.image import image_to_tensor, load_image, ltxv_preprocess
    from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample
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
    upscaler_path = (
        Path(upscaler_filename) if upscaler_filename else models_dir / _UPSCALER_DEST
    )
    # The distilled checkpoint bundles VAE weights; default to the same file.
    vae_path = Path(vae_filename) if vae_filename else unet_path

    # Load models.
    model = mm.load_unet(unet_path)
    vae = mm.load_vae(vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)
    upscale_model = mm.load_latent_upscale_model(upscaler_path)

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
    # NOTE: fps is reserved — ltxv_empty_latent_video does not yet accept fps.
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
