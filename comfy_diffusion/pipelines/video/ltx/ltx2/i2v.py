"""LTX-Video 2 image-to-video pipeline (audio-visual).

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  LoRA application, image preprocessing, conditioning, AV sampling, latent
  upsampling, and decoding.

The image-to-video variant uses:

- The dev fp8 UNet checkpoint (``ltx-2-19b-dev-fp8.safetensors``) with a
  distilled LoRA (``ltx-2-19b-distilled-lora-384.safetensors``) applied after
  loading for fast inference.
- A Gemma 3 12B text encoder loaded via
  :meth:`~comfy_diffusion.models.ModelManager.load_ltxav_text_encoder`.
- An input image preprocessed with
  :func:`~comfy_diffusion.image.ltxv_preprocess` and injected into the latent
  via :func:`~comfy_diffusion.video.ltxv_img_to_video_inplace`.
- Guide cropping via ``ltxv_crop_guides`` after image injection.
- A spatial latent upsampler (``ltx-2-spatial-upscaler-x2-1.0.safetensors``)
  applied after sampling and before VAE decoding.
- The full audio-visual (AV) sampling chain — video and audio generated together.

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
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    result = run(
        models_dir="/path/to/models",
        image="/path/to/input.png",
        prompt="the waitress smiles and turns her head",
    )
    frames = result["frames"]   # list[PIL.Image.Image]
    audio  = result["audio"]    # {"waveform": tensor, "sample_rate": int}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repositories for LTX-Video 2
# ---------------------------------------------------------------------------

_HF_REPO_LTX = "Lightricks/LTX-Video"      # UNet, LoRA, upscaler, text encoder

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
            repo_id=_HF_REPO_LTX,
            filename="ltx-2-19b-dev-fp8.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
            filename="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
            filename="ltx-2-19b-distilled-lora-384.safetensors",
            dest=_LORA_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
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
    length: int = 97,
    fps: int = 24,
    steps: int = 20,
    cfg: float = 1.0,
    seed: int = 0,
    lora_strength: float = 1.0,
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    audio_vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
    lora_filename: str | None = None,
    upscaler_filename: str | None = None,
) -> dict[str, Any]:
    """Run the LTX-Video 2 image-to-video-with-audio pipeline end-to-end.

    Single-pass sampling: the distilled LoRA is applied first, then the pipeline
    runs one full denoising pass with ``basic_scheduler``.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    image : str | Path | PIL.Image.Image
        Input image.  Accepts a file path (``str`` or :class:`~pathlib.Path`)
        or a :class:`~PIL.Image.Image` instance.
    prompt : str
        Positive text prompt describing the desired video content.
    negative_prompt : str, optional
        Negative text prompt.
        Default ``"worst quality, inconsistent motion, blurry, jittery, distorted"``.
    width : int, optional
        Output frame width in pixels.  Default ``1280``.
    height : int, optional
        Output frame height in pixels.  Default ``720``.
    length : int, optional
        Number of video frames to generate.  Default ``97``.
    fps : int, optional
        Frame rate used for the audio latent and ``LTXVConditioning``.
        Default ``24``.
    steps : int, optional
        Number of denoising steps.  Default ``20``.
    cfg : float, optional
        CFG scale.  Default ``1.0``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    lora_strength : float, optional
        Strength of the distilled LoRA.  Default ``1.0``.
    unet_filename : str | None, optional
        Override the default UNet filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the VAE filename.  When ``None`` the VAE is loaded from the
        UNet checkpoint.  Default ``None``.
    audio_vae_filename : str | None, optional
        Override the audio VAE filename.  When ``None`` falls back to
        ``vae_filename``.  Default ``None``.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.  Default ``None``.
    lora_filename : str | None, optional
        Override the default distilled LoRA filename.  Default ``None``.
    upscaler_filename : str | None, optional
        Override the default spatial upscaler filename.  Default ``None``.

    Returns
    -------
    dict[str, Any]
        ``{"frames": list[PIL.Image.Image], "audio": dict[str, Any]}``

        - ``frames`` — decoded video frames as PIL images, one per frame.
        - ``audio`` — decoded audio as ``{"waveform": tensor, "sample_rate": int}``.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.audio import ltxv_audio_vae_decode, ltxv_concat_av_latent, ltxv_empty_latent_audio, ltxv_separate_av_latent
    from comfy_diffusion.conditioning import encode_prompt, ltxv_conditioning, ltxv_crop_guides
    from comfy_diffusion.image import image_to_tensor, load_image, ltxv_preprocess
    from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import cfg_guider, get_sampler, ltxv_scheduler, random_noise, sample_custom
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
    vae_path = Path(vae_filename) if vae_filename else unet_path
    audio_vae_path = Path(audio_vae_filename) if audio_vae_filename else vae_path

    # Load models.
    # The LTX-Video 2 checkpoint bundles UNet + VAE in a single file, so use
    # load_checkpoint_from_path (mirrors CheckpointLoaderSimple) to extract both.
    ckpt = mm.load_checkpoint_from_path(unet_path)
    model = ckpt.model
    vae = ckpt.vae if vae_filename is None else mm.load_vae(vae_path)
    audio_vae = mm.load_ltxv_audio_vae(audio_vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)
    upscale_model = mm.load_latent_upscale_model(upscaler_path)

    # Apply distilled LoRA.
    model, clip = apply_lora(model, clip, lora_path, lora_strength, lora_strength)

    # Load input image and convert to BHWC tensor, then preprocess.
    if hasattr(image, "mode"):
        image_tensor = image_to_tensor(image)
    else:
        image_tensor, _ = load_image(image)
    preprocessed = ltxv_preprocess(image_tensor, width, height)

    # Text conditioning + frame-rate metadata.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    # Build video latent, inject image, crop guides.
    video_latent = ltxv_empty_latent_video(width=width, height=height, length=length)
    video_latent = ltxv_img_to_video_inplace(vae, preprocessed, video_latent)
    positive, negative, video_latent = ltxv_crop_guides(positive, negative, video_latent)

    # Build AV latent and sample.
    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    sigmas = ltxv_scheduler(steps, 2.05, 0.95, latent=av_latent)
    guider = cfg_guider(model, positive, negative, cfg)
    noise = random_noise(seed)
    sampler_obj = get_sampler("euler_ancestral")
    _, denoised = sample_custom(noise, guider, sampler_obj, sigmas, av_latent)

    # Separate, upsample video, decode.
    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised)
    video_latent_up = ltxv_latent_upsample(video_latent_out, upscale_model=upscale_model, vae=vae)
    frames = vae_decode_batch_tiled(vae, video_latent_up)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)

    return {"frames": frames, "audio": audio}
