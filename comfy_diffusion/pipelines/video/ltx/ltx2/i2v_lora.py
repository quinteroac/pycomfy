"""LTX-Video 2 image-to-video pipeline with a caller-supplied style LoRA (audio-visual).

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  dual LoRA application, conditioning, AV sampling, latent upsampling, and
  decoding.

The LoRA image-to-video variant uses:

- The dev fp8 UNet checkpoint (``ltx-2-19b-dev-fp8.safetensors``) with two LoRAs
  applied after loading — the base distilled LoRA first, then the
  caller-supplied style LoRA.
- A Gemma 3 12B text encoder loaded via
  :meth:`~comfy_diffusion.models.ModelManager.load_ltxav_text_encoder`.
- The input image loaded directly (no ``ltxv_preprocess`` resize step) and
  injected into the latent via
  :func:`~comfy_diffusion.video.ltxv_img_to_video_inplace`.
- Guide cropping via ``ltxv_crop_guides`` after image injection.
- A spatial latent upsampler (``ltx-2-spatial-upscaler-x2-1.0.safetensors``)
  applied after sampling and before VAE decoding.
- The full audio-visual (AV) sampling chain — video and audio generated together.
- Default output resolution of 1280×1280 (square) matching the reference workflow.

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
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v_lora import manifest, run

    # 1. Download base models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference with a style LoRA.
    result = run(
        models_dir="/path/to/models",
        image="/path/to/input.png",
        prompt="the waitress smiles and turns her head",
        lora_path="/path/to/style.safetensors",
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
# Canonical HuggingFace repository for LTX-Video 2
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
    """Return the list of base model files required by the LTX-Video 2 I2V LoRA pipeline.

    The style LoRA is caller-supplied and is not included here.  Each entry is
    an :class:`~comfy_diffusion.downloader.HFModelEntry` that resolves to a
    deterministic relative path under ``models_dir``.  Pass the result
    directly to :func:`~comfy_diffusion.downloader.download_models`::

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
    lora_path: str | Path,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int = 1280,
    height: int = 1280,
    length: int = 97,
    fps: int = 24,
    steps: int = 12,
    cfg: float = 1.0,
    scheduler: str = "lcm",
    seed: int = 0,
    lora_strength: float = 1.0,
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    audio_vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
    base_lora_filename: str | None = None,
    upscaler_filename: str | None = None,
) -> dict[str, Any]:
    """Run the LTX-Video 2 image-to-video-with-audio pipeline with a caller-supplied style LoRA.

    Two-pass sampling mirroring the ``video_ltx2_i2v_lora.json`` reference workflow:
    both the distilled LoRA and the style LoRA are applied throughout.  Pass 1 uses
    ``LTXVScheduler`` at half resolution; pass 2 refines at full resolution.  The
    image is only injected in pass 1 — pass 2 starts from the upscaled latent directly.

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
        Output frame width in pixels.  The latent is sampled at half this size
        and the spatial upscaler restores the full resolution.  Default ``1280``.
    height : int, optional
        Output frame height in pixels.  Default ``1280`` (square, matching the
        reference workflow).
    length : int, optional
        Number of video frames to generate.  Default ``97``.
    fps : int, optional
        Frame rate used for the audio latent and ``LTXVConditioning``.
        Default ``24``.
    steps : int, optional
        Number of denoising steps.  Default ``12``.
    cfg : float, optional
        Classifier-free guidance scale.  Default ``1.0``.
    scheduler : str, optional
        Scheduler name passed to ``basic_scheduler``.  Default ``"lcm"``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    lora_strength : float, optional
        Strength of the caller-supplied style LoRA.  Default ``1.0``.
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
    base_lora_filename : str | None, optional
        Override the default base distilled LoRA filename.  Default ``None``.
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
    from comfy_diffusion.image import image_to_tensor, load_image
    from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import basic_scheduler, cfg_guider, get_sampler, random_noise, sample_custom
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
    vae_path = Path(vae_filename) if vae_filename else unet_path
    audio_vae_path = Path(audio_vae_filename) if audio_vae_filename else vae_path
    style_lora_path = Path(lora_path)

    # Load models.
    model = mm.load_unet(unet_path)
    vae = mm.load_vae(vae_path)
    audio_vae = mm.load_ltxv_audio_vae(audio_vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)
    upscale_model = mm.load_latent_upscale_model(upscaler_path)

    # Apply base distilled LoRA first, then caller-supplied style LoRA (stacked).
    model, clip = apply_lora(model, clip, base_lora_path, 1.0, 1.0)
    model, clip = apply_lora(model, clip, style_lora_path, lora_strength, lora_strength)

    # Load input image and convert to BHWC tensor.
    if hasattr(image, "mode"):
        image_tensor = image_to_tensor(image)
    else:
        image_tensor, _ = load_image(image)

    # Text conditioning + frame-rate metadata.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    # Build video latent, inject image, crop guides.
    video_latent = ltxv_empty_latent_video(width=width, height=height, length=length)
    video_latent = ltxv_img_to_video_inplace(vae, image_tensor, video_latent)
    positive, negative, video_latent = ltxv_crop_guides(positive, negative, video_latent)

    # Build AV latent and sample.
    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    sigmas = basic_scheduler(model, scheduler, steps)
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
