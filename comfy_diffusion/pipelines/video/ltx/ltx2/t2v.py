"""LTX-Video 2 text-to-video pipeline (audio-visual).

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  LoRA application, conditioning, AV sampling, latent upsampling, and decoding.

This pipeline mirrors the ``video_ltx2_t2v.json`` reference workflow exactly:
two-pass sampling with the distilled LoRA applied only in pass 2.

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
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    result = run(
        models_dir="/path/to/models",
        prompt="a golden retriever running through a sunlit park",
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
    """Return the list of model files required by the LTX-Video 2 T2V pipeline.

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
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int = 1280,
    height: int = 720,
    length: int = 97,
    fps: int = 24,
    steps: int = 20,
    cfg_pass1: float = 4.0,
    cfg_pass2: float = 1.0,
    seed: int = 0,
    lora_strength: float = 1.0,
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    audio_vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
    lora_filename: str | None = None,
    upscaler_filename: str | None = None,
) -> dict[str, Any]:
    """Run the LTX-Video 2 text-to-video-with-audio pipeline end-to-end.

    Two-pass sampling mirroring the ``video_ltx2_t2v.json`` reference workflow:
    pass 1 uses the base model at half resolution with ``LTXVScheduler`` +
    ``euler_ancestral``; pass 2 applies the distilled LoRA at full resolution
    with ``ManualSigmas`` + ``euler_ancestral``.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    prompt : str
        Positive text prompt describing the desired video content.
    negative_prompt : str, optional
        Negative text prompt.
        Default ``"worst quality, inconsistent motion, blurry, jittery, distorted"``.
    width : int, optional
        Output frame width in pixels.  The latent is sampled at half this size
        and the spatial upscaler restores the full resolution.  Default ``1280``.
    height : int, optional
        Output frame height in pixels.  Same halving logic as ``width``.
        Default ``720``.
    length : int, optional
        Number of video frames to generate.  Default ``97``.
    fps : int, optional
        Frame rate used for the audio latent and ``LTXVConditioning``.
        Default ``25``.
    steps : int, optional
        Number of denoising steps for pass 1.  Default ``20``.
    cfg_pass1 : float, optional
        CFG scale for pass 1 (base model, half resolution).  Default ``4.0``.
    cfg_pass2 : float, optional
        CFG scale for pass 2 (distilled LoRA, full resolution).  Default ``1.0``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    lora_strength : float, optional
        Strength of the distilled LoRA applied in pass 2.  Default ``1.0``.
    unet_filename : str | None, optional
        Override the default UNet filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the VAE filename.  When ``None`` the VAE is loaded from the
        UNet checkpoint.  Default ``None``.
    audio_vae_filename : str | None, optional
        Override the audio VAE filename.  Default ``None``.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.  Default ``None``.
    lora_filename : str | None, optional
        Override the default LoRA filename.  Default ``None``.
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
    from comfy_diffusion.conditioning import encode_prompt, ltxv_conditioning
    from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import cfg_guider, get_sampler, ltxv_scheduler, manual_sigmas, random_noise, sample_custom
    from comfy_diffusion.vae import vae_decode_batch_tiled

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
        Path(text_encoder_filename) if text_encoder_filename else models_dir / _TEXT_ENCODER_DEST
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

    # Text conditioning + frame-rate metadata.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    # Build latents.
    video_latent = ltxv_empty_latent_video(width=width, height=height, length=length)
    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    sampler_obj = get_sampler("euler_ancestral")

    # Pass 1: base model, LTXVScheduler, half-resolution.
    sigmas1 = ltxv_scheduler(steps, 2.05, 0.95, latent=av_latent)
    guider1 = cfg_guider(model, positive, negative, cfg_pass1)
    noise = random_noise(seed)
    _, denoised1 = sample_custom(noise, guider1, sampler_obj, sigmas1, av_latent)

    # Separate AV latent, spatially upsample video, re-concat for pass 2.
    video_latent_out1, audio_latent_out1 = ltxv_separate_av_latent(denoised1)
    video_latent_up = ltxv_latent_upsample(video_latent_out1, upscale_model=upscale_model, vae=vae)
    av_latent2 = ltxv_concat_av_latent(video_latent_up, audio_latent_out1)

    # Pass 2: distilled LoRA model, ManualSigmas, full resolution.
    model_lora, _ = apply_lora(model, clip, lora_path, lora_strength, lora_strength)
    sigmas2 = manual_sigmas("0.909375, 0.725, 0.421875, 0.0")
    guider2 = cfg_guider(model_lora, positive, negative, cfg_pass2)
    noise2 = random_noise(seed)
    _, denoised2 = sample_custom(noise2, guider2, sampler_obj, sigmas2, av_latent2)

    # Decode video and audio from pass-2 output.
    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised2)
    frames = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)

    return {"frames": frames, "audio": audio}
