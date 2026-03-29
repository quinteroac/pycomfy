"""LTX-Video 2.3 (22B dev fp8) image+audio-to-video pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  LoRA application, image/audio preprocessing, conditioning, two-pass AV
  sampling with latent upscaling, and decoding.

This pipeline mirrors
``comfyui_official_workflows/video/ltx/ltx23/video_ltx2_3_ia2v.json`` and
uses the LTX-Video 2.3 AV (audio-visual) model.  Unlike the i2v pipeline, the
input audio is VAE-encoded and injected into the latent as a conditioning
signal (with a zero noise mask so the sampler does not regenerate it from
scratch).  The model generates video that is jointly conditioned on the
reference image and the audio content.

Sampling chain (two-pass, mirrors the reference workflow exactly):

1. The input image is preprocessed with ``LTXVPreprocess`` (img_compression 18)
   and injected into the empty video latent via ``LTXVImgToVideoInplace``
   (strength ``0.7`` for pass 1).
2. The input audio is loaded, optionally trimmed, and encoded with
   ``LTXVAudioVAEEncode``.  A zero noise mask is applied via
   ``SetLatentNoiseMask`` to preserve the audio signal during denoising.
3. ``LTXVConditioning`` injects frame-rate metadata.
4. The image-conditioned video latent and the masked audio latent are
   concatenated with ``LTXVConcatAVLatent``.
5. **Pass 1** — ``CFGGuider`` + ``ManualSigmas`` (full) +
   ``euler_ancestral_cfg_pp`` drives the first denoising loop.
6. ``LTXVSeparateAVLatent`` splits the pass-1 result into video and audio.
7. ``LTXVCropGuides`` trims guide-frame conditioning using the pass-1 video
   latent; the cropped conditioning is used for pass 2.
8. ``LTXVLatentUpsampler`` spatially upscales the pass-1 video latent.
9. The image is re-injected into the upscaled latent via
   ``LTXVImgToVideoInplace`` (strength ``1.0``).
10. The upscaled video latent is re-concatenated with the pass-1 audio latent.
11. **Pass 2** — ``CFGGuider`` + ``ManualSigmas`` (short) + ``euler_cfg_pp``
    refines the upscaled result, using a fixed noise seed (42) matching the
    reference workflow.
12. ``LTXVSeparateAVLatent`` splits again; ``VAEDecodeTiled`` decodes video;
    ``LTXVAudioVAEDecode`` decodes audio.

The audio VAE and video VAE are both loaded from the dev fp8 checkpoint so no
extra download is needed for them.

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
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    result = run(
        models_dir="/path/to/models",
        image="/path/to/first_frame.png",
        audio_path="/path/to/soundtrack.mp3",
        prompt="the musician plays the melody, warm studio acoustics",
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
# HuggingFace repositories for LTX-Video 2.3
# ---------------------------------------------------------------------------

_HF_REPO_FP8 = "Lightricks/LTX-2.3-fp8"      # dev fp8 checkpoint
_HF_REPO_LTX = "Lightricks/LTX-2.3"           # LoRAs and upscaler
_HF_REPO_COMFY = "Comfy-Org/ltx-2"            # Gemma text encoder and TE LoRA

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "ltx-2.3-22b-dev-fp8.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "gemma_3_12B_it_fp4_mixed.safetensors"
_DISTILLED_LORA_DEST = Path("loras") / "ltx-2.3-22b-distilled-lora-384.safetensors"
_TE_LORA_DEST = Path("loras") / "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
_UPSCALER_DEST = Path("upscale_models") / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# The audio VAE is bundled inside the UNet checkpoint — no separate file.
_AUDIO_VAE_DEST = _UNET_DEST

# ManualSigmas strings from the reference workflow (video_ltx2_3_ia2v.json).
# Pass 1: full denoising schedule (euler_ancestral_cfg_pp, node 308)
# Pass 2: short refinement schedule (euler_cfg_pp, node 289)
_SIGMAS_PASS1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
_SIGMAS_PASS2 = "0.85, 0.7250, 0.4219, 0.0"

# Fixed noise seed for pass 2 — hardcoded in the reference workflow (node 285).
_PASS2_NOISE_SEED = 42


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2.3 IA2V pipeline.

    The LTX 2.3 dev fp8 checkpoint bundles the UNet, video VAE, and audio VAE
    in a single file.  Two LoRAs and a spatial upscaler are also required,
    matching the reference ``video_ltx2_3_ia2v.json`` workflow exactly.

    Each entry is an :class:`~comfy_diffusion.downloader.HFModelEntry` that
    resolves to a deterministic relative path under ``models_dir``.  Pass the
    result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_FP8,
            filename="ltx-2.3-22b-dev-fp8.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_COMFY,
            filename="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
            filename="ltx-2.3-22b-distilled-lora-384.safetensors",
            dest=_DISTILLED_LORA_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_COMFY,
            filename="split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
            dest=_TE_LORA_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
            filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            dest=_UPSCALER_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    image: Any,
    audio_path: str | Path,
    prompt: str,
    negative_prompt: str = "pc game, console game, video game, cartoon, childish, ugly",
    width: int = 768,
    height: int = 512,
    length: int = 97,
    fps: int = 24,
    cfg: float = 1.0,
    seed: int = 0,
    audio_start_time: float = 0.0,
    audio_duration: float | None = None,
    guide_strength_pass1: float = 0.7,
    guide_strength_pass2: float = 1.0,
    distilled_lora_strength: float = 0.5,
    te_lora_strength: float = 1.0,
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    audio_vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
    distilled_lora_filename: str | None = None,
    te_lora_filename: str | None = None,
    upscaler_filename: str | None = None,
) -> dict[str, Any]:
    """Run the LTX-Video 2.3 (22B dev fp8) image+audio-to-video pipeline.

    Mirrors the ``video_ltx2_3_ia2v`` official workflow.  The reference image
    is injected as a conditioning frame; the audio is VAE-encoded with a zero
    noise mask so the sampler preserves it as structural guidance.  Two-pass
    sampling chain: pass 1 denoises from scratch at the target resolution,
    pass 2 refines the spatially upscaled result.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    image : str | Path | PIL.Image.Image
        Input image for the first conditioning frame.  Accepts a file path
        (``str`` or :class:`~pathlib.Path`) or a :class:`~PIL.Image.Image`
        instance.
    audio_path : str | Path
        Path to the audio file (any format supported by *torchaudio*: MP3,
        WAV, FLAC, etc.) that will condition the generated video.
    prompt : str
        Positive text prompt.  Describe both visual content and the audio
        character in the same prompt for best results.
    negative_prompt : str, optional
        Negative text prompt.
        Default ``"pc game, console game, video game, cartoon, childish, ugly"``.
    width : int, optional
        Output frame width in pixels (must be divisible by 32).  Default ``768``.
    height : int, optional
        Output frame height in pixels (must be divisible by 32).  Default ``512``.
    length : int, optional
        Number of video frames to generate.  Default ``97`` (≈ 4 s at 24 fps).
    fps : int, optional
        Frame rate used for the audio latent and ``LTXVConditioning``.
        Default ``24``.
    cfg : float, optional
        Classifier-free guidance scale.  Default ``1.0`` (distilled model).
    seed : int, optional
        Random seed for pass-1 noise.  Default ``0``.  Pass-2 noise always
        uses the fixed seed ``42`` matching the reference workflow.
    audio_start_time : float, optional
        Start offset in seconds for trimming the input audio.  Default ``0.0``.
    audio_duration : float | None, optional
        Duration in seconds of audio to use.  When ``None`` the duration is
        derived from ``length / fps``.  Default ``None``.
    guide_strength_pass1 : float, optional
        ``LTXVImgToVideoInplace`` strength before pass 1.  Default ``0.7``.
    guide_strength_pass2 : float, optional
        ``LTXVImgToVideoInplace`` strength before pass 2 (on upscaled latent).
        Default ``1.0``.
    distilled_lora_strength : float, optional
        Strength applied to the distilled LoRA (model only).  Default ``0.5``.
    te_lora_strength : float, optional
        Strength applied to the Gemma text-encoder LoRA (CLIP only).
        Default ``1.0``.
    unet_filename : str | None, optional
        Override the default UNet filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the video VAE filename.  When ``None`` the VAE is loaded from
        the UNet checkpoint.  Default ``None``.
    audio_vae_filename : str | None, optional
        Override the audio VAE filename.  When ``None`` falls back to
        ``vae_filename`` (or the UNet checkpoint).  Default ``None``.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.  Default ``None``.
    distilled_lora_filename : str | None, optional
        Override the default distilled LoRA filename.  Default ``None``.
    te_lora_filename : str | None, optional
        Override the default Gemma text-encoder LoRA filename.  Default ``None``.
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
    from comfy_diffusion.audio import (
        load_audio,
        ltxv_audio_vae_decode,
        ltxv_audio_vae_encode,
        ltxv_concat_av_latent,
        ltxv_separate_av_latent,
    )
    from comfy_diffusion.conditioning import encode_prompt, ltxv_conditioning, ltxv_crop_guides
    from comfy_diffusion.image import image_to_tensor, load_image, ltxv_preprocess
    from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample, set_latent_noise_mask
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.mask import solid_mask
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import cfg_guider, get_sampler, manual_sigmas, random_noise, sample_custom
    from comfy_diffusion.vae import vae_decode_batch_tiled
    from comfy_diffusion.video import ltxv_img_to_video_inplace

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    # Resolve model paths.
    _p = Path(unet_filename) if unet_filename else None
    unet_path = (
        (models_dir / _UNET_DEST.parent / _p if _p and not _p.is_absolute() else _p)
        if _p
        else models_dir / _UNET_DEST
    )
    te_path = (
        Path(text_encoder_filename) if text_encoder_filename else models_dir / _TEXT_ENCODER_DEST
    )
    distilled_lora_path = (
        Path(distilled_lora_filename) if distilled_lora_filename else models_dir / _DISTILLED_LORA_DEST
    )
    te_lora_path = (
        Path(te_lora_filename) if te_lora_filename else models_dir / _TE_LORA_DEST
    )
    upscaler_path = (
        Path(upscaler_filename) if upscaler_filename else models_dir / _UPSCALER_DEST
    )
    # The checkpoint bundles video VAE and audio VAE; default to the same file.
    vae_path = Path(vae_filename) if vae_filename else unet_path
    audio_vae_path = Path(audio_vae_filename) if audio_vae_filename else vae_path

    # Load models.
    model = mm.load_unet(unet_path)
    vae = mm.load_vae(vae_path)
    audio_vae = mm.load_ltxv_audio_vae(audio_vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)
    upscale_model = mm.load_latent_upscale_model(upscaler_path)

    # Apply distilled LoRA to model only (clip_strength=0.0).
    # Apply TE LoRA to CLIP only (model_strength=0.0) — mirrors the workflow's
    # parallel branches where the TE LoRA model output is not connected to the sampler.
    model, clip = apply_lora(model, clip, distilled_lora_path, distilled_lora_strength, 0.0)
    model, clip = apply_lora(model, clip, te_lora_path, 0.0, te_lora_strength)

    # Load and preprocess the input image → BHWC float32 tensor.
    # LTXVPreprocess uses img_compression=18 matching the reference workflow (node 334).
    if hasattr(image, "mode"):
        image_tensor = image_to_tensor(image)
    else:
        image_tensor, _ = load_image(image)
    preprocessed = ltxv_preprocess(image_tensor, width, height, img_compression=18)

    # Load and trim audio, then encode with the audio VAE.
    # Default duration: length / fps (exactly covers the generated video).
    effective_duration = audio_duration if audio_duration is not None else length / fps
    audio_data = load_audio(audio_path, start_time=audio_start_time, duration=effective_duration)
    audio_latent_enc = ltxv_audio_vae_encode(audio_vae, audio_data)

    # Apply a zero noise mask to the audio latent so the sampler preserves the
    # encoded audio signal as structural guidance (mask=0 → latent_mask=1 →
    # the original audio is kept, not denoised from noise).
    audio_samples = audio_latent_enc["samples"]
    if hasattr(audio_samples, "shape") and len(audio_samples.shape) >= 2:
        mask_h = int(audio_samples.shape[-2])
        mask_w = int(audio_samples.shape[-1])
    else:
        mask_h, mask_w = 1, 1
    audio_latent_masked = set_latent_noise_mask(
        audio_latent_enc, solid_mask(0.0, width=mask_w, height=mask_h)
    )

    # Text conditioning + frame-rate metadata.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    # Create empty video latent, inject image at pass-1 strength (no crop guides yet).
    video_latent = ltxv_empty_latent_video(width=width, height=height, length=length)
    video_latent = ltxv_img_to_video_inplace(vae, preprocessed, video_latent, guide_strength_pass1)

    # Concatenate the image-conditioned video latent with the masked audio latent.
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent_masked)

    # --- Pass 1: full denoising with euler_ancestral_cfg_pp ---
    # CFGGuider uses the original conditioning (guide crops applied AFTER pass 1).
    noise_1 = random_noise(seed)
    guider_1 = cfg_guider(model, positive, negative, cfg)
    sigmas_1 = manual_sigmas(_SIGMAS_PASS1)
    sampler_1 = get_sampler("euler_ancestral_cfg_pp")
    _, denoised_1 = sample_custom(noise_1, guider_1, sampler_1, sigmas_1, av_latent)

    # Separate the pass-1 output into video and audio latents.
    video_latent_1, audio_latent_1 = ltxv_separate_av_latent(denoised_1)

    # Apply LTXVCropGuides using the pass-1 video latent — trims guide-frame
    # conditioning for the refinement pass.  The output (cropped) latent is
    # not used; only the cropped positive/negative conditioning are needed.
    positive_p2, negative_p2, _ = ltxv_crop_guides(positive, negative, video_latent_1)

    # Spatially upscale the pass-1 video latent and re-inject image (pass-2 strength).
    video_latent_up = ltxv_latent_upsample(video_latent_1, upscale_model=upscale_model, vae=vae)
    video_latent_up = ltxv_img_to_video_inplace(vae, preprocessed, video_latent_up, guide_strength_pass2)

    # Re-concatenate the upscaled video latent with the pass-1 audio latent.
    av_latent_2 = ltxv_concat_av_latent(video_latent_up, audio_latent_1)

    # --- Pass 2: short refinement with euler_cfg_pp (fixed seed 42, from workflow) ---
    noise_2 = random_noise(_PASS2_NOISE_SEED)
    guider_2 = cfg_guider(model, positive_p2, negative_p2, cfg)
    sigmas_2 = manual_sigmas(_SIGMAS_PASS2)
    sampler_2 = get_sampler("euler_cfg_pp")
    _, denoised_2 = sample_custom(noise_2, guider_2, sampler_2, sigmas_2, av_latent_2)

    # Separate the final video and audio latents.
    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised_2)

    # Decode video → PIL frames; decode audio → waveform dict.
    frames = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)

    return {"frames": frames, "audio": audio}
