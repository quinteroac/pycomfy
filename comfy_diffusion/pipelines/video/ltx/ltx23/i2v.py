"""LTX-Video 2.3 (22B dev fp8) image-to-video pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  LoRA application, image preprocessing, conditioning, two-pass AV sampling
  with latent upscaling, and decoding.

This pipeline mirrors ``comfyui_official_workflows/video/ltx/ltx23/video_ltx2_3_i2v.json``
and uses the LTX-Video 2.3 AV (audio-visual) model, which generates video
**and** audio simultaneously.

Sampling chain (two-pass, mirrors the reference workflow exactly):

1. The input image is preprocessed with ``LTXVPreprocess`` and injected into
   the empty video latent via ``LTXVImgToVideoInplace`` (strength ``0.7``).
2. ``LTXVCropGuides`` trims the appended keyframe latents.
3. ``LTXVConditioning`` injects frame-rate metadata.
4. Video and audio latents are concatenated with ``LTXVConcatAVLatent``.
5. **Pass 1** — ``CFGGuider`` + ``ManualSigmas`` (full) + ``euler_ancestral_cfg_pp``
   drives the first denoising loop.
6. ``LTXVSeparateAVLatent`` splits the pass-1 result into video and audio latents.
7. ``LTXVLatentUpsampler`` spatially upscales the video latent.
8. The image is re-injected into the upscaled latent via ``LTXVImgToVideoInplace``
   (strength ``1.0``).
9. The upscaled video latent is re-concatenated with the pass-1 audio latent.
10. **Pass 2** — ``CFGGuider`` + ``ManualSigmas`` (short) + ``euler_cfg_pp``
    refines the upscaled result.
11. ``LTXVSeparateAVLatent`` splits again; ``VAEDecodeTiled`` decodes video;
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
    from comfy_diffusion.pipelines.video.ltx.ltx23.i2v import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    result = run(
        models_dir="/path/to/models",
        image="/path/to/first_frame.png",
        prompt="the queen turns her head slowly towards the camera",
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

# ManualSigmas strings from the reference workflow.
_SIGMAS_PASS1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
_SIGMAS_PASS2 = "0.85, 0.7250, 0.4219, 0.0"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2.3 I2V pipeline.

    The LTX 2.3 dev fp8 checkpoint bundles the UNet, video VAE, and audio VAE
    in a single file.  Two LoRAs and a spatial upscaler are also required,
    matching the reference ``video_ltx2_3_i2v.json`` workflow exactly.

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
    prompt: str,
    negative_prompt: str = "pc game, console game, video game, cartoon, childish, ugly",
    width: int = 768,
    height: int = 512,
    length: int = 97,
    fps: int = 25,
    cfg: float = 1.0,
    seed: int = 0,
    guide_strength_pass1: float = 0.7,
    guide_strength_pass2: float = 1.0,
    distilled_lora_strength: float = 0.5,
    te_lora_strength: float = 1.0,
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
    distilled_lora_filename: str | None = None,
    te_lora_filename: str | None = None,
    upscaler_filename: str | None = None,
) -> dict[str, Any]:
    """Run the LTX-Video 2.3 (22B dev fp8) image-to-video-with-audio pipeline.

    Mirrors the ``video_ltx2_3_i2v`` official workflow.  Uses a two-pass
    sampling chain: the input image is injected before each pass, with spatial
    upscaling between the two passes.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    image : str | Path | PIL.Image.Image
        Input image for the first conditioning frame.  Accepts a file path
        (``str`` or :class:`~pathlib.Path`) or a :class:`~PIL.Image.Image`
        instance.
    prompt : str
        Positive text prompt.
    negative_prompt : str, optional
        Negative text prompt.
        Default ``"worst quality, inconsistent motion, blurry, jittery, distorted"``.
    width : int, optional
        Output frame width in pixels (must be divisible by 32).  Default ``768``.
    height : int, optional
        Output frame height in pixels (must be divisible by 32).  Default ``512``.
    length : int, optional
        Number of video frames to generate.  Default ``97`` (≈ 4 s at 25 fps).
    fps : int, optional
        Frame rate used for the audio latent and ``LTXVConditioning``.
        Default ``25``.
    cfg : float, optional
        Classifier-free guidance scale.  Default ``1.0`` (distilled model).
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    guide_strength_pass1 : float, optional
        ``LTXVImgToVideoInplace`` strength before pass 1.  Default ``0.7``.
    guide_strength_pass2 : float, optional
        ``LTXVImgToVideoInplace`` strength before pass 2 (on upscaled latent).
        Default ``1.0``.
    distilled_lora_strength : float, optional
        Strength applied to the distilled LoRA (model only).  Default ``0.5``.
    te_lora_strength : float, optional
        Strength applied to the Gemma text-encoder LoRA (model + CLIP).
        Default ``1.0``.
    unet_filename : str | None, optional
        Override the default UNet filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the VAE filename.  When ``None`` the VAE is loaded from the
        UNet checkpoint.  Default ``None``.
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
    from comfy_diffusion.audio import ltxv_audio_vae_decode, ltxv_concat_av_latent, ltxv_empty_latent_audio, ltxv_separate_av_latent
    from comfy_diffusion.conditioning import encode_prompt, ltxv_conditioning, ltxv_crop_guides
    from comfy_diffusion.image import image_to_tensor, load_image, ltxv_preprocess
    from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample
    from comfy_diffusion.lora import apply_lora
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
    unet_path = (models_dir / _UNET_DEST.parent / _p if _p and not _p.is_absolute() else _p) if _p else models_dir / _UNET_DEST
    te_path = (
        Path(text_encoder_filename)
        if text_encoder_filename
        else models_dir / _TEXT_ENCODER_DEST
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

    # Load models.
    model = mm.load_unet(unet_path)
    vae = mm.load_vae(vae_path)
    audio_vae = mm.load_ltxv_audio_vae(vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)
    upscale_model = mm.load_latent_upscale_model(upscaler_path)

    # Apply distilled LoRA (model only, strength=0.5) then Gemma TE LoRA (model + CLIP).
    model, clip = apply_lora(model, clip, distilled_lora_path, distilled_lora_strength, 0.0)
    model, clip = apply_lora(model, clip, te_lora_path, te_lora_strength, te_lora_strength)

    # Load and preprocess the input image → BHWC float32 tensor.
    if hasattr(image, "mode"):
        image_tensor = image_to_tensor(image)
    else:
        image_tensor, _ = load_image(image)
    preprocessed = ltxv_preprocess(image_tensor, width, height)

    # Text conditioning + frame-rate metadata.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    # Create empty video latent, inject image (pass-1 strength), then crop guides.
    video_latent = ltxv_empty_latent_video(width=width, height=height, length=length)
    video_latent = ltxv_img_to_video_inplace(vae, preprocessed, video_latent, guide_strength_pass1)
    positive, negative, video_latent = ltxv_crop_guides(positive, negative, video_latent)

    # Create audio latent and concatenate into a single AV latent.
    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    # --- Pass 1: full denoising with euler_ancestral_cfg_pp ---
    guider = cfg_guider(model, positive, negative, cfg)
    noise = random_noise(seed)
    sigmas_1 = manual_sigmas(_SIGMAS_PASS1)
    sampler_1 = get_sampler("euler_ancestral_cfg_pp")
    _, denoised_1 = sample_custom(noise, guider, sampler_1, sigmas_1, av_latent)

    # Separate video and audio from the pass-1 result.
    video_latent_1, audio_latent_1 = ltxv_separate_av_latent(denoised_1)

    # Spatially upscale the video latent, then re-inject image (pass-2 strength).
    video_latent_up = ltxv_latent_upsample(video_latent_1, upscale_model=upscale_model, vae=vae)
    video_latent_up = ltxv_img_to_video_inplace(vae, preprocessed, video_latent_up, guide_strength_pass2)

    # Re-concatenate with the pass-1 audio latent for the refinement pass.
    av_latent_2 = ltxv_concat_av_latent(video_latent_up, audio_latent_1)

    # --- Pass 2: short refinement with euler_cfg_pp ---
    sigmas_2 = manual_sigmas(_SIGMAS_PASS2)
    sampler_2 = get_sampler("euler_cfg_pp")
    _, denoised_2 = sample_custom(noise, guider, sampler_2, sigmas_2, av_latent_2)

    # Separate the final video and audio latents.
    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised_2)

    # Decode video → PIL frames; decode audio → waveform dict.
    frames = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)

    return {"frames": frames, "audio": audio}
