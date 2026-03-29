"""LTX-Video 2 Canny-to-Video pipeline (dev fp8 checkpoint, audio-visual).

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: video loading,
  frame extraction, Canny edge detection, model loading, LoRA application,
  conditioning with first-frame guide, AV sampling, latent upsampling, and
  decoding.

The Canny-to-Video variant uses:

- The dev fp8 UNet checkpoint (``ltx-2-19b-dev-fp8.safetensors``) as base.
- A Canny control LoRA (``ltx-2-19b-ic-lora-canny-control.safetensors``)
  applied before pass 1 to enable structure-guided generation.
- A distilled LoRA (``ltx-2-19b-distilled-lora-384.safetensors``) stacked on
  top in pass 2 for accelerated refinement.
- A Gemma 3 12B text encoder loaded via
  :meth:`~comfy_diffusion.models.ModelManager.load_ltxav_text_encoder`.
- Canny edge detection (``low_threshold=102``, ``high_threshold=204``)
  applied to half-resolution video frames extracted from the input video.
- A first-frame guide injected via
  :func:`~comfy_diffusion.conditioning.ltxv_add_guide` into the half-resolution
  latent and via :func:`~comfy_diffusion.video.ltxv_img_to_video_inplace` at
  full resolution before pass 2.
- A spatial latent upsampler (``ltx-2-spatial-upscaler-x2-1.0.safetensors``)
  applied between passes.
- The full audio-visual (AV) sampling chain — video and audio generated together.
- Pass 1 uses ``ManualSigmas`` with 9 values and ``euler_ancestral`` at half
  resolution with CFG=3.
- Pass 2 uses ``ManualSigmas`` with 4 values and ``euler_ancestral`` at full
  resolution with CFG=1.

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
    from comfy_diffusion.pipelines.video.ltx.ltx2.canny import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    result = run(
        models_dir="/path/to/models",
        video_path="/path/to/input.mp4",
        prompt="a squirrel nibbles on a nut in a sunlit garden",
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

_HF_REPO_LTX = "Lightricks/LTX-2"
_HF_REPO_COMFY = "Comfy-Org/ltx-2"
_HF_REPO_CANNY = "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "ltx-2-19b-dev-fp8.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "gemma_3_12B_it_fp4_mixed.safetensors"
_CANNY_LORA_DEST = Path("loras") / "ltx-2-19b-ic-lora-canny-control.safetensors"
_LORA_DEST = Path("loras") / "ltx-2-19b-distilled-lora-384.safetensors"
_UPSCALER_DEST = Path("upscale_models") / "ltx-2-spatial-upscaler-x2-1.0.safetensors"

# Canny edge-detection thresholds matching the reference workflow (0.4, 0.8).
_CANNY_LOW = 102   # round(0.4 * 255)
_CANNY_HIGH = 204  # round(0.8 * 255)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2 Canny-to-Video pipeline.

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
            repo_id=_HF_REPO_COMFY,
            filename="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_CANNY,
            filename="ltx-2-19b-ic-lora-canny-control.safetensors",
            dest=_CANNY_LORA_DEST,
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
    video_path: str | Path,
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int = 1280,
    height: int = 720,
    length: int = 97,
    fps: int = 25,
    cfg_pass1: float = 3.0,
    cfg_pass2: float = 1.0,
    seed: int = 0,
    canny_lora_strength: float = 1.0,
    lora_strength: float = 1.0,
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    audio_vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
    canny_lora_filename: str | None = None,
    lora_filename: str | None = None,
    upscaler_filename: str | None = None,
) -> dict[str, Any]:
    """Run the LTX-Video 2 Canny-to-Video pipeline end-to-end.

    Two-pass sampling mirroring the ``video_ltx2_canny_to_video.json`` reference
    workflow (LTX 2.0 dev fp8 subgraph):

    - Pass 1: base model + Canny control LoRA at half resolution using
      ``ManualSigmas`` with 9 steps and ``euler_ancestral``; CFG=3.
    - Pass 2: Canny LoRA + distilled LoRA at full resolution using
      ``ManualSigmas`` with 4 steps and ``euler_ancestral``; CFG=1.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    video_path : str | Path
        Path to the input video file.  Frames are extracted, resized to the
        target resolution, and passed through Canny edge detection to generate
        the structure-control signal for pass 1.
    prompt : str
        Positive text prompt describing the desired video content.
    negative_prompt : str, optional
        Negative text prompt.
        Default ``"worst quality, inconsistent motion, blurry, jittery, distorted"``.
    width : int, optional
        Output frame width in pixels.  Pass 1 operates at half this width.
        Default ``1280``.
    height : int, optional
        Output frame height in pixels.  Pass 1 operates at half this height.
        Default ``720``.
    length : int, optional
        Number of video frames to generate.  Default ``97``.
    fps : int, optional
        Frame rate used for the audio latent and ``LTXVConditioning``.
        Default ``25``.
    cfg_pass1 : float, optional
        CFG scale for pass 1 (Canny LoRA, half resolution).  Default ``3.0``.
    cfg_pass2 : float, optional
        CFG scale for pass 2 (Canny + distilled LoRA, full resolution).
        Default ``1.0``.
    seed : int, optional
        Random seed.  Pass 1 uses ``seed + 10``, pass 2 uses ``seed``.
        Default ``0``.
    canny_lora_strength : float, optional
        Strength of the Canny control LoRA.  Default ``1.0``.
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
    canny_lora_filename : str | None, optional
        Override the default Canny control LoRA filename.  Default ``None``.
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
    from comfy_diffusion.conditioning import encode_prompt, ltxv_add_guide, ltxv_conditioning, ltxv_crop_guides
    from comfy_diffusion.image import canny, image_from_batch, image_scale_by, ltxv_preprocess
    from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import cfg_guider, get_sampler, manual_sigmas, random_noise, sample_custom
    from comfy_diffusion.vae import vae_decode_batch_tiled
    from comfy_diffusion.video import get_video_components, load_video
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
    canny_lora_path = (
        Path(canny_lora_filename) if canny_lora_filename else models_dir / _CANNY_LORA_DEST
    )
    lora_path = Path(lora_filename) if lora_filename else models_dir / _LORA_DEST
    upscaler_path = (
        Path(upscaler_filename) if upscaler_filename else models_dir / _UPSCALER_DEST
    )
    vae_path = Path(vae_filename) if vae_filename else unet_path
    audio_vae_path = Path(audio_vae_filename) if audio_vae_filename else vae_path

    # Load models.
    model_base = mm.load_unet(unet_path)
    vae = mm.load_vae(vae_path)
    audio_vae = mm.load_ltxv_audio_vae(audio_vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)
    upscale_model = mm.load_latent_upscale_model(upscaler_path)

    # Apply Canny control LoRA — model-only (LoraLoaderModelOnly in workflow).
    model_canny, _ = apply_lora(model_base, clip, canny_lora_path, canny_lora_strength, 0.0)

    # Extract video frames and preprocess for Canny edge detection.
    video = load_video(video_path)
    frames_raw, _ = get_video_components(video)
    frames = image_from_batch(frames_raw, 0, length)
    # Resize to full output resolution, then scale to half for Canny pass 1.
    frames_full = ltxv_preprocess(frames, width, height)
    frames_half = image_scale_by(frames_full, "lanczos", 0.5)
    canny_frames = canny(frames_half, _CANNY_LOW, _CANNY_HIGH)
    # First frame at full resolution — used as temporal guide.
    first_frame = image_from_batch(frames_full, 0, 1)

    # Text conditioning + frame-rate metadata.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    # Pass 1: half resolution, Canny LoRA, ManualSigmas (9 steps) + euler_ancestral.
    latent_width = width // 2
    latent_height = height // 2
    video_latent = ltxv_empty_latent_video(width=latent_width, height=latent_height, length=length)
    # Inject Canny edges for structural control.
    video_latent = ltxv_img_to_video_inplace(vae, canny_frames, video_latent)
    # Add first-frame temporal guide (updates conditioning and latent).
    positive, negative, video_latent = ltxv_add_guide(
        positive, negative, vae, video_latent, first_frame
    )
    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    sigmas_p1 = manual_sigmas("1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
    guider_p1 = cfg_guider(model_canny, positive, negative, cfg_pass1)
    noise_p1 = random_noise(seed + 10)
    sampler_obj = get_sampler("euler_ancestral")
    _, denoised_p1 = sample_custom(noise_p1, guider_p1, sampler_obj, sigmas_p1, av_latent)

    # Between passes: separate → crop guides → upsample → reinject first frame → concat.
    video_latent_p1, audio_latent_p1 = ltxv_separate_av_latent(denoised_p1)
    positive, negative, video_latent_p1 = ltxv_crop_guides(positive, negative, video_latent_p1)
    video_up = ltxv_latent_upsample(video_latent_p1, upscale_model=upscale_model, vae=vae)
    # Reinject first frame at full resolution before pass 2.
    video_up = ltxv_img_to_video_inplace(vae, first_frame, video_up)
    av_latent_p2 = ltxv_concat_av_latent(video_up, audio_latent_p1)

    # Pass 2: Canny LoRA + distilled LoRA, ManualSigmas (4 steps) + euler_ancestral.
    model_p2, clip = apply_lora(model_canny, clip, lora_path, lora_strength, 0.0)
    sigmas_p2 = manual_sigmas("0.909375, 0.725, 0.421875, 0.0")
    guider_p2 = cfg_guider(model_p2, positive, negative, cfg_pass2)
    noise_p2 = random_noise(seed)
    _, denoised_p2 = sample_custom(noise_p2, guider_p2, sampler_obj, sigmas_p2, av_latent_p2)

    # Separate video and audio; decode to PIL frames and waveform.
    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised_p2)
    frames_out = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)

    return {"frames": frames_out, "audio": audio}
