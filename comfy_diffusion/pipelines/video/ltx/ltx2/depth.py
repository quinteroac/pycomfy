"""LTX-Video 2 Depth-to-Video pipeline (dev fp8 checkpoint, audio-visual).

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: video loading,
  frame extraction, Lotus depth estimation, model loading, LoRA application,
  AV sampling, latent upsampling, and decoding.

The Depth-to-Video variant uses:

- The dev fp8 checkpoint (``ltx-2-19b-dev-fp8.safetensors``) as base, loaded
  via ``CheckpointLoaderSimple`` from ``checkpoints/``.
- A Lotus depth-estimation model (``lotus-depth-d-v1-1.safetensors``) with a
  standard SD VAE (``vae-ft-mse-840000-ema-pruned.safetensors``) to generate
  per-frame inverted depth maps from half-resolution video frames.
- A depth-control LoRA (``ltx-2-19b-ic-lora-depth-control.safetensors``)
  applied before pass 1 to enable structure-guided generation.
- A distilled LoRA (``ltx-2-19b-distilled-lora-384.safetensors``) stacked on
  top in pass 2 for accelerated refinement.
- A Gemma 3 12B text encoder loaded via
  :meth:`~comfy_diffusion.models.ModelManager.load_ltxav_text_encoder`.
- Inverted Lotus depth maps used as the ``LTXVAddGuide`` control signal,
  alongside a first-frame guide injected via
  :func:`~comfy_diffusion.video.ltxv_img_to_video_inplace`.
- A spatial latent upsampler (``ltx-2-spatial-upscaler-x2-1.0.safetensors``)
  applied between passes.
- The full audio-visual (AV) sampling chain — video and audio generated together.
- Pass 1 uses ``LTXVScheduler`` (20 steps, max_shift=2.05, base_shift=0.95)
  and ``euler`` at half resolution with CFG=3.
- Pass 2 uses ``ManualSigmas`` with 4 values and ``gradient_estimation`` at
  full resolution with CFG=1.

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
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    result = run(
        models_dir="/path/to/models",
        video_path="/path/to/input.mp4",
        prompt="a squirrel walks through a dense autumn forest",
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
# HuggingFace repositories
# ---------------------------------------------------------------------------

_HF_REPO_LTX = "Lightricks/LTX-2"
_HF_REPO_COMFY = "Comfy-Org/ltx-2"
_HF_REPO_DEPTH = "Lightricks/LTX-2-19b-IC-LoRA-Depth-Control"
_HF_REPO_LOTUS = "Comfy-Org/lotus"
_HF_REPO_SD_VAE = "stabilityai/sd-vae-ft-mse-original"

# Relative destination paths (resolved against models_dir by download_models).
_CKPT_DEST = Path("checkpoints") / "ltx-2-19b-dev-fp8.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "gemma_3_12B_it_fp4_mixed.safetensors"
_DEPTH_LORA_DEST = Path("loras") / "ltx-2-19b-ic-lora-depth-control.safetensors"
_LORA_DEST = Path("loras") / "ltx-2-19b-distilled-lora-384.safetensors"
_UPSCALER_DEST = Path("upscale_models") / "ltx-2-spatial-upscaler-x2-1.0.safetensors"
_LOTUS_MODEL_DEST = Path("diffusion_models") / "lotus-depth-d-v1-1.safetensors"
_LOTUS_VAE_DEST = Path("vae") / "vae-ft-mse-840000-ema-pruned.safetensors"

# LTXVScheduler parameters for pass 1 (matches workflow node LTXVScheduler).
_P1_STEPS = 20
_P1_MAX_SHIFT = 2.05
_P1_BASE_SHIFT = 0.95
_P1_STRETCH = True
_P1_TERMINAL = 0.1

# ManualSigmas string for pass 2 (matches workflow ManualSigmas widget value).
_P2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2 Depth-to-Video pipeline.

    Each entry is an :class:`~comfy_diffusion.downloader.HFModelEntry` that
    resolves to a deterministic relative path under ``models_dir``.  Pass the
    result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
            filename="ltx-2-19b-dev-fp8.safetensors",
            dest=_CKPT_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_COMFY,
            filename="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_DEPTH,
            filename="ltx-2-19b-ic-lora-depth-control.safetensors",
            dest=_DEPTH_LORA_DEST,
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
        HFModelEntry(
            repo_id=_HF_REPO_LOTUS,
            filename="lotus-depth-d-v1-1.safetensors",
            dest=_LOTUS_MODEL_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_SD_VAE,
            filename="vae-ft-mse-840000-ema-pruned.safetensors",
            dest=_LOTUS_VAE_DEST,
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
    length: int = 121,
    fps: int = 24,
    cfg_pass1: float = 3.0,
    cfg_pass2: float = 1.0,
    seed: int = 0,
    depth_lora_strength: float = 1.0,
    lora_strength: float = 1.0,
    ckpt_filename: str | None = None,
    vae_filename: str | None = None,
    audio_vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
    depth_lora_filename: str | None = None,
    lora_filename: str | None = None,
    upscaler_filename: str | None = None,
    lotus_model_filename: str | None = None,
    lotus_vae_filename: str | None = None,
) -> dict[str, Any]:
    """Run the LTX-Video 2 Depth-to-Video pipeline end-to-end.

    Three-phase pipeline mirroring the ``video_ltx2_depth_to_video.json``
    reference workflow:

    - **Lotus phase**: extract half-resolution video frames and run the Lotus
      depth estimator to produce per-frame inverted depth maps.
    - **Pass 1**: base model + depth-control LoRA at half resolution using
      ``LTXVScheduler`` (20 steps) and ``euler``; CFG=3.
    - **Pass 2**: depth LoRA + distilled LoRA at full resolution using
      ``ManualSigmas`` with 4 steps and ``gradient_estimation``; CFG=1.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    video_path : str | Path
        Path to the input video file.  Frames are extracted, resized to the
        target resolution, and passed through Lotus depth estimation to
        generate the structure-control signal for pass 1.
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
        Number of video frames to generate.  Default ``121``.
    fps : int, optional
        Frame rate used for the audio latent and ``LTXVConditioning``.
        Default ``24``.
    cfg_pass1 : float, optional
        CFG scale for pass 1 (depth LoRA, half resolution).  Default ``3.0``.
    cfg_pass2 : float, optional
        CFG scale for pass 2 (depth + distilled LoRA, full resolution).
        Default ``1.0``.
    seed : int, optional
        Random seed.  Both passes use ``seed``.  Default ``0``.
    depth_lora_strength : float, optional
        Strength of the depth control LoRA.  Default ``1.0``.
    lora_strength : float, optional
        Strength of the distilled LoRA applied in pass 2.  Default ``1.0``.
    ckpt_filename : str | None, optional
        Override the default checkpoint filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the VAE filename.  When ``None`` the VAE is loaded from the
        checkpoint.  Default ``None``.
    audio_vae_filename : str | None, optional
        Override the audio VAE filename.  Default ``None``.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.  Default ``None``.
    depth_lora_filename : str | None, optional
        Override the default depth control LoRA filename.  Default ``None``.
    lora_filename : str | None, optional
        Override the default distilled LoRA filename.  Default ``None``.
    upscaler_filename : str | None, optional
        Override the default spatial upscaler filename.  Default ``None``.
    lotus_model_filename : str | None, optional
        Override the default Lotus diffusion model filename.  Default ``None``.
    lotus_vae_filename : str | None, optional
        Override the default Lotus VAE filename.  Default ``None``.

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
    from comfy_diffusion.controlnet import lotus_depth_pass
    from comfy_diffusion.image import image_from_batch, image_scale_by, ltxv_preprocess
    from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import cfg_guider, get_sampler, ltxv_scheduler, manual_sigmas, random_noise, sample_custom
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
    ckpt_path = Path(ckpt_filename) if ckpt_filename else models_dir / _CKPT_DEST
    te_path = (
        Path(text_encoder_filename)
        if text_encoder_filename
        else models_dir / _TEXT_ENCODER_DEST
    )
    depth_lora_path = (
        Path(depth_lora_filename) if depth_lora_filename else models_dir / _DEPTH_LORA_DEST
    )
    lora_path = Path(lora_filename) if lora_filename else models_dir / _LORA_DEST
    upscaler_path = (
        Path(upscaler_filename) if upscaler_filename else models_dir / _UPSCALER_DEST
    )
    vae_path = Path(vae_filename) if vae_filename else ckpt_path
    audio_vae_path = Path(audio_vae_filename) if audio_vae_filename else vae_path
    lotus_model_path = (
        Path(lotus_model_filename) if lotus_model_filename else models_dir / _LOTUS_MODEL_DEST
    )
    lotus_vae_path = (
        Path(lotus_vae_filename) if lotus_vae_filename else models_dir / _LOTUS_VAE_DEST
    )

    # Load LTX-2 models.
    model_base = mm.load_unet(ckpt_path)
    vae = mm.load_vae(vae_path)
    audio_vae = mm.load_ltxv_audio_vae(audio_vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, ckpt_path)
    upscale_model = mm.load_latent_upscale_model(upscaler_path)

    # Load Lotus models (depth estimation).
    lotus_model = mm.load_unet(lotus_model_path)
    lotus_vae = mm.load_vae(lotus_vae_path)

    # Apply depth-control LoRA — model-only (LoraLoaderModelOnly in workflow).
    model_depth, _ = apply_lora(model_base, clip, depth_lora_path, depth_lora_strength, 0.0)

    # Extract video frames and preprocess for Lotus depth estimation.
    video = load_video(video_path)
    frames_raw, _ = get_video_components(video)
    frames = image_from_batch(frames_raw, 0, length)
    # Resize to full output resolution, then scale to half for pass 1.
    frames_full = ltxv_preprocess(frames, width, height)
    frames_half = image_scale_by(frames_full, "lanczos", 0.5)

    # First frame at full resolution — used as temporal guide for both passes.
    first_frame = image_from_batch(frames_full, 0, 1)

    # --- Lotus depth estimation phase ---
    # Produces an inverted depth map at half resolution for all frames.
    depth_map = lotus_depth_pass(lotus_model, lotus_vae, frames_half)

    # Text conditioning + frame-rate metadata.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    # Pass 1: half resolution, depth LoRA, LTXVScheduler (20 steps) + euler.
    latent_width = width // 2
    latent_height = height // 2
    video_latent = ltxv_empty_latent_video(width=latent_width, height=latent_height, length=length)
    # Inject first frame as temporal anchor.
    video_latent = ltxv_img_to_video_inplace(vae, first_frame, video_latent)
    # Add depth map as spatial guide (updates conditioning and latent).
    positive, negative, video_latent = ltxv_add_guide(
        positive, negative, vae, video_latent, depth_map
    )
    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    sigmas_p1 = ltxv_scheduler(
        _P1_STEPS, _P1_MAX_SHIFT, _P1_BASE_SHIFT,
        stretch=_P1_STRETCH, terminal=_P1_TERMINAL, latent=av_latent,
    )
    guider_p1 = cfg_guider(model_depth, positive, negative, cfg_pass1)
    noise_p1 = random_noise(seed)
    sampler_euler = get_sampler("euler")
    output_p1, _ = sample_custom(noise_p1, guider_p1, sampler_euler, sigmas_p1, av_latent)

    # Between passes: separate → crop guides → upsample → reinject first frame → concat.
    video_latent_p1, audio_latent_p1 = ltxv_separate_av_latent(output_p1)
    positive, negative, video_latent_p1 = ltxv_crop_guides(positive, negative, video_latent_p1)
    video_up = ltxv_latent_upsample(video_latent_p1, upscale_model=upscale_model, vae=vae)
    # Reinject first frame at full resolution before pass 2.
    video_up = ltxv_img_to_video_inplace(vae, first_frame, video_up)
    av_latent_p2 = ltxv_concat_av_latent(video_up, audio_latent_p1)

    # Pass 2: depth LoRA + distilled LoRA, ManualSigmas (4 steps) + gradient_estimation.
    model_p2, clip = apply_lora(model_depth, clip, lora_path, lora_strength, 0.0)
    sigmas_p2 = manual_sigmas(_P2_SIGMAS)
    guider_p2 = cfg_guider(model_p2, positive, negative, cfg_pass2)
    noise_p2 = random_noise(seed)
    sampler_grad = get_sampler("gradient_estimation")
    _, denoised_p2 = sample_custom(noise_p2, guider_p2, sampler_grad, sigmas_p2, av_latent_p2)

    # Separate video and audio; decode to PIL frames and waveform.
    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised_p2)
    frames_out = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)

    return {"frames": frames_out, "audio": audio}
