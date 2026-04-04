"""LTX-Video 2.3 (22B dev fp8) text-to-video pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  LoRA application, conditioning, two-pass AV sampling with latent upscaling,
  and decoding.

This pipeline mirrors ``comfyui_official_workflows/video/ltx/ltx23/video_ltx2_3_t2v.json``
and uses the LTX-Video 2.3 AV (audio-visual) model, which generates video
**and** audio simultaneously.

Sampling chain (two-pass, mirrors the reference workflow exactly):

1. ``LTXVConditioning`` injects frame-rate metadata.
2. Video and audio latents are concatenated with ``LTXVConcatAVLatent``.
3. **Pass 1** — ``CFGGuider`` + ``ManualSigmas`` (full) + ``euler_ancestral_cfg_pp``
   drives the first denoising loop.
4. ``LTXVSeparateAVLatent`` splits the result into video and audio latents.
5. ``LTXVLatentUpsampler`` spatially upscales the video latent.
6. The upscaled video latent is re-concatenated with the pass-1 audio latent.
7. **Pass 2** — ``CFGGuider`` + ``ManualSigmas`` (short) + ``euler_cfg_pp``
   refines the upscaled result.
8. ``LTXVSeparateAVLatent`` splits again; ``VAEDecodeTiled`` decodes video;
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
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest, run

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
# HuggingFace repositories for LTX-Video 2.3
# ---------------------------------------------------------------------------

_HF_REPO_CKPT = "Lightricks/LTX-2.3-fp8"
_HF_REPO_LORA = "Lightricks/LTX-2.3"
_HF_REPO_TE_LORA = "Comfy-Org/ltx-2"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("checkpoints") / "ltx-2.3-22b-dev-fp8.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "gemma_3_12B_it_fp4_mixed.safetensors"
_DISTILLED_LORA_DEST = Path("loras") / "ltx-2.3-22b-distilled-lora-384.safetensors"
_TE_LORA_DEST = Path("loras") / "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
_UPSCALER_DEST = Path("latent_upscale_models") / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# The audio VAE is bundled inside the UNet checkpoint — no separate file.
_AUDIO_VAE_DEST = _UNET_DEST

# ManualSigmas strings from the reference workflow.
_SIGMAS_PASS1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
_SIGMAS_PASS2 = "0.85, 0.7250, 0.4219, 0.0"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2.3 T2V pipeline.

    The dev fp8 checkpoint bundles the UNet, video VAE, and audio VAE
    in a single file.  Two LoRAs (distilled + TE) and a spatial upscaler are
    also required, matching the reference workflow exactly.

    Each entry is an :class:`~comfy_diffusion.downloader.HFModelEntry` that
    resolves to a deterministic relative path under ``models_dir``.  Pass the
    result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_CKPT,
            filename="ltx-2.3-22b-dev-fp8.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LORA,
            filename="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LORA,
            filename="ltx-2.3-22b-distilled-lora-384.safetensors",
            dest=_DISTILLED_LORA_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_TE_LORA,
            filename="split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
            dest=_TE_LORA_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LORA,
            filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            dest=_UPSCALER_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int = 768,
    height: int = 512,
    length: int = 97,
    fps: int = 25,
    cfg: float = 1.0,
    seed: int = 0,
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
    """Run the LTX-Video 2.3 (22B dev fp8) text-to-video-with-audio pipeline.

    Two-pass sampling with spatial upscaling, mirroring the reference workflow
    exactly.  The dev fp8 checkpoint bundles the UNet, video VAE, and audio VAE
    in a single file.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
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
        Number of video frames to generate.  Default ``97``.
    fps : int, optional
        Frame rate used for the audio latent and ``LTXVConditioning``.
        Default ``25``.
    cfg : float, optional
        Classifier-free guidance scale.  Default ``1.0`` (distilled model).
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    distilled_lora_strength : float, optional
        Strength for the distilled LoRA (model-only).  Default ``0.5``.
    te_lora_strength : float, optional
        Strength for the Gemma text-encoder LoRA (clip-only).  Default ``1.0``.
    unet_filename : str | None, optional
        Override the default UNet filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the video VAE filename.  When ``None`` the VAE is loaded from
        the UNet checkpoint.  Default ``None``.
    audio_vae_filename : str | None, optional
        Override the audio VAE filename.  When ``None`` falls back to
        ``vae_filename``.  Default ``None``.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.  Default ``None``.
    distilled_lora_filename : str | None, optional
        Override the distilled LoRA filename in ``loras/``.  Default ``None``.
    te_lora_filename : str | None, optional
        Override the TE LoRA filename in ``loras/``.  Default ``None``.
    upscaler_filename : str | None, optional
        Override the upscaler filename in ``latent_upscale_models/``.
        Default ``None``.

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
    from comfy_diffusion.sampling import cfg_guider, get_sampler, manual_sigmas, random_noise, sample_custom
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
    vae_path = Path(vae_filename) if vae_filename else unet_path
    audio_vae_path = Path(audio_vae_filename) if audio_vae_filename else vae_path
    distilled_lora_path = (
        Path(distilled_lora_filename) if distilled_lora_filename
        else models_dir / _DISTILLED_LORA_DEST
    )
    te_lora_path = (
        Path(te_lora_filename) if te_lora_filename else models_dir / _TE_LORA_DEST
    )
    upscaler_path = (
        Path(upscaler_filename) if upscaler_filename else models_dir / _UPSCALER_DEST
    )

    # Load models.
    # The LTX-Video 2.3 dev fp8 checkpoint bundles UNet + VAE.
    ckpt = mm.load_checkpoint_from_path(unet_path)
    model = ckpt.model
    vae = ckpt.vae if vae_filename is None else mm.load_vae(vae_path)
    audio_vae = mm.load_ltxv_audio_vae(audio_vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)
    upscaler_model = mm.load_latent_upscale_model(upscaler_path)

    # Apply LoRAs.
    # Distilled LoRA: model-only (mirrors LoraLoaderModelOnly in the workflow).
    model, _ = apply_lora(model, clip, distilled_lora_path, distilled_lora_strength, 0.0)
    # TE LoRA: clip-only (mirrors LoraLoader with unconnected MODEL output).
    _, clip = apply_lora(model, clip, te_lora_path, 0.0, te_lora_strength)

    # Text conditioning + frame-rate metadata.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    # Create video and audio latents, then concatenate into a single AV latent.
    video_latent = ltxv_empty_latent_video(width=width, height=height, length=length)
    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    # Pass 1 — full sigma schedule, euler_ancestral_cfg_pp.
    guider1 = cfg_guider(model, positive, negative, cfg)
    noise1 = random_noise(seed)
    sigmas1 = manual_sigmas(_SIGMAS_PASS1)
    sampler1 = get_sampler("euler_ancestral_cfg_pp")
    _, denoised1 = sample_custom(noise1, guider1, sampler1, sigmas1, av_latent)

    # Split pass-1 result and spatially upscale the video latent.
    video_latent1, audio_latent1 = ltxv_separate_av_latent(denoised1)
    video_upscaled = ltxv_latent_upsample(video_latent1, upscaler_model, vae)

    # Re-concatenate upscaled video with pass-1 audio latent.
    av_latent2 = ltxv_concat_av_latent(video_upscaled, audio_latent1)

    # Pass 2 — short sigma schedule, euler_cfg_pp.
    guider2 = cfg_guider(model, positive, negative, cfg)
    noise2 = random_noise(seed)
    sigmas2 = manual_sigmas(_SIGMAS_PASS2)
    sampler2 = get_sampler("euler_cfg_pp")
    _, denoised2 = sample_custom(noise2, guider2, sampler2, sigmas2, av_latent2)

    # Separate, decode video and audio.
    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised2)
    frames = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)

    return {"frames": frames, "audio": audio}
