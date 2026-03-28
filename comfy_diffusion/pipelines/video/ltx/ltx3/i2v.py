"""LTX-Video 2.3 (22B distilled) image-to-video pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  image preprocessing, conditioning, AV sampling, and decoding.

This pipeline mirrors the ``video_ltx2_3_i2v`` official workflow.  Key
differences from :mod:`comfy_diffusion.pipelines.video.ltx.ltx3.t2v`:

- Accepts an ``image`` parameter as the first conditioning frame.
- The preprocessed image is injected via ``LTXVAddGuide`` at ``frame_idx=0``
  (strength ``1.0``), which is the correct AV-compatible conditioning path.
- Returns **video frames and audio** (the LTX 2.3 model is audio-visual).

Pattern
-------
Every pipeline module in ``comfy_diffusion/pipelines/video/ltx/`` follows this contract:

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
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest, run

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
# Canonical HuggingFace repository for LTX-Video
# ---------------------------------------------------------------------------

_HF_REPO = "Lightricks/LTX-Video"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "ltx-2.3-22b-distilled-fp8.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "gemma_3_12B_it_fp4_mixed.safetensors"

# The audio VAE is bundled inside the UNet checkpoint — no separate file.
_AUDIO_VAE_DEST = _UNET_DEST


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2.3 I2V pipeline.

    Returns the same two entries as :func:`comfy_diffusion.pipelines.video.ltx.ltx3.t2v.manifest`:
    the 22B distilled UNet checkpoint (which bundles the video VAE and audio
    VAE) and the Gemma 3 12B text encoder.

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
    fps: int = 25,
    cfg: float = 1.0,
    seed: int = 0,
    guide_strength: float = 1.0,
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
) -> dict[str, Any]:
    """Run the LTX-Video 2.3 (22B distilled) image-to-video-with-audio pipeline.

    Mirrors the ``video_ltx2_3_i2v`` official workflow.  The first conditioning
    frame is injected via ``LTXVAddGuide`` at ``frame_idx=0``.

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
    guide_strength : float, optional
        Conditioning strength for the first-frame guide (``LTXVAddGuide``
        ``strength`` parameter, range ``[0, 1]``).  Default ``1.0``.
    unet_filename : str | None, optional
        Override the default UNet filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the VAE filename.  When ``None`` the VAE is loaded from the
        UNet checkpoint (bundles UNet + VAE + audio VAE).  Default ``None``.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.  Default ``None``.

    Returns
    -------
    dict[str, Any]
        ``{"frames": list[PIL.Image.Image], "audio": dict[str, Any]}``

        - ``frames`` — decoded video frames as PIL images, one per frame.
        - ``audio`` — decoded audio as ``{"waveform": tensor, "sample_rate": int}``.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.audio import ltxv_audio_vae_decode, ltxv_concat_av_latent, ltxv_empty_latent_audio, ltxv_separate_av_latent
    from comfy_diffusion.conditioning import encode_prompt, ltxv_add_guide, ltxv_conditioning
    from comfy_diffusion.image import image_to_tensor, load_image, ltxv_preprocess
    from comfy_diffusion.latent import ltxv_empty_latent_video
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

    # Resolve model paths.
    unet_path = Path(unet_filename) if unet_filename else models_dir / _UNET_DEST
    te_path = (
        Path(text_encoder_filename)
        if text_encoder_filename
        else models_dir / _TEXT_ENCODER_DEST
    )
    # The checkpoint bundles video VAE and audio VAE; default to the same file.
    vae_path = Path(vae_filename) if vae_filename else unet_path

    # Load models.
    model = mm.load_unet(unet_path)
    vae = mm.load_vae(vae_path)
    audio_vae = mm.load_ltxv_audio_vae(vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)

    # Load and preprocess the input image → BHWC float32 tensor.
    if hasattr(image, "mode"):
        image_tensor = image_to_tensor(image)
    else:
        image_tensor, _ = load_image(image)
    preprocessed = ltxv_preprocess(image_tensor, width, height)

    # Text conditioning + frame-rate metadata.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    # Create video latent and inject the first-frame guide.
    video_latent = ltxv_empty_latent_video(width=width, height=height, length=length)
    positive, negative, video_latent = ltxv_add_guide(
        positive, negative, vae, video_latent, preprocessed,
        frame_idx=0, strength=guide_strength,
    )

    # Create audio latent and concatenate into a single AV latent.
    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    # Build the sampling chain.
    guider = cfg_guider(model, positive, negative, cfg)
    noise = random_noise(seed)
    sigmas = manual_sigmas("1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
    sampler_obj = get_sampler("euler_ancestral")
    _, denoised = sample_custom(noise, guider, sampler_obj, sigmas, av_latent)

    # Separate video and audio from the denoised AV latent.
    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised)

    # Decode video → PIL frames; decode audio → waveform dict.
    frames = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)

    return {"frames": frames, "audio": audio}

