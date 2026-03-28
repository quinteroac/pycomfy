"""LTX-Video 2.3 (22B distilled) first-last-frame-to-video pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end, faithfully
  mirroring the ``video_ltx2_3_flf2v.json`` official workflow.

Pipeline overview
-----------------
Given a *first frame* and a *last frame* image, the pipeline generates a
video that smoothly transitions between them — including an audio track.

The model is the same LTX-Video 2.3 22B distilled checkpoint used by
:mod:`~comfy_diffusion.pipelines.video.ltx.ltx3.t2v` and
:mod:`~comfy_diffusion.pipelines.video.ltx.ltx3.i2v`.

Sampling chain (mirrors the reference workflow exactly):

1. Both images are preprocessed with ``LTXVPreprocess``.
2. Text conditioning is encoded and enriched with frame-rate metadata via
   ``LTXVConditioning``.
3. The first frame is injected at ``frame_idx=0`` and the last frame at
   ``frame_idx=-1`` via ``LTXVAddGuide`` (strength ``0.7`` each).
4. ``LTXVCropGuides`` trims the appended keyframe latents before sampling.
5. Video and audio latents are concatenated with ``LTXVConcatAVLatent``.
6. ``CFGGuider`` + ``ManualSigmas`` + ``SamplerEulerAncestral`` +
   ``SamplerCustomAdvanced`` drive the denoising loop.
7. ``LTXVSeparateAVLatent`` splits the result back into video and audio.
8. ``VAEDecodeTiled`` decodes video; ``LTXVAudioVAEDecode`` decodes audio.

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
    from comfy_diffusion.pipelines.video.ltx.ltx3.flf2v import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    result = run(
        models_dir="/path/to/models",
        first_image="/path/to/first_frame.png",
        last_image="/path/to/last_frame.png",
        prompt="the camera slowly pans across a sunlit jazz club",
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

# The audio VAE is bundled inside the same checkpoint file.
_AUDIO_VAE_DEST = _UNET_DEST

# ManualSigmas string from the reference workflow.
_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2.3 FLF2V pipeline.

    The LTX 2.3 checkpoint bundles the UNet, video VAE, and audio VAE in a
    single file, so only two distinct downloads are needed.

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
    first_image: Any,
    last_image: Any,
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int | None = None,
    height: int | None = None,
    length: int = 97,
    fps: int = 25,
    cfg: float = 1.0,
    seed: int = 0,
    first_frame_strength: float = 0.7,
    last_frame_strength: float = 0.7,
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
) -> dict[str, Any]:
    """Run the LTX-Video 2.3 first-last-frame-to-video pipeline.

    Both ``first_image`` and ``last_image`` are used as guide frames; the
    model generates a smooth video transition between them together with a
    matching audio track.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    first_image : str | Path | PIL.Image.Image
        Image for the first video frame.  Accepts a file path or a
        :class:`~PIL.Image.Image` instance.
    last_image : str | Path | PIL.Image.Image
        Image for the last video frame.
    prompt : str
        Positive text prompt.  Describe both the visual scene and the desired
        audio content (the model is audio-visual).
    negative_prompt : str, optional
        Negative text prompt.
        Default ``"worst quality, inconsistent motion, blurry, jittery, distorted"``.
    width : int | None, optional
        Output frame width in pixels (must be divisible by 32).  When ``None``
        the width is derived from ``first_image`` (rounded down to the nearest
        multiple of 32).  Default ``None``.
    height : int | None, optional
        Output frame height in pixels (must be divisible by 32).  When
        ``None`` the height is derived from ``first_image``.  Default ``None``.
    length : int, optional
        Number of video frames to generate.  Default ``97`` (≈ 4 s at 25 fps).
    fps : int, optional
        Frame rate used for the audio latent and ``LTXVConditioning``.
        Default ``25``.
    cfg : float, optional
        Classifier-free guidance scale.  Default ``1.0`` (distilled model).
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    first_frame_strength : float, optional
        ``LTXVAddGuide`` strength for the first frame (``[0, 1]``).
        Default ``0.7`` (matches the reference workflow).
    last_frame_strength : float, optional
        ``LTXVAddGuide`` strength for the last frame (``[0, 1]``).
        Default ``0.7``.
    unet_filename : str | None, optional
        Override the default UNet filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the VAE/audio-VAE filename (bundles both).  Default ``None``.
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
    from comfy_diffusion.conditioning import encode_prompt, ltxv_add_guide, ltxv_conditioning, ltxv_crop_guides
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
    # The checkpoint bundles video VAE and audio VAE.
    vae_path = Path(vae_filename) if vae_filename else unet_path

    # Load models.
    model = mm.load_unet(unet_path)
    vae = mm.load_vae(vae_path)
    audio_vae = mm.load_ltxv_audio_vae(vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)

    # Load and preprocess both input images.
    def _load_tensor(img: Any) -> Any:
        if hasattr(img, "mode"):
            return image_to_tensor(img)
        tensor, _ = load_image(img)
        return tensor

    first_tensor = _load_tensor(first_image)
    last_tensor = _load_tensor(last_image)

    # Derive width/height from the first image when not supplied.
    if width is None or height is None:
        # first_tensor shape: (B, H, W, C)
        img_h = first_tensor.shape[1]
        img_w = first_tensor.shape[2]
        if width is None:
            width = (img_w // 32) * 32
        if height is None:
            height = (img_h // 32) * 32

    first_preprocessed = ltxv_preprocess(first_tensor, width, height)
    last_preprocessed = ltxv_preprocess(last_tensor, width, height)

    # Text conditioning + frame-rate metadata.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    # Create the video latent and inject both frame guides.
    # Guide order matters: first add the first frame, then the last frame.
    video_latent = ltxv_empty_latent_video(width=width, height=height, length=length)
    positive, negative, video_latent = ltxv_add_guide(
        positive, negative, vae, video_latent, first_preprocessed,
        frame_idx=0, strength=first_frame_strength,
    )
    positive, negative, video_latent = ltxv_add_guide(
        positive, negative, vae, video_latent, last_preprocessed,
        frame_idx=-1, strength=last_frame_strength,
    )
    # Crop the appended keyframe latents from the conditioning/latent before sampling.
    positive, negative, video_latent = ltxv_crop_guides(positive, negative, video_latent)

    # Create audio latent and concatenate into a single AV latent.
    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    # Build the sampling chain (exact mirror of the reference workflow).
    guider = cfg_guider(model, positive, negative, cfg)
    noise = random_noise(seed)
    sigmas = manual_sigmas(_SIGMAS)
    sampler_obj = get_sampler("euler_ancestral")
    _, denoised = sample_custom(noise, guider, sampler_obj, sigmas, av_latent)

    # Separate video and audio from the denoised AV latent.
    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised)

    # Decode video → PIL frames; decode audio → waveform dict.
    frames = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)

    return {"frames": frames, "audio": audio}
