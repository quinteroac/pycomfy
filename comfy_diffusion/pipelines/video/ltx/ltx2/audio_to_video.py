"""LTX-Video 2 audio-to-video pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: audio preprocessing,
  image resize, model loading, conditioning, AV sampling, 4 × video extension
  passes, frame blending, and VIDEO object creation.

The audio-to-video variant uses:

- The distilled BF16 UNet (``ltx-2-19b-distilled_transformer_only_bf16.safetensors``).
- A Gemma 3 12B text encoder loaded via
  :meth:`~comfy_diffusion.models.ModelManager.load_ltxav_text_encoder`.
- An audio VAE (``LTX2_audio_vae_bf16.safetensors``) and video VAE
  (``LTX2_video_vae_bf16.safetensors``) loaded via
  :meth:`~comfy_diffusion.models.ModelManager.load_vae_kj`.
- NAG (Normalized Attention Guidance) via
  :func:`~comfy_diffusion.video.ltx2_nag`.
- ``LTXVImgToVideoInplaceKJ``-style image injection via
  :func:`~comfy_diffusion.video.ltxv_img_to_video_inplace_kj`.
- 4 video extension passes using overlapping AV sampling, blended with
  :func:`~comfy_diffusion.image.image_batch_extend_with_overlap`.

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
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    result = run(
        models_dir="/path/to/models",
        prompt="the man is singing passionately",
        image_path="/path/to/image.png",
        audio_path="/path/to/audio.mp3",
        audio_start_time=65.0,
        audio_end_time=125.0,
    )
    video  = result["video"]             # VIDEO object
    frames = result["frames"]            # list[PIL.Image.Image]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repositories for LTX-Video 2
# ---------------------------------------------------------------------------

_HF_REPO_LTX = "Lightricks/LTX-Video"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "ltx-2-19b-distilled_transformer_only_bf16.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "gemma_3_12B_it_fp8_scaled.safetensors"
_TEXT_ENCODER_2_DEST = Path("text_encoders") / "ltx-2-19b-embeddings_connector_distill_bf16.safetensors"
_AUDIO_VAE_DEST = Path("vae") / "LTX2_audio_vae_bf16.safetensors"
_VIDEO_VAE_DEST = Path("vae") / "LTX2_video_vae_bf16.safetensors"

# Pipeline-level constants
_OVERLAP_FRAMES = 25
_NAG_SCALE = 11.0
_NAG_ALPHA = 0.25
_NAG_TAU = 2.5
_NAG_NEGATIVE_PROMPT = "still image with no motion, subtitles, text, scene change"
_LTXV_SCHEDULER_MAX_SHIFT = 2.05
_LTXV_SCHEDULER_BASE_SHIFT = 0.95
_LTXV_SCHEDULER_TERMINAL = 0.1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2 audio-to-video pipeline.

    Entries correspond to the active (non-bypassed) nodes in
    ``video_ltx_2_audio_to_video.json``:

    - UNet: ``ltx-2-19b-distilled_transformer_only_bf16.safetensors``
    - Text encoder 1: ``gemma_3_12B_it_fp8_scaled.safetensors``
    - Text encoder 2: ``ltx-2-19b-embeddings_connector_distill_bf16.safetensors``
    - Audio VAE: ``LTX2_audio_vae_bf16.safetensors``
    - Video VAE: ``LTX2_video_vae_bf16.safetensors``

    Bypassed ``VHS_VideoCombine`` nodes contribute no entries.

    Pass the result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
            filename="ltx-2-19b-distilled_transformer_only_bf16.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
            filename="split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
            filename="split_files/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
            dest=_TEXT_ENCODER_2_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
            filename="LTX2_audio_vae_bf16.safetensors",
            dest=_AUDIO_VAE_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LTX,
            filename="LTX2_video_vae_bf16.safetensors",
            dest=_VIDEO_VAE_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    prompt: str,
    image_path: Any,
    audio_path: str | Path,
    audio_start_time: float = 0.0,
    audio_end_time: float = 10.0,
    width: int = 704,
    height: int = 704,
    length: int = 241,
    fps: int = 24,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int = 42,
    num_extensions: int = 4,
    overlap_frames: int = _OVERLAP_FRAMES,
    unet_filename: str | None = None,
    text_encoder_filename: str | None = None,
    text_encoder_2_filename: str | None = None,
    audio_vae_filename: str | None = None,
    video_vae_filename: str | None = None,
) -> dict[str, Any]:
    """Run the LTX-Video 2 audio-to-video pipeline end-to-end.

    Executes the full workflow in node order: audio crop → audio separation
    (vocals) → image resize → text encode → LTXV conditioning →
    LTX2_NAG patch → sampling preview override → EmptyLTXVLatentVideo →
    img-to-video inplace (KJ) → audio VAE encode → AV latent concat →
    LTXV schedule → sample → separate AV latent → VAE tiled decode →
    4 × video extension passes (EmptyLTXVLatentVideo → img-to-video → audio
    encode → concat → sample → separate → decode) →
    ImageBatchExtendWithOverlap → CreateVideo → return.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    prompt : str
        Positive text prompt describing the desired video content.
    image_path : str | Path | PIL.Image.Image
        Input image.  Accepts a file path or a :class:`~PIL.Image.Image`.
    audio_path : str | Path
        Path to the input audio file (MP3, WAV, etc.).
    audio_start_time : float, optional
        Start time in seconds to crop the audio.  Default ``0.0``.
    audio_end_time : float, optional
        End time in seconds to crop the audio.  Default ``10.0``.
    width : int, optional
        Output frame width.  Default ``704``.
    height : int, optional
        Output frame height.  Default ``704``.
    length : int, optional
        Number of video frames per sampling segment.  Default ``241``.
    fps : int, optional
        Frame rate.  Default ``24``.
    steps : int, optional
        Denoising steps per segment.  Default ``8``.
    cfg : float, optional
        CFG scale.  Default ``1.0``.
    seed : int, optional
        Random seed.  Default ``42``.
    num_extensions : int, optional
        Number of video extension passes.  Default ``4``.
    overlap_frames : int, optional
        Pixel-frame overlap between consecutive segments.  Default ``25``.
    unet_filename : str | None, optional
        Override UNet filename.
    text_encoder_filename : str | None, optional
        Override text encoder 1 filename.
    text_encoder_2_filename : str | None, optional
        Override text encoder 2 filename.
    audio_vae_filename : str | None, optional
        Override audio VAE filename.
    video_vae_filename : str | None, optional
        Override video VAE filename.

    Returns
    -------
    dict[str, Any]
        ``{"video": VIDEO, "frames": list[PIL.Image.Image]}``

        - ``video`` — VIDEO object (passable to ``get_video_components`` or
          ``save_video``).
        - ``frames`` — all decoded video frames as PIL images.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.audio import (
        audio_crop,
        audio_separation,
        load_audio,
        ltxv_audio_vae_encode,
        ltxv_audio_video_mask,
        ltxv_concat_av_latent,
        ltxv_empty_latent_audio,
        ltxv_separate_av_latent,
        trim_audio_duration,
    )
    from comfy_diffusion.conditioning import (
        conditioning_zero_out,
        encode_prompt,
        ltxv_conditioning,
    )
    from comfy_diffusion.image import (
        image_batch_extend_with_overlap,
        image_from_batch,
        image_resize_kj,
        image_to_tensor,
        load_image,
        ltxv_preprocess,
    )
    from comfy_diffusion.latent import ltxv_empty_latent_video
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import (
        cfg_guider,
        get_sampler,
        ltxv_scheduler,
        random_noise,
        sample_custom,
    )
    from comfy_diffusion.vae import vae_decode_batch_tiled
    from comfy_diffusion.video import (
        create_video,
        ltx2_nag,
        ltx2_sampling_preview_override,
        ltxv_chunk_feed_forward,
        ltxv_img_to_video_inplace_kj,
    )

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    models_dir = Path(models_dir)
    manifest_entries = manifest()

    unet_path = Path(unet_filename) if unet_filename else models_dir / _UNET_DEST
    te_path = Path(text_encoder_filename) if text_encoder_filename else models_dir / _TEXT_ENCODER_DEST
    te2_path = Path(text_encoder_2_filename) if text_encoder_2_filename else models_dir / _TEXT_ENCODER_2_DEST
    audio_vae_path = Path(audio_vae_filename) if audio_vae_filename else models_dir / _AUDIO_VAE_DEST
    video_vae_path = Path(video_vae_filename) if video_vae_filename else models_dir / _VIDEO_VAE_DEST

    # -----------------------------------------------------------------------
    # Step 1: Audio pre-processing — crop → separate (vocals) → trim to segment
    # -----------------------------------------------------------------------
    segment_seconds = (length - 1) / fps
    full_audio = load_audio(audio_path)
    cropped_audio = audio_crop(full_audio, audio_start_time, audio_end_time)
    vocals_audio = audio_separation(cropped_audio, mode="harmonic")
    segment_audio = trim_audio_duration(vocals_audio, start=0.0, duration=segment_seconds)

    # -----------------------------------------------------------------------
    # Step 2: Image resize
    # -----------------------------------------------------------------------
    if hasattr(image_path, "mode"):
        image_tensor = image_to_tensor(image_path)
    else:
        image_tensor, _ = load_image(image_path)

    resized_image, out_w, out_h = image_resize_kj(
        image_tensor, width=width, height=height, upscale_method="lanczos",
        keep_proportion="crop", divisible_by=2,
    )
    preprocessed = ltxv_preprocess(resized_image, out_w, out_h)

    # -----------------------------------------------------------------------
    # Step 3: Load models
    # -----------------------------------------------------------------------
    mm = ModelManager(models_dir)
    model = mm.load_unet(unet_path)
    clip = mm.load_ltxav_text_encoder(te_path, te2_path)
    audio_vae = mm.load_vae_kj(audio_vae_path, device="main_device", dtype="bf16")
    video_vae = mm.load_vae_kj(video_vae_path, device="main_device", dtype="bf16")

    # -----------------------------------------------------------------------
    # Step 4: Text conditioning
    # -----------------------------------------------------------------------
    positive_raw = encode_prompt(clip, prompt)
    negative_raw = conditioning_zero_out(positive_raw)
    positive, negative = ltxv_conditioning(positive_raw, negative_raw, frame_rate=fps)

    # -----------------------------------------------------------------------
    # Step 5: Model patches — chunk feed-forward → NAG → preview override
    # -----------------------------------------------------------------------
    model = ltxv_chunk_feed_forward(model, min_chunk_size=4, chunk_threshold=4096)
    nag_cond = encode_prompt(clip, _NAG_NEGATIVE_PROMPT)
    model = ltx2_nag(
        model,
        nag_scale=_NAG_SCALE,
        nag_alpha=_NAG_ALPHA,
        nag_tau=_NAG_TAU,
        nag_cond_video=nag_cond,
    )
    model = ltx2_sampling_preview_override(model, preview_rate=8)

    sampler_obj = get_sampler("lcm")
    guider = cfg_guider(model, positive, negative, cfg)
    noise = random_noise(seed)

    def _sample_segment(
        video_latent: Any, audio_latent: Any, noise_obj: Any
    ) -> tuple[Any, Any]:
        """Concatenate AV, schedule, sample, and separate."""
        av_latent = ltxv_concat_av_latent(video_latent, audio_latent)
        sigmas = ltxv_scheduler(
            steps=steps,
            max_shift=_LTXV_SCHEDULER_MAX_SHIFT,
            base_shift=_LTXV_SCHEDULER_BASE_SHIFT,
            stretch=True,
            terminal=_LTXV_SCHEDULER_TERMINAL,
            latent=av_latent,
        )
        _, denoised = sample_custom(noise_obj, guider, sampler_obj, sigmas, av_latent)
        return ltxv_separate_av_latent(denoised)

    # -----------------------------------------------------------------------
    # Step 6: Initial pass — EmptyLTXVLatentVideo → img inject → audio encode
    # -----------------------------------------------------------------------
    video_latent = ltxv_empty_latent_video(width=out_w, height=out_h, length=length)
    video_latent = ltxv_img_to_video_inplace_kj(video_vae, video_latent, preprocessed, index=0)
    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    audio_encoded = ltxv_audio_vae_encode(audio_vae, segment_audio)
    audio_latent.update(audio_encoded)

    video_lat_out, _audio_lat_out = _sample_segment(video_latent, audio_latent, noise)
    all_frames = vae_decode_batch_tiled(video_vae, video_lat_out)

    # -----------------------------------------------------------------------
    # Step 7: 4 × video extension passes
    # -----------------------------------------------------------------------
    overlap_seconds = overlap_frames / fps
    step_seconds = segment_seconds - overlap_seconds

    for ext_idx in range(num_extensions):
        # Compute audio window for this extension segment
        ext_audio_start = (ext_idx + 1) * step_seconds
        ext_audio_end = ext_audio_start + segment_seconds
        if ext_audio_start >= (audio_end_time - audio_start_time):
            break  # no more audio to drive extension

        # Trim audio for this extension window
        ext_duration = min(segment_seconds, (audio_end_time - audio_start_time) - ext_audio_start)
        ext_vocals = trim_audio_duration(vocals_audio, start=ext_audio_start, duration=ext_duration)

        # Get last `overlap_frames` from previous decode as the guide image
        last_frames = image_from_batch(all_frames, batch_index=-overlap_frames, length=overlap_frames)
        # Use the very last frame as the single injection image
        guide_frame = image_from_batch(all_frames, batch_index=-1, length=1)

        # Build extension video latent
        ext_video_latent = ltxv_empty_latent_video(width=out_w, height=out_h, length=length)
        ext_video_latent = ltxv_img_to_video_inplace_kj(
            video_vae, ext_video_latent, guide_frame, index=0
        )

        # Encode extension audio
        ext_audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
        ext_audio_encoded = ltxv_audio_vae_encode(audio_vae, ext_vocals)
        ext_audio_latent.update(ext_audio_encoded)

        # Apply AV masks for extension (preserve overlap region)
        ext_video_latent, ext_audio_latent = ltxv_audio_video_mask(
            ext_video_latent,
            ext_audio_latent,
            video_fps=float(fps),
            video_end_time=ext_audio_end,
            audio_start_time=ext_audio_start,
            audio_end_time=ext_audio_end,
            num_video_frames_to_guide=overlap_frames,
            audio_overlap_latents=overlap_frames,
        )

        ext_video_lat_out, _ = _sample_segment(ext_video_latent, ext_audio_latent, noise)
        ext_frames = vae_decode_batch_tiled(video_vae, ext_video_lat_out)

        # Blend extension frames with accumulated frames
        all_frames = image_batch_extend_with_overlap(
            all_frames, ext_frames, overlap=overlap_frames, overlap_mode="filmic_crossfade"
        )

    # -----------------------------------------------------------------------
    # Step 8: Create VIDEO object
    # -----------------------------------------------------------------------
    video_obj = create_video(all_frames, cropped_audio, fps=float(fps))

    # Convert tensor frames to PIL images
    import torch

    frames_list = []
    frames_tensor = all_frames.detach().cpu()
    frames_np = (frames_tensor.numpy() * 255.0).clip(0, 255).astype("uint8")
    from PIL import Image

    for frame in frames_np:
        frames_list.append(Image.fromarray(frame))

    return {"video": video_obj, "frames": frames_list}
