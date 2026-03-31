"""WAN 2.2 sound-to-video pipeline (S2V 14B, single-model LightX2V LoRA).

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  LoRA application, audio encoding, ``WanSoundImageToVideo`` initial pass,
  multi-pass ``WanSoundImageToVideoExtend`` + ``KSampler`` + ``LatentConcat``
  extension loop, and VAE decoding.

This pipeline mirrors the ``video_wan2_2_14B_s2v.json`` reference workflow
with two active ``Video S2V Extend`` subgraph passes (nodes 79 and 85; node 87
is bypassed in the reference workflow).

Workflow data flow
------------------
1. Load UNet (WAN 2.2 S2V 14B)
2. Apply LightX2V high-noise LoRA (LoraLoaderModelOnly, strength=1.0, clip=0)
3. Apply ModelSamplingSD3(shift=8.0)
4. Load text encoder (UMT5-XXL) + VAE (WAN 2.1) + Audio Encoder (wav2vec2)
5. Encode positive and negative text prompts
6. AudioEncoderEncode — extract audio features from the input waveform
7. WanSoundImageToVideo — initial conditioning + empty latent (length frames)
8. KSampler (uni_pc / simple) — sample initial video segment
9. For each of ``num_extend_passes`` (default 2 active passes from workflow):
   a. WanSoundImageToVideoExtend — advance audio window, build new latent
   b. KSampler — sample next video segment
   c. LatentConcat(accumulated, new_segment, dim='t') — append to video
10. LatentCut(accumulated, 't', index=0, amount=1) — extract first temporal frame
11. LatentConcat(first_frame, accumulated, dim='t') — prepend intro frame
12. VAEDecode → PIL frames

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

    from comfy_diffusion.audio import load_audio
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest, run
    from PIL import Image

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Load inputs.
    audio = load_audio("input.mp3")
    ref_image = Image.open("reference.jpg")

    # 3. Run inference.
    frames = run(
        audio,
        ref_image,
        None,
        "The man is playing the guitar, looking down at his hands",
        models_dir="/path/to/models",
        seed=42,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repository for WAN 2.2
# ---------------------------------------------------------------------------

_HF_REPO_WAN22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# Relative destination paths (resolved against models_dir by download_models).
_AUDIO_ENCODER_DEST = Path("audio_encoders") / "wav2vec2_large_english_fp16.safetensors"
_UNET_DEST = Path("diffusion_models") / "wan2.2_s2v_14B_fp8_scaled.safetensors"
_LORA_DEST = Path("loras") / "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
_VAE_DEST = Path("vae") / "wan_2.1_vae.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the WAN 2.2 S2V pipeline.

    Returns exactly 5 :class:`~comfy_diffusion.downloader.HFModelEntry` items
    matching the 5 active (non-bypassed) model-loading nodes in the reference
    workflow ``video_wan2_2_14B_s2v.json``:

    - ``audio_encoders/wav2vec2_large_english_fp16.safetensors``
      (AudioEncoderLoader, node 57)
    - ``diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors``
      (UNETLoader, node 37)
    - ``loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors``
      (LoraLoaderModelOnly, node 107)
    - ``text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors``
      (CLIPLoader, node 38)
    - ``vae/wan_2.1_vae.safetensors``
      (VAELoader, node 39)

    Bypassed nodes (147, 150, 151, 161) are excluded.

    Pass the result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors",
            dest=_AUDIO_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
            dest=_LORA_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/vae/wan_2.1_vae.safetensors",
            dest=_VAE_DEST,
        ),
    ]


def run(
    audio: dict[str, Any],
    ref_image: Any,
    control_video: Any | None,
    prompt: str,
    negative_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
        "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ),
    *,
    models_dir: str | Path,
    seed: int = 0,
    steps: int = 10,
    cfg: float = 6.0,
    width: int = 640,
    height: int = 640,
    length: int = 77,
    num_extend_passes: int = 2,
    unet_filename: str | None = None,
    lora_filename: str | None = None,
    text_encoder_filename: str | None = None,
    vae_filename: str | None = None,
    audio_encoder_filename: str | None = None,
) -> list[Any]:
    """Run the WAN 2.2 sound-to-video pipeline end-to-end.

    Mirrors the ``video_wan2_2_14B_s2v.json`` reference workflow:
    one initial ``WanSoundImageToVideo`` + ``KSampler`` pass followed by
    ``num_extend_passes`` iterations of ``WanSoundImageToVideoExtend`` +
    ``KSampler`` + ``LatentConcat``, then a ``LatentCut`` +
    ``LatentConcat`` stitch and a final ``VAEDecode``.

    The ``audio`` dict must contain:
    - ``"waveform"``: ``torch.Tensor`` of shape ``[1, C, N]`` (ComfyUI AUDIO
      format).
    - ``"sample_rate"``: ``int`` — samples per second.

    Use :func:`comfy_diffusion.audio.load_audio` to produce this dict from a
    file path.

    Parameters
    ----------
    audio : dict
        ComfyUI AUDIO dict with keys ``"waveform"`` (torch.Tensor,
        shape ``[1, C, N]``) and ``"sample_rate"`` (int).
    ref_image : PIL.Image.Image
        Reference image that anchors the video identity.  Converted to a
        float32 BHWC tensor internally.
    control_video : Any | None
        Optional control video as a ComfyUI IMAGE tensor ``[T, H, W, C]``.
        Pass ``None`` to disable (mirrors the unconnected input in the
        reference workflow).
    prompt : str
        Positive text prompt describing the desired video content.
    negative_prompt : str, optional
        Negative text prompt.  Defaults to the Chinese-language negative prompt
        from the reference workflow.
    models_dir : str | Path
        Root directory where model weights are stored.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    steps : int, optional
        Number of denoising steps per sampling pass.  Default ``10``.
    cfg : float, optional
        CFG (classifier-free guidance) scale.  Default ``6.0``.
    width : int, optional
        Output frame width in pixels.  Default ``640``.
    height : int, optional
        Output frame height in pixels.  Default ``640``.
    length : int, optional
        Number of frames per segment (initial and each extend pass).
        Default ``77`` (≈ 5 s at 16 fps).
    num_extend_passes : int, optional
        Number of ``WanSoundImageToVideoExtend`` passes to run after the
        initial segment.  Default ``2`` (matching the 2 active subgraph
        instances in the reference workflow).
    unet_filename : str | None, optional
        Override the default UNet filename.
    lora_filename : str | None, optional
        Override the default LoRA filename.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.
    vae_filename : str | None, optional
        Override the default VAE filename.
    audio_encoder_filename : str | None, optional
        Override the default audio encoder filename.

    Returns
    -------
    list[PIL.Image.Image]
        Decoded video frames as PIL images, one per frame.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.audio import audio_encoder_encode
    from comfy_diffusion.conditioning import (
        encode_prompt,
        wan_sound_image_to_video,
        wan_sound_image_to_video_extend,
    )
    from comfy_diffusion.image import image_to_tensor
    from comfy_diffusion.latent import latent_concat, latent_cut
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager, model_sampling_sd3
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode_batch

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    # Resolve model paths (allow caller overrides, fall back to manifest paths).
    unet_path = Path(unet_filename) if unet_filename else models_dir / _UNET_DEST
    lora_path = Path(lora_filename) if lora_filename else models_dir / _LORA_DEST
    te_path = (
        Path(text_encoder_filename) if text_encoder_filename else models_dir / _TEXT_ENCODER_DEST
    )
    vae_path = Path(vae_filename) if vae_filename else models_dir / _VAE_DEST
    audio_enc_path = (
        Path(audio_encoder_filename) if audio_encoder_filename else models_dir / _AUDIO_ENCODER_DEST
    )

    # Load models.
    model = mm.load_unet(unet_path)
    clip = mm.load_clip(te_path, clip_type="wan")
    vae = mm.load_vae(vae_path)
    audio_encoder = mm.load_audio_encoder(audio_enc_path)

    # Apply LightX2V LoRA (model-only — clip strength=0.0).
    model, _ = apply_lora(model, clip, lora_path, 1.0, 0.0)

    # Apply ModelSamplingSD3 patch (shift=8) as in the reference workflow.
    model = model_sampling_sd3(model, shift=8.0)

    # Text conditioning.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    # Encode audio into AUDIO_ENCODER_OUTPUT features.
    audio_encoder_output = audio_encoder_encode(audio_encoder, audio)

    # Convert PIL reference image to BHWC float32 tensor.
    ref_image_tensor = image_to_tensor(ref_image)

    # Initial WanSoundImageToVideo — builds conditioning and empty latent.
    pos_init, neg_init, latent_init = wan_sound_image_to_video(
        positive,
        negative,
        vae,
        width=width,
        height=height,
        length=length,
        audio_encoder_output=audio_encoder_output,
        ref_image=ref_image_tensor,
        control_video=control_video,
    )

    # Initial KSampler — sample the first video segment.
    accumulated = sample(
        model,
        pos_init,
        neg_init,
        latent_init,
        steps=steps,
        cfg=cfg,
        sampler_name="uni_pc",
        scheduler="simple",
        seed=seed,
    )

    # Extension loop — each pass advances the audio window and appends frames.
    for _ in range(num_extend_passes):
        pos_ext, neg_ext, latent_ext = wan_sound_image_to_video_extend(
            positive,
            negative,
            vae,
            length=length,
            video_latent=accumulated,
            audio_encoder_output=audio_encoder_output,
            ref_image=ref_image_tensor,
            control_video=control_video,
        )
        new_segment = sample(
            model,
            pos_ext,
            neg_ext,
            latent_ext,
            steps=steps,
            cfg=cfg,
            sampler_name="uni_pc",
            scheduler="simple",
            seed=seed,
        )
        accumulated = latent_concat(accumulated, new_segment, dim="t")

    # Stitch: prepend first temporal frame to the full accumulated latent
    # (mirrors LatentCut(index=0, amount=1) + LatentConcat in the workflow).
    first_frame = latent_cut(accumulated, dim="t", index=0, amount=1)
    final_latent = latent_concat(first_frame, accumulated, dim="t")

    # Decode all frames.
    return vae_decode_batch(vae, final_latent)
