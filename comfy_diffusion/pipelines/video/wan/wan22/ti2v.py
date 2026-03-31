"""WAN 2.2 text-and-image-to-video pipeline (TI2V 5B, single-model KSampler).

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  ``ModelSamplingSD3`` (shift=8), text conditioning, ``Wan22ImageToVideoLatent``
  (with optional start image), ``KSampler`` (uni_pc, 20 steps, cfg=5), and
  VAE decoding.

This pipeline mirrors the ``video_wan2_2_5B_ti2v.json`` reference workflow:
a single ``KSampler`` pass using the WAN 2.2 TI2V 5B model with
``ModelSamplingSD3`` (shift=8) patched.

Workflow data flow
------------------
1. Load UNet (WAN 2.2 TI2V 5B fp16)
2. Load text encoder (UMT5-XXL) + VAE (WAN 2.2)
3. Apply ModelSamplingSD3(shift=8) to the UNet
4. Encode positive and negative text prompts
5. Wan22ImageToVideoLatent — build TI2V latent (optionally conditioned on a
   start image); when ``start_image`` is ``None`` an empty latent is used
6. KSampler (uni_pc / simple, 20 steps, cfg=5, denoise=1.0) — full denoising
7. VAEDecode → PIL frames

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
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import manifest, run
    from PIL import Image

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference (text-only — empty latent).
    frames = run(
        "a serene mountain landscape with gentle wind",
        models_dir="/path/to/models",
    )

    # 3. Run inference conditioned on a reference image.
    start = Image.open("reference.jpg")
    frames = run(
        "a serene mountain landscape with gentle wind",
        models_dir="/path/to/models",
        start_image=start,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repositories
# ---------------------------------------------------------------------------

_HF_REPO_WAN22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
_HF_REPO_WAN21 = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "wan2.2_ti2v_5B_fp16.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
_VAE_DEST = Path("vae") / "wan2.2_vae.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the WAN 2.2 TI2V pipeline.

    Returns exactly 3 :class:`~comfy_diffusion.downloader.HFModelEntry` items
    matching the 3 active (non-bypassed) model-loading nodes in the reference
    workflow ``video_wan2_2_5B_ti2v.json``:

    - ``diffusion_models/wan2.2_ti2v_5B_fp16.safetensors``
      (UNETLoader, node 37)
    - ``text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors``
      (CLIPLoader, node 38)
    - ``vae/wan2.2_vae.safetensors``
      (VAELoader, node 39)

    Pass the result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN21,
            filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/vae/wan2.2_vae.safetensors",
            dest=_VAE_DEST,
        ),
    ]


def run(
    prompt: str,
    negative_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
        "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ),
    width: int = 1280,
    height: int = 704,
    length: int = 121,
    *,
    start_image: Any | None = None,
    models_dir: str | Path,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 5.0,
    unet_filename: str | None = None,
    text_encoder_filename: str | None = None,
    vae_filename: str | None = None,
) -> list[Any]:
    """Run the WAN 2.2 TI2V pipeline end-to-end.

    Mirrors the ``video_wan2_2_5B_ti2v.json`` reference workflow: a single
    ``KSampler`` (uni_pc, 20 steps, cfg=5) pass with ``ModelSamplingSD3``
    (shift=8) patched and ``Wan22ImageToVideoLatent`` for latent construction.

    Parameters
    ----------
    prompt : str
        Positive text prompt describing the desired video content.
    negative_prompt : str, optional
        Negative text prompt.  Defaults to the Chinese-language negative prompt
        from the reference workflow.
    width : int, optional
        Output frame width in pixels.  Default ``1280``.
    height : int, optional
        Output frame height in pixels.  Default ``704``.
    length : int, optional
        Number of video frames to generate.  Default ``121``.
    start_image : PIL.Image.Image | None, optional
        Optional reference image to condition the video.  When ``None``
        (default) an empty latent is used and no mask is injected.
        When provided the image is encoded via the VAE and the resulting
        latent is used as the starting point for generation.
    models_dir : str | Path
        Root directory where model weights are stored.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    steps : int, optional
        Number of denoising steps.  Default ``20``.
    cfg : float, optional
        CFG (classifier-free guidance) scale.  Default ``5.0``.
    unet_filename : str | None, optional
        Override the default UNet filename.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.
    vae_filename : str | None, optional
        Override the default VAE filename.

    Returns
    -------
    list[PIL.Image.Image]
        Decoded video frames as PIL images, one per frame.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.conditioning import encode_prompt, wan22_image_to_video_latent
    from comfy_diffusion.image import image_to_tensor
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
    te_path = (
        Path(text_encoder_filename) if text_encoder_filename else models_dir / _TEXT_ENCODER_DEST
    )
    vae_path = Path(vae_filename) if vae_filename else models_dir / _VAE_DEST

    # Load models.
    model = mm.load_unet(unet_path)
    clip = mm.load_clip(te_path, clip_type="wan")
    vae = mm.load_vae(vae_path)

    # Apply ModelSamplingSD3 patch (shift=8) as in the reference workflow.
    model = model_sampling_sd3(model, shift=8.0)

    # Text conditioning.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    # Convert PIL start image to BHWC float32 tensor when provided.
    start_image_tensor = image_to_tensor(start_image) if start_image is not None else None

    # Wan22ImageToVideoLatent — builds TI2V latent (empty when start_image is None).
    latent = wan22_image_to_video_latent(
        vae,
        width=width,
        height=height,
        length=length,
        start_image=start_image_tensor,
    )

    # KSampler — full denoising pass (uni_pc, simple, cfg=5, denoise=1.0).
    latent = sample(
        model,
        positive,
        negative,
        latent,
        steps=steps,
        cfg=cfg,
        sampler_name="uni_pc",
        scheduler="simple",
        seed=seed,
    )

    # Decode to frames.
    return vae_decode_batch(vae, latent)
