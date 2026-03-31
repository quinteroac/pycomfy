"""WAN 2.1 text-to-video pipeline (1.3B model).

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  conditioning, sampling, and VAE decoding.

This pipeline mirrors the ``text_to_video_wan.json`` reference workflow:
single-pass sampling with the WAN 2.1 T2V 1.3B model, UMT5-XXL text encoder,
and the WAN 2.1 VAE.  The ``ModelSamplingSD3`` patch (shift=8) is applied
before sampling.

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
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    frames = run(
        models_dir="/path/to/models",
        prompt="a fox moving quickly in a beautiful winter scenery",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repository for WAN 2.1
# ---------------------------------------------------------------------------

_HF_REPO_WAN = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "wan2.1_t2v_1.3B_fp16.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
_VAE_DEST = Path("vae") / "wan_2.1_vae.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the WAN 2.1 T2V pipeline.

    Each entry is an :class:`~comfy_diffusion.downloader.HFModelEntry` that
    resolves to a deterministic relative path under ``models_dir``.  Pass the
    result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_WAN,
            filename="split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN,
            filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN,
            filename="split_files/vae/wan_2.1_vae.safetensors",
            dest=_VAE_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    prompt: str,
    negative_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
        "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ),
    width: int = 832,
    height: int = 480,
    length: int = 33,
    fps: int = 16,
    steps: int = 30,
    cfg: float = 6.0,
    seed: int = 0,
    unet_filename: str | None = None,
    text_encoder_filename: str | None = None,
    vae_filename: str | None = None,
) -> list[Any]:
    """Run the WAN 2.1 text-to-video pipeline end-to-end.

    Single-pass sampling mirroring the ``text_to_video_wan.json`` reference
    workflow: UNet loaded via ``UNETLoader``, UMT5-XXL text encoder via
    ``CLIPLoader`` (clip_type="wan"), WAN 2.1 VAE, and ``ModelSamplingSD3``
    (shift=8) applied before sampling with ``uni_pc`` / ``simple``.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    prompt : str
        Positive text prompt describing the desired video content.
    negative_prompt : str, optional
        Negative text prompt.  Defaults to the Chinese-language negative prompt
        from the reference workflow.
    width : int, optional
        Output frame width in pixels.  Default ``832``.
    height : int, optional
        Output frame height in pixels.  Default ``480``.
    length : int, optional
        Number of video frames to generate.  Default ``33``.
    fps : int, optional
        Frame rate used when assembling the output video.  Default ``16``.
    steps : int, optional
        Number of denoising steps.  Default ``30``.
    cfg : float, optional
        CFG (classifier-free guidance) scale.  Default ``6.0``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    unet_filename : str | None, optional
        Override the default UNet filename.  Default ``None``.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the default VAE filename.  Default ``None``.

    Returns
    -------
    list[PIL.Image.Image]
        Decoded video frames as PIL images, one per frame.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.latent import empty_wan_latent_video
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

    # Text conditioning.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    # Create empty video latent.
    latent = empty_wan_latent_video(width=width, height=height, length=length)

    # Apply ModelSamplingSD3 patch (shift=8) as in the reference workflow.
    patched_model = model_sampling_sd3(model, shift=8.0)

    # Sample.
    sampled = sample(
        patched_model,
        positive,
        negative,
        latent,
        steps,
        cfg,
        "uni_pc",
        "simple",
        seed,
    )

    # Decode to frames.
    return vae_decode_batch(vae, sampled)
