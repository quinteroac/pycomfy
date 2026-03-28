"""LTX-Video 2 text-to-video pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  conditioning, sampling, and VAE decoding.

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
    from comfy_diffusion.pipelines.ltx2_t2v import manifest, run

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    frames = run(
        models_dir="/path/to/models",
        prompt="a golden retriever running through a sunlit park",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# Canonical HuggingFace repository for LTX-Video 2
# ---------------------------------------------------------------------------

_HF_REPO = "Lightricks/LTX-Video"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "ltx-video-2b-v0.9.5.safetensors"
_VAE_DEST = Path("vae") / "ltx-video-2b-v0.9.5-vae-fp32.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "t5xxl_fp16.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the LTX-Video 2 T2V pipeline.

    Each entry is an :class:`~comfy_diffusion.downloader.HFModelEntry` that
    resolves to a deterministic relative path under ``models_dir``.  Pass the
    result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="ltx-video-2b-v0.9.5.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="vae/ltx-video-2b-v0.9.5-vae-fp32.safetensors",
            dest=_VAE_DEST,
        ),
        HFModelEntry(
            repo_id="mcmonkey/google_t5-v1_1-xxl_encoderonly",
            filename="t5xxl_fp16.safetensors",
            dest=_TEXT_ENCODER_DEST,
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
    steps: int = 30,
    cfg: float = 3.0,
    seed: int = 0,
    sampler: str = "euler",
    scheduler: str = "beta",
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
) -> list[Any]:
    """Run the LTX-Video 2 text-to-video pipeline end-to-end.

    Parameters
    ----------
    models_dir:
        Root directory where model weights are stored.
    prompt:
        Positive text prompt describing the desired video content.
    negative_prompt:
        Negative text prompt.  Defaults to a standard quality-rejection string.
    width:
        Output frame width in pixels (must be divisible by 32).
    height:
        Output frame height in pixels (must be divisible by 32).
    length:
        Number of video frames to generate (default 97 ≈ ~4 s at 24 fps).
    steps:
        Number of denoising steps.
    cfg:
        Classifier-free guidance scale.
    seed:
        Random seed for reproducibility.
    sampler:
        Sampler name passed to :func:`~comfy_diffusion.sampling.sample`.
    scheduler:
        Noise scheduler name.
    unet_filename:
        Override the default UNet filename (relative to ``models_dir`` or
        absolute).  When ``None`` the path from :func:`manifest` is used.
    vae_filename:
        Override the default VAE filename.
    text_encoder_filename:
        Override the default text-encoder filename.

    Returns
    -------
    list[PIL.Image.Image]
        Decoded video frames as PIL images, one per generated frame.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.latent import ltxv_empty_latent_video
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample
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
    vae_path = Path(vae_filename) if vae_filename else models_dir / _VAE_DEST
    te_path = Path(text_encoder_filename) if text_encoder_filename else models_dir / _TEXT_ENCODER_DEST

    # Load models.
    model = mm.load_unet(unet_path)
    vae = mm.load_vae(vae_path)
    clip = mm.load_clip(te_path, clip_type="sd3")

    # Text conditioning.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    # Create empty latent.
    latent = ltxv_empty_latent_video(width=width, height=height, length=length)

    # Sample.
    samples = sample(
        model=model,
        positive=positive,
        negative=negative,
        latent_image=latent,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler=scheduler,
        seed=seed,
    )

    # Decode latent → PIL frames.
    frames = vae_decode_batch_tiled(vae, samples)
    return frames
