"""Qwen Image Layered pipeline — text-to-layers and image-to-layers.

Each pipeline module exports ``manifest()``, ``run_t2l()``, and ``run_i2l()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run_t2l()`` executes the text-to-layers pipeline end-to-end: model loading,
  AuraFlow model sampling patch (shift=1), text conditioning, empty Qwen Image
  Layered latent creation, KSampler denoising, and VAE decoding with
  ``LatentCut`` to extract generated layers.

  .. note::
     The reference ComfyUI workflow uses the ``LatentCutToBatch`` node to
     extract each layer frame.  This pipeline uses ``LatentCut(dim="t")``
     instead, which is functionally equivalent for temporal slicing and avoids
     an additional batch-dimension reshape step.

- ``run_i2l()`` executes the image-to-layers pipeline end-to-end: same as
  ``run_t2l()`` but uses the input image dimensions for the latent, encodes
  the (scaled) reference image with the VAE, and injects the reference latent
  into both positive and negative conditioning via ``ReferenceLatent``.

Both functions mirror the official ComfyUI workflow:
``comfyui_official_workflows/image/editing/qwen/qwen2512/image_qwen_image_layered.json``

Text-to-Layers subgraph node order
------------------------------------
``UNETLoader → CLIPLoader → VAELoader → ModelSamplingAuraFlow(shift=1) →
CLIPTextEncode (positive, with text) → CLIPTextEncode (negative, empty) →
EmptyQwenImageLayeredLatentImage(width, height, layers) →
KSampler(euler, simple) → LatentCut(dim="t", index=1) → VAEDecode``

Image-to-Layers subgraph node order
--------------------------------------
``UNETLoader → CLIPLoader → VAELoader → ModelSamplingAuraFlow(shift=1) →
ImageScaleToMaxDimension(image, "lanczos", 640) →
GetImageSize → EmptyQwenImageLayeredLatentImage(w, h, layers) →
VAEEncode(scaled_image) → CLIPTextEncode (positive, with text) →
CLIPTextEncode (negative, empty) → ReferenceLatent(positive, ref_latent) →
ReferenceLatent(negative, ref_latent) →
KSampler(euler, simple) → LatentCut(dim="t", index=1) → VAEDecode``

Usage
-----
::

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.qwen.layered import manifest, run_t2l, run_i2l

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Text-to-layers.
    images = run_t2l(
        prompt="a beautiful landscape with mountains",
        width=640,
        height=640,
        models_dir="/path/to/models",
    )

    # 3. Image-to-layers.
    from PIL import Image
    img = Image.open("input.png")
    layers = run_i2l(
        prompt="describe the scene",
        image=img,
        models_dir="/path/to/models",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run_t2l", "run_i2l"]

# ---------------------------------------------------------------------------
# HuggingFace repositories
# ---------------------------------------------------------------------------

_HF_REPO_QWEN_LAYERED = "Comfy-Org/Qwen-Image-Layered_ComfyUI"
_HF_REPO_HUNYUAN = "Comfy-Org/HunyuanVideo_1.5_repackaged"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "qwen_image_layered_bf16.safetensors"
_CLIP_DEST = Path("text_encoders") / "qwen_2.5_vl_7b_fp8_scaled.safetensors"
_VAE_DEST = Path("vae") / "qwen_image_layered_vae.safetensors"

# Workflow defaults.
_DEFAULT_STEPS = 20
_DEFAULT_CFG = 2.5
_DEFAULT_SAMPLER = "euler"
_DEFAULT_SCHEDULER = "simple"
_DEFAULT_LAYERS = 2
_DEFAULT_SEED = 0
_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 640
_I2L_MAX_DIMENSION = 640


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the Qwen Image Layered pipeline.

    Returns exactly 3 :class:`~comfy_diffusion.downloader.HFModelEntry` instances:

    - ``diffusion_models/qwen_image_layered_bf16.safetensors`` — Qwen Image Layered diffusion model
    - ``text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`` — Qwen 2.5 VL 7B text encoder
    - ``vae/qwen_image_layered_vae.safetensors`` — Qwen Image Layered VAE

    Pass the result directly to
    :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_QWEN_LAYERED,
            filename="split_files/diffusion_models/qwen_image_layered_bf16.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_HUNYUAN,
            filename="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            dest=_CLIP_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_QWEN_LAYERED,
            filename="split_files/vae/qwen_image_layered_vae.safetensors",
            dest=_VAE_DEST,
        ),
    ]


def run_t2l(
    prompt: str,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    layers: int = _DEFAULT_LAYERS,
    steps: int = _DEFAULT_STEPS,
    cfg: float = _DEFAULT_CFG,
    seed: int = _DEFAULT_SEED,
    models_dir: str | Path = "models",
    *,
    unet_filename: str | None = None,
    clip_filename: str | None = None,
    vae_filename: str | None = None,
) -> list[Any]:
    """Run the Qwen Image Layered text-to-layers pipeline end-to-end.

    Loads the Qwen Image Layered diffusion model, Qwen 2.5 VL 7B text encoder,
    and Qwen Image Layered VAE, applies the AuraFlow model sampling patch
    (shift=1), encodes the prompt, creates an empty Qwen Image Layered latent,
    runs the ``euler`` sampler with ``simple`` scheduler, cuts to get the
    generated layers, and decodes each layer to a PIL image.

    Parameters
    ----------
    prompt : str
        Positive text prompt describing the desired image content.
    width : int, optional
        Output image width in pixels.  Default ``640``.
    height : int, optional
        Output image height in pixels.  Default ``640``.
    layers : int, optional
        Number of generated layers.  Default ``2``.
    steps : int, optional
        Number of denoising steps.  Default ``20``.
    cfg : float, optional
        CFG guidance scale.  Default ``2.5``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    models_dir : str | Path, optional
        Root directory where model weights are stored.  Default ``"models"``.
    unet_filename : str | None, optional
        Override the default diffusion model filename.  Default ``None``.
    clip_filename : str | None, optional
        Override the default text encoder filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the default VAE filename.  Default ``None``.

    Returns
    -------
    list[PIL.Image.Image]
        A list of decoded layer images (one per layer).

    Raises
    ------
    RuntimeError
        If :func:`~comfy_diffusion.runtime.check_runtime` reports an error.
    """
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.latent import empty_qwen_image_layered_latent_image, latent_cut
    from comfy_diffusion.models import ModelManager, model_sampling_aura_flow
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    unet_path = models_dir / (unet_filename or _UNET_DEST)
    clip_path = models_dir / (clip_filename or _CLIP_DEST)
    vae_path = models_dir / (vae_filename or _VAE_DEST)

    # Load models.
    model = mm.load_unet(str(unet_path))
    clip = mm.load_clip(str(clip_path), clip_type="qwen_vl")
    vae = mm.load_vae(str(vae_path))

    # Apply AuraFlow model sampling patch (shift=1) as per workflow.
    model = model_sampling_aura_flow(model, shift=1)

    # Encode conditioning.
    positive, negative = encode_prompt(clip, prompt, "")

    # Create empty Qwen Image Layered latent.
    latent = empty_qwen_image_layered_latent_image(width, height, layers)

    # Run KSampler (euler, simple).
    latent_out = sample(
        model,
        positive,
        negative,
        latent,
        steps,
        cfg,
        _DEFAULT_SAMPLER,
        _DEFAULT_SCHEDULER,
        seed,
    )

    # Cut layers from the output latent along the temporal dimension.
    images = []
    for i in range(layers):
        layer_latent = latent_cut(latent_out, dim="t", index=i + 1, amount=1)
        images.append(vae_decode(vae, layer_latent))

    return images


def run_i2l(
    prompt: str,
    image: Any,
    layers: int = _DEFAULT_LAYERS,
    steps: int = _DEFAULT_STEPS,
    cfg: float = _DEFAULT_CFG,
    seed: int = _DEFAULT_SEED,
    models_dir: str | Path = "models",
    *,
    unet_filename: str | None = None,
    clip_filename: str | None = None,
    vae_filename: str | None = None,
) -> list[Any]:
    """Run the Qwen Image Layered image-to-layers pipeline end-to-end.

    Loads the Qwen Image Layered diffusion model, Qwen 2.5 VL 7B text encoder,
    and Qwen Image Layered VAE.  Scales the input image to a maximum of 640 px
    along the longer edge, derives the latent dimensions from the scaled image
    via ``GetImageSize``, VAE-encodes the scaled image for use as the reference
    latent, applies ``ReferenceLatent`` to both positive and negative
    conditioning, applies the AuraFlow model sampling patch (shift=1), and then
    runs the same sampler pipeline as ``run_t2l``.

    Parameters
    ----------
    prompt : str
        Positive text prompt describing the desired image content.
    image : PIL.Image.Image or torch.Tensor
        Input reference image.  Accepted as PIL :class:`~PIL.Image.Image` or
        a ComfyUI IMAGE tensor (``[B, H, W, C]`` float32).
    layers : int, optional
        Number of generated layers.  Default ``2``.
    steps : int, optional
        Number of denoising steps.  Default ``20``.
    cfg : float, optional
        CFG guidance scale.  Default ``2.5``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    models_dir : str | Path, optional
        Root directory where model weights are stored.  Default ``"models"``.
    unet_filename : str | None, optional
        Override the default diffusion model filename.  Default ``None``.
    clip_filename : str | None, optional
        Override the default text encoder filename.  Default ``None``.
    vae_filename : str | None, optional
        Override the default VAE filename.  Default ``None``.

    Returns
    -------
    list[PIL.Image.Image]
        A list of decoded layer images (one per layer).

    Raises
    ------
    RuntimeError
        If :func:`~comfy_diffusion.runtime.check_runtime` reports an error.
    """
    from comfy_diffusion.conditioning import encode_prompt, reference_latent
    from comfy_diffusion.image import get_image_size, image_scale_to_max_dimension, image_to_tensor
    from comfy_diffusion.latent import empty_qwen_image_layered_latent_image, latent_cut
    from comfy_diffusion.models import ModelManager, model_sampling_aura_flow
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode, vae_encode_tensor

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    unet_path = models_dir / (unet_filename or _UNET_DEST)
    clip_path = models_dir / (clip_filename or _CLIP_DEST)
    vae_path = models_dir / (vae_filename or _VAE_DEST)

    # Load models.
    model = mm.load_unet(str(unet_path))
    clip = mm.load_clip(str(clip_path), clip_type="qwen_vl")
    vae = mm.load_vae(str(vae_path))

    # Apply AuraFlow model sampling patch (shift=1) as per workflow.
    model = model_sampling_aura_flow(model, shift=1)

    # Convert PIL image to tensor if needed, then scale to max 640px.
    from PIL import Image as PILImage

    if isinstance(image, PILImage.Image):
        image_tensor = image_to_tensor(image)
    else:
        image_tensor = image

    scaled_image = image_scale_to_max_dimension(
        image_tensor, "lanczos", _I2L_MAX_DIMENSION
    )

    # Derive latent dimensions from scaled image.
    img_width, img_height = get_image_size(scaled_image)

    # Create empty latent sized to the scaled image.
    latent = empty_qwen_image_layered_latent_image(img_width, img_height, layers)

    # Encode reference image for ReferenceLatent conditioning.
    ref_latent = vae_encode_tensor(vae, scaled_image)

    # Encode text conditioning.
    positive, negative = encode_prompt(clip, prompt, "")

    # Inject reference latent into both positive and negative conditioning.
    positive = reference_latent(positive, ref_latent)
    negative = reference_latent(negative, ref_latent)

    # Run KSampler (euler, simple).
    latent_out = sample(
        model,
        positive,
        negative,
        latent,
        steps,
        cfg,
        _DEFAULT_SAMPLER,
        _DEFAULT_SCHEDULER,
        seed,
    )

    # Cut layers from the output latent along the temporal dimension.
    images = []
    for i in range(layers):
        layer_latent = latent_cut(latent_out, dim="t", index=i + 1, amount=1)
        images.append(vae_decode(vae, layer_latent))

    return images
