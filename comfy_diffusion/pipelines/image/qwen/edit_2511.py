"""Qwen Image Edit 2511 pipeline — image editing with optional Lightning LoRA.

Exports ``manifest()`` and ``run()`` for the Qwen Image Edit 2511 model.

- ``manifest()`` returns a ``list[ModelEntry]`` describing all four model files
  required by the pipeline.  Pass it directly to ``download_models()`` to fetch
  weights before the first inference run.

- ``run()`` executes the full image editing pipeline end-to-end: model loading,
  AuraFlow model sampling patch (shift=3.1), CFGNorm, optional Lightning LoRA,
  input image scaling, Qwen image-edit conditioning for both positive and
  negative prompts, VAE encoding of the scaled image, KSampler denoising, and
  VAE decoding.

  When ``use_lora=True`` (default), the Lightning LoRA
  ``Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`` is applied
  on top of CFGNorm.  This enables fast 4-step turbo inference; callers should
  then set ``steps=4``.  When ``use_lora=False`` the standard 40-step CFGNorm
  path is used instead.

Node execution order (mirrors the official workflow subgraph)
--------------------------------------------------------------
``UNETLoader → CLIPLoader → VAELoader →
ModelSamplingAuraFlow(shift=3.1) → CFGNorm(strength=1) →
[LoraLoaderModelOnly (if use_lora)] →
FluxKontextImageScale(image) →
TextEncodeQwenImageEditPlus(clip, vae, scaled_image, image2, image3, prompt="") →
TextEncodeQwenImageEditPlus(clip, vae, scaled_image, image2, image3, prompt) →
FluxKontextMultiReferenceLatentMethod(negative) →
FluxKontextMultiReferenceLatentMethod(positive) →
VAEEncode(scaled_image) →
KSampler(euler, simple) →
VAEDecode``

Reference workflow
------------------
``comfyui_official_workflows/image/editing/qwen/qwen2511/image_qwen_image_edit_2511.json``

Usage
-----
::

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.qwen.edit_2511 import manifest, run
    from PIL import Image

    download_models(manifest(), models_dir="/path/to/models")

    image = Image.open("input.png")
    results = run(
        prompt="Make the sofa look like it is covered in fur",
        image=image,
        models_dir="/path/to/models",
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

_HF_REPO_QWEN_EDIT = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
_HF_REPO_QWEN_IMAGE = "Comfy-Org/Qwen-Image_ComfyUI"
_HF_REPO_HUNYUAN = "Comfy-Org/HunyuanVideo_1.5_repackaged"
_HF_REPO_LORA = "lightx2v/Qwen-Image-Edit-2511-Lightning"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_DEST = Path("diffusion_models") / "qwen_image_edit_2511_bf16.safetensors"
_CLIP_DEST = Path("text_encoders") / "qwen_2.5_vl_7b_fp8_scaled.safetensors"
_VAE_DEST = Path("vae") / "qwen_image_vae.safetensors"
_LORA_DEST = Path("loras") / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# Workflow defaults.
_DEFAULT_STEPS = 40
_DEFAULT_CFG = 3.0
_DEFAULT_SEED = 0
_DEFAULT_USE_LORA = True
_AURA_FLOW_SHIFT = 3.1
_CFG_NORM_STRENGTH = 1.0
_LORA_STRENGTH = 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the Qwen Image Edit 2511 pipeline.

    Returns exactly 4 :class:`~comfy_diffusion.downloader.HFModelEntry` instances:

    - ``diffusion_models/qwen_image_edit_2511_bf16.safetensors`` — Qwen diffusion model
    - ``text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`` — Qwen 2.5 VL 7B text encoder
    - ``vae/qwen_image_vae.safetensors`` — Qwen Image VAE
    - ``loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`` — Lightning LoRA

    Pass the result directly to
    :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_QWEN_EDIT,
            filename="split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_HUNYUAN,
            filename="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            dest=_CLIP_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_QWEN_IMAGE,
            filename="split_files/vae/qwen_image_vae.safetensors",
            dest=_VAE_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LORA,
            filename="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            dest=_LORA_DEST,
        ),
    ]


def run(
    prompt: str,
    image: Any,
    image2: Any | None = None,
    image3: Any | None = None,
    models_dir: str | Path = "models",
    *,
    steps: int = _DEFAULT_STEPS,
    cfg: float = _DEFAULT_CFG,
    use_lora: bool = _DEFAULT_USE_LORA,
    seed: int = _DEFAULT_SEED,
) -> list[Any]:
    """Run the Qwen Image Edit 2511 pipeline end-to-end.

    Loads the Qwen Image Edit 2511 diffusion model, Qwen 2.5 VL 7B text
    encoder, and Qwen Image VAE.  Applies the AuraFlow model sampling patch
    (shift=3.1), CFGNorm (strength=1), and optionally the Lightning LoRA.
    Scales the input image with ``FluxKontextImageScale``, encodes positive
    and negative conditioning with ``TextEncodeQwenImageEditPlus``, applies
    ``FluxKontextMultiReferenceLatentMethod`` to both conditioning streams,
    VAE-encodes the scaled image, runs the ``euler`` sampler with ``simple``
    scheduler, and decodes the result.

    Parameters
    ----------
    prompt : str
        Text editing instruction (e.g. ``"Change the leather to fur"``).
    image : PIL.Image.Image or torch.Tensor
        Primary input image.  Accepted as PIL :class:`~PIL.Image.Image` or a
        ComfyUI IMAGE tensor (``[B, H, W, C]`` float32).
    image2 : PIL.Image.Image or torch.Tensor or None, optional
        Optional second reference image.
    image3 : PIL.Image.Image or torch.Tensor or None, optional
        Optional third reference image.
    models_dir : str | Path, optional
        Root directory where model weights are stored.  Default ``"models"``.
    steps : int, optional
        Number of denoising steps.  Default ``40``.  Use ``4`` when
        ``use_lora=True`` for Lightning-speed inference.
    cfg : float, optional
        CFG guidance scale.  Default ``3.0``.
    use_lora : bool, optional
        Whether to apply the Lightning LoRA for turbo inference.
        Default ``True``.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.

    Returns
    -------
    list[PIL.Image.Image]
        A list containing the edited output image.

    Raises
    ------
    RuntimeError
        If :func:`~comfy_diffusion.runtime.check_runtime` reports an error.
    """
    from comfy_diffusion.conditioning import (
        apply_flux_kontext_multi_reference,
        encode_qwen_image_edit_plus,
    )
    from comfy_diffusion.image import flux_kontext_image_scale, image_to_tensor
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager, model_sampling_aura_flow
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode, vae_encode_tensor
    from comfy_diffusion.video import apply_cfg_norm

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    unet_path = models_dir / _UNET_DEST
    clip_path = models_dir / _CLIP_DEST
    vae_path = models_dir / _VAE_DEST
    lora_path = models_dir / _LORA_DEST

    # Load models.
    model = mm.load_unet(str(unet_path))
    clip = mm.load_clip(str(clip_path), clip_type="qwen_image")
    vae = mm.load_vae(str(vae_path))

    # Apply AuraFlow model sampling patch (shift=3.1).
    model = model_sampling_aura_flow(model, shift=_AURA_FLOW_SHIFT)

    # Apply CFGNorm.
    model = apply_cfg_norm(model, strength=_CFG_NORM_STRENGTH)

    # Optionally apply Lightning LoRA (LoraLoaderModelOnly: clip strength = 0.0).
    if use_lora:
        model, _clip = apply_lora(model, clip, lora_path, _LORA_STRENGTH, 0.0)

    # Convert PIL images to tensors if needed.
    from PIL import Image as PILImage

    def _to_tensor(img: Any) -> Any:
        if isinstance(img, PILImage.Image):
            return image_to_tensor(img)
        return img

    image_tensor = _to_tensor(image)
    image2_tensor = _to_tensor(image2) if image2 is not None else None
    image3_tensor = _to_tensor(image3) if image3 is not None else None

    # Scale input image to Flux Kontext preferred resolution.
    scaled_image = flux_kontext_image_scale(image_tensor)

    # Encode negative conditioning (empty prompt) with reference images.
    negative = encode_qwen_image_edit_plus(
        clip,
        vae,
        scaled_image,
        image2_tensor,
        image3_tensor,
        prompt="",
    )

    # Encode positive conditioning (editing prompt) with reference images.
    positive = encode_qwen_image_edit_plus(
        clip,
        vae,
        scaled_image,
        image2_tensor,
        image3_tensor,
        prompt=prompt,
    )

    # Apply FluxKontextMultiReferenceLatentMethod to both conditioning streams.
    negative = apply_flux_kontext_multi_reference(negative, "index_timestep_zero")
    positive = apply_flux_kontext_multi_reference(positive, "index_timestep_zero")

    # Encode scaled image to latent.
    latent = vae_encode_tensor(vae, scaled_image)

    # Run KSampler (euler, simple).
    latent_out = sample(
        model,
        positive,
        negative,
        latent,
        steps,
        cfg,
        "euler",
        "simple",
        seed,
    )

    # Decode latent to PIL image.
    output_image = vae_decode(vae, latent_out)
    return [output_image]
