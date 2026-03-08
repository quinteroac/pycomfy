"""Public package entrypoint for comfy_diffusion."""

from ._runtime import ensure_comfyui_on_path
from .lora import apply_lora
from .runtime import check_runtime
from .vae import (
    vae_decode,
    vae_decode_batch,
    vae_decode_batch_tiled,
    vae_decode_tiled,
    vae_encode,
    vae_encode_batch,
    vae_encode_batch_tiled,
    vae_encode_tiled,
)

ensure_comfyui_on_path()

__all__ = [
    "check_runtime",
    "vae_decode",
    "vae_decode_batch",
    "vae_decode_batch_tiled",
    "vae_decode_tiled",
    "vae_encode",
    "vae_encode_batch",
    "vae_encode_batch_tiled",
    "vae_encode_tiled",
    "apply_lora",
]
