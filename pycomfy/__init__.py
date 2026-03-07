"""Public package entrypoint for pycomfy."""

from ._runtime import ensure_comfyui_on_path
from .runtime import check_runtime
from .vae import vae_decode

ensure_comfyui_on_path()

__all__ = ["check_runtime", "vae_decode"]
