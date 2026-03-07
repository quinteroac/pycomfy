"""Model management public API.

This module must stay import-safe in CPU-only environments. It intentionally avoids
importing ComfyUI loaders at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._runtime import ensure_comfyui_on_path


@dataclass
class CheckpointResult:
    """Container for objects produced by a ComfyUI checkpoint load."""

    model: Any
    clip: Any | None
    vae: Any | None


class ModelManager:
    """Entry point for model-loading operations.

    The implementation is intentionally deferred to later user stories; this class
    exists now to provide a stable, side-effect-free import surface.
    """

    def __init__(self, models_dir: str | Path) -> None:
        """Store and validate the models directory used by future load operations."""
        path = Path(models_dir)

        if not path.exists():
            raise ValueError(f"models_dir does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"models_dir is not a directory: {path}")

        self.models_dir = path

        ensure_comfyui_on_path()
        import folder_paths

        folder_paths.add_model_folder_path(
            "checkpoints", str(self.models_dir / "checkpoints"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "embeddings", str(self.models_dir / "embeddings"), is_default=True
        )

    def load_checkpoint(self, filename: str) -> CheckpointResult:
        """Load a checkpoint by filename from the configured checkpoints directory."""
        ensure_comfyui_on_path()

        requested_path = (self.models_dir / "checkpoints" / filename).resolve()
        if not requested_path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {requested_path}")

        import folder_paths
        from comfy import sd as comfy_sd

        checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", filename)
        loaded = comfy_sd.load_checkpoint_guess_config(
            checkpoint_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = loaded[:3]
        return CheckpointResult(model=model, clip=clip, vae=vae)

    def load_vae(self, path: str | Path) -> Any:
        """Load a standalone VAE file and return the raw ComfyUI VAE object."""
        ensure_comfyui_on_path()

        vae_path = Path(path).resolve()
        if not vae_path.is_file():
            raise FileNotFoundError(f"vae file not found: {vae_path}")

        from comfy import sd as comfy_sd
        from comfy import utils as comfy_utils

        state_dict, metadata = comfy_utils.load_torch_file(
            str(vae_path), return_metadata=True
        )
        return comfy_sd.VAE(sd=state_dict, metadata=metadata)

    def load_clip(self, path: str | Path) -> Any:
        """Load a standalone CLIP file and return the raw ComfyUI CLIP object."""
        ensure_comfyui_on_path()

        clip_path = Path(path).resolve()
        if not clip_path.is_file():
            raise FileNotFoundError(f"clip file not found: {clip_path}")

        import folder_paths
        from comfy import sd as comfy_sd

        return comfy_sd.load_clip(
            ckpt_paths=[str(clip_path)],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

    def load_unet(self, path: str | Path) -> Any:
        """Load a standalone UNet file and return the raw ComfyUI model object."""
        ensure_comfyui_on_path()

        unet_path = Path(path).resolve()
        if not unet_path.is_file():
            raise FileNotFoundError(f"unet file not found: {unet_path}")

        from comfy import sd as comfy_sd

        return comfy_sd.load_diffusion_model(str(unet_path))


__all__ = ["CheckpointResult", "ModelManager"]
