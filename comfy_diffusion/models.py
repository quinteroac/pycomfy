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
        folder_paths.add_model_folder_path(
            "diffusion_models", str(self.models_dir / "unet"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "diffusion_models", str(self.models_dir / "diffusion_models"), is_default=False
        )
        folder_paths.add_model_folder_path(
            "text_encoders", str(self.models_dir / "text_encoders"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "text_encoders", str(self.models_dir / "clip"), is_default=False
        )
        folder_paths.add_model_folder_path(
            "vae", str(self.models_dir / "vae"), is_default=True
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
        """Load a standalone VAE from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename under the ``vae`` folder.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd
        from comfy import utils as comfy_utils

        p = Path(path)
        if p.is_absolute() and p.is_file():
            vae_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"vae file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            vae_path = folder_paths.get_full_path_or_raise("vae", name)

        state_dict, metadata = comfy_utils.load_torch_file(
            vae_path, return_metadata=True
        )
        vae = comfy_sd.VAE(sd=state_dict, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    def load_clip(
        self,
        path: str | Path,
        *,
        clip_type: str = "stable_diffusion",
    ) -> Any:
        """Load a standalone text encoder (CLIP) from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename under ``text_encoders`` / ``clip``.

        ``clip_type`` selects the encoder architecture (e.g. ``"wan"`` for Wan / UMT5-XXL,
        ``"stable_diffusion"``, ``"sd3"``, ``"flux"``). Must match the model weights.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd

        p = Path(path)
        if p.is_absolute() and p.is_file():
            full_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"clip file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            full_path = folder_paths.get_full_path_or_raise("text_encoders", name)

        clip_type_enum = getattr(
            comfy_sd.CLIPType,
            clip_type.upper(),
            comfy_sd.CLIPType.STABLE_DIFFUSION,
        )
        return comfy_sd.load_clip(
            ckpt_paths=[full_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type_enum,
        )

    def load_clip_vision(self, path: str | Path) -> Any:
        """Load a CLIP vision model from a path or filename."""
        ensure_comfyui_on_path()

        import comfy.clip_vision

        p = Path(path)
        if p.is_absolute() and p.is_file():
            full_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"clip vision file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            candidate = self.models_dir / "clip_vision" / name
            if not candidate.is_file():
                raise FileNotFoundError(f"clip vision file not found: {candidate.resolve()}")
            full_path = str(candidate.resolve())

        return comfy.clip_vision.load(full_path)

    def load_unet(self, path: str | Path) -> Any:
        """Load a standalone diffusion model (UNet) from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename and resolved under the
        ``diffusion_models`` / ``unet`` folders (see ComfyUI folder layout).
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd

        p = Path(path)
        if p.is_absolute() and p.is_file():
            full_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"unet file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            full_path = folder_paths.get_full_path_or_raise("diffusion_models", name)

        return comfy_sd.load_diffusion_model(full_path)

    def load_ltxv_audio_vae(self, path: str | Path) -> object:
        """Load an LTXV audio VAE checkpoint from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename under the ``checkpoints`` folder.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import utils as comfy_utils
        from comfy.ldm.lightricks.vae.audio_vae import AudioVAE

        p = Path(path)
        if p.is_absolute() and p.is_file():
            checkpoint_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"ltxv audio vae file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", name)

        state_dict, metadata = comfy_utils.load_torch_file(
            checkpoint_path, return_metadata=True
        )
        return AudioVAE(state_dict, metadata)

    def load_ltxav_text_encoder(
        self, text_encoder_path: str | Path, checkpoint_path: str | Path
    ) -> object:
        """Load an LTXAV text encoder from two separate files.

        ``text_encoder_path`` is the text encoder file (from ``text_encoders/``).
        ``checkpoint_path`` is the companion checkpoint file (from ``checkpoints/``).
        Both can be absolute paths to existing files or relative filenames resolved
        via folder_paths.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd

        te_p = Path(text_encoder_path)
        if te_p.is_absolute() and te_p.is_file():
            resolved_te = str(te_p.resolve())
        elif te_p.is_absolute():
            raise FileNotFoundError(f"ltxav text encoder file not found: {te_p}")
        else:
            name = text_encoder_path if isinstance(text_encoder_path, str) else te_p.name
            resolved_te = folder_paths.get_full_path_or_raise("text_encoders", name)

        ckpt_p = Path(checkpoint_path)
        if ckpt_p.is_absolute() and ckpt_p.is_file():
            resolved_ckpt = str(ckpt_p.resolve())
        elif ckpt_p.is_absolute():
            raise FileNotFoundError(f"ltxav checkpoint file not found: {ckpt_p}")
        else:
            name = checkpoint_path if isinstance(checkpoint_path, str) else ckpt_p.name
            resolved_ckpt = folder_paths.get_full_path_or_raise("checkpoints", name)

        return comfy_sd.load_clip(
            ckpt_paths=[resolved_te, resolved_ckpt],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy_sd.CLIPType.LTXV,
        )

def model_sampling_flux(
    model: Any,
    max_shift: float,
    min_shift: float,
    width: int,
    height: int,
) -> Any:
    """Patch a model clone with Flux sampling shift settings.

    The shift value is interpolated from ``min_shift`` and ``max_shift`` based on
    latent token count derived from ``width`` and ``height``.
    """
    ensure_comfyui_on_path()
    import comfy.model_sampling

    patched_model = model.clone()

    x1 = 256
    x2 = 4096
    slope = (max_shift - min_shift) / (x2 - x1)
    intercept = min_shift - slope * x1
    latent_tokens = (width * height) / (8 * 8 * 2 * 2)
    shift = latent_tokens * slope + intercept

    model_sampling_type = type(
        "ModelSamplingAdvanced",
        (comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST),
        {},
    )
    model_sampling = model_sampling_type(model.model.model_config)
    model_sampling.set_parameters(shift=shift)
    patched_model.add_object_patch("model_sampling", model_sampling)
    return patched_model


def model_sampling_sd3(model: Any, shift: float) -> Any:
    """Patch a model clone with SD3 sampling shift settings."""
    ensure_comfyui_on_path()
    import comfy.model_sampling

    patched_model = model.clone()

    model_sampling_type = type(
        "ModelSamplingAdvanced",
        (comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST),
        {},
    )
    model_sampling = model_sampling_type(model.model.model_config)
    model_sampling.set_parameters(shift=shift, multiplier=1000)
    patched_model.add_object_patch("model_sampling", model_sampling)
    return patched_model


__all__ = [
    "CheckpointResult",
    "ModelManager",
    "model_sampling_flux",
    "model_sampling_sd3",
]
