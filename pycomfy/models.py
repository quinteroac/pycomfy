"""Model management public API.

This module must stay import-safe in CPU-only environments. It intentionally avoids
importing ComfyUI loaders at module import time.
"""

from __future__ import annotations

from pathlib import Path


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


__all__ = ["ModelManager"]
