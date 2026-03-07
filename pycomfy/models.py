"""Model management public API.

This module must stay import-safe in CPU-only environments. It intentionally avoids
importing ComfyUI loaders at module import time.
"""

from __future__ import annotations


class ModelManager:
    """Entry point for model-loading operations.

    The implementation is intentionally deferred to later user stories; this class
    exists now to provide a stable, side-effect-free import surface.
    """


__all__ = ["ModelManager"]
