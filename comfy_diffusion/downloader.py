"""Manifest entry types and download utilities for comfy-diffusion.

Public API
----------
- HFModelEntry   — model file hosted on Hugging Face Hub
- CivitAIModelEntry — model file hosted on CivitAI
- URLModelEntry  — model file at an arbitrary HTTPS/HTTP URL
- ModelEntry     — union alias for the three entry types

``dest`` is always resolved relative to ``models_dir`` when it is a
relative path.  Absolute ``dest`` values are used as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HFModelEntry:
    """A model file downloaded from Hugging Face Hub.

    Parameters
    ----------
    repo_id:
        Repository identifier, e.g. ``"black-forest-labs/FLUX.1-schnell"``.
    filename:
        File path within the repository, e.g. ``"flux1-schnell.safetensors"``.
    dest:
        Destination path.  Relative paths are resolved against ``models_dir``.
    sha256:
        Expected lowercase hex SHA-256 digest.  ``None`` skips verification.
    """

    repo_id: str
    filename: str
    dest: str | Path
    sha256: str | None = field(default=None)


@dataclass
class CivitAIModelEntry:
    """A model file downloaded from CivitAI.

    Parameters
    ----------
    model_id:
        Numeric CivitAI model ID.
    dest:
        Destination path.  Relative paths are resolved against ``models_dir``.
    version_id:
        Specific model-version ID.  ``None`` selects the latest version.
    sha256:
        Expected lowercase hex SHA-256 digest.  ``None`` skips verification.
    """

    model_id: int
    dest: str | Path
    version_id: int | None = field(default=None)
    sha256: str | None = field(default=None)


@dataclass
class URLModelEntry:
    """A model file downloaded from an arbitrary URL.

    Parameters
    ----------
    url:
        Direct download URL (``https://`` or ``http://``).
    dest:
        Destination path.  Relative paths are resolved against ``models_dir``.
    sha256:
        Expected lowercase hex SHA-256 digest.  ``None`` skips verification.
    """

    url: str
    dest: str | Path
    sha256: str | None = field(default=None)


# Union type alias — use as the annotation type for manifest lists.
ModelEntry = HFModelEntry | CivitAIModelEntry | URLModelEntry
