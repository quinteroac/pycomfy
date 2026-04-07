"""Shared constants and helpers for parallax CLI commands."""

from __future__ import annotations

import os
from pathlib import Path


def _get_base_dir() -> Path:
    """Return the parallax base directory, honouring ``PARALLAX_HOME`` if set."""
    parallax_home = os.environ.get("PARALLAX_HOME")
    if parallax_home:
        return Path(parallax_home)
    return Path.home() / ".parallax"


#: Root env directory.  Override by setting ``PARALLAX_HOME`` in your environment.
ENV_DIR: Path = _get_base_dir() / "env"
