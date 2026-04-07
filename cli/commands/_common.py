"""Shared constants and helpers for parallax CLI commands."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


def _get_base_dir() -> Path:
    """Return the parallax base directory, honouring ``PARALLAX_HOME`` if set."""
    parallax_home = os.environ.get("PARALLAX_HOME")
    if parallax_home:
        return Path(parallax_home)
    return Path.home() / ".parallax"


#: Root env directory.  Override by setting ``PARALLAX_HOME`` in your environment.
ENV_DIR: Path = _get_base_dir() / "env"


def ensure_env_on_path() -> None:
    """Inject ~/.parallax/env site-packages into sys.path so that comfy_diffusion
    is importable when running as a PyInstaller binary (which excludes it).

    Exits with code 1 and a helpful message if the env is not installed yet.
    """
    import typer

    if not ENV_DIR.exists():
        typer.echo(
            "Error: parallax env not found. Run `parallax install` first.",
            err=True,
        )
        raise typer.Exit(1)

    site_packages = glob.glob(str(ENV_DIR / "lib" / "python*" / "site-packages"))
    if not site_packages:
        typer.echo(
            "Error: site-packages not found in ~/.parallax/env. "
            "Run `parallax install --upgrade` to repair.",
            err=True,
        )
        raise typer.Exit(1)

    for sp in site_packages:
        if sp not in sys.path:
            sys.path.insert(0, sp)
        # Process editable-install .pth files (e.g. __editable__.comfy_diffusion-*.pth).
        # We do this manually instead of site.addsitedir() because the latter processes
        # ALL .pth files which can corrupt PyInstaller's frozen stdlib resolution.
        for pth_file in Path(sp).glob("__editable__*.pth"):
            for line in pth_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and line not in sys.path:
                    sys.path.insert(0, line)
