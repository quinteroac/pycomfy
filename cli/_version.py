"""Build-time version constant for the ``parallax`` CLI binary.

This module is the *only* authoritative source of the version string that is
baked into the standalone binary.  The value is overwritten by ``parallax.spec``
at PyInstaller build time to match ``[project].version`` in ``pyproject.toml``,
so that the binary never has to look up an installed package at runtime.

During normal development (``uv run parallax --version``) the value here is
used directly — keep it in sync with ``pyproject.toml``.
"""

__version__ = "1.3.0"
