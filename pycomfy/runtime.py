"""Runtime diagnostics for pycomfy."""

from __future__ import annotations

from typing import Any
import sys


def check_runtime() -> dict[str, Any]:
    """Return basic runtime diagnostics for the current Python process."""
    return {
        "ok": True,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
    }
