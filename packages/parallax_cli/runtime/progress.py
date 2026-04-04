"""Structured progress reporter for pipeline runtime scripts.

Usage::

    from runtime.progress import progress

    progress("download", 0.0)
    download_models(...)
    progress("download", 1.0)
    progress("model_load", 0.0)
    result = run(...)
    progress("done", 1.0)
"""

from __future__ import annotations

import json
import sys


def progress(step: str, pct: float, **kwargs) -> None:
    """Print a structured progress JSON line to stdout.

    Args:
        step: A logical milestone name (e.g. "download", "model_load", "sampling", "done").
        pct:  Progress percentage in [0.0, 1.0].
        **kwargs: Any additional fields to include in the JSON object.
    """
    print(json.dumps({"step": step, "pct": pct, **kwargs}), flush=True)
