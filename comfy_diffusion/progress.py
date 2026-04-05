"""ProgressReporter — structured NDJSON progress emitter for pipeline authors."""

from __future__ import annotations

import json


class ProgressReporter:
    """Emit structured NDJSON progress lines matching the PythonProgress schema."""

    def update(self, step: str, pct: float, **kwargs) -> None:
        """Print a single JSON line with step, pct, and any optional fields.

        Optional kwargs: frame (int), total (int), error (str).
        """
        payload: dict = {"step": step, "pct": pct}
        for key in ("frame", "total", "output", "error"):
            if key in kwargs:
                payload[key] = kwargs[key]
        print(json.dumps(payload), flush=True)

    def done(self, output_path: str) -> None:
        """Print a final progress line with pct=100.0 and the output path."""
        payload = {"step": "done", "pct": 100.0, "output": output_path}
        print(json.dumps(payload), flush=True)
