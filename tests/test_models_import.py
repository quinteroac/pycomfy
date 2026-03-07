"""Tests for US-001 import safety of pycomfy.models."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    repo_root = str(_repo_root())
    env["PYTHONPATH"] = repo_root if not existing else f"{repo_root}{os.pathsep}{existing}"

    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=env,
        cwd=_repo_root(),
    )


def test_model_manager_import_succeeds_on_cpu_only_machine() -> None:
    result = _run_python(
        "from pycomfy.models import ModelManager; "
        "assert ModelManager.__name__ == 'ModelManager'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_import_has_no_additional_side_effects_beyond_import_pycomfy() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import pycomfy\n"
        "baseline_path = list(sys.path)\n"
        "baseline_modules = set(sys.modules)\n"
        "from pycomfy.models import ModelManager\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'path_unchanged': baseline_path == list(sys.path),\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'comfy_sd_loaded': 'comfy.sd' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "  'class_name': ModelManager.__name__,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["class_name"] == "ModelManager"
    assert payload["path_unchanged"] is True
    assert payload["torch_loaded"] is False
    assert payload["comfy_sd_loaded"] is False
    # Only pycomfy.models and lightweight stdlib helpers (e.g. dataclasses) may be added.
    # Heavy modules (torch, comfy.*) must not appear.
    heavy = [m for m in payload["new_modules"] if m.startswith(("torch", "comfy", "numpy"))]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
