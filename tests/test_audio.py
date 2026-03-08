"""Tests for LTXV audio VAE helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pycomfy.audio as audio_module
from pycomfy.audio import ltxv_audio_vae_decode, ltxv_audio_vae_encode


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


def test_audio_module_exports_ltxv_audio_vae_helpers() -> None:
    assert audio_module.__all__ == ["ltxv_audio_vae_encode", "ltxv_audio_vae_decode"]


def test_ltxv_audio_vae_encode_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_audio_vae_encode)
    assert str(signature) == "(vae: '_LtxvAudioVaeEncoder', audio: 'Any') -> 'Any'"


def test_ltxv_audio_vae_encode_calls_vae_encode_and_returns_raw_latent() -> None:
    expected_audio = object()
    expected_latent = object()

    class FakeAudioVae:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def encode(self, audio: Any) -> Any:
            self.calls.append(audio)
            return expected_latent

    vae = FakeAudioVae()
    result = ltxv_audio_vae_encode(vae, expected_audio)

    assert result is expected_latent
    assert vae.calls == [expected_audio]


def test_ltxv_audio_vae_decode_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_audio_vae_decode)
    assert str(signature) == "(vae: '_LtxvAudioVaeDecoder', latent: 'Any') -> 'Any'"


def test_ltxv_audio_vae_decode_calls_vae_decode_and_returns_raw_audio_tensor() -> None:
    expected_latent = object()
    expected_audio_tensor = object()

    class FakeAudioVae:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def decode(self, latent: Any) -> Any:
            self.calls.append(latent)
            return expected_audio_tensor

    vae = FakeAudioVae()
    result = ltxv_audio_vae_decode(vae, expected_latent)

    assert result is expected_audio_tensor
    assert vae.calls == [expected_latent]


def test_ltxv_audio_vae_encode_is_importable_from_pycomfy_audio() -> None:
    result = _run_python(
        "from pycomfy.audio import ltxv_audio_vae_encode; "
        "assert ltxv_audio_vae_encode.__name__ == 'ltxv_audio_vae_encode'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_ltxv_audio_vae_decode_is_importable_from_pycomfy_audio() -> None:
    result = _run_python(
        "from pycomfy.audio import ltxv_audio_vae_decode; "
        "assert ltxv_audio_vae_decode.__name__ == 'ltxv_audio_vae_decode'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_import_pycomfy_audio_has_no_torch_or_comfy_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import pycomfy\n"
        "baseline_modules = set(sys.modules)\n"
        "from pycomfy.audio import ltxv_audio_vae_decode, ltxv_audio_vae_encode\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'func_name': ltxv_audio_vae_encode.__name__,\n"
        "  'decode_func_name': ltxv_audio_vae_decode.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'comfy_loaded': 'comfy' in sys.modules,\n"
        "  'comfy_sd_loaded': 'comfy.sd' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["func_name"] == "ltxv_audio_vae_encode"
    assert payload["decode_func_name"] == "ltxv_audio_vae_decode"
    assert payload["torch_loaded"] is False
    assert payload["comfy_loaded"] is False
    assert payload["comfy_sd_loaded"] is False
    heavy = [module for module in payload["new_modules"] if module.startswith(("torch", "comfy"))]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
