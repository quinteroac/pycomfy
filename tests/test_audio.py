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
from pycomfy.audio import (
    encode_ace_step_15_audio,
    ltxv_audio_vae_decode,
    ltxv_audio_vae_encode,
    ltxv_empty_latent_audio,
)


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
    assert audio_module.__all__ == [
        "ltxv_audio_vae_encode",
        "ltxv_audio_vae_decode",
        "ltxv_empty_latent_audio",
        "encode_ace_step_15_audio",
    ]


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


def test_ltxv_empty_latent_audio_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_empty_latent_audio)
    assert (
        str(signature)
        == "(audio_vae: '_LtxvAudioVae', frames_number: 'int', "
        "frame_rate: 'int' = 25, batch_size: 'int' = 1) -> 'dict[str, Any]'"
    )


def test_ltxv_empty_latent_audio_wraps_node_execute_and_returns_audio_latent_dict(
    monkeypatch: Any,
) -> None:
    expected_samples = object()

    class FakeAudioVae:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []
            self.sample_rate = 48000
            self.latent_channels = 32
            self.latent_frequency_bins = 128

        def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int:
            self.calls.append((frames_number, frame_rate))
            return 123

    class FakeLtxvEmptyLatentAudio:
        @classmethod
        def execute(
            cls,
            *,
            frames_number: int,
            frame_rate: int,
            batch_size: int,
            audio_vae: Any,
        ) -> tuple[dict[str, Any]]:
            num_latents = audio_vae.num_of_latents_from_frames(frames_number, frame_rate)
            return (
                {
                    "samples": expected_samples,
                    "sample_rate": int(audio_vae.sample_rate),
                    "type": "audio",
                    "num_latents": num_latents,
                    "batch_size": batch_size,
                },
            )

    monkeypatch.setattr(
        audio_module,
        "_get_ltxv_empty_latent_audio_type",
        lambda: FakeLtxvEmptyLatentAudio,
    )

    audio_vae = FakeAudioVae()
    result = ltxv_empty_latent_audio(
        audio_vae=audio_vae,
        frames_number=97,
        frame_rate=25,
        batch_size=2,
    )

    assert result["samples"] is expected_samples
    assert result["sample_rate"] == 48000
    assert result["type"] == "audio"
    assert result["num_latents"] == 123
    assert result["batch_size"] == 2
    assert audio_vae.calls == [(97, 25)]


def test_encode_ace_step_15_audio_signature_matches_contract() -> None:
    signature = inspect.signature(encode_ace_step_15_audio)
    assert (
        str(signature)
        == "(clip: '_AceStep15Clip', tags: 'str', lyrics: 'str' = '', seed: 'int' = 0, "
        "bpm: 'int' = 120, duration: 'float' = 120.0, timesignature: 'str' = '4', "
        "language: 'str' = 'en', keyscale: 'str' = 'C major', "
        "generate_audio_codes: 'bool' = True, cfg_scale: 'float' = 2.0, "
        "temperature: 'float' = 0.85, top_p: 'float' = 0.9, top_k: 'int' = 0, "
        "min_p: 'float' = 0.0) -> 'Any'"
    )
    assert "sample_rate" not in signature.parameters


def test_encode_ace_step_15_audio_wraps_tokenize_and_encode_from_tokens_scheduled() -> None:
    expected_tokens = object()
    expected_conditioning = object()

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[tuple[str, dict[str, Any]]] = []
            self.encode_calls: list[Any] = []

        def tokenize(self, tags: str, **kwargs: Any) -> Any:
            self.tokenize_calls.append((tags, kwargs))
            return expected_tokens

        def encode_from_tokens_scheduled(self, tokens: Any) -> Any:
            self.encode_calls.append(tokens)
            return expected_conditioning

    clip = FakeClip()
    result = encode_ace_step_15_audio(
        clip=clip,
        tags="electronic dance anthem",
        lyrics="we rise at dawn",
        seed=42,
        bpm=128,
        duration=91.5,
        timesignature="6",
        language="en",
        keyscale="A minor",
        generate_audio_codes=False,
        cfg_scale=3.5,
        temperature=0.7,
        top_p=0.8,
        top_k=10,
        min_p=0.05,
    )

    assert result is expected_conditioning
    assert clip.tokenize_calls == [
        (
            "electronic dance anthem",
            {
                "lyrics": "we rise at dawn",
                "bpm": 128,
                "duration": 91.5,
                "timesignature": 6,
                "language": "en",
                "keyscale": "A minor",
                "seed": 42,
                "generate_audio_codes": False,
                "cfg_scale": 3.5,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 10,
                "min_p": 0.05,
            },
        )
    ]
    assert clip.encode_calls == [expected_tokens]


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


def test_ltxv_empty_latent_audio_is_importable_from_pycomfy_audio() -> None:
    result = _run_python(
        "from pycomfy.audio import ltxv_empty_latent_audio; "
        "assert ltxv_empty_latent_audio.__name__ == 'ltxv_empty_latent_audio'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_encode_ace_step_15_audio_is_importable_from_pycomfy_audio() -> None:
    result = _run_python(
        "from pycomfy.audio import encode_ace_step_15_audio; "
        "assert encode_ace_step_15_audio.__name__ == 'encode_ace_step_15_audio'; "
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
        "from pycomfy.audio import (\n"
        "  encode_ace_step_15_audio,\n"
        "  ltxv_audio_vae_decode,\n"
        "  ltxv_audio_vae_encode,\n"
        "  ltxv_empty_latent_audio,\n"
        ")\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'ace_step_15_func_name': encode_ace_step_15_audio.__name__,\n"
        "  'func_name': ltxv_audio_vae_encode.__name__,\n"
        "  'decode_func_name': ltxv_audio_vae_decode.__name__,\n"
        "  'empty_latent_func_name': ltxv_empty_latent_audio.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'comfy_loaded': 'comfy' in sys.modules,\n"
        "  'comfy_sd_loaded': 'comfy.sd' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["ace_step_15_func_name"] == "encode_ace_step_15_audio"
    assert payload["func_name"] == "ltxv_audio_vae_encode"
    assert payload["decode_func_name"] == "ltxv_audio_vae_decode"
    assert payload["empty_latent_func_name"] == "ltxv_empty_latent_audio"
    assert payload["torch_loaded"] is False
    assert payload["comfy_loaded"] is False
    assert payload["comfy_sd_loaded"] is False
    heavy = [module for module in payload["new_modules"] if module.startswith(("torch", "comfy"))]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
