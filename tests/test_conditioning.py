"""Tests for US-001 prompt conditioning helpers."""

from __future__ import annotations

import json
import os
import inspect
import subprocess
import sys
from pathlib import Path

import pycomfy.conditioning as conditioning
from pycomfy.conditioning import encode_prompt


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


def test_encode_prompt_imports_without_models_on_cpu() -> None:
    result = _run_python(
        "from pycomfy.conditioning import encode_prompt; "
        "assert encode_prompt.__name__ == 'encode_prompt'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_encode_prompt_returns_raw_clip_conditioning_output() -> None:
    sentinel_conditioning = object()
    expected_tokens = {"tokenized": "a portrait of a woman, studio lighting"}

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[str] = []
            self.encode_calls: list[object] = []

        def tokenize(self, text: str) -> object:
            self.tokenize_calls.append(text)
            return expected_tokens

        def encode_from_tokens_scheduled(self, tokens: object) -> object:
            self.encode_calls.append(tokens)
            return sentinel_conditioning

    clip = FakeClip()

    result = encode_prompt(clip, "a portrait of a woman, studio lighting")

    assert result is not None
    assert result is sentinel_conditioning
    assert clip.tokenize_calls == ["a portrait of a woman, studio lighting"]
    assert clip.encode_calls == [expected_tokens]


def test_encode_prompt_supports_negative_prompt_text() -> None:
    sentinel_conditioning = object()
    expected_tokens = {"tokenized": "ugly, blurry"}

    class FakeClip:
        def tokenize(self, text: str) -> object:
            assert text == "ugly, blurry"
            return expected_tokens

        def encode_from_tokens_scheduled(self, tokens: object) -> object:
            assert tokens is expected_tokens
            return sentinel_conditioning

    result = encode_prompt(FakeClip(), "ugly, blurry")

    assert result is not None
    assert result is sentinel_conditioning


def test_conditioning_public_api_has_single_encode_prompt_entrypoint() -> None:
    signature = inspect.signature(encode_prompt)

    assert "is_negative" not in signature.parameters
    assert not hasattr(conditioning, "encode_negative_prompt")
    assert conditioning.__all__ == ["encode_prompt"]


def test_import_pycomfy_conditioning_has_no_torch_or_loader_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import pycomfy\n"
        "baseline_modules = set(sys.modules)\n"
        "from pycomfy.conditioning import encode_prompt\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'func_name': encode_prompt.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'folder_paths_loaded': 'folder_paths' in sys.modules,\n"
        "  'comfy_sd_loaded': 'comfy.sd' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["func_name"] == "encode_prompt"
    assert payload["torch_loaded"] is False
    assert payload["folder_paths_loaded"] is False
    assert payload["comfy_sd_loaded"] is False
    heavy = [
        module
        for module in payload["new_modules"]
        if module.startswith(("torch", "folder_paths", "comfy.sd"))
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
