"""Tests for text generation helpers."""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import comfy_diffusion.textgen as textgen_module
from comfy_diffusion.textgen import generate_ltx2_prompt, generate_text


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


def test_textgen_module_exports_generate_text() -> None:
    assert generate_text.__name__ == "generate_text"
    assert generate_ltx2_prompt.__name__ == "generate_ltx2_prompt"
    assert textgen_module.__all__ == ["generate_text", "generate_ltx2_prompt"]


def test_generate_text_signature_matches_contract() -> None:
    signature = inspect.signature(generate_text)

    assert str(signature) == (
        "(clip: 'Any', prompt: 'str', *, image: 'Any | None' = None, "
        "max_length: 'int' = 256, do_sample: 'bool' = True, "
        "temperature: 'float' = 0.7, top_k: 'int' = 64, top_p: 'float' = 0.95, "
        "min_p: 'float' = 0.05, repetition_penalty: 'float' = 1.05, "
        "seed: 'int' = 0) -> 'str'"
    )


def test_generate_ltx2_prompt_signature_matches_generate_text() -> None:
    assert inspect.signature(generate_ltx2_prompt) == inspect.signature(generate_text)
    assert inspect.signature(generate_ltx2_prompt).return_annotation == "str"


def test_generate_text_calls_tokenize_generate_decode_with_sampling() -> None:
    prompt = "describe this image"
    image = object()
    tokens = object()
    generated_ids = object()
    decoded_text = "hello world"

    class FakeClip:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

        def tokenize(self, *args: Any, **kwargs: Any) -> Any:
            self.calls.append(("tokenize", args, kwargs))
            return tokens

        def generate(self, *args: Any, **kwargs: Any) -> Any:
            self.calls.append(("generate", args, kwargs))
            return generated_ids

        def decode(self, *args: Any, **kwargs: Any) -> str:
            self.calls.append(("decode", args, kwargs))
            return decoded_text

    clip = FakeClip()

    result = generate_text(
        clip=clip,
        prompt=prompt,
        image=image,
        max_length=128,
        do_sample=True,
        temperature=0.9,
        top_k=12,
        top_p=0.8,
        min_p=0.1,
        repetition_penalty=1.2,
        seed=42,
    )

    assert result == decoded_text
    assert clip.calls == [
        (
            "tokenize",
            (prompt,),
            {"image": image, "skip_template": False, "min_length": 1},
        ),
        (
            "generate",
            (tokens,),
            {
                "do_sample": True,
                "max_length": 128,
                "temperature": 0.9,
                "top_k": 12,
                "top_p": 0.8,
                "min_p": 0.1,
                "repetition_penalty": 1.2,
                "seed": 42,
            },
        ),
        (
            "decode",
            (generated_ids,),
            {"skip_special_tokens": True},
        ),
    ]


def test_generate_text_returns_decoded_text_without_trimming() -> None:
    decoded_text = "text with trailing spaces   "

    class FakeClip:
        def tokenize(self, *_: Any, **__: Any) -> Any:
            return "tokens"

        def generate(self, *_: Any, **__: Any) -> Any:
            return "ids"

        def decode(self, *_: Any, **__: Any) -> str:
            return decoded_text

    assert generate_text(clip=FakeClip(), prompt="p") == decoded_text


def test_generate_text_does_not_forward_sampling_params_when_disabled() -> None:
    class FakeClip:
        def __init__(self) -> None:
            self.generate_kwargs: dict[str, Any] | None = None

        def tokenize(self, *_: Any, **__: Any) -> Any:
            return "tokens"

        def generate(self, *_: Any, **kwargs: Any) -> Any:
            self.generate_kwargs = kwargs
            return "ids"

        def decode(self, *_: Any, **__: Any) -> str:
            return "decoded"

    clip = FakeClip()
    result = generate_text(
        clip=clip,
        prompt="p",
        do_sample=False,
        temperature=1.5,
        top_k=999,
        top_p=0.1,
        min_p=0.2,
        repetition_penalty=3.2,
        seed=123456,
    )

    assert result == "decoded"
    assert clip.generate_kwargs == {"do_sample": False, "max_length": 256}


def test_generate_ltx2_prompt_uses_t2v_template_when_image_is_none(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_generate_text(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "enhanced"

    monkeypatch.setattr(textgen_module, "generate_text", fake_generate_text)

    result = generate_ltx2_prompt(
        clip="clip",
        prompt="raw prompt",
        image=None,
        max_length=120,
        do_sample=False,
        temperature=0.5,
        top_k=10,
        top_p=0.8,
        min_p=0.2,
        repetition_penalty=1.1,
        seed=99,
    )

    assert result == "enhanced"
    assert len(calls) == 1
    forwarded = calls[0]
    expected_prompt = (
        f"<start_of_turn>system\n{textgen_module.LTX2_T2V_SYSTEM_PROMPT.strip()}<end_of_turn>\n"
        "<start_of_turn>user\nUser Raw Input Prompt: raw prompt.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    assert forwarded["prompt"] == expected_prompt
    assert forwarded["image"] is None
    assert forwarded["clip"] == "clip"
    assert forwarded["max_length"] == 120
    assert forwarded["do_sample"] is False
    assert forwarded["temperature"] == 0.5
    assert forwarded["top_k"] == 10
    assert forwarded["top_p"] == 0.8
    assert forwarded["min_p"] == 0.2
    assert forwarded["repetition_penalty"] == 1.1
    assert forwarded["seed"] == 99


def test_generate_ltx2_prompt_uses_i2v_template_when_image_is_set(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    image = object()

    def fake_generate_text(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "enhanced"

    monkeypatch.setattr(textgen_module, "generate_text", fake_generate_text)

    result = generate_ltx2_prompt(
        clip="clip",
        prompt="raw prompt",
        image=image,
    )

    assert result == "enhanced"
    assert len(calls) == 1
    forwarded = calls[0]
    expected_prompt = (
        f"<start_of_turn>system\n{textgen_module.LTX2_I2V_SYSTEM_PROMPT.strip()}<end_of_turn>\n"
        "<start_of_turn>user\n\n<image_soft_token>\n\n"
        "User Raw Input Prompt: raw prompt.<end_of_turn>\n<start_of_turn>model\n"
    )
    assert forwarded["prompt"] == expected_prompt
    assert forwarded["image"] is image


def test_ltx2_templates_match_comfyui_verbatim() -> None:
    source_path = _repo_root() / "vendor/ComfyUI/comfy_extras/nodes_textgen.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id in {"LTX2_T2V_SYSTEM_PROMPT", "LTX2_I2V_SYSTEM_PROMPT"}
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                constants[target.id] = node.value.value

    assert constants["LTX2_T2V_SYSTEM_PROMPT"] == textgen_module.LTX2_T2V_SYSTEM_PROMPT
    assert constants["LTX2_I2V_SYSTEM_PROMPT"] == textgen_module.LTX2_I2V_SYSTEM_PROMPT


def test_textgen_import_has_no_torch_or_comfy_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion import textgen\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "heavy_modules = [\n"
        "    m for m in new_modules\n"
        "    if m == 'comfy' or m.startswith('comfy.') or m.startswith('torch')\n"
        "]\n"
        "payload = {\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'comfy_loaded': any(m == 'comfy' or m.startswith('comfy.') for m in sys.modules),\n"
        "  'heavy': heavy_modules,\n"
        "  'functions': sorted(\n"
        "      [textgen.generate_text.__name__, textgen.generate_ltx2_prompt.__name__]\n"
        "  ),\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["functions"] == ["generate_ltx2_prompt", "generate_text"]
    assert payload["torch_loaded"] is False
    assert payload["comfy_loaded"] is False
    assert payload["heavy"] == []
