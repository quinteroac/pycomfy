"""Tests for prompt conditioning helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import comfy_diffusion.conditioning as conditioning
from comfy_diffusion.conditioning import (
    conditioning_combine,
    conditioning_set_mask,
    conditioning_set_timestep_range,
    encode_prompt,
    encode_prompt_flux,
    flux_guidance,
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


def test_encode_prompt_imports_without_models_on_cpu() -> None:
    result = _run_python(
        "from comfy_diffusion.conditioning import encode_prompt; "
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


def test_encode_prompt_accepts_weighted_prompt_syntax_without_changes() -> None:
    weighted_prompt = "a portrait of a woman, (studio lighting:1.3)"
    sentinel_conditioning = object()
    expected_tokens = {"tokenized": weighted_prompt}

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
    result = encode_prompt(clip, weighted_prompt)

    assert result is not None
    assert result is sentinel_conditioning
    assert clip.tokenize_calls == [weighted_prompt]
    assert clip.encode_calls == [expected_tokens]


def test_encode_prompt_does_not_raise_for_valid_weighted_prompt_syntax() -> None:
    weighted_prompt = "a portrait of a woman, (studio lighting:1.3)"

    class FakeClip:
        def tokenize(self, text: str) -> object:
            assert text == weighted_prompt
            return ("tokens", text)

        def encode_from_tokens_scheduled(self, tokens: object) -> object:
            return {"conditioning": tokens}

    result = encode_prompt(FakeClip(), weighted_prompt)

    assert result is not None
    assert result == {"conditioning": ("tokens", weighted_prompt)}


def test_encode_prompt_accepts_empty_string_without_crashing() -> None:
    sentinel_conditioning = object()

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[str] = []

        def tokenize(self, text: str) -> object:
            self.tokenize_calls.append(text)
            if text == "":
                raise AssertionError("empty prompt should be normalized before tokenization")
            return ("tokens", text)

        def encode_from_tokens_scheduled(self, tokens: object) -> object:
            assert tokens == ("tokens", " ")
            return sentinel_conditioning

    clip = FakeClip()
    result = encode_prompt(clip, "")

    assert result is not None
    assert result is sentinel_conditioning
    assert clip.tokenize_calls == [" "]


def test_encode_prompt_flux_imports_without_models_on_cpu() -> None:
    result = _run_python(
        "from comfy_diffusion.conditioning import encode_prompt_flux; "
        "assert encode_prompt_flux.__name__ == 'encode_prompt_flux'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_encode_prompt_flux_encodes_clip_l_and_t5xxl_with_guidance() -> None:
    expected_output = object()

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[str] = []
            self.encode_calls: list[tuple[object, object]] = []

        def tokenize(self, text: str) -> object:
            self.tokenize_calls.append(text)
            if text == "clip-l detail":
                return {"clip_l": "clip-l detail"}
            if text == "t5 detail":
                return {"t5xxl": "t5 detail"}
            raise AssertionError(f"unexpected text: {text!r}")

        def encode_from_tokens_scheduled(
            self,
            tokens: object,
            add_dict: object = None,
        ) -> object:
            self.encode_calls.append((tokens, add_dict))
            return expected_output

    clip = FakeClip()

    result = encode_prompt_flux(
        clip=clip,
        text="t5 detail",
        clip_l_text="clip-l detail",
        guidance=7.0,
    )

    assert result is expected_output
    assert clip.tokenize_calls == ["clip-l detail", "t5 detail"]
    assert clip.encode_calls == [
        ({"clip_l": "clip-l detail", "t5xxl": "t5 detail"}, {"guidance": 7.0})
    ]


def test_encode_prompt_flux_uses_default_guidance_and_normalizes_empty_text() -> None:
    expected_output = object()

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[str] = []
            self.encode_calls: list[tuple[object, object]] = []

        def tokenize(self, text: str) -> object:
            self.tokenize_calls.append(text)
            if text == " ":
                return {"t5xxl": " "}
            if text == "clip-l":
                return {"clip_l": "clip-l"}
            raise AssertionError(f"unexpected text: {text!r}")

        def encode_from_tokens_scheduled(
            self,
            tokens: object,
            add_dict: object = None,
        ) -> object:
            self.encode_calls.append((tokens, add_dict))
            return expected_output

    clip = FakeClip()

    result = encode_prompt_flux(clip=clip, text="", clip_l_text="clip-l")

    assert result is expected_output
    assert clip.tokenize_calls == ["clip-l", " "]
    assert clip.encode_calls == [
        ({"clip_l": "clip-l", "t5xxl": " "}, {"guidance": 3.5})
    ]


def test_conditioning_public_api_exports_expected_entrypoints() -> None:
    signature = inspect.signature(encode_prompt)
    flux_signature = inspect.signature(encode_prompt_flux)
    flux_guidance_signature = inspect.signature(flux_guidance)

    assert "is_negative" not in signature.parameters
    assert not hasattr(conditioning, "encode_negative_prompt")
    assert flux_signature.parameters["guidance"].default == 3.5
    assert flux_guidance_signature.parameters["guidance"].default == 3.5
    assert conditioning_combine.__name__ == "conditioning_combine"
    assert conditioning_set_mask.__name__ == "conditioning_set_mask"
    assert conditioning_set_timestep_range.__name__ == "conditioning_set_timestep_range"
    assert flux_guidance.__name__ == "flux_guidance"
    assert conditioning.__all__ == [
        "encode_prompt",
        "encode_prompt_flux",
        "conditioning_combine",
        "conditioning_set_mask",
        "conditioning_set_timestep_range",
        "flux_guidance",
    ]


def test_conditioning_combine_merges_two_conditioning_lists() -> None:
    cond_a = [["a-token-1", {"meta": "a1"}], ["a-token-2", {"meta": "a2"}]]
    cond_b = [["b-token-1", {"meta": "b1"}]]

    merged = conditioning_combine(cond_a, cond_b)

    assert merged == [
        ["a-token-1", {"meta": "a1"}],
        ["a-token-2", {"meta": "a2"}],
        ["b-token-1", {"meta": "b1"}],
    ]


def test_conditioning_combine_supports_more_than_two_by_chaining() -> None:
    cond_a = [["a-token", {"w": 1.0}]]
    cond_b = [["b-token", {"w": 1.0}]]
    cond_c = [["c-token", {"w": 1.0}]]

    merged = conditioning_combine(conditioning_combine(cond_a, cond_b), cond_c)

    assert merged == [
        ["a-token", {"w": 1.0}],
        ["b-token", {"w": 1.0}],
        ["c-token", {"w": 1.0}],
    ]


def test_conditioning_combine_supports_list_input() -> None:
    cond_a = [["a-token", {"source": "base"}]]
    cond_b = [["b-token", {"source": "style"}]]
    cond_c = [["c-token", {"source": "detail"}]]

    merged = conditioning_combine([cond_a, cond_b, cond_c])

    assert merged == [
        ["a-token", {"source": "base"}],
        ["b-token", {"source": "style"}],
        ["c-token", {"source": "detail"}],
    ]


def test_conditioning_set_mask_applies_default_strength_and_default_cond_area() -> None:
    conditioning_input = [
        ["a-token", {"source": "base"}],
        ["b-token", {"source": "style"}],
    ]

    class FakeMask:
        def __init__(self) -> None:
            self.shape = (64, 64)
            self.unsqueeze_calls: list[int] = []
            self.batch_mask = object()

        def unsqueeze(self, dim: int) -> object:
            self.unsqueeze_calls.append(dim)
            assert dim == 0
            return self.batch_mask

    mask = FakeMask()

    output = conditioning_set_mask(conditioning_input, mask)

    assert output is not conditioning_input
    assert mask.unsqueeze_calls == [0]
    assert output[0][1]["source"] == "base"
    assert output[1][1]["source"] == "style"
    assert output[0][1]["mask"] is mask.batch_mask
    assert output[1][1]["mask"] is mask.batch_mask
    assert output[0][1]["mask_strength"] == 1.0
    assert output[1][1]["mask_strength"] == 1.0
    assert output[0][1]["set_area_to_bounds"] is False
    assert output[1][1]["set_area_to_bounds"] is False
    assert "mask" not in conditioning_input[0][1]
    assert "mask_strength" not in conditioning_input[0][1]


def test_conditioning_set_mask_supports_mask_bounds_and_custom_strength() -> None:
    conditioning_input = [["a-token", {"source": "base"}]]

    class FakeMask:
        def __init__(self) -> None:
            self.shape = (1, 64, 64)
            self.unsqueeze_calls: list[int] = []

        def unsqueeze(self, dim: int) -> Any:
            self.unsqueeze_calls.append(dim)
            raise AssertionError("mask already has batch dimension")

    mask = FakeMask()

    output = conditioning_set_mask(
        conditioning_input,
        mask=mask,
        strength=0.25,
        set_cond_area="mask bounds",
    )

    assert mask.unsqueeze_calls == []
    assert output[0][1]["mask"] is mask
    assert output[0][1]["mask_strength"] == 0.25
    assert output[0][1]["set_area_to_bounds"] is True


def test_conditioning_set_timestep_range_sets_start_and_end_percent_metadata() -> None:
    conditioning_input = [
        ["a-token", {"source": "base"}],
        ["b-token", {"source": "style"}],
    ]

    output = conditioning_set_timestep_range(conditioning_input, start=0.2, end=0.8)

    assert output is not conditioning_input
    assert output == [
        ["a-token", {"source": "base", "start_percent": 0.2, "end_percent": 0.8}],
        ["b-token", {"source": "style", "start_percent": 0.2, "end_percent": 0.8}],
    ]
    assert output[0][1] is not conditioning_input[0][1]
    assert output[1][1] is not conditioning_input[1][1]
    assert "start_percent" not in conditioning_input[0][1]
    assert "end_percent" not in conditioning_input[0][1]


def test_conditioning_set_timestep_range_rejects_non_float_values() -> None:
    conditioning_input = [["a-token", {"source": "base"}]]

    with pytest.raises(TypeError, match="start must be a float"):
        conditioning_set_timestep_range(conditioning_input, start=0, end=1.0)

    with pytest.raises(TypeError, match="end must be a float"):
        conditioning_set_timestep_range(conditioning_input, start=0.0, end=1)


def test_conditioning_set_timestep_range_rejects_values_outside_percentage_bounds() -> None:
    conditioning_input = [["a-token", {"source": "base"}]]

    with pytest.raises(ValueError, match="start must be between 0.0 and 1.0"):
        conditioning_set_timestep_range(conditioning_input, start=-0.01, end=1.0)

    with pytest.raises(ValueError, match="end must be between 0.0 and 1.0"):
        conditioning_set_timestep_range(conditioning_input, start=0.0, end=1.01)


def test_flux_guidance_applies_guidance_to_each_conditioning_item() -> None:
    conditioning_input = [
        ["a-token", {"source": "base"}],
        ["b-token", {"source": "style"}],
    ]

    output = flux_guidance(conditioning_input, guidance=5.25)

    assert output is not conditioning_input
    assert output == [
        ["a-token", {"source": "base", "guidance": 5.25}],
        ["b-token", {"source": "style", "guidance": 5.25}],
    ]
    assert output[0][1] is not conditioning_input[0][1]
    assert output[1][1] is not conditioning_input[1][1]
    assert "guidance" not in conditioning_input[0][1]
    assert "guidance" not in conditioning_input[1][1]


def test_flux_guidance_uses_default_guidance_value() -> None:
    conditioning_input = [["a-token", {"source": "base"}]]

    output = flux_guidance(conditioning_input)

    assert output == [["a-token", {"source": "base", "guidance": 3.5}]]


def test_import_comfy_diffusion_conditioning_has_no_torch_or_loader_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.conditioning import encode_prompt\n"
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
