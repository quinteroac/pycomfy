"""CPU-safe smoke tests for all 7 functions added in iteration 000023.

Covers:
  1. ltxv_empty_latent_video   (latent.py)
  2. ltxv_concat_av_latent      (audio.py)
  3. ltxv_separate_av_latent    (audio.py)
  4. ltxv_crop_guides           (conditioning.py)
  5. ltxv_latent_upsample       (latent.py)
  6. ModelManager.load_latent_upscale_model  (models.py)
  7. manual_sigmas              (sampling.py)

AC03: no `torch`, `comfy.*`, or `comfy_diffusion.*` import at module top level.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. ltxv_empty_latent_video
# ---------------------------------------------------------------------------

def test_ltxv_empty_latent_video_signature() -> None:
    from comfy_diffusion.latent import ltxv_empty_latent_video

    sig = inspect.signature(ltxv_empty_latent_video)
    assert str(sig) == "(width: 'int', height: 'int', length: 'int' = 97, batch_size: 'int' = 1, fps: 'int' = 24) -> 'dict[str, Any]'"


def test_ltxv_empty_latent_video_returns_correct_shape(monkeypatch: Any) -> None:
    """Use subprocess to test ltxv_empty_latent_video with a fake torch, avoiding
    sys.modules["torch"] patching that can corrupt the torch C extension in-process."""
    result = _run_python(
        "import json, sys, types\n"
        "class _FakeTensor:\n"
        "    def __init__(self, shape, **kw): self.shape = tuple(shape)\n"
        "fake_torch = types.SimpleNamespace(zeros=lambda s, device=None: _FakeTensor(s))\n"
        "fake_mm = types.SimpleNamespace(intermediate_device=lambda: 'cpu')\n"
        "fake_comfy = types.SimpleNamespace(model_management=fake_mm)\n"
        "sys.modules['torch'] = fake_torch\n"
        "sys.modules['comfy'] = fake_comfy\n"
        "sys.modules['comfy.model_management'] = fake_mm\n"
        "from comfy_diffusion.latent import ltxv_empty_latent_video\n"
        "out = ltxv_empty_latent_video(width=768, height=512, length=97, batch_size=1)\n"
        "print(json.dumps({'shape': list(out['samples'].shape), 'has_samples': 'samples' in out}))\n"
    )
    import json as _json
    payload = _json.loads(result.stdout)
    # width=768, height=512, length=97 → latent_length = (97-1)//8+1=13; 512//32=16; 768//32=24
    assert payload["has_samples"] is True
    assert payload["shape"] == [1, 128, 13, 16, 24]


def test_ltxv_empty_latent_video_in_module_all() -> None:
    import comfy_diffusion.latent as latent_module

    assert "ltxv_empty_latent_video" in latent_module.__all__


# ---------------------------------------------------------------------------
# 2. ltxv_concat_av_latent
# ---------------------------------------------------------------------------

def test_ltxv_concat_av_latent_signature() -> None:
    from comfy_diffusion.audio import ltxv_concat_av_latent

    sig = inspect.signature(ltxv_concat_av_latent)
    assert str(sig) == (
        "(video_latent: 'dict[str, Any]', audio_latent: 'dict[str, Any]') -> 'dict[str, Any]'"
    )


def test_ltxv_concat_av_latent_merges_keys(monkeypatch: Any) -> None:
    import comfy_diffusion.audio as audio_module
    from comfy_diffusion.audio import ltxv_concat_av_latent

    class _FakeTensor:
        pass

    class _FakeNestedTensor:
        class NestedTensor:
            def __init__(self, tensors: tuple[Any, Any]) -> None:
                self.tensors = tensors

    class _FakeTorch:
        float32 = "float32"

        def ones_like(self, t: Any) -> _FakeTensor:
            return _FakeTensor()

    monkeypatch.setattr(
        audio_module,
        "_get_concat_av_latent_dependencies",
        lambda: (_FakeTorch(), _FakeNestedTensor),
    )

    v_samples = _FakeTensor()
    a_samples = _FakeTensor()
    video_latent: dict[str, Any] = {"samples": v_samples, "extra": "v"}
    audio_latent: dict[str, Any] = {"samples": a_samples, "extra": "a"}

    result = ltxv_concat_av_latent(video_latent, audio_latent)

    assert "samples" in result
    assert isinstance(result["samples"], _FakeNestedTensor.NestedTensor)
    assert result["samples"].tensors == (v_samples, a_samples)
    # No noise_mask expected when neither input had one
    assert "noise_mask" not in result


def test_ltxv_concat_av_latent_in_module_all() -> None:
    import comfy_diffusion.audio as audio_module

    assert "ltxv_concat_av_latent" in audio_module.__all__


# ---------------------------------------------------------------------------
# 3. ltxv_separate_av_latent
# ---------------------------------------------------------------------------

def test_ltxv_separate_av_latent_signature() -> None:
    from comfy_diffusion.audio import ltxv_separate_av_latent

    sig = inspect.signature(ltxv_separate_av_latent)
    assert str(sig) == (
        "(av_latent: 'dict[str, Any]') -> 'tuple[dict[str, Any], dict[str, Any]]'"
    )


def test_ltxv_separate_av_latent_unbinds_samples() -> None:
    from comfy_diffusion.audio import ltxv_separate_av_latent

    class _FakeSamples:
        def __init__(self, a: object, b: object) -> None:
            self._a = a
            self._b = b

        def unbind(self) -> tuple[object, object]:
            return self._a, self._b

    v_samples = object()
    a_samples = object()
    av_latent: dict[str, Any] = {"samples": _FakeSamples(v_samples, a_samples)}

    video_latent, audio_latent = ltxv_separate_av_latent(av_latent)

    assert video_latent == {"samples": v_samples}
    assert audio_latent == {"samples": a_samples}


def test_ltxv_separate_av_latent_in_module_all() -> None:
    import comfy_diffusion.audio as audio_module

    assert "ltxv_separate_av_latent" in audio_module.__all__


# ---------------------------------------------------------------------------
# 4. ltxv_crop_guides
# ---------------------------------------------------------------------------

def test_ltxv_crop_guides_signature() -> None:
    from comfy_diffusion.conditioning import ltxv_crop_guides

    sig = inspect.signature(ltxv_crop_guides)
    assert str(sig) == (
        "(positive: 'Any', negative: 'Any', latent: 'dict[str, Any]') "
        "-> 'tuple[Any, Any, dict[str, Any]]'"
    )


def test_ltxv_crop_guides_passthrough_when_no_keyframes() -> None:
    from comfy_diffusion.conditioning import ltxv_crop_guides

    positive = [["tok", {"frame_rate": 25.0}]]
    negative = [["tok", {"frame_rate": 25.0}]]
    latent: dict[str, Any] = {"samples": object()}

    out_pos, out_neg, out_latent = ltxv_crop_guides(positive, negative, latent)

    assert out_pos is positive
    assert out_neg is negative
    assert out_latent is latent


def test_ltxv_crop_guides_in_module_all() -> None:
    import comfy_diffusion.conditioning as conditioning_module

    assert "ltxv_crop_guides" in conditioning_module.__all__


# ---------------------------------------------------------------------------
# 5. ltxv_latent_upsample
# ---------------------------------------------------------------------------

def test_ltxv_latent_upsample_signature() -> None:
    from comfy_diffusion.latent import ltxv_latent_upsample

    sig = inspect.signature(ltxv_latent_upsample)
    assert str(sig) == (
        "(samples: 'dict[str, Any]', upscale_model: 'Any', vae: 'Any') -> 'dict[str, Any]'"
    )


def test_ltxv_latent_upsample_delegates_to_node(monkeypatch: Any) -> None:
    import comfy_diffusion.latent as latent_module
    from comfy_diffusion.latent import ltxv_latent_upsample

    expected: dict[str, Any] = {"samples": object()}
    upscale_model = object()
    vae = object()
    input_samples: dict[str, Any] = {"samples": object()}

    class _FakeLTXVLatentUpsampler:
        def upsample_latent(
            self,
            samples: dict[str, Any],
            upscale_model: Any,
            vae: Any,
        ) -> tuple[dict[str, Any]]:
            return (expected,)

    monkeypatch.setattr(
        latent_module,
        "_load_latent_upscale_model",
        lambda: _FakeLTXVLatentUpsampler,
    )

    result = ltxv_latent_upsample(samples=input_samples, upscale_model=upscale_model, vae=vae)
    assert result is expected


def test_ltxv_latent_upsample_in_module_all() -> None:
    import comfy_diffusion.latent as latent_module

    assert "ltxv_latent_upsample" in latent_module.__all__


# ---------------------------------------------------------------------------
# 6. ModelManager.load_latent_upscale_model
# ---------------------------------------------------------------------------

def test_load_latent_upscale_model_method_exists() -> None:
    from comfy_diffusion.models import ModelManager

    assert callable(getattr(ModelManager, "load_latent_upscale_model", None))


def test_load_latent_upscale_model_signature() -> None:
    from comfy_diffusion.models import ModelManager

    sig = inspect.signature(ModelManager.load_latent_upscale_model)
    assert "path" in sig.parameters
    assert "->" in str(sig)


def test_load_latent_upscale_model_raises_on_missing_absolute_path(
    monkeypatch: Any, tmp_path: Any
) -> None:
    import sys
    import types

    import comfy_diffusion.models as models_module
    from comfy_diffusion.models import ModelManager

    # Stub out all comfy / folder_paths so no real GPU/model loading occurs
    fake_folder_paths = types.ModuleType("folder_paths")
    setattr(fake_folder_paths, "add_model_folder_path", lambda *a, **kw: None)
    setattr(fake_folder_paths, "get_full_path_or_raise", lambda *a, **kw: "")

    fake_comfy_utils = types.ModuleType("comfy.utils")
    setattr(fake_comfy_utils, "load_torch_file", lambda *a, **kw: ({}, None))

    fake_comfy = types.ModuleType("comfy")
    setattr(fake_comfy, "utils", fake_comfy_utils)

    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_comfy_utils)
    monkeypatch.setattr(models_module, "ensure_comfyui_on_path", lambda: None)

    missing = tmp_path / "nonexistent.safetensors"
    import pytest as _pytest

    with _pytest.raises(FileNotFoundError):
        ModelManager(models_dir=tmp_path).load_latent_upscale_model(missing)


def test_load_latent_upscale_model_accessible_via_model_manager() -> None:
    from comfy_diffusion.models import ModelManager

    assert callable(getattr(ModelManager, "load_latent_upscale_model", None))
    assert "ModelManager" in __import__("comfy_diffusion.models", fromlist=["ModelManager"]).__all__


# ---------------------------------------------------------------------------
# 7. manual_sigmas
# ---------------------------------------------------------------------------

def test_manual_sigmas_signature() -> None:
    from comfy_diffusion.sampling import manual_sigmas

    sig = inspect.signature(manual_sigmas)
    assert str(sig) == "(sigmas: 'str') -> 'Any'"


def test_manual_sigmas_parses_string() -> None:
    from comfy_diffusion.sampling import manual_sigmas

    result = manual_sigmas("14 7 3 0")
    # Result is a torch.FloatTensor — verify via its .tolist() and dtype
    assert result.tolist() == [14.0, 7.0, 3.0, 0.0]
    assert str(result.dtype) == "torch.float32"


def test_manual_sigmas_parses_comma_separated_decimals() -> None:
    from comfy_diffusion.sampling import manual_sigmas

    result = manual_sigmas("3.5, 1.75, 0.875")
    values = result.tolist()
    assert abs(values[0] - 3.5) < 1e-6
    assert abs(values[1] - 1.75) < 1e-6


def test_manual_sigmas_in_module_all() -> None:
    import comfy_diffusion.sampling as sampling_module

    assert "manual_sigmas" in sampling_module.__all__


# ---------------------------------------------------------------------------
# AC02/AC03: all 7 functions importable without side-loading torch or comfy
# ---------------------------------------------------------------------------

def test_all_seven_functions_importable_without_heavy_modules() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline = set(sys.modules)\n"
        "from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample\n"
        "from comfy_diffusion.audio import ltxv_concat_av_latent, ltxv_separate_av_latent\n"
        "from comfy_diffusion.conditioning import ltxv_crop_guides\n"
        "from comfy_diffusion.models import ModelManager\n"
        "from comfy_diffusion.sampling import manual_sigmas\n"
        "post = set(sys.modules)\n"
        "new = sorted(post - baseline)\n"
        "heavy = [m for m in new if m.startswith(('torch', 'comfy.')) or m == 'comfy']\n"
        "payload = {\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'comfy_loaded': 'comfy' in sys.modules,\n"
        "  'heavy': heavy,\n"
        "  'functions': [\n"
        "    ltxv_empty_latent_video.__name__,\n"
        "    ltxv_latent_upsample.__name__,\n"
        "    ltxv_concat_av_latent.__name__,\n"
        "    ltxv_separate_av_latent.__name__,\n"
        "    ltxv_crop_guides.__name__,\n"
        "    manual_sigmas.__name__,\n"
        "  ],\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["torch_loaded"] is False
    assert payload["comfy_loaded"] is False
    assert payload["heavy"] == [], f"Unexpected heavy modules: {payload['heavy']}"
    assert payload["functions"] == [
        "ltxv_empty_latent_video",
        "ltxv_latent_upsample",
        "ltxv_concat_av_latent",
        "ltxv_separate_av_latent",
        "ltxv_crop_guides",
        "manual_sigmas",
    ]
