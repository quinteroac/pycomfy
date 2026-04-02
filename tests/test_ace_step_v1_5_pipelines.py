"""CPU-passing tests for ACE Step 1.5 audio pipelines.

Covers all three pipeline variants in a single file:
  - checkpoint  (ace_step_1_5_t2a_checkpoint)
  - split       (ace_step_1_5_t2a_split)
  - split_4b    (ace_step_1_5_t2a_split_4b)

Acceptance criteria:
  AC01 — this file exists at tests/test_ace_step_v1_5_pipelines.py
  AC02 — manifest() returns a non-empty list of ModelEntry items
  AC03 — manifest() filenames and dest paths match expected values
  AC04 — run() with mocked model loading returns the correct dict shape
  AC05 — uv run pytest tests/test_ace_step_v1_5_pipelines.py passes on CPU
  AC06 — imports / syntax / typecheck pass
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch targets (shared across all three pipelines)
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_AURA_FLOW_PATCH = "comfy_diffusion.models.model_sampling_aura_flow"
_EMPTY_LATENT_PATCH = "comfy_diffusion.audio.empty_ace_step_15_latent_audio"
_ENCODE_AUDIO_PATCH = "comfy_diffusion.audio.encode_ace_step_15_audio"
_ZERO_OUT_PATCH = "comfy_diffusion.conditioning.conditioning_zero_out"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_AUDIO_PATCH = "comfy_diffusion.audio.vae_decode_audio"

# ---------------------------------------------------------------------------
# Pipeline module import paths (parametrize IDs)
# ---------------------------------------------------------------------------

_PIPELINE_MODULES = [
    "comfy_diffusion.pipelines.audio.ace_step.v1_5.checkpoint",
    "comfy_diffusion.pipelines.audio.ace_step.v1_5.split",
    "comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b",
]

_PIPELINE_IDS = ["checkpoint", "split", "split_4b"]

# ---------------------------------------------------------------------------
# AC01 / AC06 — File-level checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline_id", _PIPELINE_IDS)
def test_pipeline_file_exists(pipeline_id: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_file = (
        repo_root
        / "comfy_diffusion"
        / "pipelines"
        / "audio"
        / "ace_step"
        / "v1_5"
        / f"{pipeline_id}.py"
    )
    assert pipeline_file.is_file(), f"{pipeline_id}.py not found at {pipeline_file}"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_pipeline_importable(module_path: str) -> None:
    import importlib

    mod = importlib.import_module(module_path)
    assert callable(getattr(mod, "manifest", None)), f"{module_path}.manifest not callable"
    assert callable(getattr(mod, "run", None)), f"{module_path}.run not callable"


# ---------------------------------------------------------------------------
# AC02 — manifest() returns a non-empty list of ModelEntry items
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_manifest_returns_non_empty_list(module_path: str) -> None:
    import importlib

    from comfy_diffusion.downloader import ModelEntry

    mod = importlib.import_module(module_path)
    entries = mod.manifest()

    assert isinstance(entries, list), f"manifest() must return list, got {type(entries)}"
    assert len(entries) > 0, "manifest() must return at least one entry"
    for i, entry in enumerate(entries):
        assert isinstance(entry, ModelEntry), (
            f"Entry {i} must be a ModelEntry, got {type(entry)!r}"
        )


def test_checkpoint_manifest_has_one_entry() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.checkpoint import manifest

    assert len(manifest()) == 1


def test_split_manifest_has_four_entries() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split import manifest

    assert len(manifest()) == 4


def test_split_4b_manifest_has_four_entries() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    assert len(manifest()) == 4


# ---------------------------------------------------------------------------
# AC03 — manifest() filenames and dest paths are correct
# ---------------------------------------------------------------------------


def test_checkpoint_manifest_filename_and_dest() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.checkpoint import manifest

    entry = manifest()[0]
    assert isinstance(entry, HFModelEntry)
    assert entry.repo_id == "Comfy-Org/ace_step_1.5_ComfyUI_files"
    assert "ace_step_1.5_turbo_aio.safetensors" in entry.filename
    assert "checkpoints" in str(entry.dest)
    assert "ace_step_1.5_turbo_aio.safetensors" in str(entry.dest)


def test_split_manifest_filenames_and_dests() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split import manifest

    entries = manifest()
    filenames = [e.filename for e in entries]
    dests = [str(e.dest) for e in entries]

    # UNet
    assert any("acestep_v1.5_turbo" in f for f in filenames)
    assert any("diffusion_models" in d and "acestep_v1.5_turbo" in d for d in dests)

    # CLIP — 0.6B and 1.7B
    assert any("qwen_0.6b_ace15" in f for f in filenames)
    assert any("qwen_1.7b_ace15" in f for f in filenames)
    assert len([d for d in dests if "text_encoders" in d]) == 2

    # VAE
    assert any("ace_1.5_vae" in f for f in filenames)
    assert any("vae" in d and "ace_1.5_vae" in d for d in dests)


def test_split_4b_manifest_filenames_and_dests() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    entries = manifest()
    filenames = [e.filename for e in entries]
    dests = [str(e.dest) for e in entries]

    # UNet
    assert any("acestep_v1.5_turbo" in f for f in filenames)
    assert any("diffusion_models" in d and "acestep_v1.5_turbo" in d for d in dests)

    # CLIP — 0.6B and 4B
    assert any("qwen_0.6b_ace15" in f for f in filenames)
    assert any("qwen_4b_ace15" in f for f in filenames)
    assert len([d for d in dests if "text_encoders" in d]) == 2

    # VAE
    assert any("ace_1.5_vae" in f for f in filenames)
    assert any("vae" in d and "ace_1.5_vae" in d for d in dests)


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_manifest_all_entries_same_hf_repo(module_path: str) -> None:
    import importlib

    mod = importlib.import_module(module_path)
    for i, entry in enumerate(mod.manifest()):
        assert entry.repo_id == "Comfy-Org/ace_step_1.5_ComfyUI_files", (
            f"[{module_path}] entry {i}: expected Comfy-Org/ace_step_1.5_ComfyUI_files,"
            f" got {entry.repo_id!r}"
        )


# ---------------------------------------------------------------------------
# AC04 — run() with mocked models returns the correct dict shape
# ---------------------------------------------------------------------------


def _make_mock_mm_checkpoint() -> MagicMock:
    """ModelManager mock for the checkpoint pipeline (load_checkpoint)."""
    fake_vae = MagicMock(name="vae")
    fake_vae.audio_sample_rate = 44100
    mm = MagicMock()
    ckpt_result = MagicMock()
    ckpt_result.model = MagicMock(name="model")
    ckpt_result.clip = MagicMock(name="clip")
    ckpt_result.vae = fake_vae
    mm.load_checkpoint.return_value = ckpt_result
    return mm


def _make_mock_mm_split() -> MagicMock:
    """ModelManager mock for split / split_4b pipelines (load_unet, load_clip, load_vae)."""
    fake_vae = MagicMock(name="vae")
    fake_vae.audio_sample_rate = 44100
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = fake_vae
    return mm


def _run_pipeline_with_mocks(
    module_path: str,
    tmp_path: Path,
    *,
    tags: str = "neo-soul, warm groove",
    **run_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute a pipeline with all heavy dependencies mocked."""
    import importlib

    pipeline_mod = importlib.import_module(module_path)
    is_checkpoint = module_path.endswith("checkpoint")
    mm = _make_mock_mm_checkpoint() if is_checkpoint else _make_mock_mm_split()

    fake_waveform = MagicMock(name="waveform")
    captured: dict[str, Any] = {}

    def _fake_aura_flow(model: Any, shift: float) -> Any:
        captured["aura_flow_shift"] = shift
        return MagicMock(name="patched_model")

    def _fake_empty_latent(seconds: float, batch_size: int = 1) -> Any:
        captured["latent_seconds"] = seconds
        return MagicMock(name="latent")

    def _fake_encode_audio(clip: Any, tags_arg: str, **kwargs: Any) -> Any:
        captured["encode_tags"] = tags_arg
        return MagicMock(name="positive")

    def _fake_zero_out(cond: Any) -> Any:
        captured["zero_out_cond"] = cond
        return MagicMock(name="negative")

    def _fake_sample(*args: Any, **kwargs: Any) -> Any:
        captured["sample_args"] = args
        return MagicMock(name="latent_out")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_AURA_FLOW_PATCH, side_effect=_fake_aura_flow),
        patch(_EMPTY_LATENT_PATCH, side_effect=_fake_empty_latent),
        patch(_ENCODE_AUDIO_PATCH, side_effect=_fake_encode_audio),
        patch(_ZERO_OUT_PATCH, side_effect=_fake_zero_out),
        patch(_SAMPLE_PATCH, side_effect=_fake_sample),
        patch(_VAE_DECODE_AUDIO_PATCH, return_value=fake_waveform),
    ):
        result = pipeline_mod.run(models_dir=tmp_path, tags=tags, **run_kwargs)

    return result, captured


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_returns_dict_with_audio_key(module_path: str, tmp_path: Path) -> None:
    result, _ = _run_pipeline_with_mocks(module_path, tmp_path)
    assert isinstance(result, dict)
    assert "audio" in result, f"run() must return dict with 'audio' key, got {result.keys()}"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_audio_has_waveform_and_sample_rate(module_path: str, tmp_path: Path) -> None:
    result, _ = _run_pipeline_with_mocks(module_path, tmp_path)
    audio = result["audio"]
    assert "waveform" in audio, "audio dict must have 'waveform'"
    assert "sample_rate" in audio, "audio dict must have 'sample_rate'"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_audio_sample_rate_is_int(module_path: str, tmp_path: Path) -> None:
    result, _ = _run_pipeline_with_mocks(module_path, tmp_path)
    sample_rate = result["audio"]["sample_rate"]
    assert isinstance(sample_rate, int), f"sample_rate must be int, got {type(sample_rate)!r}"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_applies_aura_flow_shift_3(module_path: str, tmp_path: Path) -> None:
    _, captured = _run_pipeline_with_mocks(module_path, tmp_path)
    assert captured.get("aura_flow_shift") == 3, (
        f"ModelSamplingAuraFlow must use shift=3, got {captured.get('aura_flow_shift')!r}"
    )


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_creates_empty_latent(module_path: str, tmp_path: Path) -> None:
    _, captured = _run_pipeline_with_mocks(module_path, tmp_path)
    assert "latent_seconds" in captured, "empty_ace_step_15_latent_audio was not called"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_latent_seconds_matches_duration(module_path: str, tmp_path: Path) -> None:
    _, captured = _run_pipeline_with_mocks(module_path, tmp_path, duration=60.0)
    assert captured["latent_seconds"] == 60.0


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_uses_zero_out_for_negative(module_path: str, tmp_path: Path) -> None:
    _, captured = _run_pipeline_with_mocks(module_path, tmp_path)
    assert "zero_out_cond" in captured, "conditioning_zero_out was not called"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_sample_uses_euler_and_simple(module_path: str, tmp_path: Path) -> None:
    _, captured = _run_pipeline_with_mocks(module_path, tmp_path)
    args = captured.get("sample_args", ())
    assert len(args) >= 8
    assert args[6] == "euler", f"Expected sampler 'euler', got {args[6]!r}"
    assert args[7] == "simple", f"Expected scheduler 'simple', got {args[7]!r}"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_sample_uses_default_steps_and_cfg(module_path: str, tmp_path: Path) -> None:
    _, captured = _run_pipeline_with_mocks(module_path, tmp_path)
    args = captured.get("sample_args", ())
    assert args[4] == 8, f"Expected steps=8, got {args[4]!r}"
    assert args[5] == 1.0, f"Expected cfg=1.0, got {args[5]!r}"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_passes_tags_to_encode_audio(module_path: str, tmp_path: Path) -> None:
    _, captured = _run_pipeline_with_mocks(module_path, tmp_path, tags="lo-fi, piano")
    assert captured.get("encode_tags") == "lo-fi, piano"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_raises_on_runtime_error(module_path: str, tmp_path: Path) -> None:
    import importlib

    pipeline_mod = importlib.import_module(module_path)
    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, tags="test")


# ---------------------------------------------------------------------------
# run() signature checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_signature_has_required_params(module_path: str) -> None:
    import importlib

    mod = importlib.import_module(module_path)
    sig = inspect.signature(mod.run)
    params = set(sig.parameters.keys())
    required = {"models_dir", "tags", "lyrics", "duration", "bpm", "seed", "steps", "cfg"}
    missing = required - params
    assert not missing, f"[{module_path}] run() is missing parameters: {missing}"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_default_steps_is_eight(module_path: str) -> None:
    import importlib

    mod = importlib.import_module(module_path)
    sig = inspect.signature(mod.run)
    assert sig.parameters["steps"].default == 8


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_default_cfg_is_one(module_path: str) -> None:
    import importlib

    mod = importlib.import_module(module_path)
    sig = inspect.signature(mod.run)
    assert sig.parameters["cfg"].default == 1.0


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_default_sampler_is_euler(module_path: str) -> None:
    import importlib

    mod = importlib.import_module(module_path)
    sig = inspect.signature(mod.run)
    assert sig.parameters["sampler_name"].default == "euler"


@pytest.mark.parametrize("module_path", _PIPELINE_MODULES, ids=_PIPELINE_IDS)
def test_run_default_scheduler_is_simple(module_path: str) -> None:
    import importlib

    mod = importlib.import_module(module_path)
    sig = inspect.signature(mod.run)
    assert sig.parameters["scheduler"].default == "simple"
