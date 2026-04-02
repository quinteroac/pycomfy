"""Tests for comfy_diffusion/pipelines/audio/ace_step/v1_5/split_4b.py.

Covers:
  - File created at the correct path                                           (AC01)
  - manifest() returns four HFModelEntry items from the correct HF repo       (AC02)
  - run() loads models with load_unet, load_clip(ace, 4B), load_vae           (AC03)
  - Sampling execution follows the same order as US-002 (steps 2–7)           (AC04)
  - run() returns {"audio": {"waveform": tensor, "sample_rate": int}}          (AC05)
  - Import / parse without errors (typecheck / lint proxy)                     (AC06)
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_FILE = (
    _REPO_ROOT
    / "comfy_diffusion"
    / "pipelines"
    / "audio"
    / "ace_step"
    / "v1_5"
    / "split_4b.py"
)

# ---------------------------------------------------------------------------
# Patch targets
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
# File-level checks (AC01, AC06)
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), (
        "split_4b.py must exist under pipelines/audio/ace_step/v1_5/"
    )


def test_pipeline_parses_without_syntax_errors() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PIPELINE_FILE))
    assert isinstance(tree, ast.Module)


def test_pipeline_has_future_annotations() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert "from __future__ import annotations" in source


def test_pipeline_has_module_docstring() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PIPELINE_FILE))
    docstring = ast.get_docstring(tree)
    assert docstring, "split_4b.py must have a module-level docstring"


def test_pipeline_has_dunder_all_with_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert "__all__" in source
    assert '"manifest"' in source or "'manifest'" in source
    assert '"run"' in source or "'run'" in source


def test_no_top_level_comfy_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), (
                f"Top-level comfy import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks (AC02)
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_four_entries() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 4, f"manifest() must return exactly 4 entries, got {len(result)}"


def test_manifest_all_entries_are_hf_model_entry() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    for i, entry in enumerate(manifest()):
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {i} must be HFModelEntry, got {type(entry)!r}"
        )


def test_manifest_all_entries_same_hf_repo() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    for i, entry in enumerate(manifest()):
        assert entry.repo_id == "Comfy-Org/ace_step_1.5_ComfyUI_files", (
            f"Entry {i}: expected Comfy-Org/ace_step_1.5_ComfyUI_files, got {entry.repo_id!r}"
        )


def test_manifest_contains_unet_entry() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    filenames = [e.filename for e in manifest()]
    assert any("acestep_v1.5_turbo" in f for f in filenames), (
        f"manifest() must contain acestep_v1.5_turbo UNet entry; got {filenames}"
    )


def test_manifest_unet_dest_under_diffusion_models() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "acestep_v1.5_turbo" in d for d in dests), (
        f"manifest() must have a dest under diffusion_models/ for the UNet; got {dests}"
    )


def test_manifest_contains_0_6b_clip_entry() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    filenames = [e.filename for e in manifest()]
    assert any("qwen_0.6b_ace15" in f for f in filenames), (
        f"manifest() must contain qwen_0.6b_ace15 CLIP entry; got {filenames}"
    )


def test_manifest_contains_4b_clip_entry() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    filenames = [e.filename for e in manifest()]
    assert any("qwen_4b_ace15" in f for f in filenames), (
        f"manifest() must contain qwen_4b_ace15 CLIP entry; got {filenames}"
    )


def test_manifest_clip_dests_under_text_encoders() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    dests = [str(e.dest) for e in manifest()]
    clip_dests = [d for d in dests if "text_encoders" in d]
    assert len(clip_dests) == 2, (
        f"manifest() must have exactly 2 entries under text_encoders/; got {clip_dests}"
    )


def test_manifest_contains_vae_entry() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    filenames = [e.filename for e in manifest()]
    assert any("ace_1.5_vae" in f for f in filenames), (
        f"manifest() must contain ace_1.5_vae VAE entry; got {filenames}"
    )


def test_manifest_vae_dest_under_vae() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("vae" in d and "ace_1.5_vae" in d for d in dests), (
        f"manifest() must have a dest under vae/ for the VAE; got {dests}"
    )


# ---------------------------------------------------------------------------
# run() signature checks
# ---------------------------------------------------------------------------


def test_run_signature_includes_models_dir_and_tags() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import run

    sig = inspect.signature(run)
    required = {"models_dir", "tags"}
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_signature_has_audio_params() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import run

    sig = inspect.signature(run)
    expected = {"lyrics", "duration", "bpm", "seed", "steps", "cfg"}
    assert expected <= set(sig.parameters.keys()), (
        f"run() is missing audio parameters: {expected - set(sig.parameters.keys())}"
    )


def test_run_default_steps_is_eight() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 8


def test_run_default_cfg_is_one() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import run

    sig = inspect.signature(run)
    assert sig.parameters["cfg"].default == 1.0


def test_run_default_sampler_is_euler() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import run

    sig = inspect.signature(run)
    assert sig.parameters["sampler_name"].default == "euler"


def test_run_default_scheduler_is_simple() -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import run

    sig = inspect.signature(run)
    assert sig.parameters["scheduler"].default == "simple"


# ---------------------------------------------------------------------------
# Helper for run() behaviour tests
# ---------------------------------------------------------------------------


def _build_mock_mm(vae_sample_rate: int = 44100) -> MagicMock:
    fake_vae = MagicMock(name="vae")
    fake_vae.audio_sample_rate = vae_sample_rate
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = fake_vae
    return mm


def _run_with_mocks(
    tmp_path: Path,
    *,
    tags: str = "neo-soul, warm groove",
    **run_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], MagicMock]:
    """Run the pipeline with all heavy dependencies mocked."""
    from comfy_diffusion.pipelines.audio.ace_step.v1_5 import split_4b as pipeline_mod

    fake_waveform = MagicMock(name="waveform")
    mm = _build_mock_mm()
    captured: dict[str, Any] = {}

    def _fake_aura_flow(model: Any, shift: float) -> Any:
        captured["aura_flow_shift"] = shift
        captured["aura_flow_model"] = model
        return MagicMock(name="patched_model")

    def _fake_empty_latent(seconds: float, batch_size: int = 1) -> Any:
        captured["latent_seconds"] = seconds
        captured["latent_batch_size"] = batch_size
        return MagicMock(name="latent")

    def _fake_encode_audio(clip: Any, tags_arg: str, **kwargs: Any) -> Any:
        captured["encode_tags"] = tags_arg
        captured["encode_kwargs"] = kwargs
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
        result = pipeline_mod.run(
            models_dir=tmp_path,
            tags=tags,
            **run_kwargs,
        )

    return result, captured, mm


# ---------------------------------------------------------------------------
# run() model loading checks (AC03)
# ---------------------------------------------------------------------------


def test_run_calls_load_unet(tmp_path: Path) -> None:
    _, _, mm = _run_with_mocks(tmp_path)
    mm.load_unet.assert_called_once()
    assert "acestep_v1.5_turbo" in str(mm.load_unet.call_args)


def test_run_calls_load_clip_with_ace_type(tmp_path: Path) -> None:
    _, _, mm = _run_with_mocks(tmp_path)
    mm.load_clip.assert_called_once()
    call_kwargs = mm.load_clip.call_args
    assert call_kwargs.kwargs.get("clip_type") == "ace", (
        f"load_clip must be called with clip_type='ace', got {call_kwargs!r}"
    )


def test_run_calls_load_clip_with_0_6b_and_4b_files(tmp_path: Path) -> None:
    _, _, mm = _run_with_mocks(tmp_path)
    call_args = mm.load_clip.call_args
    positional_args = call_args.args
    assert len(positional_args) == 2, (
        f"load_clip must receive 2 positional clip paths, got {len(positional_args)}"
    )
    assert "qwen_0.6b_ace15" in positional_args[0]
    assert "qwen_4b_ace15" in positional_args[1]


def test_run_calls_load_vae(tmp_path: Path) -> None:
    _, _, mm = _run_with_mocks(tmp_path)
    mm.load_vae.assert_called_once()
    assert "ace_1.5_vae" in str(mm.load_vae.call_args)


# ---------------------------------------------------------------------------
# run() return value (AC05)
# ---------------------------------------------------------------------------


def test_run_returns_dict_with_audio_key(tmp_path: Path) -> None:
    result, _, _ = _run_with_mocks(tmp_path)
    assert isinstance(result, dict)
    assert "audio" in result, f"run() must return dict with 'audio' key, got {result.keys()}"


def test_run_audio_has_waveform_and_sample_rate(tmp_path: Path) -> None:
    result, _, _ = _run_with_mocks(tmp_path)
    audio = result["audio"]
    assert "waveform" in audio, "audio dict must have 'waveform'"
    assert "sample_rate" in audio, "audio dict must have 'sample_rate'"


def test_run_audio_sample_rate_is_int(tmp_path: Path) -> None:
    result, _, _ = _run_with_mocks(tmp_path)
    sample_rate = result["audio"]["sample_rate"]
    assert isinstance(sample_rate, int), (
        f"sample_rate must be int, got {type(sample_rate)!r}"
    )


# ---------------------------------------------------------------------------
# Node execution order — same as US-002 steps 2–7 (AC04)
# ---------------------------------------------------------------------------


def test_run_applies_model_sampling_aura_flow_shift_3(tmp_path: Path) -> None:
    _, captured, _ = _run_with_mocks(tmp_path)
    assert captured.get("aura_flow_shift") == 3, (
        f"ModelSamplingAuraFlow must use shift=3, got {captured.get('aura_flow_shift')!r}"
    )


def test_run_creates_empty_latent_audio(tmp_path: Path) -> None:
    _, captured, _ = _run_with_mocks(tmp_path)
    assert "latent_seconds" in captured, "empty_ace_step_15_latent_audio was not called"


def test_run_latent_seconds_matches_duration(tmp_path: Path) -> None:
    _, captured, _ = _run_with_mocks(tmp_path, duration=60.0)
    assert captured["latent_seconds"] == 60.0


def test_run_uses_conditioning_zero_out_for_negative(tmp_path: Path) -> None:
    _, captured, _ = _run_with_mocks(tmp_path)
    assert "zero_out_cond" in captured, "conditioning_zero_out was not called"


def test_run_calls_sample_with_euler_sampler(tmp_path: Path) -> None:
    _, captured, _ = _run_with_mocks(tmp_path)
    args = captured.get("sample_args", ())
    # sample(model, positive, negative, latent, steps, cfg, sampler_name, scheduler, seed)
    assert len(args) >= 7
    assert args[6] == "euler", f"Expected sampler 'euler', got {args[6]!r}"


def test_run_calls_sample_with_simple_scheduler(tmp_path: Path) -> None:
    _, captured, _ = _run_with_mocks(tmp_path)
    args = captured.get("sample_args", ())
    assert args[7] == "simple", f"Expected scheduler 'simple', got {args[7]!r}"


def test_run_calls_sample_with_cfg_one(tmp_path: Path) -> None:
    _, captured, _ = _run_with_mocks(tmp_path)
    args = captured.get("sample_args", ())
    assert args[5] == 1.0, f"Expected cfg=1.0, got {args[5]!r}"


def test_run_calls_sample_with_steps_eight(tmp_path: Path) -> None:
    _, captured, _ = _run_with_mocks(tmp_path)
    args = captured.get("sample_args", ())
    assert args[4] == 8, f"Expected steps=8, got {args[4]!r}"


def test_run_passes_tags_to_encode_audio(tmp_path: Path) -> None:
    _, captured, _ = _run_with_mocks(tmp_path, tags="lo-fi, piano")
    assert captured.get("encode_tags") == "lo-fi, piano"


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.audio.ace_step.v1_5 import split_4b as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, tags="test")


# ---------------------------------------------------------------------------
# download_models idempotent with manifest entries (AC02)
# ---------------------------------------------------------------------------


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.audio.ace_step.v1_5.split_4b import manifest

    entries = manifest()
    assert len(entries) == 4

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
