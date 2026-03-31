"""Tests for comfy_diffusion/pipelines/video/wan/wan21/t2v.py — WAN 2.1 T2V pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring
  - Exports manifest and run; no top-level comfy imports
  - __all__ = ["manifest", "run"]
  - manifest() returns exactly 3 HFModelEntry items with correct dest paths
  - All manifest entries use the Comfy-Org/Wan_2.1_ComfyUI_repackaged HF repo
  - run() signature: models_dir, prompt, negative_prompt, width, height, length, fps, steps, cfg, seed, filename overrides
  - run() behaviour: model loading, encode_prompt, empty_wan_latent_video, model_sampling_sd3, sample, vae_decode_batch
  - run() raises RuntimeError when check_runtime returns an error
  - download_models idempotent with 3 entries
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
    / "video"
    / "wan"
    / "wan21"
    / "t2v.py"
)

# ---------------------------------------------------------------------------
# Patch constants (source module paths for lazy imports)
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_MODEL_SAMPLING_SD3_PATCH = "comfy_diffusion.models.model_sampling_sd3"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_EMPTY_WAN_LATENT_PATCH = "comfy_diffusion.latent.empty_wan_latent_video"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_BATCH_PATCH = "comfy_diffusion.vae.vae_decode_batch"


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "t2v.py must exist"


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
    assert docstring, "t2v.py must have a module-level docstring"


def test_pipeline_all_is_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '__all__ = ["manifest", "run"]' in source


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
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks — exactly 3 HFModelEntry items
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_three_entries() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 3, f"manifest() must return exactly 3 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "wan2.1_t2v_1.3B" in d for d in dests), (
        "manifest() must include a diffusion_models/wan2.1_t2v_1.3B entry"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "umt5_xxl" in d for d in dests), (
        "manifest() must include a text_encoders/umt5_xxl entry"
    )


def test_manifest_vae_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("vae" in d and "wan_2.1_vae" in d for d in dests), (
        "manifest() must include a vae/wan_2.1_vae entry"
    )


def test_manifest_all_from_comfy_org_wan_repo() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Comfy-Org/Wan_2.1_ComfyUI_repackaged", (
            f"Entry {entry!r} must use repo_id 'Comfy-Org/Wan_2.1_ComfyUI_repackaged'"
        )


# ---------------------------------------------------------------------------
# run() signature checks
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import run

    sig = inspect.signature(run)
    required = {"models_dir", "prompt", "negative_prompt", "width", "height", "length"}
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_default_width_832() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 832


def test_run_default_height_480() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import run

    sig = inspect.signature(run)
    assert sig.parameters["height"].default == 480


def test_run_default_length_33() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import run

    sig = inspect.signature(run)
    assert sig.parameters["length"].default == 33


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import run

    sig = inspect.signature(run)
    for param in ("unet_filename", "text_encoder_filename", "vae_filename"):
        assert param in sig.parameters, f"run() must accept '{param}'"


def test_run_has_steps_cfg_seed_params() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import run

    sig = inspect.signature(run)
    assert "steps" in sig.parameters
    assert "cfg" in sig.parameters
    assert "seed" in sig.parameters


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    return mm


def _run_with_mocks(
    tmp_path: Path,
    *,
    call_order: list[str] | None = None,
    fake_frames: list[Any] | None = None,
) -> list[Any]:
    from comfy_diffusion.pipelines.video.wan.wan21 import t2v as pipeline_mod

    if fake_frames is None:
        fake_frames = [MagicMock(name="frame0")]

    fake_latent = MagicMock(name="latent")
    fake_sampled = MagicMock(name="sampled")
    fake_patched_model = MagicMock(name="patched_model")

    mm = _build_mock_mm()

    def _track(name: str, rv: Any) -> Any:
        if call_order is not None:
            call_order.append(name)
        return rv

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_WAN_LATENT_PATCH, side_effect=lambda **kw: (
            _track("empty_wan_latent_video", fake_latent)
        )),
        patch(_MODEL_SAMPLING_SD3_PATCH, side_effect=lambda m, shift: (
            _track("model_sampling_sd3", fake_patched_model)
        )),
        patch(_SAMPLE_PATCH, side_effect=lambda m, pos, neg, lat, steps, cfg, sn, sc, seed: (
            _track("sample", fake_sampled)
        )),
        patch(_VAE_DECODE_BATCH_PATCH, side_effect=lambda v, s: (
            _track("vae_decode_batch", fake_frames)
        )),
    ):
        return pipeline_mod.run(
            models_dir=tmp_path,
            prompt="a fox in a winter scenery",
        )


# ---------------------------------------------------------------------------
# run() behaviour tests
# ---------------------------------------------------------------------------


def test_run_returns_list_of_frames(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path)
    assert isinstance(result, list)


def test_run_frames_value(tmp_path: Path) -> None:
    fake_frames = [MagicMock(name="f0"), MagicMock(name="f1")]
    result = _run_with_mocks(tmp_path, fake_frames=fake_frames)
    assert result is fake_frames


def test_run_calls_empty_wan_latent_video(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "empty_wan_latent_video" in call_order


def test_run_calls_model_sampling_sd3(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "model_sampling_sd3" in call_order


def test_run_calls_sample(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "sample" in call_order


def test_run_calls_vae_decode_batch(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "vae_decode_batch" in call_order


def test_run_call_order(tmp_path: Path) -> None:
    """Pipeline must execute nodes in the correct order."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    expected = [
        "empty_wan_latent_video",
        "model_sampling_sd3",
        "sample",
        "vae_decode_batch",
    ]
    assert call_order == expected, (
        f"Expected call order {expected}, got {call_order}"
    )


def test_run_uses_load_clip_with_wan_type(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.wan.wan21 import t2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_WAN_LATENT_PATCH, return_value=MagicMock()),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_clip.assert_called_once()
    _, kwargs = mm.load_clip.call_args
    assert kwargs.get("clip_type") == "wan", (
        "load_clip must be called with clip_type='wan'"
    )


def test_run_uses_load_unet_and_load_vae(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.wan.wan21 import t2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_WAN_LATENT_PATCH, return_value=MagicMock()),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_unet.assert_called_once()
    mm.load_vae.assert_called_once()


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.wan.wan21 import t2v as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


def test_run_model_sampling_sd3_shift_is_8(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.wan.wan21 import t2v as pipeline_mod

    mm = _build_mock_mm()
    captured: dict[str, Any] = {}

    def capture_sd3(model: Any, shift: float) -> Any:
        captured["shift"] = shift
        return MagicMock(name="patched_model")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_WAN_LATENT_PATCH, return_value=MagicMock()),
        patch(_MODEL_SAMPLING_SD3_PATCH, side_effect=capture_sd3),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert captured.get("shift") == 8.0, (
        f"model_sampling_sd3 must be called with shift=8.0, got {captured.get('shift')}"
    )


def test_run_sample_uses_uni_pc_simple(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.wan.wan21 import t2v as pipeline_mod

    mm = _build_mock_mm()
    captured: dict[str, Any] = {}

    def capture_sample(
        model: Any, pos: Any, neg: Any, lat: Any,
        steps: Any, cfg: Any, sampler_name: str, scheduler: str, seed: int,
    ) -> Any:
        captured["sampler_name"] = sampler_name
        captured["scheduler"] = scheduler
        return MagicMock(name="sampled")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_WAN_LATENT_PATCH, return_value=MagicMock()),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, side_effect=capture_sample),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert captured.get("sampler_name") == "uni_pc", (
        f"sample() must use sampler_name='uni_pc', got {captured.get('sampler_name')}"
    )
    assert captured.get("scheduler") == "simple", (
        f"sample() must use scheduler='simple', got {captured.get('scheduler')}"
    )


# ---------------------------------------------------------------------------
# download_models idempotent — 3 entries
# ---------------------------------------------------------------------------


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.wan.wan21.t2v import manifest

    entries = manifest()
    assert len(entries) == 3

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
