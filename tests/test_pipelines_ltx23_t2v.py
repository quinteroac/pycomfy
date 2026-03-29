"""Tests for US-005 — comfy_diffusion/pipelines/ltx23_t2v.py pipeline.

Covers:
  AC01: manifest() returns 2 HFModelEntry items with correct dest paths
  AC03: CPU test passes with mocked inputs (AV sampling chain)
  AC04: typecheck / lint — file parses without syntax errors; no top-level comfy imports
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
_PIPELINE_FILE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx23" / "t2v.py"


# ---------------------------------------------------------------------------
# AC04 — file parses without syntax errors; conventions
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "comfy_diffusion/pipelines/video/ltx/ltx23/t2v.py must exist"


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


def test_pipeline_exports_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
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


# ---------------------------------------------------------------------------
# AC01 — manifest() returns 2 HFModelEntry items
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_manifest_returns_exactly_two_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 2, f"manifest() must return exactly 2 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any(
        "diffusion_models" in d and "ltx-2.3-22b-distilled-fp8" in d for d in dests
    ), "manifest() must include an entry with dest under diffusion_models/ for ltx-2.3-22b-distilled-fp8"


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include a text_encoders/gemma_3_12B_it_fp4_mixed entry"
    )


def test_manifest_all_from_lightricks_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Lightricks/LTX-Video", (
            f"Entry {entry!r} must use repo_id 'Lightricks/LTX-Video'"
        )


def test_manifest_entries_have_nonempty_dest() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest

    for entry in manifest():
        assert entry.dest, f"manifest() entry {entry!r} must have a non-empty dest"


# ---------------------------------------------------------------------------
# AC03 — run() signature
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import run

    sig = inspect.signature(run)
    params = set(sig.parameters.keys())
    required = {"models_dir", "prompt", "negative_prompt", "width", "height"}
    assert required <= params, f"run() is missing parameters: {required - params}"


def test_run_has_fps_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import run

    sig = inspect.signature(run)
    assert "fps" in sig.parameters, "run() must have an 'fps' parameter"


def test_run_fps_default_is_25() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import run

    sig = inspect.signature(run)
    assert sig.parameters["fps"].default == 25, "fps default must be 25"


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import run

    sig = inspect.signature(run)
    assert "unet_filename" in sig.parameters
    assert "text_encoder_filename" in sig.parameters
    assert "vae_filename" in sig.parameters


# ---------------------------------------------------------------------------
# AC03 — CPU test passes with mocked inputs (AV sampling chain)
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_LTXV_COND_PATCH = "comfy_diffusion.conditioning.ltxv_conditioning"
_EMPTY_VIDEO_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_EMPTY_AUDIO_PATCH = "comfy_diffusion.audio.ltxv_empty_latent_audio"
_CONCAT_AV_PATCH = "comfy_diffusion.audio.ltxv_concat_av_latent"
_CFG_GUIDER_PATCH = "comfy_diffusion.sampling.cfg_guider"
_RANDOM_NOISE_PATCH = "comfy_diffusion.sampling.random_noise"
_MANUAL_SIGMAS_PATCH = "comfy_diffusion.sampling.manual_sigmas"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_SAMPLE_CUSTOM_PATCH = "comfy_diffusion.sampling.sample_custom"
_SEPARATE_AV_PATCH = "comfy_diffusion.audio.ltxv_separate_av_latent"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode_batch_tiled"
_AUDIO_VAE_DECODE_PATCH = "comfy_diffusion.audio.ltxv_audio_vae_decode"


def _build_mock_mm() -> MagicMock:
    """Return a ModelManager mock with sensible return values."""
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_ltxv_audio_vae.return_value = MagicMock(name="audio_vae")
    mm.load_ltxav_text_encoder.return_value = MagicMock(name="clip")
    return mm


def _run_mocked(
    tmp_path: Path,
    *,
    prompt: str = "test prompt",
    extra_patches: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], MagicMock]:
    """Helper: run pipeline with all heavy dependencies mocked."""
    from comfy_diffusion.pipelines.video.ltx.ltx23 import t2v as pipeline_mod

    mm = _build_mock_mm()
    fake_frames = [MagicMock()]
    fake_audio = {"waveform": MagicMock(), "sample_rate": 44100}

    patches: dict[str, Any] = {
        _RUNTIME_PATCH: {"python_version": "3.12.0"},
        _MM_PATCH: mm,
        _ENCODE_PATCH: (MagicMock(), MagicMock()),
        _LTXV_COND_PATCH: (MagicMock(), MagicMock()),
        _EMPTY_VIDEO_PATCH: {"samples": MagicMock()},
        _EMPTY_AUDIO_PATCH: {"samples": MagicMock()},
        _CONCAT_AV_PATCH: {"samples": MagicMock()},
        _CFG_GUIDER_PATCH: MagicMock(),
        _RANDOM_NOISE_PATCH: MagicMock(),
        _MANUAL_SIGMAS_PATCH: MagicMock(),
        _GET_SAMPLER_PATCH: MagicMock(),
        _SAMPLE_CUSTOM_PATCH: (MagicMock(), MagicMock()),
        _SEPARATE_AV_PATCH: (MagicMock(), MagicMock()),
        _VAE_DECODE_PATCH: fake_frames,
        _AUDIO_VAE_DECODE_PATCH: fake_audio,
    }
    if extra_patches:
        patches.update(extra_patches)

    with (
        patch(_RUNTIME_PATCH, return_value=patches[_RUNTIME_PATCH]),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=patches[_ENCODE_PATCH]),
        patch(_LTXV_COND_PATCH, return_value=patches[_LTXV_COND_PATCH]),
        patch(_EMPTY_VIDEO_PATCH, return_value=patches[_EMPTY_VIDEO_PATCH]),
        patch(_EMPTY_AUDIO_PATCH, return_value=patches[_EMPTY_AUDIO_PATCH]),
        patch(_CONCAT_AV_PATCH, return_value=patches[_CONCAT_AV_PATCH]),
        patch(_CFG_GUIDER_PATCH, return_value=patches[_CFG_GUIDER_PATCH]),
        patch(_RANDOM_NOISE_PATCH, return_value=patches[_RANDOM_NOISE_PATCH]),
        patch(_MANUAL_SIGMAS_PATCH, return_value=patches[_MANUAL_SIGMAS_PATCH]),
        patch(_GET_SAMPLER_PATCH, return_value=patches[_GET_SAMPLER_PATCH]),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=patches[_SAMPLE_CUSTOM_PATCH]),
        patch(_SEPARATE_AV_PATCH, return_value=patches[_SEPARATE_AV_PATCH]),
        patch(_VAE_DECODE_PATCH, return_value=patches[_VAE_DECODE_PATCH]),
        patch(_AUDIO_VAE_DECODE_PATCH, return_value=patches[_AUDIO_VAE_DECODE_PATCH]),
    ):
        result = pipeline_mod.run(models_dir=tmp_path, prompt=prompt)

    return result, mm


def test_run_returns_dict_with_frames_and_audio(tmp_path: Path) -> None:
    """run() must return a dict with 'frames' and 'audio' keys."""
    result, _ = _run_mocked(tmp_path)
    assert isinstance(result, dict), "run() must return a dict"
    assert "frames" in result, "result must have a 'frames' key"
    assert "audio" in result, "result must have an 'audio' key"
    assert isinstance(result["frames"], list)
    assert isinstance(result["audio"], dict)


def test_run_calls_sample_custom_and_separates_av_latent(tmp_path: Path) -> None:
    """AC03: verify call order: cfg_guider → sample_custom → ltxv_separate_av_latent
    → vae_decode_batch_tiled → ltxv_audio_vae_decode."""
    from comfy_diffusion.pipelines.video.ltx.ltx23 import t2v as pipeline_mod

    call_order: list[str] = []
    mm = _build_mock_mm()

    def _tracker(name: str) -> Any:
        def fn(*args: Any, **kwargs: Any) -> Any:
            call_order.append(name)
            return MagicMock()
        return fn

    fake_denoised = MagicMock()
    fake_video_latent = MagicMock()
    fake_audio_latent = MagicMock()
    fake_frames = [MagicMock()]
    fake_audio = {"waveform": MagicMock(), "sample_rate": 44100}

    def fake_cfg_guider(*args: Any, **kwargs: Any) -> MagicMock:
        call_order.append("cfg_guider")
        return MagicMock()

    def fake_sample_custom(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        call_order.append("sample_custom")
        return MagicMock(), fake_denoised

    def fake_separate_av(latent: Any) -> tuple[Any, Any]:
        call_order.append("ltxv_separate_av_latent")
        return fake_video_latent, fake_audio_latent

    def fake_vae_decode(vae: Any, latent: Any) -> list[Any]:
        call_order.append("vae_decode_batch_tiled")
        return fake_frames

    def fake_audio_vae_decode(audio_vae: Any, latent: Any) -> dict[str, Any]:
        call_order.append("ltxv_audio_vae_decode")
        return fake_audio

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_EMPTY_AUDIO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CONCAT_AV_PATCH, return_value={"samples": MagicMock()}),
        patch(_CFG_GUIDER_PATCH, fake_cfg_guider),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, fake_sample_custom),
        patch(_SEPARATE_AV_PATCH, fake_separate_av),
        patch(_VAE_DECODE_PATCH, fake_vae_decode),
        patch(_AUDIO_VAE_DECODE_PATCH, fake_audio_vae_decode),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            prompt="a golden retriever running through a sunlit park",
        )

    assert call_order == [
        "cfg_guider",
        "sample_custom",
        "ltxv_separate_av_latent",
        "vae_decode_batch_tiled",
        "ltxv_audio_vae_decode",
    ], f"Unexpected call order: {call_order}"
    assert result["frames"] is fake_frames
    assert result["audio"] is fake_audio


def test_run_uses_load_ltxv_audio_vae(tmp_path: Path) -> None:
    """Audio VAE must be loaded via mm.load_ltxv_audio_vae."""
    _, mm = _run_mocked(tmp_path)
    mm.load_ltxv_audio_vae.assert_called_once()


def test_run_uses_load_ltxav_text_encoder(tmp_path: Path) -> None:
    """Gemma 3 text encoder must be loaded via load_ltxav_text_encoder."""
    _, mm = _run_mocked(tmp_path)
    mm.load_ltxav_text_encoder.assert_called_once()
    mm.load_clip.assert_not_called()


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    """run() must raise RuntimeError when check_runtime() returns an error."""
    from comfy_diffusion.pipelines.video.ltx.ltx23 import t2v as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    """download_models(manifest()) completes without error when all 2 files are present."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest

    entries = manifest()
    assert len(entries) == 2

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
