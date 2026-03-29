"""Tests for LTX-Video 2.3 first-last-frame-to-video pipeline.

Covers:
  AC01: manifest() returns 2 HFModelEntry items matching ltx23_t2v
  AC02: run() parameter signature includes first_image, last_image, first_frame_strength, last_frame_strength
  AC03: CPU test passes with mocked inputs
  AC04: typecheck / lint — file parses without syntax errors; no top-level comfy imports
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Paths under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx23" / "flf2v.py"
)


# ---------------------------------------------------------------------------
# AC04 — file parses without syntax errors; conventions
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), (
        "comfy_diffusion/pipelines/video/ltx/ltx23/flf2v.py must exist"
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
    assert docstring, "flf2v.py must have a module-level docstring"


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
# AC01 — manifest() returns 2 HFModelEntry items matching ltx23 t2v
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_manifest_returns_exactly_two_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 2, f"manifest() must return exactly 2 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_matches_ltx23_t2v() -> None:
    """flf2v manifest must be identical to t2v manifest (same weights)."""
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import manifest as flf2v_manifest
    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest as t2v_manifest

    flf2v_entries = flf2v_manifest()
    t2v_entries = t2v_manifest()

    assert len(flf2v_entries) == len(t2v_entries)
    for flf2v_e, t2v_e in zip(flf2v_entries, t2v_entries):
        assert flf2v_e.repo_id == t2v_e.repo_id, (
            f"repo_id mismatch: {flf2v_e.repo_id!r} != {t2v_e.repo_id!r}"
        )
        assert str(flf2v_e.dest) == str(t2v_e.dest), (
            f"dest mismatch: {flf2v_e.dest!r} != {t2v_e.dest!r}"
        )
        assert flf2v_e.filename == t2v_e.filename, (
            f"filename mismatch: {flf2v_e.filename!r} != {t2v_e.filename!r}"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any(
        "diffusion_models" in d and "ltx-2.3-22b-distilled-fp8" in d for d in dests
    ), "manifest() must include an entry with dest under diffusion_models/ for ltx-2.3-22b-distilled-fp8"


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include a text_encoders/gemma_3_12B_it_fp4_mixed entry"
    )


def test_manifest_all_from_lightricks_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Lightricks/LTX-Video", (
            f"Entry {entry!r} must use repo_id 'Lightricks/LTX-Video'"
        )


# ---------------------------------------------------------------------------
# AC02 — run() signature includes flf2v-specific parameters
# ---------------------------------------------------------------------------


def test_run_has_first_image_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import run

    sig = inspect.signature(run)
    assert "first_image" in sig.parameters, "run() must have a 'first_image' parameter"


def test_run_has_last_image_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import run

    sig = inspect.signature(run)
    assert "last_image" in sig.parameters, "run() must have a 'last_image' parameter"


def test_run_has_first_frame_strength_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import run

    sig = inspect.signature(run)
    assert "first_frame_strength" in sig.parameters, (
        "run() must have a 'first_frame_strength' parameter"
    )
    assert sig.parameters["first_frame_strength"].default == 0.7, (
        "first_frame_strength default must be 0.7"
    )


def test_run_has_last_frame_strength_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import run

    sig = inspect.signature(run)
    assert "last_frame_strength" in sig.parameters, (
        "run() must have a 'last_frame_strength' parameter"
    )
    assert sig.parameters["last_frame_strength"].default == 0.7, (
        "last_frame_strength default must be 0.7"
    )


def test_run_has_fps_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import run

    sig = inspect.signature(run)
    assert "fps" in sig.parameters, "run() must have an 'fps' parameter"
    assert sig.parameters["fps"].default == 25, "fps default must be 25"


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import run

    sig = inspect.signature(run)
    assert "unet_filename" in sig.parameters
    assert "text_encoder_filename" in sig.parameters
    assert "vae_filename" in sig.parameters
    assert "upscaler_filename" not in sig.parameters, (
        "flf2v run() must not have an upscaler_filename parameter"
    )


def test_run_has_required_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import run

    sig = inspect.signature(run)
    assert "models_dir" in sig.parameters
    assert "prompt" in sig.parameters


# ---------------------------------------------------------------------------
# AC03 — CPU tests with mocked inputs
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_LTXV_COND_PATCH = "comfy_diffusion.conditioning.ltxv_conditioning"
_EMPTY_VIDEO_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_ADD_GUIDE_PATCH = "comfy_diffusion.conditioning.ltxv_add_guide"
_CROP_GUIDES_PATCH = "comfy_diffusion.conditioning.ltxv_crop_guides"
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
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"
_LOAD_IMAGE_PATCH = "comfy_diffusion.image.load_image"
_LTXV_PREPROCESS_PATCH = "comfy_diffusion.image.ltxv_preprocess"


def _build_mock_mm() -> MagicMock:
    """Return a ModelManager mock with sensible return values."""
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_ltxv_audio_vae.return_value = MagicMock(name="audio_vae")
    mm.load_ltxav_text_encoder.return_value = MagicMock(name="clip")
    return mm


def _make_add_guide_side_effect() -> tuple[list[tuple[Any, ...]], Any]:
    """Return a (calls_log, side_effect_fn) pair for ltxv_add_guide."""
    guide_calls: list[tuple[Any, ...]] = []

    def _add_guide(pos: Any, neg: Any, vae: Any, latent: Any, image: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        guide_calls.append((pos, neg, vae, latent, image, kwargs))
        return MagicMock(), MagicMock(), {"samples": MagicMock()}

    return guide_calls, _add_guide


def _run_mocked(
    tmp_path: Path,
    *,
    prompt: str = "test prompt",
    extra_run_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], MagicMock, MagicMock]:
    """Run the flf2v pipeline with all heavy deps mocked.

    Returns ``(result, mm_mock, add_guide_mock)``.
    """
    import comfy_diffusion.pipelines.video.ltx.ltx23.flf2v as pipeline_mod

    mm = _build_mock_mm()
    fake_tensor = MagicMock()
    fake_tensor.shape = (1, 64, 64, 3)

    fake_frames = [MagicMock()]
    fake_audio = {"waveform": MagicMock(), "sample_rate": 44100}

    first_image = MagicMock(spec=["mode"])
    last_image = MagicMock(spec=["mode"])

    add_guide_mock = MagicMock(
        side_effect=lambda pos, neg, vae, latent, img, **kw: (
            MagicMock(), MagicMock(), {"samples": MagicMock()}
        )
    )

    run_kwargs: dict[str, Any] = {
        "models_dir": tmp_path,
        "first_image": first_image,
        "last_image": last_image,
        "prompt": prompt,
        **(extra_run_kwargs or {}),
    }

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_ADD_GUIDE_PATCH, add_guide_mock),
        patch(_CROP_GUIDES_PATCH, return_value=(MagicMock(), MagicMock(), {"samples": MagicMock()})),
        patch(_EMPTY_AUDIO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CONCAT_AV_PATCH, return_value={"samples": MagicMock()}),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=fake_frames),
        patch(_AUDIO_VAE_DECODE_PATCH, return_value=fake_audio),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=fake_tensor),
        patch(_LOAD_IMAGE_PATCH, return_value=(fake_tensor, None)),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        result = pipeline_mod.run(**run_kwargs)

    return result, mm, add_guide_mock


def test_run_returns_dict_with_frames_and_audio(tmp_path: Path) -> None:
    """run() must return a dict with 'frames' and 'audio' keys."""
    result, _, _ = _run_mocked(tmp_path)
    assert isinstance(result, dict), "run() must return a dict"
    assert "frames" in result, "result must have a 'frames' key"
    assert "audio" in result, "result must have an 'audio' key"
    assert isinstance(result["frames"], list)
    assert isinstance(result["audio"], dict)


def test_run_calls_ltxv_add_guide_twice(tmp_path: Path) -> None:
    """ltxv_add_guide must be called exactly twice: first frame (idx=0), last frame (idx=-1)."""
    _, _, add_guide_mock = _run_mocked(tmp_path)
    assert add_guide_mock.call_count == 2, (
        f"ltxv_add_guide must be called exactly twice, got {add_guide_mock.call_count}"
    )
    first_call_kwargs = add_guide_mock.call_args_list[0].kwargs
    second_call_kwargs = add_guide_mock.call_args_list[1].kwargs
    assert first_call_kwargs.get("frame_idx") == 0, (
        f"First ltxv_add_guide call must use frame_idx=0, got {first_call_kwargs.get('frame_idx')!r}"
    )
    assert second_call_kwargs.get("frame_idx") == -1, (
        f"Second ltxv_add_guide call must use frame_idx=-1, got {second_call_kwargs.get('frame_idx')!r}"
    )


def test_run_calls_ltxv_crop_guides(tmp_path: Path) -> None:
    """ltxv_crop_guides must be called exactly once after both add_guide calls."""
    import comfy_diffusion.pipelines.video.ltx.ltx23.flf2v as pipeline_mod

    mm = _build_mock_mm()
    fake_tensor = MagicMock()
    fake_tensor.shape = (1, 64, 64, 3)
    first_image = MagicMock(spec=["mode"])
    last_image = MagicMock(spec=["mode"])
    crop_guides_mock = MagicMock(
        return_value=(MagicMock(), MagicMock(), {"samples": MagicMock()})
    )

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_ADD_GUIDE_PATCH, return_value=(MagicMock(), MagicMock(), {"samples": MagicMock()})),
        patch(_CROP_GUIDES_PATCH, crop_guides_mock),
        patch(_EMPTY_AUDIO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CONCAT_AV_PATCH, return_value={"samples": MagicMock()}),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=[MagicMock()]),
        patch(_AUDIO_VAE_DECODE_PATCH, return_value={"waveform": MagicMock(), "sample_rate": 44100}),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=fake_tensor),
        patch(_LOAD_IMAGE_PATCH, return_value=(fake_tensor, None)),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            first_image=first_image,
            last_image=last_image,
            prompt="test",
        )

    crop_guides_mock.assert_called_once()


def test_run_calls_sample_custom_and_separates_av_latent(tmp_path: Path) -> None:
    """AC03: verify call order — sample_custom → ltxv_separate_av_latent
    → vae_decode_batch_tiled → ltxv_audio_vae_decode."""
    import comfy_diffusion.pipelines.video.ltx.ltx23.flf2v as pipeline_mod

    call_order: list[str] = []
    mm = _build_mock_mm()
    fake_tensor = MagicMock()
    fake_tensor.shape = (1, 64, 64, 3)
    first_image = MagicMock(spec=["mode"])
    last_image = MagicMock(spec=["mode"])
    fake_frames = [MagicMock()]
    fake_audio = {"waveform": MagicMock(), "sample_rate": 44100}

    def fake_cfg_guider(*args: Any, **kwargs: Any) -> MagicMock:
        call_order.append("cfg_guider")
        return MagicMock()

    def fake_sample_custom(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        call_order.append("sample_custom")
        return MagicMock(), MagicMock()

    def fake_separate_av(latent: Any) -> tuple[Any, Any]:
        call_order.append("ltxv_separate_av_latent")
        return MagicMock(), MagicMock()

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
        patch(_ADD_GUIDE_PATCH, return_value=(MagicMock(), MagicMock(), {"samples": MagicMock()})),
        patch(_CROP_GUIDES_PATCH, return_value=(MagicMock(), MagicMock(), {"samples": MagicMock()})),
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
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=fake_tensor),
        patch(_LOAD_IMAGE_PATCH, return_value=(fake_tensor, None)),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            first_image=first_image,
            last_image=last_image,
            prompt="a sunlit jazz club",
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
    _, mm, _ = _run_mocked(tmp_path)
    mm.load_ltxv_audio_vae.assert_called_once()


def test_run_uses_load_ltxav_text_encoder(tmp_path: Path) -> None:
    """Gemma 3 text encoder must be loaded via load_ltxav_text_encoder, not load_clip."""
    _, mm, _ = _run_mocked(tmp_path)
    mm.load_ltxav_text_encoder.assert_called_once()
    mm.load_clip.assert_not_called()


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    """run() must raise RuntimeError when check_runtime() returns an error."""
    import comfy_diffusion.pipelines.video.ltx.ltx23.flf2v as pipeline_mod

    first_image = MagicMock(spec=["mode"])
    last_image = MagicMock(spec=["mode"])

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(
                models_dir=tmp_path,
                first_image=first_image,
                last_image=last_image,
                prompt="test",
            )


def test_run_uses_image_to_tensor_for_pil_images(tmp_path: Path) -> None:
    """image_to_tensor must be called for both first_image and last_image (PIL inputs)."""
    import comfy_diffusion.pipelines.video.ltx.ltx23.flf2v as pipeline_mod

    mm = _build_mock_mm()
    fake_tensor = MagicMock()
    fake_tensor.shape = (1, 64, 64, 3)
    first_image = MagicMock(spec=["mode"])
    last_image = MagicMock(spec=["mode"])
    image_to_tensor_mock = MagicMock(return_value=fake_tensor)

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_ADD_GUIDE_PATCH, return_value=(MagicMock(), MagicMock(), {"samples": MagicMock()})),
        patch(_CROP_GUIDES_PATCH, return_value=(MagicMock(), MagicMock(), {"samples": MagicMock()})),
        patch(_EMPTY_AUDIO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CONCAT_AV_PATCH, return_value={"samples": MagicMock()}),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=[MagicMock()]),
        patch(_AUDIO_VAE_DECODE_PATCH, return_value={"waveform": MagicMock(), "sample_rate": 44100}),
        patch(_IMAGE_TO_TENSOR_PATCH, image_to_tensor_mock),
        patch(_LOAD_IMAGE_PATCH, return_value=(fake_tensor, None)),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            first_image=first_image,
            last_image=last_image,
            prompt="test",
        )

    assert image_to_tensor_mock.call_count == 2, (
        f"image_to_tensor must be called twice (once per PIL image), "
        f"got {image_to_tensor_mock.call_count}"
    )


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    """download_models(manifest()) completes without error when all 2 files are present."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import manifest

    entries = manifest()
    assert len(entries) == 2

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
