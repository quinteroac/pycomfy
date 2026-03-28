"""Tests for US-006 — comfy_diffusion/pipelines/ltx3_i2v.py pipeline.

Covers:
  AC01: manifest() returns same 2 entries as ltx3_t2v
  AC02: run() adds image, fps, and guide_strength parameters vs ltx3_t2v.run()
  AC04: CPU test passes with mocked inputs (AV sampling chain + ltxv_add_guide)
  AC05: typecheck / lint — file parses without syntax errors; no top-level comfy imports
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
_PIPELINE_FILE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx3" / "i2v.py"
_T2V_FILE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx3" / "t2v.py"


# ---------------------------------------------------------------------------
# AC05 — file parses without syntax errors; conventions
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "comfy_diffusion/pipelines/video/ltx/ltx3/i2v.py must exist"


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
    assert docstring, "i2v.py must have a module-level docstring"


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
# AC01 — manifest() returns same 2 entries as ltx3_t2v
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_manifest_returns_exactly_two_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 2, f"manifest() must return exactly 2 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_matches_ltx3_t2v() -> None:
    """AC01: manifest() returns the same 2 entries as ltx3_t2v.manifest()."""
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest as i2v_manifest
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import manifest as t2v_manifest

    i2v_entries = i2v_manifest()
    t2v_entries = t2v_manifest()

    assert len(i2v_entries) == len(t2v_entries), (
        f"ltx3_i2v manifest has {len(i2v_entries)} entries; "
        f"ltx3_t2v manifest has {len(t2v_entries)} entries — must match"
    )
    for i2v_e, t2v_e in zip(i2v_entries, t2v_entries):
        assert i2v_e.repo_id == t2v_e.repo_id
        assert str(i2v_e.dest) == str(t2v_e.dest)
        assert i2v_e.filename == t2v_e.filename


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any(
        "diffusion_models" in d and "ltx-2.3-22b-distilled-fp8" in d for d in dests
    ), "manifest() must include diffusion_models/ltx-2.3-22b-distilled-fp8"


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include text_encoders/gemma_3_12B_it_fp4_mixed"
    )


def test_manifest_all_from_lightricks_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Lightricks/LTX-Video"


# ---------------------------------------------------------------------------
# AC02 — run() adds image, fps, guide_strength parameters vs ltx3_t2v.run()
# ---------------------------------------------------------------------------


def test_run_has_image_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    assert "image" in sig.parameters, "run() must have an 'image' parameter"


def test_run_has_fps_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    assert "fps" in sig.parameters, "run() must have an 'fps' parameter"


def test_run_fps_default_is_25() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    assert sig.parameters["fps"].default == 25, "fps default must be 25"


def test_run_fps_annotation_is_int() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    # Under from __future__ import annotations, annotation is a string
    ann = sig.parameters["fps"].annotation
    assert ann in (int, "int"), f"fps must be annotated as int, got {ann!r}"


def test_run_has_guide_strength_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    assert "guide_strength" in sig.parameters, "run() must have a 'guide_strength' parameter"
    assert sig.parameters["guide_strength"].default == 1.0, "guide_strength default must be 1.0"


def test_run_has_all_t2v_params_plus_image_and_fps() -> None:
    """run() must include all ltx3_t2v params plus image, fps, and guide_strength."""
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run as i2v_run
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import run as t2v_run

    i2v_params = set(inspect.signature(i2v_run).parameters)
    t2v_params = set(inspect.signature(t2v_run).parameters)

    # i2v must contain everything from t2v
    assert t2v_params <= i2v_params, (
        f"ltx3_i2v.run() is missing t2v parameters: {t2v_params - i2v_params}"
    )
    # Plus image, fps, guide_strength
    assert "image" in i2v_params
    assert "fps" in i2v_params
    assert "guide_strength" in i2v_params
    # And nothing extra beyond image, fps, guide_strength
    extra = i2v_params - t2v_params - {"image", "fps", "guide_strength"}
    assert not extra, f"ltx3_i2v.run() has unexpected extra parameters: {extra}"


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    assert "unet_filename" in sig.parameters
    assert "text_encoder_filename" in sig.parameters
    assert "vae_filename" in sig.parameters


# ---------------------------------------------------------------------------
# AC04 — CPU test passes with mocked inputs (AV sampling chain)
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_LTXV_COND_PATCH = "comfy_diffusion.conditioning.ltxv_conditioning"
_EMPTY_VIDEO_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_ADD_GUIDE_PATCH = "comfy_diffusion.conditioning.ltxv_add_guide"
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
_LTXV_PREPROCESS_PATCH = "comfy_diffusion.image.ltxv_preprocess"


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
    image: Any = None,
    prompt: str = "test prompt",
    fps: int = 25,
    extra_patches: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], MagicMock]:
    """Helper: run pipeline with all heavy dependencies mocked."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import i2v as pipeline_mod

    if image is None:
        image = MagicMock(spec=["mode"])

    mm = _build_mock_mm()
    fake_frames = [MagicMock()]
    fake_audio = {"waveform": MagicMock(), "sample_rate": 44100}

    patches: dict[str, Any] = {
        _RUNTIME_PATCH: {"python_version": "3.12.0"},
        _MM_PATCH: mm,
        _ENCODE_PATCH: (MagicMock(), MagicMock()),
        _LTXV_COND_PATCH: (MagicMock(), MagicMock()),
        _EMPTY_VIDEO_PATCH: {"samples": MagicMock()},
        _ADD_GUIDE_PATCH: (MagicMock(), MagicMock(), {"samples": MagicMock()}),
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
        _IMAGE_TO_TENSOR_PATCH: MagicMock(),
        _LTXV_PREPROCESS_PATCH: MagicMock(),
    }
    if extra_patches:
        patches.update(extra_patches)

    with (
        patch(_RUNTIME_PATCH, return_value=patches[_RUNTIME_PATCH]),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=patches[_ENCODE_PATCH]),
        patch(_LTXV_COND_PATCH, return_value=patches[_LTXV_COND_PATCH]),
        patch(_EMPTY_VIDEO_PATCH, return_value=patches[_EMPTY_VIDEO_PATCH]),
        patch(_ADD_GUIDE_PATCH, return_value=patches[_ADD_GUIDE_PATCH]),
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
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=patches[_IMAGE_TO_TENSOR_PATCH]),
        patch(_LTXV_PREPROCESS_PATCH, return_value=patches[_LTXV_PREPROCESS_PATCH]),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            image=image,
            prompt=prompt,
            fps=fps,
        )

    return result, mm


def test_run_returns_dict_with_frames_and_audio(tmp_path: Path) -> None:
    """run() must return a dict with 'frames' and 'audio' keys."""
    result, _ = _run_mocked(tmp_path)
    assert isinstance(result, dict), "run() must return a dict"
    assert "frames" in result, "result must have a 'frames' key"
    assert "audio" in result, "result must have an 'audio' key"
    assert isinstance(result["frames"], list)
    assert isinstance(result["audio"], dict)


def test_run_calls_ltxv_add_guide(tmp_path: Path) -> None:
    """AC04: ltxv_add_guide() must be called with frame_idx=0."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import i2v as pipeline_mod

    mm = _build_mock_mm()
    add_guide_calls: list[dict[str, Any]] = []

    def fake_add_guide(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        add_guide_calls.append(kwargs)
        return MagicMock(), MagicMock(), {"samples": MagicMock(name="guided")}

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_ADD_GUIDE_PATCH, fake_add_guide),
        patch(_EMPTY_AUDIO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CONCAT_AV_PATCH, return_value={"samples": MagicMock()}),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_VAE_DECODE_PATCH, return_value={"waveform": MagicMock(), "sample_rate": 44100}),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=MagicMock(spec=["mode"]),
            prompt="test",
        )

    assert len(add_guide_calls) == 1, "ltxv_add_guide() must be called exactly once"
    assert add_guide_calls[0].get("frame_idx") == 0, (
        f"ltxv_add_guide() must be called with frame_idx=0, got {add_guide_calls[0]}"
    )


def test_run_accepts_fps_and_guide_strength(tmp_path: Path) -> None:
    """run() must accept fps and guide_strength without raising."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import i2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_ADD_GUIDE_PATCH, return_value=(MagicMock(), MagicMock(), {"samples": MagicMock()})),
        patch(_EMPTY_AUDIO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CONCAT_AV_PATCH, return_value={"samples": MagicMock()}),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_VAE_DECODE_PATCH, return_value={"waveform": MagicMock(), "sample_rate": 44100}),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=MagicMock(spec=["mode"]),
            prompt="test",
            fps=30,
            guide_strength=0.8,
        )


def test_run_uses_load_ltxav_text_encoder(tmp_path: Path) -> None:
    """Gemma 3 text encoder must be loaded via load_ltxav_text_encoder."""
    _, mm = _run_mocked(tmp_path)
    mm.load_ltxav_text_encoder.assert_called_once()
    mm.load_clip.assert_not_called()


def test_run_uses_image_to_tensor_for_pil_image(tmp_path: Path) -> None:
    """PIL Image (has .mode attr) must be converted via image_to_tensor."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import i2v as pipeline_mod

    mm = _build_mock_mm()
    pil_image = MagicMock(spec=["mode"])
    image_to_tensor_calls: list[Any] = []

    def fake_image_to_tensor(img: Any) -> Any:
        image_to_tensor_calls.append(img)
        return MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_ADD_GUIDE_PATCH, return_value=(MagicMock(), MagicMock(), {"samples": MagicMock()})),
        patch(_EMPTY_AUDIO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CONCAT_AV_PATCH, return_value={"samples": MagicMock()}),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_MANUAL_SIGMAS_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_VAE_DECODE_PATCH, return_value={"waveform": MagicMock(), "sample_rate": 44100}),
        patch(_IMAGE_TO_TENSOR_PATCH, fake_image_to_tensor),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=pil_image,
            prompt="test",
        )

    assert len(image_to_tensor_calls) == 1
    assert image_to_tensor_calls[0] is pil_image


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    """run() must raise RuntimeError when check_runtime() returns an error."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import i2v as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={
            "error": "ComfyUI submodule not initialized",
            "python_version": "3.12.0",
        },
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(
                models_dir=tmp_path,
                image=MagicMock(spec=["mode"]),
                prompt="test",
            )


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    """download_models(manifest()) completes without error when all 2 files exist."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    entries = manifest()
    assert len(entries) == 2

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
