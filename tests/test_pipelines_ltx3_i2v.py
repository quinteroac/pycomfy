"""Tests for US-006 — comfy_diffusion/pipelines/ltx3_i2v.py pipeline.

Covers:
  AC01: manifest() returns same 3 entries as ltx3_t2v
  AC02: run() adds image and fps parameters vs ltx3_t2v.run()
  AC03: fps is passed to ltxv_empty_latent_video() if supported, else documented as reserved
  AC04: CPU test passes with mocked inputs
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
# AC01 — manifest() returns same 3 entries as ltx3_t2v
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_manifest_returns_exactly_three_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 3, f"manifest() must return exactly 3 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_matches_ltx3_t2v() -> None:
    """AC01: manifest() returns the same 3 entries as ltx3_t2v.manifest()."""
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


def test_manifest_upscaler_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("upscale_models" in d and "ltx-2-spatial-upscaler" in d for d in dests), (
        "manifest() must include upscale_models/ltx-2-spatial-upscaler"
    )


def test_manifest_all_from_lightricks_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Lightricks/LTX-Video"


# ---------------------------------------------------------------------------
# AC02 — run() adds image and fps parameters vs ltx3_t2v.run()
# ---------------------------------------------------------------------------


def test_run_has_image_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    assert "image" in sig.parameters, "run() must have an 'image' parameter"


def test_run_has_fps_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    assert "fps" in sig.parameters, "run() must have an 'fps' parameter"


def test_run_fps_default_is_24() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    assert sig.parameters["fps"].default == 24, "fps default must be 24"


def test_run_fps_annotation_is_int() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    # Under from __future__ import annotations, annotation is a string
    ann = sig.parameters["fps"].annotation
    assert ann in (int, "int"), f"fps must be annotated as int, got {ann!r}"


def test_run_has_all_t2v_params_plus_image_and_fps() -> None:
    """run() must include all ltx3_t2v params plus image and fps."""
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run as i2v_run
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import run as t2v_run

    i2v_params = set(inspect.signature(i2v_run).parameters)
    t2v_params = set(inspect.signature(t2v_run).parameters)

    # i2v must contain everything from t2v
    assert t2v_params <= i2v_params, (
        f"ltx3_i2v.run() is missing t2v parameters: {t2v_params - i2v_params}"
    )
    # Plus image and fps
    assert "image" in i2v_params
    assert "fps" in i2v_params
    # And nothing extra beyond image and fps
    extra = i2v_params - t2v_params - {"image", "fps"}
    assert not extra, f"ltx3_i2v.run() has unexpected extra parameters: {extra}"


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    assert "unet_filename" in sig.parameters
    assert "text_encoder_filename" in sig.parameters
    assert "upscaler_filename" in sig.parameters
    assert "vae_filename" in sig.parameters


def test_run_default_steps_is_eight() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 8


# ---------------------------------------------------------------------------
# AC03 — fps is documented as reserved (ltxv_empty_latent_video does not
#         accept fps yet); verify it is accepted by run() and not forwarded
# ---------------------------------------------------------------------------


def test_fps_documented_as_reserved_in_docstring() -> None:
    """AC03: fps docstring must mention 'reserved' since the latent function
    does not yet accept an fps argument."""
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import run

    doc = inspect.getdoc(run) or ""
    assert "reserved" in doc.lower(), (
        "run() docstring must document fps as reserved until ltxv_empty_latent_video "
        "supports it"
    )


def test_fps_accepted_without_error(tmp_path: Path) -> None:
    """AC03: run() must accept fps without raising, even if not forwarded."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import i2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_SAMPLE_PATCH, return_value={"samples": MagicMock()}),
        patch(_UPSAMPLE_PATCH, return_value={"samples": MagicMock()}),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=MagicMock(spec=["mode"]),
            prompt="test",
            fps=30,
        )


# ---------------------------------------------------------------------------
# AC04 — CPU test passes with mocked inputs
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_UPSAMPLE_PATCH = "comfy_diffusion.latent.ltxv_latent_upsample"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode_batch_tiled"
_IMG_TO_VIDEO_PATCH = "comfy_diffusion.video.ltxv_img_to_video_inplace"
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"
_LOAD_IMAGE_PATCH = "comfy_diffusion.image.load_image"
_LTXV_PREPROCESS_PATCH = "comfy_diffusion.image.ltxv_preprocess"


def _build_mock_mm() -> MagicMock:
    """Return a ModelManager mock with sensible return values."""
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_ltxav_text_encoder.return_value = MagicMock(name="clip")
    mm.load_latent_upscale_model.return_value = MagicMock(name="upscale_model")
    return mm


def _run_mocked(
    tmp_path: Path,
    *,
    image: Any = None,
    prompt: str = "test prompt",
    fps: int = 24,
    extra_patches: dict[str, Any] | None = None,
) -> tuple[list[Any], MagicMock]:
    """Helper: run pipeline with all heavy dependencies mocked."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import i2v as pipeline_mod

    if image is None:
        image = MagicMock(spec=["mode"])

    mm = _build_mock_mm()
    fake_frames = [MagicMock()]

    patches: dict[str, Any] = {
        _RUNTIME_PATCH: {"python_version": "3.12.0"},
        _MM_PATCH: mm,
        _ENCODE_PATCH: (MagicMock(), MagicMock()),
        _EMPTY_LATENT_PATCH: {"samples": MagicMock()},
        _IMG_TO_VIDEO_PATCH: {"samples": MagicMock()},
        _SAMPLE_PATCH: {"samples": MagicMock()},
        _UPSAMPLE_PATCH: {"samples": MagicMock()},
        _VAE_DECODE_PATCH: fake_frames,
        _IMAGE_TO_TENSOR_PATCH: MagicMock(),
        _LTXV_PREPROCESS_PATCH: MagicMock(),
    }
    if extra_patches:
        patches.update(extra_patches)

    ctx = [
        patch(target, return_value=val) if not callable(val) else patch(target, val)
        for target, val in patches.items()
    ]

    # Use nested patches for simplicity
    with (
        patch(_RUNTIME_PATCH, return_value=patches[_RUNTIME_PATCH]),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=patches[_ENCODE_PATCH]),
        patch(_EMPTY_LATENT_PATCH, return_value=patches[_EMPTY_LATENT_PATCH]),
        patch(_IMG_TO_VIDEO_PATCH, return_value=patches[_IMG_TO_VIDEO_PATCH]),
        patch(_SAMPLE_PATCH, return_value=patches[_SAMPLE_PATCH]),
        patch(_UPSAMPLE_PATCH, return_value=patches[_UPSAMPLE_PATCH]),
        patch(_VAE_DECODE_PATCH, return_value=patches[_VAE_DECODE_PATCH]),
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


def test_run_returns_list_of_frames(tmp_path: Path) -> None:
    frames, _ = _run_mocked(tmp_path)
    assert isinstance(frames, list)
    assert len(frames) == 1


def test_run_calls_ltxv_img_to_video_inplace(tmp_path: Path) -> None:
    """AC04: ltxv_img_to_video_inplace() must be called during run()."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import i2v as pipeline_mod

    mm = _build_mock_mm()
    img_to_video_calls: list[Any] = []

    def fake_img_to_video(vae: Any, image: Any, latent: Any, **kwargs: Any) -> dict[str, Any]:
        img_to_video_calls.append((vae, image, latent))
        return {"samples": MagicMock(name="injected")}

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, fake_img_to_video),
        patch(_SAMPLE_PATCH, return_value={"samples": MagicMock()}),
        patch(_UPSAMPLE_PATCH, return_value={"samples": MagicMock()}),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=MagicMock(spec=["mode"]),
            prompt="test",
        )

    assert len(img_to_video_calls) == 1, (
        "ltxv_img_to_video_inplace() must be called exactly once"
    )


def test_run_calls_ltxv_latent_upsample_before_vae_decode(tmp_path: Path) -> None:
    """AC04: pipeline calls ltxv_latent_upsample() before vae_decode_batch_tiled()."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import i2v as pipeline_mod

    call_order: list[str] = []
    fake_samples = {"samples": MagicMock()}
    fake_upsampled = {"samples": MagicMock()}
    fake_frames = [MagicMock()]

    mm = _build_mock_mm()

    def fake_sample(**kwargs: Any) -> dict[str, Any]:
        call_order.append("sample")
        return fake_samples

    def fake_ltxv_latent_upsample(
        samples: Any, *, upscale_model: Any, vae: Any
    ) -> dict[str, Any]:
        call_order.append("ltxv_latent_upsample")
        return fake_upsampled

    def fake_vae_decode_batch_tiled(vae: Any, samples: Any) -> list[Any]:
        call_order.append("vae_decode_batch_tiled")
        return fake_frames

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_SAMPLE_PATCH, fake_sample),
        patch(_UPSAMPLE_PATCH, fake_ltxv_latent_upsample),
        patch(_VAE_DECODE_PATCH, fake_vae_decode_batch_tiled),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            image=MagicMock(spec=["mode"]),
            prompt="a golden retriever running through a sunlit park",
        )

    assert result is fake_frames
    assert call_order == ["sample", "ltxv_latent_upsample", "vae_decode_batch_tiled"], (
        f"Expected sample → ltxv_latent_upsample → vae_decode_batch_tiled, got: {call_order}"
    )


def test_run_passes_upsampled_samples_to_vae_decode(tmp_path: Path) -> None:
    """Upsampled latent is passed to vae_decode_batch_tiled, not the original."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import i2v as pipeline_mod

    original_samples = {"samples": MagicMock(name="original")}
    upsampled_samples = {"samples": MagicMock(name="upsampled")}
    received_samples: list[Any] = []

    mm = _build_mock_mm()

    def fake_ltxv_latent_upsample(
        samples: Any, *, upscale_model: Any, vae: Any
    ) -> dict[str, Any]:
        assert samples is original_samples
        return upsampled_samples

    def fake_vae_decode_batch_tiled(vae: Any, samples: Any) -> list[Any]:
        received_samples.append(samples)
        return []

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_SAMPLE_PATCH, return_value=original_samples),
        patch(_UPSAMPLE_PATCH, fake_ltxv_latent_upsample),
        patch(_VAE_DECODE_BATCH_TILED_PATCH := _VAE_DECODE_PATCH, fake_vae_decode_batch_tiled),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_LTXV_PREPROCESS_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            image=MagicMock(spec=["mode"]),
            prompt="test prompt",
        )

    assert len(received_samples) == 1
    assert received_samples[0] is upsampled_samples


def test_run_uses_load_ltxav_text_encoder(tmp_path: Path) -> None:
    """Gemma 3 text encoder must be loaded via load_ltxav_text_encoder."""
    _, mm = _run_mocked(tmp_path)
    mm.load_ltxav_text_encoder.assert_called_once()
    mm.load_clip.assert_not_called()


def test_run_uses_load_latent_upscale_model(tmp_path: Path) -> None:
    """Upscale model must be loaded via load_latent_upscale_model."""
    _, mm = _run_mocked(tmp_path)
    mm.load_latent_upscale_model.assert_called_once()


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
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_SAMPLE_PATCH, return_value={"samples": MagicMock()}),
        patch(_UPSAMPLE_PATCH, return_value={"samples": MagicMock()}),
        patch(_VAE_DECODE_PATCH, return_value=[]),
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
    """download_models(manifest()) completes without error when all 3 files exist."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx3.i2v import manifest

    entries = manifest()
    assert len(entries) == 3

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
