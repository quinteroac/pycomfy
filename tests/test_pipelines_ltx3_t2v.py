"""Tests for US-005 — comfy_diffusion/pipelines/ltx3_t2v.py pipeline.

Covers:
  AC01: manifest() returns 3 HFModelEntry items with correct dest paths
  AC02: run() matches same parameter signature as ltx2_t2v_distilled.run()
  AC03: CPU test passes with mocked inputs
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
_PIPELINE_FILE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx3" / "t2v.py"
_DISTILLED_FILE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx2" / "t2v_distilled.py"


# ---------------------------------------------------------------------------
# AC04 — file parses without syntax errors; conventions
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "comfy_diffusion/pipelines/ltx3_t2v.py must exist"


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
# AC01 — manifest() returns 3 HFModelEntry items
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_manifest_returns_exactly_three_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 3, f"manifest() must return exactly 3 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any(
        "diffusion_models" in d and "ltx-2.3-22b-distilled-fp8" in d for d in dests
    ), "manifest() must include an entry with dest under diffusion_models/ for ltx-2.3-22b-distilled-fp8"


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include a text_encoders/gemma_3_12B_it_fp4_mixed entry"
    )


def test_manifest_upscaler_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any("upscale_models" in d and "ltx-2-spatial-upscaler" in d for d in dests), (
        "manifest() must include an upscale_models/ltx-2-spatial-upscaler entry"
    )


def test_manifest_all_from_lightricks_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Lightricks/LTX-Video", (
            f"Entry {entry!r} must use repo_id 'Lightricks/LTX-Video'"
        )


def test_manifest_entries_have_nonempty_dest() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import manifest

    for entry in manifest():
        assert entry.dest, f"manifest() entry {entry!r} must have a non-empty dest"


# ---------------------------------------------------------------------------
# AC02 — run() matches same parameter signature as ltx2_t2v_distilled.run()
# ---------------------------------------------------------------------------


def test_run_signature_matches_distilled_pipeline() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v_distilled import run as distilled_run
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import run

    sig_ltx3 = inspect.signature(run)
    sig_distilled = inspect.signature(distilled_run)

    ltx3_params = set(sig_ltx3.parameters.keys())
    distilled_params = set(sig_distilled.parameters.keys())

    assert ltx3_params == distilled_params, (
        f"ltx3_t2v.run() parameters differ from ltx2_t2v_distilled.run().\n"
        f"  Extra in ltx3: {ltx3_params - distilled_params}\n"
        f"  Missing from ltx3: {distilled_params - ltx3_params}"
    )


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import run

    sig = inspect.signature(run)
    params = set(sig.parameters.keys())
    required = {"models_dir", "prompt", "negative_prompt", "width", "height"}
    assert required <= params, f"run() is missing parameters: {required - params}"


def test_run_default_steps_is_eight() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 8, (
        "run() default steps must be 8 for the distilled model"
    )


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import run

    sig = inspect.signature(run)
    assert "unet_filename" in sig.parameters
    assert "text_encoder_filename" in sig.parameters
    assert "upscaler_filename" in sig.parameters
    assert "vae_filename" in sig.parameters


# ---------------------------------------------------------------------------
# AC03 — CPU test passes with mocked inputs
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_UPSAMPLE_PATCH = "comfy_diffusion.latent.ltxv_latent_upsample"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode_batch_tiled"


def _build_mock_mm() -> MagicMock:
    """Return a ModelManager mock with sensible return values."""
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_ltxav_text_encoder.return_value = MagicMock(name="clip")
    mm.load_latent_upscale_model.return_value = MagicMock(name="upscale_model")
    return mm


def test_run_calls_ltxv_latent_upsample_before_vae_decode(tmp_path: Path) -> None:
    """AC03: pipeline calls ltxv_latent_upsample() before vae_decode_batch_tiled()."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import t2v as pipeline_mod

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
        patch(_SAMPLE_PATCH, fake_sample),
        patch(_UPSAMPLE_PATCH, fake_ltxv_latent_upsample),
        patch(_VAE_DECODE_PATCH, fake_vae_decode_batch_tiled),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            prompt="a golden retriever running through a sunlit park",
        )

    assert result is fake_frames
    assert call_order == ["sample", "ltxv_latent_upsample", "vae_decode_batch_tiled"], (
        f"Expected sample → ltxv_latent_upsample → vae_decode_batch_tiled, got: {call_order}"
    )


def test_run_passes_upsampled_samples_to_vae_decode(tmp_path: Path) -> None:
    """Upsampled latent is passed to vae_decode_batch_tiled, not the original."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import t2v as pipeline_mod

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
        patch(_SAMPLE_PATCH, return_value=original_samples),
        patch(_UPSAMPLE_PATCH, fake_ltxv_latent_upsample),
        patch(_VAE_DECODE_PATCH, fake_vae_decode_batch_tiled),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test prompt")

    assert len(received_samples) == 1
    assert received_samples[0] is upsampled_samples


def test_run_uses_load_ltxav_text_encoder(tmp_path: Path) -> None:
    """Gemma 3 text encoder must be loaded via load_ltxav_text_encoder."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import t2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_SAMPLE_PATCH, return_value={"samples": MagicMock()}),
        patch(_UPSAMPLE_PATCH, return_value={"samples": MagicMock()}),
        patch(_VAE_DECODE_PATCH, return_value=[]),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_ltxav_text_encoder.assert_called_once()
    mm.load_clip.assert_not_called()


def test_run_uses_load_latent_upscale_model(tmp_path: Path) -> None:
    """Upscale model must be loaded via load_latent_upscale_model."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import t2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_SAMPLE_PATCH, return_value={"samples": MagicMock()}),
        patch(_UPSAMPLE_PATCH, return_value={"samples": MagicMock()}),
        patch(_VAE_DECODE_PATCH, return_value=[]),
    ):
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_latent_upscale_model.assert_called_once()


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    """run() must raise RuntimeError when check_runtime() returns an error."""
    from comfy_diffusion.pipelines.video.ltx.ltx3 import t2v as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    """download_models(manifest()) completes without error when all 3 files are present."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx3.t2v import manifest

    entries = manifest()
    assert len(entries) == 3

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
