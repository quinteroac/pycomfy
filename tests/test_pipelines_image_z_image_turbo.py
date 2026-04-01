"""CPU smoke tests for comfy_diffusion/pipelines/image/z_image/turbo.py.

Covers:
  - File exists, parses, has future annotations, module docstring
  - __all__ = ["manifest", "run"]
  - No top-level comfy/torch imports
  - manifest() returns exactly 3 HFModelEntry items with correct dest paths
  - run() signature includes required parameters
  - run() calls model loading + sampling stubs via unittest.mock.patch
  - run() uses empty_sd3_latent_image and conditioning_zero_out
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
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "z_image" / "turbo.py"
)

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_ZERO_OUT_PATCH = "comfy_diffusion.conditioning.conditioning_zero_out"
_EMPTY_SD3_LATENT_PATCH = "comfy_diffusion.latent.empty_sd3_latent_image"
_MODEL_SAMPLING_AURA_FLOW_PATCH = "comfy_diffusion.models.model_sampling_aura_flow"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "z_image/turbo.py must exist"


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
    assert ast.get_docstring(tree), "z_image/turbo.py must have a module-level docstring"


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.z_image import turbo

    assert set(turbo.__all__) == {"manifest", "run"}


def test_no_top_level_comfy_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), (
                f"Top-level comfy import at line {i}: {line!r}"
            )


def test_no_top_level_torch_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            assert line.startswith("    "), (
                f"Top-level torch import at line {i}: {line!r}"
            )


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks — exactly 3 HFModelEntry items
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_three_entries() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 3, f"manifest() must return exactly 3 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry)


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any("diffusion_models" in d and "z_image_turbo" in d for d in dests)


def test_manifest_clip_dest_path() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any("text_encoders" in d and "qwen" in d.lower() for d in dests)


def test_manifest_vae_dest_path() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any("vae" in d and "ae" in d for d in dests)


def test_manifest_uses_comfy_org_repo() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    repos = [entry.repo_id for entry in manifest()]
    assert any("Comfy-Org/z_image_turbo" in r for r in repos)


# ---------------------------------------------------------------------------
# run() signature checks
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import run

    sig = inspect.signature(run)
    required = {"models_dir", "prompt", "width", "height", "steps", "seed"}
    assert required <= set(sig.parameters.keys())


def test_run_default_width_height_and_steps() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 1024
    assert sig.parameters["height"].default == 1024
    assert sig.parameters["steps"].default == 4


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    return mm


def _run_with_mocks(tmp_path: Path, **run_kwargs: Any) -> list[Any]:
    from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod

    mm = _build_mock_mm()
    fake_model_patched = MagicMock(name="model_patched")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_SD3_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_MODEL_SAMPLING_AURA_FLOW_PATCH, return_value=fake_model_patched),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock(name="image")),
    ):
        return pipeline_mod.run(models_dir=tmp_path, prompt="test prompt", **run_kwargs)


# ---------------------------------------------------------------------------
# run() behaviour tests
# ---------------------------------------------------------------------------


def test_run_returns_list_of_images(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_run_returns_vae_decoded_image(tmp_path: Path) -> None:
    fake_image = MagicMock(name="image")
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_SD3_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_MODEL_SAMPLING_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod
        result = pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert result[0] is fake_image


def test_run_calls_empty_sd3_latent_image(tmp_path: Path) -> None:
    """run() must use empty_sd3_latent_image (16-channel SD3-family latent)."""
    sd3_calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> dict[str, MagicMock]:
        sd3_calls.append((args, kwargs))
        return {"samples": MagicMock()}

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_SD3_LATENT_PATCH, side_effect=_capture),
        patch(_MODEL_SAMPLING_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert len(sd3_calls) == 1, "run() must call empty_sd3_latent_image once"


def test_run_calls_conditioning_zero_out(tmp_path: Path) -> None:
    """run() must derive negative conditioning via conditioning_zero_out."""
    zero_out_calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        zero_out_calls.append((args, kwargs))
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, side_effect=_capture),
        patch(_EMPTY_SD3_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_MODEL_SAMPLING_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert len(zero_out_calls) == 1, "run() must call conditioning_zero_out once"


def test_run_calls_model_sampling_aura_flow(tmp_path: Path) -> None:
    """run() must apply model_sampling_aura_flow patch before sampling."""
    aura_calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        aura_calls.append((args, kwargs))
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_SD3_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_MODEL_SAMPLING_AURA_FLOW_PATCH, side_effect=_capture),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert len(aura_calls) == 1, "run() must call model_sampling_aura_flow once"


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    entries = manifest()
    assert len(entries) == 3

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
