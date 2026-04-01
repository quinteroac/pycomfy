"""CPU smoke tests for comfy_diffusion/pipelines/image/sdxl/turbo.py.

Covers:
  - File exists, parses, has future annotations, module docstring
  - __all__ = ["manifest", "run"]
  - No top-level comfy/torch imports
  - manifest() returns exactly 1 HFModelEntry
  - run() signature includes required parameters
  - run() calls model loading + sampling stubs via unittest.mock.patch
  - run() uses sd_turbo_scheduler and sample_custom_simple
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
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "sdxl" / "turbo.py"
)

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.empty_latent_image"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_SD_TURBO_PATCH = "comfy_diffusion.sampling.sd_turbo_scheduler"
_SAMPLE_CUSTOM_SIMPLE_PATCH = "comfy_diffusion.sampling.sample_custom_simple"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "sdxl/turbo.py must exist"


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
    assert ast.get_docstring(tree), "sdxl/turbo.py must have a module-level docstring"


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.sdxl import turbo

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
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks — exactly 1 HFModelEntry
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_one_entry() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 1, f"manifest() must return exactly 1 entry, got {len(result)}"


def test_manifest_entry_is_hf_model_entry() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest

    assert isinstance(manifest()[0], HFModelEntry)


def test_manifest_checkpoint_dest_path() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any("checkpoints" in d and "turbo" in d.lower() for d in dests), (
        "manifest() must include a checkpoints/sd_xl_turbo_*.safetensors entry"
    )


def test_manifest_uses_stabilityai_sdxl_turbo_repo() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest

    repos = [entry.repo_id for entry in manifest()]
    assert any("sdxl-turbo" in r.lower() for r in repos)


# ---------------------------------------------------------------------------
# run() signature checks
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import run

    sig = inspect.signature(run)
    required = {"models_dir", "prompt", "negative_prompt", "width", "height", "steps", "cfg", "seed"}
    assert required <= set(sig.parameters.keys())


def test_run_default_width_height_and_steps() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 512
    assert sig.parameters["height"].default == 512
    assert sig.parameters["steps"].default == 1


def test_run_default_cfg_is_zero() -> None:
    """SDXL Turbo default CFG is 0.0 (guidance-free distilled model)."""
    from comfy_diffusion.pipelines.image.sdxl.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["cfg"].default == 0.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    result = MagicMock()
    result.model = MagicMock(name="model")
    result.clip = MagicMock(name="clip")
    result.vae = MagicMock(name="vae")
    mm.load_checkpoint.return_value = result
    return mm


def _run_with_mocks(tmp_path: Path, **run_kwargs: Any) -> list[Any]:
    from comfy_diffusion.pipelines.image.sdxl import turbo as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock(name="sampler")),
        patch(_SD_TURBO_PATCH, return_value=MagicMock(name="sigmas")),
        patch(_SAMPLE_CUSTOM_SIMPLE_PATCH, return_value=MagicMock(name="latent_out")),
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
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SD_TURBO_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_SIMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        from comfy_diffusion.pipelines.image.sdxl import turbo as pipeline_mod
        result = pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert result[0] is fake_image


def test_run_loads_exactly_one_checkpoint(tmp_path: Path) -> None:
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SD_TURBO_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_SIMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import turbo as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert mm.load_checkpoint.call_count == 1


def test_run_calls_sd_turbo_scheduler(tmp_path: Path) -> None:
    sd_turbo_calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        sd_turbo_calls.append((args, kwargs))
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SD_TURBO_PATCH, side_effect=_capture),
        patch(_SAMPLE_CUSTOM_SIMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import turbo as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert len(sd_turbo_calls) == 1, "run() must call sd_turbo_scheduler once"


def test_run_calls_sample_custom_simple(tmp_path: Path) -> None:
    sample_calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        sample_calls.append((args, kwargs))
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SD_TURBO_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_SIMPLE_PATCH, side_effect=_capture),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import turbo as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert len(sample_calls) == 1, "run() must call sample_custom_simple once"


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.sdxl import turbo as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest

    entries = manifest()
    assert len(entries) == 1

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
