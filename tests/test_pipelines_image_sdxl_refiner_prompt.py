"""CPU smoke tests for comfy_diffusion/pipelines/image/sdxl/t2i_refiner_prompt.py.

Covers:
  - File exists, parses, has future annotations, module docstring
  - __all__ = ["manifest", "run"]
  - No top-level comfy/torch imports
  - manifest() returns exactly 2 HFModelEntry items
  - run() signature includes required parameters (including refiner_prompt params)
  - run() calls model loading + sampling stubs via unittest.mock.patch
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
    / "image"
    / "sdxl"
    / "t2i_refiner_prompt.py"
)

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.empty_latent_image"
_SAMPLE_ADVANCED_PATCH = "comfy_diffusion.sampling.sample_advanced"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "sdxl/t2i_refiner_prompt.py must exist"


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
    assert ast.get_docstring(tree), "t2i_refiner_prompt.py must have a module-level docstring"


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as pipeline_mod

    assert set(pipeline_mod.__all__) == {"manifest", "run"}


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
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import (  # noqa: F401
        manifest,
        run,
    )

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks — exactly 2 HFModelEntry items
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_two_entries() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 2


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry)


def test_manifest_base_dest_path() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any("checkpoints" in d and "sd_xl_base" in d for d in dests)


def test_manifest_refiner_dest_path() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any("checkpoints" in d and "sd_xl_refiner" in d for d in dests)


# ---------------------------------------------------------------------------
# run() signature checks
# ---------------------------------------------------------------------------


def test_run_signature_includes_base_params() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    required = {
        "models_dir",
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "steps",
        "base_end_step",
        "cfg",
        "seed",
    }
    assert required <= set(sig.parameters.keys())


def test_run_has_refiner_prompt_param() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    assert "refiner_prompt" in sig.parameters
    assert sig.parameters["refiner_prompt"].default is None


def test_run_has_refiner_negative_prompt_param() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    assert "refiner_negative_prompt" in sig.parameters
    assert sig.parameters["refiner_negative_prompt"].default is None


def test_run_default_width_and_height() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 1024
    assert sig.parameters["height"].default == 1024


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
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        return pipeline_mod.run(models_dir=tmp_path, prompt="test prompt", **run_kwargs)


# ---------------------------------------------------------------------------
# run() behaviour tests
# ---------------------------------------------------------------------------


def test_run_returns_list_of_images(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_run_loads_two_checkpoints(tmp_path: Path) -> None:
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert mm.load_checkpoint.call_count == 2


def test_run_calls_sample_advanced_twice(tmp_path: Path) -> None:
    captured: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        captured.append((args, kwargs))
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=_capture),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert len(captured) == 2, "run() must call sample_advanced twice (base + refiner pass)"


def test_run_encodes_prompt_twice_for_base_and_refiner(tmp_path: Path) -> None:
    encode_calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> tuple[MagicMock, MagicMock]:
        encode_calls.append((args, kwargs))
        return (MagicMock(), MagicMock())

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, side_effect=_capture),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="base prompt")

    assert len(encode_calls) == 2, "run() must call encode_prompt twice (base + refiner)"


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    entries = manifest()
    assert len(entries) == 2

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
