"""CPU smoke tests for comfy_diffusion/pipelines/image/flux_klein/.

Covers (US-005):
  - AC01: t2i_4b_base.manifest() returns 3 entries with correct filenames/dirs
  - AC02: t2i_4b_base.run() follows node order and returns list[PIL.Image]
  - AC03: t2i_4b_distilled.manifest() returns 3 entries with flux-2-klein-4b.safetensors
  - AC04: t2i_4b_distilled.run() uses cfg=1, conditioning_zero_out, steps=4 default
  - AC05: Both pipelines call check_runtime() and raise RuntimeError on failure
  - AC06: CPU tests via mocks verify manifest length, field names, run returns list[PIL.Image]
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
_BASE_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "flux_klein" / "t2i_4b_base.py"
)
_DISTILLED_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "flux_klein" / "t2i_4b_distilled.py"
)

# ---------------------------------------------------------------------------
# Patch targets — base
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_ZERO_OUT_PATCH = "comfy_diffusion.conditioning.conditioning_zero_out"
_EMPTY_FLUX2_LATENT_PATCH = "comfy_diffusion.latent.empty_flux2_latent_image"
_RANDOM_NOISE_PATCH = "comfy_diffusion.sampling.random_noise"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_FLUX2_SCHEDULER_PATCH = "comfy_diffusion.sampling.flux2_scheduler"
_CFG_GUIDER_PATCH = "comfy_diffusion.sampling.cfg_guider"
_SAMPLE_CUSTOM_PATCH = "comfy_diffusion.sampling.sample_custom"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"

_OK_RUNTIME = {"python_version": "3.12.0"}
_ERR_RUNTIME = {"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"}


# ===========================================================================
# Shared helpers
# ===========================================================================


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    return mm


def _run_base_with_mocks(tmp_path: Path, **run_kwargs: Any) -> list[Any]:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    mm = _build_mock_mm()
    fake_latent_out = MagicMock(name="latent_out")

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(fake_latent_out, MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock(name="image")),
    ):
        return mod.run(models_dir=tmp_path, prompt="test prompt", **run_kwargs)


def _run_distilled_with_mocks(tmp_path: Path, **run_kwargs: Any) -> list[Any]:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_distilled as mod

    mm = _build_mock_mm()
    fake_latent_out = MagicMock(name="latent_out")

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(fake_latent_out, MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock(name="image")),
    ):
        return mod.run(models_dir=tmp_path, prompt="test prompt", **run_kwargs)


# ===========================================================================
# File-level checks — both files
# ===========================================================================


@pytest.mark.parametrize("fpath", [_BASE_FILE, _DISTILLED_FILE])
def test_pipeline_file_exists(fpath: Path) -> None:
    assert fpath.is_file(), f"{fpath.name} must exist"


@pytest.mark.parametrize("fpath", [_BASE_FILE, _DISTILLED_FILE])
def test_pipeline_parses_without_syntax_errors(fpath: Path) -> None:
    source = fpath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(fpath))
    assert isinstance(tree, ast.Module)


@pytest.mark.parametrize("fpath", [_BASE_FILE, _DISTILLED_FILE])
def test_pipeline_has_future_annotations(fpath: Path) -> None:
    assert "from __future__ import annotations" in fpath.read_text(encoding="utf-8")


@pytest.mark.parametrize("fpath", [_BASE_FILE, _DISTILLED_FILE])
def test_pipeline_has_module_docstring(fpath: Path) -> None:
    source = fpath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(fpath))
    assert ast.get_docstring(tree), f"{fpath.name} must have a module-level docstring"


@pytest.mark.parametrize("fpath", [_BASE_FILE, _DISTILLED_FILE])
def test_no_top_level_comfy_imports(fpath: Path) -> None:
    for i, line in enumerate(fpath.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), f"Top-level comfy import at line {i}: {line!r}"


@pytest.mark.parametrize("fpath", [_BASE_FILE, _DISTILLED_FILE])
def test_no_top_level_torch_imports(fpath: Path) -> None:
    for i, line in enumerate(fpath.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            assert line.startswith("    "), f"Top-level torch import at line {i}: {line!r}"


# ===========================================================================
# __all__ checks
# ===========================================================================


def test_base_dunder_all() -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    assert set(mod.__all__) == {"manifest", "run"}


def test_distilled_dunder_all() -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_distilled as mod

    assert set(mod.__all__) == {"manifest", "run"}


# ===========================================================================
# AC01 — t2i_4b_base.manifest(): 3 entries with correct filenames/directories
# ===========================================================================


def test_base_manifest_returns_exactly_three_entries() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 3, f"manifest() must return exactly 3 entries, got {len(result)}"


def test_base_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry)


def test_base_manifest_unet_filename() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any(
        "diffusion_models" in d and "flux-2-klein-base-4b.safetensors" in d for d in dests
    ), f"Expected flux-2-klein-base-4b.safetensors in diffusion_models, got: {dests}"


def test_base_manifest_clip_filename() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any(
        "text_encoders" in d and "qwen_3_4b.safetensors" in d for d in dests
    ), f"Expected qwen_3_4b.safetensors in text_encoders, got: {dests}"


def test_base_manifest_vae_filename() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any(
        "vae" in d and "flux2-vae.safetensors" in d for d in dests
    ), f"Expected flux2-vae.safetensors in vae, got: {dests}"


# ===========================================================================
# AC02 — t2i_4b_base.run() signature and node order
# ===========================================================================


def test_base_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import run

    sig = inspect.signature(run)
    required = {"models_dir", "prompt", "width", "height", "steps", "cfg", "seed"}
    assert required <= set(sig.parameters.keys())


def test_base_run_default_steps() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 20


def test_base_run_default_cfg() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import run

    sig = inspect.signature(run)
    assert sig.parameters["cfg"].default == 5.0


def test_base_run_returns_list_of_images(tmp_path: Path) -> None:
    result = _run_base_with_mocks(tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_base_run_returns_vae_decoded_image(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    fake_image = MagicMock(name="decoded_image")
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        result = mod.run(models_dir=tmp_path, prompt="test")

    assert result[0] is fake_image


def test_base_run_calls_empty_flux2_latent_image(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> dict[str, MagicMock]:
        calls.append((args, kwargs))
        return {"samples": MagicMock()}

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_FLUX2_LATENT_PATCH, side_effect=_capture),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="test")

    assert len(calls) == 1, "run() must call empty_flux2_latent_image exactly once"


def test_base_run_calls_flux2_scheduler(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append((args, kwargs))
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, side_effect=_capture),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="test")

    assert len(calls) == 1, "run() must call flux2_scheduler exactly once"


def test_base_run_calls_cfg_guider(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append((args, kwargs))
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, side_effect=_capture),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="test", cfg=5.0)

    assert len(calls) == 1, "run() must call cfg_guider exactly once"
    # CFG value passed correctly.
    assert calls[0][0][3] == 5.0


def test_base_run_calls_sample_custom(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> tuple[MagicMock, MagicMock]:
        calls.append((args, kwargs))
        return MagicMock(), MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, side_effect=_capture),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="test")

    assert len(calls) == 1, "run() must call sample_custom exactly once"


def test_base_run_uses_euler_sampler(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    sampler_calls: list[Any] = []

    def _capture(name: str) -> MagicMock:
        sampler_calls.append(name)
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, side_effect=_capture),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="test")

    assert sampler_calls == ["euler"], f"Expected euler sampler, got: {sampler_calls}"


def test_base_run_loads_clip_with_flux2_type(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="test")

    clip_call = mm.load_clip.call_args
    assert clip_call is not None
    clip_type = clip_call[1].get("clip_type") or clip_call[0][1]
    assert clip_type == "flux2", f"Expected clip_type='flux2', got: {clip_type!r}"


# ===========================================================================
# AC03 — t2i_4b_distilled.manifest(): flux-2-klein-4b.safetensors in diffusion_models
# ===========================================================================


def test_distilled_manifest_returns_exactly_three_entries() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 3, f"manifest() must return exactly 3 entries, got {len(result)}"


def test_distilled_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry)


def test_distilled_manifest_unet_filename() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any(
        "diffusion_models" in d and "flux-2-klein-4b.safetensors" in d
        and "base" not in d
        for d in dests
    ), f"Expected flux-2-klein-4b.safetensors (not base) in diffusion_models, got: {dests}"


def test_distilled_manifest_clip_filename() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any(
        "text_encoders" in d and "qwen_3_4b.safetensors" in d for d in dests
    ), f"Expected qwen_3_4b.safetensors in text_encoders, got: {dests}"


def test_distilled_manifest_vae_filename() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any(
        "vae" in d and "flux2-vae.safetensors" in d for d in dests
    ), f"Expected flux2-vae.safetensors in vae, got: {dests}"


# ===========================================================================
# AC04 — t2i_4b_distilled.run(): cfg=1, conditioning_zero_out, steps=4 default
# ===========================================================================


def test_distilled_run_default_steps() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 4


def test_distilled_run_default_cfg() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import run

    sig = inspect.signature(run)
    assert sig.parameters["cfg"].default == 1.0


def test_distilled_run_calls_conditioning_zero_out(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_distilled as mod

    calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append((args, kwargs))
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, side_effect=_capture),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="test")

    assert len(calls) == 1, "distilled run() must call conditioning_zero_out exactly once"


def test_distilled_run_passes_cfg_one_to_guider(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_distilled as mod

    calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append((args, kwargs))
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, side_effect=_capture),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="test")  # default cfg=1.0

    assert len(calls) == 1
    assert calls[0][0][3] == 1.0, f"Expected cfg=1.0, got: {calls[0][0][3]}"


def test_distilled_run_returns_list_of_images(tmp_path: Path) -> None:
    result = _run_distilled_with_mocks(tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_distilled_run_does_not_call_conditioning_zero_out_in_base(tmp_path: Path) -> None:
    """Base pipeline must NOT call conditioning_zero_out."""
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    calls: list[Any] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append((args, kwargs))
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, side_effect=_capture),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="test")

    assert len(calls) == 0, "Base pipeline must NOT call conditioning_zero_out"


# ===========================================================================
# AC05 — Both pipelines raise RuntimeError on check_runtime() failure
# ===========================================================================


def test_base_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    with patch(_RUNTIME_PATCH, return_value=_ERR_RUNTIME):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            mod.run(models_dir=tmp_path, prompt="test")


def test_distilled_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_distilled as mod

    with patch(_RUNTIME_PATCH, return_value=_ERR_RUNTIME):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            mod.run(models_dir=tmp_path, prompt="test")


# ===========================================================================
# AC06 — path override params
# ===========================================================================


def test_base_run_signature_includes_path_override_params() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import run

    sig = inspect.signature(run)
    assert "unet_filename" in sig.parameters
    assert "clip_filename" in sig.parameters
    assert "vae_filename" in sig.parameters
    assert sig.parameters["unet_filename"].default is None
    assert sig.parameters["clip_filename"].default is None
    assert sig.parameters["vae_filename"].default is None


def test_distilled_run_signature_includes_path_override_params() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import run

    sig = inspect.signature(run)
    assert "unet_filename" in sig.parameters
    assert "clip_filename" in sig.parameters
    assert "vae_filename" in sig.parameters


def test_base_run_uses_custom_filenames(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_base as mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(
            models_dir=tmp_path,
            prompt="test",
            unet_filename="custom_unet.safetensors",
            clip_filename="custom_clip.safetensors",
            vae_filename="custom_vae.safetensors",
        )

    assert str(mm.load_unet.call_args[0][0]).endswith("custom_unet.safetensors")
    assert str(mm.load_clip.call_args[0][0]).endswith("custom_clip.safetensors")
    assert str(mm.load_vae.call_args[0][0]).endswith("custom_vae.safetensors")


def test_distilled_run_uses_custom_filenames(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import t2i_4b_distilled as mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(
            models_dir=tmp_path,
            prompt="test",
            unet_filename="custom_unet.safetensors",
            clip_filename="custom_clip.safetensors",
            vae_filename="custom_vae.safetensors",
        )

    assert str(mm.load_unet.call_args[0][0]).endswith("custom_unet.safetensors")
    assert str(mm.load_clip.call_args[0][0]).endswith("custom_clip.safetensors")
    assert str(mm.load_vae.call_args[0][0]).endswith("custom_vae.safetensors")


# ===========================================================================
# download_models idempotency check
# ===========================================================================


def test_download_models_idempotent_base(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_base import manifest

    entries = manifest()
    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)


def test_download_models_idempotent_distilled(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.flux_klein.t2i_4b_distilled import manifest

    entries = manifest()
    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
