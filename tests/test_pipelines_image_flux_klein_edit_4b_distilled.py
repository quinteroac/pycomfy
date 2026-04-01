"""CPU smoke tests for comfy_diffusion/pipelines/image/flux_klein/edit_4b_distilled.py.

Covers (US-006):
  - AC01: edit_4b_distilled.py exists at the correct path
  - AC02: manifest() returns exactly 3 entries with correct filenames/dirs
  - AC03: run() follows node order and returns list[PIL.Image]
  - AC05: Distilled variant defaults cfg=1, steps=4
  - AC06: run() calls check_runtime() and raises RuntimeError on failure
"""

from __future__ import annotations

import ast
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
    / "flux_klein"
    / "edit_4b_distilled.py"
)

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_ZERO_OUT_PATCH = "comfy_diffusion.conditioning.conditioning_zero_out"
_REFERENCE_LATENT_PATCH = "comfy_diffusion.conditioning.reference_latent"
_EMPTY_FLUX2_LATENT_PATCH = "comfy_diffusion.latent.empty_flux2_latent_image"
_RANDOM_NOISE_PATCH = "comfy_diffusion.sampling.random_noise"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_FLUX2_SCHEDULER_PATCH = "comfy_diffusion.sampling.flux2_scheduler"
_CFG_GUIDER_PATCH = "comfy_diffusion.sampling.cfg_guider"
_SAMPLE_CUSTOM_PATCH = "comfy_diffusion.sampling.sample_custom"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"
_VAE_ENCODE_PATCH = "comfy_diffusion.vae.vae_encode"

_OK_RUNTIME = {"python_version": "3.12.0"}
_ERR_RUNTIME = {"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    return mm


def _run_with_mocks(tmp_path: Path, **run_kwargs: Any) -> list[Any]:
    from comfy_diffusion.pipelines.image.flux_klein import edit_4b_distilled as mod

    mm = _build_mock_mm()
    fake_latent_out = MagicMock(name="latent_out")

    with (
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_REFERENCE_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(fake_latent_out, MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock(name="image")),
        patch(_VAE_ENCODE_PATCH, return_value={"samples": MagicMock()}),
    ):
        return mod.run(
            models_dir=tmp_path,
            prompt="test prompt",
            image=MagicMock(name="ref_image"),
            **run_kwargs,
        )


# ===========================================================================
# File-level checks
# ===========================================================================


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "edit_4b_distilled.py must exist"


def test_pipeline_parses_without_syntax_errors() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PIPELINE_FILE))
    assert isinstance(tree, ast.Module)


def test_pipeline_has_future_annotations() -> None:
    assert "from __future__ import annotations" in _PIPELINE_FILE.read_text(encoding="utf-8")


def test_pipeline_has_module_docstring() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PIPELINE_FILE))
    assert ast.get_docstring(tree), "edit_4b_distilled.py must have a module-level docstring"


def test_no_top_level_comfy_imports() -> None:
    for i, line in enumerate(_PIPELINE_FILE.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), f"Top-level comfy import at line {i}: {line!r}"


# ===========================================================================
# AC02: manifest() — 3 entries with correct filenames
# ===========================================================================


def test_manifest_returns_three_entries() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.edit_4b_distilled import manifest

    entries = manifest()
    assert len(entries) == 3, f"Expected 3 manifest entries, got {len(entries)}"


def test_manifest_unet_filename() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.edit_4b_distilled import manifest

    entries = manifest()
    dest_names = [str(e.dest) for e in entries]
    assert any("flux-2-klein-4b-fp8.safetensors" in d for d in dest_names)


def test_manifest_clip_filename() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.edit_4b_distilled import manifest

    entries = manifest()
    dest_names = [str(e.dest) for e in entries]
    assert any("qwen_3_4b.safetensors" in d for d in dest_names)


def test_manifest_vae_filename() -> None:
    from comfy_diffusion.pipelines.image.flux_klein.edit_4b_distilled import manifest

    entries = manifest()
    dest_names = [str(e.dest) for e in entries]
    assert any("flux2-vae.safetensors" in d for d in dest_names)


# ===========================================================================
# AC03: run() returns list[PIL.Image]
# ===========================================================================


def test_run_returns_list(tmp_path: Path) -> None:
    with patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME):
        result = _run_with_mocks(tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_run_uses_conditioning_zero_out(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import edit_4b_distilled as mod

    mm = _build_mock_mm()
    fake_latent_out = MagicMock(name="latent_out")

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()) as mock_zero,
        patch(_REFERENCE_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(fake_latent_out, MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock(name="image")),
        patch(_VAE_ENCODE_PATCH, return_value={"samples": MagicMock()}),
    ):
        mod.run(models_dir=tmp_path, prompt="test", image=MagicMock())
    mock_zero.assert_called_once()


def test_run_calls_reference_latent_twice(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import edit_4b_distilled as mod

    mm = _build_mock_mm()
    fake_latent_out = MagicMock(name="latent_out")

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_REFERENCE_LATENT_PATCH, return_value=MagicMock()) as mock_ref,
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(fake_latent_out, MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock(name="image")),
        patch(_VAE_ENCODE_PATCH, return_value={"samples": MagicMock()}),
    ):
        mod.run(models_dir=tmp_path, prompt="test", image=MagicMock())
    assert mock_ref.call_count == 2


# ===========================================================================
# AC05: default cfg=1, steps=4 (distilled)
# ===========================================================================


def test_default_cfg_is_1(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import edit_4b_distilled as mod

    mm = _build_mock_mm()
    fake_latent_out = MagicMock(name="latent_out")

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_REFERENCE_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()) as mock_cfg,
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(fake_latent_out, MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock(name="image")),
        patch(_VAE_ENCODE_PATCH, return_value={"samples": MagicMock()}),
    ):
        mod.run(models_dir=tmp_path, prompt="test", image=MagicMock())
    cfg_val = mock_cfg.call_args[0][3]
    assert cfg_val == 1.0


def test_default_steps_is_4(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import edit_4b_distilled as mod

    mm = _build_mock_mm()
    fake_latent_out = MagicMock(name="latent_out")

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_REFERENCE_LATENT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_FLUX2_SCHEDULER_PATCH, return_value=MagicMock()) as mock_sched,
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(fake_latent_out, MagicMock())),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock(name="image")),
        patch(_VAE_ENCODE_PATCH, return_value={"samples": MagicMock()}),
    ):
        mod.run(models_dir=tmp_path, prompt="test", image=MagicMock())
    steps_val = mock_sched.call_args[0][0]
    assert steps_val == 4


# ===========================================================================
# AC06: check_runtime() is called; RuntimeError raised on failure
# ===========================================================================


def test_check_runtime_called(tmp_path: Path) -> None:
    with patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME) as mock_rt:
        try:
            _run_with_mocks(tmp_path)
        except Exception:
            pass
    mock_rt.assert_called()


def test_raises_runtime_error_on_failure(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.flux_klein import edit_4b_distilled as mod

    with (
        patch(_RUNTIME_PATCH, return_value=_ERR_RUNTIME),
        pytest.raises(RuntimeError, match="ComfyUI runtime not available"),
    ):
        mod.run(models_dir=tmp_path, prompt="test", image=MagicMock())
