"""Tests for comfy_diffusion/pipelines/image/sdxl/turbo.py — SDXL Turbo T2I pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring
  - Exports manifest and run; no top-level comfy/torch imports
  - __all__ = ["manifest", "run"]
  - manifest() returns exactly 1 HFModelEntry with correct dest path
  - run() signature: models_dir, prompt, negative_prompt, width, height, steps, cfg, seed
  - run() defaults: width=512, height=512, steps=1, cfg=0.0, seed=0
  - run() uses euler_ancestral sampler via get_sampler()
  - run() uses sd_turbo_scheduler for sigmas
  - run() calls sample_custom_simple with add_noise=True
  - run() returns list[PIL.Image.Image]
  - sd_turbo_scheduler and sample_custom_simple exist in sampling module
  - download_models idempotent with 1 entry
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
_PIPELINE_FILE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "sdxl" / "turbo.py"
_SAMPLING_FILE = _REPO_ROOT / "comfy_diffusion" / "sampling.py"

# ---------------------------------------------------------------------------
# Patch targets (source module paths for lazy imports)
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.empty_latent_image"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_SD_TURBO_SCHEDULER_PATCH = "comfy_diffusion.sampling.sd_turbo_scheduler"
_SAMPLE_CUSTOM_SIMPLE_PATCH = "comfy_diffusion.sampling.sample_custom_simple"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"


# ---------------------------------------------------------------------------
# File-level checks (AC01, AC07, AC08)
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "turbo.py must exist"


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
    assert docstring, "turbo.py must have a module-level docstring"


def test_pipeline_has_dunder_all_with_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert "__all__" in source
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


def test_no_top_level_torch_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            assert line.startswith("    "), (
                f"Top-level torch import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.sdxl import turbo

    assert hasattr(turbo, "__all__")
    assert set(turbo.__all__) == {"manifest", "run"}


# ---------------------------------------------------------------------------
# Manifest checks (AC02)
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_one_entry() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 1, f"manifest() must return exactly 1 entry, got {len(result)}"


def test_manifest_entry_is_hf_model_entry() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest

    entry = manifest()[0]
    assert isinstance(entry, HFModelEntry), (
        f"manifest() entry must be an HFModelEntry, got {type(entry)!r}"
    )


def test_manifest_dest_path_is_checkpoints_turbo() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest

    dest = str(manifest()[0].dest)
    assert "checkpoints" in dest, "dest must be under checkpoints/"
    assert "sd_xl_turbo_1.0_fp16" in dest, (
        "dest must reference sd_xl_turbo_1.0_fp16.safetensors"
    )


def test_manifest_uses_stabilityai_sdxl_turbo_repo() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest

    entry = manifest()[0]
    assert "stabilityai" in entry.repo_id
    assert "turbo" in entry.repo_id.lower()


def test_manifest_filename_is_fp16_safetensors() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import manifest

    entry = manifest()[0]
    assert entry.filename == "sd_xl_turbo_1.0_fp16.safetensors"


# ---------------------------------------------------------------------------
# run() signature checks (AC03)
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import run

    sig = inspect.signature(run)
    required = {"models_dir", "prompt", "negative_prompt", "width", "height",
                "steps", "cfg", "seed"}
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_default_width_height() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 512
    assert sig.parameters["height"].default == 512


def test_run_default_steps_is_one() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 1


def test_run_default_cfg_is_zero() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["cfg"].default == 0.0


def test_run_default_seed_is_zero() -> None:
    from comfy_diffusion.pipelines.image.sdxl.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["seed"].default == 0


# ---------------------------------------------------------------------------
# Sampling module — new wrappers exist (AC05, AC06)
# ---------------------------------------------------------------------------


def test_sd_turbo_scheduler_exists_in_sampling() -> None:
    from comfy_diffusion import sampling

    assert hasattr(sampling, "sd_turbo_scheduler"), (
        "sd_turbo_scheduler must be defined in comfy_diffusion.sampling"
    )
    assert callable(sampling.sd_turbo_scheduler)


def test_sd_turbo_scheduler_in_dunder_all() -> None:
    from comfy_diffusion import sampling

    assert "sd_turbo_scheduler" in sampling.__all__


def test_sample_custom_simple_exists_in_sampling() -> None:
    from comfy_diffusion import sampling

    assert hasattr(sampling, "sample_custom_simple"), (
        "sample_custom_simple must be defined in comfy_diffusion.sampling"
    )
    assert callable(sampling.sample_custom_simple)


def test_sample_custom_simple_in_dunder_all() -> None:
    from comfy_diffusion import sampling

    assert "sample_custom_simple" in sampling.__all__


def test_sample_custom_simple_signature() -> None:
    from comfy_diffusion.sampling import sample_custom_simple

    sig = inspect.signature(sample_custom_simple)
    params = set(sig.parameters.keys())
    required = {"model", "add_noise", "noise_seed", "cfg", "positive", "negative", "sampler", "sigmas", "latent_image"}
    assert required <= params, f"sample_custom_simple missing params: {required - params}"


# ---------------------------------------------------------------------------
# Helper for run() behaviour tests
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    result = MagicMock()
    result.model = MagicMock(name="model")
    result.clip = MagicMock(name="clip")
    result.vae = MagicMock(name="vae")
    mm.load_checkpoint.return_value = result
    return mm


def _run_with_mocks(
    tmp_path: Path,
    *,
    prompt: str = "a golden retriever puppy on a summer beach",
    **run_kwargs: Any,
) -> tuple[list[Any], dict[str, Any]]:
    """Run the pipeline with all heavy dependencies mocked.

    Returns ``(result, captured)`` where ``captured`` holds the args passed
    to the key sampling functions.
    """
    from comfy_diffusion.pipelines.image.sdxl import turbo as pipeline_mod

    fake_latent = MagicMock(name="latent")
    fake_sigmas = MagicMock(name="sigmas")
    fake_sampler = MagicMock(name="sampler")
    fake_latent_out = MagicMock(name="latent_out")
    fake_image = MagicMock(name="image")
    mm = _build_mock_mm()

    captured: dict[str, Any] = {}

    def _fake_get_sampler(name: str) -> Any:
        captured["get_sampler_name"] = name
        return fake_sampler

    def _fake_sd_turbo_scheduler(model: Any, steps: int, denoise: float) -> Any:
        captured["sd_turbo_steps"] = steps
        captured["sd_turbo_denoise"] = denoise
        return fake_sigmas

    def _fake_sample_custom_simple(*args: Any, **kwargs: Any) -> Any:
        captured["sample_custom_simple_args"] = args
        captured["sample_custom_simple_kwargs"] = kwargs
        return fake_latent_out

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=fake_latent),
        patch(_GET_SAMPLER_PATCH, side_effect=_fake_get_sampler),
        patch(_SD_TURBO_SCHEDULER_PATCH, side_effect=_fake_sd_turbo_scheduler),
        patch(_SAMPLE_CUSTOM_SIMPLE_PATCH, side_effect=_fake_sample_custom_simple),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            prompt=prompt,
            **run_kwargs,
        )

    return result, captured


# ---------------------------------------------------------------------------
# run() behaviour tests (AC03–AC06)
# ---------------------------------------------------------------------------


def test_run_returns_list_of_images(tmp_path: Path) -> None:
    result, _ = _run_with_mocks(tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_run_returns_pil_image(tmp_path: Path) -> None:
    fake_image = MagicMock(name="image")
    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SD_TURBO_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_SIMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        from comfy_diffusion.pipelines.image.sdxl import turbo as pipeline_mod
        result = pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert result[0] is fake_image


def test_run_uses_euler_ancestral_sampler(tmp_path: Path) -> None:
    """get_sampler must be called with 'euler_ancestral'."""
    _, captured = _run_with_mocks(tmp_path)
    assert captured.get("get_sampler_name") == "euler_ancestral", (
        f"Expected sampler 'euler_ancestral', got {captured.get('get_sampler_name')!r}"
    )


def test_run_uses_sd_turbo_scheduler_with_steps(tmp_path: Path) -> None:
    """sd_turbo_scheduler must be called with the steps parameter."""
    _, captured = _run_with_mocks(tmp_path, steps=2)
    assert captured.get("sd_turbo_steps") == 2


def test_run_sd_turbo_scheduler_denoise_is_one(tmp_path: Path) -> None:
    """sd_turbo_scheduler must always use denoise=1.0."""
    _, captured = _run_with_mocks(tmp_path)
    assert captured.get("sd_turbo_denoise") == 1.0


def test_run_calls_sample_custom_simple_with_add_noise(tmp_path: Path) -> None:
    """sample_custom_simple must be called with add_noise=True (second positional arg)."""
    _, captured = _run_with_mocks(tmp_path)
    args = captured.get("sample_custom_simple_args", ())
    # add_noise is the second positional argument (index 1) in the new signature
    assert len(args) >= 2 and args[1] is True, (
        f"Expected add_noise=True as second positional arg, got args={args!r}"
    )


def test_run_loads_single_checkpoint(tmp_path: Path) -> None:
    """Exactly one load_checkpoint call for the turbo checkpoint."""
    mm = _build_mock_mm()
    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SD_TURBO_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_SIMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import turbo as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert mm.load_checkpoint.call_count == 1


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.sdxl import turbo as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


# ---------------------------------------------------------------------------
# download_models idempotent — 1 entry (AC02)
# ---------------------------------------------------------------------------


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
