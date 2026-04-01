"""Tests for comfy_diffusion/pipelines/image/sdxl/t2i.py — SDXL base + refiner T2I pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring
  - Exports manifest and run; no top-level comfy imports
  - __all__ = ["manifest", "run"]
  - manifest() returns exactly 2 HFModelEntry items with correct dest paths
  - run() signature: models_dir, prompt, negative_prompt, width, height, steps,
    base_end_step, cfg, seed
  - run() behaviour: two-pass KSamplerAdvanced, refiner VAE decode, correct call order
  - download_models idempotent with 2 entries
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
_PIPELINE_FILE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "sdxl" / "t2i.py"

# ---------------------------------------------------------------------------
# Patch targets (source module paths for lazy imports)
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
    assert _PIPELINE_FILE.is_file(), "t2i.py must exist"


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
    assert docstring, "t2i.py must have a module-level docstring"


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
    from comfy_diffusion.pipelines.image.sdxl.t2i import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.sdxl import t2i

    assert hasattr(t2i, "__all__")
    assert set(t2i.__all__) == {"manifest", "run"}


# ---------------------------------------------------------------------------
# Manifest checks — exactly 2 HFModelEntry items
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_two_entries() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 2, f"manifest() must return exactly 2 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.sdxl.t2i import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_base_dest_path() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("checkpoints" in d and "sd_xl_base_1.0" in d for d in dests), (
        "manifest() must include a checkpoints/sd_xl_base_1.0 entry"
    )


def test_manifest_refiner_dest_path() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("checkpoints" in d and "sd_xl_refiner_1.0" in d for d in dests), (
        "manifest() must include a checkpoints/sd_xl_refiner_1.0 entry"
    )


def test_manifest_base_uses_stabilityai_repo() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i import manifest

    base_entry = next(e for e in manifest() if "base" in str(e.dest))
    assert "stabilityai" in base_entry.repo_id


def test_manifest_refiner_uses_stabilityai_repo() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i import manifest

    refiner_entry = next(e for e in manifest() if "refiner" in str(e.dest))
    assert "stabilityai" in refiner_entry.repo_id


# ---------------------------------------------------------------------------
# run() signature checks
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i import run

    sig = inspect.signature(run)
    required = {"models_dir", "prompt", "negative_prompt", "width", "height",
                "steps", "base_end_step", "cfg", "seed"}
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_default_width_height() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 1024
    assert sig.parameters["height"].default == 1024


def test_run_default_steps_and_base_end_step() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 25
    assert sig.parameters["base_end_step"].default == 20


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i import run

    sig = inspect.signature(run)
    for param in ("base_filename", "refiner_filename"):
        assert param in sig.parameters, f"run() must accept '{param}'"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()

    base_result = MagicMock()
    base_result.model = MagicMock(name="base_model")
    base_result.clip = MagicMock(name="base_clip")
    base_result.vae = MagicMock(name="base_vae")
    mm.load_checkpoint.side_effect = [base_result, MagicMock(
        model=MagicMock(name="refiner_model"),
        clip=MagicMock(name="refiner_clip"),
        vae=MagicMock(name="refiner_vae"),
    )]
    return mm


def _run_with_mocks(
    tmp_path: Path,
    *,
    call_order: list[str] | None = None,
    prompt: str = "a majestic eagle soaring over snow-capped mountains",
    **run_kwargs: Any,
) -> list[Any]:
    from comfy_diffusion.pipelines.image.sdxl import t2i as pipeline_mod

    fake_latent = MagicMock(name="latent")
    fake_latent_pass1 = MagicMock(name="latent_pass1")
    fake_latent_pass2 = MagicMock(name="latent_pass2")
    fake_image = MagicMock(name="image")
    mm = _build_mock_mm()

    def _track(name: str, rv: Any) -> Any:
        if call_order is not None:
            call_order.append(name)
        return rv

    sample_call_count: list[int] = [0]

    def _fake_sample_advanced(*args: Any, **kwargs: Any) -> Any:
        sample_call_count[0] += 1
        n = sample_call_count[0]
        name = f"sample_advanced_pass{n}"
        if call_order is not None:
            call_order.append(name)
        return fake_latent_pass1 if n == 1 else fake_latent_pass2

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, side_effect=lambda w, h, **kw: (
            _track("empty_latent_image", fake_latent)
        )),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=_fake_sample_advanced),
        patch(_VAE_DECODE_PATCH, side_effect=lambda vae, latent: (
            _track("vae_decode", fake_image)
        )),
    ):
        return pipeline_mod.run(
            models_dir=tmp_path,
            prompt=prompt,
            **run_kwargs,
        )


# ---------------------------------------------------------------------------
# run() behaviour tests
# ---------------------------------------------------------------------------


def test_run_returns_list_of_images(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_run_returns_pil_image(tmp_path: Path) -> None:
    fake_image = MagicMock(name="image")
    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        from comfy_diffusion.pipelines.image.sdxl import t2i as pipeline_mod
        result = pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert result[0] is fake_image


def test_run_calls_sample_advanced_twice(tmp_path: Path) -> None:
    """Two calls to sample_advanced: one for base, one for refiner."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    sample_calls = [c for c in call_order if "sample_advanced" in c]
    assert len(sample_calls) == 2


def test_run_pass1_uses_add_noise_and_leftover_noise(tmp_path: Path) -> None:
    """Pass 1 must set add_noise=True, return_with_leftover_noise=True."""
    captured_calls: list[dict[str, Any]] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        captured_calls.append(kwargs)
        return MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=_capture),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    pass1_kwargs = captured_calls[0]
    assert pass1_kwargs.get("add_noise") is True
    assert pass1_kwargs.get("return_with_leftover_noise") is True
    assert pass1_kwargs.get("start_at_step") == 0


def test_run_pass1_end_at_step_equals_base_end_step(tmp_path: Path) -> None:
    """Pass 1 end_at_step must equal base_end_step."""
    captured_calls: list[dict[str, Any]] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        captured_calls.append(kwargs)
        return MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=_capture),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test", base_end_step=15)

    assert captured_calls[0].get("end_at_step") == 15


def test_run_pass2_no_noise_full_denoise(tmp_path: Path) -> None:
    """Pass 2 must set add_noise=False, return_with_leftover_noise=False, end_at_step=10000."""
    captured_calls: list[dict[str, Any]] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        captured_calls.append(kwargs)
        return MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=_capture),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    pass2_kwargs = captured_calls[1]
    assert pass2_kwargs.get("add_noise") is False
    assert pass2_kwargs.get("return_with_leftover_noise") is False
    assert pass2_kwargs.get("end_at_step") == 10000


def test_run_pass2_start_at_step_equals_base_end_step(tmp_path: Path) -> None:
    """Pass 2 start_at_step must equal base_end_step."""
    captured_calls: list[dict[str, Any]] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        captured_calls.append(kwargs)
        return MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=_capture),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test", base_end_step=18)

    assert captured_calls[1].get("start_at_step") == 18


def test_run_uses_refiner_vae_for_decode(tmp_path: Path) -> None:
    """vae_decode must be called with the refiner VAE, not base VAE."""
    base_vae = MagicMock(name="base_vae")
    refiner_vae = MagicMock(name="refiner_vae")

    base_result = MagicMock(model=MagicMock(), clip=MagicMock(), vae=base_vae)
    refiner_result = MagicMock(model=MagicMock(), clip=MagicMock(), vae=refiner_vae)

    mm = MagicMock()
    mm.load_checkpoint.side_effect = [base_result, refiner_result]

    decode_vae_arg: list[Any] = []

    def _capture_decode(vae: Any, latent: Any) -> MagicMock:
        decode_vae_arg.append(vae)
        return MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, side_effect=_capture_decode),
    ):
        from comfy_diffusion.pipelines.image.sdxl import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert decode_vae_arg[0] is refiner_vae, (
        "vae_decode must use the refiner VAE, not the base VAE"
    )


def test_run_loads_two_checkpoints(tmp_path: Path) -> None:
    """Two load_checkpoint calls: base then refiner."""
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.sdxl import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert mm.load_checkpoint.call_count == 2


def test_full_pipeline_call_order(tmp_path: Path) -> None:
    """Pipeline must execute in the correct order."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)

    expected = [
        "empty_latent_image",
        "sample_advanced_pass1",
        "sample_advanced_pass2",
        "vae_decode",
    ]
    assert call_order == expected, (
        f"Expected call order {expected}, got {call_order}"
    )


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.sdxl import t2i as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


# ---------------------------------------------------------------------------
# download_models idempotent — 2 entries
# ---------------------------------------------------------------------------


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.sdxl.t2i import manifest

    entries = manifest()
    assert len(entries) == 2

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
