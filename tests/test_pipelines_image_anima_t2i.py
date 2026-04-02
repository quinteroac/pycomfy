"""Tests for comfy_diffusion/pipelines/image/anima/t2i.py — Anima Preview T2I pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring
  - Exports manifest and run; no top-level comfy/torch imports
  - __all__ = ["manifest", "run"]
  - manifest() returns exactly 3 HFModelEntry items with correct dest paths
  - run() signature: models_dir, prompt, negative_prompt, width, height, steps, cfg, seed
  - run() defaults: sampler er_sde, cfg 4.0, steps 30
  - run() loads unet, clip (stable_diffusion type), vae; encodes prompt; creates latent; samples; decodes
  - download_models idempotent with 3 entries
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
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "anima" / "t2i.py"
)

# ---------------------------------------------------------------------------
# Patch targets (source module paths for lazy imports inside run())
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.empty_latent_image"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"


# ---------------------------------------------------------------------------
# File-level checks (AC07, AC08)
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
    from comfy_diffusion.pipelines.image.anima.t2i import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.anima import t2i

    assert hasattr(t2i, "__all__")
    assert set(t2i.__all__) == {"manifest", "run"}


# ---------------------------------------------------------------------------
# Manifest checks — exactly 3 HFModelEntry items (AC01, AC02)
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_three_entries() -> None:
    from comfy_diffusion.pipelines.image.anima.t2i import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 3, f"manifest() must return exactly 3 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.anima.t2i import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_diffusion_model_dest() -> None:
    from comfy_diffusion.pipelines.image.anima.t2i import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any("diffusion_models" in d and "anima-preview2" in d for d in dests), (
        "manifest() must include a diffusion_models/anima-preview2.safetensors entry"
    )


def test_manifest_text_encoder_dest() -> None:
    from comfy_diffusion.pipelines.image.anima.t2i import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any("text_encoders" in d and "qwen_3_06b_base" in d for d in dests), (
        "manifest() must include a text_encoders/qwen_3_06b_base.safetensors entry"
    )


def test_manifest_vae_dest() -> None:
    from comfy_diffusion.pipelines.image.anima.t2i import manifest

    dests = [str(entry.dest) for entry in manifest()]
    assert any("vae" in d and "qwen_image_vae" in d for d in dests), (
        "manifest() must include a vae/qwen_image_vae.safetensors entry"
    )


def test_manifest_uses_split_files_hf_paths() -> None:
    from comfy_diffusion.pipelines.image.anima.t2i import manifest

    filenames = {str(entry.filename) for entry in manifest()}
    assert filenames == {
        "split_files/diffusion_models/anima-preview2.safetensors",
        "split_files/text_encoders/qwen_3_06b_base.safetensors",
        "split_files/vae/qwen_image_vae.safetensors",
    }


# ---------------------------------------------------------------------------
# run() signature checks (AC03)
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.image.anima.t2i import run

    sig = inspect.signature(run)
    required = {"models_dir", "prompt", "negative_prompt", "width", "height", "steps", "cfg", "seed"}
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_default_width_and_height() -> None:
    from comfy_diffusion.pipelines.image.anima.t2i import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 1024
    assert sig.parameters["height"].default == 1024


def test_run_default_cfg_is_4() -> None:
    """Workflow default CFG is 4.0 (AC06)."""
    from comfy_diffusion.pipelines.image.anima.t2i import run

    sig = inspect.signature(run)
    assert sig.parameters["cfg"].default == 4.0


def test_run_default_steps_is_30() -> None:
    """Workflow default steps is 30 (AC06)."""
    from comfy_diffusion.pipelines.image.anima.t2i import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 30


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    return mm


def _run_with_mocks(
    tmp_path: Path,
    *,
    prompt: str = "1girl, anime style",
    **run_kwargs: Any,
) -> list[Any]:
    from comfy_diffusion.pipelines.image.anima import t2i as pipeline_mod

    fake_latent = MagicMock(name="latent")
    fake_latent_out = MagicMock(name="latent_out")
    fake_image = MagicMock(name="image")
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=fake_latent),
        patch(_SAMPLE_PATCH, return_value=fake_latent_out),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        return pipeline_mod.run(models_dir=tmp_path, prompt=prompt, **run_kwargs)


# ---------------------------------------------------------------------------
# run() behaviour tests (AC03, AC04, AC05, AC06)
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
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        from comfy_diffusion.pipelines.image.anima import t2i as pipeline_mod
        result = pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert result[0] is fake_image


def test_run_calls_load_unet(tmp_path: Path) -> None:
    """load_unet must be called exactly once (AC04)."""
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.anima import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_unet.assert_called_once()


def test_run_calls_load_clip_with_stable_diffusion_type(tmp_path: Path) -> None:
    """load_clip must be called with clip_type='stable_diffusion' (AC04)."""
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.anima import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_clip.assert_called_once()
    _, kwargs = mm.load_clip.call_args
    assert kwargs.get("clip_type") == "stable_diffusion", (
        "load_clip must be called with clip_type='stable_diffusion'"
    )


def test_run_calls_load_vae(tmp_path: Path) -> None:
    """load_vae must be called exactly once (AC04)."""
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.anima import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_vae.assert_called_once()


def test_run_calls_empty_latent_image_with_batch_size_1(tmp_path: Path) -> None:
    """empty_latent_image must be called with batch_size=1 (AC05)."""
    captured: list[dict[str, Any]] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        captured.append({"args": args, "kwargs": kwargs})
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, side_effect=_capture),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.anima import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test", width=512, height=768)

    assert len(captured) == 1
    call_info = captured[0]
    args = call_info["args"]
    kwargs = call_info["kwargs"]
    # width and height passed as positional or keyword
    all_args = list(args) + [kwargs.get("width"), kwargs.get("height")]
    assert 512 in all_args or args[0] == 512, "width must be passed to empty_latent_image"
    assert kwargs.get("batch_size", 1) == 1, "batch_size must be 1"


def test_run_uses_er_sde_sampler(tmp_path: Path) -> None:
    """sample() must be called with sampler_name='er_sde' (AC06)."""
    captured_args: list[tuple[Any, ...]] = []
    captured_kwargs: list[dict[str, Any]] = []

    def _capture(*args: Any, **kwargs: Any) -> MagicMock:
        captured_args.append(args)
        captured_kwargs.append(kwargs)
        return MagicMock()

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, side_effect=_capture),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.anima import t2i as pipeline_mod
        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert len(captured_args) == 1
    # sampler_name is the 7th positional arg: (model, positive, negative, latent, steps, cfg, sampler_name, scheduler, seed)
    sampler_name = captured_args[0][6]
    assert sampler_name == "er_sde", (
        f"run() must use sampler 'er_sde', got '{sampler_name}'"
    )


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.anima import t2i as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


# ---------------------------------------------------------------------------
# download_models idempotent — 3 entries
# ---------------------------------------------------------------------------


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.anima.t2i import manifest

    entries = manifest()
    assert len(entries) == 3

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
