"""Tests for comfy_diffusion/pipelines/image/z_image/turbo.py — Z-Image Turbo pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring                      (AC01, AC09)
  - __all__ = ["manifest", "run"]                                                       (AC09)
  - No top-level comfy/torch imports (lazy import pattern)                              (AC09)
  - manifest() returns exactly 3 HFModelEntry items with correct dest paths             (AC02)
  - manifest() uses Comfy-Org/z_image_turbo HF repo                                    (AC02)
  - run() signature: models_dir, prompt, width, height, steps, seed                    (AC03)
  - run() returns list[PIL.Image.Image]                                                 (AC03)
  - run() calls load_unet() for diffusion model                                         (AC04)
  - run() calls load_clip() with clip_type="lumina2"                                   (AC04)
  - run() calls load_vae()                                                              (AC04)
  - run() applies model_sampling_aura_flow(model, shift=3)                             (AC05)
  - run() creates latent via empty_sd3_latent_image()                                  (AC06)
  - run() uses conditioning_zero_out() for negative conditioning                        (AC07)
  - run() uses sampler "res_multistep", scheduler "simple", cfg=1.0                    (AC08)
  - empty_sd3_latent_image exists in latent module and is exported                     (AC06, FR-3)
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
_PIPELINE_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "z_image" / "turbo.py"
)
_LATENT_FILE = _REPO_ROOT / "comfy_diffusion" / "latent.py"

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_ZERO_OUT_PATCH = "comfy_diffusion.conditioning.conditioning_zero_out"
_EMPTY_SD3_PATCH = "comfy_diffusion.latent.empty_sd3_latent_image"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"
_AURA_FLOW_PATCH = "comfy_diffusion.models.model_sampling_aura_flow"


# ---------------------------------------------------------------------------
# File-level checks (AC01, AC09)
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "turbo.py must exist under pipelines/image/z_image/"


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


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.z_image import turbo

    assert hasattr(turbo, "__all__")
    assert set(turbo.__all__) == {"manifest", "run"}


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
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks (AC02)
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
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry must be an HFModelEntry, got {type(entry)!r}"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "z_image_turbo_bf16" in d for d in dests), (
        f"No unet dest matching diffusion_models/z_image_turbo_bf16.safetensors in {dests}"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "qwen_3_4b" in d for d in dests), (
        f"No text_encoder dest matching text_encoders/qwen_3_4b.safetensors in {dests}"
    )


def test_manifest_vae_dest_path() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("vae" in d and "ae.safetensors" in d for d in dests), (
        f"No vae dest matching vae/ae.safetensors in {dests}"
    )


def test_manifest_all_from_comfy_org_z_image_turbo_repo() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import manifest

    for entry in manifest():
        assert "Comfy-Org" in entry.repo_id, (
            f"Expected Comfy-Org repo, got {entry.repo_id!r}"
        )
        assert "z_image_turbo" in entry.repo_id, (
            f"Expected z_image_turbo repo, got {entry.repo_id!r}"
        )


# ---------------------------------------------------------------------------
# run() signature checks (AC03)
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import run

    sig = inspect.signature(run)
    required = {"models_dir", "prompt", "width", "height", "steps", "seed"}
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_default_width_is_1024() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 1024


def test_run_default_height_is_1024() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["height"].default == 1024


def test_run_default_steps_is_four() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 4


def test_run_default_seed_is_zero() -> None:
    from comfy_diffusion.pipelines.image.z_image.turbo import run

    sig = inspect.signature(run)
    assert sig.parameters["seed"].default == 0


# ---------------------------------------------------------------------------
# empty_sd3_latent_image exists in latent module (AC06 / FR-3)
# ---------------------------------------------------------------------------


def test_empty_sd3_latent_image_exists_in_latent_module() -> None:
    from comfy_diffusion import latent as latent_module

    assert hasattr(latent_module, "empty_sd3_latent_image"), (
        "empty_sd3_latent_image must be defined in comfy_diffusion.latent"
    )
    assert callable(latent_module.empty_sd3_latent_image)


def test_empty_sd3_latent_image_in_dunder_all() -> None:
    from comfy_diffusion import latent as latent_module

    assert "empty_sd3_latent_image" in latent_module.__all__


def test_empty_sd3_latent_image_signature() -> None:
    from comfy_diffusion.latent import empty_sd3_latent_image

    sig = inspect.signature(empty_sd3_latent_image)
    assert "width" in sig.parameters
    assert "height" in sig.parameters
    assert "batch_size" in sig.parameters
    assert sig.parameters["batch_size"].default == 1


# ---------------------------------------------------------------------------
# Helper for run() behaviour tests
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
    prompt: str = "a cinematic close-up portrait",
    **run_kwargs: Any,
) -> tuple[list[Any], dict[str, Any]]:
    """Run the pipeline with all heavy dependencies mocked."""
    from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod

    fake_model = MagicMock(name="model")
    fake_patched_model = MagicMock(name="patched_model")
    fake_clip = MagicMock(name="clip")
    fake_vae = MagicMock(name="vae")
    fake_positive = MagicMock(name="positive")
    fake_negative = MagicMock(name="negative")
    fake_latent = MagicMock(name="latent")
    fake_latent_out = MagicMock(name="latent_out")
    fake_image = MagicMock(name="image")
    mm = _build_mock_mm()
    mm.load_unet.return_value = fake_model
    mm.load_clip.return_value = fake_clip
    mm.load_vae.return_value = fake_vae

    captured: dict[str, Any] = {}

    def _fake_aura_flow(model: Any, shift: float) -> Any:
        captured["aura_flow_model"] = model
        captured["aura_flow_shift"] = shift
        return fake_patched_model

    def _fake_encode_prompt(clip: Any, text: str, negative_text: str) -> Any:
        captured["encode_clip"] = clip
        captured["encode_text"] = text
        return (fake_positive, MagicMock())

    def _fake_zero_out(cond: Any) -> Any:
        captured["zero_out_cond"] = cond
        return fake_negative

    def _fake_empty_sd3(width: int, height: int, batch_size: int = 1) -> Any:
        captured["sd3_width"] = width
        captured["sd3_height"] = height
        captured["sd3_batch_size"] = batch_size
        return fake_latent

    def _fake_sample(*args: Any, **kwargs: Any) -> Any:
        captured["sample_args"] = args
        captured["sample_kwargs"] = kwargs
        return fake_latent_out

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_AURA_FLOW_PATCH, side_effect=_fake_aura_flow),
        patch(_ENCODE_PATCH, side_effect=_fake_encode_prompt),
        patch(_ZERO_OUT_PATCH, side_effect=_fake_zero_out),
        patch(_EMPTY_SD3_PATCH, side_effect=_fake_empty_sd3),
        patch(_SAMPLE_PATCH, side_effect=_fake_sample),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            prompt=prompt,
            **run_kwargs,
        )

    return result, captured


# ---------------------------------------------------------------------------
# run() behaviour tests (AC03–AC08)
# ---------------------------------------------------------------------------


def test_run_returns_list_of_images(tmp_path: Path) -> None:
    result, _ = _run_with_mocks(tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_run_returns_correct_image(tmp_path: Path) -> None:
    fake_image = MagicMock(name="image")
    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=_build_mock_mm()),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=fake_image),
    ):
        from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod

        result = pipeline_mod.run(models_dir=tmp_path, prompt="test")

    assert result[0] is fake_image


def test_run_calls_load_unet(tmp_path: Path) -> None:
    """load_unet() must be called for the diffusion model (AC04)."""
    mm = _build_mock_mm()
    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod

        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_unet.assert_called_once()
    assert "z_image_turbo_bf16" in str(mm.load_unet.call_args)


def test_run_calls_load_clip_with_lumina2(tmp_path: Path) -> None:
    """load_clip() must be called with clip_type='lumina2' (AC04)."""
    mm = _build_mock_mm()
    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod

        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_clip.assert_called_once()
    _, kwargs = mm.load_clip.call_args
    assert kwargs.get("clip_type") == "lumina2", (
        f"load_clip must use clip_type='lumina2', got {kwargs.get('clip_type')!r}"
    )


def test_run_calls_load_vae(tmp_path: Path) -> None:
    """load_vae() must be called for the VAE (AC04)."""
    mm = _build_mock_mm()
    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_AURA_FLOW_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod

        pipeline_mod.run(models_dir=tmp_path, prompt="test")

    mm.load_vae.assert_called_once()
    assert "ae.safetensors" in str(mm.load_vae.call_args)


def test_run_applies_model_sampling_aura_flow_shift_3(tmp_path: Path) -> None:
    """model_sampling_aura_flow must be called with shift=3 (AC05)."""
    _, captured = _run_with_mocks(tmp_path)
    assert captured.get("aura_flow_shift") == 3, (
        f"Expected shift=3 for model_sampling_aura_flow, got {captured.get('aura_flow_shift')!r}"
    )


def test_run_uses_empty_sd3_latent_image(tmp_path: Path) -> None:
    """empty_sd3_latent_image must be called (AC06)."""
    _, captured = _run_with_mocks(tmp_path)
    assert "sd3_width" in captured, "empty_sd3_latent_image was not called"


def test_run_passes_correct_dimensions_to_empty_sd3(tmp_path: Path) -> None:
    """empty_sd3_latent_image must receive width and height from run() (AC06)."""
    _, captured = _run_with_mocks(tmp_path, width=512, height=768)
    assert captured.get("sd3_width") == 512
    assert captured.get("sd3_height") == 768


def test_run_uses_conditioning_zero_out_for_negative(tmp_path: Path) -> None:
    """conditioning_zero_out must be called with the positive conditioning (AC07)."""
    _, captured = _run_with_mocks(tmp_path)
    assert "zero_out_cond" in captured, "conditioning_zero_out was not called"


def test_run_uses_res_multistep_sampler(tmp_path: Path) -> None:
    """sample() must be called with sampler_name='res_multistep' (AC08)."""
    _, captured = _run_with_mocks(tmp_path)
    args = captured.get("sample_args", ())
    # sample(model, positive, negative, latent, steps, cfg, sampler_name, scheduler, seed)
    assert len(args) >= 7, f"sample() received fewer args than expected: {args}"
    sampler_name = args[6]
    assert sampler_name == "res_multistep", (
        f"Expected sampler 'res_multistep', got {sampler_name!r}"
    )


def test_run_uses_simple_scheduler(tmp_path: Path) -> None:
    """sample() must be called with scheduler='simple' (AC08)."""
    _, captured = _run_with_mocks(tmp_path)
    args = captured.get("sample_args", ())
    scheduler = args[7]
    assert scheduler == "simple", (
        f"Expected scheduler 'simple', got {scheduler!r}"
    )


def test_run_uses_cfg_one(tmp_path: Path) -> None:
    """sample() must be called with cfg=1.0 (AC08)."""
    _, captured = _run_with_mocks(tmp_path)
    args = captured.get("sample_args", ())
    cfg = args[5]
    assert cfg == 1.0, (
        f"Expected cfg=1.0 for turbo pipeline, got {cfg!r}"
    )


def test_run_passes_steps_to_sample(tmp_path: Path) -> None:
    """sample() must receive the steps argument from run() (AC08)."""
    _, captured = _run_with_mocks(tmp_path, steps=8)
    args = captured.get("sample_args", ())
    steps_arg = args[4]
    assert steps_arg == 8, f"Expected steps=8, got {steps_arg!r}"


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.z_image import turbo as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(models_dir=tmp_path, prompt="test")


# ---------------------------------------------------------------------------
# download_models idempotent with 3 entries (AC02)
# ---------------------------------------------------------------------------


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
