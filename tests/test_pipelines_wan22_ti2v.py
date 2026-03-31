"""Tests for comfy_diffusion/pipelines/video/wan/wan22/ti2v.py — WAN 2.2 TI2V pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring (AC-06)
  - Exports manifest and run; no top-level comfy imports (AC-06)
  - __all__ = ["manifest", "run"]
  - manifest() returns exactly 3 HFModelEntry items (AC-01)
  - manifest() has unet, text_encoder, vae entries (AC-01)
  - run() signature: prompt, negative_prompt, width, height, length,
    *, start_image, models_dir, seed, steps, cfg (AC-02)
  - run() calls wan22_image_to_video_latent, ModelSamplingSD3 (shift=8),
    KSampler (uni_pc, 20 steps, cfg=5), and vae_decode_batch (AC-02)
  - When start_image is None, image_to_tensor is not called (AC-03)
  - wan22 sub-package exports ti2v (AC-04)
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
    _REPO_ROOT
    / "comfy_diffusion"
    / "pipelines"
    / "video"
    / "wan"
    / "wan22"
    / "ti2v.py"
)

# ---------------------------------------------------------------------------
# Patch constants (source module paths for lazy imports inside run())
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_MODEL_SAMPLING_SD3_PATCH = "comfy_diffusion.models.model_sampling_sd3"
_ENCODE_PROMPT_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_WAN22_I2V_LATENT_PATCH = "comfy_diffusion.conditioning.wan22_image_to_video_latent"
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_BATCH_PATCH = "comfy_diffusion.vae.vae_decode_batch"


# ---------------------------------------------------------------------------
# File-level checks (AC-06)
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "ti2v.py must exist under wan22/"


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
    assert docstring, "ti2v.py must have a module-level docstring"


def test_pipeline_all_is_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '__all__ = ["manifest", "run"]' in source


def test_no_top_level_comfy_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), (
                f"Top-level comfy import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks — exactly 3 HFModelEntry items (AC-01)
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_three_entries() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 3, f"manifest() must return exactly 3 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_has_unet_entry() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "ti2v" in d for d in dests), (
        "manifest() must include a diffusion_models/...ti2v... UNet entry"
    )


def test_manifest_has_text_encoder_entry() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "umt5_xxl" in d for d in dests), (
        "manifest() must include a text_encoders/umt5_xxl entry"
    )


def test_manifest_has_vae_entry() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("vae" in d and "wan2.2_vae" in d for d in dests), (
        "manifest() must include a vae/wan2.2_vae entry"
    )


def test_manifest_field_names() -> None:
    """Each entry must have repo_id, filename, and dest fields (AC-01, AC-04)."""
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import manifest

    for entry in manifest():
        assert hasattr(entry, "repo_id"), f"Entry {entry!r} must have repo_id"
        assert hasattr(entry, "filename"), f"Entry {entry!r} must have filename"
        assert hasattr(entry, "dest"), f"Entry {entry!r} must have dest"


# ---------------------------------------------------------------------------
# run() signature checks (AC-02)
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import run

    sig = inspect.signature(run)
    required = {
        "prompt", "negative_prompt", "width", "height", "length",
        "start_image", "models_dir", "seed", "steps", "cfg",
    }
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_models_dir_is_keyword_only() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import run

    sig = inspect.signature(run)
    param = sig.parameters["models_dir"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
        "models_dir must be keyword-only"
    )


def test_run_start_image_is_keyword_only() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import run

    sig = inspect.signature(run)
    assert sig.parameters["start_image"].kind == inspect.Parameter.KEYWORD_ONLY


def test_run_seed_is_keyword_only() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import run

    sig = inspect.signature(run)
    assert sig.parameters["seed"].kind == inspect.Parameter.KEYWORD_ONLY


def test_run_default_steps_20() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 20


def test_run_default_cfg_5() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import run

    sig = inspect.signature(run)
    assert sig.parameters["cfg"].default == 5.0


def test_run_start_image_default_none() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.ti2v import run

    sig = inspect.signature(run)
    assert sig.parameters["start_image"].default is None


# ---------------------------------------------------------------------------
# Helpers
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
    start_image: Any | None = None,
    call_order: list[str] | None = None,
    fake_frames: list[Any] | None = None,
) -> list[Any]:
    from comfy_diffusion.pipelines.video.wan.wan22 import ti2v as pipeline_mod

    if fake_frames is None:
        fake_frames = [MagicMock(name="frame0")]

    fake_patched_model = MagicMock(name="patched_model")
    fake_img_tensor = MagicMock(name="img_tensor")
    fake_latent = MagicMock(name="latent")
    fake_sampled_latent = MagicMock(name="sampled_latent")

    mm = _build_mock_mm()

    def _track(name: str, rv: Any) -> Any:
        if call_order is not None:
            call_order.append(name)
        return rv

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_MODEL_SAMPLING_SD3_PATCH, side_effect=lambda m, shift: (
            _track("model_sampling_sd3", fake_patched_model)
        )),
        patch(_ENCODE_PROMPT_PATCH, side_effect=lambda c, p, n: (
            _track("encode_prompt", (MagicMock(), MagicMock()))
        )),
        patch(_IMAGE_TO_TENSOR_PATCH, side_effect=lambda img: (
            _track("image_to_tensor", fake_img_tensor)
        )),
        patch(_WAN22_I2V_LATENT_PATCH, side_effect=lambda vae, **kw: (
            _track("wan22_image_to_video_latent", fake_latent)
        )),
        patch(_SAMPLE_PATCH, side_effect=lambda m, p, n, l, **kw: (
            _track("sample", fake_sampled_latent)
        )),
        patch(_VAE_DECODE_BATCH_PATCH, side_effect=lambda v, l: (
            _track("vae_decode_batch", fake_frames)
        )),
    ):
        return pipeline_mod.run(
            "test prompt",
            models_dir=tmp_path,
            start_image=start_image,
        )


# ---------------------------------------------------------------------------
# run() behaviour tests (AC-02)
# ---------------------------------------------------------------------------


def test_run_returns_list_of_frames(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path)
    assert isinstance(result, list)


def test_run_frames_value(tmp_path: Path) -> None:
    fake_frames = [MagicMock(name="f0"), MagicMock(name="f1")]
    result = _run_with_mocks(tmp_path, fake_frames=fake_frames)
    assert result is fake_frames


def test_run_calls_model_sampling_sd3(tmp_path: Path) -> None:
    """ModelSamplingSD3 must be called with shift=8 (AC-02)."""
    call_order: list[str] = []

    from comfy_diffusion.pipelines.video.wan.wan22 import ti2v as pipeline_mod

    captured_shifts: list[float] = []
    fake_patched_model = MagicMock(name="patched_model")
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_MODEL_SAMPLING_SD3_PATCH, side_effect=lambda m, shift: (
            captured_shifts.append(shift) or call_order.append("model_sampling_sd3") or fake_patched_model
        )),
        patch(_ENCODE_PROMPT_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN22_I2V_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[MagicMock()]),
    ):
        pipeline_mod.run("test prompt", models_dir=tmp_path)

    assert "model_sampling_sd3" in call_order
    assert captured_shifts == [8.0], (
        f"ModelSamplingSD3 must be called with shift=8.0, got {captured_shifts}"
    )


def test_run_calls_wan22_image_to_video_latent(tmp_path: Path) -> None:
    """wan22_image_to_video_latent must be called (AC-02)."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "wan22_image_to_video_latent" in call_order


def test_run_calls_sample_with_uni_pc(tmp_path: Path) -> None:
    """KSampler must be called with sampler_name='uni_pc' (AC-02)."""
    from comfy_diffusion.pipelines.video.wan.wan22 import ti2v as pipeline_mod

    captured_kwargs: list[dict] = []
    mm = _build_mock_mm()

    def capture_sample(model: Any, pos: Any, neg: Any, lat: Any, **kwargs: Any) -> Any:
        captured_kwargs.append(kwargs)
        return MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PROMPT_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN22_I2V_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_PATCH, side_effect=capture_sample),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[MagicMock()]),
    ):
        pipeline_mod.run("test prompt", models_dir=tmp_path, steps=20, cfg=5.0)

    assert len(captured_kwargs) == 1
    kw = captured_kwargs[0]
    assert kw.get("sampler_name") == "uni_pc", (
        f"KSampler must use sampler_name='uni_pc', got {kw.get('sampler_name')!r}"
    )
    assert kw.get("steps") == 20, f"KSampler must use steps=20, got {kw.get('steps')}"
    assert kw.get("cfg") == 5.0, f"KSampler must use cfg=5.0, got {kw.get('cfg')}"


def test_run_calls_vae_decode_batch(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "vae_decode_batch" in call_order


def test_run_call_order(tmp_path: Path) -> None:
    """model_sampling_sd3 → encode_prompt → wan22_image_to_video_latent → sample → vae_decode_batch."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)

    expected = [
        "model_sampling_sd3",
        "encode_prompt",
        "wan22_image_to_video_latent",
        "sample",
        "vae_decode_batch",
    ]
    for step in expected:
        assert step in call_order, f"run() must call {step}"

    for i in range(len(expected) - 1):
        a, b = expected[i], expected[i + 1]
        assert call_order.index(a) < call_order.index(b), (
            f"{a} must be called before {b}, got order: {call_order}"
        )


# ---------------------------------------------------------------------------
# AC-03: start_image=None → empty latent, no image_to_tensor call
# ---------------------------------------------------------------------------


def test_run_no_start_image_skips_image_to_tensor(tmp_path: Path) -> None:
    """When start_image is None, image_to_tensor must NOT be called (AC-03)."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, start_image=None, call_order=call_order)
    assert "image_to_tensor" not in call_order, (
        "image_to_tensor must NOT be called when start_image is None"
    )


def test_run_no_start_image_passes_none_to_latent(tmp_path: Path) -> None:
    """When start_image is None, wan22_image_to_video_latent gets start_image=None (AC-03)."""
    from comfy_diffusion.pipelines.video.wan.wan22 import ti2v as pipeline_mod

    captured_kw: list[dict] = []
    mm = _build_mock_mm()

    def capture_latent(vae: Any, **kwargs: Any) -> Any:
        captured_kw.append(kwargs)
        return MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PROMPT_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN22_I2V_LATENT_PATCH, side_effect=capture_latent),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[MagicMock()]),
    ):
        pipeline_mod.run("test prompt", models_dir=tmp_path, start_image=None)

    assert len(captured_kw) == 1
    assert captured_kw[0].get("start_image") is None, (
        "wan22_image_to_video_latent must receive start_image=None when no image is given"
    )


def test_run_with_start_image_calls_image_to_tensor(tmp_path: Path) -> None:
    """When start_image is provided, image_to_tensor must be called (AC-03)."""
    call_order: list[str] = []
    fake_image = MagicMock(name="pil_image")
    _run_with_mocks(tmp_path, start_image=fake_image, call_order=call_order)
    assert "image_to_tensor" in call_order, (
        "image_to_tensor must be called when start_image is provided"
    )


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.wan.wan22 import ti2v as pipeline_mod

    with (
        patch(_RUNTIME_PATCH, return_value={"error": "ComfyUI not found", "python_version": "3.12.0"}),
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run("prompt", models_dir=tmp_path)


# ---------------------------------------------------------------------------
# AC-04: wan22 sub-package exports ti2v
# ---------------------------------------------------------------------------


def test_wan22_package_exports_ti2v() -> None:
    from comfy_diffusion.pipelines.video.wan import wan22

    assert "ti2v" in wan22.__all__
