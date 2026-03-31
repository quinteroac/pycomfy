"""Tests for comfy_diffusion/pipelines/video/wan/wan21/flf2v.py — WAN 2.1 FLF2V pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring
  - Exports manifest and run; no top-level comfy imports
  - __all__ = ["manifest", "run"]
  - manifest() returns exactly 4 HFModelEntry items with correct dest paths
  - All manifest entries use the Comfy-Org/Wan_2.1_ComfyUI_repackaged HF repo
  - run() signature: models_dir, start_image, end_image, prompt, negative_prompt,
    width, height, length, fps, steps, cfg, seed, filename overrides
  - run() default width=720, height=1280
  - run() behaviour: model loading, encode_prompt, encode_clip_vision (x2),
    wan_first_last_frame_to_video, model_sampling_sd3, sample, vae_decode_batch
  - start_image and end_image accept str | Path | PIL.Image.Image
  - run() raises RuntimeError when check_runtime returns an error
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
    / "video"
    / "wan"
    / "wan21"
    / "flf2v.py"
)

# ---------------------------------------------------------------------------
# Patch constants (source module paths for lazy imports)
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_MODEL_SAMPLING_SD3_PATCH = "comfy_diffusion.models.model_sampling_sd3"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_ENCODE_CV_PATCH = "comfy_diffusion.conditioning.encode_clip_vision"
_WAN_FLF2V_PATCH = "comfy_diffusion.conditioning.wan_first_last_frame_to_video"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_BATCH_PATCH = "comfy_diffusion.vae.vae_decode_batch"
_LOAD_IMAGE_PATCH = "comfy_diffusion.image.load_image"
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "flf2v.py must exist"


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
    assert docstring, "flf2v.py must have a module-level docstring"


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
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# AC01 — manifest() returns exactly 4 HFModelEntry items
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_four_entries() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 4, f"manifest() must return exactly 4 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "wan2.1_flf2v_720p_14B" in d for d in dests), (
        "manifest() must include a diffusion_models/wan2.1_flf2v_720p_14B entry"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "umt5_xxl" in d for d in dests), (
        "manifest() must include a text_encoders/umt5_xxl entry"
    )


def test_manifest_vae_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("vae" in d and "wan_2.1_vae" in d for d in dests), (
        "manifest() must include a vae/wan_2.1_vae entry"
    )


def test_manifest_clip_vision_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("clip_vision" in d and "clip_vision_h" in d for d in dests), (
        "manifest() must include a clip_vision/clip_vision_h entry"
    )


def test_manifest_all_from_comfy_org_wan_repo() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Comfy-Org/Wan_2.1_ComfyUI_repackaged", (
            f"Entry {entry!r} must use repo_id 'Comfy-Org/Wan_2.1_ComfyUI_repackaged'"
        )


# ---------------------------------------------------------------------------
# AC02 — run() signature checks
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import run

    sig = inspect.signature(run)
    required = {
        "models_dir", "start_image", "end_image", "prompt", "negative_prompt",
        "width", "height", "length",
    }
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_default_width_720() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 720


def test_run_default_height_1280() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import run

    sig = inspect.signature(run)
    assert sig.parameters["height"].default == 1280


def test_run_default_length_33() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import run

    sig = inspect.signature(run)
    assert sig.parameters["length"].default == 33


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import run

    sig = inspect.signature(run)
    for param in ("unet_filename", "text_encoder_filename", "vae_filename", "clip_vision_filename"):
        assert param in sig.parameters, f"run() must accept '{param}'"


def test_run_has_steps_cfg_seed_params() -> None:
    from comfy_diffusion.pipelines.video.wan.wan21.flf2v import run

    sig = inspect.signature(run)
    assert "steps" in sig.parameters
    assert "cfg" in sig.parameters
    assert "seed" in sig.parameters


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_clip_vision.return_value = MagicMock(name="clip_vision")
    return mm


def _run_with_mocks(
    tmp_path: Path,
    *,
    start_image: Any = None,
    end_image: Any = None,
    call_order: list[str] | None = None,
    fake_frames: list[Any] | None = None,
) -> list[Any]:
    from comfy_diffusion.pipelines.video.wan.wan21 import flf2v as pipeline_mod

    if fake_frames is None:
        fake_frames = [MagicMock(name="frame0")]
    if start_image is None:
        start_image = str(tmp_path / "start.png")
    if end_image is None:
        end_image = str(tmp_path / "end.png")

    fake_start_tensor = MagicMock(name="start_tensor")
    fake_end_tensor = MagicMock(name="end_tensor")
    fake_cv_start = MagicMock(name="cv_start")
    fake_cv_end = MagicMock(name="cv_end")
    fake_pos = MagicMock(name="pos")
    fake_neg = MagicMock(name="neg")
    fake_latent: dict[str, Any] = {"samples": MagicMock(name="latent_samples")}
    fake_sampled = MagicMock(name="sampled")
    fake_patched_model = MagicMock(name="patched_model")

    mm = _build_mock_mm()

    cv_call_count: list[int] = [0]

    def _fake_encode_cv(cv: Any, img: Any, **kw: Any) -> Any:
        cv_call_count[0] += 1
        if call_order is not None:
            call_order.append(f"encode_clip_vision_{cv_call_count[0]}")
        return fake_cv_start if cv_call_count[0] == 1 else fake_cv_end

    def _track(name: str, rv: Any) -> Any:
        if call_order is not None:
            call_order.append(name)
        return rv

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_LOAD_IMAGE_PATCH, side_effect=[
            (fake_start_tensor, None),
            (fake_end_tensor, None),
        ]),
        patch(_ENCODE_PATCH, return_value=(fake_pos, fake_neg)),
        patch(_ENCODE_CV_PATCH, side_effect=_fake_encode_cv),
        patch(_WAN_FLF2V_PATCH, side_effect=lambda *a, **kw: (
            _track("wan_first_last_frame_to_video", (fake_pos, fake_neg, fake_latent))
        )),
        patch(_MODEL_SAMPLING_SD3_PATCH, side_effect=lambda m, shift: (
            _track("model_sampling_sd3", fake_patched_model)
        )),
        patch(_SAMPLE_PATCH, side_effect=lambda m, pos, neg, lat, steps, cfg, sn, sc, seed: (
            _track("sample", fake_sampled)
        )),
        patch(_VAE_DECODE_BATCH_PATCH, side_effect=lambda v, s: (
            _track("vae_decode_batch", fake_frames)
        )),
    ):
        return pipeline_mod.run(
            models_dir=tmp_path,
            start_image=start_image,
            end_image=end_image,
            prompt="glass flower blossom",
        )


# ---------------------------------------------------------------------------
# run() behaviour tests
# ---------------------------------------------------------------------------


def test_run_returns_list_of_frames(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path)
    assert isinstance(result, list)


def test_run_frames_value(tmp_path: Path) -> None:
    fake_frames = [MagicMock(name="f0"), MagicMock(name="f1")]
    result = _run_with_mocks(tmp_path, fake_frames=fake_frames)
    assert result is fake_frames


def test_run_calls_encode_clip_vision_twice(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    cv_calls = [c for c in call_order if c.startswith("encode_clip_vision")]
    assert len(cv_calls) == 2, f"encode_clip_vision must be called twice, got: {cv_calls}"


def test_run_calls_wan_first_last_frame_to_video(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "wan_first_last_frame_to_video" in call_order


def test_run_calls_model_sampling_sd3(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "model_sampling_sd3" in call_order


def test_run_calls_sample(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "sample" in call_order


def test_run_calls_vae_decode_batch(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "vae_decode_batch" in call_order


# ---------------------------------------------------------------------------
# AC03 — start_image and end_image accept str | Path | PIL.Image.Image
# ---------------------------------------------------------------------------


def test_run_accepts_start_image_as_string(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path, start_image=str(tmp_path / "start.png"))
    assert isinstance(result, list)


def test_run_accepts_start_image_as_path(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path, start_image=tmp_path / "start.png")
    assert isinstance(result, list)


def test_run_accepts_start_image_as_pil(tmp_path: Path) -> None:
    fake_pil = MagicMock()
    fake_pil.mode = "RGB"
    fake_pil.size = (720, 1280)
    fake_tensor = MagicMock(name="start_tensor_from_pil")

    from comfy_diffusion.pipelines.video.wan.wan21 import flf2v as pipeline_mod

    fake_end_tensor = MagicMock(name="end_tensor")
    fake_pos = MagicMock(name="pos")
    fake_neg = MagicMock(name="neg")
    fake_latent: dict[str, Any] = {"samples": MagicMock()}
    fake_frames = [MagicMock(name="frame0")]
    fake_patched_model = MagicMock(name="patched_model")
    fake_sampled = MagicMock(name="sampled")
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=fake_tensor),
        patch(_LOAD_IMAGE_PATCH, return_value=(fake_end_tensor, None)),
        patch(_ENCODE_PATCH, return_value=(fake_pos, fake_neg)),
        patch(_ENCODE_CV_PATCH, return_value=MagicMock()),
        patch(_WAN_FLF2V_PATCH, return_value=(fake_pos, fake_neg, fake_latent)),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=fake_patched_model),
        patch(_SAMPLE_PATCH, return_value=fake_sampled),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=fake_frames),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            start_image=fake_pil,
            end_image=str(tmp_path / "end.png"),
            prompt="glass flower blossom",
        )
    assert isinstance(result, list)


def test_run_accepts_end_image_as_pil(tmp_path: Path) -> None:
    fake_pil = MagicMock()
    fake_pil.mode = "RGB"
    fake_pil.size = (720, 1280)
    fake_tensor = MagicMock(name="end_tensor_from_pil")

    from comfy_diffusion.pipelines.video.wan.wan21 import flf2v as pipeline_mod

    fake_start_tensor = MagicMock(name="start_tensor")
    fake_pos = MagicMock(name="pos")
    fake_neg = MagicMock(name="neg")
    fake_latent: dict[str, Any] = {"samples": MagicMock()}
    fake_frames = [MagicMock(name="frame0")]
    fake_patched_model = MagicMock(name="patched_model")
    fake_sampled = MagicMock(name="sampled")
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_LOAD_IMAGE_PATCH, return_value=(fake_start_tensor, None)),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=fake_tensor),
        patch(_ENCODE_PATCH, return_value=(fake_pos, fake_neg)),
        patch(_ENCODE_CV_PATCH, return_value=MagicMock()),
        patch(_WAN_FLF2V_PATCH, return_value=(fake_pos, fake_neg, fake_latent)),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=fake_patched_model),
        patch(_SAMPLE_PATCH, return_value=fake_sampled),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=fake_frames),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            start_image=str(tmp_path / "start.png"),
            end_image=fake_pil,
            prompt="glass flower blossom",
        )
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.wan.wan21 import flf2v as pipeline_mod

    with patch(_RUNTIME_PATCH, return_value={"error": "ComfyUI not found", "python_version": "3.12.0"}):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(
                models_dir=tmp_path,
                start_image=str(tmp_path / "start.png"),
                end_image=str(tmp_path / "end.png"),
                prompt="test",
            )
