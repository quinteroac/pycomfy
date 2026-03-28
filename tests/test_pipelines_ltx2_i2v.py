"""Tests for US-002 — comfy_diffusion/pipelines/ltx2_i2v.py pipeline (AV chain).

Covers:
  AC01: manifest() returns 4 HFModelEntry items with correct dest paths
  AC02: run() accepts image (path or PIL) in addition to all t2v parameters
  AC03: Image is preprocessed with ltxv_preprocess() before ltxv_img_to_video_inplace()
  AC04: LoRA (ltx-2-19b-distilled-lora-384) is applied via apply_lora()
  AC05: CPU test passes with mocked inputs; AV chain is executed in correct order
  AC06: typecheck / lint — file parses without syntax errors; no top-level comfy imports
"""

from __future__ import annotations

import ast
import contextlib
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_FILE = _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx2" / "i2v.py"


# ---------------------------------------------------------------------------
# AC06 — file parses without syntax errors; conventions
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), (
        "comfy_diffusion/pipelines/ltx2_i2v.py must exist"
    )


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
    assert docstring, "i2v.py must have a module-level docstring"


def test_pipeline_exports_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
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


# ---------------------------------------------------------------------------
# AC01 — manifest() returns 4 HFModelEntry items
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_manifest_returns_exactly_four_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 4, f"manifest() must return exactly 4 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "ltx-2-19b-dev-fp8" in d for d in dests), (
        "manifest() must include an entry with dest under diffusion_models/ltx-2-19b-dev-fp8"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include a text_encoders/gemma_3_12B_it_fp4_mixed entry"
    )


def test_manifest_lora_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d and "ltx-2-19b-distilled-lora-384" in d for d in dests), (
        "manifest() must include a loras/ltx-2-19b-distilled-lora-384 entry"
    )


def test_manifest_upscaler_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("upscale_models" in d and "ltx-2-spatial-upscaler" in d for d in dests), (
        "manifest() must include an upscale_models/ltx-2-spatial-upscaler entry"
    )


def test_manifest_all_from_lightricks_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Lightricks/LTX-Video", (
            f"Entry {entry!r} must use repo_id 'Lightricks/LTX-Video'"
        )


# ---------------------------------------------------------------------------
# AC02 — run() accepts image and all t2v parameters
# ---------------------------------------------------------------------------


def test_run_signature_includes_image_param() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import run

    sig = inspect.signature(run)
    assert "image" in sig.parameters, "run() must accept an 'image' parameter"


def test_run_signature_includes_all_required_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import run

    sig = inspect.signature(run)
    required = {
        "models_dir",
        "image",
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "length",
        "steps",
        "cfg",
        "seed",
    }
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_has_fps_param_with_default_24() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import run

    sig = inspect.signature(run)
    assert "fps" in sig.parameters
    assert sig.parameters["fps"].default == 24


def test_run_has_lora_and_upscaler_override_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import run

    sig = inspect.signature(run)
    assert "lora_filename" in sig.parameters
    assert "upscaler_filename" in sig.parameters
    assert "lora_strength" in sig.parameters


# ---------------------------------------------------------------------------
# Helper: mock ModelManager
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_ltxv_audio_vae.return_value = MagicMock(name="audio_vae")
    mm.load_ltxav_text_encoder.return_value = MagicMock(name="clip")
    mm.load_latent_upscale_model.return_value = MagicMock(name="upscale_model")
    return mm


# Patch targets (lazy imports inside run())
_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_LTXV_COND_PATCH = "comfy_diffusion.conditioning.ltxv_conditioning"
_CROP_GUIDES_PATCH = "comfy_diffusion.conditioning.ltxv_crop_guides"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_UPSAMPLE_PATCH = "comfy_diffusion.latent.ltxv_latent_upsample"
_SAMPLE_CUSTOM_PATCH = "comfy_diffusion.sampling.sample_custom"
_CFG_GUIDER_PATCH = "comfy_diffusion.sampling.cfg_guider"
_BASIC_SCHEDULER_PATCH = "comfy_diffusion.sampling.basic_scheduler"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_RANDOM_NOISE_PATCH = "comfy_diffusion.sampling.random_noise"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode_batch_tiled"
_APPLY_LORA_PATCH = "comfy_diffusion.lora.apply_lora"
_PREPROCESS_PATCH = "comfy_diffusion.image.ltxv_preprocess"
_IMG_TO_VIDEO_PATCH = "comfy_diffusion.video.ltxv_img_to_video_inplace"
_LOAD_IMAGE_PATCH = "comfy_diffusion.image.load_image"
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"
_EMPTY_AUDIO_PATCH = "comfy_diffusion.audio.ltxv_empty_latent_audio"
_CONCAT_AV_PATCH = "comfy_diffusion.audio.ltxv_concat_av_latent"
_SEPARATE_AV_PATCH = "comfy_diffusion.audio.ltxv_separate_av_latent"
_AUDIO_DECODE_PATCH = "comfy_diffusion.audio.ltxv_audio_vae_decode"


def _run_with_mocks(
    tmp_path: Path,
    image: Any,
    *,
    call_order: list[str] | None = None,
    fake_latent: Any = None,
    fake_img_latent: Any = None,
    fake_frames: list[Any] | None = None,
    fake_audio: Any = None,
) -> dict[str, Any]:
    """Execute pipeline.run() with all heavy dependencies mocked."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v as pipeline_mod

    if fake_latent is None:
        fake_latent = {"samples": MagicMock(name="empty_latent")}
    if fake_img_latent is None:
        fake_img_latent = {"samples": MagicMock(name="img_latent")}
    if fake_frames is None:
        fake_frames = [MagicMock(name="frame0")]
    if fake_audio is None:
        fake_audio = MagicMock(name="audio")

    fake_tensor = MagicMock(name="image_tensor")
    fake_preprocessed = MagicMock(name="preprocessed")
    fake_cropped_latent = MagicMock(name="cropped_latent")
    fake_av_latent = MagicMock(name="av_latent")
    fake_video_out = MagicMock(name="video_out")
    fake_audio_out = MagicMock(name="audio_out")
    fake_video_up = MagicMock(name="video_up")
    patched_model = MagicMock(name="patched_model")
    patched_clip = MagicMock(name="patched_clip")

    mm = _build_mock_mm()

    def _track(name: str, rv: Any) -> Any:
        if call_order is not None:
            call_order.append(name)
        return rv

    patches = [
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, side_effect=lambda m, c, p, sm, sc: (
            _track("apply_lora", (patched_model, patched_clip))
        )),
        patch(_LOAD_IMAGE_PATCH, return_value=(fake_tensor, MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=fake_tensor),
        patch(_PREPROCESS_PATCH, side_effect=lambda img, w, h: (
            _track("ltxv_preprocess", fake_preprocessed)
        )),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (
            _track("ltxv_conditioning", (p, n))
        )),
        patch(_EMPTY_LATENT_PATCH, return_value=fake_latent),
        patch(_IMG_TO_VIDEO_PATCH, side_effect=lambda vae, img, lat, **kw: (
            _track("ltxv_img_to_video_inplace", fake_img_latent)
        )),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (
            _track("ltxv_crop_guides", (p, n, fake_cropped_latent))
        )),
        patch(_EMPTY_AUDIO_PATCH, side_effect=lambda av, **kw: (
            _track("ltxv_empty_latent_audio", MagicMock())
        )),
        patch(_CONCAT_AV_PATCH, side_effect=lambda vl, al: (
            _track("ltxv_concat_av_latent", fake_av_latent)
        )),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, side_effect=lambda n, g, s, sig, lat: (
            _track("sample_custom", (MagicMock(), MagicMock()))
        )),
        patch(_SEPARATE_AV_PATCH, side_effect=lambda d: (
            _track("ltxv_separate_av_latent", (fake_video_out, fake_audio_out))
        )),
        patch(_UPSAMPLE_PATCH, side_effect=lambda s, **kw: (
            _track("ltxv_latent_upsample", fake_video_up)
        )),
        patch(_VAE_DECODE_PATCH, side_effect=lambda v, s: (
            _track("vae_decode_batch_tiled", fake_frames)
        )),
        patch(_AUDIO_DECODE_PATCH, return_value=fake_audio),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return pipeline_mod.run(
            models_dir=tmp_path,
            image=image,
            prompt="a waitress smiles",
        )


# ---------------------------------------------------------------------------
# AC03 — ltxv_preprocess called before ltxv_img_to_video_inplace
# ---------------------------------------------------------------------------


def test_preprocess_called_before_img_to_video_inplace(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, "/fake/image.png", call_order=call_order)
    assert "ltxv_preprocess" in call_order
    assert "ltxv_img_to_video_inplace" in call_order
    preprocess_idx = call_order.index("ltxv_preprocess")
    inject_idx = call_order.index("ltxv_img_to_video_inplace")
    assert preprocess_idx < inject_idx, (
        f"ltxv_preprocess must be called before ltxv_img_to_video_inplace; "
        f"got order: {call_order}"
    )


def test_preprocessed_image_passed_to_img_to_video_inplace(tmp_path: Path) -> None:
    """The output of ltxv_preprocess must be passed to ltxv_img_to_video_inplace."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v as pipeline_mod

    fake_preprocessed = MagicMock(name="preprocessed_image")
    received_image: list[Any] = []

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LOAD_IMAGE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_PREPROCESS_PATCH, return_value=fake_preprocessed),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(
            _IMG_TO_VIDEO_PATCH,
            side_effect=lambda vae, img, lat, **kw: received_image.append(img) or {"samples": MagicMock()},
        ),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, image="/fake/image.png", prompt="test")

    assert len(received_image) == 1
    assert received_image[0] is fake_preprocessed, (
        "ltxv_img_to_video_inplace must receive the output of ltxv_preprocess"
    )


# ---------------------------------------------------------------------------
# AC04 — LoRA applied via apply_lora()
# ---------------------------------------------------------------------------


def test_apply_lora_is_called(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v as pipeline_mod

    mm = _build_mock_mm()
    lora_calls: list[tuple[Any, ...]] = []

    def fake_apply_lora(model: Any, clip: Any, path: Any, sm: float, sc: float) -> tuple:
        lora_calls.append((model, clip, path, sm, sc))
        return MagicMock(name="patched_model"), MagicMock(name="patched_clip")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, side_effect=fake_apply_lora),
        patch(_LOAD_IMAGE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_PREPROCESS_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, image="/fake/image.png", prompt="test")

    assert len(lora_calls) == 1, f"apply_lora must be called exactly once; got {len(lora_calls)}"


def test_apply_lora_uses_lora_dest_path(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v as pipeline_mod

    mm = _build_mock_mm()
    lora_paths: list[Any] = []

    def fake_apply_lora(model: Any, clip: Any, path: Any, sm: float, sc: float) -> tuple:
        lora_paths.append(path)
        return MagicMock(), MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, side_effect=fake_apply_lora),
        patch(_LOAD_IMAGE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_PREPROCESS_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, image="/fake/image.png", prompt="test")

    assert len(lora_paths) == 1
    assert "ltx-2-19b-distilled-lora-384" in str(lora_paths[0]), (
        f"apply_lora path must reference ltx-2-19b-distilled-lora-384; got {lora_paths[0]}"
    )


def test_apply_lora_called_before_sampling(tmp_path: Path) -> None:
    """LoRA must be applied after model load but before sampling."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, "/fake/image.png", call_order=call_order)
    assert "apply_lora" in call_order
    assert "sample_custom" in call_order
    assert call_order.index("apply_lora") < call_order.index("sample_custom"), (
        f"apply_lora must be called before sample_custom; got order: {call_order}"
    )


# ---------------------------------------------------------------------------
# AC02 — image accepts PIL.Image as well as path
# ---------------------------------------------------------------------------


def test_run_accepts_pil_image(tmp_path: Path) -> None:
    """run() must accept a PIL.Image.Image without raising."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v as pipeline_mod

    pil_image = MagicMock()
    pil_image.mode = "RGB"

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock(name="tensor")),
        patch(_PREPROCESS_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[MagicMock()]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        result = pipeline_mod.run(models_dir=tmp_path, image=pil_image, prompt="test")

    assert isinstance(result, dict)


def test_run_accepts_path_image(tmp_path: Path) -> None:
    """run() must accept a Path without raising."""
    result = _run_with_mocks(tmp_path, tmp_path / "image.png")
    assert isinstance(result, dict)


def test_run_accepts_str_image(tmp_path: Path) -> None:
    """run() must accept a str path without raising."""
    result = _run_with_mocks(tmp_path, "/some/path/image.png")
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# AC05 — CPU test: full pipeline call order
# ---------------------------------------------------------------------------


def test_full_pipeline_call_order(tmp_path: Path) -> None:
    """AC05: end-to-end AV pipeline runs in correct order on CPU with mocked inputs."""
    call_order: list[str] = []
    result = _run_with_mocks(tmp_path, "/fake/image.png", call_order=call_order)

    expected = [
        "apply_lora",
        "ltxv_preprocess",
        "ltxv_conditioning",
        "ltxv_img_to_video_inplace",
        "ltxv_crop_guides",
        "ltxv_empty_latent_audio",
        "ltxv_concat_av_latent",
        "sample_custom",
        "ltxv_separate_av_latent",
        "ltxv_latent_upsample",
        "vae_decode_batch_tiled",
    ]
    assert call_order == expected, (
        f"Expected call order {expected}, got {call_order}"
    )
    assert isinstance(result, dict)


def test_run_returns_dict_with_frames_and_audio(tmp_path: Path) -> None:
    fake_frames = [MagicMock(name="f0")]
    fake_audio = MagicMock(name="audio")
    result = _run_with_mocks(tmp_path, "/fake/image.png",
                             fake_frames=fake_frames, fake_audio=fake_audio)
    assert isinstance(result, dict)
    assert result["frames"] is fake_frames
    assert result["audio"] is fake_audio


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    """run() must raise RuntimeError when check_runtime() returns an error."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(
                models_dir=tmp_path, image="/fake/image.png", prompt="test"
            )


def test_run_uses_load_ltxav_text_encoder(tmp_path: Path) -> None:
    """Gemma 3 text encoder must be loaded via load_ltxav_text_encoder, not load_clip."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LOAD_IMAGE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_PREPROCESS_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, image="/fake/image.png", prompt="test")

    mm.load_ltxav_text_encoder.assert_called_once()
    mm.load_clip.assert_not_called()


def test_run_uses_load_latent_upscale_model(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LOAD_IMAGE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_PREPROCESS_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, image="/fake/image.png", prompt="test")

    mm.load_latent_upscale_model.assert_called_once()


def test_run_calls_ltxv_conditioning(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, "/fake/image.png", call_order=call_order)
    assert "ltxv_conditioning" in call_order


def test_run_calls_ltxv_crop_guides(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, "/fake/image.png", call_order=call_order)
    assert "ltxv_crop_guides" in call_order


def test_run_calls_ltxv_separate_av_latent(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, "/fake/image.png", call_order=call_order)
    assert "ltxv_separate_av_latent" in call_order


def test_run_uses_load_ltxv_audio_vae(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import i2v as pipeline_mod

    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LOAD_IMAGE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_PREPROCESS_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_LTXV_COND_PATCH, side_effect=lambda p, n, **kw: (p, n)),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_IMG_TO_VIDEO_PATCH, return_value={"samples": MagicMock()}),
        patch(_CROP_GUIDES_PATCH, side_effect=lambda p, n, lat: (p, n, lat)),
        patch(_EMPTY_AUDIO_PATCH, return_value=MagicMock()),
        patch(_CONCAT_AV_PATCH, return_value=MagicMock()),
        patch(_CFG_GUIDER_PATCH, return_value=MagicMock()),
        patch(_RANDOM_NOISE_PATCH, return_value=MagicMock()),
        patch(_BASIC_SCHEDULER_PATCH, return_value=MagicMock()),
        patch(_GET_SAMPLER_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_SEPARATE_AV_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_UPSAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=[]),
        patch(_AUDIO_DECODE_PATCH, return_value=MagicMock()),
    ):
        pipeline_mod.run(models_dir=tmp_path, image="/fake/image.png", prompt="test")

    mm.load_ltxv_audio_vae.assert_called_once()


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    """download_models(manifest()) completes without error when all 4 files are present."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx2.i2v import manifest

    entries = manifest()
    assert len(entries) == 4

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
