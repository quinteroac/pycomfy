"""Tests for comfy_diffusion/pipelines/video/ltx/ltx2/depth.py — LTX2 Depth-to-Video pipeline.

Covers:
  AC01: manifest() returns exactly 7 HFModelEntry items with correct dest paths
  AC02: run() follows the workflow node execution order end-to-end
  AC03: Pass 1 uses LTXVScheduler + euler; pass 2 uses ManualSigmas(4 values) +
        gradient_estimation; CFG 3/1
  AC04: run() accepts video_path, prompt, negative_prompt, width, height, length,
        fps, cfg_pass1, cfg_pass2, seed, depth_lora_strength, lora_strength,
        and all filename-override parameters
  AC05: Pipeline file is at comfy_diffusion/pipelines/video/ltx/ltx2/depth.py
  AC06: Typecheck / lint — file parses without syntax errors; no top-level comfy imports
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
_PIPELINE_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx2" / "depth.py"
)

# ---------------------------------------------------------------------------
# Patch targets (lazy imports inside run())
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_LTXV_COND_PATCH = "comfy_diffusion.conditioning.ltxv_conditioning"
_CROP_GUIDES_PATCH = "comfy_diffusion.conditioning.ltxv_crop_guides"
_ADD_GUIDE_PATCH = "comfy_diffusion.conditioning.ltxv_add_guide"
_EMPTY_VIDEO_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_EMPTY_AUDIO_PATCH = "comfy_diffusion.audio.ltxv_empty_latent_audio"
_CONCAT_AV_PATCH = "comfy_diffusion.audio.ltxv_concat_av_latent"
_SEPARATE_AV_PATCH = "comfy_diffusion.audio.ltxv_separate_av_latent"
_AUDIO_DECODE_PATCH = "comfy_diffusion.audio.ltxv_audio_vae_decode"
_CFG_GUIDER_PATCH = "comfy_diffusion.sampling.cfg_guider"
_RANDOM_NOISE_PATCH = "comfy_diffusion.sampling.random_noise"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_LTXV_SCHEDULER_PATCH = "comfy_diffusion.sampling.ltxv_scheduler"
_MANUAL_SIGMAS_PATCH = "comfy_diffusion.sampling.manual_sigmas"
_SAMPLE_CUSTOM_PATCH = "comfy_diffusion.sampling.sample_custom"
_UPSAMPLE_PATCH = "comfy_diffusion.latent.ltxv_latent_upsample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode_batch_tiled"
_APPLY_LORA_PATCH = "comfy_diffusion.lora.apply_lora"
_LOAD_VIDEO_PATCH = "comfy_diffusion.video.load_video"
_GET_VIDEO_COMPONENTS_PATCH = "comfy_diffusion.video.get_video_components"
_IMG_TO_VIDEO_PATCH = "comfy_diffusion.video.ltxv_img_to_video_inplace"
_PREPROCESS_PATCH = "comfy_diffusion.image.ltxv_preprocess"
_SCALE_BY_PATCH = "comfy_diffusion.image.image_scale_by"
_IMAGE_FROM_BATCH_PATCH = "comfy_diffusion.image.image_from_batch"
_LOTUS_DEPTH_PATCH = "comfy_diffusion.controlnet.lotus_depth_pass"


# ---------------------------------------------------------------------------
# AC05 / AC06 — file-level checks
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "depth.py must exist at the expected path"


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
    assert docstring, "depth.py must have a module-level docstring"


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


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# AC01 — manifest() returns exactly 7 HFModelEntry items with correct paths
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_seven_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 7, f"manifest() must return exactly 7 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_ckpt_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("checkpoints" in d and "ltx-2-19b-dev-fp8" in d for d in dests), (
        "manifest() must include a checkpoints/ltx-2-19b-dev-fp8 entry"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include a text_encoders/gemma_3_12B_it_fp4_mixed entry"
    )


def test_manifest_depth_lora_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d and "ic-lora-depth-control" in d for d in dests), (
        "manifest() must include a loras/ltx-2-19b-ic-lora-depth-control entry"
    )


def test_manifest_distilled_lora_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d and "distilled-lora-384" in d for d in dests), (
        "manifest() must include a loras/ltx-2-19b-distilled-lora-384 entry"
    )


def test_manifest_upscaler_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("upscale_models" in d and "ltx-2-spatial-upscaler" in d for d in dests), (
        "manifest() must include an upscale_models/ltx-2-spatial-upscaler entry"
    )


def test_manifest_lotus_model_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "lotus-depth" in d for d in dests), (
        "manifest() must include a diffusion_models/lotus-depth entry"
    )


def test_manifest_lotus_vae_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("vae" in d and "vae-ft-mse-840000" in d for d in dests), (
        "manifest() must include a vae/vae-ft-mse-840000 entry"
    )


def test_manifest_depth_lora_uses_correct_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    depth_entries = [e for e in manifest() if "ic-lora-depth-control" in str(e.dest)]
    assert len(depth_entries) == 1, "manifest() must have exactly one depth control LoRA entry"
    assert "Depth-Control" in depth_entries[0].repo_id, (
        "Depth control LoRA must be fetched from the Lightricks Depth-Control HF repo"
    )


def test_manifest_lotus_model_uses_comfy_org_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import manifest

    lotus_entries = [e for e in manifest() if "lotus-depth" in str(e.dest)]
    assert len(lotus_entries) == 1, "manifest() must have exactly one Lotus model entry"
    assert "Comfy-Org" in lotus_entries[0].repo_id or "lotus" in lotus_entries[0].repo_id.lower(), (
        "Lotus model must be fetched from the Comfy-Org/lotus HF repo"
    )


# ---------------------------------------------------------------------------
# AC04 — run() signature
# ---------------------------------------------------------------------------


def test_run_accepts_video_path_param() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import run

    sig = inspect.signature(run)
    assert "video_path" in sig.parameters, "run() must accept a 'video_path' parameter"


def test_run_accepts_prompt_and_negative_prompt() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import run

    sig = inspect.signature(run)
    assert "prompt" in sig.parameters
    assert "negative_prompt" in sig.parameters


def test_run_accepts_resolution_and_length_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import run

    sig = inspect.signature(run)
    for param in ("width", "height", "length", "fps"):
        assert param in sig.parameters, f"run() must accept '{param}'"


def test_run_accepts_cfg_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import run

    sig = inspect.signature(run)
    assert "cfg_pass1" in sig.parameters
    assert "cfg_pass2" in sig.parameters
    assert sig.parameters["cfg_pass1"].default == 3.0
    assert sig.parameters["cfg_pass2"].default == 1.0


def test_run_accepts_lora_strength_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import run

    sig = inspect.signature(run)
    assert "depth_lora_strength" in sig.parameters
    assert "lora_strength" in sig.parameters


def test_run_accepts_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import run

    sig = inspect.signature(run)
    for param in (
        "ckpt_filename",
        "text_encoder_filename",
        "vae_filename",
        "audio_vae_filename",
        "depth_lora_filename",
        "lora_filename",
        "upscaler_filename",
        "lotus_model_filename",
        "lotus_vae_filename",
    ):
        assert param in sig.parameters, f"run() must accept '{param}'"


def test_run_length_default_is_121() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import run

    sig = inspect.signature(run)
    assert sig.parameters["length"].default == 121, (
        "run() default length must be 121 (from workflow PrimitiveInt node)"
    )


def test_run_fps_default_is_24() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.depth import run

    sig = inspect.signature(run)
    assert sig.parameters["fps"].default == 24, (
        "run() default fps must be 24 (from workflow PrimitiveFloat/PrimitiveInt nodes)"
    )


# ---------------------------------------------------------------------------
# Helper: build mocked ModelManager and run pipeline
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_ltxv_audio_vae.return_value = MagicMock(name="audio_vae")
    mm.load_ltxav_text_encoder.return_value = MagicMock(name="clip")
    mm.load_latent_upscale_model.return_value = MagicMock(name="upscale_model")
    return mm


def _run_with_mocks(
    tmp_path: Path,
    *,
    call_order: list[str] | None = None,
    fake_frames: list[Any] | None = None,
    fake_audio: Any = None,
    manual_sigmas_calls: list[str] | None = None,
    ltxv_scheduler_calls: list[tuple[Any, ...]] | None = None,
    cfg_guider_calls: list[float] | None = None,
    sampler_calls: list[str] | None = None,
) -> dict[str, Any]:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import depth as pipeline_mod

    if fake_frames is None:
        fake_frames = [MagicMock(name="frame0")]
    if fake_audio is None:
        fake_audio = MagicMock(name="audio")

    fake_video_latent = MagicMock(name="video_latent")
    fake_audio_latent = MagicMock(name="audio_latent")
    fake_av_latent = MagicMock(name="av_latent")
    fake_output_p1 = MagicMock(name="output_p1")
    fake_denoised_p2 = MagicMock(name="denoised_p2")
    fake_video_out = MagicMock(name="video_out")
    fake_audio_out = MagicMock(name="audio_out")
    fake_video_up = MagicMock(name="video_up")
    fake_frames_tensor = MagicMock(name="frames_tensor")
    fake_frames_full = MagicMock(name="frames_full")
    fake_frames_half = MagicMock(name="frames_half")
    fake_depth_map = MagicMock(name="depth_map")
    fake_first_frame = MagicMock(name="first_frame")
    fake_inplace_latent = MagicMock(name="inplace_latent")
    fake_add_guide_latent = MagicMock(name="add_guide_latent")
    patched_model = MagicMock(name="patched_model")
    patched_clip = MagicMock(name="patched_clip")

    mm = _build_mock_mm()

    def _track(name: str, rv: Any) -> Any:
        if call_order is not None:
            call_order.append(name)
        return rv

    apply_lora_call_count: list[int] = [0]

    def _apply_lora(m: Any, c: Any, p: Any, sm: float, sc: float) -> tuple[Any, Any]:
        apply_lora_call_count[0] += 1
        return _track("apply_lora", (patched_model, patched_clip))

    def _manual_sigmas(s: str) -> Any:
        if manual_sigmas_calls is not None:
            manual_sigmas_calls.append(s)
        return _track("manual_sigmas", MagicMock(name=f"sigmas_{s[:4]}"))

    def _ltxv_scheduler(steps: Any, max_shift: Any, base_shift: Any, **kw: Any) -> Any:
        if ltxv_scheduler_calls is not None:
            ltxv_scheduler_calls.append((steps, max_shift, base_shift))
        return _track("ltxv_scheduler", MagicMock(name="sigmas_ltxv"))

    def _cfg_guider(m: Any, p: Any, n: Any, c: float) -> Any:
        if cfg_guider_calls is not None:
            cfg_guider_calls.append(c)
        return _track("cfg_guider", MagicMock(name=f"guider_cfg{c}"))

    def _get_sampler(name: str) -> Any:
        if sampler_calls is not None:
            sampler_calls.append(name)
        return _track("get_sampler", MagicMock(name=f"sampler_{name}"))

    sample_custom_call_count: list[int] = [0]

    def _sample_custom(
        noise: Any, guider: Any, sampler: Any, sigmas: Any, latent: Any
    ) -> tuple[Any, Any]:
        sample_custom_call_count[0] += 1
        if sample_custom_call_count[0] == 1:
            # Pass 1: return (output, _) — pipeline uses index 0.
            return _track("sample_custom", (fake_output_p1, MagicMock(name="denoised_p1_unused")))
        else:
            # Pass 2: return (_, denoised) — pipeline uses index 1.
            return _track("sample_custom", (MagicMock(name="output_p2_unused"), fake_denoised_p2))

    separate_av_call_count: list[int] = [0]

    def _separate_av(d: Any) -> tuple[Any, Any]:
        separate_av_call_count[0] += 1
        return _track("ltxv_separate_av_latent", (fake_video_out, fake_audio_out))

    patches = [
        (_RUNTIME_PATCH, dict(return_value={"python_version": "3.12.0"})),
        (_MM_PATCH, dict(return_value=mm)),
        (_APPLY_LORA_PATCH, dict(side_effect=_apply_lora)),
        (_LOAD_VIDEO_PATCH, dict(return_value=MagicMock(name="video"))),
        (
            _GET_VIDEO_COMPONENTS_PATCH,
            dict(return_value=(fake_frames_tensor, MagicMock(name="vid_audio"))),
        ),
        (_IMAGE_FROM_BATCH_PATCH, dict(side_effect=[fake_frames_tensor, fake_first_frame])),
        (_PREPROCESS_PATCH, dict(return_value=fake_frames_full)),
        (_SCALE_BY_PATCH, dict(return_value=fake_frames_half)),
        (_LOTUS_DEPTH_PATCH, dict(side_effect=lambda m, v, img: _track("lotus_depth_pass", fake_depth_map))),
        (_ENCODE_PATCH, dict(return_value=(MagicMock(), MagicMock()))),
        (_LTXV_COND_PATCH, dict(side_effect=lambda pos, neg, **kw: _track("ltxv_conditioning", (pos, neg)))),
        (_EMPTY_VIDEO_PATCH, dict(return_value=fake_video_latent)),
        (_IMG_TO_VIDEO_PATCH, dict(side_effect=lambda vae, img, lat, **kw: _track("ltxv_img_to_video_inplace", fake_inplace_latent))),
        (_ADD_GUIDE_PATCH, dict(side_effect=lambda pos, neg, vae, lat, img, **kw: _track("ltxv_add_guide", (pos, neg, fake_add_guide_latent)))),
        (_EMPTY_AUDIO_PATCH, dict(side_effect=lambda av, **kw: _track("ltxv_empty_latent_audio", fake_audio_latent))),
        (_CONCAT_AV_PATCH, dict(side_effect=lambda vl, al: _track("ltxv_concat_av_latent", fake_av_latent))),
        (_LTXV_SCHEDULER_PATCH, dict(side_effect=_ltxv_scheduler)),
        (_MANUAL_SIGMAS_PATCH, dict(side_effect=_manual_sigmas)),
        (_CFG_GUIDER_PATCH, dict(side_effect=_cfg_guider)),
        (_RANDOM_NOISE_PATCH, dict(return_value=MagicMock(name="noise"))),
        (_GET_SAMPLER_PATCH, dict(side_effect=_get_sampler)),
        (_SAMPLE_CUSTOM_PATCH, dict(side_effect=_sample_custom)),
        (_SEPARATE_AV_PATCH, dict(side_effect=_separate_av)),
        (_CROP_GUIDES_PATCH, dict(side_effect=lambda p, n, lat: _track("ltxv_crop_guides", (p, n, fake_video_out)))),
        (_UPSAMPLE_PATCH, dict(side_effect=lambda s, **kw: _track("ltxv_latent_upsample", fake_video_up))),
        (_VAE_DECODE_PATCH, dict(side_effect=lambda v, s: _track("vae_decode_batch_tiled", fake_frames))),
        (_AUDIO_DECODE_PATCH, dict(side_effect=lambda av, al: _track("ltxv_audio_vae_decode", fake_audio))),
    ]

    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        return pipeline_mod.run(
            models_dir=tmp_path,
            video_path=tmp_path / "input.mp4",
            prompt="a squirrel walks through a dense autumn forest",
        )


# ---------------------------------------------------------------------------
# AC02 — run() executes end-to-end in the correct order
# ---------------------------------------------------------------------------


def test_run_returns_frames_and_audio_keys(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path)
    assert "frames" in result
    assert "audio" in result


def test_run_frames_value_is_correct(tmp_path: Path) -> None:
    fake_frames = [MagicMock(name="f0")]
    result = _run_with_mocks(tmp_path, fake_frames=fake_frames)
    assert result["frames"] is fake_frames


def test_run_audio_value_is_correct(tmp_path: Path) -> None:
    fake_audio = MagicMock(name="my_audio")
    result = _run_with_mocks(tmp_path, fake_audio=fake_audio)
    assert result["audio"] is fake_audio


def test_run_calls_lotus_depth_pass(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "lotus_depth_pass" in call_order


def test_run_lotus_depth_pass_before_encode_prompt(tmp_path: Path) -> None:
    """Lotus depth estimation must occur before text conditioning."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "lotus_depth_pass" in call_order
    # Conditioning is tracked through ltxv_conditioning
    if "ltxv_conditioning" in call_order:
        assert call_order.index("lotus_depth_pass") < call_order.index("ltxv_conditioning")


def test_run_calls_apply_lora_twice(tmp_path: Path) -> None:
    """Depth control LoRA (pass 1) and distilled LoRA (pass 2) — two apply_lora calls."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert call_order.count("apply_lora") == 2


def test_run_calls_ltxv_conditioning(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_conditioning" in call_order


def test_run_calls_ltxv_add_guide(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_add_guide" in call_order


def test_run_calls_ltxv_img_to_video_inplace(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_img_to_video_inplace" in call_order


def test_run_calls_ltxv_empty_latent_audio(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_empty_latent_audio" in call_order


def test_run_calls_ltxv_concat_av_latent(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_concat_av_latent" in call_order


def test_run_calls_sample_custom_twice(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert call_order.count("sample_custom") == 2


def test_run_calls_ltxv_separate_av_latent(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_separate_av_latent" in call_order


def test_run_calls_ltxv_crop_guides(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_crop_guides" in call_order


def test_run_calls_ltxv_latent_upsample(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_latent_upsample" in call_order


def test_run_calls_vae_decode_batch_tiled(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "vae_decode_batch_tiled" in call_order


def test_run_calls_ltxv_audio_vae_decode(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_audio_vae_decode" in call_order


# ---------------------------------------------------------------------------
# AC03 — Sampler, sigmas, and CFG match the workflow
# ---------------------------------------------------------------------------


def test_run_pass1_uses_ltxv_scheduler(tmp_path: Path) -> None:
    ltxv_scheduler_calls: list[tuple[Any, ...]] = []
    _run_with_mocks(tmp_path, ltxv_scheduler_calls=ltxv_scheduler_calls)
    assert len(ltxv_scheduler_calls) == 1, (
        f"ltxv_scheduler must be called exactly once (pass 1), got {ltxv_scheduler_calls}"
    )
    steps, max_shift, base_shift = ltxv_scheduler_calls[0]
    assert steps == 20, f"Pass 1 LTXVScheduler must use 20 steps, got {steps}"
    assert max_shift == pytest.approx(2.05), f"max_shift must be 2.05, got {max_shift}"
    assert base_shift == pytest.approx(0.95), f"base_shift must be 0.95, got {base_shift}"


def test_run_pass2_uses_4_step_manual_sigmas(tmp_path: Path) -> None:
    manual_sigmas_calls: list[str] = []
    _run_with_mocks(tmp_path, manual_sigmas_calls=manual_sigmas_calls)
    assert len(manual_sigmas_calls) == 1, (
        f"manual_sigmas must be called exactly once (pass 2), got {manual_sigmas_calls}"
    )
    pass2_sigmas = manual_sigmas_calls[0]
    values = [v.strip() for v in pass2_sigmas.split(",")]
    assert len(values) == 4, (
        f"Pass 2 ManualSigmas must have 4 values, got {len(values)}: {pass2_sigmas!r}"
    )
    assert "0.909375" in pass2_sigmas, (
        f"Pass 2 ManualSigmas must start with 0.909375, got {pass2_sigmas!r}"
    )


def test_run_pass1_uses_euler_sampler(tmp_path: Path) -> None:
    sampler_calls: list[str] = []
    _run_with_mocks(tmp_path, sampler_calls=sampler_calls)
    assert len(sampler_calls) >= 1, "get_sampler must be called at least once"
    assert sampler_calls[0] == "euler", (
        f"Pass 1 must use 'euler' sampler, got {sampler_calls[0]!r}"
    )


def test_run_pass2_uses_gradient_estimation_sampler(tmp_path: Path) -> None:
    sampler_calls: list[str] = []
    _run_with_mocks(tmp_path, sampler_calls=sampler_calls)
    assert len(sampler_calls) >= 2, "get_sampler must be called twice (once per pass)"
    assert sampler_calls[1] == "gradient_estimation", (
        f"Pass 2 must use 'gradient_estimation' sampler, got {sampler_calls[1]!r}"
    )


def test_run_pass1_cfg_is_3(tmp_path: Path) -> None:
    cfg_calls: list[float] = []
    _run_with_mocks(tmp_path, cfg_guider_calls=cfg_calls)
    assert len(cfg_calls) == 2, f"Expected 2 cfg_guider calls, got {cfg_calls}"
    assert cfg_calls[0] == 3.0, (
        f"Pass 1 CFG must be 3.0, got {cfg_calls[0]}"
    )


def test_run_pass2_cfg_is_1(tmp_path: Path) -> None:
    cfg_calls: list[float] = []
    _run_with_mocks(tmp_path, cfg_guider_calls=cfg_calls)
    assert cfg_calls[1] == 1.0, (
        f"Pass 2 CFG must be 1.0, got {cfg_calls[1]}"
    )


def test_run_uses_ltxv_scheduler_not_manual_sigmas_for_pass1(tmp_path: Path) -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert "ltxv_scheduler" in source, (
        "Depth pipeline must use ltxv_scheduler for pass 1"
    )


def test_run_does_not_use_euler_ancestral(tmp_path: Path) -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert "euler_ancestral" not in source, (
        "Depth pipeline must use 'euler' (not euler_ancestral) for pass 1"
    )
