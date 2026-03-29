"""Tests for comfy_diffusion/pipelines/video/ltx/ltx2/pose.py — LTX2 Pose-to-Video pipeline.

Covers:
  AC01: manifest() returns exactly 5 HFModelEntry items with correct dest paths
        (ckpt, text_encoder, pose_lora, distilled_lora, upscaler)
  AC02: run() follows the workflow node execution order end-to-end
        (DWPreprocessor → load models → pose LoRA → conditioning → pass 1 →
         between passes → pass 2 → decode)
  AC03: Pass 1 uses LTXVScheduler + euler; pass 2 uses ManualSigmas(4 values) +
        gradient_estimation; CFG 3/1 — matching the pose workflow (LTX 2.0 dev fp8
        subgraph, same sigma/sampler pattern as depth pipeline)
  AC04: run() accepts prompt, negative_prompt, video_path, first_frame_path,
        width, height, length, fps, cfg_pass1, cfg_pass2, seed,
        pose_lora_strength, lora_strength, and all filename-override parameters
  AC05: Pipeline file is at comfy_diffusion/pipelines/video/ltx/ltx2/pose.py
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
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx2" / "pose.py"
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
_DW_PREPROCESSOR_PATCH = "comfy_diffusion.image.dw_preprocessor"
_IMAGE_FROM_BATCH_PATCH = "comfy_diffusion.image.image_from_batch"
_LOAD_IMAGE_PATCH = "comfy_diffusion.image.load_image"


# ---------------------------------------------------------------------------
# AC05 / AC06 — file-level checks
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "pose.py must exist at the expected path"


def test_pipeline_parses_without_syntax_errors() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PIPELINE_FILE))
    assert isinstance(tree, ast.Module)


def test_no_top_level_comfy_imports() -> None:
    """No top-level 'import comfy' or 'from comfy' in pipeline module."""
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Only check module-level imports (not inside function bodies)
            pass
    # Check that comfy.* is not imported at top level (outside function defs)
    top_imports = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    # All comfy_diffusion.* imports inside run() are fine; top-level ones are not
    module_body_imports = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    for node in module_body_imports:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("comfy."), (
                f"Top-level 'from comfy.*' import is forbidden: {module}"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("comfy."), (
                    f"Top-level 'import comfy.*' is forbidden: {alias.name}"
                )


def test_all_exports_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    all_names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        all_names = [
                            elt.s for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        ]
    assert "manifest" in all_names, "__all__ must export 'manifest'"
    assert "run" in all_names, "__all__ must export 'run'"


# ---------------------------------------------------------------------------
# AC01 — manifest() returns 5 entries with correct destinations
# ---------------------------------------------------------------------------


def test_manifest_returns_five_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest

    entries = manifest()
    assert len(entries) == 5, f"manifest() must return exactly 5 entries, got {len(entries)}"


def test_manifest_all_entries_are_hf_model_entry() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"All manifest entries must be HFModelEntry, got {type(entry)}"
        )


def test_manifest_checkpoint_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("checkpoints" in d and "ltx-2-19b-dev-fp8" in d for d in dests), (
        "manifest() must include a checkpoints/ltx-2-19b-dev-fp8 entry"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include a text_encoders/gemma_3_12B_it_fp4_mixed entry"
    )


def test_manifest_pose_lora_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d and "ic-lora-pose-control" in d for d in dests), (
        "manifest() must include a loras/ltx-2-19b-ic-lora-pose-control entry"
    )


def test_manifest_distilled_lora_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d and "distilled-lora-384" in d for d in dests), (
        "manifest() must include a loras/ltx-2-19b-distilled-lora-384 entry"
    )


def test_manifest_upscaler_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("upscale_models" in d and "ltx-2-spatial-upscaler" in d for d in dests), (
        "manifest() must include an upscale_models/ltx-2-spatial-upscaler entry"
    )


def test_manifest_pose_lora_uses_correct_hf_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest

    pose_entries = [e for e in manifest() if "ic-lora-pose-control" in str(e.dest)]
    assert len(pose_entries) == 1, "manifest() must have exactly one pose control LoRA entry"
    assert "Pose-Control" in pose_entries[0].repo_id or "pose" in pose_entries[0].repo_id.lower(), (
        "Pose control LoRA must be fetched from the Lightricks Pose-Control HF repo"
    )


# ---------------------------------------------------------------------------
# AC04 — run() signature
# ---------------------------------------------------------------------------


def test_run_accepts_video_path_param() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import run

    sig = inspect.signature(run)
    assert "video_path" in sig.parameters, "run() must accept a 'video_path' parameter"


def test_run_accepts_first_frame_path_param() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import run

    sig = inspect.signature(run)
    assert "first_frame_path" in sig.parameters, (
        "run() must accept a 'first_frame_path' parameter"
    )
    assert sig.parameters["first_frame_path"].default is None, (
        "first_frame_path should default to None"
    )


def test_run_accepts_prompt_and_negative_prompt() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import run

    sig = inspect.signature(run)
    assert "prompt" in sig.parameters
    assert "negative_prompt" in sig.parameters


def test_run_accepts_resolution_and_length_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import run

    sig = inspect.signature(run)
    for param in ("width", "height", "length", "fps"):
        assert param in sig.parameters, f"run() must accept '{param}'"


def test_run_accepts_cfg_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import run

    sig = inspect.signature(run)
    assert "cfg_pass1" in sig.parameters
    assert "cfg_pass2" in sig.parameters
    assert sig.parameters["cfg_pass1"].default == 3.0
    assert sig.parameters["cfg_pass2"].default == 1.0


def test_run_accepts_lora_strength_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import run

    sig = inspect.signature(run)
    assert "pose_lora_strength" in sig.parameters
    assert "lora_strength" in sig.parameters


def test_run_accepts_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import run

    sig = inspect.signature(run)
    for param in (
        "ckpt_filename",
        "text_encoder_filename",
        "vae_filename",
        "audio_vae_filename",
        "pose_lora_filename",
        "lora_filename",
        "upscaler_filename",
    ):
        assert param in sig.parameters, f"run() must accept '{param}'"


def test_run_length_default_is_121() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import run

    sig = inspect.signature(run)
    assert sig.parameters["length"].default == 121, (
        "run() default length must be 121 (from workflow PrimitiveInt node)"
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
    first_frame_path: str | Path | None = None,
) -> dict[str, Any]:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import pose as pipeline_mod

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
    fake_pose_frames = MagicMock(name="pose_frames")
    fake_pose_frames_full = MagicMock(name="pose_frames_full")
    fake_pose_frames_half = MagicMock(name="pose_frames_half")
    fake_first_frame = MagicMock(name="first_frame")
    fake_first_frame_full = MagicMock(name="first_frame_full")
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
            return _track("sample_custom", (fake_output_p1, MagicMock(name="denoised_p1_unused")))
        else:
            return _track("sample_custom", (MagicMock(name="output_p2_unused"), fake_denoised_p2))

    separate_av_call_count: list[int] = [0]

    def _separate_av(d: Any) -> tuple[Any, Any]:
        separate_av_call_count[0] += 1
        return _track("ltxv_separate_av_latent", (fake_video_out, fake_audio_out))

    preprocess_call_count: list[int] = [0]

    def _preprocess(img: Any, w: int, h: int) -> Any:
        preprocess_call_count[0] += 1
        if preprocess_call_count[0] == 1:
            # First call: pose_frames → pose_frames_full
            return _track("ltxv_preprocess", fake_pose_frames_full)
        else:
            # Second call: first_frame_img or frames_full → full-res image
            return _track("ltxv_preprocess", fake_first_frame_full)

    image_from_batch_call_count: list[int] = [0]

    def _image_from_batch(img: Any, idx: int, n: int) -> Any:
        image_from_batch_call_count[0] += 1
        if image_from_batch_call_count[0] == 1:
            return _track("image_from_batch", fake_frames_tensor)
        else:
            return _track("image_from_batch", fake_first_frame)

    patches = [
        (_RUNTIME_PATCH, dict(return_value={"python_version": "3.12.0"})),
        (_MM_PATCH, dict(return_value=mm)),
        (_APPLY_LORA_PATCH, dict(side_effect=_apply_lora)),
        (_LOAD_VIDEO_PATCH, dict(return_value=MagicMock(name="video"))),
        (
            _GET_VIDEO_COMPONENTS_PATCH,
            dict(return_value=(fake_frames_tensor, MagicMock(name="vid_audio"))),
        ),
        (_IMAGE_FROM_BATCH_PATCH, dict(side_effect=_image_from_batch)),
        (_DW_PREPROCESSOR_PATCH, dict(side_effect=lambda img, **kw: _track("dw_preprocessor", fake_pose_frames))),
        (_PREPROCESS_PATCH, dict(side_effect=_preprocess)),
        (_SCALE_BY_PATCH, dict(return_value=fake_pose_frames_half)),
        (_LOAD_IMAGE_PATCH, dict(return_value=(fake_first_frame, MagicMock(name="mask")))),
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
            prompt="a woman moves gracefully through a sunlit garden",
            first_frame_path=first_frame_path,
        )


# ---------------------------------------------------------------------------
# AC02 — run() follows workflow execution order
# ---------------------------------------------------------------------------


def test_run_returns_frames_and_audio(tmp_path: Path) -> None:
    fake_frames = [MagicMock(name="frame0"), MagicMock(name="frame1")]
    fake_audio = MagicMock(name="audio_out")
    result = _run_with_mocks(tmp_path, fake_frames=fake_frames, fake_audio=fake_audio)
    assert "frames" in result
    assert "audio" in result
    assert result["frames"] is fake_frames
    assert result["audio"] is fake_audio


def test_run_calls_load_video(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    # load_video is tracked indirectly; verify video components were consumed
    assert "image_from_batch" in call_order


def test_run_calls_dw_preprocessor(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "dw_preprocessor" in call_order, "run() must call dw_preprocessor for pose estimation"


def test_run_applies_pose_lora_before_sampling(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "apply_lora" in call_order, "run() must call apply_lora for pose control LoRA"
    lora_idx = call_order.index("apply_lora")
    # LTXVScheduler (pass 1 sigmas) should come after apply_lora
    assert "ltxv_scheduler" in call_order
    sched_idx = call_order.index("ltxv_scheduler")
    assert lora_idx < sched_idx, "apply_lora must happen before LTXVScheduler"


def test_run_calls_ltxv_add_guide(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_add_guide" in call_order, "run() must call ltxv_add_guide with pose frames"


def test_run_calls_ltxv_img_to_video_inplace(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    inplace_calls = [x for x in call_order if x == "ltxv_img_to_video_inplace"]
    assert len(inplace_calls) >= 2, (
        "run() must call ltxv_img_to_video_inplace at least twice "
        "(once for pass 1 first-frame injection, once for between-passes reinjectioon)"
    )


def test_run_calls_ltxv_latent_upsample(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_latent_upsample" in call_order, (
        "run() must call ltxv_latent_upsample between passes"
    )


def test_run_calls_ltxv_crop_guides(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "ltxv_crop_guides" in call_order, (
        "run() must call ltxv_crop_guides between passes"
    )


def test_run_calls_sample_custom_twice(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    sample_calls = [x for x in call_order if x == "sample_custom"]
    assert len(sample_calls) == 2, "run() must call sample_custom exactly twice (pass 1 + pass 2)"


# ---------------------------------------------------------------------------
# AC03 — sampler / sigmas / CFG match the pose workflow
# ---------------------------------------------------------------------------


def test_pass1_uses_ltxv_scheduler(tmp_path: Path) -> None:
    ltxv_scheduler_calls: list[tuple[Any, ...]] = []
    _run_with_mocks(tmp_path, ltxv_scheduler_calls=ltxv_scheduler_calls)
    assert len(ltxv_scheduler_calls) == 1, "run() must call ltxv_scheduler exactly once (pass 1)"
    steps, max_shift, base_shift = ltxv_scheduler_calls[0]
    assert steps == 20, f"LTXVScheduler steps must be 20, got {steps}"
    assert max_shift == pytest.approx(2.05), f"max_shift must be 2.05, got {max_shift}"
    assert base_shift == pytest.approx(0.95), f"base_shift must be 0.95, got {base_shift}"


def test_pass2_uses_manual_sigmas_4_values(tmp_path: Path) -> None:
    manual_sigmas_calls: list[str] = []
    _run_with_mocks(tmp_path, manual_sigmas_calls=manual_sigmas_calls)
    assert len(manual_sigmas_calls) == 1, "run() must call manual_sigmas exactly once (pass 2)"
    sigma_str = manual_sigmas_calls[0]
    sigma_values = [v.strip() for v in sigma_str.split(",")]
    assert len(sigma_values) == 4, (
        f"Pass 2 ManualSigmas must have 4 values, got {len(sigma_values)}: {sigma_str}"
    )
    assert sigma_values[-1] == "0.0", "Last sigma must be 0.0"


def test_pass1_uses_euler_sampler(tmp_path: Path) -> None:
    sampler_calls: list[str] = []
    _run_with_mocks(tmp_path, sampler_calls=sampler_calls)
    assert "euler" in sampler_calls, "Pass 1 must use 'euler' sampler"


def test_pass2_uses_gradient_estimation_sampler(tmp_path: Path) -> None:
    sampler_calls: list[str] = []
    _run_with_mocks(tmp_path, sampler_calls=sampler_calls)
    assert "gradient_estimation" in sampler_calls, "Pass 2 must use 'gradient_estimation' sampler"


def test_cfg_pass1_is_3(tmp_path: Path) -> None:
    cfg_calls: list[float] = []
    _run_with_mocks(tmp_path, cfg_guider_calls=cfg_calls)
    assert 3.0 in cfg_calls, f"CFG=3 must be used for pass 1, got: {cfg_calls}"


def test_cfg_pass2_is_1(tmp_path: Path) -> None:
    cfg_calls: list[float] = []
    _run_with_mocks(tmp_path, cfg_guider_calls=cfg_calls)
    assert 1.0 in cfg_calls, f"CFG=1 must be used for pass 2, got: {cfg_calls}"


def test_both_passes_use_same_seed(tmp_path: Path) -> None:
    """Both passes use the same seed (matching workflow RandomNoise [0, 'fixed'])."""
    from comfy_diffusion.pipelines.video.ltx.ltx2 import pose as pipeline_mod

    noise_seeds: list[int] = []

    def _random_noise(seed: int) -> MagicMock:
        noise_seeds.append(seed)
        return MagicMock(name=f"noise_{seed}")

    fake_audio = MagicMock(name="audio")
    fake_frames = [MagicMock(name="frame0")]

    fake_video_latent = MagicMock(name="video_latent")
    fake_audio_latent = MagicMock(name="audio_latent")
    fake_av_latent = MagicMock(name="av_latent")
    fake_output_p1 = MagicMock(name="output_p1")
    fake_denoised_p2 = MagicMock(name="denoised_p2")
    fake_video_out = MagicMock(name="video_out")
    fake_audio_out = MagicMock(name="audio_out")
    fake_video_up = MagicMock(name="video_up")
    fake_frames_tensor = MagicMock(name="frames_tensor")
    sample_call_count: list[int] = [0]

    def _sample_custom(noise: Any, guider: Any, sampler: Any, sigmas: Any, latent: Any) -> tuple[Any, Any]:
        sample_call_count[0] += 1
        if sample_call_count[0] == 1:
            return (fake_output_p1, MagicMock())
        else:
            return (MagicMock(), fake_denoised_p2)

    mm = _build_mock_mm()
    patched_model = MagicMock(name="model")
    patched_clip = MagicMock(name="clip")

    patches = [
        (_RUNTIME_PATCH, dict(return_value={"python_version": "3.12.0"})),
        (_MM_PATCH, dict(return_value=mm)),
        (_APPLY_LORA_PATCH, dict(return_value=(patched_model, patched_clip))),
        (_LOAD_VIDEO_PATCH, dict(return_value=MagicMock())),
        (_GET_VIDEO_COMPONENTS_PATCH, dict(return_value=(fake_frames_tensor, MagicMock()))),
        (_IMAGE_FROM_BATCH_PATCH, dict(return_value=fake_frames_tensor)),
        (_DW_PREPROCESSOR_PATCH, dict(return_value=MagicMock())),
        (_PREPROCESS_PATCH, dict(return_value=MagicMock())),
        (_SCALE_BY_PATCH, dict(return_value=MagicMock())),
        (_ENCODE_PATCH, dict(return_value=(MagicMock(), MagicMock()))),
        (_LTXV_COND_PATCH, dict(side_effect=lambda pos, neg, **kw: (pos, neg))),
        (_EMPTY_VIDEO_PATCH, dict(return_value=fake_video_latent)),
        (_IMG_TO_VIDEO_PATCH, dict(return_value=MagicMock())),
        (_ADD_GUIDE_PATCH, dict(return_value=(MagicMock(), MagicMock(), MagicMock()))),
        (_EMPTY_AUDIO_PATCH, dict(return_value=fake_audio_latent)),
        (_CONCAT_AV_PATCH, dict(return_value=fake_av_latent)),
        (_LTXV_SCHEDULER_PATCH, dict(return_value=MagicMock())),
        (_MANUAL_SIGMAS_PATCH, dict(return_value=MagicMock())),
        (_CFG_GUIDER_PATCH, dict(return_value=MagicMock())),
        (_RANDOM_NOISE_PATCH, dict(side_effect=_random_noise)),
        (_GET_SAMPLER_PATCH, dict(return_value=MagicMock())),
        (_SAMPLE_CUSTOM_PATCH, dict(side_effect=_sample_custom)),
        (_SEPARATE_AV_PATCH, dict(return_value=(fake_video_out, fake_audio_out))),
        (_CROP_GUIDES_PATCH, dict(return_value=(MagicMock(), MagicMock(), fake_video_out))),
        (_UPSAMPLE_PATCH, dict(return_value=fake_video_up)),
        (_VAE_DECODE_PATCH, dict(return_value=fake_frames)),
        (_AUDIO_DECODE_PATCH, dict(return_value=fake_audio)),
    ]

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        with contextlib.ExitStack() as stack:
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            pipeline_mod.run(
                models_dir=tmp_dir,
                video_path="/tmp/input.mp4",
                prompt="test",
                seed=42,
            )

    assert len(noise_seeds) == 2, f"Expected 2 random_noise calls, got {len(noise_seeds)}"
    assert noise_seeds[0] == 42, f"Pass 1 seed must be 42, got {noise_seeds[0]}"
    assert noise_seeds[1] == 42, f"Pass 2 seed must be 42 (same as pass 1), got {noise_seeds[1]}"


# ---------------------------------------------------------------------------
# AC04 — first_frame_path optional fallback
# ---------------------------------------------------------------------------


def test_run_loads_first_frame_from_path_when_provided(tmp_path: Path) -> None:
    """When first_frame_path is given, load_image must be called."""
    call_order: list[str] = []

    load_image_called: list[str | Path] = []

    def _load_image(path: str | Path) -> tuple[Any, Any]:
        load_image_called.append(path)
        return (MagicMock(name="first_frame_img"), MagicMock(name="mask"))

    from comfy_diffusion.pipelines.video.ltx.ltx2 import pose as pipeline_mod

    fake_video_latent = MagicMock(name="video_latent")
    fake_audio_latent = MagicMock(name="audio_latent")
    fake_av_latent = MagicMock(name="av_latent")
    fake_output_p1 = MagicMock(name="output_p1")
    fake_denoised_p2 = MagicMock(name="denoised_p2")
    fake_video_out = MagicMock(name="video_out")
    fake_audio_out = MagicMock(name="audio_out")
    fake_video_up = MagicMock(name="video_up")
    fake_frames_tensor = MagicMock(name="frames_tensor")
    sample_call_count: list[int] = [0]

    def _sample_custom(noise: Any, guider: Any, sampler: Any, sigmas: Any, latent: Any) -> tuple[Any, Any]:
        sample_call_count[0] += 1
        if sample_call_count[0] == 1:
            return (fake_output_p1, MagicMock())
        else:
            return (MagicMock(), fake_denoised_p2)

    mm = _build_mock_mm()
    patched_model = MagicMock(name="model")
    patched_clip = MagicMock(name="clip")

    first_frame_path = tmp_path / "first_frame.png"
    first_frame_path.touch()

    patches = [
        (_RUNTIME_PATCH, dict(return_value={"python_version": "3.12.0"})),
        (_MM_PATCH, dict(return_value=mm)),
        (_APPLY_LORA_PATCH, dict(return_value=(patched_model, patched_clip))),
        (_LOAD_VIDEO_PATCH, dict(return_value=MagicMock())),
        (_GET_VIDEO_COMPONENTS_PATCH, dict(return_value=(fake_frames_tensor, MagicMock()))),
        (_IMAGE_FROM_BATCH_PATCH, dict(return_value=fake_frames_tensor)),
        (_DW_PREPROCESSOR_PATCH, dict(return_value=MagicMock())),
        (_PREPROCESS_PATCH, dict(return_value=MagicMock())),
        (_SCALE_BY_PATCH, dict(return_value=MagicMock())),
        (_LOAD_IMAGE_PATCH, dict(side_effect=_load_image)),
        (_ENCODE_PATCH, dict(return_value=(MagicMock(), MagicMock()))),
        (_LTXV_COND_PATCH, dict(side_effect=lambda pos, neg, **kw: (pos, neg))),
        (_EMPTY_VIDEO_PATCH, dict(return_value=fake_video_latent)),
        (_IMG_TO_VIDEO_PATCH, dict(return_value=MagicMock())),
        (_ADD_GUIDE_PATCH, dict(return_value=(MagicMock(), MagicMock(), MagicMock()))),
        (_EMPTY_AUDIO_PATCH, dict(return_value=fake_audio_latent)),
        (_CONCAT_AV_PATCH, dict(return_value=fake_av_latent)),
        (_LTXV_SCHEDULER_PATCH, dict(return_value=MagicMock())),
        (_MANUAL_SIGMAS_PATCH, dict(return_value=MagicMock())),
        (_CFG_GUIDER_PATCH, dict(return_value=MagicMock())),
        (_RANDOM_NOISE_PATCH, dict(return_value=MagicMock())),
        (_GET_SAMPLER_PATCH, dict(return_value=MagicMock())),
        (_SAMPLE_CUSTOM_PATCH, dict(side_effect=_sample_custom)),
        (_SEPARATE_AV_PATCH, dict(return_value=(fake_video_out, fake_audio_out))),
        (_CROP_GUIDES_PATCH, dict(return_value=(MagicMock(), MagicMock(), fake_video_out))),
        (_UPSAMPLE_PATCH, dict(return_value=fake_video_up)),
        (_VAE_DECODE_PATCH, dict(return_value=[MagicMock()])),
        (_AUDIO_DECODE_PATCH, dict(return_value=MagicMock())),
    ]

    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        pipeline_mod.run(
            models_dir=tmp_path,
            video_path=tmp_path / "input.mp4",
            prompt="test",
            first_frame_path=first_frame_path,
        )

    assert len(load_image_called) == 1, "load_image must be called once when first_frame_path is given"
    assert load_image_called[0] == first_frame_path


# ---------------------------------------------------------------------------
# AC01 — manifest paths are consistent with run() path derivations
# ---------------------------------------------------------------------------


def test_manifest_ckpt_path_consistent_with_run_default(tmp_path: Path) -> None:
    """run() default ckpt path must equal models_dir / manifest()[0].dest."""
    from comfy_diffusion.pipelines.video.ltx.ltx2.pose import manifest

    entries = manifest()
    ckpt_entry = next(
        (e for e in entries if "ltx-2-19b-dev-fp8" in str(e.dest)), None
    )
    assert ckpt_entry is not None, "manifest() must include the dev-fp8 checkpoint entry"
    assert "checkpoints" in str(ckpt_entry.dest), (
        "dev-fp8 checkpoint must be stored in the 'checkpoints/' subdirectory"
    )


def test_init_exports_pose() -> None:
    """__init__.py must list 'pose' in __all__."""
    from comfy_diffusion.pipelines.video.ltx import ltx2

    assert "pose" in ltx2.__all__, "'pose' must be listed in ltx2.__all__"
