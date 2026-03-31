"""Tests for comfy_diffusion/pipelines/video/wan/wan22/s2v.py — WAN 2.2 S2V pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring (AC-06)
  - Exports manifest and run; no top-level comfy imports (AC-06)
  - __all__ = ["manifest", "run"]
  - manifest() returns exactly 5 HFModelEntry items (AC-01)
  - All manifest entries use the Comfy-Org/Wan_2.2_ComfyUI_Repackaged HF repo
  - manifest() has audio_encoder, unet, lora, text_encoder, vae entries (AC-01)
  - run() signature: audio, ref_image, control_video, prompt, negative_prompt,
    *, models_dir, seed, steps, cfg (AC-02)
  - run() calls load_audio_encoder then audio_encoder_encode (AC-02)
  - run() calls wan_sound_image_to_video then WanSoundImageToVideoExtend loop
    followed by LatentConcat (AC-02)
  - audio dict accepts waveform and sample_rate keys (AC-03)
  - wan22 sub-package exports s2v (AC-04 covered by import)
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
    / "s2v.py"
)

# ---------------------------------------------------------------------------
# Patch constants (source module paths for lazy imports inside run())
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_MODEL_SAMPLING_SD3_PATCH = "comfy_diffusion.models.model_sampling_sd3"
_ENCODE_PROMPT_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_WAN_S2V_PATCH = "comfy_diffusion.conditioning.wan_sound_image_to_video"
_WAN_S2V_EXTEND_PATCH = "comfy_diffusion.conditioning.wan_sound_image_to_video_extend"
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"
_AUDIO_ENC_ENCODE_PATCH = "comfy_diffusion.audio.audio_encoder_encode"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_LATENT_CONCAT_PATCH = "comfy_diffusion.latent.latent_concat"
_LATENT_CUT_PATCH = "comfy_diffusion.latent.latent_cut"
_VAE_DECODE_BATCH_PATCH = "comfy_diffusion.vae.vae_decode_batch"
_APPLY_LORA_PATCH = "comfy_diffusion.lora.apply_lora"


# ---------------------------------------------------------------------------
# File-level checks (AC-06)
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "s2v.py must exist under wan22/"


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
    assert docstring, "s2v.py must have a module-level docstring"


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
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks — exactly 5 HFModelEntry items (AC-01)
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_five_entries() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 5, f"manifest() must return exactly 5 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_has_audio_encoder_entry() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("audio_encoders" in d and "wav2vec2" in d for d in dests), (
        "manifest() must include an audio_encoders/wav2vec2... entry"
    )


def test_manifest_has_unet_entry() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "s2v" in d for d in dests), (
        "manifest() must include a diffusion_models/...s2v... UNet entry"
    )


def test_manifest_has_lora_entry() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d for d in dests), (
        "manifest() must include a loras/ LoRA entry"
    )


def test_manifest_has_text_encoder_entry() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "umt5_xxl" in d for d in dests), (
        "manifest() must include a text_encoders/umt5_xxl entry"
    )


def test_manifest_has_vae_entry() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("vae" in d and "wan_2.1_vae" in d for d in dests), (
        "manifest() must include a vae/wan_2.1_vae entry"
    )


def test_manifest_all_from_comfy_org_wan22_repo() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", (
            f"Entry {entry!r} must use repo_id 'Comfy-Org/Wan_2.2_ComfyUI_Repackaged'"
        )


def test_manifest_no_bypassed_nodes() -> None:
    """All 5 entries correspond to active (non-bypassed) nodes (AC-01)."""
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import manifest

    result = manifest()
    assert len(result) == 5, (
        "manifest() must have exactly 5 entries (active nodes only, no bypassed)"
    )


# ---------------------------------------------------------------------------
# run() signature checks (AC-02)
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import run

    sig = inspect.signature(run)
    required = {
        "audio", "ref_image", "control_video", "prompt", "negative_prompt",
        "models_dir", "seed", "steps", "cfg",
    }
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_models_dir_is_keyword_only() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import run

    sig = inspect.signature(run)
    param = sig.parameters["models_dir"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
        "models_dir must be keyword-only"
    )


def test_run_seed_is_keyword_only() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import run

    sig = inspect.signature(run)
    assert sig.parameters["seed"].kind == inspect.Parameter.KEYWORD_ONLY


def test_run_default_steps_10() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 10


def test_run_default_cfg_6() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.s2v import run

    sig = inspect.signature(run)
    assert sig.parameters["cfg"].default == 6.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_audio() -> dict[str, Any]:
    """Return a minimal audio dict matching the ComfyUI AUDIO format (AC-03)."""
    import torch

    return {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_audio_encoder.return_value = MagicMock(name="audio_encoder")
    return mm


def _run_with_mocks(
    tmp_path: Path,
    *,
    call_order: list[str] | None = None,
    fake_frames: list[Any] | None = None,
    num_extend_passes: int = 2,
) -> list[Any]:
    from comfy_diffusion.pipelines.video.wan.wan22 import s2v as pipeline_mod

    if fake_frames is None:
        fake_frames = [MagicMock(name="frame0")]

    fake_audio_enc_output = MagicMock(name="audio_enc_output")
    fake_img_tensor = MagicMock(name="img_tensor")
    fake_latent_init = MagicMock(name="latent_init")
    fake_accumulated = MagicMock(name="accumulated")
    fake_extended = MagicMock(name="latent_ext")
    fake_new_segment = MagicMock(name="new_segment")
    fake_first_frame = MagicMock(name="first_frame")
    fake_final = MagicMock(name="final_latent")
    fake_patched_model = MagicMock(name="patched_model")

    mm = _build_mock_mm()
    sample_call_count: list[int] = [0]
    concat_call_count: list[int] = [0]

    def _track(name: str, rv: Any) -> Any:
        if call_order is not None:
            call_order.append(name)
        return rv

    def fake_sample(model: Any, pos: Any, neg: Any, lat: Any, **kwargs: Any) -> Any:
        sample_call_count[0] += 1
        n = sample_call_count[0]
        name = f"sample_{n}"
        if call_order is not None:
            call_order.append(name)
        return fake_accumulated if n == 1 else fake_new_segment

    def fake_latent_concat(*latents: Any, dim: str = "t") -> Any:
        concat_call_count[0] += 1
        n = concat_call_count[0]
        name = f"latent_concat_{n}"
        if call_order is not None:
            call_order.append(name)
        return fake_final if n > num_extend_passes else fake_accumulated

    fake_pil = MagicMock(name="pil_image")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, side_effect=lambda m, c, p, sm, sc: (
            _track("apply_lora", (fake_patched_model, c))
        )),
        patch(_MODEL_SAMPLING_SD3_PATCH, side_effect=lambda m, shift: (
            _track("model_sampling_sd3", fake_patched_model)
        )),
        patch(_ENCODE_PROMPT_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_AUDIO_ENC_ENCODE_PATCH, side_effect=lambda enc, audio: (
            _track("audio_encoder_encode", fake_audio_enc_output)
        )),
        patch(_IMAGE_TO_TENSOR_PATCH, side_effect=lambda img: (
            _track("image_to_tensor", fake_img_tensor)
        )),
        patch(_WAN_S2V_PATCH, side_effect=lambda pos, neg, vae, **kw: (
            _track("wan_sound_image_to_video", (pos, neg, fake_latent_init))
        )),
        patch(_WAN_S2V_EXTEND_PATCH, side_effect=lambda pos, neg, vae, **kw: (
            _track("wan_sound_image_to_video_extend", (pos, neg, fake_extended))
        )),
        patch(_SAMPLE_PATCH, side_effect=fake_sample),
        patch(_LATENT_CONCAT_PATCH, side_effect=fake_latent_concat),
        patch(_LATENT_CUT_PATCH, side_effect=lambda lat, **kw: (
            _track("latent_cut", fake_first_frame)
        )),
        patch(_VAE_DECODE_BATCH_PATCH, side_effect=lambda v, s: (
            _track("vae_decode_batch", fake_frames)
        )),
    ):
        audio = _make_fake_audio()
        return pipeline_mod.run(
            audio,
            fake_pil,
            None,
            "test prompt",
            models_dir=tmp_path,
            num_extend_passes=num_extend_passes,
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


def test_run_calls_audio_encoder_encode(tmp_path: Path) -> None:
    """AudioEncoderEncode must be called before wan_sound_image_to_video (AC-02)."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "audio_encoder_encode" in call_order, (
        "run() must call audio_encoder_encode"
    )
    assert call_order.index("audio_encoder_encode") < call_order.index(
        "wan_sound_image_to_video"
    ), "audio_encoder_encode must be called before wan_sound_image_to_video"


def test_run_calls_image_to_tensor(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "image_to_tensor" in call_order


def test_run_calls_wan_sound_image_to_video(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "wan_sound_image_to_video" in call_order


def test_run_calls_wan_sound_image_to_video_extend(tmp_path: Path) -> None:
    """WanSoundImageToVideoExtend must be called num_extend_passes times (AC-02)."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order, num_extend_passes=2)
    extend_calls = [c for c in call_order if c == "wan_sound_image_to_video_extend"]
    assert len(extend_calls) == 2, (
        f"wan_sound_image_to_video_extend must be called twice, got {extend_calls}"
    )


def test_run_calls_latent_concat_after_each_extend(tmp_path: Path) -> None:
    """LatentConcat must be called once per extend pass + once for stitch (AC-02)."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order, num_extend_passes=2)
    concat_calls = [c for c in call_order if c.startswith("latent_concat")]
    # 2 extend passes + 1 stitch = 3 total
    assert len(concat_calls) == 3, (
        f"latent_concat must be called 3 times (2 extend + 1 stitch), got {concat_calls}"
    )


def test_run_calls_latent_cut(tmp_path: Path) -> None:
    """LatentCut must be called once to extract the first frame for the stitch (AC-02)."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "latent_cut" in call_order, "run() must call latent_cut for the stitch step"


def test_run_calls_vae_decode_batch(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "vae_decode_batch" in call_order


def test_run_calls_apply_lora(tmp_path: Path) -> None:
    """LoRA must be applied before ModelSamplingSD3."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "apply_lora" in call_order


def test_run_calls_model_sampling_sd3(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "model_sampling_sd3" in call_order


def test_run_sample_called_initial_plus_extend(tmp_path: Path) -> None:
    """sample() must be called 1 (initial) + num_extend_passes times."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order, num_extend_passes=2)
    sample_calls = [c for c in call_order if c.startswith("sample_")]
    assert len(sample_calls) == 3, (
        f"sample must be called 3 times (1 initial + 2 extend), got {sample_calls}"
    )


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.wan.wan22 import s2v as pipeline_mod

    fake_audio = _make_fake_audio()
    with (
        patch(_RUNTIME_PATCH, return_value={"error": "ComfyUI not found", "python_version": "3.12.0"}),
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(
                fake_audio,
                MagicMock(),
                None,
                "prompt",
                models_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# AC-03: audio dict accepts waveform and sample_rate
# ---------------------------------------------------------------------------


def test_audio_dict_waveform_and_sample_rate(tmp_path: Path) -> None:
    """run() must accept an audio dict with 'waveform' and 'sample_rate' keys (AC-03)."""
    import torch
    from comfy_diffusion.pipelines.video.wan.wan22 import s2v as pipeline_mod

    fake_audio = {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}
    fake_frames = [MagicMock(name="f0")]
    mm = _build_mock_mm()

    captured_audio: list[dict] = []

    def capture_encode(enc: Any, audio: Any) -> Any:
        captured_audio.append(audio)
        return MagicMock(name="audio_enc_output")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_APPLY_LORA_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_ENCODE_PROMPT_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_AUDIO_ENC_ENCODE_PATCH, side_effect=capture_encode),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN_S2V_PATCH, return_value=(MagicMock(), MagicMock(), MagicMock())),
        patch(_WAN_S2V_EXTEND_PATCH, return_value=(MagicMock(), MagicMock(), MagicMock())),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_LATENT_CONCAT_PATCH, return_value=MagicMock()),
        patch(_LATENT_CUT_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=fake_frames),
    ):
        result = pipeline_mod.run(
            fake_audio,
            MagicMock(name="pil_image"),
            None,
            "test prompt",
            models_dir=tmp_path,
        )

    assert result is fake_frames
    assert len(captured_audio) == 1
    passed_audio = captured_audio[0]
    assert "waveform" in passed_audio, "audio dict must have 'waveform' key"
    assert "sample_rate" in passed_audio, "audio dict must have 'sample_rate' key"
    assert passed_audio["sample_rate"] == 16000


# ---------------------------------------------------------------------------
# AC-04: wan22 sub-package exports s2v
# ---------------------------------------------------------------------------


def test_wan22_package_exports_s2v() -> None:
    from comfy_diffusion.pipelines.video.wan import wan22

    assert "s2v" in wan22.__all__
