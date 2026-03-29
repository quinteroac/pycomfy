"""Tests for US-007 — comfy_diffusion/pipelines/video/ltx/ltx23/ia2v.py pipeline.

Covers:
  AC01: manifest() returns 5 HFModelEntry items matching ltx23_i2v
  AC02: run() follows the two-pass workflow order with audio loading/encoding
  AC03: Samplers match the workflow — pass-1 uses euler_ancestral_cfg_pp,
        pass-2 uses euler_cfg_pp; manual sigmas from workflow; pass-2 uses
        fixed noise seed 42
  AC04: run() accepts prompt, negative_prompt, image, audio_path, seed, fps,
        width, height, length, cfg, audio_start_time, audio_duration,
        guide_strength_pass1, guide_strength_pass2, distilled_lora_strength,
        te_lora_strength, and filename-override params
  AC05: Pipeline file is at the correct path
  AC06: ltx23/__init__.py exports ia2v and has no "Not yet implemented" notice
  AC07: Typecheck / lint — file parses without syntax errors; no top-level
        comfy imports
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
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx23" / "ia2v.py"
)
_INIT_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx23" / "__init__.py"
)

# ---------------------------------------------------------------------------
# Patch target constants
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_LTXV_COND_PATCH = "comfy_diffusion.conditioning.ltxv_conditioning"
_LTXV_CROP_PATCH = "comfy_diffusion.conditioning.ltxv_crop_guides"
_EMPTY_VIDEO_PATCH = "comfy_diffusion.latent.ltxv_empty_latent_video"
_LORA_PATCH = "comfy_diffusion.lora.apply_lora"
_LOAD_AUDIO_PATCH = "comfy_diffusion.audio.load_audio"
_AUDIO_VAE_ENCODE_PATCH = "comfy_diffusion.audio.ltxv_audio_vae_encode"
_CONCAT_AV_PATCH = "comfy_diffusion.audio.ltxv_concat_av_latent"
_SEPARATE_AV_PATCH = "comfy_diffusion.audio.ltxv_separate_av_latent"
_AUDIO_VAE_DECODE_PATCH = "comfy_diffusion.audio.ltxv_audio_vae_decode"
_CFG_GUIDER_PATCH = "comfy_diffusion.sampling.cfg_guider"
_RANDOM_NOISE_PATCH = "comfy_diffusion.sampling.random_noise"
_MANUAL_SIGMAS_PATCH = "comfy_diffusion.sampling.manual_sigmas"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_SAMPLE_CUSTOM_PATCH = "comfy_diffusion.sampling.sample_custom"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode_batch_tiled"
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"
_LTXV_PREPROCESS_PATCH = "comfy_diffusion.image.ltxv_preprocess"
_IMG_TO_VIDEO_PATCH = "comfy_diffusion.video.ltxv_img_to_video_inplace"
_LATENT_UPSAMPLE_PATCH = "comfy_diffusion.latent.ltxv_latent_upsample"
_SET_NOISE_MASK_PATCH = "comfy_diffusion.latent.set_latent_noise_mask"
_SOLID_MASK_PATCH = "comfy_diffusion.mask.solid_mask"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_ltxv_audio_vae.return_value = MagicMock(name="audio_vae")
    mm.load_ltxav_text_encoder.return_value = MagicMock(name="clip")
    mm.load_latent_upscale_model.return_value = MagicMock(name="upscale_model")
    return mm


def _fake_audio_latent() -> dict[str, Any]:
    samples = MagicMock(name="audio_samples")
    samples.shape = (1, 8, 12, 16)
    return {"samples": samples, "sample_rate": 24000, "type": "audio"}


def _build_default_patches(mm: MagicMock) -> dict[str, Any]:
    """Return the default mock values used by _run_mocked."""
    fake_latent = _fake_audio_latent()
    return {
        _RUNTIME_PATCH: {"python_version": "3.12.0"},
        _MM_PATCH: mm,
        _ENCODE_PATCH: (MagicMock(name="pos"), MagicMock(name="neg")),
        _LTXV_COND_PATCH: (MagicMock(name="pos_cond"), MagicMock(name="neg_cond")),
        _LTXV_CROP_PATCH: (MagicMock(name="pos_crop"), MagicMock(name="neg_crop"), MagicMock()),
        _EMPTY_VIDEO_PATCH: {"samples": MagicMock()},
        _LORA_PATCH: (MagicMock(name="model_lora"), MagicMock(name="clip_lora")),
        _LOAD_AUDIO_PATCH: {"waveform": MagicMock(), "sample_rate": 24000},
        _AUDIO_VAE_ENCODE_PATCH: fake_latent,
        _CONCAT_AV_PATCH: {"samples": MagicMock()},
        _SEPARATE_AV_PATCH: (MagicMock(), MagicMock()),
        _AUDIO_VAE_DECODE_PATCH: {"waveform": MagicMock(), "sample_rate": 44100},
        _CFG_GUIDER_PATCH: MagicMock(),
        _RANDOM_NOISE_PATCH: MagicMock(),
        _MANUAL_SIGMAS_PATCH: MagicMock(),
        _GET_SAMPLER_PATCH: MagicMock(),
        _SAMPLE_CUSTOM_PATCH: (MagicMock(), MagicMock()),
        _VAE_DECODE_PATCH: [MagicMock()],
        _IMAGE_TO_TENSOR_PATCH: MagicMock(),
        _LTXV_PREPROCESS_PATCH: MagicMock(),
        _IMG_TO_VIDEO_PATCH: {"samples": MagicMock()},
        _LATENT_UPSAMPLE_PATCH: {"samples": MagicMock()},
        _SET_NOISE_MASK_PATCH: fake_latent,
        _SOLID_MASK_PATCH: MagicMock(name="mask"),
    }


def _apply_patches(patches: dict[str, Any]) -> contextlib.ExitStack:
    """Enter all patches into an ExitStack and return it (use as context manager)."""
    stack = contextlib.ExitStack()
    for target, value in patches.items():
        if callable(value) and not isinstance(value, MagicMock):
            stack.enter_context(patch(target, value))
        else:
            stack.enter_context(patch(target, return_value=value))
    return stack


def _run_mocked(
    tmp_path: Path,
    *,
    image: Any = None,
    audio_path: str | Path = "/fake/audio.mp3",
    prompt: str = "test prompt",
    overrides: dict[str, Any] | None = None,
    **run_kwargs: Any,
) -> tuple[dict[str, Any], MagicMock]:
    """Run the ia2v pipeline with all heavy dependencies mocked."""
    from comfy_diffusion.pipelines.video.ltx.ltx23 import ia2v as pipeline_mod

    if image is None:
        image = MagicMock(spec=["mode"])

    mm = _build_mock_mm()
    patches = _build_default_patches(mm)
    if overrides:
        patches.update(overrides)

    with _apply_patches(patches):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            image=image,
            audio_path=audio_path,
            prompt=prompt,
            **run_kwargs,
        )

    return result, mm


# ---------------------------------------------------------------------------
# AC05 / AC07 — file exists, parses, conventions
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    """AC05: Pipeline file must exist at the correct path."""
    assert _PIPELINE_FILE.is_file(), (
        "comfy_diffusion/pipelines/video/ltx/ltx23/ia2v.py must exist"
    )


def test_pipeline_parses_without_syntax_errors() -> None:
    """AC07: File must parse as valid Python."""
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
    assert docstring, "ia2v.py must have a module-level docstring"


def test_pipeline_exports_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '"manifest"' in source or "'manifest'" in source
    assert '"run"' in source or "'run'" in source


def test_no_top_level_comfy_imports() -> None:
    """AC07: No top-level comfy.* imports allowed — use lazy imports."""
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), (
                f"Top-level comfy import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


# ---------------------------------------------------------------------------
# AC06 — ltx23/__init__.py exports ia2v; no "Not yet implemented" notice
# ---------------------------------------------------------------------------


def test_init_exports_ia2v() -> None:
    """AC06: ltx23/__init__.py must list 'ia2v' in __all__."""
    source = _INIT_FILE.read_text(encoding="utf-8")
    assert '"ia2v"' in source or "'ia2v'" in source, (
        "ltx23/__init__.py must export 'ia2v'"
    )


def test_init_has_no_not_yet_implemented_notice() -> None:
    """AC06: The 'Not yet implemented' section must be removed."""
    source = _INIT_FILE.read_text(encoding="utf-8")
    assert "Not yet implemented" not in source, (
        "ltx23/__init__.py must not contain 'Not yet implemented'"
    )


# ---------------------------------------------------------------------------
# AC01 — manifest() returns 5 HFModelEntry items matching ltx23_i2v
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_manifest_returns_five_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 5, f"manifest() must return 5 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_matches_ltx23_i2v() -> None:
    """AC01: manifest() must return the same 5 entries as ltx23_i2v.manifest()."""
    from comfy_diffusion.pipelines.video.ltx.ltx23.i2v import manifest as i2v_manifest
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest as ia2v_manifest

    ia2v_entries = ia2v_manifest()
    i2v_entries = i2v_manifest()

    assert len(ia2v_entries) == len(i2v_entries), (
        f"ltx23_ia2v manifest has {len(ia2v_entries)} entries; "
        f"ltx23_i2v manifest has {len(i2v_entries)} entries — must match"
    )
    for ia2v_e, i2v_e in zip(ia2v_entries, i2v_entries):
        assert ia2v_e.repo_id == i2v_e.repo_id
        assert str(ia2v_e.dest) == str(i2v_e.dest)
        assert ia2v_e.filename == i2v_e.filename


def test_manifest_unet_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "ltx-2.3-22b-dev-fp8" in d for d in dests), (
        "manifest() must include diffusion_models/ltx-2.3-22b-dev-fp8"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "gemma_3_12B_it_fp4_mixed" in d for d in dests), (
        "manifest() must include text_encoders/gemma_3_12B_it_fp4_mixed"
    )


def test_manifest_includes_upscaler() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("upscale_models" in d and "spatial-upscaler" in d for d in dests), (
        "manifest() must include upscale_models/ltx-2.3-spatial-upscaler"
    )


def test_manifest_includes_both_loras() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest

    dests = [str(e.dest) for e in manifest()]
    lora_dests = [d for d in dests if "loras" in d]
    assert len(lora_dests) == 2, (
        f"manifest() must include exactly 2 LoRA entries, got {lora_dests}"
    )


# ---------------------------------------------------------------------------
# AC04 — run() parameter signature
# ---------------------------------------------------------------------------


def test_run_has_audio_path_parameter() -> None:
    """AC04: run() must accept audio_path."""
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import run

    sig = inspect.signature(run)
    assert "audio_path" in sig.parameters, "run() must have an 'audio_path' parameter"


def test_run_has_image_parameter() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import run

    sig = inspect.signature(run)
    assert "image" in sig.parameters, "run() must have an 'image' parameter"


def test_run_has_prompt_and_negative_prompt() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import run

    sig = inspect.signature(run)
    assert "prompt" in sig.parameters
    assert "negative_prompt" in sig.parameters


def test_run_has_seed_parameter_with_default_zero() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import run

    sig = inspect.signature(run)
    assert "seed" in sig.parameters
    assert sig.parameters["seed"].default == 0


def test_run_has_fps_parameter_with_default_24() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import run

    sig = inspect.signature(run)
    assert "fps" in sig.parameters
    assert sig.parameters["fps"].default == 24


def test_run_has_audio_trim_parameters() -> None:
    """AC04: run() must accept audio_start_time and audio_duration."""
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import run

    sig = inspect.signature(run)
    assert "audio_start_time" in sig.parameters
    assert sig.parameters["audio_start_time"].default == 0.0
    assert "audio_duration" in sig.parameters
    assert sig.parameters["audio_duration"].default is None


def test_run_has_guide_strength_parameters() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import run

    sig = inspect.signature(run)
    assert "guide_strength_pass1" in sig.parameters
    assert sig.parameters["guide_strength_pass1"].default == 0.7
    assert "guide_strength_pass2" in sig.parameters
    assert sig.parameters["guide_strength_pass2"].default == 1.0


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import run

    sig = inspect.signature(run)
    for param in (
        "unet_filename",
        "vae_filename",
        "audio_vae_filename",
        "text_encoder_filename",
        "distilled_lora_filename",
        "te_lora_filename",
        "upscaler_filename",
    ):
        assert param in sig.parameters, f"run() must have a '{param}' parameter"


# ---------------------------------------------------------------------------
# AC02 / AC03 — CPU tests with mocked inputs
# ---------------------------------------------------------------------------


def test_run_returns_dict_with_frames_and_audio(tmp_path: Path) -> None:
    """run() must return a dict with 'frames' and 'audio' keys."""
    result, _ = _run_mocked(tmp_path)
    assert isinstance(result, dict), "run() must return a dict"
    assert "frames" in result, "result must have a 'frames' key"
    assert "audio" in result, "result must have an 'audio' key"
    assert isinstance(result["frames"], list)
    assert isinstance(result["audio"], dict)


def test_run_calls_load_audio(tmp_path: Path) -> None:
    """AC02: load_audio must be called with the provided audio_path."""
    load_audio_calls: list[Any] = []

    def fake_load_audio(path: Any, **kwargs: Any) -> dict[str, Any]:
        load_audio_calls.append(path)
        return {"waveform": MagicMock(), "sample_rate": 24000}

    _run_mocked(tmp_path, overrides={_LOAD_AUDIO_PATCH: fake_load_audio})

    assert len(load_audio_calls) == 1, "load_audio() must be called exactly once"


def test_run_calls_ltxv_audio_vae_encode(tmp_path: Path) -> None:
    """AC02: ltxv_audio_vae_encode must be called to encode the input audio."""
    encode_calls: list[Any] = []
    fake_latent = _fake_audio_latent()

    def fake_encode(vae: Any, audio: Any) -> dict[str, Any]:
        encode_calls.append((vae, audio))
        return fake_latent

    _run_mocked(
        tmp_path,
        overrides={
            _AUDIO_VAE_ENCODE_PATCH: fake_encode,
            _SET_NOISE_MASK_PATCH: fake_latent,
        },
    )

    assert len(encode_calls) == 1, "ltxv_audio_vae_encode() must be called exactly once"


def test_run_calls_set_latent_noise_mask(tmp_path: Path) -> None:
    """AC02: set_latent_noise_mask must be called on the encoded audio latent."""
    mask_calls: list[Any] = []
    fake_latent = _fake_audio_latent()

    def fake_set_mask(latent: Any, mask: Any) -> dict[str, Any]:
        mask_calls.append((latent, mask))
        return fake_latent

    _run_mocked(tmp_path, overrides={_SET_NOISE_MASK_PATCH: fake_set_mask})

    assert len(mask_calls) == 1, "set_latent_noise_mask() must be called exactly once"


def test_run_calls_sample_custom_twice(tmp_path: Path) -> None:
    """AC02: sample_custom() must be called twice (pass 1 and pass 2)."""
    sample_calls: list[Any] = []

    def fake_sample(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        sample_calls.append(args)
        return MagicMock(), MagicMock()

    _run_mocked(tmp_path, overrides={_SAMPLE_CUSTOM_PATCH: fake_sample})

    assert len(sample_calls) == 2, (
        f"sample_custom() must be called exactly twice (pass 1 + pass 2), got {len(sample_calls)}"
    )


def test_run_uses_correct_samplers(tmp_path: Path) -> None:
    """AC03: Pass-1 uses euler_ancestral_cfg_pp; pass-2 uses euler_cfg_pp."""
    sampler_calls: list[str] = []

    def fake_get_sampler(name: str) -> MagicMock:
        sampler_calls.append(name)
        return MagicMock(name=f"sampler_{name}")

    _run_mocked(tmp_path, overrides={_GET_SAMPLER_PATCH: fake_get_sampler})

    assert len(sampler_calls) == 2, f"get_sampler must be called twice, got {sampler_calls}"
    assert sampler_calls[0] == "euler_ancestral_cfg_pp", (
        f"Pass-1 sampler must be 'euler_ancestral_cfg_pp', got {sampler_calls[0]!r}"
    )
    assert sampler_calls[1] == "euler_cfg_pp", (
        f"Pass-2 sampler must be 'euler_cfg_pp', got {sampler_calls[1]!r}"
    )


def test_run_uses_fixed_noise_seed_for_pass2(tmp_path: Path) -> None:
    """AC03: Pass-2 noise must use fixed seed 42 (matching reference workflow node 285)."""
    noise_seeds: list[int] = []

    def fake_random_noise(seed: int) -> MagicMock:
        noise_seeds.append(seed)
        return MagicMock(name=f"noise_{seed}")

    _run_mocked(
        tmp_path,
        seed=99,
        overrides={_RANDOM_NOISE_PATCH: fake_random_noise},
    )

    assert len(noise_seeds) == 2, f"random_noise must be called twice, got seeds {noise_seeds}"
    assert noise_seeds[0] == 99, (
        f"Pass-1 seed must be the user-provided seed 99, got {noise_seeds[0]}"
    )
    assert noise_seeds[1] == 42, (
        f"Pass-2 seed must be the fixed seed 42, got {noise_seeds[1]}"
    )


def test_run_uses_correct_sigmas(tmp_path: Path) -> None:
    """AC03: Manual sigma strings must match the reference workflow exactly."""
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import _SIGMAS_PASS1, _SIGMAS_PASS2

    sigma_calls: list[str] = []

    def fake_manual_sigmas(sigmas: str) -> MagicMock:
        sigma_calls.append(sigmas)
        return MagicMock(name=f"sigmas_{sigmas[:10]}")

    _run_mocked(tmp_path, overrides={_MANUAL_SIGMAS_PATCH: fake_manual_sigmas})

    assert len(sigma_calls) == 2, f"manual_sigmas must be called twice, got {sigma_calls}"
    assert sigma_calls[0] == _SIGMAS_PASS1, (
        f"Pass-1 sigmas must be '{_SIGMAS_PASS1}', got '{sigma_calls[0]}'"
    )
    assert sigma_calls[1] == _SIGMAS_PASS2, (
        f"Pass-2 sigmas must be '{_SIGMAS_PASS2}', got '{sigma_calls[1]}'"
    )


def test_run_calls_ltxv_latent_upsample(tmp_path: Path) -> None:
    """AC02: ltxv_latent_upsample must be called for spatial upscaling."""
    upsample_calls: list[Any] = []

    def fake_upsample(latent: Any, **kwargs: Any) -> dict[str, Any]:
        upsample_calls.append((latent, kwargs))
        return {"samples": MagicMock()}

    _run_mocked(tmp_path, overrides={_LATENT_UPSAMPLE_PATCH: fake_upsample})

    assert len(upsample_calls) == 1, "ltxv_latent_upsample must be called exactly once"


def test_run_calls_ltxv_crop_guides(tmp_path: Path) -> None:
    """AC02: ltxv_crop_guides must be called (after pass 1, workflow order)."""
    crop_calls: list[Any] = []

    def fake_crop_guides(positive: Any, negative: Any, latent: Any) -> tuple[Any, Any, Any]:
        crop_calls.append((positive, negative, latent))
        return MagicMock(), MagicMock(), MagicMock()

    _run_mocked(tmp_path, overrides={_LTXV_CROP_PATCH: fake_crop_guides})

    assert len(crop_calls) == 1, (
        f"ltxv_crop_guides must be called exactly once, got {len(crop_calls)}"
    )


def test_run_uses_load_ltxav_text_encoder(tmp_path: Path) -> None:
    """Gemma 3 text encoder must be loaded via load_ltxav_text_encoder."""
    _, mm = _run_mocked(tmp_path)
    mm.load_ltxav_text_encoder.assert_called_once()
    mm.load_clip.assert_not_called()


def test_run_uses_image_to_tensor_for_pil_image(tmp_path: Path) -> None:
    """PIL Image (has .mode attr) must be converted via image_to_tensor."""
    pil_image = MagicMock(spec=["mode"])
    image_to_tensor_calls: list[Any] = []

    def fake_image_to_tensor(img: Any) -> Any:
        image_to_tensor_calls.append(img)
        return MagicMock()

    _run_mocked(
        tmp_path,
        image=pil_image,
        overrides={_IMAGE_TO_TENSOR_PATCH: fake_image_to_tensor},
    )

    assert len(image_to_tensor_calls) == 1
    assert image_to_tensor_calls[0] is pil_image


def test_run_loads_latent_upscale_model(tmp_path: Path) -> None:
    """AC02: ModelManager.load_latent_upscale_model must be called for the upscaler."""
    _, mm = _run_mocked(tmp_path)
    mm.load_latent_upscale_model.assert_called_once()


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    """run() must raise RuntimeError when check_runtime() returns an error."""
    from comfy_diffusion.pipelines.video.ltx.ltx23 import ia2v as pipeline_mod

    with patch(
        _RUNTIME_PATCH,
        return_value={
            "error": "ComfyUI submodule not initialized",
            "python_version": "3.12.0",
        },
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(
                models_dir=Path("/tmp"),
                image=MagicMock(spec=["mode"]),
                audio_path="/fake/audio.mp3",
                prompt="test",
            )


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    """download_models(manifest()) completes without error when all files exist."""
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest

    entries = manifest()
    assert len(entries) == 5

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
