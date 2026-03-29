"""Tests for US-004 — comfy_diffusion/pipelines/ltx2/audio_to_video pipeline.

Covers:
  AC01: manifest() returns exactly 5 HFModelEntry items with correct dest paths
  AC02: run() accepts the required keyword arguments
  AC03: run() returns a dict with "video" and "frames" keys
  AC04: __all__ = ["manifest", "run"]
  AC05: ltx2/__init__.py exports "audio_to_video"
  AC06: Typecheck / lint — file parses without syntax errors; no top-level comfy imports
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx2" / "audio_to_video.py"
)
_INIT_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "video" / "ltx" / "ltx2" / "__init__.py"
)


# ---------------------------------------------------------------------------
# AC06 — file parses without syntax errors; conventions
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "audio_to_video.py must exist"


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
    assert docstring, "audio_to_video.py must have a module-level docstring"


def test_no_top_level_comfy_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), (
                f"Top-level comfy import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


def test_no_top_level_comfy_diffusion_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    lines = source.splitlines()
    in_function = False
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Track function scope (rough check: any indented import is inside a function)
        if stripped.startswith("def ") or stripped.startswith("class "):
            in_function = True
        if (
            (stripped.startswith("from comfy_diffusion.") or stripped.startswith("import comfy_diffusion."))
            and not stripped.startswith("from comfy_diffusion.downloader import")
        ):
            assert line.startswith("    "), (
                f"Top-level comfy_diffusion import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


# ---------------------------------------------------------------------------
# AC04 — __all__ = ["manifest", "run"]
# ---------------------------------------------------------------------------


def test_pipeline_exports_manifest_and_run_in_all() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '"manifest"' in source or "'manifest'" in source
    assert '"run"' in source or "'run'" in source
    # Check __all__ specifically
    assert '__all__' in source
    assert '"manifest"' in source
    assert '"run"' in source


def test_all_contains_only_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import __all__ as pipeline_all

    assert set(pipeline_all) == {"manifest", "run"}, (
        f"__all__ must be exactly ['manifest', 'run'], got {pipeline_all!r}"
    )


# ---------------------------------------------------------------------------
# AC05 — ltx2/__init__.py exports "audio_to_video"
# ---------------------------------------------------------------------------


def test_ltx2_init_exports_audio_to_video() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2 import __all__ as ltx2_all

    assert "audio_to_video" in ltx2_all, (
        f"ltx2/__init__.py __all__ must include 'audio_to_video', got {ltx2_all!r}"
    )


def test_ltx2_init_file_contains_audio_to_video() -> None:
    source = _INIT_FILE.read_text(encoding="utf-8")
    assert "audio_to_video" in source


# ---------------------------------------------------------------------------
# AC01 — manifest() returns exactly 5 HFModelEntry items
# ---------------------------------------------------------------------------


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


def test_manifest_returns_exactly_five_entries() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 5, f"manifest() must return exactly 5 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), f"Expected HFModelEntry, got {type(entry)}"


def test_manifest_unet_dest() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any("diffusion_models" in d and "ltx-2-19b-distilled_transformer_only_bf16" in d for d in dests), (
        f"Expected UNet in diffusion_models, got dests: {dests}"
    )


def test_manifest_text_encoder_dests() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any("gemma_3_12B_it_fp8_scaled" in d for d in dests), (
        f"Expected gemma_3_12B_it_fp8_scaled in dests: {dests}"
    )
    assert any("ltx-2-19b-embeddings_connector_distill_bf16" in d for d in dests), (
        f"Expected ltx-2-19b-embeddings_connector_distill_bf16 in dests: {dests}"
    )


def test_manifest_audio_vae_dest() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any("vae" in d and "LTX2_audio_vae_bf16" in d for d in dests), (
        f"Expected LTX2_audio_vae_bf16 in vae/, got dests: {dests}"
    )


def test_manifest_video_vae_dest() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import manifest

    entries = manifest()
    dests = [str(e.dest) for e in entries]
    assert any("vae" in d and "LTX2_video_vae_bf16" in d for d in dests), (
        f"Expected LTX2_video_vae_bf16 in vae/, got dests: {dests}"
    )


def test_manifest_all_from_lightricks_repo() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import manifest

    for entry in manifest():
        assert entry.repo_id == "Lightricks/LTX-Video", (
            f"Expected repo_id='Lightricks/LTX-Video', got {entry.repo_id!r}"
        )


# ---------------------------------------------------------------------------
# AC02 — run() accepts the required keyword arguments
# ---------------------------------------------------------------------------


def test_run_has_correct_signature() -> None:
    import inspect

    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import run

    sig = inspect.signature(run)
    required = {
        "models_dir", "prompt", "image_path", "audio_path",
        "audio_start_time", "audio_end_time",
        "width", "height", "length", "fps", "steps", "cfg", "seed",
    }
    for param_name in required:
        assert param_name in sig.parameters, (
            f"run() is missing required parameter: {param_name!r}"
        )


def test_run_all_params_are_keyword_only() -> None:
    import inspect

    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import run

    sig = inspect.signature(run)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        assert param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        ), f"Parameter {name!r} must be keyword-only"


# ---------------------------------------------------------------------------
# AC03 — run() returns dict with "video" and "frames" keys (CPU mock test)
# ---------------------------------------------------------------------------


def _build_mock_tensor(shape: tuple[int, ...]) -> Any:
    """Return a torch.zeros tensor of the given shape."""
    import torch

    return torch.zeros(shape)


def test_run_returns_video_and_frames_keys() -> None:
    """CPU-safe smoke test: mock all heavy ComfyUI/model calls, verify return shape."""
    import contextlib
    import tempfile

    import torch
    from PIL import Image as PILImage
    from unittest.mock import MagicMock, patch

    fake_pil = PILImage.new("RGB", (64, 64), color=(128, 128, 128))
    fake_frames_tensor = torch.zeros(4, 64, 64, 3)
    fake_video_obj = object()
    fake_audio_dict = {"waveform": torch.zeros(1, 2, 1000), "sample_rate": 44100}
    fake_latent: dict[str, Any] = {"samples": torch.zeros(1, 16, 4, 8, 8)}
    fake_audio_latent: dict[str, Any] = {"samples": torch.zeros(1, 8, 4)}
    fake_conditioning: list[Any] = [[torch.zeros(1, 4, 64), {}]]
    fake_sigmas = torch.linspace(1.0, 0.0, 9)

    # Patch at source module level (lazy imports inside run() use the source binding)
    patches: dict[str, Any] = {
        "comfy_diffusion.runtime.check_runtime": MagicMock(return_value={}),
        "comfy_diffusion.audio.load_audio": MagicMock(return_value=fake_audio_dict),
        "comfy_diffusion.audio.audio_crop": MagicMock(return_value=fake_audio_dict),
        "comfy_diffusion.audio.audio_separation": MagicMock(return_value=fake_audio_dict),
        "comfy_diffusion.audio.trim_audio_duration": MagicMock(return_value=fake_audio_dict),
        "comfy_diffusion.audio.ltxv_empty_latent_audio": MagicMock(return_value=dict(fake_audio_latent)),
        "comfy_diffusion.audio.ltxv_audio_vae_encode": MagicMock(return_value={"samples": fake_audio_latent["samples"]}),
        "comfy_diffusion.audio.ltxv_audio_video_mask": MagicMock(return_value=(dict(fake_latent), dict(fake_audio_latent))),
        "comfy_diffusion.audio.ltxv_concat_av_latent": MagicMock(return_value=dict(fake_latent)),
        "comfy_diffusion.audio.ltxv_separate_av_latent": MagicMock(return_value=(dict(fake_latent), dict(fake_audio_latent))),
        "comfy_diffusion.image.load_image": MagicMock(return_value=(fake_frames_tensor[:1], None)),
        "comfy_diffusion.image.image_to_tensor": MagicMock(return_value=fake_frames_tensor[:1]),
        "comfy_diffusion.image.image_resize_kj": MagicMock(return_value=(fake_frames_tensor[:1], 64, 64)),
        "comfy_diffusion.image.ltxv_preprocess": MagicMock(return_value=fake_frames_tensor[:1]),
        "comfy_diffusion.image.image_from_batch": MagicMock(return_value=fake_frames_tensor[:1]),
        "comfy_diffusion.image.image_batch_extend_with_overlap": MagicMock(return_value=fake_frames_tensor),
        "comfy_diffusion.conditioning.encode_prompt": MagicMock(return_value=fake_conditioning),
        "comfy_diffusion.conditioning.conditioning_zero_out": MagicMock(return_value=fake_conditioning),
        "comfy_diffusion.conditioning.ltxv_conditioning": MagicMock(return_value=(fake_conditioning, fake_conditioning)),
        "comfy_diffusion.video.ltxv_chunk_feed_forward": MagicMock(side_effect=lambda m, **kw: m),
        "comfy_diffusion.video.ltx2_nag": MagicMock(side_effect=lambda m, **kw: m),
        "comfy_diffusion.video.ltx2_sampling_preview_override": MagicMock(side_effect=lambda m, **kw: m),
        "comfy_diffusion.video.ltxv_img_to_video_inplace_kj": MagicMock(return_value=dict(fake_latent)),
        "comfy_diffusion.video.create_video": MagicMock(return_value=fake_video_obj),
        "comfy_diffusion.sampling.get_sampler": MagicMock(return_value=MagicMock()),
        "comfy_diffusion.sampling.cfg_guider": MagicMock(return_value=MagicMock()),
        "comfy_diffusion.sampling.random_noise": MagicMock(return_value=MagicMock()),
        "comfy_diffusion.sampling.ltxv_scheduler": MagicMock(return_value=fake_sigmas),
        "comfy_diffusion.sampling.sample_custom": MagicMock(return_value=(dict(fake_latent), dict(fake_latent))),
        "comfy_diffusion.latent.ltxv_empty_latent_video": MagicMock(return_value=dict(fake_latent)),
        "comfy_diffusion.vae.vae_decode_batch_tiled": MagicMock(return_value=fake_frames_tensor),
    }

    # Configure ModelManager mock separately (needs method returns)
    fake_model = MagicMock()
    fake_clip = MagicMock()
    fake_vae = MagicMock()
    fake_mm = MagicMock()
    fake_mm.load_unet.return_value = fake_model
    fake_mm.load_ltxav_text_encoder.return_value = fake_clip
    fake_mm.load_vae_kj.return_value = fake_vae
    patches["comfy_diffusion.models.ModelManager"] = MagicMock(return_value=fake_mm)

    with contextlib.ExitStack() as stack:
        for key, val in patches.items():
            stack.enter_context(patch(key, new=val))

        with tempfile.TemporaryDirectory() as tmpdir:
            from comfy_diffusion.pipelines.video.ltx.ltx2 import audio_to_video as _atv_mod
            import importlib
            importlib.reload(_atv_mod)

            result = _atv_mod.run(
                models_dir=tmpdir,
                prompt="singing man",
                image_path=fake_pil,
                audio_path="/fake/audio.mp3",
                audio_start_time=0.0,
                audio_end_time=10.0,
                num_extensions=0,  # skip extension passes for speed
            )

    assert isinstance(result, dict), "run() must return a dict"
    assert "video" in result, "result must have 'video' key"
    assert "frames" in result, "result must have 'frames' key"
    assert isinstance(result["frames"], list), "result['frames'] must be a list"
    assert result["video"] is fake_video_obj


def test_run_calls_check_runtime() -> None:
    """run() must call check_runtime() and raise on error."""
    import tempfile

    from comfy_diffusion.pipelines.video.ltx.ltx2.audio_to_video import run

    with patch(
        "comfy_diffusion.runtime.check_runtime",
        return_value={"error": "ComfyUI not initialised"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            with tempfile.TemporaryDirectory() as tmpdir:
                run(
                    models_dir=tmpdir,
                    prompt="test",
                    image_path="/fake/image.png",
                    audio_path="/fake/audio.mp3",
                )
