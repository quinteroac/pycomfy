"""Tests for comfy_diffusion/pipelines/image/sdxl/t2i_refiner_prompt.py.

Covers:
  - AC01: file exists, parses, has future annotations, module docstring
  - AC02: manifest() is identical to t2i.py (same 2 checkpoints)
  - AC03: run() accepts refiner_prompt and refiner_negative_prompt (both optional)
  - AC04: base stage uses base CLIP with prompt; refiner stage uses refiner CLIP with refiner_prompt
  - AC05: two-pass KSamplerAdvanced flow identical to t2i.py
  - AC06: __all__ = ["manifest", "run"], lazy imports, module docstring
  - AC07: typecheck / lint (no top-level comfy/torch imports, from __future__ import annotations)
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
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "sdxl" / "t2i_refiner_prompt.py"
)

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.empty_latent_image"
_SAMPLE_ADVANCED_PATCH = "comfy_diffusion.sampling.sample_advanced"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"


# ---------------------------------------------------------------------------
# AC01 / AC06 / AC07 — file-level checks
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "t2i_refiner_prompt.py must exist"


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
    assert docstring, "t2i_refiner_prompt.py must have a module-level docstring"


def test_pipeline_has_dunder_all_with_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert "__all__" in source
    assert '"manifest"' in source or "'manifest'" in source
    assert '"run"' in source or "'run'" in source


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as mod

    assert hasattr(mod, "__all__")
    assert set(mod.__all__) == {"manifest", "run"}


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
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# AC02 — manifest() identical to t2i.py
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_two_entries() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 2


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry)


def test_manifest_base_dest_path() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("checkpoints" in d and "sd_xl_base_1.0" in d for d in dests)


def test_manifest_refiner_dest_path() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("checkpoints" in d and "sd_xl_refiner_1.0" in d for d in dests)


def test_manifest_base_uses_stabilityai_repo() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    base_entry = next(e for e in manifest() if "base" in str(e.dest))
    assert "stabilityai" in base_entry.repo_id


def test_manifest_refiner_uses_stabilityai_repo() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    refiner_entry = next(e for e in manifest() if "refiner" in str(e.dest))
    assert "stabilityai" in refiner_entry.repo_id


def test_manifest_identical_to_t2i() -> None:
    """AC02: manifest() must return the same 2 checkpoints as t2i.manifest()."""
    from comfy_diffusion.pipelines.image.sdxl import t2i, t2i_refiner_prompt

    t2i_entries = t2i.manifest()
    refiner_entries = t2i_refiner_prompt.manifest()

    assert len(refiner_entries) == len(t2i_entries)
    for a, b in zip(t2i_entries, refiner_entries):
        assert type(a) is type(b)
        assert a.dest == b.dest
        assert a.repo_id == b.repo_id
        assert a.filename == b.filename


# ---------------------------------------------------------------------------
# AC03 — run() signature
# ---------------------------------------------------------------------------


def test_run_has_refiner_prompt_param() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    assert "refiner_prompt" in sig.parameters


def test_run_refiner_prompt_defaults_to_none() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    assert sig.parameters["refiner_prompt"].default is None


def test_run_has_refiner_negative_prompt_param() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    assert "refiner_negative_prompt" in sig.parameters


def test_run_refiner_negative_prompt_defaults_to_none() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    assert sig.parameters["refiner_negative_prompt"].default is None


def test_run_signature_includes_base_params() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    required = {
        "models_dir", "prompt", "negative_prompt", "width", "height",
        "steps", "base_end_step", "cfg", "seed",
    }
    assert required <= set(sig.parameters.keys())


def test_run_default_width_height() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 1024
    assert sig.parameters["height"].default == 1024


def test_run_default_steps_and_base_end_step() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 25
    assert sig.parameters["base_end_step"].default == 20


def test_run_has_filename_override_params() -> None:
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import run

    sig = inspect.signature(run)
    for param in ("base_filename", "refiner_filename"):
        assert param in sig.parameters


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    base_result = MagicMock(
        model=MagicMock(name="base_model"),
        clip=MagicMock(name="base_clip"),
        vae=MagicMock(name="base_vae"),
    )
    refiner_result = MagicMock(
        model=MagicMock(name="refiner_model"),
        clip=MagicMock(name="refiner_clip"),
        vae=MagicMock(name="refiner_vae"),
    )
    mm.load_checkpoint.side_effect = [base_result, refiner_result]
    return mm


def _run_pipeline(
    tmp_path: Path,
    *,
    call_order: list[str] | None = None,
    encode_calls: list[tuple[Any, ...]] | None = None,
    sample_calls: list[dict[str, Any]] | None = None,
    **run_kwargs: Any,
) -> list[Any]:
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as mod

    fake_latent = MagicMock(name="latent")
    fake_latent_p1 = MagicMock(name="latent_pass1")
    fake_latent_p2 = MagicMock(name="latent_pass2")
    fake_image = MagicMock(name="image")
    mm = _build_mock_mm()

    encode_call_count: list[int] = [0]
    sample_call_count: list[int] = [0]

    def _fake_encode(clip: Any, pos: Any, neg: Any) -> tuple[MagicMock, MagicMock]:
        encode_call_count[0] += 1
        if encode_calls is not None:
            encode_calls.append((clip, pos, neg))
        return MagicMock(), MagicMock()

    def _fake_sample(*args: Any, **kwargs: Any) -> Any:
        sample_call_count[0] += 1
        n = sample_call_count[0]
        if call_order is not None:
            call_order.append(f"sample_advanced_pass{n}")
        if sample_calls is not None:
            sample_calls.append(kwargs)
        return fake_latent_p1 if n == 1 else fake_latent_p2

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, side_effect=_fake_encode),
        patch(_EMPTY_LATENT_PATCH, side_effect=lambda w, h, **kw: (
            call_order.append("empty_latent_image") or fake_latent
            if call_order is not None
            else fake_latent
        )),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=_fake_sample),
        patch(_VAE_DECODE_PATCH, side_effect=lambda vae, latent: (
            call_order.append("vae_decode") or fake_image
            if call_order is not None
            else fake_image
        )),
    ):
        return mod.run(models_dir=tmp_path, **run_kwargs)


# ---------------------------------------------------------------------------
# AC04 — base encodes prompt; refiner encodes refiner_prompt
# ---------------------------------------------------------------------------


def test_base_clip_encodes_base_prompt(tmp_path: Path) -> None:
    """Base stage must encode the base prompt with the base CLIP."""
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as mod

    mm2 = MagicMock()
    base_r = MagicMock(model=MagicMock(), clip=MagicMock(name="base_clip"), vae=MagicMock())
    refiner_r = MagicMock(
        model=MagicMock(), clip=MagicMock(name="refiner_clip"), vae=MagicMock()
    )
    mm2.load_checkpoint.side_effect = [base_r, refiner_r]

    actual_encode_calls: list[tuple[Any, ...]] = []

    def _cap_encode(clip: Any, pos: Any, neg: Any) -> tuple[MagicMock, MagicMock]:
        actual_encode_calls.append((clip, pos, neg))
        return MagicMock(), MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm2),
        patch(_ENCODE_PATCH, side_effect=_cap_encode),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(
            models_dir=tmp_path,
            prompt="base prompt",
            negative_prompt="base neg",
            refiner_prompt="refiner prompt",
            refiner_negative_prompt="refiner neg",
        )

    # First encode call: base CLIP with base prompts
    assert actual_encode_calls[0][0] is base_r.clip
    assert actual_encode_calls[0][1] == "base prompt"
    assert actual_encode_calls[0][2] == "base neg"


def test_refiner_clip_encodes_refiner_prompt(tmp_path: Path) -> None:
    """Refiner stage must encode refiner_prompt with the refiner CLIP."""
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as mod

    mm2 = MagicMock()
    base_r = MagicMock(model=MagicMock(), clip=MagicMock(name="base_clip"), vae=MagicMock())
    refiner_r = MagicMock(
        model=MagicMock(), clip=MagicMock(name="refiner_clip"), vae=MagicMock()
    )
    mm2.load_checkpoint.side_effect = [base_r, refiner_r]

    actual_encode_calls: list[tuple[Any, ...]] = []

    def _cap_encode(clip: Any, pos: Any, neg: Any) -> tuple[MagicMock, MagicMock]:
        actual_encode_calls.append((clip, pos, neg))
        return MagicMock(), MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm2),
        patch(_ENCODE_PATCH, side_effect=_cap_encode),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(
            models_dir=tmp_path,
            prompt="base prompt",
            negative_prompt="base neg",
            refiner_prompt="refiner prompt",
            refiner_negative_prompt="refiner neg",
        )

    # Second encode call: refiner CLIP with refiner prompts
    assert actual_encode_calls[1][0] is refiner_r.clip
    assert actual_encode_calls[1][1] == "refiner prompt"
    assert actual_encode_calls[1][2] == "refiner neg"


def test_refiner_prompt_defaults_to_base_prompt(tmp_path: Path) -> None:
    """When refiner_prompt is None, encode_prompt for refiner must use base prompt."""
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as mod

    mm2 = MagicMock()
    base_r = MagicMock(model=MagicMock(), clip=MagicMock(name="base_clip"), vae=MagicMock())
    refiner_r = MagicMock(
        model=MagicMock(), clip=MagicMock(name="refiner_clip"), vae=MagicMock()
    )
    mm2.load_checkpoint.side_effect = [base_r, refiner_r]

    actual_encode_calls: list[tuple[Any, ...]] = []

    def _cap_encode(clip: Any, pos: Any, neg: Any) -> tuple[MagicMock, MagicMock]:
        actual_encode_calls.append((clip, pos, neg))
        return MagicMock(), MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm2),
        patch(_ENCODE_PATCH, side_effect=_cap_encode),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="only prompt", negative_prompt="only neg")

    # refiner_prompt=None → refiner encode must use "only prompt"
    assert actual_encode_calls[1][1] == "only prompt"
    # refiner_negative_prompt=None → refiner encode must use "only neg"
    assert actual_encode_calls[1][2] == "only neg"


def test_refiner_negative_prompt_defaults_to_negative_prompt(tmp_path: Path) -> None:
    """When refiner_negative_prompt is None, refiner uses negative_prompt."""
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as mod

    mm2 = MagicMock()
    base_r = MagicMock(model=MagicMock(), clip=MagicMock(), vae=MagicMock())
    refiner_r = MagicMock(model=MagicMock(), clip=MagicMock(), vae=MagicMock())
    mm2.load_checkpoint.side_effect = [base_r, refiner_r]

    actual_encode_calls: list[tuple[Any, ...]] = []

    def _cap_encode(clip: Any, pos: Any, neg: Any) -> tuple[MagicMock, MagicMock]:
        actual_encode_calls.append((clip, pos, neg))
        return MagicMock(), MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm2),
        patch(_ENCODE_PATCH, side_effect=_cap_encode),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(
            models_dir=tmp_path,
            prompt="prompt",
            negative_prompt="custom negative",
            refiner_prompt="refiner specific",
        )

    # refiner_negative_prompt=None → falls back to negative_prompt="custom negative"
    assert actual_encode_calls[1][2] == "custom negative"


# ---------------------------------------------------------------------------
# AC05 — two-pass KSamplerAdvanced flow identical to t2i.py
# ---------------------------------------------------------------------------


def test_run_returns_list_of_images(tmp_path: Path) -> None:
    result = _run_pipeline(tmp_path, prompt="test")
    assert isinstance(result, list)
    assert len(result) == 1


def test_run_calls_sample_advanced_twice(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_pipeline(tmp_path, prompt="test", call_order=call_order)
    sample_calls = [c for c in call_order if "sample_advanced" in c]
    assert len(sample_calls) == 2


def test_run_full_call_order(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_pipeline(tmp_path, prompt="test", call_order=call_order)
    expected = [
        "empty_latent_image",
        "sample_advanced_pass1",
        "sample_advanced_pass2",
        "vae_decode",
    ]
    assert call_order == expected


def test_run_pass1_add_noise_and_leftover_noise(tmp_path: Path) -> None:
    sample_calls: list[dict[str, Any]] = []
    _run_pipeline(tmp_path, prompt="test", sample_calls=sample_calls)
    assert sample_calls[0].get("add_noise") is True
    assert sample_calls[0].get("return_with_leftover_noise") is True
    assert sample_calls[0].get("start_at_step") == 0


def test_run_pass1_end_at_step_equals_base_end_step(tmp_path: Path) -> None:
    sample_calls: list[dict[str, Any]] = []
    _run_pipeline(tmp_path, prompt="test", base_end_step=15, sample_calls=sample_calls)
    assert sample_calls[0].get("end_at_step") == 15


def test_run_pass2_no_noise_full_denoise(tmp_path: Path) -> None:
    sample_calls: list[dict[str, Any]] = []
    _run_pipeline(tmp_path, prompt="test", sample_calls=sample_calls)
    assert sample_calls[1].get("add_noise") is False
    assert sample_calls[1].get("return_with_leftover_noise") is False
    assert sample_calls[1].get("end_at_step") == 10000


def test_run_pass2_start_at_step_equals_base_end_step(tmp_path: Path) -> None:
    sample_calls: list[dict[str, Any]] = []
    _run_pipeline(tmp_path, prompt="test", base_end_step=18, sample_calls=sample_calls)
    assert sample_calls[1].get("start_at_step") == 18


def test_run_uses_refiner_vae_for_decode(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as mod

    refiner_vae = MagicMock(name="refiner_vae")
    mm2 = MagicMock()
    mm2.load_checkpoint.side_effect = [
        MagicMock(model=MagicMock(), clip=MagicMock(), vae=MagicMock(name="base_vae")),
        MagicMock(model=MagicMock(), clip=MagicMock(), vae=refiner_vae),
    ]

    decode_vae_arg: list[Any] = []

    def _capture_decode(vae: Any, latent: Any) -> MagicMock:
        decode_vae_arg.append(vae)
        return MagicMock()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm2),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, side_effect=_capture_decode),
    ):
        mod.run(models_dir=tmp_path, prompt="test")

    assert decode_vae_arg[0] is refiner_vae


def test_run_loads_two_checkpoints(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as mod

    mm2 = _build_mock_mm()
    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm2),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_EMPTY_LATENT_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        mod.run(models_dir=tmp_path, prompt="test")

    assert mm2.load_checkpoint.call_count == 2


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.sdxl import t2i_refiner_prompt as mod

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            mod.run(models_dir=tmp_path, prompt="test")


# ---------------------------------------------------------------------------
# download_models idempotent — 2 entries
# ---------------------------------------------------------------------------


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.image.sdxl.t2i_refiner_prompt import manifest

    entries = manifest()
    assert len(entries) == 2

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
