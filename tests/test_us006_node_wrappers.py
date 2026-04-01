"""Tests for US-006 — New node wrappers for image pipelines.

Covers:
  - AC01: empty_sd3_latent_image in latent module, in __all__, correct signature
  - AC02: sd_turbo_scheduler in sampling module, in __all__, correct signature
  - AC03: sample_custom_simple in sampling module, in __all__, wraps SamplerCustom, correct signature
  - AC04: all three wrappers follow the lazy-import pattern (no top-level comfy.*)
  - AC05: module-level imports are clean (no torch/comfy at top level)
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LATENT_FILE = _REPO_ROOT / "comfy_diffusion" / "latent.py"
_SAMPLING_FILE = _REPO_ROOT / "comfy_diffusion" / "sampling.py"


# ---------------------------------------------------------------------------
# AC01: empty_sd3_latent_image
# ---------------------------------------------------------------------------


def test_empty_sd3_latent_image_exists() -> None:
    import comfy_diffusion.latent as latent_mod

    assert hasattr(latent_mod, "empty_sd3_latent_image")
    assert callable(latent_mod.empty_sd3_latent_image)


def test_empty_sd3_latent_image_in_dunder_all() -> None:
    import comfy_diffusion.latent as latent_mod

    assert "empty_sd3_latent_image" in latent_mod.__all__


def test_empty_sd3_latent_image_signature() -> None:
    from comfy_diffusion.latent import empty_sd3_latent_image

    sig = inspect.signature(empty_sd3_latent_image)
    params = sig.parameters
    assert "width" in params
    assert "height" in params
    assert "batch_size" in params
    assert params["batch_size"].default == 1


# ---------------------------------------------------------------------------
# AC02: sd_turbo_scheduler
# ---------------------------------------------------------------------------


def test_sd_turbo_scheduler_exists() -> None:
    import comfy_diffusion.sampling as sampling_mod

    assert hasattr(sampling_mod, "sd_turbo_scheduler")
    assert callable(sampling_mod.sd_turbo_scheduler)


def test_sd_turbo_scheduler_in_dunder_all() -> None:
    import comfy_diffusion.sampling as sampling_mod

    assert "sd_turbo_scheduler" in sampling_mod.__all__


def test_sd_turbo_scheduler_signature() -> None:
    from comfy_diffusion.sampling import sd_turbo_scheduler

    sig = inspect.signature(sd_turbo_scheduler)
    params = sig.parameters
    assert "model" in params
    assert "steps" in params
    assert "denoise" in params
    assert params["denoise"].default == 1.0


# ---------------------------------------------------------------------------
# AC03: sample_custom_simple wraps SamplerCustom
# ---------------------------------------------------------------------------


def test_sample_custom_simple_exists() -> None:
    import comfy_diffusion.sampling as sampling_mod

    assert hasattr(sampling_mod, "sample_custom_simple")
    assert callable(sampling_mod.sample_custom_simple)


def test_sample_custom_simple_in_dunder_all() -> None:
    import comfy_diffusion.sampling as sampling_mod

    assert "sample_custom_simple" in sampling_mod.__all__


def test_sample_custom_simple_signature() -> None:
    from comfy_diffusion.sampling import sample_custom_simple

    sig = inspect.signature(sample_custom_simple)
    params = set(sig.parameters.keys())
    required = {
        "model",
        "add_noise",
        "noise_seed",
        "cfg",
        "positive",
        "negative",
        "sampler",
        "sigmas",
        "latent_image",
    }
    assert required <= params, f"sample_custom_simple missing params: {required - params}"


def test_sample_custom_simple_wraps_sampler_custom() -> None:
    """Verify sample_custom_simple calls SamplerCustom.execute internally."""
    source = _SAMPLING_FILE.read_text(encoding="utf-8")
    assert "SamplerCustom" in source, "SamplerCustom must be referenced in sampling.py"
    # Ensure it references SamplerCustom (not only SamplerCustomAdvanced)
    assert "_get_sampler_custom_type" in source


def test_sample_custom_simple_returns_first_output() -> None:
    """sample_custom_simple must return the first latent dict output."""
    from unittest.mock import MagicMock, patch

    fake_out = MagicMock(name="out_latent")
    fake_denoised = MagicMock(name="denoised_latent")

    class FakeSamplerCustom:
        @classmethod
        def execute(cls, *args, **kwargs):
            result = MagicMock()
            result.result = [fake_out, fake_denoised]
            return result

    with patch(
        "comfy_diffusion.sampling._get_sampler_custom_type",
        return_value=FakeSamplerCustom,
    ):
        from comfy_diffusion.sampling import sample_custom_simple

        result = sample_custom_simple(
            model=MagicMock(),
            add_noise=False,
            noise_seed=0,
            cfg=1.0,
            positive=MagicMock(),
            negative=MagicMock(),
            sampler=MagicMock(),
            sigmas=MagicMock(),
            latent_image={"samples": MagicMock()},
        )

    assert result is fake_out


# ---------------------------------------------------------------------------
# AC04 / AC05: Lazy-import — no top-level comfy.* or torch imports
# ---------------------------------------------------------------------------


def _get_top_level_imports(filepath: Path) -> list[str]:
    """Return all module names imported at module top level (not inside functions)."""
    tree = ast.parse(filepath.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        # Only consider statements directly inside the module body (col_offset 0)
        if node.col_offset != 0:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_latent_no_top_level_comfy_import() -> None:
    top_level = _get_top_level_imports(_LATENT_FILE)
    comfy_imports = [m for m in top_level if m.startswith("comfy") or m == "torch"]
    assert not comfy_imports, (
        f"latent.py has top-level comfy/torch imports: {comfy_imports}"
    )


def test_sampling_no_top_level_comfy_import() -> None:
    top_level = _get_top_level_imports(_SAMPLING_FILE)
    comfy_imports = [m for m in top_level if m.startswith("comfy") or m == "torch"]
    assert not comfy_imports, (
        f"sampling.py has top-level comfy/torch imports: {comfy_imports}"
    )
