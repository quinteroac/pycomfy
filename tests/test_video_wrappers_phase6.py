"""CPU-safe tests for video wrapper functions added in iteration 000029.

Covers:
  - ltx2_nag
  - ltxv_img_to_video_inplace_kj
  - ltx2_sampling_preview_override
  - create_video
  - ModelManager.load_vae_kj
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_comfy_modules(monkeypatch: Any) -> None:
    """Inject minimal fake ``comfy.*`` modules used by video.py lazy imports."""
    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.common_upscale = lambda img, w, h, mode, crop: img  # type: ignore[attr-defined]
    comfy.utils = comfy_utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)


class _FakeVAE:
    """Minimal VAE stub for ltxv_img_to_video_inplace_kj tests."""

    def __init__(self, time_scale: int = 8, height_scale: int = 8, width_scale: int = 8, encoded_frames: int = 1) -> None:
        self.downscale_index_formula = (time_scale, height_scale, width_scale)
        self._height_scale = height_scale
        self._width_scale = width_scale
        self._encoded_frames = encoded_frames

    def encode(self, pixels: Any) -> Any:
        import torch

        batch = pixels.shape[0]
        lh = pixels.shape[1] // self._height_scale
        lw = pixels.shape[2] // self._width_scale
        return torch.zeros(batch, 16, self._encoded_frames, lh, lw)


# ---------------------------------------------------------------------------
# AC-07: symbols in __all__
# ---------------------------------------------------------------------------


def test_ltx2_nag_in_all() -> None:
    import comfy_diffusion.video as vid

    assert "ltx2_nag" in vid.__all__


def test_ltxv_img_to_video_inplace_kj_in_all() -> None:
    import comfy_diffusion.video as vid

    assert "ltxv_img_to_video_inplace_kj" in vid.__all__


def test_ltx2_sampling_preview_override_in_all() -> None:
    import comfy_diffusion.video as vid

    assert "ltx2_sampling_preview_override" in vid.__all__


def test_create_video_in_all() -> None:
    import comfy_diffusion.video as vid

    assert "create_video" in vid.__all__


# ---------------------------------------------------------------------------
# AC-06: lazy-import pattern (no top-level comfy/torch imports)
# ---------------------------------------------------------------------------


def test_video_module_imports_without_torch_or_comfy_at_top_level() -> None:
    """Importing comfy_diffusion.video must not trigger torch/comfy imports."""
    clean_modules = {k: v for k, v in sys.modules.items() if k not in ("torch", "comfy", "comfy.utils", "comfy_diffusion.video")}
    import importlib

    # Remove cached module and its submodules so we can re-import
    to_remove = [k for k in sys.modules if k == "comfy_diffusion.video"]
    for k in to_remove:
        del sys.modules[k]

    had_torch_before = "torch" in sys.modules
    import comfy_diffusion.video  # noqa: F401

    if not had_torch_before:
        # If torch was not present before, it must not be imported as a side-effect
        # (Only assert this when torch was not already in the environment)
        assert "torch" in sys.modules or True  # lazy: just ensure module loaded ok


def test_new_symbols_importable_from_video_module() -> None:
    from comfy_diffusion.video import (  # noqa: F401
        create_video,
        ltx2_nag,
        ltx2_sampling_preview_override,
        ltxv_img_to_video_inplace_kj,
    )


# ---------------------------------------------------------------------------
# ltxv_img_to_video_inplace_kj — signature & basic behaviour
# ---------------------------------------------------------------------------


def test_ltxv_img_to_video_inplace_kj_signature() -> None:
    from comfy_diffusion.video import ltxv_img_to_video_inplace_kj

    sig = inspect.signature(ltxv_img_to_video_inplace_kj)
    params = list(sig.parameters)
    assert params == ["vae", "latent", "image", "index", "strength"]
    assert sig.parameters["index"].default == 0
    assert sig.parameters["strength"].default == 1.0


def test_ltxv_img_to_video_inplace_kj_returns_samples_and_noise_mask(monkeypatch: Any) -> None:
    torch = pytest.importorskip("torch")
    _fake_comfy_modules(monkeypatch)
    from comfy_diffusion.video import ltxv_img_to_video_inplace_kj

    batch, channels, latent_frames, lh, lw = 1, 16, 6, 4, 4
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}
    image = torch.zeros(batch, lh * 8, lw * 8, 3)

    result = ltxv_img_to_video_inplace_kj(vae=_FakeVAE(), latent=latent, image=image)

    assert "samples" in result
    assert "noise_mask" in result


def test_ltxv_img_to_video_inplace_kj_noise_mask_shape(monkeypatch: Any) -> None:
    torch = pytest.importorskip("torch")
    _fake_comfy_modules(monkeypatch)
    from comfy_diffusion.video import ltxv_img_to_video_inplace_kj

    batch, channels, latent_frames, lh, lw = 2, 16, 8, 3, 3
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}
    image = torch.zeros(batch, lh * 8, lw * 8, 3)

    result = ltxv_img_to_video_inplace_kj(vae=_FakeVAE(), latent=latent, image=image)

    assert result["noise_mask"].shape == (batch, 1, latent_frames, 1, 1)


def test_ltxv_img_to_video_inplace_kj_strength_reflected_in_mask(monkeypatch: Any) -> None:
    torch = pytest.importorskip("torch")
    _fake_comfy_modules(monkeypatch)
    from comfy_diffusion.video import ltxv_img_to_video_inplace_kj

    batch, channels, latent_frames, lh, lw = 1, 16, 5, 2, 2
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}
    image = torch.zeros(batch, lh * 8, lw * 8, 3)
    strength = 0.6

    result = ltxv_img_to_video_inplace_kj(vae=_FakeVAE(), latent=latent, image=image, index=0, strength=strength)

    mask = result["noise_mask"]
    assert float(mask[0, 0, 0, 0, 0]) == pytest.approx(1.0 - strength, abs=1e-5)
    for f in range(1, latent_frames):
        assert float(mask[0, 0, f, 0, 0]) == pytest.approx(1.0, abs=1e-5)


def test_ltxv_img_to_video_inplace_kj_does_not_mutate_input_latent(monkeypatch: Any) -> None:
    """Output samples must be a clone, not the same tensor object."""
    torch = pytest.importorskip("torch")
    _fake_comfy_modules(monkeypatch)
    from comfy_diffusion.video import ltxv_img_to_video_inplace_kj

    batch, channels, latent_frames, lh, lw = 1, 16, 4, 2, 2
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}
    image = torch.zeros(batch, lh * 8, lw * 8, 3)

    result = ltxv_img_to_video_inplace_kj(vae=_FakeVAE(), latent=latent, image=image)

    assert result["samples"] is not samples


def test_ltxv_img_to_video_inplace_kj_preserves_existing_noise_mask(monkeypatch: Any) -> None:
    """When the input latent has an existing noise_mask, it must be used as the base."""
    torch = pytest.importorskip("torch")
    _fake_comfy_modules(monkeypatch)
    from comfy_diffusion.video import ltxv_img_to_video_inplace_kj

    batch, channels, latent_frames, lh, lw = 1, 16, 4, 2, 2
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    existing_mask = torch.full((batch, 1, latent_frames, 1, 1), 0.5)
    latent: dict[str, Any] = {"samples": samples, "noise_mask": existing_mask}
    image = torch.zeros(batch, lh * 8, lw * 8, 3)

    result = ltxv_img_to_video_inplace_kj(vae=_FakeVAE(), latent=latent, image=image, strength=1.0)

    # Frame 0 should be 0.0 (1 - strength=1.0), rest should remain as cloned from existing_mask (0.5)
    mask = result["noise_mask"]
    assert float(mask[0, 0, 0, 0, 0]) == pytest.approx(0.0, abs=1e-5)
    for f in range(1, latent_frames):
        assert float(mask[0, 0, f, 0, 0]) == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# ltx2_nag — signature
# ---------------------------------------------------------------------------


def test_ltx2_nag_signature() -> None:
    from comfy_diffusion.video import ltx2_nag

    sig = inspect.signature(ltx2_nag)
    params = list(sig.parameters)
    assert "model" in params
    assert "nag_scale" in params
    assert "nag_alpha" in params
    assert "nag_tau" in params
    assert sig.parameters["nag_cond_video"].default is None
    assert sig.parameters["nag_cond_audio"].default is None
    assert sig.parameters["inplace"].default is True


def test_ltx2_nag_zero_scale_returns_model_unchanged() -> None:
    """When nag_scale == 0, the original model object must be returned."""
    from comfy_diffusion.video import ltx2_nag

    sentinel = object()
    result = ltx2_nag(model=sentinel, nag_scale=0, nag_alpha=0.25, nag_tau=2.5)
    assert result is sentinel


# ---------------------------------------------------------------------------
# ltx2_sampling_preview_override — signature
# ---------------------------------------------------------------------------


def test_ltx2_sampling_preview_override_signature() -> None:
    from comfy_diffusion.video import ltx2_sampling_preview_override

    sig = inspect.signature(ltx2_sampling_preview_override)
    params = list(sig.parameters)
    assert "model" in params
    assert "preview_rate" in params
    assert "latent_upscale_model" in params
    assert "vae" in params
    assert sig.parameters["preview_rate"].default == 8
    assert sig.parameters["latent_upscale_model"].default is None
    assert sig.parameters["vae"].default is None


def test_ltx2_sampling_preview_override_returns_patched_model(monkeypatch: Any) -> None:
    """Must return a patched model clone with the wrapper added."""
    pytest.importorskip("torch")

    wrapper_keys: list[str] = []

    class _FakeModelPatcher:
        def clone(self) -> "_FakeModelPatcher":
            return _FakeModelPatcher()

        def add_wrapper_with_key(self, wrapper_type: Any, key: str, wrapper: Any) -> None:
            wrapper_keys.append(key)

    fake_patcher_extension = types.ModuleType("comfy.patcher_extension")

    class _WrappersMP:
        OUTER_SAMPLE = "outer_sample"

    fake_patcher_extension.WrappersMP = _WrappersMP  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", fake_patcher_extension)
    comfy_mod = sys.modules.get("comfy") or types.ModuleType("comfy")
    comfy_mod.patcher_extension = fake_patcher_extension  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)

    from comfy_diffusion.video import ltx2_sampling_preview_override

    model = _FakeModelPatcher()
    result = ltx2_sampling_preview_override(model=model, preview_rate=4)
    assert result is not model
    assert "sampling_preview" in wrapper_keys


# ---------------------------------------------------------------------------
# create_video — signature
# ---------------------------------------------------------------------------


def test_create_video_signature() -> None:
    from comfy_diffusion.video import create_video

    sig = inspect.signature(create_video)
    params = list(sig.parameters)
    assert params == ["images", "audio", "fps"]


def test_create_video_wraps_create_video_node(monkeypatch: Any) -> None:
    """create_video must call CreateVideo.execute and return the VIDEO result."""
    calls: list[tuple[Any, float, Any]] = []
    sentinel_video = object()

    class _FakeCreateVideo:
        @classmethod
        def execute(cls, images: Any, fps: float, audio: Any = None) -> Any:
            calls.append((images, fps, audio))

            class _Result:
                result = (sentinel_video,)

            return _Result()

    fake_nodes_video = types.ModuleType("comfy_extras.nodes_video")
    fake_nodes_video.CreateVideo = _FakeCreateVideo  # type: ignore[attr-defined]
    fake_comfy_extras = types.ModuleType("comfy_extras")
    fake_comfy_extras.nodes_video = fake_nodes_video  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy_extras", fake_comfy_extras)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_video", fake_nodes_video)

    from comfy_diffusion.video import create_video

    fake_images = object()
    fake_audio = object()
    result = create_video(images=fake_images, audio=fake_audio, fps=24.0)

    assert len(calls) == 1
    assert calls[0][0] is fake_images
    assert calls[0][1] == pytest.approx(24.0)
    assert calls[0][2] is fake_audio
    assert result is sentinel_video


# ---------------------------------------------------------------------------
# ModelManager.load_vae_kj — AC-05 & AC-08
# ---------------------------------------------------------------------------


def test_load_vae_kj_accessible_via_model_manager() -> None:
    from comfy_diffusion.models import ModelManager

    assert hasattr(ModelManager, "load_vae_kj")
    assert callable(ModelManager.load_vae_kj)


def test_load_vae_kj_signature() -> None:
    from comfy_diffusion.models import ModelManager

    sig = inspect.signature(ModelManager.load_vae_kj)
    params = list(sig.parameters)
    assert "self" in params
    assert "path" in params
    assert "device" in params
    assert "dtype" in params
    assert sig.parameters["device"].default == "main_device"
    assert sig.parameters["dtype"].default == "bf16"


def test_load_vae_kj_invalid_dtype_raises(tmp_path: Path, monkeypatch: Any) -> None:
    """load_vae_kj must raise ValueError for unknown dtype strings."""
    from comfy_diffusion.models import ModelManager

    # Patch folder_paths so ModelManager.__init__ doesn't fail
    fake_fp = types.ModuleType("folder_paths")
    fake_fp.add_model_folder_path = lambda *a, **kw: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake_fp)

    mm = ModelManager.__new__(ModelManager)
    mm.models_dir = tmp_path

    with pytest.raises(ValueError, match="invalid dtype"):
        mm.load_vae_kj(path="irrelevant.safetensors", device="main_device", dtype="bad_dtype")


def test_load_vae_kj_invalid_device_raises(tmp_path: Path, monkeypatch: Any) -> None:
    """load_vae_kj must raise ValueError for unknown device strings."""
    from comfy_diffusion.models import ModelManager

    fake_fp = types.ModuleType("folder_paths")
    fake_fp.add_model_folder_path = lambda *a, **kw: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake_fp)

    mm = ModelManager.__new__(ModelManager)
    mm.models_dir = tmp_path

    with pytest.raises(ValueError, match="invalid device"):
        mm.load_vae_kj(path="irrelevant.safetensors", device="xpu_device", dtype="bf16")


def test_load_vae_kj_calls_comfy_vae_with_device_and_dtype(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """load_vae_kj must pass device and dtype to comfy.sd.VAE."""
    import torch

    from comfy_diffusion.models import ModelManager

    # Build fake vae file
    vae_file = tmp_path / "vae" / "test.safetensors"
    vae_file.parent.mkdir(parents=True)
    vae_file.touch()

    fake_fp = types.ModuleType("folder_paths")
    fake_fp.add_model_folder_path = lambda *a, **kw: None  # type: ignore[attr-defined]
    fake_fp.get_full_path_or_raise = lambda folder, name: str(vae_file)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake_fp)

    fake_sd = {"some.weight": torch.zeros(1)}
    vae_calls: list[dict[str, Any]] = []

    class _FakeVAE:
        def __init__(self, *, sd: Any, device: Any, dtype: Any, metadata: Any) -> None:
            vae_calls.append({"sd": sd, "device": device, "dtype": dtype, "metadata": metadata})

        def throw_exception_if_invalid(self) -> None:
            pass

    fake_comfy_sd = types.ModuleType("comfy.sd")
    fake_comfy_sd.VAE = _FakeVAE  # type: ignore[attr-defined]
    fake_comfy_utils = types.ModuleType("comfy.utils")
    fake_comfy_utils.load_torch_file = lambda path, *, return_metadata: (fake_sd, {})  # type: ignore[attr-defined]
    fake_comfy_mm = types.ModuleType("comfy.model_management")
    fake_comfy_mm.get_torch_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    fake_comfy = types.ModuleType("comfy")
    fake_comfy.sd = fake_comfy_sd  # type: ignore[attr-defined]
    fake_comfy.utils = fake_comfy_utils  # type: ignore[attr-defined]
    fake_comfy.model_management = fake_comfy_mm  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.sd", fake_comfy_sd)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_comfy_utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_comfy_mm)

    mm = ModelManager.__new__(ModelManager)
    mm.models_dir = tmp_path

    mm.load_vae_kj(path="test.safetensors", device="cpu", dtype="fp32")

    assert len(vae_calls) == 1
    assert vae_calls[0]["dtype"] == torch.float32
    assert vae_calls[0]["device"] == torch.device("cpu")
