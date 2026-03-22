"""Tests for US-006 ModelManager.load_latent_upscale_model()."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from comfy_diffusion.models import ModelManager


# ---------------------------------------------------------------------------
# Fake model classes
# ---------------------------------------------------------------------------


class _FakeLatentUpsampler:
    """Stand-in for comfy.ldm.lightricks.latent_upsampler.LatentUpsampler."""

    def __init__(self) -> None:
        self._dtype: Any = None
        self._state_dict: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "_FakeLatentUpsampler":
        return cls()

    def to(self, *, dtype: Any = None) -> "_FakeLatentUpsampler":
        self._dtype = dtype
        return self

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._state_dict = sd


def _install_fake_latent_upscale_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state_dict: dict[str, Any],
    metadata: dict[str, Any] | None,
    resolved_path_for_relative: str | None = None,
    fake_upsampler: _FakeLatentUpsampler | None = None,
) -> dict[str, list[Any]]:
    """Install fake modules into sys.modules and return a call log."""
    calls: dict[str, list[Any]] = {
        "add_model_folder_path": [],
        "get_full_path_or_raise": [],
        "load_torch_file": [],
        "LatentUpsampler_from_config": [],
    }

    # --- folder_paths ---
    folder_paths_module = ModuleType("folder_paths")

    def add_model_folder_path(
        folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        calls["add_model_folder_path"].append((folder_name, full_folder_path, is_default))

    def get_full_path_or_raise(folder_name: str, filename: str) -> str:
        calls["get_full_path_or_raise"].append((folder_name, filename))
        if resolved_path_for_relative is None:
            raise AssertionError(f"unexpected get_full_path_or_raise call: {folder_name}/{filename}")
        return resolved_path_for_relative

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)
    setattr(folder_paths_module, "get_full_path_or_raise", get_full_path_or_raise)

    # --- comfy.utils ---
    comfy_utils_module = ModuleType("comfy.utils")

    def load_torch_file(
        path: str, *, safe_load: bool = False, return_metadata: bool = False
    ) -> Any:
        calls["load_torch_file"].append((path, safe_load, return_metadata))
        return state_dict, metadata

    setattr(comfy_utils_module, "load_torch_file", load_torch_file)

    # --- comfy.model_management ---
    comfy_mm_module = ModuleType("comfy.model_management")

    def vae_dtype(allowed_dtypes: list[Any] | None = None) -> Any:
        return "bfloat16"

    setattr(comfy_mm_module, "vae_dtype", vae_dtype)

    # --- comfy.ldm.lightricks.latent_upsampler ---
    _upsampler = fake_upsampler if fake_upsampler is not None else _FakeLatentUpsampler()
    latent_upsampler_module = ModuleType("comfy.ldm.lightricks.latent_upsampler")

    class _TrackedLatentUpsampler(_FakeLatentUpsampler):
        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "_TrackedLatentUpsampler":  # type: ignore[override]
            calls["LatentUpsampler_from_config"].append(config)
            return _upsampler  # type: ignore[return-value]

    setattr(latent_upsampler_module, "LatentUpsampler", _TrackedLatentUpsampler)

    # --- torch stub (for bfloat16 / float32 references) ---
    torch_module = ModuleType("torch")
    setattr(torch_module, "bfloat16", "bfloat16")
    setattr(torch_module, "float32", "float32")

    # --- comfy main module ---
    comfy_module = ModuleType("comfy")
    comfy_ldm_module = ModuleType("comfy.ldm")
    comfy_lightricks_module = ModuleType("comfy.ldm.lightricks")

    setattr(comfy_module, "utils", comfy_utils_module)
    setattr(comfy_module, "model_management", comfy_mm_module)
    setattr(comfy_module, "ldm", comfy_ldm_module)
    setattr(comfy_ldm_module, "lightricks", comfy_lightricks_module)
    setattr(comfy_lightricks_module, "latent_upsampler", latent_upsampler_module)

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)
    monkeypatch.setitem(sys.modules, "comfy.model_management", comfy_mm_module)
    monkeypatch.setitem(sys.modules, "comfy.ldm", comfy_ldm_module)
    monkeypatch.setitem(sys.modules, "comfy.ldm.lightricks", comfy_lightricks_module)
    monkeypatch.setitem(sys.modules, "comfy.ldm.lightricks.latent_upsampler", latent_upsampler_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    return calls


def _ltxv_state_dict() -> dict[str, Any]:
    """Minimal LTXV state dict (triggers the LTXV branch in load_latent_upscale_model)."""
    return {"post_upsample_res_blocks.0.conv2.bias": object()}


def _ltxv_metadata() -> dict[str, Any]:
    return {"config": '{"in_channels": 128}'}


def _make_models_dir(tmp_path: Path) -> Path:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)
    return models_dir


# ---------------------------------------------------------------------------
# AC01 — method exists on ModelManager
# ---------------------------------------------------------------------------


def test_load_latent_upscale_model_method_exists_on_model_manager() -> None:
    assert callable(getattr(ModelManager, "load_latent_upscale_model", None))


# ---------------------------------------------------------------------------
# AC02 — absolute path to existing file is loaded directly
# ---------------------------------------------------------------------------


def test_load_latent_upscale_model_absolute_existing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)
    model_file = tmp_path / "weights" / "ltxv_upscale.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("stub latent upscale model")

    fake_upsampler = _FakeLatentUpsampler()
    calls = _install_fake_latent_upscale_loader_modules(
        monkeypatch,
        state_dict=_ltxv_state_dict(),
        metadata=_ltxv_metadata(),
        fake_upsampler=fake_upsampler,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_latent_upscale_model(model_file)

    assert result is fake_upsampler
    assert calls["get_full_path_or_raise"] == []
    assert calls["load_torch_file"] == [(str(model_file.resolve()), True, True)]


# ---------------------------------------------------------------------------
# AC03 — relative filename resolved against models_dir/upscale/
# ---------------------------------------------------------------------------


def test_load_latent_upscale_model_relative_path_resolved_against_upscale_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)
    upscale_dir = models_dir / "upscale"
    upscale_dir.mkdir(parents=True)
    model_file = upscale_dir / "ltxv_upscale.safetensors"
    model_file.write_text("stub latent upscale model")
    resolved_path = str(model_file.resolve())

    fake_upsampler = _FakeLatentUpsampler()
    calls = _install_fake_latent_upscale_loader_modules(
        monkeypatch,
        state_dict=_ltxv_state_dict(),
        metadata=_ltxv_metadata(),
        resolved_path_for_relative=resolved_path,
        fake_upsampler=fake_upsampler,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_latent_upscale_model("ltxv_upscale.safetensors")

    assert result is fake_upsampler
    assert calls["get_full_path_or_raise"] == [("latent_upscale_models", "ltxv_upscale.safetensors")]
    assert calls["load_torch_file"] == [(resolved_path, True, True)]


# ---------------------------------------------------------------------------
# AC04 — absolute path that does not exist raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_load_latent_upscale_model_absolute_missing_path_raises_file_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)
    missing = tmp_path / "weights" / "nonexistent.safetensors"

    calls = _install_fake_latent_upscale_loader_modules(
        monkeypatch,
        state_dict={},
        metadata=None,
    )

    with pytest.raises(FileNotFoundError, match="latent upscale model file not found"):
        ModelManager(models_dir=models_dir).load_latent_upscale_model(missing)

    assert calls["load_torch_file"] == []


# ---------------------------------------------------------------------------
# AC05 — uses comfy.utils.load_torch_file(safe_load=True, return_metadata=True)
# ---------------------------------------------------------------------------


def test_load_latent_upscale_model_uses_load_torch_file_safe_load_and_return_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)
    model_file = tmp_path / "weights" / "model.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("stub")

    calls = _install_fake_latent_upscale_loader_modules(
        monkeypatch,
        state_dict=_ltxv_state_dict(),
        metadata=_ltxv_metadata(),
    )

    ModelManager(models_dir=models_dir).load_latent_upscale_model(model_file)

    assert len(calls["load_torch_file"]) == 1
    _path, safe_load, return_metadata = calls["load_torch_file"][0]
    assert safe_load is True
    assert return_metadata is True


# ---------------------------------------------------------------------------
# AC06 — folder_paths.add_model_folder_path("latent_upscale_models", ...) in __init__
# ---------------------------------------------------------------------------


def test_load_latent_upscale_model_registers_latent_upscale_models_folder_on_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = _make_models_dir(tmp_path)

    calls = _install_fake_latent_upscale_loader_modules(
        monkeypatch,
        state_dict={},
        metadata=None,
    )

    ModelManager(models_dir=models_dir)

    latent_upscale_registrations = [
        c for c in calls["add_model_folder_path"] if c[0] == "latent_upscale_models"
    ]
    assert len(latent_upscale_registrations) == 1
    folder_name, folder_path, is_default = latent_upscale_registrations[0]
    assert folder_name == "latent_upscale_models"
    assert folder_path == str(models_dir / "upscale")
    assert is_default is True


# ---------------------------------------------------------------------------
# AC07 — lazy imports: no heavy modules at import time
# ---------------------------------------------------------------------------


def test_load_latent_upscale_model_lazy_imports_no_heavy_modules_at_import_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mod_name in list(sys.modules.keys()):
        if (
            mod_name.startswith("comfy.ldm.lightricks.latent_upsampler")
            or mod_name.startswith("comfy_extras.nodes_hunyuan")
        ):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    import importlib
    import comfy_diffusion.models as models_module

    importlib.reload(models_module)

    assert "comfy.ldm.lightricks.latent_upsampler" not in sys.modules
    assert "comfy_extras.nodes_hunyuan" not in sys.modules


# ---------------------------------------------------------------------------
# AC08 — load_latent_upscale_model in models.py __all__
# ---------------------------------------------------------------------------


def test_load_latent_upscale_model_in_module_all() -> None:
    from comfy_diffusion import models as models_module

    assert "load_latent_upscale_model" in models_module.__all__
    assert hasattr(ModelManager, "load_latent_upscale_model")
