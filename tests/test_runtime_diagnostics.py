"""Tests for US-005 structured runtime diagnostics."""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import dataclass
from typing import Any

import pytest

from comfy_diffusion.runtime import check_runtime


@dataclass
class _FakeDevice:
    type: str
    index: int | None = None

    def __str__(self) -> str:
        if self.index is None:
            return self.type
        return f"{self.type}:{self.index}"


def test_check_runtime_returns_required_keys_and_types_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(module_name: str) -> Any:
        if module_name == "comfyui_version":
            return type("ComfyVersion", (), {"__version__": "0.16.3"})()

        if module_name == "comfy.model_management":
            return type(
                "ModelManagement",
                (),
                {
                    "get_torch_device": staticmethod(lambda: _FakeDevice("cpu")),
                    "get_total_memory": staticmethod(lambda _dev: 16 * 1024 * 1024 * 1024),
                    "get_free_memory": staticmethod(lambda _dev: 8 * 1024 * 1024 * 1024),
                },
            )()

        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    payload = check_runtime()

    assert isinstance(payload["comfyui_version"], str)
    assert payload["device"] == "cpu"
    assert isinstance(payload["vram_total_mb"], int)
    assert isinstance(payload["vram_free_mb"], int)
    assert isinstance(payload["python_version"], str)
    assert payload["vram_total_mb"] == 0
    assert payload["vram_free_mb"] == 0
    assert "error" not in payload


def test_check_runtime_reports_cuda_device_and_vram_via_model_management(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {"device": 0, "total": 0, "free": 0}
    fake_device = _FakeDevice("cuda", index=0)
    total_memory = 10 * 1024 * 1024 * 1024
    free_memory = 4 * 1024 * 1024 * 1024

    class _FakeModelManagement:
        @staticmethod
        def get_torch_device() -> _FakeDevice:
            calls["device"] += 1
            return fake_device

        @staticmethod
        def get_total_memory(device: _FakeDevice) -> int:
            calls["total"] += 1
            assert device is fake_device
            return total_memory

        @staticmethod
        def get_free_memory(device: _FakeDevice) -> int:
            calls["free"] += 1
            assert device is fake_device
            return free_memory

    def fake_import_module(module_name: str) -> Any:
        if module_name == "comfyui_version":
            return type("ComfyVersion", (), {"__version__": "0.16.3"})()

        if module_name == "comfy.model_management":
            return _FakeModelManagement

        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    payload = check_runtime()

    assert payload["device"] == "cuda:0"
    assert payload["vram_total_mb"] == 10240
    assert payload["vram_free_mb"] == 4096
    assert payload["comfyui_version"] == "0.16.3"
    assert calls == {"device": 1, "total": 1, "free": 1}


def test_check_runtime_returns_error_dict_when_runtime_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(module_name: str) -> Any:
        if module_name == "comfyui_version":
            return type("ComfyVersion", (), {"__version__": "0.16.3"})()
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    payload = check_runtime()

    assert "error" in payload
    assert "git submodule update --init" in payload["error"]
    assert payload["comfyui_version"] is None
    assert payload["device"] is None
    assert payload["vram_total_mb"] is None
    assert payload["vram_free_mb"] is None
    assert isinstance(payload["python_version"], str)


def test_check_runtime_returns_error_dict_when_runtime_is_not_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingModelManagement:
        @staticmethod
        def get_torch_device() -> _FakeDevice:
            raise RuntimeError("model management unavailable")

    def fake_import_module(module_name: str) -> Any:
        if module_name == "comfyui_version":
            return type("ComfyVersion", (), {"__version__": "0.16.3"})()
        if module_name == "comfy.model_management":
            return _FailingModelManagement
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    payload = check_runtime()

    assert "error" in payload
    assert "not responsive" in payload["error"]
    assert payload["comfyui_version"] is None
    assert payload["device"] is None
    assert payload["vram_total_mb"] is None
    assert payload["vram_free_mb"] is None
    assert isinstance(payload["python_version"], str)


def test_runtime_smoke_one_liner_succeeds() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion import check_runtime; print(check_runtime())",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "python_version" in result.stdout
    assert "comfyui_version" in result.stdout
