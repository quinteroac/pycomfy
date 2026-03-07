"""Tests for sampling helpers."""

from __future__ import annotations

import inspect
import sys
from types import ModuleType
from typing import Any

import pytest

import pycomfy.sampling as sampling_module
from pycomfy.sampling import sample


def test_sampling_public_api_exports_sample_only() -> None:
    assert sample.__name__ == "sample"
    assert sampling_module.__all__ == ["sample"]


def test_sample_signature_matches_contract() -> None:
    signature = inspect.signature(sample)

    assert str(signature) == (
        "(model: 'Any', positive: 'Any', negative: 'Any', latent: 'Any', "
        "steps: 'Any', cfg: 'Any', sampler_name: 'str', scheduler: 'str', "
        "seed: 'int', *, denoise: 'float' = 1.0) -> 'Any'"
    )


def test_sample_returns_raw_denoised_latent_without_transformation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_latent = {"samples": object(), "batch_index": [0]}

    nodes_module = ModuleType("nodes")

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        return (raw_latent,)

    setattr(nodes_module, "common_ksampler", common_ksampler)
    monkeypatch.setitem(sys.modules, "nodes", nodes_module)
    monkeypatch.setattr(sampling_module, "ensure_comfyui_on_path", lambda: None)

    result = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=123,
    )

    assert result is raw_latent


def test_sample_uses_common_ksampler_call_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    positive = object()
    negative = object()
    latent = {"samples": object()}
    expected = object()
    recorded: dict[str, Any] = {}

    nodes_module = ModuleType("nodes")

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return (expected,)

    setattr(nodes_module, "common_ksampler", common_ksampler)
    monkeypatch.setitem(sys.modules, "nodes", nodes_module)
    monkeypatch.setattr(sampling_module, "ensure_comfyui_on_path", lambda: None)

    result = sample(
        model=model,
        positive=positive,
        negative=negative,
        latent=latent,
        steps=30,
        cfg=5.5,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        seed=42,
        denoise=0.65,
    )

    assert result is expected
    assert recorded["args"] == (
        model,
        42,
        30,
        5.5,
        "dpmpp_2m",
        "karras",
        positive,
        negative,
        latent,
    )
    assert recorded["kwargs"] == {"denoise": 0.65}


def test_sample_passes_sampler_and_scheduler_strings_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}
    sampler_name = " Euler++ custom "
    scheduler = "Normal/alt schedule "

    nodes_module = ModuleType("nodes")

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["args"] = args
        return (object(),)

    setattr(nodes_module, "common_ksampler", common_ksampler)
    monkeypatch.setitem(sys.modules, "nodes", nodes_module)
    monkeypatch.setattr(sampling_module, "ensure_comfyui_on_path", lambda: None)

    sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=10,
        cfg=4.0,
        sampler_name=sampler_name,
        scheduler=scheduler,
        seed=999,
    )

    assert recorded["args"][4] == sampler_name
    assert recorded["args"][5] == scheduler


def test_sample_passes_seed_to_common_ksampler_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}
    seed = 4_294_967_295

    nodes_module = ModuleType("nodes")

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["args"] = args
        return (object(),)

    setattr(nodes_module, "common_ksampler", common_ksampler)
    monkeypatch.setitem(sys.modules, "nodes", nodes_module)
    monkeypatch.setattr(sampling_module, "ensure_comfyui_on_path", lambda: None)

    sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=20,
        cfg=6.5,
        sampler_name="euler",
        scheduler="normal",
        seed=seed,
    )

    assert recorded["args"][1] == seed
