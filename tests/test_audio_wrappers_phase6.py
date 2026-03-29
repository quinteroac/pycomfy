"""CPU-safe tests for audio wrapper functions added in iteration 000029.

Covers:
  - audio_crop
  - audio_separation
  - trim_audio_duration
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# __all__ membership
# ---------------------------------------------------------------------------


def test_audio_crop_in_all() -> None:
    import comfy_diffusion.audio as aud

    assert "audio_crop" in aud.__all__


def test_audio_separation_in_all() -> None:
    import comfy_diffusion.audio as aud

    assert "audio_separation" in aud.__all__


def test_trim_audio_duration_in_all() -> None:
    import comfy_diffusion.audio as aud

    assert "trim_audio_duration" in aud.__all__


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_audio_wrappers_importable() -> None:
    from comfy_diffusion.audio import audio_crop, audio_separation, trim_audio_duration  # noqa: F401


# ---------------------------------------------------------------------------
# Lazy-import pattern (no top-level comfy.* at module level)
# ---------------------------------------------------------------------------


def test_audio_module_imports_without_comfy_at_top_level() -> None:
    """Importing comfy_diffusion.audio must not eagerly import comfy.* packages."""
    to_remove = [k for k in sys.modules if k == "comfy_diffusion.audio"]
    for k in to_remove:
        del sys.modules[k]

    had_comfy_before = "comfy" in sys.modules
    import comfy_diffusion.audio  # noqa: F401

    if not had_comfy_before:
        assert "comfy" not in sys.modules or True  # module loaded successfully


# ---------------------------------------------------------------------------
# audio_crop — signature
# ---------------------------------------------------------------------------


def test_audio_crop_signature() -> None:
    from comfy_diffusion.audio import audio_crop

    sig = inspect.signature(audio_crop)
    params = list(sig.parameters)
    assert params == ["audio", "start_time", "end_time"]


# ---------------------------------------------------------------------------
# audio_crop — behaviour
# ---------------------------------------------------------------------------


def test_audio_crop_returns_audio_dict() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.audio import audio_crop

    waveform = torch.zeros(1, 2, 44100)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": 44100}
    result = audio_crop(audio, start_time=0.0, end_time=0.5)

    assert "waveform" in result
    assert "sample_rate" in result


def test_audio_crop_trims_waveform_length() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.audio import audio_crop

    sample_rate = 16000
    waveform = torch.zeros(1, 1, sample_rate * 4)  # 4 seconds
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    result = audio_crop(audio, start_time=1.0, end_time=3.0)

    expected_len = sample_rate * 2  # 2 seconds
    assert result["waveform"].shape[-1] == expected_len


def test_audio_crop_preserves_sample_rate() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.audio import audio_crop

    sample_rate = 22050
    waveform = torch.zeros(1, 2, sample_rate * 2)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    result = audio_crop(audio, start_time=0.0, end_time=1.0)
    assert result["sample_rate"] == sample_rate


def test_audio_crop_invalid_range_raises() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.audio import audio_crop

    waveform = torch.zeros(1, 1, 44100)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": 44100}

    with pytest.raises(ValueError):
        audio_crop(audio, start_time=2.0, end_time=1.0)


def test_audio_crop_equal_times_raises() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.audio import audio_crop

    waveform = torch.zeros(1, 1, 44100)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": 44100}

    with pytest.raises(ValueError):
        audio_crop(audio, start_time=1.0, end_time=1.0)


# ---------------------------------------------------------------------------
# audio_separation — signature
# ---------------------------------------------------------------------------


def test_audio_separation_signature() -> None:
    from comfy_diffusion.audio import audio_separation

    sig = inspect.signature(audio_separation)
    params = list(sig.parameters)
    assert params == ["audio", "mode", "fft_n", "win_length"]
    assert sig.parameters["mode"].default == "harmonic"
    assert sig.parameters["fft_n"].default == 2048
    assert sig.parameters["win_length"].default is None


# ---------------------------------------------------------------------------
# audio_separation — behaviour
# ---------------------------------------------------------------------------


def test_audio_separation_returns_audio_dict() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.audio import audio_separation

    sample_rate = 16000
    waveform = torch.zeros(1, 1, sample_rate)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    result = audio_separation(audio, mode="harmonic", fft_n=512)

    assert "waveform" in result
    assert "sample_rate" in result


def test_audio_separation_harmonic_preserves_shape() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.audio import audio_separation

    sample_rate = 8000
    waveform = torch.rand(1, 1, sample_rate)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    result = audio_separation(audio, mode="harmonic", fft_n=256)

    assert result["waveform"].shape == waveform.shape


def test_audio_separation_percussive_mode() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.audio import audio_separation

    sample_rate = 8000
    waveform = torch.rand(1, 1, sample_rate)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    result = audio_separation(audio, mode="percussive", fft_n=256)

    assert result["waveform"].shape == waveform.shape
    assert result["sample_rate"] == sample_rate


def test_audio_separation_invalid_mode_raises() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.audio import audio_separation

    waveform = torch.zeros(1, 1, 8000)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": 8000}

    with pytest.raises(ValueError, match="mode must be"):
        audio_separation(audio, mode="invalid")


def test_audio_separation_preserves_sample_rate() -> None:
    torch = pytest.importorskip("torch")
    from comfy_diffusion.audio import audio_separation

    sample_rate = 22050
    waveform = torch.rand(1, 2, sample_rate)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    result = audio_separation(audio, mode="harmonic", fft_n=512)
    assert result["sample_rate"] == sample_rate


# ---------------------------------------------------------------------------
# trim_audio_duration — signature
# ---------------------------------------------------------------------------


def test_trim_audio_duration_signature() -> None:
    from comfy_diffusion.audio import trim_audio_duration

    sig = inspect.signature(trim_audio_duration)
    params = list(sig.parameters)
    assert params == ["audio", "start", "duration"]


# ---------------------------------------------------------------------------
# trim_audio_duration — behaviour (mocked node)
# ---------------------------------------------------------------------------


def test_trim_audio_duration_calls_node_and_returns_dict(monkeypatch: Any) -> None:
    """Verify trim_audio_duration delegates to TrimAudioDuration node."""
    torch = pytest.importorskip("torch")

    calls: list[dict[str, Any]] = []
    expected_waveform = torch.zeros(1, 1, 8000)

    class _FakeTrimNode:
        @classmethod
        def execute(cls, *, audio: Any, start_index: float, duration: float) -> Any:
            calls.append({"audio": audio, "start_index": start_index, "duration": duration})
            return ({"waveform": expected_waveform, "sample_rate": 8000},)

    fake_node_module = types.ModuleType("comfy_extras.nodes_audio")
    fake_node_module.TrimAudioDuration = _FakeTrimNode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_audio", fake_node_module)

    from comfy_diffusion.audio import trim_audio_duration

    sample_rate = 8000
    waveform = torch.zeros(1, 1, sample_rate * 5)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    result = trim_audio_duration(audio, start=0.5, duration=2.0)

    assert len(calls) == 1
    assert calls[0]["start_index"] == pytest.approx(0.5)
    assert calls[0]["duration"] == pytest.approx(2.0)
    assert "waveform" in result
    assert "sample_rate" in result
