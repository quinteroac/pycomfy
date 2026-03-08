"""Audio helpers."""

from __future__ import annotations

from typing import Any, Protocol


class _LtxvAudioVaeEncoder(Protocol):
    def encode(self, audio: Any) -> Any: ...


def ltxv_audio_vae_encode(vae: _LtxvAudioVaeEncoder, audio: Any) -> Any:
    """Encode raw audio with an LTXV audio VAE."""
    return vae.encode(audio)


__all__ = ["ltxv_audio_vae_encode"]
