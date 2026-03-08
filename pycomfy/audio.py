"""Audio helpers."""

from __future__ import annotations

from typing import Any, Protocol


class _LtxvAudioVaeEncoder(Protocol):
    def encode(self, audio: Any) -> Any: ...


class _LtxvAudioVaeDecoder(Protocol):
    def decode(self, latent: Any) -> Any: ...


def ltxv_audio_vae_encode(vae: _LtxvAudioVaeEncoder, audio: Any) -> Any:
    """Encode raw audio with an LTXV audio VAE."""
    return vae.encode(audio)


def ltxv_audio_vae_decode(vae: _LtxvAudioVaeDecoder, latent: Any) -> Any:
    """Decode latent audio with an LTXV audio VAE."""
    return vae.decode(latent)


__all__ = ["ltxv_audio_vae_encode", "ltxv_audio_vae_decode"]
