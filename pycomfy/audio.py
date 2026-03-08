"""Audio helpers."""

from __future__ import annotations

from typing import Any, Protocol, cast


class _LtxvAudioVaeEncoder(Protocol):
    def encode(self, audio: Any) -> Any: ...


class _LtxvAudioVaeDecoder(Protocol):
    def decode(self, latent: Any) -> Any: ...


class _LtxvAudioVae(Protocol):
    sample_rate: int
    latent_channels: int
    latent_frequency_bins: int

    def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int: ...


def _get_ltxv_empty_latent_audio_type() -> Any:
    """Resolve ComfyUI LTXVEmptyLatentAudio node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio

    return LTXVEmptyLatentAudio


def _unwrap_node_output(output: Any) -> Any:
    """Return first output for ComfyUI V3 nodes and tuple-style APIs."""
    if hasattr(output, "result"):
        return output.result[0]
    if isinstance(output, tuple):
        return output[0]
    return output


def ltxv_audio_vae_encode(vae: _LtxvAudioVaeEncoder, audio: Any) -> Any:
    """Encode raw audio with an LTXV audio VAE."""
    return vae.encode(audio)


def ltxv_audio_vae_decode(vae: _LtxvAudioVaeDecoder, latent: Any) -> Any:
    """Decode latent audio with an LTXV audio VAE."""
    return vae.decode(latent)


def ltxv_empty_latent_audio(
    audio_vae: _LtxvAudioVae,
    frames_number: int,
    frame_rate: int = 25,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Create empty LTXV audio latents compatible with ComfyUI's audio pipeline."""
    ltxv_empty_latent_audio_type = _get_ltxv_empty_latent_audio_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            ltxv_empty_latent_audio_type.execute(
                frames_number=frames_number,
                frame_rate=frame_rate,
                batch_size=batch_size,
                audio_vae=audio_vae,
            )
        )
    )


__all__ = [
    "ltxv_audio_vae_encode",
    "ltxv_audio_vae_decode",
    "ltxv_empty_latent_audio",
]
