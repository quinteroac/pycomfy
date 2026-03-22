"""Audio helpers."""

from __future__ import annotations

from typing import Any, Protocol, cast


class _LtxvAudioVaeEncoder(Protocol):
    sample_rate: int

    def encode(self, audio: Any) -> Any: ...


class _LtxvAudioVaeDecoder(Protocol):
    output_sample_rate: int

    def decode(self, latent: Any) -> Any: ...


class _LtxvAudioVae(Protocol):
    sample_rate: int
    latent_channels: int
    latent_frequency_bins: int

    def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int: ...


class _AceStep15Clip(Protocol):
    def tokenize(
        self,
        tags: str,
        *,
        lyrics: str,
        bpm: int,
        duration: float,
        timesignature: int,
        language: str,
        keyscale: str,
        seed: int,
        generate_audio_codes: bool,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
    ) -> Any: ...

    def encode_from_tokens_scheduled(self, tokens: Any) -> Any: ...


def _get_ltxv_empty_latent_audio_type() -> Any:
    """Resolve ComfyUI LTXVEmptyLatentAudio node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio

    return LTXVEmptyLatentAudio


def _get_ace_step_15_latent_audio_dependencies() -> tuple[Any, Any]:
    """Resolve torch and ComfyUI model management at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management
    import torch

    return torch, comfy.model_management


def _get_concat_av_latent_dependencies() -> tuple[Any, Any]:
    """Resolve torch and comfy.nested_tensor at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.nested_tensor
    import torch

    return torch, comfy.nested_tensor


def _unwrap_node_output(output: Any) -> Any:
    """Return first output for ComfyUI V3 nodes and tuple-style APIs."""
    if hasattr(output, "result"):
        return output.result[0]
    if isinstance(output, tuple):
        return output[0]
    return output


def ltxv_audio_vae_encode(vae: _LtxvAudioVaeEncoder, audio: Any) -> dict[str, Any]:
    """Encode raw audio with an LTXV audio VAE."""
    audio_latents = vae.encode(audio)
    return {"samples": audio_latents, "sample_rate": int(vae.sample_rate), "type": "audio"}


def ltxv_audio_vae_decode(vae: _LtxvAudioVaeDecoder, latent: Any) -> dict[str, Any]:
    """Decode latent audio with an LTXV audio VAE."""
    latent_tensor = latent["samples"] if isinstance(latent, dict) else latent
    if getattr(latent_tensor, "is_nested", False):
        latent_tensor = latent_tensor.unbind()[-1]
    audio = vae.decode(latent_tensor).to(latent_tensor.device).detach()
    return {"waveform": audio, "sample_rate": int(vae.output_sample_rate)}


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


def encode_ace_step_15_audio(
    clip: _AceStep15Clip,
    tags: str,
    lyrics: str = "",
    seed: int = 0,
    bpm: int = 120,
    duration: float = 120.0,
    timesignature: str = "4",
    language: str = "en",
    keyscale: str = "C major",
    generate_audio_codes: bool = True,
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
) -> Any:
    """Encode ACE Step 1.5 text/audio metadata conditioning."""
    tokens = clip.tokenize(
        tags,
        lyrics=lyrics,
        bpm=bpm,
        duration=duration,
        timesignature=int(timesignature),
        language=language,
        keyscale=keyscale,
        seed=seed,
        generate_audio_codes=generate_audio_codes,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    return clip.encode_from_tokens_scheduled(tokens)


def ltxv_concat_av_latent(
    video_latent: dict[str, Any],
    audio_latent: dict[str, Any],
) -> dict[str, Any]:
    """Concatenate video and audio latents into a single NestedTensor latent for joint denoising."""
    torch, comfy_nested_tensor = _get_concat_av_latent_dependencies()

    output: dict[str, Any] = {}
    output.update(video_latent)
    output.update(audio_latent)

    video_noise_mask = video_latent.get("noise_mask", None)
    audio_noise_mask = audio_latent.get("noise_mask", None)

    if video_noise_mask is not None or audio_noise_mask is not None:
        if video_noise_mask is None:
            video_noise_mask = torch.ones_like(video_latent["samples"])
        if audio_noise_mask is None:
            audio_noise_mask = torch.ones_like(audio_latent["samples"])
        output["noise_mask"] = comfy_nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask))

    output["samples"] = comfy_nested_tensor.NestedTensor((video_latent["samples"], audio_latent["samples"]))
    return output


def ltxv_separate_av_latent(
    av_latent: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate a joint AV NestedTensor latent into individual video and audio latents."""
    video_samples, audio_samples = av_latent["samples"].unbind()

    video_latent: dict[str, Any] = {"samples": video_samples}
    audio_latent: dict[str, Any] = {"samples": audio_samples}

    if "noise_mask" in av_latent:
        video_mask, audio_mask = av_latent["noise_mask"].unbind()
        video_latent["noise_mask"] = video_mask
        audio_latent["noise_mask"] = audio_mask

    return video_latent, audio_latent


def empty_ace_step_15_latent_audio(seconds: float, batch_size: int = 1) -> dict[str, Any]:
    """Create empty ACE Step 1.5 latents used as sampler noise input."""
    torch, model_management = _get_ace_step_15_latent_audio_dependencies()
    length = round(seconds * 48000 / 1920)
    latent = torch.zeros([batch_size, 64, length], device=model_management.intermediate_device())
    return {"samples": latent, "type": "audio"}


__all__ = [
    "ltxv_audio_vae_encode",
    "ltxv_audio_vae_decode",
    "ltxv_empty_latent_audio",
    "encode_ace_step_15_audio",
    "empty_ace_step_15_latent_audio",
    "ltxv_concat_av_latent",
    "ltxv_separate_av_latent",
]
