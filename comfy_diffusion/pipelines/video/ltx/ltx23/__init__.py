"""LTX-Video 2.3 (22B dev fp8) pipeline modules.

All three pipelines use the dev fp8 checkpoint (``ltx-2.3-22b-dev-fp8.safetensors``)
with the distilled LoRA and Gemma text-encoder LoRA applied at runtime.  They
follow a two-pass sampling chain: a full denoising pass followed by spatial
upscaling and a short refinement pass.

Available pipelines
-------------------
t2v
    Text-to-video-with-audio.  Mirrors ``video_ltx2_3_t2v.json``.
i2v
    Image-to-video-with-audio.  Mirrors ``video_ltx2_3_i2v.json``.
flf2v
    First-last-frame-to-video-with-audio.  Generates a smooth video transition
    between two guide images.  Mirrors ``video_ltx2_3_flf2v.json``.

Not yet implemented
-------------------
ia2v
    Image+audio-to-video.  Mirrors ``video_ltx2_3_ia2v.json``.  Requires
    ``load_audio`` and ``trim_audio`` helpers not yet available in the
    ``comfy_diffusion`` public API.

Usage::

    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest, run
    from comfy_diffusion.pipelines.video.ltx.ltx23.i2v import manifest, run
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import manifest, run
"""

__all__ = ["t2v", "i2v", "flf2v"]
