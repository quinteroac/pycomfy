"""LTX-Video 2 pipeline modules.

Available pipelines
-------------------
t2v
    Text-to-video with the LTX-Video 2 dev model.
t2v_distilled
    Text-to-video with the LTX-Video 2 distilled checkpoint (fewer steps).
i2v
    Image-to-video with the LTX-Video 2 dev fp8 model + distilled LoRA.
i2v_distilled
    Image-to-video with the LTX-Video 2 distilled checkpoint (no LoRA).
i2v_lora
    Image-to-video with an additional caller-supplied style LoRA.
canny
    Canny-to-video with the LTX-Video 2 dev fp8 model + Canny control LoRA.
depth
    Depth-to-video with Lotus depth estimation + LTX-Video 2 dev fp8 model.

Usage::

    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest, run
"""

__all__ = ["t2v", "t2v_distilled", "i2v", "i2v_distilled", "i2v_lora", "canny", "depth"]
