"""Audio generation pipeline sub-packages for comfy-diffusion.

Each sub-package groups pipelines by model family, mirroring the layout of
``comfyui_official_workflows/audio/``.

Import directly from the specific pipeline module, e.g.::

    from comfy_diffusion.pipelines.audio.ace_step.v1_5.checkpoint import manifest, run
"""

__all__ = ["ace_step"]
