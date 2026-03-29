"""Pipeline modules for comfy-diffusion.

Each sub-package groups pipelines by model family, mirroring the layout of
``comfyui_official_workflows/``.

Import directly from the specific pipeline module, e.g.::

    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest, run
    from comfy_diffusion.pipelines.video.ltx.ltx23.i2v import manifest, run

Sub-packages
------------
video
    Video generation pipelines organised by model family.
"""

__all__ = ["video"]
