"""Pipeline modules for comfy-diffusion.

Each submodule in this package is a self-contained pipeline that exports:
- ``manifest() -> list[ModelEntry]``  — the models required by the pipeline.
- ``run(...)``                         — execute the full inference pipeline.

The package itself exports nothing; import from the specific pipeline module.
"""
