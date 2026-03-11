# Comfy Diffusion Skills

This package contains distributable skill documents that can be discovered at runtime with:

```python
from importlib.resources import files

skills_root = files("comfy_diffusion.skills")
```

These files are part of the installable `comfy_diffusion` package and are independent from the
repository-local `.agents/skills/` developer workflow assets.

