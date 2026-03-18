# Comfy Diffusion Skills

This package contains distributable skill documents (skills 2.0 format) that can be
discovered at runtime with:

```python
from comfy_diffusion.skills import get_skills_path

skills_root = get_skills_path()
```

Each skill lives in its own named subdirectory and contains a `SKILL.md` file with
YAML frontmatter:

```
comfy_diffusion/skills/
└── comfy-diffusion-reference/
    └── SKILL.md
```

These files are part of the installable `comfy_diffusion` package and are independent from the
repository-local `.agents/skills/` developer workflow assets.
