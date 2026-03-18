"""Distributable agent skills bundled with comfy_diffusion."""

from importlib.resources import files
from importlib.resources.abc import Traversable


def get_skills_path() -> Traversable:
    """Return the root path of bundled distributable skills.

    Each skill lives in its own named subdirectory containing a ``SKILL.md``
    file with YAML frontmatter (skills 2.0 format).

    Usage::

        from comfy_diffusion.skills import get_skills_path

        skills_root = get_skills_path()
        # list available skills
        skill_dirs = [p for p in skills_root.iterdir() if not p.name.startswith("_")]
        # read a specific skill
        skill_text = (skills_root / "comfy-diffusion-reference" / "SKILL.md").read_text(encoding="utf-8")
    """
    return files(__name__)


__all__ = ["get_skills_path"]
