"""Distributable agent skills bundled with comfy_diffusion."""

from importlib.resources import files
from importlib.resources.abc import Traversable


def get_skills_path() -> Traversable:
    """Return the root path of bundled distributable skills.

    Usage::

        from comfy_diffusion.skills import get_skills_path

        skills_root = get_skills_path()
        skill_text = (skills_root / "SKILL.md").read_text(encoding="utf-8")
    """
    return files(__name__)


__all__ = ["get_skills_path"]
