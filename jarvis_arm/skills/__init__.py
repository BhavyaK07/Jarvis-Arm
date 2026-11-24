from . import home, pick, place, scan, talk
from .context import SkillContext
from .registry import SkillRegistry, SkillResult


def build_registry() -> SkillRegistry:
    reg = SkillRegistry()
    for mod in (scan, pick, place, home, talk):
        mod.register(reg)
    return reg


__all__ = ["SkillContext", "SkillRegistry", "SkillResult", "build_registry"]
