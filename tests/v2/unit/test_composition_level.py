"""composition_level invariants — every registered skill must have a sane
composition_level (0..5), and the level must be consistent with the folder
the skill lives in. This is a forward-compat gate: new skills that forget
to set composition_level get caught here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if _MASTV2_ROOT not in sys.path:
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.core.registry import SkillRegistry  # noqa: E402


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    r = SkillRegistry()
    r.discover("mast.skills.builtins", "mast.skills.composite", "mast.skills.paper")
    return r


def test_every_skill_has_valid_level(registry):
    bad: list[tuple[str, int]] = []
    for meta in registry.list_skills():
        if not (0 <= meta.composition_level <= 5):
            bad.append((meta.name, meta.composition_level))
    assert not bad, f"out-of-range composition_level: {bad}"


def test_composite_skills_at_least_L2(registry):
    """Anything living in mast/skills/composite/ orchestrates other skills,
    so its level must be ≥ 2."""
    violators: list[tuple[str, int]] = []
    for meta in registry.list_skills():
        cls = registry.get(meta.name)
        mod = cls.__module__
        if ".composite." in mod and meta.composition_level < 2:
            violators.append((meta.name, meta.composition_level))
    assert not violators, f"composite skills under L2: {violators}"


def test_paper_skills_at_least_L3(registry):
    """Paper-replication / RL skills (mast/skills/paper/) are multi-stage
    autonomous procedures; level must be ≥ 3."""
    violators: list[tuple[str, int]] = []
    for meta in registry.list_skills():
        cls = registry.get(meta.name)
        mod = cls.__module__
        if ".paper." in mod and meta.composition_level < 3:
            violators.append((meta.name, meta.composition_level))
    assert not violators, f"paper skills under L3: {violators}"


def test_registry_is_populated(registry):
    """Catch a global import failure that would silently empty the registry."""
    n = len(registry.list_skills())
    assert n >= 200, f"expected ≥200 registered skills, got {n}"


def test_level_distribution_is_reasonable(registry):
    """At least some skills at L0 (atomic wrappers), L3 (composites), and
    L4 (paper). Catches a regression where everything ends up at L0."""
    from collections import Counter
    counts = Counter(m.composition_level for m in registry.list_skills())
    assert counts.get(0, 0) > 50, f"too few L0 atomic skills: {counts}"
    assert counts.get(3, 0) >= 5, f"too few L3 composite skills: {counts}"
    assert counts.get(4, 0) >= 5, f"too few L4 paper skills: {counts}"
