"""A duplicate skill name silently DELETES a skill. Make it fatal.

``SkillRegistry.discover`` logs a warning on a name+version collision and then
overwrites the entry. Nothing fails; the tool count barely moves; the tests that
exercise the *other* skill keep passing because a skill by that name still exists.
The original — with all its behaviour — is simply gone.

That happened for real on 2026-07-13: a new ``GetLockInConfig`` in readback.py
overwrote the existing one in lockin.py, which had careful scalar parsing of the
amplitude/frequency/phase. The only evidence was one WARNING line in a log nobody
reads. This test turns that warning into a failure.

Names are also the gate key for the optional-hardware modules and for the HITL map,
so a collision does not merely lose behaviour — it can silently move a skill from
one safety class to another.
"""

from __future__ import annotations

from collections import Counter

from mast.agents.instrument_control.tools import discover_instrument_skills


def test_no_two_skills_share_a_name():
    """SkillRegistry keys on name — a repeat is a deletion, not a duplication."""
    registry = discover_instrument_skills()
    names = [m.name for m in registry.list_skills()]
    dupes = sorted(n for n, c in Counter(names).items() if c > 1)
    assert not dupes, (
        f"skill 重名：{dupes}。SkillRegistry 只打一条 WARNING 然后覆盖——"
        "先注册的那个连同它的全部行为就此消失，而且什么都不会失败。"
    )


def test_discovery_registers_every_class_it_finds():
    """The count check the collision above would have caught.

    Walk the builtins package and count the concrete BaseSkill subclasses; every one
    of them must be in the registry. If two classes claim the same name, the registry
    will hold fewer skills than there are classes — which is exactly the symptom, and
    exactly what nobody noticed.
    """
    import importlib
    import inspect
    import pkgutil

    import mast.skills.builtins as builtins_pkg
    from mast.skills.base import BaseSkill

    classes: dict[str, list[str]] = {}
    for mod_info in pkgutil.iter_modules(builtins_pkg.__path__):
        mod = importlib.import_module(f"mast.skills.builtins.{mod_info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, BaseSkill) and obj is not BaseSkill
                    and obj.__module__ == mod.__name__
                    and not inspect.isabstract(obj)):
                try:
                    name = obj().metadata().name
                except Exception:  # noqa: BLE001 — a skill whose metadata() throws
                    continue       # is a different bug; test_hardware_modules covers it
                classes.setdefault(name, []).append(f"{mod_info.name}.{obj.__name__}")

    collisions = {n: where for n, where in classes.items() if len(where) > 1}
    assert not collisions, (
        "两个 skill 类声明了同一个 name（后注册的会覆盖先注册的）：\n"
        + "\n".join(f"  {n}: {' 与 '.join(w)}" for n, w in sorted(collisions.items()))
    )

    registry = discover_instrument_skills()
    registered = {m.name for m in registry.list_skills()}
    missing = sorted(set(classes) - registered)
    assert not missing, f"这些 skill 类没能注册进 registry：{missing}"
