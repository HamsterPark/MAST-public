"""修复项 (2026-06-11): composite store mutations hot-(un)register in the LIVE registry.

Before this fix load_spec_skills() ran only at startup, so:
  * a freshly cloned composite was NOT runnable until restart,
  * a rollback left the OLD version registered,
  * a delete left a ghost skill callable by agents.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/gui/test_composite_panel_hot_register.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.core.registry import SkillRegistry
from mast.webui import composite_panel as cp
from mast.skills.composite.spec import CompositeSpec, ParamSpec
from mast.skills.composite.version_store import CompositeVersionStore


def _spec(name="Demo", desc="d"):
    return CompositeSpec(
        name=name, description=desc, safety_level="confirm",
        params=[ParamSpec("n", "int", 2)],
        nodes=[{"type": "step", "id": "a", "skill": "GetBias", "params": {}}],
    )


@pytest.fixture()
def wired(tmp_path):
    """Fresh store + fresh live registry wired into composite_panel; module
    globals restored afterwards so other tests see a clean panel."""
    store = CompositeVersionStore(root=tmp_path / "composite_skills")
    reg = SkillRegistry()
    old_store, old_reg = cp._store, cp._live_registry
    cp._store = store
    cp.set_live_registry(reg)
    try:
        yield store, reg
    finally:
        cp._store = old_store
        cp._live_registry = old_reg


def test_clone_hot_registers(wired):
    store, reg = wired
    store.save(_spec("Demo"))
    msg = cp.clone_composite("Demo", "Demo2")
    assert "已克隆" in msg and "热注册" in msg
    assert reg.has("Demo2")
    assert reg.get("Demo2")().metadata().name == "Demo2"


def test_delete_hot_unregisters(wired):
    store, reg = wired
    store.save(_spec("Demo"))
    cp.clone_composite("Demo", "Doomed")
    assert reg.has("Doomed")
    msg = cp.delete_composite("Doomed")
    assert "已删除" in msg and "移除" in msg
    assert not reg.has("Doomed")  # no ghost skill


def test_restore_hot_registers_rolled_back_content(wired):
    store, reg = wired
    store.save(_spec("Demo", desc="v1"))
    store.save(_spec("Demo", desc="v2"))
    msg = cp.restore_composite("Demo", 1)
    assert "已回滚" in msg and "热注册" in msg
    assert reg.has("Demo")
    # restore writes forward as v3 → registry's latest must carry v1 content
    meta = reg.get("Demo")().metadata()
    assert meta.version == "3.0.0"
    assert meta.description == "v1"


def test_unwired_registry_is_silent_noop(wired, tmp_path):
    store, _reg = wired
    store.save(_spec("Demo"))
    cp.set_live_registry(None)
    msg = cp.clone_composite("Demo", "Demo3")
    assert "已克隆" in msg
    assert "热注册" not in msg  # no registry → no suffix, no crash


# ── name-collision guard: a composite named like a builtin must neither
#    shadow it on register nor DEREGISTER it on delete ──────────────────────

from mast.core.types import SkillCategory, SkillMetadata, SkillResult  # noqa: E402
from mast.skills.base import BaseSkill  # noqa: E402


class _FakeBuiltin(BaseSkill):
    def metadata(self):
        return SkillMetadata(name="Collide", version="1.0.0",
                             category=SkillCategory.READ, description="")
    def execute(self, ctx, params):
        return SkillResult(skill_name="Collide", success=True)


def test_register_spec_refuses_collision_with_non_spec_skill(wired):
    from mast.skills.composite.loader import register_spec
    store, reg = wired
    reg.register(_FakeBuiltin)
    with pytest.raises(ValueError, match="collides"):
        register_spec(reg, _spec("Collide"))
    assert reg.get("Collide") is _FakeBuiltin  # builtin untouched


def test_delete_never_deregisters_colliding_builtin(wired):
    store, reg = wired
    reg.register(_FakeBuiltin)
    store.save(_spec("Collide"))           # store accepts the name…
    msg = cp.delete_composite("Collide")   # …but unregister must refuse
    assert "已删除" in msg
    assert "移除" not in msg
    assert reg.has("Collide") and reg.get("Collide") is _FakeBuiltin


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
