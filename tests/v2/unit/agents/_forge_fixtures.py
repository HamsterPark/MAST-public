"""技能工坊测试的共用零件。

注册表的 discover() 要走 ~460 个技能,每个测试各建一次就太贵了 —— 建一次缓存住,
但**版本库每个测试各给一份 tmp**(共用 store 会让「保存」互相看见,而 CAS 与
「不许改别人的作品」这两条恰恰是靠 store 的内容判的)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core.registry import SkillRegistry  # noqa: E402
from mast.core.types import SkillResult  # noqa: E402

_REG: SkillRegistry | None = None


def registry() -> SkillRegistry:
    """A registry with the real builtin + composite skills (built once)."""
    global _REG
    if _REG is None:
        reg = SkillRegistry()
        reg.discover("mast.skills.builtins", "mast.skills.composite")
        _REG = reg
    return _REG


def fresh_registry() -> SkillRegistry:
    """A private copy — for tests that register into it (hot-register, twins)."""
    reg = SkillRegistry()
    reg.discover("mast.skills.builtins", "mast.skills.composite")
    return reg


def store(tmp_path) -> "object":
    from mast.skills.composite.version_store import CompositeVersionStore
    return CompositeVersionStore(Path(tmp_path))


class StubCtx:
    """An ExecutionContext stand-in that records every ``run`` (sub-step).

    Only ``run`` matters here: a composite reaches the instrument exclusively
    through ``ExecutionContext.run``, so recording it *is* recording what would
    have happened. ``safe_call`` raises on purpose — a composite that talks to
    the instrument directly would be bypassing the收口, and that must fail loud.
    """

    def __init__(self, *, aborted: bool = False, fail: set[str] | None = None):
        self.ran: list[tuple[str, dict]] = []
        self.state = None
        self._aborted = aborted
        self._fail = fail or set()

    def check_abort(self) -> bool:
        return self._aborted

    def abort_reason(self) -> str:
        return "测试里挂的中止" if self._aborted else ""

    def run(self, skill_name, params=None, **kw):
        self.ran.append((skill_name, dict(params or {})))
        if skill_name in self._fail:
            return SkillResult(skill_name=skill_name, success=False,
                               error=f"{skill_name} refused by the stub")
        return SkillResult(skill_name=skill_name, success=True,
                           data={"fft_quality": 0.9}, summary="ok")

    def safe_call(self, method, *a, **k):  # pragma: no cover — must never run
        raise AssertionError(
            f"composite 直接调了 safe_call({method!r}) —— 那绕过了 "
            "ExecutionContext.run 这个收口")


def tools(reg=None, ctx=None, st=None, **kw) -> dict:
    from mast.agents._shared.skill_forge_tools import make_skill_forge_tools
    reg = reg if reg is not None else registry()
    provider = (lambda: ctx) if ctx is not None else None
    built = make_skill_forge_tools(reg, provider, store=st, **kw)
    return {t.name: t for t in built}


def call(tool, **kw) -> dict:
    """Call a forge tool and parse its JSON envelope."""
    return json.loads(tool.func(**kw))


# ── spec 素材 ────────────────────────────────────────────────────────

def two_step_spec(name: str = "ScanThenCheck") -> dict:
    """一份合法的两步 spec:扫一张 + 判读质量。"""
    return {
        "name": name, "description": "扫一张并判读质量",
        "safety_level": "confirm",
        "params": [{"name": "x_m", "type": "number", "required": True},
                   {"name": "y_m", "type": "number", "required": True}],
        "nodes": [
            {"type": "step", "id": "scan", "skill": "ScanAt",
             "params": {"center_x_m": {"$expr": "x_m"},
                        "center_y_m": {"$expr": "y_m"}, "size_m": 1e-7}},
            {"type": "step", "id": "q", "skill": "AssessImageQuality", "params": {}},
        ],
    }


def spec_with(nodes: list, name: str = "T", **kw) -> dict:
    d = {"name": name, "description": "d", "safety_level": "confirm",
         "params": [], "nodes": nodes}
    d.update(kw)
    return d
