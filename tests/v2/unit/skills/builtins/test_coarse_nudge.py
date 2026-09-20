# -*- coding: utf-8 -*-
"""StepCoarseXY —— 「挪一步」这件事以前两边都过不去。

缺一个「位移一步」的 skill。

* 裸 `MotorMove(x±/y±)` → `is_unguarded_lateral_coarse_move` 硬闸
* `RelocateCoarseXY` → 落点要离已访问站点 ≥200 步

那 200 步是**效率**约束（别重复用同一片表面），它自己的注释就这么写的。
所以修法是给那条效率约束一个明说的出口 `allow_revisit`，
再给它一个名字对得上的入口 —— **不写第二份实现**。
"""
from __future__ import annotations

import pytest

from mast.skills.builtins import StepCoarseXY
from mast.skills.builtins.coarse_nudge import _NUDGE_MAX_STEPS
from mast.skills.composite import RelocateCoarseXY


class _Res:
    def __init__(self, success=True, data=None, error=None):
        self.success = success
        self.data = data or {}
        self.error = error


class _Ctx:
    def __init__(self, inner=None):
        self.inner = inner or _Res(data={"moved_steps": 1})
        self.calls: list[tuple] = []

    def run(self, name, params=None):
        self.calls.append((name, dict(params or {})))
        return self.inner


def _run(ctx, **kw):
    p = {"axis": "x", "direction": "+"}
    p.update(kw)
    return StepCoarseXY().execute(ctx, p)


# ── ★ 委托，不重写 ─────────────────────────────────────────────────────
def test_delegates_to_relocate_and_does_not_touch_the_motor_itself():
    """**不写第二份实现。** 清障/互锁/里程表全靠那一份跑。"""
    ctx = _Ctx()
    res = _run(ctx, steps=1)
    assert res.success is True
    assert [n for n, _ in ctx.calls] == ["RelocateCoarseXY"]
    assert "MotorMove" not in [n for n, _ in ctx.calls]
    assert res.data["delegated_to"] == "RelocateCoarseXY"


def test_it_asks_for_allow_revisit():
    """这是它存在的全部理由 —— 不传就还是被 200 步那条规则挡住。"""
    ctx = _Ctx()
    _run(ctx, steps=2)
    assert ctx.calls[0][1]["allow_revisit"] is True


def test_defaults_to_one_step():
    ctx = _Ctx()
    _run(ctx)
    assert ctx.calls[0][1]["steps"] == 1


def test_passes_axis_direction_through():
    ctx = _Ctx()
    _run(ctx, axis="y", direction="-", steps=3)
    p = ctx.calls[0][1]
    assert (p["axis"], p["direction"], p["steps"]) == ("y", "-", 3)


def test_prewithdraw_is_omitted_when_not_given():
    """留空要让下游用 instrument_profile 的默认，别在这里替它编一个数。"""
    ctx = _Ctx()
    _run(ctx, steps=1)
    assert "prewithdraw_steps" not in ctx.calls[0][1]


def test_prewithdraw_is_passed_when_given():
    ctx = _Ctx()
    _run(ctx, steps=1, prewithdraw_steps=5)
    assert ctx.calls[0][1]["prewithdraw_steps"] == 5


def test_dry_run_reaches_the_inner_skill():
    ctx = _Ctx()
    _run(ctx, steps=1, dry_run=True)
    assert ctx.calls[0][1]["dry_run"] is True


# ── 用途上限 ───────────────────────────────────────────────────────────
def test_big_moves_are_refused_before_anything_happens():
    """大距离是换区，不该走这条明说跳过了效率约束的路 ——
    否则那条约束就被整个架空了。"""
    ctx = _Ctx()
    res = _run(ctx, steps=_NUDGE_MAX_STEPS + 1)
    assert res.success is False
    assert "RelocateCoarseXY" in (res.error or "")
    assert ctx.calls == [], "拒绝要发生在委托之前"


def test_the_limit_is_small_enough_to_stay_inside_a_scan_field():
    """微小粗动应受有限步数上限约束，避免单次指令演变成长距离移动。"""
    assert _NUDGE_MAX_STEPS <= 200, "到 200 步就是 RelocateCoarseXY 的地盘了"


def test_failure_from_the_inner_skill_is_propagated_not_swallowed():
    ctx = _Ctx(inner=_Res(success=False, error="真空互锁未通过", data={"checks": {}}))
    res = _run(ctx, steps=1)
    assert res.success is False
    assert "真空互锁" in (res.error or "")


# ── allow_revisit 在下游确实起作用 ─────────────────────────────────────
def test_relocate_declares_allow_revisit():
    names = {p.name for p in RelocateCoarseXY().metadata().parameters}
    assert "allow_revisit" in names


def test_allow_revisit_short_circuits_the_site_check():
    """★ 这一条钉住出口本身：开了就不再问地图。"""
    sk = RelocateCoarseXY()

    class Boom:
        def safe_call(self, *a, **k):
            raise AssertionError("allow_revisit 开着时不该去读地图")

    out = sk._check_destination(Boom(), {"axis": "x", "direction": "+",
                                         "steps": 2, "allow_revisit": True})
    assert out["blocked"] is False
    assert "allow_revisit" in out["reason"]


def test_without_allow_revisit_the_site_check_still_runs():
    """反面：不开就还得走原来那条路，否则上一条会因为「永远短路」而空过。"""
    sk = RelocateCoarseXY()
    calls = []

    class Ctx:
        def safe_call(self, *a, **k):
            calls.append(a)
            raise RuntimeError("no map here")

    out = sk._check_destination(Ctx(), {"axis": "x", "direction": "+", "steps": 2})
    # 地图读不到时降级放行，但**理由不同** —— 说明它真的走了那条路
    assert out["blocked"] is False
    assert "allow_revisit" not in out.get("reason", "")


def test_metadata_is_sane():
    m = StepCoarseXY().metadata()
    assert m.name == "StepCoarseXY"
    assert m.safety_level is not None
    steps = next(p for p in m.parameters if p.name == "steps")
    assert steps.min_value == 1
    assert steps.max_value == _NUDGE_MAX_STEPS
