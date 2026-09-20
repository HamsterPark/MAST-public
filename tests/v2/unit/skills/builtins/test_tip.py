"""v2 unit tests for mast.skills.builtins.tip.

Skills covered: SafeRetract (AUTO), EmergencyRetract (AUTO) — 2 skills total.

## SafeRetract 的确认式退针(2026-08-14,修复项)

从前它发出 ``ZCtrl_Withdraw(0, 1)`` 就报 ``retracted: True`` —— ``(0, 1)`` 是
「不等待、1 ms 超时」,命令一发出压电还在往上爬。它验证的是**发过命令**,
不是**动作生效**;而它正是 planner 提示词里「出事就退针」的首选技能。

现在下发之后要有界轮询 ``core.tip_park.tip_parked`` 回读确认,``retracted`` 三态:

* ``True``  确认到位;
* ``False`` 读得到状态、状态说没到位(**确定的否定** ⇒ ``success=False``);
* ``None``  判不了(``success=True`` + 一句「已下发未确认」)。

**「没查」永远不折叠成 ``True``。** 而「读不到」也不折叠成「失败」——
把读不到当成出故障是本仓在案的旧错(那次它退了一次不该退的针)。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_tip.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.tip import EmergencyRetract, SafeRetract


# ── FakeCtx ──────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def make_provider(canned: dict[str, Any] | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# ── 确认式退针的共用件 ────────────────────────────────────────────────────────
#
# 独立合成 Z 全程与收回端，方向由测试配置显式提供。
Z_FULL_M = 400e-9
RAIL_HIGH_M = +200e-9
ENGAGED_Z_M = -120e-9          # 还在隧穿区,离收回端 289 nm


def _ok(*vals):
    return {"return_value": ("", b"", list(vals))}


def canned_rig(*, feedback_on: int = 0, z_m: float = RAIL_HIGH_M,
               module_code: int = 1) -> dict[str, Any]:
    canned = {
        "ZCtrl_Withdraw": _ok(),
        "ZCtrl_OnOffGet": _ok(feedback_on),
        "ZCtrl_StatusGet": _ok(module_code),
        "ZCtrl_ZPosGet": _ok(z_m),
        "Piezo_RangeGet": _ok(1e-6, 1e-6, Z_FULL_M),
        "ZCtrl_LimitsGet": _ok(RAIL_HIGH_M, -RAIL_HIGH_M),
        "ZCtrl_LimitsEnabledGet": _ok(0),
    }
    return canned


@pytest.fixture(autouse=True)
def _fast_confirmation(monkeypatch):
    """确认预算清零 ⇒ do-while 只读一次,墙钟不进测试。

    真值是 5 s;这里调到 0 不是在测超时长度,是为了让「读不到就等满预算」这条
    真实行为不把测试拖成 5 s 一条。要测「轮询确实在轮询」的用例自己调回来。
    """
    monkeypatch.setattr(SafeRetract, "_confirm_budget_s", 0.0)
    monkeypatch.setattr(SafeRetract, "_confirm_poll_s", 0.0)


@pytest.fixture
def declared_sign(monkeypatch):
    """合成配置显式声明 z_extend_sign=-1，用于检验退针方向。"""
    import mast.core.tip_park as tp
    monkeypatch.setattr(tp, "z_extend_sign_or_none", lambda: -1)


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_safe_retract_shape():
    tool = wrap_skill(SafeRetract, make_provider())
    assert tool.name == "SafeRetract"
    assert tool.metadata["danger_level"] == "AUTO"
    # No required parameters
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_emergency_retract_shape():
    tool = wrap_skill(EmergencyRetract, make_provider())
    assert tool.name == "EmergencyRetract"
    # AUTO: an emergency tip retract only moves the tip AWAY from the sample
    # (fine-Z withdraw), which can never damage the instrument, so it runs ungated.
    assert tool.metadata["danger_level"] == "AUTO"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_skill_source_points_to_tip_module():
    tool = wrap_skill(SafeRetract, make_provider())
    assert tool.metadata["skill_source"].endswith(".tip")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_safe_retract_executes():
    canned = {"ZCtrl_Withdraw": {"return_value": ("", b"", [])}}
    tool = wrap_skill(SafeRetract, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["SafeRetract"]


def test_safe_retract_calls_correct_method():
    canned = {"ZCtrl_Withdraw": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SafeRetract, capturing_provider)
    _invoke(tool)
    last_ctx = instances[-1]
    withdraw_calls = [c for c in last_ctx.calls if c[0] == "ZCtrl_Withdraw"]
    assert len(withdraw_calls) == 1
    assert withdraw_calls[0][1] == (0, 1)


def test_emergency_retract_executes():
    canned = {
        "ZCtrl_OnOffSet": {"return_value": ("", b"", [])},
        "Scan_Action": {"return_value": ("", b"", [])},
        "ZCtrl_Withdraw": {"return_value": ("", b"", [])},
    }
    tool = wrap_skill(EmergencyRetract, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["EmergencyRetract"]


def test_emergency_retract_calls_three_methods():
    canned = {
        "ZCtrl_OnOffSet": {"return_value": ("", b"", [])},
        "Scan_Action": {"return_value": ("", b"", [])},
        "ZCtrl_Withdraw": {"return_value": ("", b"", [])},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(EmergencyRetract, capturing_provider)
    _invoke(tool)
    last_ctx = instances[-1]
    methods = [c[0] for c in last_ctx.calls]
    assert "ZCtrl_OnOffSet" in methods
    assert "Scan_Action" in methods
    assert "ZCtrl_Withdraw" in methods


# ── 确认式退针:三态 ───────────────────────────────────────────────────────────

def test_confirmed_retract_reports_true_with_evidence(declared_sign):
    """针停在收回端、反馈已断 ⇒ True,而且证据跟着结论走。"""
    res = SafeRetract().execute(FakeCtx(canned=canned_rig()), {})
    assert res.success is True
    assert res.data["retracted"] is True
    assert res.data["park"]["state"] == "parked"
    assert "nm" in res.data["park"]["evidence"]


def test_still_engaged_after_the_budget_is_a_definite_no(declared_sign):
    """回读状态明确未到位时，应同时返回 retracted=False 和技能失败。"""
    ctx = FakeCtx(canned=canned_rig(z_m=ENGAGED_Z_M))
    res = SafeRetract().execute(ctx, {})
    assert res.data["retracted"] is False
    assert res.success is False
    # 两种可能都说,不替仪器断定是哪一种。
    assert "还在走" in res.error and "没生效" in res.error
    # 别把它说成一次成功的退针。
    assert res.data["park"]["state"] == "not_parked"


def test_unreadable_is_dispatched_but_unconfirmed_never_true(declared_sign):
    """判不了 ⇒ None + 一句「已下发未确认」。读不到既不是退到了也不是没退到。

    符号显式钉成「已声明」,所以这里测的是**读失败**那一种 unreadable
    (另一种「本机没声明」由下一条单独测)—— 否则同一条断言在两台机器上
    可能因为完全不同的原因通过。
    """
    # canned 里只有 ZCtrl_Withdraw:回读全部读不到(FakeCtx 对未列出的 verb 报错)。
    res = SafeRetract().execute(FakeCtx(canned={"ZCtrl_Withdraw": _ok()}), {})
    assert res.data["retracted"] is None
    assert res.data["retracted"] is not True     # 「没查」永不折叠成 True
    assert "已下发未确认" in (res.summary or "")
    # 「读不到」不折叠成「出故障」:判据链断了不构成「退针失败」这个断言。
    assert res.success is True
    assert res.data["park"]["state"] == "unreadable"
    assert "feedback_on" in res.data["park"]["unreadable"]
    assert res.data["park"]["undeclared"] == []


def test_undeclared_sign_short_circuits_instead_of_burning_the_budget(monkeypatch):
    """本机没声明 ``z_extend_sign`` ⇒ 立刻收工,只读一次。

    等多久都不会变的东西不该被报成「超时未确认」—— 那会把人指向没坏的东西。
    """
    import mast.core.tip_park as tp
    monkeypatch.setattr(tp, "z_extend_sign_or_none", lambda: None)
    monkeypatch.setattr(SafeRetract, "_confirm_budget_s", 30.0)  # 预算充足
    ctx = FakeCtx(canned=canned_rig())
    res = SafeRetract().execute(ctx, {})
    assert res.data["retracted"] is None
    assert res.data["park"]["undeclared"] == ["z_extend_sign"]
    assert res.data["confirm_waited_s"] < 1.0, "没声明的配置项被当成瞬时故障轮询了"
    assert len([c for c in ctx.calls if c[0] == "ZCtrl_ZPosGet"]) == 1


def test_confirmation_actually_polls_until_the_tip_arrives(monkeypatch, declared_sign):
    """压电还在爬的那几百毫秒必须等 —— 否则这个修复只是换了个地方报早。"""
    monkeypatch.setattr(SafeRetract, "_confirm_budget_s", 2.0)
    monkeypatch.setattr(SafeRetract, "_confirm_poll_s", 0.0)

    climbing = FakeCtx(canned=canned_rig(z_m=ENGAGED_Z_M))
    reads = {"n": 0}
    base_safe_call = climbing.safe_call

    def climbing_safe_call(method, *args, **kw):
        if method == "ZCtrl_ZPosGet":
            reads["n"] += 1
            z = ENGAGED_Z_M if reads["n"] < 3 else RAIL_HIGH_M
            climbing.calls.append((method, args))
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [z]))
        return base_safe_call(method, *args, **kw)

    climbing.safe_call = climbing_safe_call  # type: ignore[method-assign]
    res = SafeRetract().execute(climbing, {})
    assert res.data["retracted"] is True
    assert reads["n"] >= 3, "一次就收工了,根本没轮询"


def test_withdraw_is_dispatched_once_even_with_the_confirmation_loop(declared_sign):
    """轮询是**只读**的:退针命令只该发一次,不能每轮补一发。"""
    ctx = FakeCtx(canned=canned_rig(z_m=ENGAGED_Z_M))
    SafeRetract().execute(ctx, {})
    assert len([c for c in ctx.calls if c[0] == "ZCtrl_Withdraw"]) == 1


def test_failed_withdraw_call_is_not_a_confirmed_retract():
    canned = canned_rig()
    canned["ZCtrl_Withdraw"] = {"error": "TCP timeout"}
    res = SafeRetract().execute(FakeCtx(canned=canned), {})
    assert res.success is False
    assert res.data["retracted"] is None      # 没发出去,更谈不上到位
    assert res.data["retracted"] is not False  # 也不是「读了,没到位」


# ── 变异验证:先证明变异生效,再看测试红 ──────────────────────────────────────

def test_mutation_a_predicate_that_always_says_parked_flips_the_timeout_case(
        monkeypatch, declared_sign):
    """把谓词改成恒报 ``parked``,``test_still_engaged_after_the_budget_is_a_definite_no``
    的断言就必须站不住 —— 否则那条测试测的不是它自称的东西。

    两步都要做:**先证明变异确实落到了被调用的那个名字上**,再证明结论翻了。
    """
    import mast.skills.builtins.tip as tip_mod
    from mast.core.tip_park import PARKED, ParkVerdict

    def always_parked(context) -> ParkVerdict:
        return ParkVerdict(
            state=PARKED, reason="(变异)恒报已到位", feedback_on=False,
            module_status="Off", z_m=ENGAGED_Z_M, rail_m=RAIL_HIGH_M,
            rail_side="high", gap_m=0.0, tolerance_m=1e-9,
            travel_source="(变异)", unreadable=(), undeclared=(), read_at=0.0,
        )

    monkeypatch.setattr(tip_mod, "tip_parked", always_parked)
    # ① 变异已应用:SafeRetract 用的就是这个名字。
    assert tip_mod.tip_parked(None).state == PARKED

    # ② 同一个「针还在隧穿区」的场景,结论翻了 ⇒ 原测试会红。
    res = SafeRetract().execute(FakeCtx(canned=canned_rig(z_m=ENGAGED_Z_M)), {})
    assert res.data["retracted"] is True
    assert res.success is True


# ── 消费方 ────────────────────────────────────────────────────────────────────

def test_emergency_retract_still_reports_a_plain_true(declared_sign):
    """``EmergencyRetract`` 有意不动:紧急路径的判断保持独立,不与常规路径共享失败模式。"""
    canned = {
        "ZCtrl_OnOffSet": _ok(),
        "Scan_Action": _ok(),
        "ZCtrl_Withdraw": _ok(),
    }
    res = EmergencyRetract().execute(FakeCtx(canned=canned), {})
    assert res.data == {"retracted": True, "emergency": True}


def test_watchdog_does_not_go_through_this_skill():
    """看门狗自己发 ``ZCtrl_Withdraw(1, -1)`` 并自己确认(``core/executor.py``),
    **不调这个技能** —— 所以三态语义不会顺着紧急路径漂过去。

    日志里那句 "executing SafeRetract via emergency port" 说的是动作,不是技能;
    有人照字面把它改成调用本技能时,这条会拦住。
    """
    import inspect

    from mast.core import executor as executor_mod

    src = inspect.getsource(executor_mod)
    assert "ZCtrl_Withdraw" in src
    for wired in ("SafeRetract()", "SafeRetract().execute", "wrap_skill(SafeRetract"):
        assert wired not in src, f"看门狗接上了常规退针技能:{wired}"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
