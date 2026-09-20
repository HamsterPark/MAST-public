"""End-to-end operating-mode tip-processing gate for the instrument_control agent.

Drives the safe / semi / auto gate through a REAL compiled IC agent (fake LLM,
InMemorySaver checkpointer), exercising all three middlewares wired by build():

  * SAFE  — SafetyGate Layer-0d hard-blocks BOTH electrical pulses and mechanical
            tip shaping (status=error ToolMessage; skill never executes; no pause).
  * SEMI  — electrical pulses PAUSE for human confirmation (ModeGatedPulseHITL →
            __interrupt__), approve executes / reject skips; a too-deep mechanical
            plunge is blocked (semi_mode_shallow_only); a shallow pure-mechanical
            plunge is allowed autonomously (no block, no pause); a voltage-assisted
            TipShape (change_bias=True) is treated as a pulse → pauses.
  * AUTO  — everything runs straight through (current behaviour).
  * no get_mode — backward-compatible: no gating (pulses run).

Mirrors tests/.../test_hitl_interrupt_resume.py's harness (fake LLM + FakeCtx).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/instrument_control/test_operating_mode_gates.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found above " + str(Path(__file__).resolve()))


_MASTV2_ROOT = _find_mastv2_root()
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
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from mast.agents.instrument_control.graph import build
from mast.core.types import NanonisCallRecord


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(method=method, args=args,
                                     return_value=entry.get("return_value"),
                                     error=entry.get("error", ""))
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def _shared_ctx(canned: dict | None = None):
    ctx = FakeCtx(canned=canned or {})
    return ctx, (lambda: ctx)


_CANNED = {"Bias_Pulse": {"return_value": ("", b"", [])}}


def _biaspulse_llm(bias_v: float = 5.0):
    return _FakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "BiasPulse",
            "args": {"width_s": "10m", "bias_v": str(bias_v)},
            "id": "tc-bp-1", "type": "tool_call"}]),
        AIMessage(content="BiasPulse handled."),
    ]))


def _tipshape_llm(tip_lift_m: float = -2e-9, change_bias: bool = False, **extra):
    return _FakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "TipShape",
            "args": {"tip_lift_m": tip_lift_m, "change_bias": change_bias, **extra},
            "id": "tc-ts-1", "type": "tool_call"}]),
        AIMessage(content="TipShape handled."),
    ]))


def _build(llm, prov, *, get_mode, canned=None):
    return build(buf=None, context_provider=prov, model=llm,
                 checkpointer=InMemorySaver(), enable_hitl=True, get_mode=get_mode)


def _joined(result) -> str:
    return "\n".join(str(getattr(m, "content", "")) for m in result.get("messages", []))


# ── SAFE — hard block, no pause, never executes ──────────────────────────────
class TestSafeMode:
    def test_safe_blocks_bias_pulse(self):
        ctx, prov = _shared_ctx(_CANNED)
        agent = _build(_biaspulse_llm(), prov, get_mode=lambda: "safe")
        cfg = {"configurable": {"thread_id": "safe-bp"}}
        result = agent.invoke({"messages": [("user", "pulse")]}, config=cfg)
        assert not agent.get_state(cfg).next, "SAFE is a hard block, not a pause"
        assert not result.get("__interrupt__")
        assert not any(m == "Bias_Pulse" for m, _ in ctx.calls), "pulse must not execute in SAFE"
        assert "safe_mode_tip_processing_blocked" in _joined(result)

    def test_safe_blocks_tip_shape(self):
        ctx, prov = _shared_ctx({})
        agent = _build(_tipshape_llm(tip_lift_m=-2e-9, change_bias=False), prov, get_mode=lambda: "safe")
        cfg = {"configurable": {"thread_id": "safe-ts"}}
        result = agent.invoke({"messages": [("user", "shape")]}, config=cfg)
        assert not agent.get_state(cfg).next
        assert "safe_mode_tip_processing_blocked" in _joined(result)
        assert not any(m.startswith("TipShaper") for m, _ in ctx.calls), "shaping must not execute in SAFE"


# ── SEMI — pulses confirm; shallow mechanical allowed; deep mechanical blocked ─
class TestSemiMode:
    """⑰(2026-08-08):SEMI 的**电脉冲确认框**没了,机械深度限制原样保留。

    这一节原来有三条测试钉着「SEMI 的脉冲要暂停等人 → approve 执行 / reject 跳过」。
    那个中间件(``ModeGatedPulseHITLMiddleware``)随整条确认框链路一起删除,依据是
    两天实机里这类确认**零真阳性**、把自动运行打成了值守运行。

    **要注意 SEMI 剩下什么**:``test_deep_mechanical_blocked`` 一个字没改,而且它
    才是这个模式真正的保护 —— 那是**拒绝型**的(不弹框、不等人,超过深度直接说不)。
    「SEMI = 半自动」现在的含义是「深压受限」,不是「凡事先问」。

    要把脉冲确认框加回来,需要观测到:一次「确认框拦下了真实损害、而 SafetyGate 的
    深度/包络硬闸接不住」的实例。截至 2026-08-08 零例。
    """

    def test_pulse_runs_without_a_confirmation_dialog(self):
        """**语义反转的正主。** 原名 ``test_pulse_pauses_for_confirmation``。

        同样的输入,断言反过来:不暂停、没有 ``__interrupt__``,而且脉冲**真的打到
        了仪器上**(最后那条断言是关键 —— 只断言「没暂停」的话,一个把工具整个吞掉
        的 bug 也会让它绿)。"""
        ctx, prov = _shared_ctx(_CANNED)
        agent = _build(_biaspulse_llm(), prov, get_mode=lambda: "semi")
        cfg = {"configurable": {"thread_id": "semi-bp-nopause"}}
        result = agent.invoke({"messages": [("user", "pulse")]}, config=cfg)
        assert not agent.get_state(cfg).next, "SEMI pulse 又开始等人确认了"
        assert not result.get("__interrupt__")
        assert any(m == "Bias_Pulse" for m, _ in ctx.calls), (
            "脉冲没有到达仪器 —— 「不再确认」不等于「不再执行」")

    def test_the_pulse_leaves_a_notice_behind(self):
        """执行 + 留痕 + **通知**:少了通知这一半,这条改动就只是「悄悄放开」。

        按 subject 过滤而不是读最新一条:``diagnostics`` 是进程级环形缓冲,同一次
        pytest 里别的文件写进去的行会让「读最新」读到不属于本测试的东西。"""
        from mast.core import diagnostics as diag

        ctx, prov = _shared_ctx(_CANNED)
        agent = _build(_biaspulse_llm(), prov, get_mode=lambda: "semi")
        cfg = {"configurable": {"thread_id": "semi-bp-notice"}}
        agent.invoke({"messages": [("user", "pulse")]}, config=cfg)

        rows = [r for r in diag.recent(200, kinds=("notice_only",))
                if r.get("subject") == "BiasPulse"]
        assert rows, "SEMI 的电脉冲跑了却没有留下任何通知"
        assert "半自动模式下的电脉冲" in rows[0]["reason"]

    def test_deep_mechanical_blocked(self):
        """**这条一个字没改。** SEMI 真正的保护是拒绝型的,⑰ 不碰拒绝型防护。"""
        ctx, prov = _shared_ctx({})
        agent = _build(_tipshape_llm(tip_lift_m=-3e-8, change_bias=False), prov, get_mode=lambda: "semi")
        cfg = {"configurable": {"thread_id": "semi-ts-deep"}}
        result = agent.invoke({"messages": [("user", "deep shape")]}, config=cfg)
        assert not agent.get_state(cfg).next
        assert "semi_mode_shallow_only" in _joined(result), "a 30 nm plunge must be blocked in SEMI"

    def test_shallow_mechanical_allowed(self):
        ctx, prov = _shared_ctx({})
        agent = _build(_tipshape_llm(tip_lift_m=-2e-9, change_bias=False), prov, get_mode=lambda: "semi")
        cfg = {"configurable": {"thread_id": "semi-ts-shallow"}}
        result = agent.invoke({"messages": [("user", "shallow shape")]}, config=cfg)
        # shallow pure-mechanical shaping runs autonomously: not blocked, not paused
        assert not agent.get_state(cfg).next
        j = _joined(result)
        assert "safe_mode" not in j and "semi_mode_shallow_only" not in j

    def test_voltage_assisted_tipshape_runs_and_is_noticed(self):
        """原名 ``test_voltage_assisted_tipshape_pauses``。

        ``change_bias=True`` 让 TipShape 变成带电压的下压 → 归类为电脉冲。这个
        **归类判据**(``is_electrical_pulse`` 看参数,不只看名字)一个字没改 ——
        改的只是它的下游:从「暂停等人」变成「通知」。所以这条测试仍然是那个判据的
        钉子,只是断言换了对象。"""
        from mast.core import diagnostics as diag

        ctx, prov = _shared_ctx({})
        agent = _build(_tipshape_llm(tip_lift_m=-2e-9, change_bias=True), prov, get_mode=lambda: "semi")
        cfg = {"configurable": {"thread_id": "semi-ts-voltage"}}
        result = agent.invoke({"messages": [("user", "voltage shape")]}, config=cfg)
        assert not agent.get_state(cfg).next
        assert not result.get("__interrupt__")
        rows = [r for r in diag.recent(200, kinds=("notice_only",))
                if r.get("subject") == "TipShape"]
        assert rows, "带电压的 TipShape 没有被认成电脉冲 —— 归类判据掉了"
        assert "半自动模式下的电脉冲" in rows[0]["reason"]

    def test_a_purely_mechanical_shallow_shape_is_not_noticed(self):
        """归类判据的**另一侧**:纯机械的浅层 shaping 不是电脉冲,不该发通知。

        只钉「命中会通知」的话,一个恒真的判据也会绿。

        ⚠️ 2026-08-11:「纯机械」的**写法**改了 —— 以前只写 ``change_bias=False``,
        而那**从来就不是纯机械**。TipShaper 有两个 bias 字段,``Bias Lift (V)``
        是**无条件**施加的(厂商:「applied just after the first Z ramping」,
        没有条件从句),``change_bias`` 关不掉它;当时它的默认值还是 **3.0 V**。
        所以这条测试从前断言的「不发通知」是**对的判断、错的理由**:判据确实没
        认出电脉冲,但那一下真的有 3 V。
        要表达「纯机械」,现在必须把两个 bias 都显式写成 0 —— 这也正是判据保留
        分辨力的那一侧。"""
        from mast.core import diagnostics as diag

        ctx, prov = _shared_ctx({})
        before = len([r for r in diag.recent(200, kinds=("notice_only",))
                      if r.get("subject") == "TipShape"])
        agent = _build(_tipshape_llm(tip_lift_m=-2e-9, change_bias=False,
                                     # 「0」不是「0V」:判据看的是**模型原样发来的
                                     # 实参**(SI 解析发生在它之后),而 SI 解析器
                                     # 收的是带前缀的数值串,不带单位符号。
                                     bias_v="0", bias_lift_v="0"), prov,
                       get_mode=lambda: "semi")
        cfg = {"configurable": {"thread_id": "semi-ts-mech-notice"}}
        agent.invoke({"messages": [("user", "shallow shape")]}, config=cfg)
        after = len([r for r in diag.recent(200, kinds=("notice_only",))
                     if r.get("subject") == "TipShape"])
        assert after == before, "纯机械浅层 shaping 被误认成了电脉冲"


# ── AUTO / no-mode — nothing gated ───────────────────────────────────────────
class TestAutoAndDefault:
    def test_auto_runs_pulse(self):
        ctx, prov = _shared_ctx(_CANNED)
        agent = _build(_biaspulse_llm(), prov, get_mode=lambda: "auto")
        cfg = {"configurable": {"thread_id": "auto-bp"}}
        agent.invoke({"messages": [("user", "pulse")]}, config=cfg)
        assert not agent.get_state(cfg).next
        assert any(m == "Bias_Pulse" for m, _ in ctx.calls), "AUTO runs the pulse straight through"

    def test_no_get_mode_runs_pulse(self):
        ctx, prov = _shared_ctx(_CANNED)
        agent = build(buf=None, context_provider=prov, model=_biaspulse_llm(),
                      checkpointer=InMemorySaver(), enable_hitl=True)  # no get_mode
        cfg = {"configurable": {"thread_id": "no-mode-bp"}}
        agent.invoke({"messages": [("user", "pulse")]}, config=cfg)
        assert any(m == "Bias_Pulse" for m, _ in ctx.calls), "no get_mode → no gating (back-compat)"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
