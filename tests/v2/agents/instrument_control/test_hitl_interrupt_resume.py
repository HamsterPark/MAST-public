"""instrument_control 的 DANGEROUS 技能：从「暂停等人批」到「照跑并通知」。

## ⑷（2026-08-08）这份文件的语义变了

它原本在 LangGraph 层钉 **HITL interrupt/resume 机制**：假 LLM 发一个被
``interrupt_on`` 门控的技能调用 → agent（带 InMemorySaver +
HumanInTheLoopMiddleware）**暂停**在 ``__interrupt__`` 而不执行；随后的
``Command(resume=...)`` 驱动四个分支（approve / reject / edit / edit→超限）。

那条链路整个被割掉了（定案，依据是两天实机里确认框**零真阳性**），
所以四个裁决分支合并成两条断言：**它跑了**、**它留下了通知**。

## 什么没变（本文件同样钉着）

``TestDefaultBuildAndCoarseApproachGate`` 一个字没改：默认构建不暂停，
而那一个真正危险的动作 —— 开环粗动 Z 向样品进针
（``MotorMove direction='z-approach'``）—— 仍然由 SafetyGateMiddleware
**fail-closed 拒绝**。那是**拒绝型防护**：不弹框、不等人、越界就说不。
⑷ 割的是「等人」，不是「说不」。
"""
from __future__ import annotations

# ── path bootstrap (robust for any test depth) ───────────────────────
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
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
from langchain_core.messages import AIMessage, ToolMessage
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
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def _shared_ctx(canned: dict | None = None):
    """A single FakeCtx instance shared across calls so we can inspect .calls."""
    ctx = FakeCtx(canned=canned or {})
    return ctx, (lambda: ctx)


# Force the gate ON for BiasPulse regardless of production metadata, so these
# tests pin the interrupt/resume MECHANISM (Step 1-4) independent of the
# metadata flip (Step 5).
_FORCE_BIASPULSE_HITL = {"BiasPulse": {"allowed_decisions": ["approve", "edit", "reject"]}}


def _biaspulse_llm(bias_v: float = 5.0):
    return _FakeChatModel(messages=iter([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "BiasPulse",
                "args": {"width_s": "10m", "bias_v": str(bias_v)},
                "id": "tc-bp-1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="BiasPulse handled."),
    ]))


def _build_agent(llm, ctx_provider):
    return build(
        buf=None,
        context_provider=ctx_provider,
        model=llm,
        checkpointer=InMemorySaver(),
        enable_hitl=True,
        interrupt_on=_FORCE_BIASPULSE_HITL,
    )


_CANNED = {"Bias_Pulse": {"return_value": ("", b"", [])}}


def _motormove_llm(direction: str = "z-approach", steps: int = 50):
    """A fake LLM that emits a single MotorMove tool_call, then a final answer."""
    return _FakeChatModel(messages=iter([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "MotorMove",
                "args": {"direction": direction, "steps": steps},
                "id": "tc-mm-1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="MotorMove handled."),
    ]))


# Motor_StartMove is the underlying Nanonis call MotorMove.execute would make if
# it were ever allowed to run — the coarse-approach gate must prevent it.
_CANNED_MOTOR = {"Motor_StartMove": {"return_value": ("",)}}


class TestTheApprovalDialogIsGone:
    """⑰(2026-08-08):DANGEROUS 技能不再暂停等人批准 —— 照跑、留痕、通知。

    这一节原来有四个类,钉的是 **interrupt/resume 机制**本身:
    ``TestInterruptPauses``(必须暂停)、``TestResumeApprove``(批准后执行)、
    ``TestResumeReject``(拒绝后不执行)、``TestResumeEdit``(编辑后按新值执行)。
    它们在当时都是对的,而且是 GUI 之外唯一一处端到端验证这条链路的地方。

    用户在两天实机之后把整条确认框链路割掉:

        「绝大多数要求用户批准的安全设定都可以做成只提醒不阻碍的。现在我们的实验
          的真正阻碍就是这些没有意义的安全设定,就像官僚体系一样。」

    所以四个「裁决分支」合并成两条断言:**它跑了**,以及**它留下了通知**。
    ``build(..., interrupt_on=...)`` 这个参数仍然接受(六处调用方按名传),但它现在
    只影响启动日志 —— 运行时判据是 ``mast.core.auto_approval.would_have_asked``,
    直接读 ``metadata.safety_level``。

    **SafetyGate 那一半没有合并进来**,见本类下面第三条:参数越界仍然被拦,而且拦
    在执行之前。它是拒绝型防护,⑰ 一个字没动。
    """

    def test_a_gated_skill_runs_without_pausing(self):
        """原 ``test_dangerous_biaspulse_pauses_on_interrupt``,断言反过来。

        三件事一起钉,少一件都会被一个「把工具整个吞掉」的 bug 骗过去:
        不暂停 / 没有 ``__interrupt__`` / **脉冲真的到了仪器上**。"""
        ctx, prov = _shared_ctx(_CANNED)
        agent = _build_agent(_biaspulse_llm(bias_v=5.0), prov)
        cfg = {"configurable": {"thread_id": "hitl-nopause"}}
        result = agent.invoke({"messages": [("user", "pulse the tip")]}, config=cfg)

        assert not agent.get_state(cfg).next, "又开始等人批准了"
        assert not result.get("__interrupt__")
        bias_calls = [a for m, a in ctx.calls if m == "Bias_Pulse"]
        assert bias_calls, "脉冲没有到达仪器 —— 「不再审批」不等于「不再执行」"
        # Bias_Pulse(Wait, width_s, bias_v, z_hold, abs_rel) → arg[2] is bias_v
        assert bias_calls[0][2] == 5.0

    def test_it_leaves_a_notice_behind(self):
        """执行 + 留痕 + **通知**。少了通知,这条改动就只是「悄悄放开」。

        按 subject 过滤而不是读最新一条:``diagnostics`` 是进程级环形缓冲,同一次
        pytest 里别的文件写进去的行会让「读最新」读到不属于本测试的东西。"""
        from mast.core import diagnostics as diag

        ctx, prov = _shared_ctx(_CANNED)
        agent = _build_agent(_biaspulse_llm(), prov)
        cfg = {"configurable": {"thread_id": "hitl-notice"}}
        before = len([r for r in diag.recent(300, kinds=("notice_only",))
                      if r.get("subject") == "BiasPulse"])
        agent.invoke({"messages": [("user", "pulse")]}, config=cfg)

        rows = [r for r in diag.recent(300, kinds=("notice_only",))
                if r.get("subject") == "BiasPulse"]
        assert len(rows) > before, "被门控的技能跑了却没有留下通知"
        # 这里走的是**显式名单**那一支:BiasPulse 在 2026-06-11 的重新分级之后是
        # CONFIRM 而不是 DANGEROUS,本文件一直用 ``interrupt_on`` 覆盖把它强行拉进
        # 门控(好处是不必为了测这条路径去翻生产 metadata)。⑰ 之后这个覆盖仍然
        # 有效 —— 见 ``AutoApprovalNoticeMiddleware(extra_names=...)``,那正是为了
        # 不让 ``interrupt_on`` 退化成一个调了没反应的死参数。
        assert "显式门控名单" in rows[0]["reason"]
        assert "不再等待人工批准" in rows[0]["reason"]

    def test_a_really_dangerous_skill_is_noticed_without_any_override(self):
        """上一条走的是覆盖名单;**生产上靠的是 metadata**,那一支也要钉。

        否则「覆盖名单能用」会掩盖「派生判据坏了」——两条路只测了一条,而生产只走
        另一条。这里直接对着注册表里**真的** DANGEROUS 的那些技能核判据。"""
        from mast.agents.instrument_control.tools import discover_instrument_skills
        from mast.core.auto_approval import would_have_asked
        from mast.core.types import SafetyLevel

        metas = discover_instrument_skills().list_skills()
        dangerous = [m for m in metas
                     if getattr(m, "safety_level", None) == SafetyLevel.DANGEROUS]
        assert dangerous, "注册表里一个 DANGEROUS 技能都没有 —— 这条测试失去了对象"
        for meta in dangerous:
            assert would_have_asked(meta) is not None, (
                f"{meta.name} 是 DANGEROUS,却不会触发通知")
        # 反向面:非 DANGEROUS 的不该被误报,否则判据恒真也会绿。
        for meta in metas:
            if getattr(meta, "safety_level", None) != SafetyLevel.DANGEROUS:
                assert would_have_asked(meta, mode="auto") is None, meta.name

    def test_safety_gate_still_blocks_an_out_of_bounds_value(self):
        """原 ``test_edit_above_global_cap_blocked_by_safety_gate`` 的**保留部分**。

        原来那条走的是「用户把 bias_v 编辑成 50 → SafetyGate 在送去执行的路上
        重新校验并拦下」。编辑这条路随审批框一起没了,但它真正要证明的性质没变:
        **参数越界在执行之前被拦**。所以现在直接让模型发一个 50 V 出来。

        这一条是 ⑰ 的边界线:审批(等人)割掉了,**拒绝型防护(越界即拒)一个字没动**。
        两天实机里真正拦下危险的正是这一类。"""
        ctx, prov = _shared_ctx(_CANNED)
        agent = _build_agent(_biaspulse_llm(bias_v=50.0), prov)
        cfg = {"configurable": {"thread_id": "hitl-oob"}}
        result = agent.invoke({"messages": [("user", "pulse")]}, config=cfg)

        assert not any(m == "Bias_Pulse" for m, _ in ctx.calls), \
            "SafetyGate must block bias_v=50 before execution"
        joined = "\n".join(
            str(getattr(m, "content", "")) for m in result.get("messages", []))
        assert "global_bounds_violation" in joined or "above global safety" in joined

    def test_the_out_of_bounds_refusal_is_not_an_approval_prompt(self):
        """上一条的**反向面**:被拒不等于被挂起。

        拒绝型防护的定义就是「不弹框、不等人」。如果 SafetyGate 哪天改成
        「拦下来问问人」,上一条测试照样绿 —— 而那正是 ⑰ 要消灭的形状。"""
        ctx, prov = _shared_ctx(_CANNED)
        agent = _build_agent(_biaspulse_llm(bias_v=50.0), prov)
        cfg = {"configurable": {"thread_id": "hitl-oob-nopause"}}
        result = agent.invoke({"messages": [("user", "pulse")]}, config=cfg)
        assert not agent.get_state(cfg).next
        assert not result.get("__interrupt__")


class TestDefaultBuildAndCoarseApproachGate:
    """PRODUCTION-path policy after the 2026-06-11 safety re-scoping.

    No builtin skill is DANGEROUS anymore, so the default-derived HITL map is
    empty and the HITL middleware is a no-op. The hardware-danger gate is no
    longer a static-safety_level human弹框 but a PARAMETER condition — an
    open-loop coarse Z step toward the sample (``MotorMove
    direction='z-approach'``) — which SafetyGateMiddleware blocks FAIL-CLOSED on
    the autonomous agent path (see mast.core.safety.is_coarse_sample_approach).
    """

    def test_default_build_does_not_interrupt_on_biaspulse(self):
        # With no interrupt_on override, _derive_hitl_map() returns {} (BiasPulse
        # is now AUTO) → the agent does NOT pause and BiasPulse runs straight
        # through. This is the inverse of the old DANGEROUS-gate behaviour.
        ctx, prov = _shared_ctx(_CANNED)
        agent = build(
            buf=None, context_provider=prov, model=_biaspulse_llm(),
            checkpointer=InMemorySaver(), enable_hitl=True,
            # NOTE: no interrupt_on — derivation from metadata yields an empty map.
        )
        cfg = {"configurable": {"thread_id": "hitl-default"}}
        result = agent.invoke({"messages": [("user", "pulse")]}, config=cfg)
        assert not agent.get_state(cfg).next, (
            "no builtin skill is DANGEROUS post-rescoping → default build must NOT pause"
        )
        assert not result.get("__interrupt__"), "default build must not surface __interrupt__"
        # BiasPulse (AUTO, Nanonis-bounded) runs autonomously.
        assert any(m == "Bias_Pulse" for m, _ in ctx.calls), \
            "AUTO BiasPulse must execute on the default path (no HITL gate)"

    def test_coarse_sample_approach_blocked_fail_closed(self):
        # The ONE physically-dangerous action — an open-loop coarse Z step TOWARD
        # the sample — must be BLOCKED by SafetyGateMiddleware on the autonomous
        # agent path: status=error ToolMessage, content carries
        # 'coarse_sample_approach_blocked', and the skill handler is NEVER
        # reached (no Motor_StartMove safe_call).
        ctx, prov = _shared_ctx(_CANNED_MOTOR)
        agent = build(
            buf=None, context_provider=prov,
            model=_motormove_llm(direction="z-approach", steps=50),
            checkpointer=InMemorySaver(), enable_hitl=True,
        )
        cfg = {"configurable": {"thread_id": "coarse-approach-blocked"}}
        result = agent.invoke({"messages": [("user", "approach the sample")]}, config=cfg)
        # It must NOT pause for human approval (this is a fail-closed BLOCK, not a
        # HITL弹框 — autonomous agents may not coarse-approach at all).
        assert not agent.get_state(cfg).next, \
            "coarse approach is fail-closed blocked, not paused for approval"
        assert not result.get("__interrupt__")
        # Handler NEVER ran: the underlying Nanonis Motor_StartMove was not called.
        assert not any(m == "Motor_StartMove" for m, _ in ctx.calls), \
            "SafetyGate must block the coarse approach before the handler runs"
        # A status=error ToolMessage carrying the gate marker must be present.
        msgs = result.get("messages", [])
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        blocked = [
            m for m in tool_msgs
            if "coarse_sample_approach_blocked" in str(getattr(m, "content", ""))
        ]
        assert blocked, (
            "a coarse_sample_approach_blocked ToolMessage must be injected; "
            f"tool messages: {[str(getattr(m, 'content', '')) for m in tool_msgs]}"
        )
        assert getattr(blocked[0], "status", None) == "error", \
            "the coarse-approach block must be a status=error ToolMessage"

    def test_lateral_motormove_is_gated_but_not_as_a_sample_approach(self):
        """A bare lateral step is blocked — by its OWN gate, not the approach one.

        Reversed on 2026-07-31, deliberately. A lateral move is not dangerous
        because it heads toward the sample (it does not); it is dangerous because
        ``MotorMove`` performs it with the tip hanging over the surface on ~1 µm
        of fine-Z clearance — a check that also passes when the state is unknown
        — with no coarse-Z retract, no chamber-pressure check, no drive-voltage
        readback, no current watch and no record of where the stage has been.

        The two gates stay distinct on purpose: their remedies differ. A blocked
        sample-approach means "a human must run this"; a blocked lateral move
        means "call RelocateCoarseXY instead", which the agent can do itself."""
        ctx, prov = _shared_ctx(_CANNED_MOTOR)
        agent = build(
            buf=None, context_provider=prov,
            model=_motormove_llm(direction="x+", steps=10),
            checkpointer=InMemorySaver(), enable_hitl=True,
        )
        cfg = {"configurable": {"thread_id": "lateral-move-gated"}}
        result = agent.invoke({"messages": [("user", "step x")]}, config=cfg)
        joined = "\n".join(
            str(getattr(m, "content", "")) for m in result.get("messages", []))
        assert "coarse_sample_approach_blocked" not in joined, \
            "a lateral move is not the open-loop sample-approach danger"
        assert "unguarded_lateral_coarse_move_blocked" in joined, \
            "a BARE lateral coarse step must be refused on the autonomous path"
        assert "RelocateCoarseXY" in joined, \
            "the refusal must name the guarded path, which IS autonomous"
        assert not any(m == "Motor_StartMove" for m, _ in ctx.calls), \
            "the stage must not move on a refused request"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
