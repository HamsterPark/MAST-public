"""2026-07-28 dead-path sweep — 「写了但那条路径从来没生效过」.

One confirmed finding, pinned here.

## buffer_hitl: the block message ignored SAFE mode

``_suggest_action(kind, safe_mode)`` exists precisely because SAFE mode changes
what MAST is willing to do: in SAFE the SafetyGate hard-blocks tip conditioning,
so recommending ``tip_prep`` is advice against the system's own behaviour. That
is — 「明明是safe模式，还是要修针尖」.

The fix reached ONE of the two call sites. ``_publish_meta`` passed
``self._safe_mode()``; ``_gate_block`` — the message the AGENT actually receives
when a critical vision event blocks its next tool call — called
``_suggest_action(ev.kind)`` and took the ``safe_mode=False`` default.

So for the SAME unresolved event, in SAFE mode:
  * the interrupt shown to the operator said  manual_tip_check
  * the block message handed to the agent said 建议动作：['tip_prep']

The agent follows the message it is given, attempts the conditioning tool, and
SafetyGate blocks it — burning super-steps on an action the mode forbids, and
reproducing #43 through a second door. Nothing failed loudly; the parameter was
simply never passed.

Test coverage was part of the problem: the existing buffer_hitl tests build the
middleware with no ``get_mode`` at all, so ``_safe_mode()`` is always False and
neither call site was ever exercised in SAFE mode. The one assertion on
``suggested_action`` (tests/v2/agents/_shared/test_buffer_hitl.py) checks the
interrupt-payload path in NORMAL mode — the path that was already correct.

## ⑷（2026-08-08）：两个门变成一个门，这条要求没有过期

打断链整条割掉之后，``interrupt``与 ``_gate_block`` 都不存在了，所以「两个视图各说各话」
在结构上不可能再发生。但 ``_suggest_action`` 没死：它现在给**通知**用，而一条自相
矛盾的通知照样会指挥模型去做一件系统自己不肯做的事。下面的测试因此改钉通知语，
并多一条反向守卫钉住「第二个门是被删掉了，不是被修好了」。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_dead_path_sweep_20260728.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
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

import asyncio  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from mast.agents._shared.buffer_hitl import (  # noqa: E402
    _suggest_action,
    make_buffer_hitl_middleware,
)
from mast.agents._shared.skill_adapter import wrap_skill  # noqa: E402
from mast.buffer.schemas import Severity, VisionEvent, VisionEventType  # noqa: E402
from mast.buffer.service import BufferService  # noqa: E402
from mast.skills.builtins.bias import SetBias  # noqa: E402


# ── helpers (mirror tests/v2/unit/test_forensics_20260727_safety_gates.py) ────


class _Capture:
    """interrupt() that returns instead of raising — the field shape."""

    def __init__(self) -> None:
        self.payloads: list[Any] = []

    def __call__(self, payload: Any) -> Any:
        self.payloads.append(payload)
        return None


@pytest_asyncio.fixture
async def buffer(tmp_path):
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        yield buf
    finally:
        await buf.stop()


def _drop(seqno: int = 1) -> VisionEvent:
    """A CRITICAL tip_quality_drop that OPENS the gate — i.e. a physical one.

    These tests are about the WORDING the agent gets when the gate refuses it
    (SAFE mode must not recommend an action SAFE itself blocks).
    They need an open gate; which events open one is a different question,
    answered in ``buffer_hitl.escalates_to_operator``.

    The payload used to be the vision edge-detector shape. Since 2026-08-05 that
    shape is recorded but never escalates, so it would leave the gate open and
    these tests would be asserting on a message nobody was ever shown."""
    return VisionEvent(
        seqno=seqno,
        kind=VisionEventType.TIP_QUALITY_DROP,
        severity=Severity.CRITICAL,
        payload={"signal": "current_saturation", "source": "current_monitor",
                 "scan_id": "s", "summary_zh": "前放到轨"},
        cause_ref=f"current_monitor#{seqno}",
    )


def _make_request(tool, args: dict, call_id: str = "t-1") -> Any:
    req = MagicMock()
    req.tool = tool
    req.tool_call = {"name": tool.name, "args": args, "id": call_id,
                     "type": "tool_call"}
    req.state = {}
    req.runtime = MagicMock()
    return req


def _tool(skill_cls):
    return wrap_skill(skill_cls, lambda: MagicMock(safe_call=lambda *a, **k: None))


async def _tick() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def _notice_reason(buffer, *, mode: str | None) -> str:
    """发一条 critical tip drop,把这条事件留在诊断台账里的**通知语**读回来。

    ⑰(2026-08-08)之前这个 helper 读的是「agent 被拒绝时收到的拦截语」。拦截语
    随打断链一起没了 —— 这个缺陷的两个「门」如今只剩一个,而剩下的这个门必须继承
    #43 的要求。

    刻意按 ``cause_ref`` 过滤:``diagnostics`` 是**进程级**环形缓冲,同一次 pytest
    里别的文件写进去的 ``notice_only`` 行会让「读最新一条」读到不属于本测试的东西
    (本轮改造中真的踩到过一次,断言绿了而这条测试自己一行都没写)。"""
    from mast.core import diagnostics as diag

    kw = {} if mode is None else {"get_mode": lambda: mode}
    mw = make_buffer_hitl_middleware(buffer=buffer, **kw)
    try:
        ev = _drop()
        buffer.emit_event(ev)
        await _tick()
        mw.before_model({}, MagicMock())
        rows = [r for r in diag.recent(200, kinds=("notice_only",))
                if r.get("cause_ref") == ev.cause_ref]
        assert rows, "关键事件没有留下任何通知 —— 记录那一半掉了"
        return str(rows[0]["reason"])
    finally:
        mw.close()


# ── the pure mapping (what the parameter is for) ─────────────────────────────


def test_safe_mode_changes_the_recommended_action() -> None:
    """Baseline: the parameter is not decorative — it flips exactly one answer."""
    assert _suggest_action(VisionEventType.TIP_QUALITY_DROP, False) == "tip_prep"
    assert _suggest_action(VisionEventType.TIP_QUALITY_DROP, True) == "manual_tip_check"
    # stops/retracts are allowed in every mode and must NOT change
    assert _suggest_action(VisionEventType.E_STOP, True) == "halt"
    assert _suggest_action(VisionEventType.EMERGENCY_RETRACT_NEEDED, True) == "retract"


# ── the path that was dead(现在是**通知**这条路)────────────────────────────


@pytest.mark.asyncio
async def test_the_notice_honours_safe_mode(buffer) -> None:
    """THE REGRESSION,搬到新家。

    原来的缺陷是:同一条事件,给用户的 interrupt 说 ``manual_tip_check``,而给
    agent 的拦截语说 ``tip_prep`` —— 一个参数没传,两个视图各说各话。⑰ 之后只剩
    **通知**这一个视图,而它必须继续按模式给答案:SAFE 下 SafetyGate 硬拦修针,
    再建议修针就是在推荐一件系统自己不肯做的事。"""
    reason = await _notice_reason(buffer, mode="safe")
    assert "manual_tip_check" in reason, "SAFE 模式的通知丢了按模式给的建议"
    assert "tip_prep" not in reason, (
        "SAFE 模式的通知又开始建议修针 —— 正是这个模式硬拦的动作")


@pytest.mark.asyncio
async def test_the_notice_still_recommends_tip_prep_outside_safe_mode(buffer) -> None:
    """修复不能让每个模式都畏手畏脚:SAFE 之外,修针本来就是该做的下一步。"""
    reason = await _notice_reason(buffer, mode="normal")
    assert "tip_prep" in reason
    assert "manual_tip_check" not in reason


@pytest.mark.asyncio
async def test_unwired_mode_reader_keeps_the_old_behaviour(buffer) -> None:
    """没接 get_mode(standalone / 测试)→ ``_safe_mode()`` 为 False,通知不变。
    修复不能依赖一个可能根本不存在的读取器。"""
    reason = await _notice_reason(buffer, mode=None)
    assert "tip_prep" in reason


@pytest.mark.asyncio
async def test_the_second_door_is_gone_not_just_fixed(buffer) -> None:
    """反向守卫:这个缺陷的形状是「同一件事有两个视图,其中一个忘了传参」。

    ⑰ 把第二个视图(拦截语)整个删掉 —— 一个不存在的视图不会再和别人说法不一。
    直接钉这件事,否则上面三条只证明「剩下那个视图对」,而缺陷的根因是**有两个**。"""
    from mast.agents._shared.buffer_hitl import BufferHITLMiddleware

    assert not hasattr(BufferHITLMiddleware, "_gate_block")
    assert not hasattr(BufferHITLMiddleware, "_build_interrupt_payload")

    mw = make_buffer_hitl_middleware(buffer=buffer, get_mode=lambda: "safe")
    try:
        buffer.emit_event(_drop())
        await _tick()
        mw.before_model({}, MagicMock())
        # …而且 agent 的下一个推进类工具照跑,拿不到任何「建议动作」的字样。
        req = _make_request(_tool(SetBias), {"bias_v": 1.0})
        handler = MagicMock(return_value="ok")
        assert mw.wrap_tool_call(req, handler) == "ok"
        handler.assert_called_once()
    finally:
        mw.close()
