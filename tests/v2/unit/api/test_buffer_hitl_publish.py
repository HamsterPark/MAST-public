"""orchestrator run-task surfaces a buffer_hitl interrupt (not "unparseable").

审查: BufferHITLMiddleware raised interrupt({"kind":"buffer_hitl",
"events":[...]}) on a CRITICAL hardware event (tip crash / e-stop). _publish_interrupt
only knew workflow_human + DANGEROUS(action_requests), so the buffer_hitl chunk fell
through to an empty result and the run-task loop logged "unparseable __interrupt__"
and silently aborted — the operator never got the prompt. This locks the new branch.

## ⑰（2026-08-08）：这条分支**没有生产者了**，而它留着

这是本仓那条历史 bug 「buffer_hitl 中断被丢」的测试。语义变了，而且是彻底反过来：
当年的毛病是「**该弹的没弹**」（事件到了 orchestrator 却被当成无法解析而丢掉），
现在的策略是「**都不弹**」—— 用户把整条确认框链路割掉了（见
``MASTv2/mast/agents/_shared/buffer_hitl.py`` 顶部，依据是两天实机零真阳性）。

所以：``BufferHITLMiddleware`` 不再产生这个形状，本文件里的 chunk 是**手写**的。
测试没有删，因为它守的是 ``_publish_interrupt`` 的**分派表**，而分派表还在服务
两个活着的生产者（``ask_user`` 与工作流 ``human`` 节点）；下面第二条
（``test_unknown_interrupt_still_empty``）钉的正是「不认识的形状不能被 buffer_hitl
分支顺手吞掉」——那条要求与生产者在不在无关。

把这两条删掉的代价是：分派表的兼容分支变成没人守的代码，而它是万一要把 buffer_hitl
接回来时的落点。**一条被删掉的测试和一条从没写过的测试长得一模一样。**

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/api/test_buffer_hitl_publish.py -x -v
"""
from __future__ import annotations

import threading

from mast.api.routes.orchestrator import _publish_interrupt


class _App:
    pass


def _fresh_app() -> _App:
    app = _App()
    app._orch_interrupts = {"pending": {}, "events": {}, "resolved": {}, "lock": threading.Lock()}
    return app


def test_buffer_hitl_interrupt_is_published():
    """⑰ 之后这个 chunk 是**手写**的 —— 树里已经没有东西会产生它（见模块 docstring）。

    它钉的是 ``_publish_interrupt`` 的兼容分支还在，以及万一有人把 buffer_hitl 接
    回来，落点是什么样。**不要**把这条读成「buffer_hitl 还会弹框」。
    """
    app = _fresh_app()
    # A plain dict works: _publish_interrupt does getattr(intr, "value", intr), and
    # a dict has no .value, so it is treated as the interrupt value directly.
    chunk = (
        {
            "kind": "buffer_hitl",
            "events": [
                {"event_id": "e1", "kind": "tip_quality_drop",
                 "severity": "critical", "seqno": 28, "suggested_action": "tip_prep"}
            ],
        },
    )
    published = _publish_interrupt(app, "instrument_control", chunk, "t1")

    assert len(published) == 1, "buffer_hitl must NOT fall through to empty (was 'unparseable')"
    p = published[0]
    assert p["kind"] == "buffer_hitl"
    assert "tip_quality_drop" in p["skill"]
    assert "tip_prep" in p["rationale"]
    assert p["allowed_decisions"] == ["approve", "reject"]
    # registered in the live store so agents_control.resolve can drain + resume it
    assert p["event_id"] in app._orch_interrupts["pending"]
    assert p["event_id"] in app._orch_interrupts["events"]


def test_unknown_interrupt_still_empty():
    # a shape we genuinely can't map must still yield nothing (caller degrades) —
    # the buffer_hitl branch must not accidentally swallow arbitrary dicts.
    app = _fresh_app()
    published = _publish_interrupt(app, "instrument_control", ({"kind": "mystery"},), "t2")
    assert published == []


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
