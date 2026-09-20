"""Unit tests for the migrated hold-honouring + live task-slot projection.

Covers routes.orchestrator._honor_holds (the _wait_while_held port) and
CoreRuntime.agents_snapshot / agents_interrupts / agents_artifacts (the live
overlay hooks bootstrap wires onto ctx).
"""
from __future__ import annotations

import threading
import time

import pytest

from mast.api.routes.orchestrator import _honor_holds
from mast.core.runtime import CoreRuntime


def _frames(gen) -> list[str]:
    return list(gen)


def test_honor_holds_noop_when_not_held() -> None:
    st = {"holds": {}, "task": {"events": []}}
    assert _frames(_honor_holds(st, "instrument_control", None)) == []


def test_honor_holds_noop_when_st_missing() -> None:
    assert _frames(_honor_holds(None, "x", None)) == []
    assert _frames(_honor_holds({}, "x", None)) == []


def test_honor_holds_escapes_on_abort() -> None:
    abort = threading.Event()
    abort.set()
    st = {"holds": {"instrument_control": True}, "task": {"events": []}}
    out = _frames(_honor_holds(st, "instrument_control", abort))
    # one 'held' status frame, then abort breaks the loop before 'resumed'
    assert any('"held": true' in f for f in out)
    assert all('"held": false' not in f for f in out)


def test_honor_holds_pauses_then_resumes() -> None:
    st = {"holds": {"__all__": True}, "task": {"events": []}}
    # release the hold shortly after the generator starts blocking
    def _release() -> None:
        time.sleep(0.4)
        st["holds"]["__all__"] = False
    t = threading.Thread(target=_release, daemon=True)
    t.start()
    out = _frames(_honor_holds(st, "literature", None))
    t.join(timeout=2)
    assert any('"held": true' in f for f in out)
    assert any('"held": false' in f for f in out)
    kinds = [e["kind"] for e in st["task"]["events"]]
    assert "held" in kinds and "resumed" in kinds


# ── CoreRuntime live-overlay hooks (called on a stub holding only the state) ──
class _Stub:
    """Minimal stand-in carrying just the attributes the hooks read."""
    def __init__(self, api_state=None, interrupts=None, buffer=None):
        self._agents_api_state = api_state
        self._orch_interrupts = interrupts
        self._buffer = buffer


def test_agents_snapshot_projects_active_task() -> None:
    st = {
        "holds": {"data_processing": True},
        "task": {
            "active": True, "id": "t1", "description": "扫描规划",
            "active_agent_id": "experiment_design",
            "handoffs": [{"t": 1.0, "kind": "handoff", "text": "SUP → XD"}],
            "threads": {"experiment_design": [{}, {}]},
            "final_text": "", "error": None,
        },
    }
    snap = CoreRuntime.agents_snapshot(_Stub(api_state=st, buffer=object()))
    assert snap["active_agent_id"] == "experiment_design"
    assert snap["holds"] == {"data_processing": True}
    assert snap["active_task"]["active"] is True
    assert snap["active_task"]["id"] == "t1"
    assert snap["threads_index"]["experiment_design"] == 2
    assert len(snap["handoff_events"]) == 1
    caps = snap["capabilities"]
    assert caps["hold"] and caps["interject"] and caps["abort"]
    assert caps["buffer_summarizer_active"] is True


def test_agents_snapshot_degrades_empty() -> None:
    snap = CoreRuntime.agents_snapshot(_Stub(api_state=None))
    assert snap["active_task"] is None
    assert snap["holds"] == {}
    assert snap["active_agent_id"] is None


def test_agents_interrupts_filters_by_agent() -> None:
    store = {
        "lock": threading.RLock(),
        "pending": {
            "a": {"event_id": "a", "agent_id": "instrument_control", "skill": "SetBias"},
            "b": {"event_id": "b", "agent_id": "data_processing", "skill": "Analyze"},
        },
    }
    stub = _Stub(interrupts=store)
    allrows = CoreRuntime.agents_interrupts(stub, "__all__")
    assert len(allrows) == 2
    ic = CoreRuntime.agents_interrupts(stub, "instrument_control")
    assert [r["event_id"] for r in ic] == ["a"]


def test_agents_artifacts_merges_produced_and_edits() -> None:
    st = {
        "task": {"artifacts": {"session_summary": {"path": "x.md"}}},
        "artifact_edits": {"session_summary": {"body": "edited", "t": 2.0}},
    }
    arts = CoreRuntime.agents_artifacts(_Stub(api_state=st))
    assert arts["produced"]["session_summary"]["path"] == "x.md"
    assert arts["edits"]["session_summary"]["body"] == "edited"


# ── 「读不到」不许折成「零条待批」（普查 A1，2026-08-15）──────────────────
#
# `agents_interrupts` 回 `[]` 是一句**正面断言**：「问过活闸门了，零条待批」。
# 它原来在 `except Exception:` 里也回 `[]`，于是路由发出的是
# `count=0, interrupt_gating=True, degraded=False` ——「门禁开着，没人在等」。
#
# 特别值钱的一点：**下游早就写对了**。`routes/agents_topology._agents_interrupts`
# 明确区分「拿不到 ⇒ None」与「拿到一个 list」，`get_agent_interrupts` 再把
# None 变成 `degraded=True`。那条路一直是活的，只是生产方递过去的永远是一个
# 合法的空 list，于是它永远走不到。所以这里的测试有两层：生产方要会说「读不到」，
# 以及那句话要**真的落到**下游那条路上。

class _ExplodingPending:
    """一个 pending 表，`.values()` 会抛 —— 模拟并发改写/坏掉的存储。"""

    def values(self):
        raise RuntimeError("pending 表在迭代时被改了")


def test_interrupts_read_failure_raises_instead_of_answering_zero() -> None:
    stub = _Stub(interrupts={"lock": threading.RLock(),
                             "pending": _ExplodingPending()})
    with pytest.raises(RuntimeError):
        CoreRuntime.agents_interrupts(stub, "__all__")


def test_interrupts_with_no_store_raises_too() -> None:
    """第二道门：`getattr(...) or {}` 会让「没有闸门」和「闸门是空的」一模一样。"""
    with pytest.raises(RuntimeError, match="不是「零条待批」"):
        CoreRuntime.agents_interrupts(_Stub(interrupts=None), "__all__")


def test_a_real_empty_store_still_answers_zero() -> None:
    """反向对照：**该放行的时候要放行。**

    没有这一条，上面两条可以被一个「永远抛」的实现满足 —— 那会把面板从
    「假装太平」换成「永远打不开」，是另一种坏法。`[]` 必须仍然说得出口。
    """
    stub = _Stub(interrupts={"lock": threading.RLock(), "pending": {}})
    assert CoreRuntime.agents_interrupts(stub, "__all__") == []


def test_the_refusal_lands_on_the_routes_degraded_path() -> None:
    """接缝：生产方抛 ⇒ 路由的 `None` ⇒ 面板 `degraded=True`。

    分开测两端都绿、而中间没接上，正是这一族缺陷的成因。这里把真方法
    喂进真的路由帮助函数，证明那条路是连着的。

    ## 这一行就是缺陷本身（修之前它是这样红的）

        assert _agents_interrupts(_Ctx(_Stub(interrupts=None))) is None
        E   AssertionError: assert [] is None

    路由的三态**一直是活的** —— `_agents_interrupts` 从第一天就分「拿不到 ⇒
    None」与「拿到一个 list」，`get_agent_interrupts` 也一直把 None 变成
    `degraded=True`。只是生产方递过去的永远是一个**合法的空 list**，于是那条
    路永远走不到。修的是生产方，而证据在这里：`[] is None` 为假。
    """
    from mast.api.routes.agents_topology import _agents_interrupts

    class _Ctx:
        def __init__(self, stub):
            self.agents_interrupts = (
                lambda agent_id="__all__": CoreRuntime.agents_interrupts(
                    stub, agent_id))

    # 读不到 ⇒ None ⇒ 路由会答 degraded=True
    assert _agents_interrupts(_Ctx(_Stub(interrupts=None))) is None
    # 真的零条 ⇒ 一个空 list（**不是** None）⇒ 路由会答 degraded=False, count=0
    ok = _agents_interrupts(
        _Ctx(_Stub(interrupts={"lock": threading.RLock(), "pending": {}})))
    assert ok == []
