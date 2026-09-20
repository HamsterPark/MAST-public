"""用户的「停止」必须真的到得了正在跑的工具 —— 端到端，跨线程，不碰仪器。

## 为什么这条**必须**是端到端的

这个缺陷的全部内容就是「跨线程」。单元测试证明不了它，而且仓里那条单元测试
(``tests/v2/unit/chat/test_abort_reaches_the_skill.py``)恰好示范了怎么证伪失败：
它在新线程里**自己先调了** ``turn_context.set_turn(conversation_id=...)``，然后断言
``active_abort_event()`` 取得到事件。可生产里那个线程是 langgraph 的执行器
起的，**没有任何人在那里调 set_turn** —— 测试自己制造了被测条件，于是它绿着，
而现场三次停不下来。

## 真正的断点在哪(2026-08-11 实测，探针见提交信息)

三跳，坏的是最后一跳：

* 跳一 SSE：``chat_stream`` 用 ``api.sse.with_heartbeat`` 包了内层 generator，而
  ``with_heartbeat`` 是在**一根固定的 daemon 线程**上 ``for item in inner`` 消费它的。
  所以 ``_stream_turn_impl`` 的每次恢复执行都落在同一根线程上 —— engine 的
  docstring 说的「``iterate_in_threadpool`` 每次 next() 可能换线程」在这条路上
  已经被心跳泵挡住了，**不是断点**。
* 跳二 pregel：私聊走 ``graph.stream(..., subgraphs=True)``，``subgraphs=True`` 让
  pregel 装上 ``get_waiter``，于是 ``PregelRunner.tick`` 那条「单任务就地跑」的快路径
  被跳过，每个节点都进 ``submit()``。
* 跳三 ``ToolNode``：**技能就住在这后面**。
  ``langgraph/prebuilt/tool_node.py`` 的 ``_func`` 里是
  ``with get_executor_for_config(config) as executor: executor.map(self._run_one, ...)``
  —— **每次工具节点调用都新起一个 ``ContextThreadPoolExecutor``**，哪怕只有一个工具调用。

  ``ContextThreadPoolExecutor.submit`` 做的是 ``copy_context().run(...)``：
  它复制的是 **contextvars**，**不复制 threading.local**。

  ⇒ 工具线程上 ``engine._tl.abort`` 没有、``turn_context.current_turn()`` 全是 None
  ⇒ ``active_abort_event()`` 返回 None（thread-local 快路径**和**会话回落**双双落空**，
     因为会话回落的钥匙 ``conversation_id`` 自己也存在 threading.local 里）
  ⇒ ``skill_adapter`` 在工具体内 ``context_provider()`` 建出的 ``ExecutionContext``，
     abort 并集里只剩 E-STOP
  ⇒ 用户按下的「停止」到不了正在跑的技能。等待循环确实每个 poll 都在查
  ``check_abort()``，它查的那个集合里没有用户那一个。

## 它从来没好过（不是回归）

``KNOWN_ISSUES §2.38`` 标着「已修 ``543b485`` + ``0462014``」。查了历史：
``active_abort_event()`` 2026-06-16 出生时就是一句裸的 thread-local 读；``543b485``
(2026-08-06) 是它**唯一**一次被改，加的正是那个会话回落 —— 而回落的钥匙也是
thread-local。所以那次修复**从未成立**，不存在「修好过又回归」。这条测试因此是
永久的防回归钉子，不是一次性的验尸。

## 这个文件钉什么

1. **停得下来**：一个真的会跑很久的假动作(纯 sleep 轮询，不碰仪器)，中途 POST
   abort，动作必须在 ``_STOP_DEADLINE_S`` 秒内停下；
2. **回包不许撒谎**：停下来了才 ``ok=true``，并说出停了哪一轮；
3. **没有正在跑的轮次时不许报 ok** —— 一个撒谎的停止按钮比一个明显坏掉的停止
   按钮更危险，用户会以为已经停了然后走开。

变异验证(见提交信息)：把 ``core/turn_context`` 的 ContextVar 改回
``threading.local()``，① 必须红。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mast.api.context import AppContext  # noqa: E402
from mast.api.routes import agents as agents_routes  # noqa: E402
from mast.api.routes import chat_stream  # noqa: E402
from mast.chat.engine import ConversationEngine  # noqa: E402
from mast.core.execution_context import ExecutionContext  # noqa: E402

# 假动作最长跑多久(它自己会到点收工，所以测试挂了也不会吊死)。要比
# _STOP_DEADLINE_S 大得多，否则「跑完了」和「被停下了」看起来一样 —— 那就成了
# 一条自己制造成功条件的测试。
_FAKE_ACTION_CAP_S = 25.0
# 收到停止后允许的最大反应时间。轮询间隔 20 ms，留足 Windows 调度余量。
_STOP_DEADLINE_S = 5.0

_AGENT = "instrument_control"
_CONV = "conv-stop-e2e"


class _Store:
    """够 ConversationEngine 用的最小会话库(不碰磁盘)。"""

    def __init__(self) -> None:
        self._rows = {
            _CONV: {"conversation_id": _CONV, "agent_id": _AGENT,
                    "thread_id": "thread-stop-e2e", "title": "新对话",
                    "last_message_preview": ""},
        }

    def get(self, cid):
        return self._rows.get(cid)

    def touch(self, cid, **_kw):
        return None


class _Probe:
    """假动作在**哪根线程**上跑、跑了多久、是被停的还是跑完的。"""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.finished = threading.Event()
        self.thread_id: int | None = None
        self.aborted: bool | None = None
        self.elapsed: float = -1.0
        self.saw_session_event: bool | None = None


def _build(probe: _Probe):
    """一个真的用 langgraph + **真 ToolNode** 跑、真的走 ``ExecutionContext.check_abort()``
    的私聊。

    技能必须住在 ``ToolNode`` 后面，因为断点就在那儿：生产里 ``skill_adapter`` 是在
    **工具体内**调 ``context_provider()`` 的，而工具体跑在 ``ToolNode`` 自己起的
    ``ContextThreadPoolExecutor`` 上。把假动作直接写成一个普通图节点也能复现，但那
    是靠 ``subgraphs=True`` 那一跳 —— 用 ToolNode 才是**生产的那个排布**。
    """
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    holder: dict = {}

    def _context_provider():
        """**逐字照抄** runtime.py 里私聊那个 ``_ctx()`` 的取用方式：会话 Stop 与
        E-STOP 取并集。这里被测的就是第一项取不取得到。"""
        eng = holder["engine"]
        session_abort = eng.active_abort_event()
        probe.saw_session_event = session_abort is not None
        aborts = [e for e in (session_abort, holder["orch_abort"]) if e is not None]
        return ExecutionContext(pool=None, state=None, registry=None,
                                abort_event=aborts, run_id="run-stop-e2e",
                                owner="测试/私聊")

    @tool
    def long_running_action() -> str:
        """会跑很久的假动作。**不碰仪器** —— 只是一个查 abort 的 sleep 轮询，形状与
        WaitScanComplete / 进针等待相那类等待循环一致。"""
        ctx = _context_provider()          # 生产里 skill_adapter 也是在这里建 ctx
        probe.thread_id = threading.get_ident()
        t0 = time.monotonic()
        probe.started.set()
        try:
            while time.monotonic() - t0 < _FAKE_ACTION_CAP_S:
                if ctx.check_abort():
                    probe.aborted = True
                    return "动作已按用户要求停止"
                time.sleep(0.02)
            probe.aborted = False
            return "动作跑完了整整一轮(没被停下)"
        finally:
            probe.elapsed = time.monotonic() - t0
            probe.finished.set()

    def _fake_model(state):
        """第一次发一个工具调用，拿到 ToolMessage 之后收尾。不联网、不调 LLM。"""
        if any(isinstance(m, ToolMessage) for m in state["messages"]):
            return {"messages": [AIMessage(content="好了。")]}
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "long_running_action", "args": {},
             "id": "call-stop-e2e", "type": "tool_call"}])]}

    def _graph_factory(_agent_id):
        g = StateGraph(MessagesState)
        g.add_node("model", _fake_model)
        g.add_node("tools", ToolNode([long_running_action]))
        g.add_edge(START, "model")
        g.add_conditional_edges(
            "model",
            lambda s: "tools" if getattr(s["messages"][-1], "tool_calls", None) else END,
            {"tools": "tools", END: END})
        g.add_edge("tools", "model")
        return g.compile(checkpointer=holder["checkpointer"])

    holder["checkpointer"] = MemorySaver()
    holder["orch_abort"] = threading.Event()
    holder["engine"] = ConversationEngine(
        graph_factory=_graph_factory,
        checkpointer=holder["checkpointer"],
        store=_Store(),
        recursion_limit=12,
    )

    ctx = AppContext()
    ctx.conversation_engine = holder["engine"]
    ctx.chat_abort = chat_stream.chat_abort_hook
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(chat_stream.router, prefix="/api")
    app.include_router(agents_routes.router, prefix="/api")
    return TestClient(app), holder


@pytest.fixture()
def _clean_registry():
    """每条用例前后都清干净进程级登记 —— 上一条留下的活跃轮次会让
    「没有正在跑的轮次」那条断言凭空变绿。"""
    chat_stream._reset_turn_registry_for_tests()
    yield
    chat_stream._reset_turn_registry_for_tests()


def test_operator_stop_reaches_a_long_running_tool(_clean_registry):
    """① 根因那一条：动作在工具线程上跑，用户的停止到得了它。"""
    probe = _Probe()
    client, _holder = _build(probe)

    frames: list[dict] = []
    stream_done = threading.Event()

    def _drive():
        try:
            with client.stream("POST", f"/api/agents/{_AGENT}/chat",
                               json={"conversation_id": _CONV, "user_text": "跑一个很久的动作"}) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        frames.append(json.loads(line[len("data:"):].strip()))
        finally:
            stream_done.set()

    t = threading.Thread(target=_drive, daemon=True)
    t.start()
    assert probe.started.wait(20), "假动作没跑起来 —— 这条测试什么都没测到"

    # 证据有效性：动作确实在**另一根线程**上。同线程的话这条用例证明不了跨线程。
    assert probe.thread_id is not None
    assert probe.thread_id != threading.get_ident()

    time.sleep(0.3)          # 让它确实进入等待循环，而不是刚起步就被停
    resp = client.post(f"/api/agents/{_AGENT}/chat/abort")
    body = resp.json()

    assert probe.finished.wait(_STOP_DEADLINE_S), (
        f"用户按了停止，动作没停 —— 它已经跑了 {probe.elapsed:.1f}s，"
        f"上限 {_FAKE_ACTION_CAP_S}s。abort 回包={body!r}")
    assert probe.aborted is True, (
        f"动作是**自己跑完的**，不是被停的(elapsed={probe.elapsed:.1f}s)。"
        f"abort 回包={body!r}")
    assert probe.elapsed < _STOP_DEADLINE_S
    assert probe.saw_session_event is True, (
        "工具线程上根本没拿到本回合的 Stop 事件 —— abort 并集里只剩 E-STOP")

    # ② 回包必须如实说它停了这一轮。
    assert body["ok"] is True, body
    assert body["degraded"] is False, body
    assert body["signalled"] == 1, body
    assert _CONV in body["conversation_ids"], body

    stream_done.wait(15)
    t.join(timeout=5)


def test_abort_with_no_live_turn_must_not_claim_ok(_clean_registry):
    """② 诚实性：没有正在跑的轮次时**不许**报 ok。

    这是原来那个 bug 最危险的一半：``ok:true`` 的含义曾经是「这个 agent_id 名下
    有一个 Event 对象」，也就是**进程发生过第一次对话之后永远为真**。用户点了
    停止、看到成功、然后走开 —— 而机器照跑。
    """
    probe = _Probe()
    client, _holder = _build(probe)

    # 从没跑过任何一轮。
    first = client.post(f"/api/agents/{_AGENT}/chat/abort").json()
    assert first["ok"] is False, first
    assert first["signalled"] == 0, first
    assert first["reason"], "拒绝要说人话，不能只给一个 false"

    # 跑完整整一轮之后再问一次 —— 这正是旧实现开始永远撒谎的时刻。
    with client.stream("POST", f"/api/agents/{_AGENT}/chat",
                       json={"conversation_id": _CONV, "user_text": "起一轮"}) as r:
        assert probe.started.wait(20)
        client.post(f"/api/agents/{_AGENT}/chat/abort")      # 停掉，别等满 25 秒
        for _line in r.iter_lines():
            pass
    assert probe.finished.wait(_STOP_DEADLINE_S)

    after = client.post(f"/api/agents/{_AGENT}/chat/abort").json()
    assert after["ok"] is False, (
        "回合已经结束，abort 却还报 ok —— 这就是那个撒谎的停止按钮")
    assert after["signalled"] == 0, after
    assert after["reason"], after


def test_stopping_one_conversation_does_not_stop_another(_clean_registry):
    """隔离的单位是**会话**：停一个聊天不许连带停掉并发的另一个。

    旧实现按 agent_id 存一个 Event，而私聊的会话都挂在同一个 agent 下 ⇒ 停一个
    就停了全部；更糟的是新回合开头那句 ``ev.clear()`` 会把另一条正在跑的回合刚
    收到的停止信号**清掉**。
    """
    probe_a, probe_b = _Probe(), _Probe()
    client_a, _ = _build(probe_a)
    client_b, holder_b = _build(probe_b)
    # 两条不同会话(不同 conversation_id)，同一个 agent。
    holder_b["engine"]._store._rows["conv-other"] = {
        "conversation_id": "conv-other", "agent_id": _AGENT,
        "thread_id": "thread-other", "title": "新对话", "last_message_preview": ""}

    threads = []
    for cli, conv in ((client_a, _CONV), (client_b, "conv-other")):
        def _drive(cli=cli, conv=conv):
            with cli.stream("POST", f"/api/agents/{_AGENT}/chat",
                            json={"conversation_id": conv, "user_text": "跑"}) as r:
                for _line in r.iter_lines():
                    pass
        th = threading.Thread(target=_drive, daemon=True)
        th.start()
        threads.append(th)
    assert probe_a.started.wait(20) and probe_b.started.wait(20)
    time.sleep(0.3)

    stopped = client_a.post(f"/api/agents/{_AGENT}/chat/abort",
                            json={"conversation_id": _CONV}).json()
    assert stopped["ok"] is True and stopped["conversation_ids"] == [_CONV], stopped

    assert probe_a.finished.wait(_STOP_DEADLINE_S), "被点名的那一轮没停"
    assert probe_a.aborted is True
    assert not probe_b.finished.is_set(), "另一条会话被连带停了"

    # 收尾：把 B 也停掉，别让它跑满 25 秒。
    client_b.post(f"/api/agents/{_AGENT}/chat/abort", json={"conversation_id": "conv-other"})
    probe_b.finished.wait(_STOP_DEADLINE_S)
    for th in threads:
        th.join(timeout=10)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
