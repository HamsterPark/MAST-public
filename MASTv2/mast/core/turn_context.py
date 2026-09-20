"""当前回合的 provenance —— 「这份文档是哪次对话、哪个 run 产出的」。

设计文档：``docs/v2/design/document_and_library_management.md`` §3.4

为什么需要它
------------

文档保存时想记下 ``conversation_id`` / ``run_id``，但保存工具是裸 ``@tool``
函数（``paper_writing.save_draft`` 等），没有 ctx 注入、也拿不到图状态。

对话引擎其实**已经**把这两个值放进了自己的 thread-local（``chat/engine.py`` 在
``stream_turn`` / ``stream_events`` 开头就设），而且「thread-local 在工具执行期
可见」这件事有既有证据：``active_run_id()`` 已经被 runtime 的仪器仲裁在工具调用
期间成功消费。缺的只是一个**中立的、谁都能读**的落点 —— engine 的
``conversation_id`` 至今没有公共访问器，零消费者。

这个模块就是那个落点：驱动方（对话引擎、orchestrator 群跑循环）在回合开始时
``set``，任何工具在回合内 ``current_turn()`` 就能读到。

三条纪律
--------

1. **best-effort，绝不阻塞。** 读不到就返回空 dict。provenance 缺一个字段是记账
   损失；因为拿不到 run_id 而拒绝保存 LLM 写好的整篇报告是内容损失。孰轻孰重
   已经在 ``sample_gate`` 的 fail-open 论证里定过调：歧义朝放行解。
2. **按执行上下文隔离，不是全局。** 群聊、私聊、后台 run 三条线程同时在跑；一个
   全局变量会让后台 run 的文档记上前台对话的 id。
3. **只读不猜。** 没有「取最近一次」这种回退 —— 猜错的 provenance 比没有更糟，
   它会让人相信一个错的因果链。

为什么落点是 ContextVar 而不是 threading.local（2026-08-11 改，实测）
------------------------------------------------------------------

原来这里是 ``threading.local()``。它对**驱动线程**成立，对**工具线程**不成立 ——
而工具正是这个模块最重要的读者：

* langgraph 的 ``ToolNode`` 每次调用都 ``with get_executor_for_config(config) as ex:
  ex.map(self._run_one, ...)``，即**每个工具都在一根新的
  ``ContextThreadPoolExecutor`` 工作线程上跑**（哪怕只有一个工具调用）；
* 私聊还额外走 ``graph.stream(..., subgraphs=True)``，这会让 pregel 装上
  ``get_waiter``，连 ``PregelRunner.tick`` 里「单任务就地跑」的快路径也被跳过，
  于是**每个节点**都进 ``submit()``；
* ``ContextThreadPoolExecutor.submit`` 做的是 ``copy_context().run(fn, ...)`` ——
  它复制的是 **contextvars**，**threading.local 一个字节都不带过去**。

实测（同一个 venv，langgraph 1.1.9 / langchain-core 1.3.2）：

    生产调用形状 stream_mode='updates', subgraphs=True
       节点线程 != 调用线程
       threading.local 读到的 conversation_id = None
       ContextVar     读到的 conversation_id = 'conv-XYZ'

这不只是记账损失。``ConversationEngine.active_abort_event()`` 的**会话回落用的钥匙
就是这里的 ``conversation_id``** —— 钥匙在工具线程上是 None，回落就永远落空，于是
用户按下的「停止」到不了正在跑的技能（``KNOWN_ISSUES §2.38`` 标着「已修」，实际
从未成立：那次修复把回落键换成了 conversation_id，而 conversation_id 自己也是
thread-local）。

**这跟 ``reassert_turn_each_resume`` docstring 里那句「ContextVar 解决不了」不矛盾，
它们说的是两个方向**：那里说的是**写回**（worker 里 set 的值传不回父上下文，所以
「回合开头设一次」仍然不够，那个包装仍然必需）；这里要的是**读下去**（父上下文的值
传得进子线程），而 ``copy_context()`` 恰好只保证后者。两件事都要，缺一不可。

隔离性没有变松：``ContextVar`` 的可见范围是**当前执行上下文**，比线程更细；
``reassert_turn_each_resume`` 每次恢复执行前重设，所以并发两条流各自看到自己的值
（``tests/v2/unit/documents/test_provenance_concurrency.py`` 钉着这条）。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

# 三个独立的 ContextVar，不是一个 dict：dict 会被子上下文**就地改**，那等于共享，
# 隔离就没了。分开存 + 只写不改，子上下文的 set 只影响它自己。
_conversation_id: ContextVar[str | None] = ContextVar(
    "mast_turn_conversation_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("mast_turn_run_id", default=None)
_agent_id: ContextVar[str | None] = ContextVar("mast_turn_agent_id", default=None)

_VARS = {"conversation_id": _conversation_id, "run_id": _run_id,
         "agent_id": _agent_id}

__all__ = ["set_turn", "clear_turn", "current_turn", "turn_scope",
           "reassert_turn_each_resume"]


def set_turn(*, conversation_id: str | None = None, run_id: str | None = None,
             agent_id: str | None = None) -> None:
    """记下当前回合的标识。只覆盖显式传入的字段。"""
    if conversation_id is not None:
        _conversation_id.set(str(conversation_id or "") or None)
    if run_id is not None:
        _run_id.set(str(run_id or "") or None)
    if agent_id is not None:
        _agent_id.set(str(agent_id or "") or None)


def clear_turn() -> None:
    """回合结束时清掉。**必须在 finally 里调** —— 线程/上下文会被复用，残留的 id
    会让下一个回合的产物记上上一个回合的来历。"""
    for var in _VARS.values():
        var.set(None)


def current_turn() -> dict[str, str | None]:
    """当前回合的 ``{conversation_id, run_id, agent_id}``。读不到就是 None。"""
    return {name: (var.get() or None) for name, var in _VARS.items()}


def reassert_turn_each_resume(gen, holder: dict):
    """包一个**同步 generator**，让它每次被恢复执行前重新 set 一遍回合标识。

    为什么「开头设一次」不够（2026-07-29 架构审查发现，实测复现）
    ----------------------------------------------------------

    对话流和群跑流都是同步 generator，交给 ``StreamingResponse`` 后由 Starlette 的
    ``iterate_in_threadpool`` 消费 —— **每次 ``next()`` 单独走一趟
    ``anyio.to_thread.run_sync``，不保证落在同一个 worker 线程上**。于是回合开头
    在线程 A 上设好的 thread-local，下一个 super-step 可能在线程 B 上执行，读不到；
    更糟的是两条流并发时，B 上残留的是**另一条对话**的值：

        两条并发流实测：
          ('tool','B',77252,'run-C')   ← B 的工具读到了 C 的 run_id
          ('clear','B',77252)          ← B 的 finally 清掉了 C 的 provenance

    结果是把**一条错的因果链当事实存进** ``doc.json`` / ``versions.jsonl`` / DB ——
    正撞本模块第 3 条纪律「猜错的 provenance 比没有更糟」。

    为什么这个包装有效：generator 恢复执行的代码跑在**调用 ``next()`` 的那个线程**上。
    这里在 ``next(gen)`` **之前**设值，紧随其后的 super-step 因此必然看到正确的值。

    ⚠️ 「改成 ``ContextVar`` 就不用这个包装了」是**错的**，而且这条曾经被写反过：
    每趟 ``to_thread`` 拿到的是 context 的**拷贝**，generator 内部（在 worker 里）
    ``set`` 的值**写不回**父上下文，所以「回合开头设一次」无论用哪种存储都不够 ——
    这个包装必需。落点换成 ContextVar 修的是**另一个方向**：父上下文的值能不能
    **读下去**传进 ``ToolNode`` 起的执行器子线程（``copy_context()`` 保证能）。
    两件事都要，见模块 docstring。

    ``holder`` 是个可变 dict，由内层实现在拿到 id 后填（``conversation_id`` /
    ``run_id``）；填之前包装什么都不设，所以早退路径不会留下脏值。
    """
    try:
        while True:
            if holder:
                set_turn(conversation_id=holder.get("conversation_id"),
                         run_id=holder.get("run_id"))
            try:
                item = next(gen)
            except StopIteration:
                return
            yield item
    finally:
        clear_turn()
        close = getattr(gen, "close", None)
        if callable(close):
            close()


@contextmanager
def turn_scope(*, conversation_id: str | None = None, run_id: str | None = None,
               agent_id: str | None = None):
    """``with turn_scope(...)``：进入时设、离开时**恢复**（不是清空）。

    恢复而非清空是因为回合可以嵌套（群聊的一次 super-step 里派生子 agent 调用），
    清空会让外层回合剩下的工具调用突然失去 provenance。
    """
    prev = current_turn()
    set_turn(conversation_id=conversation_id, run_id=run_id, agent_id=agent_id)
    try:
        yield
    finally:
        # 直接 set 回旧值，**不用 ContextVar.reset(token)**：token 只能在**同一个
        # 上下文**里 reset，而这个 with 完全可能跨 ``to_thread``/执行器边界地进出
        # （群聊 super-step 里派生子 agent 就是这个形状），那时 reset 会抛
        # ValueError 并把 finally 炸掉。写回旧值语义一样、且到哪儿都成立。
        for name, var in _VARS.items():
            var.set(prev[name])
