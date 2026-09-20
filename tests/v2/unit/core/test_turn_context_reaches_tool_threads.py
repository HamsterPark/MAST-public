"""回合标识必须**读得下去** —— 传得进 langgraph 起工具的那种执行器。

这条钉的是 ``core/turn_context`` 的落点选择本身，跟对话、跟图都无关。

## 为什么这件事值得单独钉一条

``turn_context`` 原来存在 ``threading.local()`` 里。它对驱动线程成立，对**工具线程**
不成立 —— 而工具是这个模块最重要的读者：``ToolNode`` 每次调用都新起一个
``ContextThreadPoolExecutor``，而那个执行器 ``copy_context().run(...)``，**复制
contextvars，不复制 threading.local**。

后果不止是记账损失：``ConversationEngine.active_abort_event()`` 的会话回落用的钥匙
就是这里的 ``conversation_id``，钥匙在工具线程上是 None ⇒ 回落永远落空 ⇒ 用户按下
的「停止」到不了正在跑的技能（``KNOWN_ISSUES §2.38`` 标着「已修」，实则从未成立）。

## 为什么这里直接用 ``ContextThreadPoolExecutor``

因为**被测的正是「能不能穿过它」**。用 ``threading.Thread`` 写这条测试会永远是绿的
（裸线程当然读不到，两种存储都读不到），那就成了一条回答别的问题的证据。这里导入
langgraph 实际使用的那一个类，是为了让「库换了实现」这件事也能把这条测试打红。
"""

from __future__ import annotations

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

import threading  # noqa: E402

import pytest  # noqa: E402
from langchain_core.runnables.config import ContextThreadPoolExecutor  # noqa: E402

from mast.core import turn_context  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    turn_context.clear_turn()
    yield
    turn_context.clear_turn()


def _read_in(executor) -> dict:
    return executor.submit(turn_context.current_turn).result(timeout=10)


def test_the_turn_context_reaches_a_context_executor_child():
    """**根因那一条。** 工具跑在这种执行器上，它必须读得到本回合的标识。"""
    turn_context.set_turn(conversation_id="conv-1", run_id="run-1", agent_id="ic")
    with ContextThreadPoolExecutor(max_workers=2) as ex:
        seen = _read_in(ex)
    assert seen == {"conversation_id": "conv-1", "run_id": "run-1", "agent_id": "ic"}


def test_the_child_really_is_another_thread():
    """证据有效性：上一条如果碰巧同线程执行，它证明不了任何事。"""
    with ContextThreadPoolExecutor(max_workers=1) as ex:
        child = ex.submit(threading.get_ident).result(timeout=10)
    assert child != threading.get_ident()


def test_a_plain_thread_still_sees_nothing():
    """隔离没被换成「全局」。

    ``ContextVar`` 的可见范围是**执行上下文**：从当前上下文派生出去的子任务看得到，
    一根自己起的、跟本回合无关的裸线程看不到。这正是要的语义 —— 后台 run 的文档
    不该记上前台对话的 id。
    """
    turn_context.set_turn(conversation_id="conv-1")
    box: dict = {}
    t = threading.Thread(target=lambda: box.update(turn_context.current_turn()))
    t.start()
    t.join()
    assert box["conversation_id"] is None


def test_a_child_cannot_write_back_into_the_parent():
    """写回**仍然**传不上来 —— 所以 ``reassert_turn_each_resume`` 那个包装依旧必需。

    这条把「换成 ContextVar 就不用每次恢复前重设了」这个被否掉的推论钉死。它是
    真的：``copy_context()`` 给子任务的是**拷贝**，子任务的 set 不回到父上下文。
    """
    turn_context.set_turn(conversation_id="parent")
    with ContextThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(turn_context.set_turn, conversation_id="child").result(timeout=10)
    assert turn_context.current_turn()["conversation_id"] == "parent"


def test_two_sibling_contexts_do_not_leak_into_each_other():
    """并发两条流各自看到自己的值（哪怕复用同一根工作线程）。"""
    import contextvars

    def _run_as(cid):
        ctx = contextvars.copy_context()

        def _fn():
            turn_context.set_turn(conversation_id=cid)
            return turn_context.current_turn()["conversation_id"]

        return ctx.run(_fn)

    with ContextThreadPoolExecutor(max_workers=1) as ex:      # 一根线程，故意的
        a = ex.submit(_run_as, "conv-A").result(timeout=10)
        b = ex.submit(_run_as, "conv-B").result(timeout=10)
    assert (a, b) == ("conv-A", "conv-B")


def test_clear_turn_leaves_nothing_behind():
    """回合结束必须清干净 —— 残留会让下一轮的产物记上上一轮的来历。"""
    turn_context.set_turn(conversation_id="c", run_id="r", agent_id="a")
    turn_context.clear_turn()
    assert turn_context.current_turn() == {
        "conversation_id": None, "run_id": None, "agent_id": None}


def test_turn_scope_restores_instead_of_clearing():
    """嵌套回合：离开内层要**恢复**外层，不是清空。"""
    turn_context.set_turn(conversation_id="outer", run_id="ro")
    with turn_context.turn_scope(conversation_id="inner"):
        assert turn_context.current_turn()["conversation_id"] == "inner"
    now = turn_context.current_turn()
    assert now["conversation_id"] == "outer" and now["run_id"] == "ro"


def test_turn_scope_restores_even_across_a_context_boundary():
    """``turn_scope`` 不许用 ``ContextVar.reset(token)``。

    token 只能在**同一个上下文**里 reset；这个 with 完全可能跨执行器边界地进出
    （群聊 super-step 里派生子 agent 就是这个形状），那时 reset 抛 ValueError 会把
    finally 一起炸掉。写回旧值到哪儿都成立。
    """
    turn_context.set_turn(conversation_id="outer")

    def _nested():
        with turn_context.turn_scope(conversation_id="inner"):
            pass
        return turn_context.current_turn()["conversation_id"]

    with ContextThreadPoolExecutor(max_workers=1) as ex:
        assert ex.submit(_nested).result(timeout=10) == "outer"
    assert turn_context.current_turn()["conversation_id"] == "outer"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
