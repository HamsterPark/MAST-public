"""用户的「停止」必须穿透到正在跑的技能(缺陷⑬,2026-08-06 实机,两天撞三次)。

## 根因不是「循环没查 abort」

长等待循环**确实**每个 poll 都在查 `ctx.check_abort()`(AutoApproach 的等待相、
WaitScanComplete、Z 稳定循环都查)。坏的是它们查的那个**事件集**:

回合开头在某一个 worker 线程上设了 `engine._tl.abort`。但这条流是同步 generator,
交给 `StreamingResponse` 后由 `iterate_in_threadpool` 消费 —— **每次 `next()` 单独走
一趟 `anyio.to_thread.run_sync`,不保证落在同一个线程上**(仓里
`turn_context.reassert_turn_each_resume` 的 docstring 记着实测:两条并发流互相读到
对方的值)。恢复执行落到线程 B 时,`_tl.abort` 在 B 上根本没有 ⇒ 那一步建的
`ExecutionContext` 的 abort 并集里只剩 `_orch_abort`(E_STOP)⇒ 聊天的 Stop 到不了
硬件。

**这也解释了为什么是「三次」而不是「每次」** —— 落在哪个线程是随机的。

当初为 `conversation_id` / `run_id` 修过同一个坑(每次恢复前重设),唯独漏了 abort。

## 这个文件钉什么

1. 换线程后仍然拿得到本回合的 Stop 事件(根因那一条);
2. 隔离没被破坏:另一个会话的 Stop 取不到我的事件;
3. 回合结束必须摘登记 —— 否则下一个回合会读到上一回合那个**已经 set 过**的事件,
   然后第一个技能什么都没干就说自己被停了。
"""
from __future__ import annotations

import sys
import threading
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

import pytest  # noqa: E402

from mast.chat.engine import ConversationEngine  # noqa: E402
from mast.core import turn_context  # noqa: E402


def _engine() -> ConversationEngine:
    """一个不跑图的引擎:这些钉子只关心 abort 的取用,不关心对话。"""
    eng = ConversationEngine.__new__(ConversationEngine)
    eng._tl = threading.local()
    eng._abort_by_conv = {}
    eng._abort_by_conv_lock = threading.Lock()
    return eng


def _on_another_thread(fn):
    """在**另一个**线程上取值 —— 模拟 generator 恢复执行落到别的 worker。"""
    box: dict = {}
    t = threading.Thread(target=lambda: box.setdefault("v", fn()))
    t.start()
    t.join()
    return box.get("v")


def test_the_stop_event_survives_a_thread_hop():
    """**根因那一条。** 换线程后仍然拿得到本回合的 Stop 事件。"""
    eng = _engine()
    ev = threading.Event()
    eng._tl.abort = ev                      # 回合开头(线程 A)
    eng._register_abort("conv-1", ev)

    # 恢复执行落到线程 B:thread-local 读不到,但会话登记读得到。
    turn_context.set_turn(conversation_id="conv-1")

    def _read():
        turn_context.set_turn(conversation_id="conv-1")   # 每次恢复前重设(仓里已有)
        return eng.active_abort_event()

    assert _on_another_thread(_read) is ev

    # 探针有效性:没有这次修复的话,换线程就是 None —— 把登记摘掉再看一次。
    eng._unregister_abort("conv-1")
    assert _on_another_thread(_read) is None


def test_the_same_thread_fast_path_still_wins():
    """同线程时走 thread-local:它一定是对的,而且不用拿锁。"""
    eng = _engine()
    mine, other = threading.Event(), threading.Event()
    eng._tl.abort = mine
    eng._register_abort("conv-1", other)
    turn_context.set_turn(conversation_id="conv-1")
    assert eng.active_abort_event() is mine


def test_one_conversations_stop_is_not_another_conversations():
    """隔离的**单位是会话**。停一个聊天不许连带停掉并发的另一个。"""
    eng = _engine()
    a, b = threading.Event(), threading.Event()
    eng._register_abort("conv-A", a)
    eng._register_abort("conv-B", b)

    def _read_as(cid):
        def _fn():
            turn_context.set_turn(conversation_id=cid)
            return eng.active_abort_event()
        return _on_another_thread(_fn)

    assert _read_as("conv-A") is a
    assert _read_as("conv-B") is b
    assert _read_as("conv-C") is None


def test_a_finished_turn_leaves_no_stale_event():
    """回合结束必须摘登记。

    留着的话,下一个回合会读到上一回合那个**已经 set 过**的事件,于是第一个技能
    什么都没干就说自己被停了 —— 而这种症状最难查:它看起来像「用户点了停止」。
    """
    eng = _engine()
    ev = threading.Event()
    eng._register_abort("conv-1", ev)
    ev.set()                                  # 上一回合确实被停过
    eng._unregister_abort("conv-1")

    def _fn():
        turn_context.set_turn(conversation_id="conv-1")
        return eng.active_abort_event()

    assert _on_another_thread(_fn) is None


def test_no_turn_context_means_no_event_not_someone_elses():
    """读不到会话 id 就返回 None,**不猜**一个。

    「不知道现在是哪个会话」和「这个会话没有停止事件」都必须导向同一个安全结果:
    不给一个别人的事件。给错了的话,一个会话的 Stop 会停掉另一个会话的技能。
    """
    eng = _engine()
    ev = threading.Event()
    eng._register_abort("conv-1", ev)

    def _fn():
        turn_context.clear_turn()
        return eng.active_abort_event()

    assert _on_another_thread(_fn) is None


def test_registering_nothing_is_a_no_op():
    """没有 abort 事件的回合(内部调用)不该在表里留一个 None。"""
    eng = _engine()
    eng._register_abort("conv-1", None)
    eng._register_abort("", threading.Event())
    assert eng._abort_by_conv == {}


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
