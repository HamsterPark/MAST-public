"""并发流式输出下,provenance 不能记成**另一条对话**的 id。

架构审查 2026-07-29 H1。这条测试用真实的 ``anyio.to_thread`` 驱动两条同步
generator —— 那正是 Starlette 的 ``iterate_in_threadpool`` 消费
``StreamingResponse(sync_generator)`` 的方式。

为什么「回合开头设一次 thread-local」不够:每次 ``next()`` 单独走一趟
``to_thread.run_sync``,**不保证落在同一个 worker 线程上**。于是开头在线程 A 上设的
值,下一个 super-step 可能在线程 B 上执行 —— 读不到是小事,两条流并发时 B 上残留的是
**另一条对话**的值,那就把一条错的因果链当事实存进了 ``doc.json``。

「猜错的 provenance 比没有更糟」是 ``core/turn_context`` 自己的第 3 条纪律。
"""

from __future__ import annotations

import anyio
import pytest

from mast.core.turn_context import (
    clear_turn,
    current_turn,
    reassert_turn_each_resume,
    set_turn,
)

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of


def _one_turn(name: str, holder: dict, seen: list, *, reassert: bool):
    """模拟一条流:开头设回合标识,之后每个 super-step 读一次(工具就是这么读的)。"""

    def impl():
        holder["conversation_id"] = f"conv-{name}"
        holder["run_id"] = f"run-{name}"
        set_turn(conversation_id=f"conv-{name}", run_id=f"run-{name}")
        try:
            for step in range(4):
                seen.append((name, step, current_turn()["conversation_id"]))
                yield step
        finally:
            clear_turn()

    gen = impl()
    return reassert_turn_each_resume(gen, holder) if reassert else gen


_DONE = object()


def _next_or_done(gen):
    """``StopIteration`` 不能穿过 coroutine（会变成 RuntimeError），所以在线程里
    就把它翻译成哨兵 —— Starlette 的 ``iterate_in_threadpool`` 也是这么做的。"""
    try:
        return next(gen)
    except StopIteration:
        return _DONE


def _drain_two_concurrently(reassert: bool) -> list:
    """两条流交替被不同 worker 线程 next() —— 复刻 iterate_in_threadpool 的形状。"""
    seen: list = []

    async def main():
        gens = {n: _one_turn(n, {}, seen, reassert=reassert) for n in ("B", "C")}
        alive = dict.fromkeys(gens, True)
        while any(alive.values()):
            for name, gen in gens.items():
                if not alive[name]:
                    continue
                got = await anyio.to_thread.run_sync(lambda g=gen: _next_or_done(g))
                if got is _DONE:
                    alive[name] = False

    anyio.run(main)
    return seen


def _mismatches(seen: list) -> list:
    return [row for row in seen if row[2] != f"conv-{row[0]}"]


@pytest.fixture(autouse=True)
def _clean():
    clear_turn()
    yield
    clear_turn()


def test_two_concurrent_streams_never_read_each_others_conversation():
    """★ 这是 H1 的回归:B 的工具绝不能读到 C 的 conversation_id。"""
    seen = _drain_two_concurrently(reassert=True)
    assert seen, "没有采到任何 super-step"
    bad = _mismatches(seen)
    assert not bad, f"provenance 串档（把别的对话当成了自己的来历）：{bad}"


def test_the_bug_is_real_without_the_wrapper():
    """反面:不加包装时确实会串档 —— 否则上面那条测试是空过的。

    ``to_thread`` 的线程分配不是确定的,所以这里只要求「至少出现过一次串档或读空」,
    不要求每次都串。真串起来的概率在实测里很高(审查者一次就复现了)。
    """
    seen = _drain_two_concurrently(reassert=False)
    wrong_or_missing = [row for row in seen if row[2] != f"conv-{row[0]}"]
    if not wrong_or_missing:
        pytest.skip("这一次线程分配恰好没有串档；包装层的正确性由上一条测试保证")
    assert wrong_or_missing


def test_holder_filled_late_does_not_leak_a_stale_value():
    """``holder`` 还没被填时包装什么都不设 —— 早退路径不该留下上一个回合的值。"""
    set_turn(conversation_id="previous", run_id="previous-run")
    holder: dict = {}

    def impl():
        yield 1

    list(reassert_turn_each_resume(impl(), holder))
    # 包装的 finally 会 clear，所以离开后是空的（而不是 previous）
    assert current_turn()["conversation_id"] is None


def test_wrapper_propagates_values_and_clears_on_exit():
    holder: dict = {}
    inside: list = []

    def impl():
        holder["conversation_id"] = "conv-X"
        holder["run_id"] = "run-X"
        yield 1
        inside.append(current_turn())
        yield 2

    out = list(reassert_turn_each_resume(impl(), holder))
    assert out == [1, 2]
    assert inside and inside[0]["conversation_id"] == "conv-X"
    assert inside[0]["run_id"] == "run-X"
    assert current_turn()["conversation_id"] is None, "离开后必须清掉"


def test_engine_streams_are_wrapped():
    """引擎的两个流式入口都必须经过包装层（否则修了也白修）。"""
    import inspect

    from mast.chat.engine import ConversationEngine

    for name in ("stream_turn", "stream_events"):
        src = source_of(getattr(ConversationEngine, name))
        assert "reassert_turn_each_resume" in src, f"{name} 没有经过重设包装"
        assert not inspect.isgeneratorfunction(getattr(ConversationEngine, name)), (
            f"{name} 仍是 generator function —— 说明它没有把实现拆到 _impl 里")
