"""Regression tests for the per-agent call-limit refactor (2026-07-07).

A durable chat / group thread must NOT be bricked by the cumulative
ModelCallLimit thread cap. Covers make_call_limit_middleware config +
CoreRuntime._chat_call_limits.
"""
from mast.agents._shared.call_limits import make_call_limit_middleware


def test_thread_off_disables_cumulative_cap():
    tool_mw, model_mw = make_call_limit_middleware(
        max_model_calls=None, max_tool_calls=None,
        max_model_calls_per_run=30, max_tool_calls_per_run=80)
    assert model_mw.thread_limit is None
    assert model_mw.run_limit == 30
    assert tool_mw.thread_limit is None
    assert tool_mw.run_limit == 80


def test_zero_means_off():
    _, model_mw = make_call_limit_middleware(max_model_calls=0, max_tool_calls=0)
    assert model_mw.thread_limit is None


def test_run_clamped_to_thread():
    tool_mw, model_mw = make_call_limit_middleware(
        max_model_calls=12, max_tool_calls=40,
        max_model_calls_per_run=30, max_tool_calls_per_run=80)
    assert (tool_mw.thread_limit, tool_mw.run_limit) == (40, 40)
    assert (model_mw.thread_limit, model_mw.run_limit) == (12, 12)


def test_defaults_keep_oneshot_backstop():
    """一次性路径:thread 上限仍在,run 上限被**夹到 thread 上限**。

    2026-08-10 出厂 run 上限 30 → 500,而这条路的 thread 上限是 40 ——
    ``make_call_limit_middleware`` 里那条既有的 ``m_run > m_thread`` 夹紧于是生效,
    run 变成 40。**这不是回归,是那条夹紧在做它该做的事**:
    一个小的累计上限不该被一个更大的每轮上限架空。
    """
    tool_mw, model_mw = make_call_limit_middleware()
    assert (model_mw.thread_limit, model_mw.run_limit) == (40, 40)
    assert (tool_mw.thread_limit, tool_mw.run_limit) == (120, 80)


def test_tool_exit_behavior_end():
    tool_mw, _ = make_call_limit_middleware(tool_exit_behavior="end")
    assert tool_mw.exit_behavior == "end"


def test_run_cap_always_positive():
    """run 上限永远是正整数(中间件的不变式:两个上限不能都是 None)。

    ## 2026-08-10:``0`` 的含义**被刻意反转了**

    这条测试原来断言 ``max_model_calls_per_run=0 → run_limit == 30``,
    也就是**「填 0 拿到出厂值」** —— 那正是用户那句「模型不要设调用上限了吧」
    会踩的坑:他去填 0,拿到的是一个比默认更小的限制,而且没有任何提示。

    现在 ``0`` = **不限**(一个够不到的大数,见 ``UNLIMITED_RUN_CALLS``——
    不能真传 None,否则两个上限都是 None,中间件构造器直接 ValueError)。
    **负数**仍然是「打错了」→ 出厂值,因为「关掉」有它自己的写法(0)。
    """
    from mast.agents._shared.call_limits import (
        DEFAULT_TOOL_CALLS_PER_RUN, UNLIMITED_RUN_CALLS,
    )

    tool_mw, model_mw = make_call_limit_middleware(
        max_model_calls=None, max_tool_calls=None,
        max_model_calls_per_run=0, max_tool_calls_per_run=-5)
    assert model_mw.thread_limit is None
    assert model_mw.run_limit == UNLIMITED_RUN_CALLS, "0 又被读成出厂值了"
    assert tool_mw.run_limit == DEFAULT_TOOL_CALLS_PER_RUN, "负数不该是「关掉」"


def _chat_limits(settings):
    from mast.core.runtime import CoreRuntime

    class _Stub:
        _settings = settings

    return CoreRuntime._chat_call_limits(_Stub())


def test_chat_call_limits_default_thread_off():
    out = _chat_limits(None)
    assert out["max_model_calls"] is None
    assert out["max_tool_calls"] is None
    # 出厂值从 call_limits 派生,不在这里抄一个数字 —— 抄了就是第七份副本。
    from mast.agents._shared.call_limits import DEFAULT_MODEL_CALLS_PER_RUN

    assert out["max_model_calls_per_run"] == DEFAULT_MODEL_CALLS_PER_RUN
    assert out["max_tool_calls_per_run"] == 80


def test_chat_call_limits_configurable():
    class _FakeSettings:
        def __init__(self, d):
            self._d = d

        def get(self, k, default=None):
            return self._d.get(k, default)

    out = _chat_limits(_FakeSettings({
        "chat_model_calls_per_thread": 200,
        "chat_tool_calls_per_run": 50,
    }))
    assert out["max_model_calls"] == 200
    assert out["max_tool_calls"] is None
    from mast.agents._shared.call_limits import DEFAULT_MODEL_CALLS_PER_RUN

    assert out["max_model_calls_per_run"] == DEFAULT_MODEL_CALLS_PER_RUN
    assert out["max_tool_calls_per_run"] == 50
