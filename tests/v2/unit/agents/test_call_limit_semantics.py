"""每轮调用上限:``0`` 的含义、出厂值的唯一性,以及**可读的守卫必须先响**。

## 为什么有这个文件

要求：调用上限必须可关闭或调大，不能是不可变的常量。

复杂工作流可能跨多次工具调用，因此调用预算必须可配置，且设置写入口必须实际生效。

## 这个文件钉的三件事

1. **``0`` 真的表示「不限」** —— 而不是「用出厂值」。
   ``_pos()`` 那个「v<=0 → 出厂默认」的形状本仓今天已经付过三次学费
   (ForgeAuTip 的 per_run、看门狗的 ``disable(for_s)``、这里)。
2. **出厂值只有一份** —— 六张 agent 图曾经各自硬编码 30,和常量是六份副本。
3. ⚠️ **可读的守卫必须先响。** 这条最容易被漏:把上限从 30 提到 500,
   如果不同时抬 ``MAX_RECURSION_LIMIT``,拿到的**不是 500**,
   是 ≈109 次调用 + 一条读不懂的 ``GraphRecursionError``。
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

import inspect  # noqa: E402
import logging  # noqa: E402

# 源码级断言一律走它,不用 ``inspect.getsource``(2026-08-15)——
# 后者按 import 那一刻的行号切当前文件,别人同时在改就返回错位切片:
# ``in`` 那半给假红(吵、会被查),``not in`` 那半给**假绿**(不吵、没人会查)。
from tests.v2.srcref import source_of  # noqa: E402

from mast.agents._shared.call_limits import (  # noqa: E402
    DEFAULT_MODEL_CALLS_PER_RUN,
    MAX_RECURSION_LIMIT,
    UNLIMITED_RUN_CALLS,
    _norm_run,
    derive_recursion_limit,
)


class _Graph:
    """一张有 ``per_cycle + 2`` 个节点的假图(``__start__`` / ``__end__`` 各一)。"""

    def __init__(self, per_cycle: int):
        self._n = per_cycle + 2

    def get_graph(self):
        return type("g", (), {"nodes": list(range(self._n))})()


# ══════════════════════════════════════════════════════════════════════
# 1. 0 = 不限,而且大声说;负数 = 打错了,不是「关掉」
# ══════════════════════════════════════════════════════════════════════

def test_zero_means_unlimited_not_the_factory_default(caplog):
    """``0`` 必须表示「不限」,不能悄悄退回一个更小的出厂值。

    修复前,填 0 拿到的是 30 —— 比预期**更小**,而且没有任何提示。
    """
    with caplog.at_level(logging.WARNING, logger="mast.agents._shared.call_limits"):
        got = _norm_run(0, DEFAULT_MODEL_CALLS_PER_RUN)
    assert got == UNLIMITED_RUN_CALLS
    assert got != DEFAULT_MODEL_CALLS_PER_RUN, "填 0 又拿到出厂值了"
    assert any(r.levelno >= logging.WARNING for r in caplog.records), (
        "把唯一那道刹车关掉了,却一个字都没说")


def test_a_negative_is_treated_as_a_typo_not_as_off(caplog):
    """负数不是「关掉」(关掉请填 0),是打错了 —— 用出厂值并说一句。"""
    with caplog.at_level(logging.WARNING, logger="mast.agents._shared.call_limits"):
        got = _norm_run(-1, DEFAULT_MODEL_CALLS_PER_RUN)
    assert got == DEFAULT_MODEL_CALLS_PER_RUN
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_a_positive_value_is_used_as_is():
    assert _norm_run(7, DEFAULT_MODEL_CALLS_PER_RUN) == 7


def test_none_falls_back_quietly():
    """没设过就是没设过 —— 那不是降级,不该刷日志。"""
    assert _norm_run(None, DEFAULT_MODEL_CALLS_PER_RUN) == DEFAULT_MODEL_CALLS_PER_RUN


# ══════════════════════════════════════════════════════════════════════
# 2. 出厂值只有一份
# ══════════════════════════════════════════════════════════════════════

def test_the_default_is_big_enough_for_one_basic_experiment_unit():
    """默认预算应给正常的多步骤工作流留有余量。"""
    assert DEFAULT_MODEL_CALLS_PER_RUN >= 300, (
        f"出厂值 {DEFAULT_MODEL_CALLS_PER_RUN} 不满足此测试声明的多步骤预算要求")


def test_no_agent_graph_hardcodes_its_own_default():
    """六张图不许各自写一个字面量 —— 那是六份会漂开的副本。

    2026-08-10 之前它们各自写着 ``= 30``;把常量提到 500 时,那六处会**静静地留在 30**。
    """
    import re

    # 只挑**赋了数字字面量**的那些。``max_model_calls_per_run=max_model_calls_per_run``
    # 是把入参往下传,不是自带出厂值 —— 把它算进来会让这条闸门恒红,
    # 而一条恒红的闸门和一条恒绿的一样没用。
    numeric_default = re.compile(
        r"max_model_calls_per_run\s*(?::\s*[\w\[\], |]+\s*)?=\s*(\d+)")
    offenders = []
    for name in ("data_processing", "experiment_design", "instrument_control",
                 "literature", "paper_review", "paper_writing"):
        mod = __import__(f"mast.agents.{name}.graph", fromlist=["build"])
        for line in inspect.getsource(mod).splitlines():
            m = numeric_default.search(line)
            if m:
                offenders.append(f"{name}: {line.strip()}")
    assert not offenders, (
        "这些图自带了数字出厂值,没有从 call_limits 派生:\n" + "\n".join(offenders))

    # 自检:这条闸门确实认得出违规的写法(否则它可能只是**扫不到任何东西**)。
    assert numeric_default.search("    max_model_calls_per_run: int = 30,"), \
        "闸门连一个明显的违规样本都认不出来 —— 它现在的绿是假的"


def test_the_chat_path_derives_its_default_too():
    """``runtime._chat_call_limits`` 曾经写着字面量 30。

    ⚠️ **取源用 ``source_of``,不用 ``inspect.getsource``**(2026-08-15)。

    这条测试是那一族的**第四个实例,而且它咬人的时候正好是在做提交验收**:
    全量 1 failed / 2979 passed,单跑 10 passed —— 十几分钟花在排除「是不是真回归」上,
    而真因是 ``runtime.py`` 当时有别人在改。``getsource`` 按 import 那一刻记下的行号去切
    **当前磁盘上**的文件,行一移就返回错位切片。

    同一处**两半都在**,而两半的坏法不同:

        assert "DEFAULT_MODEL_CALLS_PER_RUN" in src      ← 假红:吵,会被查
        assert 'chat_model_calls_per_run", 30' not in src ← 假绿:不吵,没人会查

    假红只是浪费时间,假绿是**这条测试可能已经悄悄不守了,而没人知道多久**。
    """
    from mast.core.runtime import CoreRuntime

    src = source_of(CoreRuntime._chat_call_limits)
    # 自检:先证明取到的确实是**这个**方法的源码。取错/取空时下面那条 ``not in``
    # 恒真,而恒真的断言与守得住的断言在报告里长得一模一样。
    assert "def _chat_call_limits" in src, (
        f"取到的不是 _chat_call_limits 的源码(前 80 字:{src[:80]!r})—— "
        "下面那条 not in 会因此恒真,先修取源")
    assert "DEFAULT_MODEL_CALLS_PER_RUN" in src, "聊天路径没派生出厂值"
    assert "chat_model_calls_per_run\", 30" not in src, "字面量 30 又回来了"


# ══════════════════════════════════════════════════════════════════════
# 3. ⚠️ 可读的守卫必须先响 —— 这条是整个改动真正承重的地方
# ══════════════════════════════════════════════════════════════════════

class TestTheReadableGuardBindsFirst:
    """``ModelCallLimitMiddleware`` 用一句用户看得懂的话结束回合;
    ``recursion_limit`` 用 ``GraphRecursionError`` 结束回合。
    **必须是前者先响**,否则「上限 500」是句空话。

    实测(2026-08-10)把 30 提到 500 而不动 ``MAX_RECURSION_LIMIT`` 的后果:

        per_cycle=11 → want = 11×501 = 5511 > 1200 ⇒ 夹到 1200
                     ⇒ 实际 ≈109 次调用,而且以 GraphRecursionError 结束
    """

    def test_at_shipped_defaults_the_model_cap_binds_for_realistic_graphs(self):
        """生产 IC 图今天是 11;每加一个带 before/after_model 的中间件 +1。
        留到 17 的余量。"""
        for per_cycle in range(5, 18):
            want = per_cycle * (DEFAULT_MODEL_CALLS_PER_RUN + 1)
            got = derive_recursion_limit(
                _Graph(per_cycle), model_calls_per_run=DEFAULT_MODEL_CALLS_PER_RUN)
            assert got >= want, (
                f"per_cycle={per_cycle}:recursion 预算 {got} < 需要的 {want} —— "
                f"回合会在 ≈{got // per_cycle} 次调用时以 GraphRecursionError 结束,"
                f"而不是以那条可读的「Model call limits exceeded」结束")

    def test_the_cap_is_what_makes_this_possible(self):
        """把不变式和那个数字绑在一起:``MAX_RECURSION_LIMIT`` 必须够大。

        这条会在有人「顺手把 MAX_RECURSION_LIMIT 调回去」时红。
        """
        need = 17 * (DEFAULT_MODEL_CALLS_PER_RUN + 1)
        assert MAX_RECURSION_LIMIT >= need, (
            f"MAX_RECURSION_LIMIT={MAX_RECURSION_LIMIT} 撑不住 "
            f"{DEFAULT_MODEL_CALLS_PER_RUN} 次调用(需要 {need})")

    def test_unlimited_still_gets_a_recursion_backstop(self):
        """填 0(不限)之后 recursion 成为**唯一**的兜底 —— 它必须仍然存在。"""
        got = derive_recursion_limit(_Graph(11), model_calls_per_run=0)
        assert got == MAX_RECURSION_LIMIT
        assert got > 0
