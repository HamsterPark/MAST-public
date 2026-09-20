"""可中止性是**动作的属性**，不是一句形容词 —— 这张表得说真话。

钉两件事：

1. **表本身自洽**：每个「能打断阻塞调用的停止动词」都必须在
   ``execution_context._ABORT_SAFE_WRITES`` 的放行名单里。不然中止之后那个停止动词
   自己会被 abort 闸门拒掉 —— 收尾路径把自己掐死，而症状是「我明明发了停止」。
2. **告知不许撒谎**：``BLOCKING_HELD`` 那一类的说明里，不许出现「可以随时中止」
   这种句式；而且只要 ``soft_stop_sends_hardware_verbs()`` 还是 False，
   面向用户的那句话就必须说明**软停不会替你发那个动词**。

外加一条 ③ 的钉子：``WaitForScanEndBlocking`` 曾经是一条最长 1800 秒、中间零检查点
的阻塞调用，而它的说明写着「中止 still works」。现在它切片了，这里钉住它真的会在
中途查停止，并且**不把「停止等待」说成「停止扫描」**。
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

import pytest  # noqa: E402

from mast.core import abortability as ab  # noqa: E402
from mast.core.execution_context import _ABORT_SAFE_WRITES  # noqa: E402


def test_every_stop_verb_survives_the_abort_gate():
    """收尾动词必须能在中止之后发出去。

    这是 ``_panic`` 那套能成立的**前提**：中止闩扳下之后，``safe_call`` 会拒掉写命令，
    只放行读取和名单里的停止动词。一个不在名单里的停止动词 = 一条永远发不出去的
    收尾路径。
    """
    missing = [v for v in ab.INTERRUPTED_BY.values() if v not in _ABORT_SAFE_WRITES]
    assert not missing, f"这些停止动词会被 abort 闸门自己拒掉：{missing}"


def test_every_stop_verb_is_a_real_nanonis_method():
    """表里写错一个名字 = 一个永远发不出去、且**没人会发现**的停止动词。"""
    from nanonis_spm import Nanonis

    unknown = [v for v in ab.INTERRUPTED_BY.values() if not hasattr(Nanonis, v)]
    assert not unknown, f"这些不是真的 Nanonis 方法：{unknown}"


def test_blocked_verbs_are_real_nanonis_methods():
    from nanonis_spm import Nanonis

    unknown = [k for k in ab.INTERRUPTED_BY if not hasattr(Nanonis, k)]
    assert not unknown, f"这些不是真的 Nanonis 方法：{unknown}"


def test_the_blocking_class_does_not_claim_you_can_stop_it_any_time():
    """④：**不许**出现「有风险但可以随时中止」这种句式。"""
    text = ab.describe(ab.Abortability.BLOCKING_HELD, verb="Motor_StartMove")
    assert "随时中止" not in text
    assert "打断不了" in text


def test_the_blocking_class_says_what_WOULD_interrupt_it_and_who_sends_it():
    """「停不了」和「没人去发那个动词」是两回事，混成一句会让能修的事被当成不能修。

    ``Motor_StartMove`` 的 wait 标志按 nanonis 文档是「到位**或运动停止**才返回」，
    而 ``Motor_StopMove`` 存在且无条件放行 —— 所以它**不是不可逆的**。缺的是有人
    从急停 socket 上去发它。
    """
    text = ab.describe(ab.Abortability.BLOCKING_HELD, verb="Motor_StartMove")
    assert "Motor_StopMove" in text
    assert "不是不可逆" in text
    # 只要软停还不下发硬件动词，这句话就必须在。
    assert ab.soft_stop_sends_hardware_verbs() is False
    assert "紧急停止" in text


def test_the_polled_class_is_allowed_to_promise_a_stop():
    text = ab.describe(ab.Abortability.POLLED)
    assert "可中止" in text and "打断不了" not in text


def test_between_steps_says_the_latency_is_not_zero():
    text = ab.describe(ab.Abortability.BETWEEN_STEPS)
    assert "步与步之间" in text and "不是零" in text


def test_an_unknown_blocking_verb_is_not_given_a_reassurance():
    """表里没登记 ⇒ 按最保守的说 —— 不发明一个不存在的停止动词。"""
    text = ab.describe(ab.Abortability.BLOCKING_HELD, verb="Some_UnknownStart")
    assert "发出即跑完" in text


def test_the_api_caveat_comes_from_this_module():
    """`/chat/abort` 的告知文案必须是**取**来的，不是另抄一份。

    抄一份就是第二真源，然后在某次「顺手改一下」里静静变成假话。
    """
    from mast.api.routes.agents import _stop_caveat

    assert _stop_caveat() == ab.SOFT_STOP_CAVEAT
    assert "软停不会替你下发任何硬件停止动词" in ab.SOFT_STOP_CAVEAT


# ── ③ WaitForScanEndBlocking 真的会在中途查停止 ────────────────────────


class _Rec:
    def __init__(self, rv=(1, 0, ""), error=""):
        self.return_value, self.error = rv, error
        self.method, self.args = "Scan_WaitEndOfScan", ()


class _Ctx:
    """停止在第 N 次调用后到位。不碰仪器。"""

    def __init__(self, abort_after: int = 2, ends_at: int | None = None):
        self.calls: list = []
        self._abort_after, self._ends_at = abort_after, ends_at

    def check_abort(self) -> bool:
        return len(self.calls) >= self._abort_after

    def safe_call(self, method, *args):
        self.calls.append((method, args))
        ended = self._ends_at is not None and len(self.calls) >= self._ends_at
        return _Rec(rv=(0 if ended else 1, 0, ""))


def _skill():
    from mast.skills.builtins.advanced_ops import WaitForScanEndBlocking

    return WaitForScanEndBlocking()


def test_the_blocking_wait_is_sliced_not_one_long_call():
    """一条 1800 秒、零检查点的阻塞调用是这个缺陷最纯粹的形态。"""
    ctx = _Ctx(abort_after=3)
    _skill().execute(ctx, {"timeout_s": 600.0})
    assert len(ctx.calls) >= 2, "还是一条长调用 —— 中途没有任何检查点"
    for _m, args in ctx.calls:
        assert args[0] <= 1000, f"切片 {args[0]} ms 太长，停止延迟会超过 1 秒"


def test_the_operator_stop_ends_the_wait():
    ctx = _Ctx(abort_after=2)
    out = _skill().execute(ctx, {"timeout_s": 600.0})
    assert out.success is False
    assert out.data["aborted"] is True
    assert len(ctx.calls) == 2, "停止到位之后还在继续等"


def test_being_stopped_does_not_claim_the_scan_was_stopped():
    """「停止等待」不是「停止扫描」。把两者说成一件事，用户会以为扫描停了。"""
    out = _skill().execute(_Ctx(abort_after=1), {"timeout_s": 600.0})
    assert "扫描本身没有被停" in (out.error or "")


def test_the_exact_end_of_scan_moment_is_not_lost():
    """切片的代价不能是精度 —— 扫描结束的那一刻仍然立刻返回。"""
    ctx = _Ctx(abort_after=99, ends_at=3)
    out = _skill().execute(ctx, {"timeout_s": 600.0})
    assert out.success is True and out.data["timed_out"] is False
    assert len(ctx.calls) == 3, "扫描结束了还在继续切片等待"


def test_the_description_no_longer_claims_a_stop_path_nobody_walks():
    """原说明写着「中止 still works（急停口是另一条 socket）」 —— 急停口确实在，
    但软停不往那条 socket 上发任何东西，所以那句话描述的是一个没人走的通道。"""
    desc = _skill().metadata().description
    assert "the emergency port is a separate socket" not in desc
    # 说明译成中文之后，只挡英文的那道否定断言就再也响不了了 —— 那句假话
    # 若要回来，回来的会是中文版。两边都挡住，这道防护才还活着。
    assert "急停口是另一条 socket" not in desc
    assert "切成" in desc


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
