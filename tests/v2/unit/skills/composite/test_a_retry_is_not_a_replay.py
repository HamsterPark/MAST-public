# -*- coding: utf-8 -*-
"""恢复既往进度时，上一次的中止状态**不许**跟着回来。

## 事故（2026-08-23，用户报的第 4 条）

一个 composite 因为配置不对中止了。摆正配置后重跑 ——

    回包用时 **0.0 分钟**，``error`` 字段**一字不差**还是上一轮那句

**根本没跑。** ``aborted`` / ``aborted_reason`` 是持久化字段，恢复既往进度时被
原样带了回来，于是执行器起手就是 ``aborted=True``，每一步都被
``if progress.aborted: return`` 跳过，直接把上一轮的结局重放一遍。

一次「重试」在无人值守里静默变成了「**重放**」，而重放的结论**看起来和真跑
一模一样** —— 这正是本仓记过的那条签名：``elapsed=0`` + 数字一模一样。
当时的绕开办法是改一个显式参数让步骤签名不同，但**调用方不该需要知道这件事**。

## 两个入口

既往进度可以从**上下文**来（``get_progress``），也可以从**磁盘 sidecar** 来。
两条路都走 ``CompositeProgress.from_dict``，都会把 ``aborted`` 恢复回来。
**只堵一个等于没堵** —— 所以两条都钉。

## 这不会削弱用户的中止

活的中止走 ``context.check_abort()``（每一步独立回调），它**不读**
``progress.aborted``。按着中止按钮的话下一步照样停，而且停的理由会是**这一次**
的，不是上一次的。本文件最后一条就在钉这件事。
"""
from __future__ import annotations

import time

from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
    _drop_stale_abort,
)


class _Ctx:
    """最小上下文：只提供执行器真正会问的那几个鸭子方法。"""

    def __init__(self, prior=None, abort=False):
        self._prior = prior
        self._abort = abort
        self.ran: list[str] = []

    def get_progress(self, name):
        return self._prior

    def check_abort(self):
        return self._abort

    def run(self, skill, params=None, **kw):
        self.ran.append(skill)

        class _R:
            success = True
            data: dict = {}
            error = ""
        return _R()


def _aborted_prior(name="X", n=2):
    return CompositeProgress(
        composite_name=name, total_steps=5,
        completed_steps=["s%d" % i for i in range(n)],
        aborted=True, aborted_reason="上一轮那句：Precondition not met",
        last_update_at=time.time())


# ── 入口一：上下文里的既往进度 ──────────────────────────────────────────────

def test_a_prior_progress_that_aborted_does_not_abort_the_new_run():
    ex = GraphExecutor("X", _Ctx(prior=_aborted_prior()))
    assert ex.progress.aborted is False, (
        "起手就带着上一次的中止状态 —— 这一次「重试」会一步不跑地重放上一轮结局")
    assert ex.progress.aborted_reason == "", (
        "中止理由也要清掉：留着它，报告会把上一轮的原因说成这一轮的")


def test_the_completed_steps_are_still_inherited():
    """反证：清的是中止状态，**不是**整个既往进度。

    否则「恢复」就退化成「从头再跑一遍」，而那正是 sidecar 当初要解决的问题
    （中断后重放会把仪器动作再做一遍）。
    """
    ex = GraphExecutor("X", _Ctx(prior=_aborted_prior(n=3)))
    assert len(ex.progress.completed_steps) == 3, ex.progress.completed_steps


# ── 入口二：磁盘 sidecar ────────────────────────────────────────────────────

def test_a_sidecar_that_aborted_does_not_abort_the_new_run(monkeypatch, tmp_path):
    import mast.skills.composite.graph_executor as G

    monkeypatch.setattr(G, "_sidecar_dir", lambda: tmp_path)
    side = _aborted_prior(n=4)
    path = G._sidecar_path("X", "run-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.write_text(json.dumps(side.to_dict()), encoding="utf-8")

    class _CtxRun(_Ctx):
        run_id = "run-1"

    ex = GraphExecutor("X", _CtxRun(prior=None))
    assert len(ex.progress.completed_steps) == 4, (
        "sidecar 没被读进来，这条测试测不到它要测的东西：%s"
        % ex.progress.completed_steps)
    assert ex.progress.aborted is False, (
        "sidecar 那条路把上一次的中止状态带回来了 —— 两个入口只堵了一个")


# ── 同一扇门的第二把锁:**跑完了的**既往进度不许当成断点 ────────────────────
#
# 2026-08-24。上面那两条堵的是 ``aborted``;``completed_steps`` 是同一个入口上的
# 另一半,而它以前**只在 sidecar 那条路上**有守卫。
#
# 后果:一份「每一步都完成了」的快照从 ``ctx.get_progress()`` 原样变成本次的起点
# ⇒ ``is_completed()`` 对每一步都为真 ⇒ **整个计划被跳过** ⇒ ``run_plan`` 回 True
# ⇒ ``aggregate`` 读的是快照里**上一次**的读数 ⇒ 一次零仪器调用的假成功。
#
# 这条路真机可达:``MASTState.composite_progress`` 的 reducer 是 merge_dicts、
# 键是**技能名**,而 ``FullScan`` 本身就是一个可以被直接调用的工具。agent 直接
# 调过一次 FullScan,这个线程往后**每一个**嵌在别人里面的 FullScan 都会读到那份
# terminal 快照 —— 这正是 2026-08-24 那三个「写入时间相差 1 秒、读数逐位相同」
# 的文件的来路之一(见 test_one_attempt_is_one_acquisition.py)。
#
# 同一失败模式的第四次出货,前三次(07-10 假进针 / 07-27 BatchRegionsScan /
# 08-12 ForgeAuTip)都堵在 sidecar 那扇门上。**一个坑的两个入口,只堵一个等于没堵。**


def _finished_prior(name="FullScan", n=4):
    return CompositeProgress(
        composite_name=name, total_steps=n,
        completed_steps=["s%d" % i for i in range(n)],
        partial_data={"scan_lines_done": 256, "wait_outcome": "completed"},
        last_update_at=time.time())


_FOUR = [CompositeStep(step_id="s%d" % i, skill_name="Sub%d" % i, params={})
         for i in range(4)]


def test_a_finished_prior_progress_does_not_skip_the_whole_plan():
    ctx = _Ctx(prior=_finished_prior())
    ex = GraphExecutor("FullScan", ctx)
    ok = ex.run_plan(iter(_FOUR))
    assert ok is True
    assert ctx.ran == ["Sub0", "Sub1", "Sub2", "Sub3"], (
        "一步都没跑就报成功 —— 零仪器调用的假成功:%s" % ctx.ran)


def test_the_stale_readings_do_not_come_back_with_it():
    """跳步的第二重伤害:``aggregate`` 会把**上一次**的读数当成这一次的。

    「512/512 行、completed」这种句子在报告里和真跑一模一样,而它描述的是
    另一次采集 —— 报告里最难发现的错误,就是一个合理的旧数字。
    """
    ex = GraphExecutor("FullScan", _Ctx(prior=_finished_prior()))
    assert ex.progress.partial_data.get("scan_lines_done") is None, (
        "上一次的读数跟着回来了:%s" % ex.progress.partial_data)


def test_an_interrupted_prior_progress_still_resumes():
    """**安全对照。**丢掉的只是「跑完了的」那种,真正被打断的照旧续跑。

    否则「恢复」退化成「从头再跑一遍」—— 重放硬件动作,并把用户已经答过的
    问题再问一遍。这条不过,上面那两条就只是把功能删了。
    """
    ctx = _Ctx(prior=CompositeProgress(
        composite_name="FullScan", total_steps=4,
        completed_steps=["s0", "s1"], last_update_at=time.time()))
    ex = GraphExecutor("FullScan", ctx)
    ok = ex.run_plan(iter(_FOUR))
    assert ok is True
    assert ctx.ran == ["Sub2", "Sub3"], (
        "被打断的续跑被一起丢掉了 —— 这会重放硬件动作:%s" % ctx.ran)


def test_a_streaming_prior_progress_with_unknown_total_still_resumes():
    """``total_steps == 0``(流式 composite 步数事先不知道)时**不判 terminal**。

    分母不存在就不该下结论 —— 宁可多恢复一次,也别把一次真正的中断续跑判死。
    与 sidecar 那条路上的同一条纪律。
    """
    ctx = _Ctx(prior=CompositeProgress(
        composite_name="WaitScanComplete", total_steps=0,
        completed_steps=["s0", "s1"], last_update_at=time.time()))
    ex = GraphExecutor("WaitScanComplete", ctx)
    ex.run_plan(iter(_FOUR))
    assert ctx.ran == ["Sub2", "Sub3"], ctx.ran


def test_a_dict_shaped_finished_progress_is_caught_too():
    """真机上这份快照是**字典**(它从 checkpointer 的 state 里出来),不是对象。

    只认对象的话,这道守卫在生产路径上一次都不会触发 —— 一道
    「看着在防护、其实从没触发过」的守卫。
    """
    ctx = _Ctx(prior=_finished_prior().to_dict())
    ex = GraphExecutor("FullScan", ctx)
    ex.run_plan(iter(_FOUR))
    assert ctx.ran == ["Sub0", "Sub1", "Sub2", "Sub3"], ctx.ran


# ── 助手本身 ────────────────────────────────────────────────────────────────

def test_the_helper_leaves_a_clean_progress_alone():
    pr = CompositeProgress("X", completed_steps=["a"])
    _drop_stale_abort(pr, "X", "test")
    assert pr.aborted is False and pr.completed_steps == ["a"]


# ── 用户的中止不受影响 ────────────────────────────────────────────────────

def test_a_live_abort_still_stops_the_run():
    """按着中止按钮时照样停 —— 活的中止不走 ``progress.aborted``。

    这一条是上面那三条的**安全对照**：把陈旧标志清掉，不等于把停止按钮也拆了。
    """
    ctx = _Ctx(prior=_aborted_prior(), abort=True)
    ex = GraphExecutor("X", ctx)
    assert ex.progress.aborted is False          # 陈旧的那个已清
    ok = ex.run_plan([CompositeStep(step_id="s9", skill_name="Noop", params={})])
    assert ok is False, "中止按钮按着，却把计划跑完了"
    assert ex.progress.aborted is True, "活的中止应当把**这一次**标成中止"
    assert "s9" not in ctx.ran, "中止按着还执行了步骤：%s" % ctx.ran
