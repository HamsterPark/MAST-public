"""阶段失败之后叫醒 L2：包络内它说了算，包络外交给人。

## 这一组的重点在「没做成的时候会怎样」

一个诊断席最危险的失效不是判错，是**看起来判过了**。所以这里的用例密度压在
四种「没判成」上：没接席位、已经问过一次、越权、席位坏了。每一种都必须
① 转人 ② 在审计流里留下一条能读的原因。

另有一条顺带复活的死字段：`StageSpec.mandatory` 在此之前**零执行读者**（只有
API 和 journal 拿去显示）。它的语义正是「必做的段不许被跳过」——现在由 L2 的
skip_stage 分支真的读它了。
"""

from __future__ import annotations

import pytest

from mast.conduct.l2_seat import EscalationAdvice
from mast.conduct.llm_seat import SeatUnavailable

from ._harness import FakeExecutor, build, outcome, spec, stage, step
from mast.conduct.spec import StageFailPolicy


def _failing_stage(*, then="escalate", allowed=("continue_retry", "wait_operator"),
                   mandatory=False, stage_id="S1"):
    return stage(
        stage_id, steps=(step(f"{stage_id}.00", skill="ScanAt"),),
        on_fail=StageFailPolicy(then=then, max_retries=0),
        allowed_escalations=frozenset(allowed),
        mandatory=mandatory)


def _rig(tmp_path, *, advisor=None, allowed=("continue_retry", "wait_operator"),
         mandatory=False, second_stage=False):
    stages = [_failing_stage(allowed=allowed, mandatory=mandatory)]
    if second_stage:
        stages.append(stage("S2", steps=(step("S2.00", skill="PreScanCheck"),)))
    # S1 用一个**只有 S1 会调**的技能名，好让「这一步失败」不牵连别的阶段。
    ex = FakeExecutor({"ScanAt": [outcome(ok=False, error="扫不出原子分辨")]})
    return build(tmp_path, spec(tuple(stages)), executor=ex,
                 escalation_advisor=advisor)


# ── 没接席位 ──────────────────────────────────────────────────────

def test_with_no_seat_wired_it_says_so_instead_of_pretending(tmp_path):
    rig = _rig(tmp_path)
    rig.run_until("waiting_operator")
    row = rig.row()
    assert row["status"] == "waiting_operator"
    assert "没有接 L2" in (row["status_reason"] or ""), (
        "没接席位时必须说出来 —— 一句「已升级」而实际什么都没发生，"
        "会让下一个人去查一份不存在的诊断结果")
    assert not rig.events(kind="escalation_started"), "没接席位不该记成叫醒过"


# ── 包络内：它说了算 ──────────────────────────────────────────────

def test_a_route_inside_the_envelope_is_carried_out_without_asking(tmp_path):
    """这就是「授权包络」：模板写进 allowed 的处置，L2 说了算。"""
    rig = _rig(tmp_path,
               advisor=lambda ctx: EscalationAdvice("continue_retry", "再试一次"))
    rig.tick(6)
    kinds = rig.event_kinds()
    assert "escalation_started" in kinds and "escalation_verdict" in kinds
    v = rig.events(kind="escalation_verdict")[0]["payload"]
    assert v["ok"] is True and v["route"] == "continue_retry"

    # 「它自己决定了」的证据是那条裁决被**执行**了，不是最终状态。
    # 这里最终仍然会停在等人上，而那是对的：continue_retry 之后重试又失败，
    # 第二次升级撞上「一个代次只问一次」——一次诊断给不出第二个答案。
    # 把这两件事分开断言，是因为它们会因完全不同的原因坏掉。
    retried = [e for e in rig.events(kind="status_change")
               if (e.get("payload") or {}).get("l2_retry")]
    assert retried, "L2 说了 continue_retry，但没有任何东西记下它被执行过"
    reason = rig.row()["status_reason"] or ""
    assert "已经诊断过一次" in reason, (
        f"最终停下来的原因应该是「问过了」，实际是：{reason}")


def test_skip_stage_moves_on_when_the_stage_is_not_mandatory(tmp_path):
    rig = _rig(tmp_path, allowed=("skip_stage", "wait_operator"),
               advisor=lambda ctx: EscalationAdvice("skip_stage", "这段可跳"),
               second_stage=True)
    rig.tick(8)
    assert rig.row()["stage_idx"] >= 1, "非 mandatory 段没有被跳过去"


def test_skip_stage_is_refused_on_a_mandatory_stage(tmp_path):
    """`mandatory` 从此有了执行读者。

    在此之前这个字段零消费者（只有 API 与 journal 显示它）。一个「声明了却
    没人读」的字段，和没有这个字段的区别只是它看起来像有保护。
    """
    rig = _rig(tmp_path, allowed=("skip_stage", "wait_operator"), mandatory=True,
               advisor=lambda ctx: EscalationAdvice("skip_stage", "这段可跳"),
               second_stage=True)
    rig.run_until("waiting_operator")
    row = rig.row()
    assert row["status"] == "waiting_operator"
    assert "mandatory" in (row["status_reason"] or "")
    assert row["stage_idx"] == 0, "mandatory 段被跳过去了"


# ── 包络外 / 坏了：交给人，且说清是哪一种 ─────────────────────────

def test_an_out_of_envelope_advice_is_recorded_as_such(tmp_path):
    """越权要按越权记，不改写成「模型建议叫人」。

    两句话指向的下一步不同：前者要人去看「模型为什么想做这件没授权的事」，
    后者只会让人去看现场。
    """
    def _advisor(ctx):
        raise SeatUnavailable("L2 给的处置 'abort' 不在模板允许的 (...) 里 —— 越权")

    rig = _rig(tmp_path, advisor=_advisor)
    rig.run_until("waiting_operator")
    assert rig.row()["status"] == "waiting_operator"
    v = rig.events(kind="escalation_verdict")[-1]["payload"]
    assert v["ok"] is False and "越权" in v["unavailable"]


def test_a_broken_seat_is_not_a_verdict(tmp_path):
    def _advisor(ctx):
        raise RuntimeError("provider 500")

    rig = _rig(tmp_path, advisor=_advisor)
    rig.run_until("waiting_operator")
    v = rig.events(kind="escalation_verdict")[-1]["payload"]
    assert v["ok"] is False and "provider 500" in v["error"]
    assert "L2 诊断抛异常" in (rig.row()["status_reason"] or "")


# ── 一个阶段一个代次只叫一次 ──────────────────────────────────────

def test_the_seat_is_woken_at_most_once_per_stage_per_epoch(tmp_path):
    calls = []

    def _advisor(ctx):
        calls.append(ctx.stage_id)
        return EscalationAdvice("continue_retry", "再来")

    rig = _rig(tmp_path, advisor=_advisor)
    rig.tick(20)
    assert len(calls) == 1, (
        f"L2 被叫醒了 {len(calls)} 次。同样的输入不会给出不同的答案，"
        f"而每一次都是一整个 agent run。")
    assert "已经诊断过一次" in (rig.row()["status_reason"] or "")


def test_the_wake_is_counted_before_the_call_not_after(tmp_path):
    """先记「叫醒了」再去问 —— 顺序是有意的。

    如果这次调用把进程带走了，重启之后计数里仍然有这一条；否则一个每次都让
    Director 崩溃的诊断，会在每次重启之后重来一遍。
    """
    def _kaboom(ctx):
        raise RuntimeError("boom")

    rig = _rig(tmp_path, advisor=_kaboom)
    rig.tick(4)
    assert rig.events(kind="escalation_started"), (
        "调用炸了之后没有留下「叫醒过」的痕迹")


def test_the_seat_sees_the_envelope_and_the_failure(tmp_path):
    seen = {}

    def _advisor(ctx):
        seen.update(stage=ctx.stage_id, why=ctx.why, allowed=ctx.allowed,
                    epoch=ctx.evidence_epoch)
        return EscalationAdvice("continue_retry", "ok")

    rig = _rig(tmp_path, advisor=_advisor,
               allowed=("continue_retry", "detour", "wait_operator"))
    rig.tick(6)
    assert seen["stage"] == "S1"
    assert "扫不出原子分辨" in seen["why"]
    assert set(seen["allowed"]) == {"continue_retry", "detour", "wait_operator"}
