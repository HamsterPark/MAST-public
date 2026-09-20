"""L2 值守席：只读是结构，授权来自模板。

## 这一组要证明的两件事

1. **「只读」拦得住一个不配合的调用方。** 提示词和工具清单只决定模型看得见
   什么；真正的边界在注册表视图上——即使凭空说出一个写技能的名字，也查不到。
   所以这里的关键用例是**塞一个写技能进去，然后确认它取不出来**（变异式），
   而不是「列一列视图里有什么」。
2. **权限来自 `allowed_escalations`，不来自这一席自己。** 越权的建议必须以
   「越权」的形式留痕并转人 —— 悄悄改写成 `wait_operator` 会让记录里长出一句
   「模型建议叫人」，而实际发生的是「模型提议了一件没人授权的事」。
"""

from __future__ import annotations

import pytest

from mast.conduct.l2_seat import (
    READONLY_CATEGORIES,
    EscalationContext,
    FilteredRegistry,
    make_escalation_advisor,
    readonly_registry_view,
)
from mast.conduct.llm_seat import SeatUnavailable


# ── 替身：一张最小的注册表 ────────────────────────────────────────

class _Meta:
    def __init__(self, name, category):
        self.name = name
        self.category = category


class _Skill:
    def __init__(self, name, category):
        self.metadata = _Meta(name, category)


class _Registry:
    def __init__(self, **skills):
        self._m = dict(skills)

    def get(self, name, version=None):
        return self._m.get(name)

    def list_skills(self):
        return list(self._m.values())

    def register(self, name, cls):     # 写接口 —— 视图不该透传
        self._m[name] = cls


def _reg():
    return _Registry(
        ReadScan=_Skill("ReadScan", "READ"),
        AssessThing=_Skill("AssessThing", "ANALYSIS"),
        MoveTip=_Skill("MoveTip", "WRITE"),
        PulseTip=_Skill("PulseTip", "DANGEROUS"),
    )


def _ctx(allowed=("continue_retry", "wait_operator")):
    return EscalationContext(
        conduct_id="c1", stage_id="S2", why="扫了三张都没有原子分辨",
        attempts=1, evidence_epoch=3, allowed=tuple(allowed))


# ── 第一层：注册表视图 ────────────────────────────────────────────

def test_only_read_and_analysis_are_visible():
    v = readonly_registry_view(_reg())
    names = {s.metadata.name for s in v.list_skills()}
    assert names == {"ReadScan", "AssessThing"}


@pytest.mark.parametrize("writer", ["MoveTip", "PulseTip"])
def test_a_write_skill_looks_like_it_does_not_exist(writer):
    """关键用例：不是「拒绝」，是「查无此技能」。

    两者对调用方是不同的信号：被拒绝的人会去找绕过拒绝的路，而「不存在」
    没有什么可绕的。执行层每个子步都按名字回注册表查类，所以这一条同时
    盖住了「模型凭空说出一个写技能名」的情况。
    """
    v = readonly_registry_view(_reg())
    assert v.get(writer) is None
    assert v.get("ReadScan") is not None


def test_a_write_skill_added_later_is_still_invisible():
    """变异式：视图是包装不是快照 —— 后加进去的写技能同样进不来。

    如果这里改成「构造时复制一份只读清单」，这条会绿得一样漂亮，而真实系统里
    技能是会热注册的（覆盖层、composite 热加载）。「看到的和能跑的不是同一张
    表」在本仓踩过。
    """
    inner = _reg()
    v = readonly_registry_view(inner)
    inner.register("SneakyWrite", _Skill("SneakyWrite", "WRITE"))
    assert v.get("SneakyWrite") is None
    inner.register("LateRead", _Skill("LateRead", "READ"))
    assert v.get("LateRead") is not None, "视图不该是构造时的快照"


def test_a_skill_with_no_metadata_is_refused():
    """读不到元数据 ⇒ 判不了它是不是只读 ⇒ 不给。

    「读不到」不是「安全」。这条在本仓记过一整天的账。
    """
    class _Bare:
        pass

    v = readonly_registry_view(_Registry(Mystery=_Bare()))
    assert v.get("Mystery") is None


def test_the_view_has_no_write_side_at_all():
    v = readonly_registry_view(_reg())
    for attr in ("register", "unregister", "discover", "clear"):
        with pytest.raises(AttributeError):
            getattr(v, attr)


def test_nothing_the_seat_can_run_would_ever_take_the_instrument_token():
    """第三层：仲裁面。

    对账方式是**真的去问 `needs_token`**，不是在它的源码里搜字符串——后者会
    因为一个大小写、一次重构而假绿或假红，测的是文字不是行为。

    这一条同时是「L2 的只读定义」与「仲裁层的只读定义」之间的**对账**：
    只要视图里出现了一个仲裁层认为需要令牌的技能，这条就红。那时要改的不是
    这条断言，是想清楚两处对「什么叫只读」为什么有了不同意见。
    """
    from enum import Enum

    from mast.core.instrument_lock import needs_token

    class _Cat(Enum):
        READ = "read"
        ANALYSIS = "analysis"
        WRITE = "write"

    class _M:
        def __init__(self, cat):
            self.name = "X"
            self.category = cat
            self.tags = ()

    # 视图允许的每一类，仲裁层都必须认为它不需要令牌。
    for cat in READONLY_CATEGORIES:
        assert needs_token(_M(_Cat[cat])) is False, (
            f"L2 视图放行 {cat}，而 instrument_lock 认为它要取仪器令牌 —— "
            f"一个「只读」诊断会在真机上和用户抢仪器")
    # 反向：写类确实要令牌（证明上面那条不是恒真）。
    assert needs_token(_M(_Cat.WRITE)) is True


# ── 第二层：闭集与授权 ────────────────────────────────────────────

def test_a_route_inside_the_envelope_comes_back():
    advise = make_escalation_advisor(
        ask=lambda ctx, reg: {"route": "continue_retry", "reason": "再扫一张看看"})
    adv = advise(_ctx())
    assert adv.route == "continue_retry" and "再扫" in adv.reason


def test_a_route_outside_the_envelope_is_not_quietly_rewritten():
    """越权 ⇒ SeatUnavailable，**不是** wait_operator。

    Director 收到这个异常之后照样会转人，但记录里留下的是「越权的提议」，
    不是「模型建议叫人」。两句话指向的下一步不同。
    """
    advise = make_escalation_advisor(
        ask=lambda ctx, reg: {"route": "abort", "reason": "我觉得该停"})
    with pytest.raises(SeatUnavailable) as ei:
        advise(_ctx(allowed=("continue_retry", "wait_operator")))
    assert "越权" in str(ei.value)


def test_an_empty_envelope_means_there_is_nothing_to_authorise():
    advise = make_escalation_advisor(ask=lambda ctx, reg: {"route": "abort"})
    with pytest.raises(SeatUnavailable):
        advise(_ctx(allowed=()))


def test_a_seat_that_throws_is_not_a_verdict():
    def _boom(ctx, reg):
        raise RuntimeError("provider 500")

    advise = make_escalation_advisor(ask=_boom)
    with pytest.raises(SeatUnavailable):
        advise(_ctx())


def test_a_seat_that_returns_junk_is_not_a_verdict():
    advise = make_escalation_advisor(ask=lambda ctx, reg: "continue_retry")
    with pytest.raises(SeatUnavailable):
        advise(_ctx())


def test_a_seat_that_never_returns_is_bounded():
    import time

    def _hang(ctx, reg):
        time.sleep(5)
        return {"route": "continue_retry"}

    advise = make_escalation_advisor(ask=_hang, deadline_s=0.3)
    with pytest.raises(SeatUnavailable) as ei:
        advise(_ctx())
    assert "没回来" in str(ei.value) or "超过" in str(ei.value)


def test_the_envelope_is_shown_to_the_model():
    """模板允许什么，要逐字交给模型看 —— 它是权限边界，不是内部实现。"""
    seen = {}

    def _ask(ctx, reg):
        seen["allowed"] = ctx.allowed
        return {"route": "wait_operator"}

    make_escalation_advisor(ask=_ask)(_ctx(allowed=("continue_retry", "detour",
                                                    "wait_operator")))
    assert seen["allowed"] == ("continue_retry", "detour", "wait_operator")


def test_the_registry_handed_to_the_seat_is_already_filtered():
    """这一层负责收窄，调用方只管给完整表。

    「谁负责收窄」必须只有一个答案：如果调用方也能传一个包好的视图，
    那么某天有人传进来一个没包的，这里看不出区别。
    """
    got = {}

    def _ask(ctx, reg):
        got["reg"] = reg
        return {"route": "wait_operator"}

    make_escalation_advisor(registry_provider=_reg, ask=_ask)(_ctx())
    assert isinstance(got["reg"], FilteredRegistry)
    assert got["reg"].get("MoveTip") is None
