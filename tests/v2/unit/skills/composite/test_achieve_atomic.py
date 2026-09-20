# -*- coding: utf-8 -*-
"""AchieveAtomicResolution 的合成编排测试。

验证分级处理、预算与重新判定；证据不足时不得据此干预针尖。
所有判据结果来自测试内构造的输入，不代表实验验证。
"""
from __future__ import annotations

import pytest

from mast.skills.composite.achieve_atomic import AchieveAtomicResolution


class _Res:
    def __init__(self, data):
        self.data = dict(data or {})


def _drive(params=None, scripted=None, max_steps=40):
    """跑 ``plan_dynamic``，按 ``scripted`` 给每一步喂结果。

    回 ``(走过的 step_id 列表, executor)``。**不执行**任何真步骤 ——
    这里测的是编排，不是硬件。
    """
    from mast.skills.composite.graph_executor import CompositeProgress

    scripted = dict(scripted or {})

    class _Ex:
        def __init__(self):
            self.progress = CompositeProgress("AchieveAtomicResolution")
            self.sub_results = {}

        def set_partial(self, k, v):
            self.progress.partial_data[k] = v

        def set_total_steps(self, n):
            pass

    ex = _Ex()
    seen: list[str] = []
    # 默认现状：反馈开着、针在隧道结上
    scripted.setdefault("p0:zctrl", {"controller_on": True, "setpoint": 5e-10})
    for st in AchieveAtomicResolution().plan_dynamic(dict(params or {}), ex):
        seen.append(st.step_id)
        ex.sub_results[st.step_id] = _Res(scripted.get(st.step_id, {}))
        if len(seen) > max_steps:  # pragma: no cover — 防跑飞
            pytest.fail("步骤停不下来：%s" % seen)
    return seen, ex


def _skills_of(seen):
    return [s.split(":")[0] for s in seen]


def _absent(n=3):
    # 合成历史带路径，让表面分诊的 FindFlatRegion 分支也能被执行。
    return {"found": False, "attempts": n, "n_undetermined": 0,
            "history": [{"attempt": i + 1, "verdict": "absent",
                         "concentration": 5.0, "path": "f%d.sxm" % (i + 1)}
                        for i in range(n)]}


def _undetermined(n=3):
    return {"found": False, "attempts": n, "n_undetermined": n,
            "history": [{"attempt": i + 1, "verdict": "undetermined"}
                        for i in range(n)]}


def _found(path="X.sxm"):
    return {"found": True, "found_at_attempt": 1, "found_path": path,
            "history": [{"attempt": 1, "verdict": "atomic",
                         "concentration": 240.0}]}


# ── 零必填参数：这是「简单指令能不能成功」的前提 ────────────────────────────

def test_it_takes_no_required_parameters():
    """用户只说「我想要原子分辨」时，没有任何数需要别人替它填。"""
    md = AchieveAtomicResolution().metadata()
    assert [p.name for p in md.parameters if p.required] == []


# ── 安全属性：判不了 ≠ 没有 ────────────────────────────────────────────────

def test_every_frame_undecidable_never_touches_the_tip():
    """每一帧都「判不了」时，**一次针尖都不许动**。

    这是本技能最重要的一条。判不了是证据不足，不是关于针尖的证据；
    在这上面修针 = 在自己造出来的空白上判读。
    """
    seen, ex = _drive(scripted={"r1": _undetermined(3)})
    skills = _skills_of(seen)
    for forbidden in ("r2", "r3a", "r3b", "r4"):
        assert forbidden not in skills, (
            "判不了却动了针尖（%s）—— 走过的步骤：%s" % (forbidden, seen))
    out = AchieveAtomicResolution().aggregate({}, ex.progress)
    assert "没有动针尖" in (out.get("advice") or ""), out.get("advice")


def test_absent_frames_do_let_it_climb():
    """反证：判据说「**没有**」时，阶梯必须照爬 —— 否则上面那条闸门是过严的。"""
    seen, _ = _drive(scripted={"r1": _absent(3)})
    assert "r2" in _skills_of(seen), seen


# ── 不在成像状态：如实停下，不替用户进针 ────────────────────────────────────

def test_feedback_off_stops_before_anything_else():
    seen, ex = _drive(scripted={"p0:zctrl": {"controller_on": False}})
    assert seen == ["p0:zctrl"], seen
    out = AchieveAtomicResolution().aggregate({}, ex.progress)
    assert "不在成像状态" in (out.get("advice") or "")
    assert "不替你进针" in (out.get("advice") or "")


# ── 阶梯从便宜那头爬 ────────────────────────────────────────────────────────

def test_a_hit_on_the_cheapest_rung_stops_there():
    """最便宜那档就拿到了就不许再往上爬 —— 每一档都更贵、对现场破坏更大。"""
    seen, _ = _drive(scripted={"r1": _found()})
    assert "verify" in seen, seen
    assert "r2" not in _skills_of(seen), "第一档就成了却还去动针尖：%s" % seen


def test_the_ladder_is_climbed_from_the_cheap_end_in_order():
    """分级流程默认保留常规处理机会，不因失败直接启用 ForgeAuTip。"""
    seen, _ = _drive(scripted={"r1": _absent(), "r2": {"outcome": "no_tip"},
                               "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    order = [s for s in seen if s in ("r1", "r2", "r3a", "r3a:recheck",
                                      "r3b", "r3b:stabilise", "r3b:recheck")]
    assert order == ["r1", "r2", "r3a", "r3a:recheck",
                     "r3b", "r3b:stabilise", "r3b:recheck"], order


def test_the_poke_is_rechecked_before_reaching_for_a_pulse():
    """浅扎后先复评，再决定是否增加脉冲处理；脉冲后的稳定步骤另有测试。"""
    seen, _ = _drive(scripted={"r1": _absent(), "r2": {"outcome": "no_tip"},
                               "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    assert seen.index("r3a:recheck") < seen.index("r3b"), seen


def test_a_recheck_that_finds_it_stops_the_climb():
    seen, _ = _drive(scripted={"r1": _absent(), "r2": {"outcome": "no_tip"},
                               "r3a:recheck": _found("after_poke.sxm")})
    assert "verify" in seen
    assert "r3b" not in seen, "浅扎后复评已经拿到了，却还去打脉冲：%s" % seen


def test_a_pulse_is_always_followed_by_a_stabilising_poke():
    """验证脉冲之后的稳定步骤到达主流程，再进入复评。"""
    seen, _ = _drive(scripted={"r1": _absent(), "r2": {"outcome": "no_tip"},
                               "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    assert "r3b" in seen, seen
    assert "r3b:stabilise" in seen, "打了脉冲却没有扎针稳定：%s" % seen
    assert seen.index("r3b") < seen.index("r3b:stabilise") < seen.index("r3b:recheck"),         "稳定那一步必须夹在脉冲与复评之间：%s" % seen
    # 浅扎那一档自己不需要再补稳定 —— 它本来就是扎针
    assert "r3a:stabilise" not in seen, "浅扎档不该再补一次扎针：%s" % seen


# ── 最贵那档默认关 ──────────────────────────────────────────────────────────

def test_forge_is_not_attempted_unless_explicitly_allowed():
    seen, ex = _drive(scripted={"r1": _absent(), "r2": {"outcome": "no_tip"},
                                "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    assert "r4" not in seen, seen
    out = AchieveAtomicResolution().aggregate({}, ex.progress)
    assert "ForgeAuTip" in (out.get("advice") or "")


def test_forge_runs_when_the_operator_opens_it():
    seen, _ = _drive(params={"allow_forge": True},
                     scripted={"r1": _absent(), "r2": {"outcome": "no_tip"},
                               "r3a:recheck": _absent(), "r3b:recheck": _absent(),
                               "r4:recheck": _absent()})
    assert "r4" in seen and "r4:recheck" in seen, seen


def test_tip_conditioning_can_be_switched_off_entirely():
    """关掉动针尖之后**一次都不动针尖** —— 但表面那道问句照问。

    问表面是只读的（扫一帧、判一次），它回答的是「该不该换区」，
    与「允不允许动针尖」是两件事。
    """
    seen, _ = _drive(params={"allow_tip_conditioning": False},
                     scripted={"r1": _absent(), "surface": _HAS_REGION})
    assert _skills_of(seen) == ["p0", "r1", "surface"], seen
    for forbidden in ("r2", "r3a", "r3b", "r4"):
        assert forbidden not in _skills_of(seen), seen


# ── 时间不是问题 ────────────────────────────────────────────────────────────

def test_the_default_budget_is_a_whole_night():
    """默认允许长时间编排，预算仅决定是否启动下一档，不把超时当作针尖判据。"""
    md = AchieveAtomicResolution().metadata()
    budget = [p for p in md.parameters if p.name == "total_budget_min"][0]
    assert budget.default >= 480.0, "默认预算 %s 分钟 —— 整夜跑不完" % budget.default
    assert budget.max_value >= 1440.0, "上限装不下一整天"


def test_an_exhausted_budget_says_so_instead_of_blaming_the_tip():
    """预算用完还有没开的档时，报告要说「预算用完了」，不要说「针尖不行」。"""
    seen, ex = _drive(params={"total_budget_min": 5.0},
                      scripted={"r1": _absent()})
    ex.progress.partial_data["rungs"] = list(
        ex.progress.partial_data.get("rungs") or []) + [
        {"rung": "r2", "outcome": "skipped_budget", "at_min": 5.1}]
    out = AchieveAtomicResolution().aggregate({}, ex.progress)
    assert "预算用完了" in (out.get("advice") or ""), out.get("advice")


# ── 拿到了就要验 ────────────────────────────────────────────────────────────

def test_a_hit_is_always_verified():
    """初判过了不等于成立 —— 验证要求正反扫都有晶格、整帧采满、帧内两半不突变。"""
    seen, ex = _drive(scripted={"r1": _found(), "verify": {
        "verdict": "atomic_resolved", "angular_concentration": 240.0}})
    assert "verify" in seen
    out = AchieveAtomicResolution().aggregate({}, ex.progress)
    assert out["achieved"] is True
    assert out["verified"] == "atomic_resolved"
    assert out["angular_concentration"] == pytest.approx(240.0)


def test_a_hit_that_fails_verification_is_not_sold_as_success():
    seen, ex = _drive(scripted={"r1": _found(), "verify": {
        "verdict": "undecidable", "angular_concentration": 12.0}})
    out = AchieveAtomicResolution().aggregate({}, ex.progress)
    assert "验证没过" in (out.get("advice") or ""), out.get("advice")


# ── 路由：阶梯上的每一档都要指回入口 ────────────────────────────────────────
#
# 更重要的问题：如果终端用户不是资深 STM 使用者，只是想让 mast 帮忙扫原子
# 分辨，简单指令（例如「我想要原子分辨」）能否成功？
#
# 造出入口只是一半，另一半是**让 agent 选中它**。而路由信息必须住在 agent 真正
# 读得到的地方 —— **技能目录**，不是提示词。在提示词里说服模型这条路本仓已经
# 走过四次，四次都退回来了（见记忆 remove_incentive_not_persuade）。
#
# 这是一道**结构闸门**：新增一档而忘了指路，这里当场红，而不是等到某天有人
# 发现「简单指令又走岔了」。

def test_every_rung_points_back_at_the_entry_point():
    from mast.skills.composite.make_special_tip import MakeAtomicResolutionTip
    from mast.skills.composite.scan_until_atomic import ScanUntilAtomicResolution

    entry = AchieveAtomicResolution().metadata().description
    assert "我想要原子分辨" in entry, "入口的描述里要有用户会说的那句原话"
    assert "零必填参数" in entry, "入口要明说不需要调用方填数"

    for cls in (ScanUntilAtomicResolution, MakeAtomicResolutionTip):
        md = cls().metadata()
        assert "AchieveAtomicResolution" in md.description, (
            "%s 的目录描述里没有指回入口 —— 用户说「我想要原子分辨」时，"
            "agent 读到的就是这段话，指路要写在这里而不是提示词里" % md.name)
        assert "不是入口" in md.description, (
            "%s 要明说自己只是阶梯里的一档" % md.name)


def test_forge_states_its_indication_is_a_ruined_apex():
    """明确区分顶端重塑、常规修针与表面换区；高成本不能作为更高成功率的证据。"""
    from mast.skills.composite.forge_au_tip import ForgeAuTip

    d = ForgeAuTip().metadata().description
    assert "极其糟糕" in d, "适应症没写清顶端已毁"
    assert "重新锻造" in d, "要说明它是**重造顶端**，不是把现有顶端修一修"
    assert "再给常规修针一次机会" in d, (
        "「常规修针没成功」不是它的适应症 —— 那种情况该再试一次便宜的")
    assert "RelocateCoarseXY" in d, (
        "「这片表面用完了」要指向换区技能，不能落在 forge 头上")
    assert "代价高不等于成功率高" in d
    assert "AchieveAtomicResolution" in d, "最贵那档也要指回入口"


def test_relocate_is_the_answer_to_a_used_up_surface():
    """反证：换区技能自己要认领「换一片新表面」这件事。

    否则上一条只是把责任推给了一个不接的人。
    """
    from mast.skills.composite.relocate_coarse_xy import RelocateCoarseXY

    d = RelocateCoarseXY().metadata().description
    assert "换区专用" in d and "新表面" in d, d[:200]


# ── 第二个分诊分支：表面的问题 vs 针尖的问题 ────────────────────────────────
#
# 要求：「这片表面用完了」应由粗动换区处理 ——`RelocateCoarseXY`，
# 而且它自述「换区专用……唯一应该
# 自主使用的换区方式」。接进来时**必须**由表面判据触发，不能由「别的都失败了」
# 触发 —— 后者正是同一天被纠正过的那个错。
#
# 触发源不是我发明的：`flat_region.py:688` 自己写着
#     verdict == "no_usable_region" ⇒「我找过了，这里没有」⇒ 换一块再扫
#     别的失败                       ⇒「我没法找」⇒ 判不了
# 那句「换一块再扫」一直没有调用方在这一层接它。

_NO_REGION = {"verdict": "no_usable_region"}
_HAS_REGION = {"verdict": "ok", "spots": [{"x_m": 0.0, "y_m": 0.0}]}
_SUGGESTION = {"axis": "x", "direction": "+", "steps": 300,
               "reason": "往 +x 还有没去过的地方"}


def _patch_suggestion(monkeypatch, sug=_SUGGESTION, note="ok"):
    from mast.skills.composite import achieve_atomic as A
    monkeypatch.setattr(A, "_coarse_suggestion", lambda: (sug, note))


def test_a_used_up_surface_relocates_instead_of_conditioning_the_tip(monkeypatch):
    """表面判据说「没有可用区域」⇒ 换区，**不要**去修针。

    这是此前修正过的那条的正面版本：治病要看**失效模式**。
    表面用完了却去修针，和「常规修针没成功就上 forge」是同一种错。
    """
    _patch_suggestion(monkeypatch)
    seen, ex = _drive(scripted={"r1": _absent(), "surface": _NO_REGION,
                                "r1#2": _absent(), "surface#2": _HAS_REGION,
                                "r2#2": {"outcome": "no_tip"},
                                "r3a#2:recheck": _absent(),
                                "r3b#2:recheck": _absent()})
    assert "relocate" in seen, "表面用完了却没换区：%s" % seen
    assert seen.index("relocate") < seen.index("r2#2"), (
        "换区必须发生在动针尖之前 —— 否则就是在治错的病：%s" % seen)
    assert "r2" not in seen, "第一个站点上不该动针尖（表面判据已经说没有台面）"


def test_a_usable_surface_climbs_the_tip_ladder_instead(monkeypatch):
    """反证：表面还有台面 ⇒ 这是**针尖**的问题 ⇒ 照常爬阶梯，不许换区。

    没有这一条，上面那条可以靠「永远换区」通过。
    """
    _patch_suggestion(monkeypatch)
    seen, _ = _drive(scripted={"r1": _absent(), "surface": _HAS_REGION,
                               "r2": {"outcome": "no_tip"},
                               "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    assert "relocate" not in seen, "表面还能用却换了区：%s" % seen
    assert "r2" in seen, seen


def test_undecidable_frames_never_trigger_a_relocation(monkeypatch):
    """判不了 ⇒ 既不动针尖，**也不换区**。

    粗动是本流程里影响面最大的动作（新坐标代次 + 撞针风险）。拿「读不到」
    去触发它，和拿「读不到」去修针是同一种错，只是更贵。
    """
    _patch_suggestion(monkeypatch)
    seen, _ = _drive(scripted={"r1": _undetermined(3)})
    assert "relocate" not in seen and "surface" not in seen, seen


def test_relocation_can_be_switched_off(monkeypatch):
    _patch_suggestion(monkeypatch)
    seen, _ = _drive(params={"allow_relocate": False},
                     scripted={"r1": _absent(), "surface": _NO_REGION,
                               "r2": {"outcome": "no_tip"},
                               "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    assert "relocate" not in seen and "surface" not in seen, seen


def test_no_next_site_is_reported_honestly_not_blamed_on_the_tip(monkeypatch):
    """粗动地图说没有下一站 ⇒ 如实停下，**不要**接着去修针。

    ``suggestion is None`` 的含义是「这一带已经用完」，通常意味着要换样品 ——
    那不是这个流程能解决的，硬爬阶梯只会白白扎针。
    """
    _patch_suggestion(monkeypatch, sug=None, note="这一带已经用完，建议换样品")
    seen, ex = _drive(scripted={"r1": _absent(), "surface": _NO_REGION})
    assert "relocate" not in seen, seen
    assert "r2" not in seen, "没有下一站却还去修针：%s" % seen
    out = AchieveAtomicResolution().aggregate({}, ex.progress)
    assert "换样品" in (out.get("advice") or ""), out.get("advice")


def test_the_move_is_planned_by_the_map_not_by_this_skill(monkeypatch):
    """axis/direction/steps **原样**来自粗动地图的 suggestion。

    这个技能一个数都不许拍：粗动步长随驱动幅度/负载/温度漂移，
    低温下同样步数走的距离可以差几倍 —— **绝不要用步数去算米**。
    """
    _patch_suggestion(monkeypatch)
    steps = []

    from mast.skills.composite.graph_executor import CompositeProgress

    class _Ex:
        def __init__(self):
            self.progress = CompositeProgress("AchieveAtomicResolution")
            self.sub_results = {}

        def set_partial(self, k, v):
            self.progress.partial_data[k] = v

        def set_total_steps(self, n):
            pass

    scripted = {"p0:zctrl": {"controller_on": True}, "r1": _absent(),
                "surface": _NO_REGION, "r1#2": _absent(),
                "surface#2": _HAS_REGION, "r2#2": {"outcome": "no_tip"},
                "r3a#2:recheck": _absent(), "r3b#2:recheck": _absent()}
    ex = _Ex()
    for st in AchieveAtomicResolution().plan_dynamic({}, ex):
        steps.append(st)
        ex.sub_results[st.step_id] = _Res(scripted.get(st.step_id, {}))
    mv = [s for s in steps if s.skill_name == "RelocateCoarseXY"]
    assert len(mv) == 1, [s.step_id for s in steps]
    assert mv[0].params["axis"] == "x"
    assert mv[0].params["direction"] == "+"
    assert mv[0].params["steps"] == 300
