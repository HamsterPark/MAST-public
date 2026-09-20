"""判据③(跨点聚合)的闭集三值、转移矩阵、epoch 作废与被否方案的钉子。

设计:``docs/v2/design/synthetic_sample_s1_tip_cycle_design.md`` D4 / §3.4 / §5.2 / §5.3。

跑:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/conduct/test_cross_check.py -q

## 转移矩阵为什么要**手写**一张期望表

用「judged 全票判坏 ⇒ bad_tip」这条规则去算期望值,等于拿实现去测实现 ——
两边一起错的时候全绿。所以 :data:`EXPECTED` 是按 (判坏, 判好, 判不了) 三个计数
逐格写下来的十行,N=3 的全部可能;测试再把 27 种排列都跑一遍(顺序不该影响结论)。
"""

from __future__ import annotations

import itertools
import json

import pytest

from mast.conduct import cross_check as cc
from mast.conduct.cross_check import (
    CROSS_VERDICTS,
    PointVerdict,
    aggregate_cross_points,
)

EPOCH = 7


def bad(label="", **kw) -> PointVerdict:
    return PointVerdict(coord_epoch=EPOCH, tip_ready=False, label=label, **kw)


def good(label="", **kw) -> PointVerdict:
    return PointVerdict(coord_epoch=EPOCH, tip_ready=True,
                        corrugation_verdict="normal", label=label, **kw)


def und(label="", **kw) -> PointVerdict:
    return PointVerdict(coord_epoch=EPOCH, tip_ready=None, label=label,
                        abstain_reason="行数不足", **kw)


#: (判坏, 判好, 判不了) → 结论。**手写的**,N=3 的全部十种计数。
EXPECTED = {
    (3, 0, 0): "bad_tip",
    (2, 0, 1): "bad_tip",
    (2, 1, 0): "surface_feature",
    (1, 2, 0): "surface_feature",
    (1, 1, 1): "surface_feature",
    (0, 3, 0): "surface_feature",
    (0, 2, 1): "surface_feature",
    (1, 0, 2): "undecidable",   # judged=1 —— 一个点不构成「换位置复测」
    (0, 1, 2): "undecidable",
    (0, 0, 3): "undecidable",
}


# ═══════════════════════════════════════════════════════════════════════
# 1. 转移矩阵:N=3 全组合逐格
# ═══════════════════════════════════════════════════════════════════════

def _assert_transition_matrix():
    makers = {"bad": bad, "good": good, "und": und}
    for combo in itertools.product(makers, repeat=3):
        pts = [makers[k](label=f"P{i + 1}") for i, k in enumerate(combo)]
        counts = (combo.count("bad"), combo.count("good"), combo.count("und"))
        res = aggregate_cross_points(pts)
        assert res.verdict == EXPECTED[counts], (
            f"{combo} → {res.verdict},期望 {EXPECTED[counts]};{res.reason}")
        assert res.n_bad == counts[0] and res.n_good == counts[1]
        assert res.n_judged == counts[0] + counts[1]
        assert res.n_points == 3


def test_transition_matrix_n3():
    _assert_transition_matrix()


@pytest.mark.parametrize("pts,want", [
    ([bad(), bad(), und()], "bad_tip"),
    ([bad(), good(), und()], "surface_feature"),
    ([bad(), und(), und()], "undecidable"),
    ([und(), und(), und()], "undecidable"),
])
def test_the_four_cases_the_design_pins_by_name(pts, want):
    assert aggregate_cross_points(pts).verdict == want


def test_all_good_is_surface_feature_not_a_pass_for_the_tip():
    """稳定的合成针尖序列在连续多个点上都不应误判为 bad_tip。"""
    res = aggregate_cross_points([good(), good(), good()])
    assert res.verdict == "surface_feature" and res.n_bad == 0


def test_two_points_are_enough_and_one_is_not():
    assert aggregate_cross_points([bad(), bad()]).verdict == "bad_tip"
    assert aggregate_cross_points([bad(), und()]).verdict == "undecidable"
    assert aggregate_cross_points([bad()]).verdict == "undecidable"
    assert aggregate_cross_points([]).verdict == "undecidable"


def test_min_judged_below_two_is_refused():
    """``min_judged=1`` 会让 ``bad_tip`` 退化成单帧结论 —— 拒绝,不夹紧。"""
    with pytest.raises(ValueError) as exc:
        aggregate_cross_points([bad(), bad()], min_judged=1)
    assert "换位置" in str(exc.value)
    # 更严是允许的
    assert aggregate_cross_points(
        [bad(), bad()], min_judged=3).verdict == "undecidable"


# ═══════════════════════════════════════════════════════════════════════
# 2. 一个点算判坏 / 判好 / 不计
# ═══════════════════════════════════════════════════════════════════════

def test_corrugation_high_counts_as_bad_but_never_alone():
    """§5.3:判据②**不单独下结论**。一个点的 ``high`` 顶多让这个点算判坏。"""
    hi = PointVerdict(coord_epoch=EPOCH, tip_ready=None,
                      corrugation_verdict="high", label="P1")
    assert aggregate_cross_points([hi]).verdict == "undecidable"
    assert aggregate_cross_points([hi, und()]).verdict == "undecidable"
    assert aggregate_cross_points([hi, bad()]).verdict == "bad_tip"


def test_tip_ready_none_is_not_folded_into_either_side():
    """判据①判不了的点既不投坏票也不投好票。"""
    res = aggregate_cross_points([und(), und(), good()])
    assert res.n_bad == 0 and res.n_good == 1 and res.n_judged == 1
    assert res.verdict == "undecidable"


def test_a_point_that_is_ready_but_high_is_bad_not_good():
    p = PointVerdict(coord_epoch=EPOCH, tip_ready=True,
                     corrugation_verdict="high")
    assert aggregate_cross_points([p, bad()]).verdict == "bad_tip"


def test_an_unrecognised_corrugation_word_does_not_buy_a_good_vote():
    """不认的 verdict 字符串 = 读不到,不是「不是 high 所以算好」。"""
    p = PointVerdict(coord_epoch=EPOCH, tip_ready=True,
                     corrugation_verdict="bad_tip")   # 词表里没有这个值
    res = aggregate_cross_points([p, good()])
    assert res.n_good == 1 and res.verdict == "undecidable"


# ── 判据④:不对称使用 ──────────────────────────────────────────────────

def test_spectrum_fired_counts_as_bad_even_with_no_frame():
    """谱触发 ⇒ 已拿到针尖不稳的证据,该点计判坏并省掉那张 5 min 的帧。"""
    fired = PointVerdict(coord_epoch=EPOCH, spectrum_hysteresis="fired",
                         frame_skipped_by_prescreen=True, similarity=None,
                         label="P1")
    assert aggregate_cross_points([fired, bad()]).verdict == "bad_tip"


def _assert_not_fired_is_not_evidence_of_good():
    pts = [PointVerdict(coord_epoch=EPOCH, spectrum_hysteresis="not_fired",
                        label=f"P{i}") for i in range(3)]
    res = aggregate_cross_points(pts)
    assert res.verdict == "undecidable", (
        "「谱没触发」被当成了「这个点好」—— 那是把「没看见」当成「没有」")
    assert res.n_good == 0


def test_spectrum_not_fired_is_not_evidence_of_good():
    _assert_not_fired_is_not_evidence_of_good()


@pytest.mark.parametrize("state", ["unrated", "skipped", "not_fired"])
def test_only_fired_moves_anything(state):
    p = PointVerdict(coord_epoch=EPOCH, tip_ready=True,
                     corrugation_verdict="normal", spectrum_hysteresis=state)
    assert aggregate_cross_points([p, good()]).verdict == "surface_feature"


# ═══════════════════════════════════════════════════════════════════════
# 3. coord_epoch
# ═══════════════════════════════════════════════════════════════════════

def _assert_epoch_mismatch_voids_the_batch():
    # **经模块属性**调用 —— 变异验证要 monkeypatch 得进来,而断言体只能有一份。
    pts = [bad("P1"), PointVerdict(coord_epoch=EPOCH + 1, tip_ready=False,
                                   label="P2"), bad("P3")]
    res = cc.aggregate_cross_points(pts)
    assert res.verdict == "undecidable", "中途粗动了,这批点不再是同一片区域"
    assert res.coord_epoch is None
    assert "P2" in res.reason and str(EPOCH + 1) in res.reason, (
        "整批作废却没点名是哪一点 —— 人没法处置")


def test_epoch_mismatch_voids_the_batch():
    _assert_epoch_mismatch_voids_the_batch()


def test_unknown_epoch_is_not_a_match():
    """代次读不到与不一致同样作废 —— 两个未知不构成「一致」。"""
    res = aggregate_cross_points([bad("P1"), PointVerdict(tip_ready=False,
                                                          label="P2"), bad("P3")])
    assert res.verdict == "undecidable" and res.coord_epoch is None
    assert "P2" in res.reason and "读不到" in res.reason


def test_a_consistent_epoch_is_carried_out():
    assert aggregate_cross_points([bad(), bad()]).coord_epoch == EPOCH


# ═══════════════════════════════════════════════════════════════════════
# 4. map_known / 记账 / 词表
# ═══════════════════════════════════════════════════════════════════════

def test_map_known_false_propagates():
    """地图读不到与「表面干净」几何上不可区分 —— 必须一路传到报告。"""
    pts = [bad("P1"), bad("P2", map_known=False)]
    res = aggregate_cross_points(pts)
    assert res.map_known is False and res.verdict == "bad_tip"
    assert aggregate_cross_points([bad(), bad()]).map_known is True


def test_verdict_vocabulary_is_closed():
    assert set(CROSS_VERDICTS) == {"bad_tip", "surface_feature", "undecidable"}
    makers = (bad, good, und)
    for combo in itertools.product(makers, repeat=3):
        assert aggregate_cross_points([m() for m in combo]).verdict in CROSS_VERDICTS


def test_reason_names_the_points_and_what_each_said():
    res = aggregate_cross_points([bad("P1"), good("P2"), und("P3")])
    for name in ("P1", "P2", "P3"):
        assert name in res.reason
    assert "判据①不合格" in res.reason and "行数不足" in res.reason


def test_points_survive_into_the_payload_as_json():
    res = aggregate_cross_points([bad("P1", x_m=1e-8, y_m=-2e-8), good("P2")])
    payload = res.as_dict()
    assert len(payload["points"]) == 2
    assert payload["cross_verdict"] if "cross_verdict" in payload else True
    json.dumps(payload)


# ═══════════════════════════════════════════════════════════════════════
# 5. conduct 分析注册表
# ═══════════════════════════════════════════════════════════════════════

def test_registered_as_an_analysis_step():
    from mast.conduct import analyses

    assert "aggregate_cross_points" in analyses.known_names()
    fn = analyses.get("aggregate_cross_points")
    out = fn({"points": [
        {"coord_epoch": EPOCH, "tip_ready": False, "label": "P1"},
        {"coord_epoch": EPOCH, "tip_ready": False, "label": "P2"},
    ]})
    assert out["cross_verdict"] == "bad_tip" and out["verdict"] == "bad_tip"
    assert out["n_judged"] == 2
    json.dumps(out)


def test_analysis_step_accepts_json_text_too():
    from mast.conduct import analyses

    fn = analyses.get("aggregate_cross_points")
    out = fn({"points": json.dumps([
        {"coord_epoch": EPOCH, "tip_ready": False},
        {"coord_epoch": EPOCH, "tip_ready": True, "corrugation_verdict": "normal"},
    ])})
    assert out["cross_verdict"] == "surface_feature"


def test_analysis_step_refuses_to_invent_points():
    from mast.conduct import analyses
    from mast.conduct.analyses import AnalysisError

    fn = analyses.get("aggregate_cross_points")
    with pytest.raises(AnalysisError):
        fn({})
    with pytest.raises(AnalysisError):
        fn({"points": [{"coord_epoch": "第七代"}]})


@pytest.mark.parametrize("bad_point", [
    {"tip_ready": "false"},                       # 字符串会静默变成「判不了」
    {"frame_usable": "yes"},
    {"corrugation_verdict": "bad_tip"},           # 判据②不产出这个词
    {"corrugation_verdict": "High"},
    {"spectrum_hysteresis": "maybe"},
    {"map_known": "true"},
])
def test_analysis_step_refuses_a_miswired_point(bad_point):
    """纯函数那一侧对不认的值保守;**边界要吵** —— 保守加沉默 = 查不出来的接错线。"""
    from mast.conduct import analyses
    from mast.conduct.analyses import AnalysisError

    fn = analyses.get("aggregate_cross_points")
    with pytest.raises(AnalysisError):
        fn({"points": [{"coord_epoch": EPOCH, **bad_point},
                       {"coord_epoch": EPOCH, "tip_ready": False}]})


def test_analysis_step_does_not_fill_in_a_missing_tip_ready():
    """缺席的 ``tip_ready`` 是 ``None``(判不了),绝不当成 True/False。"""
    from mast.conduct import analyses

    fn = analyses.get("aggregate_cross_points")
    out = fn({"points": [{"coord_epoch": EPOCH}, {"coord_epoch": EPOCH}]})
    assert out["cross_verdict"] == "undecidable" and out["n_judged"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 6. 变异验证
# ═══════════════════════════════════════════════════════════════════════

def test_mutation_counting_undecidable_as_judged_turns_the_matrix_red(monkeypatch):
    """把「判不了」算进 judged ⇒ 「1 坏 + 2 判不了」会变成 bad_tip。"""
    monkeypatch.setattr(cc, "_is_good", lambda p: not cc._is_bad(p))
    assert cc._is_good(und()) is True                      # 变异已应用
    assert aggregate_cross_points([bad(), und(), und()]).verdict != "undecidable"
    with pytest.raises(AssertionError):
        _assert_transition_matrix()


def test_mutation_letting_not_fired_vote_good_turns_its_test_red(monkeypatch):
    orig = cc._is_good
    monkeypatch.setattr(
        cc, "_is_good",
        lambda p: orig(p) or p.spectrum_hysteresis == "not_fired")
    probe = PointVerdict(coord_epoch=EPOCH, spectrum_hysteresis="not_fired")
    assert cc._is_good(probe) is True                      # 变异已应用
    with pytest.raises(AssertionError):
        _assert_not_fired_is_not_evidence_of_good()


def test_mutation_ignoring_the_epoch_turns_its_test_red(monkeypatch):
    """把 epoch 检查拿掉 ⇒ 混着代次的一批点会给出一个结论。"""
    orig = aggregate_cross_points

    def mutant(points, **kw):
        return orig([p if p.coord_epoch is not None else p
                     for p in [type(p)(**{**p.__dict__, "coord_epoch": EPOCH})
                               for p in points]], **kw)

    monkeypatch.setattr(cc, "aggregate_cross_points", mutant)
    mixed = [bad("P1"), PointVerdict(coord_epoch=EPOCH + 1, tip_ready=False,
                                     label="P2"), bad("P3")]
    assert cc.aggregate_cross_points(mixed).verdict == "bad_tip"   # 变异已应用
    with pytest.raises(AssertionError):
        _assert_epoch_mismatch_voids_the_batch()
