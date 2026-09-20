"""STSConditionSeries —— S4 STS 设计 D18/D19/D20、陷阱 6。

同一个位置、一串条件组、逐条取谱。这一层最容易**静默**出错的四件事:

1. 结果列表叫 ``points`` —— 记录层会在**同一个 xy** 上叠 N 个标记(陷阱 6),
   而针尖只去过一个地方。
2. 一个坏组名废掉整条序列 —— 或者反过来,被记成「跑了但失败」,把人送去查仪器,
   而真正要查的是那张表。
3. 逐条共用一个 ``run_tag`` —— 第 2 条的 .dat 归属会拿第 1 条的文件当自己的
   (文件名那一层认不出来),两份数据长得一模一样。
4. 穿零那一步没被指出来 —— 事后发现针尖变了,没人知道该看哪一步。

跑:
    .venv-v2-py313/Scripts/python.exe -m pytest \
      tests/v2/unit/skills/composite/test_sts_condition_series.py -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import json  # noqa: E402

import pytest  # noqa: E402

from mast.core.sts_workflow import STSConditionSpec  # noqa: E402
from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite.sts_condition_series import (  # noqa: E402
    MAX_CONDITIONS,
    POINT_ENGINE_SKILL,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_REFUSED,
    STATUS_SKIPPED,
    STSConditionSeries,
    parse_conditions,
)

X_M, Y_M, EPOCH = 2e-8, -1e-8, 5

#: 三个条件组:一条负偏压综览、一条正偏压综览(⇒ 中间穿零)、一条精细谱。
GROUPS = {
    "wide_neg": dict(stab_bias_v=-1.0, stab_setpoint_a=100e-12,
                     start_v=-1.5, end_v=1.5, mod_amp_v=0.015),
    "wide_pos": dict(stab_bias_v=1.0, stab_setpoint_a=100e-12,
                     start_v=-1.5, end_v=1.5, mod_amp_v=0.015),
    "fine": dict(stab_bias_v=-0.1, stab_setpoint_a=500e-12,
                 start_v=-0.1, end_v=0.1, mod_amp_v=0.001),
}


@pytest.fixture(autouse=True)
def _table(monkeypatch):
    """把三个标定过的组挂进**真表** —— 走的是真的组名解析,不是替身。"""
    from mast.core import sts_workflow as W

    for name, vals in GROUPS.items():
        monkeypatch.setitem(W.CONDITIONS, name,
                            W.STSConditionSpec(label=name, **vals))
    return GROUPS


class FakeCtx:
    """只回答 ``SpectroscopyAtPositions``。别的技能名直接抛 —— 这个流程按设计
    **不自己采谱**:归属那三层只该有一份实现,抄第二份的那天只有一份是对的。"""

    def __init__(self, verdicts=None, fail_at=()):
        self.calls: list[tuple[str, dict]] = []
        self.run_id = "test-series"
        self.state = None
        self._verdicts = dict(verdicts or {})
        self._fail = set(fail_at)
        self._n = 0

    def run(self, skill_name, params, version=None):
        self.calls.append((skill_name, dict(params)))
        if skill_name != POINT_ENGINE_SKILL:
            raise AssertionError(
                f"这个流程只该按名字调 {POINT_ENGINE_SKILL}，却调了 {skill_name!r} —— "
                f".dat 归属那三层只该有一份实现")
        i = self._n
        self._n += 1
        if i in self._fail:
            return SkillResult(skill_name=skill_name, success=False,
                               error="引擎失败")
        verdict = self._verdicts.get(i, "keep")
        return SkillResult(
            skill_name=skill_name, success=True,
            data={"points": [{"index": 1, "success": True, "status": "ok",
                              "verdict": verdict,
                              "path": f"/tmp/{params['run_tag']}_00001.dat"}],
                  "lockin_readback": {"amplitude": 0.015}})

    def check_abort(self):
        return False

    def check_halt(self):
        return ""

    def engine_params(self):
        return [p for s, p in self.calls if s == POINT_ENGINE_SKILL]


def run_series(ctx=None, conditions="wide_neg,wide_pos,fine", **kw):
    ctx = ctx if ctx is not None else FakeCtx()
    p = {"conditions": conditions, "x_m": X_M, "y_m": Y_M,
         "expected_coord_epoch": EPOCH}
    p.update(kw)
    return ctx, STSConditionSeries().execute(ctx, p)


# ═══════════════════════════════════════════════════════════════════════
# 1. 结果的键叫 spectra,不叫 points(设计陷阱 6)
# ═══════════════════════════════════════════════════════════════════════

def _assert_one_position_one_marker(data):
    from mast.core.runtime import _marker_subrecords

    assert "points" not in data, (
        "结果里出现了 points —— 记录层会给列表里每一项画一个地图标记，"
        "而这 N 条谱在**同一个 xy** 上：针尖只去过一个地方")
    assert _marker_subrecords(data) == [], (
        "逐点定位记录那条准入被触发了 —— 一个位置上会叠 N 个标记")


def test_the_result_list_is_called_spectra_so_one_position_gets_one_marker():
    """陷阱 6：叫 ``points`` 会在一个 xy 上叠 N 个标记，叫 ``spectra`` 走单标记路径。

    这条测试很像在钉一个字段名，而它挡住的是一张被 N 个重叠标记糊住的地图 ——
    找干净地方的路径靠地图避开用过的位置。
    """
    _, res = run_series()
    assert len(res.data["spectra"]) == 3
    _assert_one_position_one_marker(res.data)


def test_every_spectrum_is_taken_at_the_one_position():
    """序列的全部意义就是「同一片表面、不同条件」——换点等于换了被测对象。"""
    ctx, _ = run_series()
    for p in ctx.engine_params():
        pos = json.loads(p["positions"])
        assert len(pos) == 1
        assert (pos[0]["x_m"], pos[0]["y_m"]) == (X_M, Y_M)


# ═══════════════════════════════════════════════════════════════════════
# 2. 一条坏组名不拖垮其余的
# ═══════════════════════════════════════════════════════════════════════

def _assert_bad_name_is_isolated(res, ctx):
    recs = res.data["spectra"]
    assert [r["status"] for r in recs] == [STATUS_OK, STATUS_REFUSED, STATUS_OK]
    assert res.data["n_acquired"] == 2 and res.data["n_refused"] == 1
    assert len(ctx.engine_params()) == 2, "坏组名那一条还是发出去了"
    assert "nope" in (recs[1]["error"] or "")


def test_one_unusable_condition_does_not_take_down_the_series():
    """坏组名那一条**不发出去**，其余照跑。

    把它记成「跑了但失败」会把人送去查仪器，而真正要查的是那张表 —— 所以
    ``refused`` 与 ``failed`` 是两个状态，不是一个。
    """
    ctx, res = run_series(conditions="wide_neg,nope,fine")
    assert res.success is True
    _assert_bad_name_is_isolated(res, ctx)
    assert res.data["dat_paths"] and len(res.data["dat_paths"]) == 2


def test_an_uncalibrated_group_is_refused_per_entry_naming_the_gaps(monkeypatch):
    from mast.core import sts_workflow as W

    monkeypatch.setitem(W.CONDITIONS, "raw", STSConditionSpec(label="raw"))
    _, res = run_series(conditions="wide_neg,raw")
    bad = res.data["spectra"][1]
    assert bad["status"] == STATUS_REFUSED
    assert "stab_bias_v" in bad["error"] and "start_v" in bad["error"]
    assert "没发出去" in (res.summary or "")


def test_zero_usable_conditions_is_a_hard_failure():
    """一条都没采到 ⇒ 失败。报成功会让上层拿着一个空产物往下走。"""
    _, res = run_series(conditions="nope1,nope2")
    assert res.success is False
    assert "0/2" in (res.error or "")


# ═══════════════════════════════════════════════════════════════════════
# 3. 逐条一个 run_tag(否则第 2 条会拿第 1 条的文件)
# ═══════════════════════════════════════════════════════════════════════

def _assert_tags_are_distinct(ctx):
    tags = [p["run_tag"] for p in ctx.engine_params()]
    assert len(set(tags)) == len(tags), (
        f"逐条共用了 run_tag {tags} —— 第 2 条的归属校验会拿第 1 条的文件当自己的，"
        f"文件名那一层认不出来，而两份数据长得一模一样")


def test_each_condition_gets_its_own_run_tag():
    ctx, _ = run_series()
    _assert_tags_are_distinct(ctx)


def test_an_operator_run_tag_is_still_per_condition():
    ctx, _ = run_series(run_tag="night")
    _assert_tags_are_distinct(ctx)
    assert all(p["run_tag"].startswith("night") for p in ctx.engine_params())


# ═══════════════════════════════════════════════════════════════════════
# 4. 穿零要指出来(D20)
# ═══════════════════════════════════════════════════════════════════════

def _assert_zero_crossing_is_flagged(res):
    recs = res.data["spectra"]
    assert [r["crosses_zero"] for r in recs] == [False, True, True], (
        "穿零那一步没被指出来 —— 恒流反馈下偏压穿零会把针尖推进样品，"
        "事后发现针尖变了，没人知道该看哪一步")
    assert res.data["n_zero_crossings"] == 2
    assert "穿零" in (res.summary or "")


def test_a_stabilisation_bias_crossing_zero_is_reported():
    """−1 V 综览 → +1 V 综览 → −0.1 V 精细：两次穿零。"""
    _, res = run_series()
    _assert_zero_crossing_is_flagged(res)


def test_the_first_condition_never_counts_as_a_crossing():
    """第一条没有「上一条」——从仪器当前偏压过来那一步不归这个流程管。"""
    _, res = run_series(conditions="wide_pos")
    assert res.data["spectra"][0]["crosses_zero"] is False
    assert res.data["n_zero_crossings"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 5. 早停预算
# ═══════════════════════════════════════════════════════════════════════

def test_consecutive_discards_stop_the_series():
    ctx, res = run_series(verdicts_ctx := FakeCtx(verdicts={0: "discard",
                                                           1: "discard"}),
                          stop_after_consecutive_discard=2)
    assert res.data["stopped_early"] is True
    assert len(verdicts_ctx.engine_params()) == 2, "停了还接着采"
    assert res.data["spectra"][2]["status"] == STATUS_SKIPPED


def test_unrated_does_not_count_towards_the_stop_budget():
    """「判不了」不是坏谱 —— 它对「针尖在变坏吗」这个问题一个字都没说。"""
    ctx, res = run_series(FakeCtx(verdicts={0: "unrated", 1: "unrated",
                                            2: "unrated"}),
                          stop_after_consecutive_discard=2)
    assert res.data["stopped_early"] is False
    assert len(ctx.engine_params()) == 3


def test_stop_after_zero_disables_the_budget():
    ctx, res = run_series(FakeCtx(verdicts={i: "discard" for i in range(3)}),
                          stop_after_consecutive_discard=0)
    assert res.data["stopped_early"] is False
    assert len(ctx.engine_params()) == 3


# ═══════════════════════════════════════════════════════════════════════
# 6. 输入解析
# ═══════════════════════════════════════════════════════════════════════

def test_both_comma_and_json_forms_parse():
    assert parse_conditions("a, b") == (["a", "b"], "")
    assert parse_conditions('["a","b"]') == (["a", "b"], "")


def test_an_empty_list_is_refused_not_defaulted():
    """空**不当成「用默认」** —— 那会让一次打错的输入变成一条谁也没要的谱。"""
    names, err = parse_conditions("")
    assert names == [] and "空" in err
    _, res = run_series(conditions="")
    assert res.success is False


def test_more_than_the_budget_is_refused():
    names, err = parse_conditions(",".join(["a"] * (MAX_CONDITIONS + 1)))
    assert names == [] and str(MAX_CONDITIONS) in err
    assert "预算" in err


def test_a_failed_engine_call_is_failed_not_refused():
    """发出去了没成 ≠ 根本没发出去。两者的下一步完全不同。"""
    ctx, res = run_series(FakeCtx(fail_at={1}))
    recs = res.data["spectra"]
    assert [r["status"] for r in recs] == [STATUS_OK, STATUS_FAILED, STATUS_OK]
    assert res.data["n_failed"] == 1 and res.data["n_refused"] == 0
    assert len(res.data["dat_paths"]) == 2, "失败那条的路径漏进了产物"


def test_partial_success_says_the_gap_in_the_summary():
    """缺口必须进 summary —— 躺在 data 里没人看（本仓「假成功」那一族）。"""
    _, res = run_series(conditions="wide_neg,nope,fine")
    assert res.success is True
    assert "缺口" in (res.summary or "") and "3 条里采到 2 条" in res.summary


# ═══════════════════════════════════════════════════════════════════════
# 7. 变异验证 —— 先证明变异已应用，再证明测试红了
# ═══════════════════════════════════════════════════════════════════════

def test_mutation_renaming_spectra_to_points_turns_its_test_red():
    """把结果的键改回 ``points`` —— 也就是陷阱 6 描述的那个行为。"""
    orig = STSConditionSeries.aggregate

    def _as_points(self, sub_results, progress):
        data = orig(self, sub_results, progress)
        data["points"] = data.pop("spectra")
        return data

    STSConditionSeries.aggregate = _as_points
    try:
        _, res = run_series()
        assert "points" in res.data                      # 变异已应用
        with pytest.raises(AssertionError):
            _assert_one_position_one_marker(res.data)
    finally:
        STSConditionSeries.aggregate = orig


def test_mutation_sharing_one_run_tag_turns_its_test_red():
    """逐条共用一个 tag —— 第 2 条会拿第 1 条的文件当自己的证据。"""
    orig = STSConditionSeries.plan_dynamic

    def _shared(self, params, executor):
        for step in orig(self, params, executor):
            step.params["run_tag"] = "same"
            yield step

    STSConditionSeries.plan_dynamic = _shared
    try:
        ctx, _ = run_series()
        assert {p["run_tag"] for p in ctx.engine_params()} == {"same"}  # 变异已应用
        with pytest.raises(AssertionError):
            _assert_tags_are_distinct(ctx)
    finally:
        STSConditionSeries.plan_dynamic = orig


def test_mutation_treating_a_bad_group_as_a_run_failure_turns_its_test_red():
    """把「组名坏了」记成「跑了但失败」—— 把人送去查仪器，而该查的是那张表。

    变异打在 ``aggregate`` 上而不是那个常量上：改常量是自相抵消的（写状态和数状态
    读的是同一个名字，两边一起变，结果一模一样）。这本身是一条小教训——**变异必须
    先证明自己动了手**，而「改一个两边共用的常量」经常什么都没动。
    """
    orig = STSConditionSeries.aggregate

    def _refused_as_failed(self, sub_results, progress):
        for rec in self._records:
            if rec["status"] == STATUS_REFUSED:
                rec["status"] = STATUS_FAILED
        return orig(self, sub_results, progress)

    STSConditionSeries.aggregate = _refused_as_failed
    try:
        ctx, res = run_series(conditions="wide_neg,nope,fine")
        assert res.data["n_refused"] == 0 and res.data["n_failed"] == 1  # 变异已应用
        with pytest.raises(AssertionError):
            _assert_bad_name_is_isolated(res, ctx)
    finally:
        STSConditionSeries.aggregate = orig


def test_mutation_dropping_the_zero_crossing_flag_turns_its_test_red():
    """不再指出穿零 —— 事后发现针尖变了，没人知道该看哪一步。"""
    from mast.skills.composite import sts_condition_series as M

    orig = M.crosses_zero
    M.crosses_zero = lambda a, b: False
    try:
        _, res = run_series()
        assert res.data["n_zero_crossings"] == 0         # 变异已应用
        with pytest.raises(AssertionError):
            _assert_zero_crossing_is_flagged(res)
    finally:
        M.crosses_zero = orig
