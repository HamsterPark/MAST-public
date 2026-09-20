"""``CrossPointTipCheck`` —— 判据③的执行体(采点 + 只读复测 + 聚合)。

S1 修针循环设计 D4 / D5 / §5.2。这里盯的**不是**聚合规则本身(那是
``tests/v2/unit/campaign/test_cross_check.py`` 的活),而是这一层最容易悄悄坏掉的
四件事:

1. **采点的两条后置筛真的在筛**(两两不重叠、不在自己炸过的坑上);
2. **代次每点各记一次**,中途变了整批作废并**点名**是哪一点;
3. **候选不够时不凑数** —— 而且不拿一个短批次去下 ``bad_tip``;
4. **只读** —— 不记撞针、不写地图标记。``surface_feature`` 的点一旦上了避让地图,
   后面找畴界的搜索就会躲开自己的目标。

⚠️ 假上下文用**显式哨兵**区分「没配置这个技能」与「配置成失败」:前者当场报错,
后者返回一个失败的 ``SkillResult``。把两者都写成 ``None`` 会让一个本该失败的步骤
悄悄变成成功 —— 脚手架自己的那条教训。
"""
from __future__ import annotations

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

import pytest  # noqa: E402

from mast.core.types import SafetyLevel, SkillCategory, SkillResult  # noqa: E402
from mast.skills.composite import cross_point_tip_check as CPT  # noqa: E402
from mast.skills.composite.cross_point_tip_check import (  # noqa: E402
    ALLOWED_SUBSKILLS,
    DEFAULT_N_POINTS,
    MIN_SEPARATION_FRAMES,
    REJECT_CRASH_HISTORY,
    REJECT_TOO_CLOSE,
    CrossPointTipCheck,
    pick_candidate,
)

#: 「这个技能配置成失败」的哨兵。**与「没配置」不是同一件事** —— 见模块注释。
FAIL = object()

SCAN_NM = 100.0
NM = 1e-9
#: 过筛的最小间距(米):帧宽 × 1.5。测试里所有「远点」都比它远。
FAR = SCAN_NM * NM * MIN_SEPARATION_FRAMES * 1.2


class FakeCtx:
    """按技能名派活的假上下文。**没配置的技能当场报错**,不静默成功。"""

    def __init__(self, script: dict):
        self.script = dict(script)
        self.calls: list[tuple[str, dict]] = []
        #: 调了但**没配置**的技能。执行器会把子技能抛的异常吞成一次「步骤失败」
        #: (optional 步的既有行为),所以光靠抛异常拦不住 —— 记在这里,让断言
        #: 能分辨「没配置」和「配置成失败」。
        self.unscripted: list[str] = []
        self._n: dict[str, int] = {}
        self.run_id = "test-cross-point"

    def run(self, skill_name, params, version=None):
        self.calls.append((skill_name, dict(params)))
        if skill_name not in self.script:
            self.unscripted.append(skill_name)
            raise AssertionError(
                f"假上下文没配置 {skill_name!r} —— 「没配置」不许静默变成"
                f"「跑成功了」。已配置: {sorted(self.script)}")
        n = self._n.get(skill_name, 0)
        self._n[skill_name] = n + 1
        fn = self.script[skill_name]
        data = fn(params, n) if callable(fn) else fn
        if data is FAIL:
            return SkillResult(skill_name=skill_name, success=False,
                               error="scripted failure")
        return SkillResult(skill_name=skill_name, success=True, data=dict(data))

    def check_abort(self):
        return False

    def check_halt(self):
        return ""

    def count(self, name):
        return sum(1 for c, _ in self.calls if c == name)

    def params_for(self, name):
        return [p for c, p in self.calls if c == name]


# ── 脚本片段 ──────────────────────────────────────────────────────────────────

def spots(*xy, map_known=True, extra=None):
    """``FindCleanSpot``:第 n 次调用把 ``xy[n]`` 放在候选队首。

    ``extra`` 可以在队首**之前**插一串会被筛掉的候选(第 n 次用 ``extra[n]``)。
    """
    def _f(params, n):
        head = list((extra or {}).get(n, []))
        if n >= len(xy):
            return FAIL
        x, y = xy[n]
        head.append({"x_m": x, "y_m": y, "distance_m": 0.0})
        return {"x_m": head[0]["x_m"], "y_m": head[0]["y_m"],
                "candidates": head, "map_known": map_known,
                # FindCleanSpot 回传的这个代次是从截断窗口推导的 —— 本技能
                # **不该**用它,测试里故意给一个假得离谱的值。
                "coord_epoch": -999}
    return _f


def prescan(*ready):
    """``PreScanCheck``:``tip_ready`` 三态序列。``None`` = 判不了。"""
    def _f(params, n):
        r = ready[n] if n < len(ready) else None
        out = {"tip_ready": r, "similarity": 0.92 if r else 0.41}
        if r is None:
            out["tip_ready"] = None
            out["abstain_reason"] = "起伏不足,判据弃权"
        return out
    return _f


def saved(ok=True, timed_out=False):
    def _f(params, n):
        if not ok:
            return FAIL
        return {"timed_out": timed_out, "saved_path": f"/tmp/frame_{n}.sxm"}
    return _f


def corrugation(*verdicts, width_nm=SCAN_NM):
    def _f(params, n):
        v = verdicts[n] if n < len(verdicts) else "normal"
        return {"verdict": v, "value_pm": 55.0 if v == "high" else 18.0,
                "detrend": "ols", "statistic": "std", "frame_usable": True,
                "width_nm": width_nm, "threshold_pm": 40.0}
    return _f


def script(*, ready=(True, True, True), verdicts=("normal",) * 3,
           points=((0.0, 0.0), (FAR, 0.0), (0.0, FAR)),
           map_known=True, extra=None, save_ok=True, save_timeout=False,
           width_nm=SCAN_NM):
    return {
        "FindCleanSpot": spots(*points, map_known=map_known, extra=extra),
        "MoveToXY": lambda p, n: {"moved": True},
        "PreScanCheck": prescan(*ready),
        "SaveScan": saved(save_ok, save_timeout),
        "AssessFrameCorrugation": corrugation(*verdicts, width_nm=width_nm),
    }


@pytest.fixture(autouse=True)
def _clean_crash_memory():
    from mast.core.tip_crash_tracker import reset_tip_crash_tracker

    reset_tip_crash_tracker()
    yield
    reset_tip_crash_tracker()


@pytest.fixture
def epoch(monkeypatch):
    """把权威代次查询换成一个可编排的序列。默认恒为 7。"""
    from mast.core import coord_epoch as ce

    box = {"values": [7], "calls": 0}

    def _fake():
        i = box["calls"]
        box["calls"] += 1
        vals = box["values"]
        return vals[i] if i < len(vals) else vals[-1]

    monkeypatch.setattr(ce, "read_current_epoch", _fake)
    return box


def run_skill(scr, *, ctx=None, **params):
    ctx = ctx or FakeCtx(scr)
    p = {"scan_nm": SCAN_NM, "n_points": 3}
    p.update(params)
    return ctx, CrossPointTipCheck().execute(ctx, p)


# ══════════════════════════════════════════════════════════════════════
# 1. 元数据 —— default 陷阱与安全级
# ══════════════════════════════════════════════════════════════════════

def test_metadata_is_confirm_and_reuses_an_existing_precondition():
    meta = CrossPointTipCheck().metadata()
    assert meta.category is SkillCategory.COMPOSITE
    assert meta.safety_level is SafetyLevel.CONFIRM, "它会移动针尖"
    # D9:本设计不新增任何 precondition —— 这一条是子技能已有的。
    assert meta.preconditions == ["z_controller_on"]


def test_threshold_has_no_default_in_the_spec():
    """有 default,「没传」就会被 pydantic 变成「传了那个数」,查 profile 那行
    永远到不了。本仓第三次记这个形状。"""
    specs = {p.name: p for p in CrossPointTipCheck().metadata().parameters}
    for name in ("corrugation_threshold_pm", "scan_nm"):
        assert specs[name].required is False
        assert specs[name].default is None, (
            f"{name} 有了 default —— 那会把「没传」物化成「传了」")
    assert specs["n_points"].default == DEFAULT_N_POINTS
    assert specs["n_points"].min_value == 2, "低于 2 就不是「换位置复测」"


def test_the_pm_and_nm_params_are_plain_floats_not_si_text():
    """不许给它们写 ``unit=``:``"40p"`` 会被 SI 解析成 4e-11,差 1e12。"""
    from mast.agents._shared.skill_adapter import _si_params

    meta = CrossPointTipCheck().metadata()
    assert not (set(_si_params(meta))
                & {"corrugation_threshold_pm", "scan_nm"})


def _through_the_tool_schema(**model_args) -> dict:
    """走模型真正会走的那条路 —— **填默认值那一层就在这里。**"""
    from mast.agents._shared.skill_adapter import (
        _coerce_si_params,
        _schema_from_metadata,
    )

    meta = CrossPointTipCheck().metadata()
    kw = _schema_from_metadata(meta)(**model_args).model_dump()
    kw.pop("tool_call_id", None)
    out, errs = _coerce_si_params(meta, kw)
    assert not errs, f"SI 转换报错: {errs}"
    return out


def test_omitting_the_threshold_reaches_the_profile(epoch, monkeypatch):
    """从**生产入口**进:不传阈值 ⇒ 一个数都不往下游塞,让它去查 profile。"""
    from mast.vision import scan_prep_thresholds as spt

    monkeypatch.setattr(
        spt, "resolve",
        lambda name=None: type("T", (), {"corrugation_ref_scan_nm": SCAN_NM})())

    ctx = FakeCtx(script())
    params = _through_the_tool_schema(n_points=3)
    assert params.get("corrugation_threshold_pm") is None, (
        "「没传」在 schema 那一层就被填成了一个数 —— 查 profile 那行永远到不了")
    result = CrossPointTipCheck().execute(ctx, params)

    assert result.success is True
    assert result.data["threshold_source"] == "profile"
    assert result.data["scan_nm_source"].startswith("profile:")
    for p in ctx.params_for("AssessFrameCorrugation"):
        assert "threshold_pm" not in p and "ref_scan_nm" not in p, (
            "没传阈值却往判据②塞了一个 —— 它就再也查不到 profile 了")


def test_explicit_threshold_without_an_explicit_field_of_view_is_refused(epoch):
    """显式阈值配 profile 声明的视野 = 把两次不同标定的数拼成一个判据。"""
    ctx, result = run_skill(script(), scan_nm=None,
                            corrugation_threshold_pm=40.0)
    assert result.success is True
    assert result.data["cross_verdict"] == "undecidable"
    assert "scan_nm" in result.data["reason"]
    assert ctx.calls == [], "参数不成对就该在动手之前拒绝,一步都不该走"


def test_explicit_threshold_travels_with_its_field_of_view(epoch):
    ctx, result = run_skill(script(), corrugation_threshold_pm=40.0)
    assert result.success is True
    for p in ctx.params_for("AssessFrameCorrugation"):
        assert p["threshold_pm"] == 40.0
        assert p["ref_scan_nm"] == SCAN_NM, "阈值和它标定时的视野必须同行"


# ══════════════════════════════════════════════════════════════════════
# 2. 三值转移:坏 / 分歧 / 判不了
# ══════════════════════════════════════════════════════════════════════

def test_three_bad_points_give_bad_tip(epoch):
    ctx, result = run_skill(script(ready=(False, False, False)))
    assert result.success is True
    assert result.data["cross_verdict"] == "bad_tip"
    assert result.data["verdict"] == "bad_tip"
    assert (result.data["n_points"], result.data["n_judged"],
            result.data["n_bad"]) == (3, 3, 3)
    assert ctx.count("PreScanCheck") == 3
    assert len(result.data["points"]) == 3


def test_high_corrugation_alone_also_counts_as_bad(epoch):
    """判坏是**析取**:判据①合格、判据②说 high,这个点仍然判坏。"""
    ctx, result = run_skill(
        script(ready=(True, True, True), verdicts=("high", "high", "high")))
    assert result.data["cross_verdict"] == "bad_tip"
    assert result.data["n_bad"] == 3


def test_one_bad_one_good_one_undecidable_gives_surface_feature(epoch):
    ctx, result = run_skill(script(ready=(False, True, None)))
    assert result.data["cross_verdict"] == "surface_feature"
    assert (result.data["n_judged"], result.data["n_bad"],
            result.data["n_good"]) == (2, 1, 1)
    # 摘要要说清下一步:指向表面就**不要**据此去修针尖或换样品。
    assert "不是针" in result.summary


def test_one_bad_two_undecidable_gives_undecidable(epoch):
    """judged=1 —— 一个点不构成「换位置复测」。**判不了不许折叠成判坏。**"""
    ctx, result = run_skill(script(ready=(False, None, None)))
    assert result.data["cross_verdict"] == "undecidable"
    assert result.data["n_judged"] == 1
    assert result.data["n_bad"] == 1


def test_a_point_with_no_saved_frame_still_carries_criterion_one(epoch):
    """存盘失败 ⇒ 判据②判不了,但判据①的票还在 —— 两个判据各有各的取帧路径。"""
    scr = script(ready=(False, False, False))
    scr["SaveScan"] = saved(ok=False)
    scr.pop("AssessFrameCorrugation")      # 没有路径就根本不该调它
    ctx, result = run_skill(scr)
    assert ctx.unscripted == [], "没有路径却去调了判据②"
    assert result.data["cross_verdict"] == "bad_tip"
    assert all(p["corrugation_verdict"] == "undecidable"
               for p in result.data["points"])
    assert "没存下来" in result.data["points"][0]["abstain_reason"]


def test_a_timed_out_save_never_borrows_the_latest_file(epoch):
    """存盘超时 ⇒ 不拿「最近 120 s 的最新 .sxm」冒充这一帧。"""
    scr = script(save_timeout=True)
    scr.pop("AssessFrameCorrugation")
    ctx, result = run_skill(scr)
    assert ctx.count("AssessFrameCorrugation") == 0
    assert ctx.unscripted == []
    assert "存盘超时" in result.data["points"][0]["abstain_reason"]


def test_a_frame_of_the_wrong_width_has_its_corrugation_discarded(epoch):
    """存下来的帧宽对不上 ⇒ 这个文件多半不是刚才那一帧,它的起伏结论作废。"""
    ctx, result = run_skill(
        script(ready=(True, True, True), verdicts=("high",) * 3,
               width_nm=SCAN_NM * 2))
    pts = result.data["points"]
    assert all(p["corrugation_verdict"] == "undecidable" for p in pts)
    assert "不是刚才那一帧" in pts[0]["abstain_reason"]
    # 判据①仍然合格 ⇒ 三点全判好 ⇒ 不是针的问题
    assert result.data["cross_verdict"] == "surface_feature"


def test_a_failed_move_never_measures_the_old_spot_again(epoch):
    """没走到就地扫 = 把同一片表面测第二次,冒充独立证据。"""
    scr = script()
    calls = {"n": 0}

    def _move(p, n):
        calls["n"] += 1
        return FAIL if n == 1 else {"moved": True}

    scr["MoveToXY"] = _move
    ctx, result = run_skill(scr)
    assert ctx.count("MoveToXY") == 3
    assert ctx.count("PreScanCheck") == 2, "移动失败的那个点不该再扫一帧"
    assert "移动失败" in result.data["points"][1]["abstain_reason"]
    assert result.data["points"][1]["tip_ready"] is None


# ══════════════════════════════════════════════════════════════════════
# 3. 坐标代次:每点各记一次,中途变了整批作废并点名
# ══════════════════════════════════════════════════════════════════════

def test_epoch_change_mid_batch_voids_everything_and_names_the_point(epoch):
    epoch["values"] = [5, 5, 9]
    ctx, result = run_skill(script(ready=(False, False, False)))
    assert result.data["cross_verdict"] == "undecidable", (
        "三点全判坏,但中途粗动过 —— 这批证据不再是「同一片区域的换位置复测」")
    assert result.data["coord_epoch"] is None
    assert "P3" in result.data["reason"], "整批作废必须点名是哪一点"
    assert "代次 9" in result.data["reason"]
    assert result.data["stopped_early"] == "coord_epoch_changed"


def test_epoch_is_read_once_per_point_from_the_authoritative_query(epoch):
    ctx, result = run_skill(script())
    assert epoch["calls"] == 3, "每点各记一次,不是整批记一次"
    assert result.data["coord_epoch"] == 7
    assert [p["coord_epoch"] for p in result.data["points"]] == [7, 7, 7]


def test_the_derived_epoch_from_the_map_is_never_used(epoch):
    """``FindCleanSpot`` 回传的代次是从 2000 行截断窗口推导的,不是权威值。"""
    ctx, result = run_skill(script())
    assert all(p["coord_epoch"] == 7 for p in result.data["points"]), (
        "用了地图窗口推导出来的那个代次(-999)")
    for p in ctx.params_for("MoveToXY"):
        assert "coord_epoch" not in p or p["coord_epoch"] != -999


def test_unreadable_epoch_voids_the_batch(epoch):
    """代次读不到 ⇒ 整批作废。两个未知不构成「一致」。"""
    epoch["values"] = [None]
    ctx, result = run_skill(script(ready=(False, False, False)))
    assert result.data["cross_verdict"] == "undecidable"
    assert "读不到" in result.data["reason"]


# ══════════════════════════════════════════════════════════════════════
# 4. 选点的两条后置筛
# ══════════════════════════════════════════════════════════════════════

def test_pick_candidate_rejects_an_overlapping_frame():
    chosen, rejected = pick_candidate(
        [{"x_m": 10 * NM, "y_m": 0.0}, {"x_m": FAR, "y_m": 0.0}],
        taken=[(0.0, 0.0)], min_sep_m=SCAN_NM * NM * MIN_SEPARATION_FRAMES,
        crash_count=lambda x, y: 0)
    assert chosen["x_m"] == FAR
    assert [r["why"] for r in rejected] == [REJECT_TOO_CLOSE]


def test_pick_candidate_rejects_a_crater_we_made():
    seen: list = []

    def _crash(x, y):
        seen.append((x, y))
        return 1 if x == 0.0 else 0

    chosen, rejected = pick_candidate(
        [{"x_m": 0.0, "y_m": 0.0}, {"x_m": FAR, "y_m": 0.0}],
        taken=[], min_sep_m=1.0, crash_count=_crash)
    assert chosen["x_m"] == FAR
    assert [r["why"] for r in rejected] == [REJECT_CRASH_HISTORY]


def test_the_separation_filter_is_live_in_the_composite(epoch):
    """队首放一个与 P1 重叠的候选 ⇒ 必须被跳过,选后面那个远的。"""
    near = {"x_m": 5 * NM, "y_m": 0.0, "distance_m": 5 * NM}
    ctx, result = run_skill(script(extra={1: [near]}))
    moved = [(p["x_m"], p["y_m"]) for p in ctx.params_for("MoveToXY")]
    assert moved[1] == (FAR, 0.0), "选了一个与上一帧重叠的点 —— 那不是独立证据"
    assert result.data["rejected_by_reason"][REJECT_TOO_CLOSE] >= 1


def test_the_crash_filter_is_live_in_the_composite(epoch):
    """在自己刚炸出来的坑上复测「仍然差」,差的可能是坑,不是针。"""
    from mast.core.tip_crash_tracker import get_tip_crash_tracker

    crater = {"x_m": 3 * FAR, "y_m": 0.0, "distance_m": 0.0}
    get_tip_crash_tracker().record_crash(crater["x_m"], crater["y_m"])

    ctx, result = run_skill(script(extra={0: [crater]}))
    moved = [(p["x_m"], p["y_m"]) for p in ctx.params_for("MoveToXY")]
    assert moved[0] == (0.0, 0.0), "选了一个撞过针的格子"
    assert result.data["rejected_by_reason"][REJECT_CRASH_HISTORY] == 1


def test_the_crash_memory_says_it_is_process_local(epoch):
    """``crash_count == 0`` 有两种读法,返回值必须让人分得开。"""
    ctx, result = run_skill(script())
    mem = result.data["crash_memory"]
    assert mem["scope"] == "process_local"
    assert "不等于" in mem["note"]


def test_the_spot_search_never_invents_a_third_purpose(epoch):
    """``purpose`` 的取值映射到「这里被我们弄脏了」的损伤词表 —— 复测不是损伤。"""
    ctx, _ = run_skill(script())
    assert all(p["purpose"] == "tip_shape" for p in ctx.params_for("FindCleanSpot"))


def test_already_used_spots_are_handed_back_as_exclusions(epoch):
    ctx, _ = run_skill(script())
    excl = [p["exclude_spots"] for p in ctx.params_for("FindCleanSpot")]
    assert excl[0] == ""
    assert excl[1].count(";") == 0 and excl[1]
    assert excl[2].count(";") == 1


# ══════════════════════════════════════════════════════════════════════
# 5. 候选不足:说出来,不凑数
# ══════════════════════════════════════════════════════════════════════

def test_insufficient_candidates_says_so_instead_of_padding(epoch):
    """第三次找不到点 ⇒ 明说候选不足,**不拿两个点去凑一个 bad_tip**。"""
    ctx, result = run_skill(
        script(ready=(False, False), points=((0.0, 0.0), (FAR, 0.0))))
    assert result.success is True
    assert result.data["cross_verdict"] == "undecidable"
    assert result.data["insufficient_candidates"] is True
    assert result.data["stopped_early"] == "no_candidate"
    assert "不凑数" in result.data["reason"]
    assert "只凑出 2 个" in result.data["reason"]
    assert ctx.count("PreScanCheck") == 2, "凑数了 —— 它多测了一个点"
    # 两个点的原始证据一条都不许丢
    assert len(result.data["points"]) == 2
    assert result.data["n_bad"] == 2


def test_a_short_batch_never_escalates_to_bad_tip(epoch):
    """一片用完的表面上判「仍然差」,差的可能正是用完它的那些动作。"""
    ctx, result = run_skill(
        script(ready=(False, False), points=((0.0, 0.0), (FAR, 0.0))))
    assert result.data["cross_verdict"] != "bad_tip"
    # 但记账仍然如实:两票都是判坏
    assert (result.data["n_judged"], result.data["n_bad"]) == (2, 2)


def test_no_candidate_at_all_is_undecidable_not_a_failure(epoch):
    scr = script(points=())
    ctx, result = run_skill(scr)
    assert result.success is True, (
        "「判不了」不该走成「步骤失败」—— 下游闸门按 verdict 分三路")
    assert result.data["cross_verdict"] == "undecidable"
    assert result.data["n_points"] == 0
    assert ctx.count("MoveToXY") == 0


def test_n_points_below_two_is_refused_before_touching_anything(epoch):
    ctx, result = run_skill(script(), n_points=1)
    assert result.data["cross_verdict"] == "undecidable"
    assert "换位置复测" in result.data["reason"]
    assert ctx.calls == []


# ══════════════════════════════════════════════════════════════════════
# 6. 只读性钉子
# ══════════════════════════════════════════════════════════════════════

def test_it_only_ever_calls_the_five_read_and_move_skills(epoch):
    ctx, _ = run_skill(script())
    used = {name for name, _ in ctx.calls}
    assert used <= set(ALLOWED_SUBSKILLS), f"多调了: {used - set(ALLOWED_SUBSKILLS)}"


def test_it_never_records_a_crash(epoch, monkeypatch):
    """复测只观察。记一次撞针,下一轮的选点筛就会绕开自己刚判过的点。"""
    from mast.core.tip_crash_tracker import get_tip_crash_tracker

    tracker = get_tip_crash_tracker()

    def _boom(*a, **k):
        raise AssertionError("复测记了一次撞针 —— 它应该是只读的")

    monkeypatch.setattr(tracker, "record_crash", _boom)
    monkeypatch.setattr(tracker, "note_recovery", _boom)
    _, result = run_skill(script(ready=(False, False, False)))
    assert result.data["cross_verdict"] == "bad_tip"


def test_the_source_writes_no_marker_and_no_crash():
    """源码级钉子:``surface_feature`` 的点一旦上了避让地图,后面找畴界的搜索
    就会躲开自己的目标。这里连**能**写的入口都不许出现。"""
    src = Path(CPT.__file__).read_text(encoding="utf-8")
    for forbidden in ("record_crash(", "log_marker", "note_recovery(",
                      "record_scan_path("):
        assert forbidden not in src, f"只读技能里出现了写入口: {forbidden}"


def test_dry_run_moves_nothing_and_produces_no_evidence(epoch):
    scr = script()
    scr.pop("MoveToXY")
    scr.pop("PreScanCheck")
    scr.pop("SaveScan")
    scr.pop("AssessFrameCorrugation")
    ctx, result = run_skill(scr, dry_run=True)
    assert result.success is True
    assert ctx.unscripted == [], "彩排却动了硬件"
    assert {name for name, _ in ctx.calls} == {"FindCleanSpot"}
    assert result.data["cross_verdict"] == "undecidable"
    assert len(result.data["planned_points"]) == 3
    assert result.data["points"] == []
    assert "彩排" in result.summary


# ══════════════════════════════════════════════════════════════════════
# 7. map_known 一路传出
# ══════════════════════════════════════════════════════════════════════

def test_map_unknown_travels_all_the_way_out(epoch):
    """「读不到实验记录」与「表面干净」在几何上不可区分 —— 必须传到报告。"""
    ctx, result = run_skill(script(map_known=False, ready=(False, False, False)))
    assert result.data["map_known"] is False
    assert all(p["map_known"] is False for p in result.data["points"])
    assert "不知道 ≠ 干净" in result.summary
    # 结论本身照常给 —— 地图读不到不是拒判的理由,是要说出来的事
    assert result.data["cross_verdict"] == "bad_tip"


def test_map_known_defaults_to_true_when_every_point_could_read_it(epoch):
    ctx, result = run_skill(script())
    assert result.data["map_known"] is True


# ══════════════════════════════════════════════════════════════════════
# 8. 变异验证 —— 先证明自己动了手,再看测试红没红
# ══════════════════════════════════════════════════════════════════════

def test_mutation_counting_undecidable_as_judged_turns_a_test_red(
        epoch, monkeypatch):
    """变异:把「判不了」计入 judged(当成判好)。"""
    from mast.conduct import cross_check as cc

    probe = cc.PointVerdict(tip_ready=None, coord_epoch=1,
                            corrugation_verdict="normal")
    before = cc.aggregate_cross_points([probe, probe]).n_judged
    monkeypatch.setattr(cc, "_is_good", lambda p: not cc._is_bad(p))
    after = cc.aggregate_cross_points([probe, probe]).n_judged
    assert (before, after) == (0, 2), "变异没落盘 —— 后面的红没有意义"

    _, result = run_skill(script(ready=(False, None, None)))
    assert result.data["cross_verdict"] != "undecidable", (
        "「1 坏 + 2 判不了 ⇒ undecidable」这条用例抓不住这个变异")


def test_mutation_dropping_the_crash_filter_turns_a_test_red(epoch, monkeypatch):
    """变异:选点时不再看撞针记忆。"""
    from mast.core.tip_crash_tracker import get_tip_crash_tracker

    crater_x = 3 * FAR
    get_tip_crash_tracker().record_crash(crater_x, 0.0)
    assert CrossPointTipCheck._crash_count(crater_x, 0.0) == 1
    monkeypatch.setattr(CrossPointTipCheck, "_crash_count",
                        staticmethod(lambda x, y: 0))
    assert CrossPointTipCheck._crash_count(crater_x, 0.0) == 0, "变异没落盘"

    crater = {"x_m": crater_x, "y_m": 0.0, "distance_m": 0.0}
    ctx, _ = run_skill(script(extra={0: [crater]}))
    moved = [(p["x_m"], p["y_m"]) for p in ctx.params_for("MoveToXY")]
    assert moved[0] != (0.0, 0.0), "选点筛的用例抓不住「撞针筛被拿掉」"


def test_mutation_dropping_the_per_point_epoch_turns_a_test_red(
        epoch, monkeypatch):
    """变异:代次不再每点各记一次(整批记一次)。"""
    epoch["values"] = [5, 5, 9]
    monkeypatch.setattr(CrossPointTipCheck, "_read_epoch",
                        staticmethod(lambda: 5))
    assert CrossPointTipCheck._read_epoch() == CrossPointTipCheck._read_epoch()

    _, result = run_skill(script(ready=(False, False, False)))
    assert result.data["cross_verdict"] != "undecidable", (
        "「代次中途变了 ⇒ 整批作废」这条用例抓不住这个变异")
    assert result.data["cross_verdict"] == "bad_tip"


def test_mutation_dropping_the_separation_filter_turns_a_test_red(
        epoch, monkeypatch):
    """变异:两两距离筛的倍数改成 0 ⇒ 重叠的帧会被选上。"""
    assert MIN_SEPARATION_FRAMES > 1.0
    monkeypatch.setattr(CPT, "MIN_SEPARATION_FRAMES", 0.0)

    near = {"x_m": 5 * NM, "y_m": 0.0, "distance_m": 5 * NM}
    ctx, _ = run_skill(script(extra={1: [near]}))
    moved = [(p["x_m"], p["y_m"]) for p in ctx.params_for("MoveToXY")]
    assert moved[1] == (5 * NM, 0.0), "间距筛的用例抓不住「筛被拿掉」"


# ══════════════════════════════════════════════════════════════════════
# 9. 通用层纪律
# ══════════════════════════════════════════════════════════════════════

def test_the_module_names_no_sample():
    """通用层零样品名 —— 这是一个 STM 系统,不是某一个样品的系统。"""
    import re

    src = Path(CPT.__file__).read_text(encoding="utf-8")
    hits = re.findall(r"(?<![A-Za-z0-9])(wo2i2|woi2?|au111)(?![A-Za-z0-9])",
                      src, re.IGNORECASE)
    assert not hits, f"通用层出现样品名: {set(hits)}"
