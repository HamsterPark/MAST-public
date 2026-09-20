"""``SearchDomainBoundary`` —— 畴界二分(``mode="bisect"``)。

S3 畴搜索设计 §5.5 的整张表 + §5.4 里与状态机有关的那几条。这里盯的**不是**指纹算
得对不对(那是 ``tests/v2/unit/vision/test_domain_phase*.py`` 的活),而是这个状态机
六件写错了**零报错**的事:

1. 终止条件被当成一个「精度参数」往下调 —— 压电还能细三四个数量级,于是二分空转到
   预算耗尽,每一次空转都真的去扫了一帧,而且每一帧「都成功了」。
2. 二分中途发生粗动,状态机接着用旧的米坐标 —— 它们已经指向另一片表面,而剩下的
   每一帧照样扫得很成功。**这是最重要的负向测试。**
3. bracket 的两端分属粗动前后 —— 「取中点」这个动作没有定义,算出来的中点会是一个
   看上去完全正常的坐标。
4. 跨站点的畴界被报成米坐标 —— 步数乘一个标称步长得到的数,和真坐标长得一模一样。
5. 二分落点走了 ``pick_next_position`` —— ``reuse_overlap_frac`` 把中点静默丢掉,
   表现成「这片表面在当前策略下没有位置了」。
6. 「判不了」时下一个坐标由模型发明 —— 那是一个几何问题,答案由三个数唯一确定。

跑:
    .venv-v2-py313/Scripts/python.exe -m pytest \
      tests/v2/unit/skills/composite/test_search_domain_boundary_bisect.py -q
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

import math  # noqa: E402
from dataclasses import replace  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from mast.core.types import SkillResult  # noqa: E402
from mast.io.map_analysis import AnalysisConfig  # noqa: E402
from mast.skills.composite import search_domain_boundary as SDB  # noqa: E402
from mast.skills.composite.search_domain_boundary import (  # noqa: E402
    END_MIXED,
    END_NOT_A_PAIR,
    END_TOLERANCE,
    END_UNDETERMINED,
    MODE_BISECT,
    POINT_ASSESSED,
    POINT_EPOCH_STALE,
    POINT_PLANNED,
    POINT_UNREACHABLE,
    REFUSE_BRACKET_EPOCH_SPLIT,
    REFUSE_BRACKET_NOT_A_PAIR,
    REFUSE_BRACKET_NOT_FOUND,
    REFUSE_CROSS_SITE_BISECT,
    REFUSE_NO_BRACKET,
    REFUSE_NO_REFERENCE,
    ROLE_HI,
    ROLE_LO,
    ROLE_MID,
    ROLE_OFFSET,
    STATE_ABANDONED,
    STATE_CONVERGED,
    STATE_INVALIDATED,
    BisectMachine,
    SearchDomainBoundary,
    bracket_gap_m,
    check_bracket_epochs,
    cross_site_downgrade,
    locate_tolerance,
    midpoint,
    offset_probe,
    probe_position_problem,
    rebuild_brackets,
)

#: 这台仪器有 XY 粗动 ⇒ 中心区 500 nm + 脉冲避让 500 nm 那一对。二分**不**解除中心
#: 区:它的落点闸看的是 ``piezo_half_range_m``(硬边界),中心区改的是
#: ``effective_half_range_m``(排路线时的可用区)。
BASE_CFG = AnalysisConfig(
    piezo_half_range_m=1.5e-6,
    frame_size_m=100e-9,
    strategy="center_first",
    has_xy_coarse_motion=True,
    center_zone_side_m=500e-9,
    pulse_r_m=500e-9,
)

ATOMIC_FRAME_M = 5e-9
EPOCH = 2

#: 一对端点。间距 640 nm ⇒ 在 5 nm 的容差上正好要 7 次收窄(2⁷ = 128,640/128 = 5)。
LO = (-320e-9, 0.0)
HI = (320e-9, 0.0)
#: 真畴界的位置。刻意不落在任何一个中点上 —— 落在中点上会让「收敛」变成运气。
BOUNDARY_X = 37.3e-9

LABEL_LO, LABEL_HI = "alpha", "beta"


# ── 参照系:二分要求它已经标定 ────────────────────────────────────────────────

def calibrated_reference():
    """一份**标定过的**参照系。二分没有它开不了头(设计 §3.3)。"""
    from mast.vision.domain_reference import DomainPrototype, DomainReference

    return DomainReference(
        version="v001", sample="test", symmetry_deg=60.0,
        labels=(LABEL_LO, LABEL_HI),
        prototypes={
            LABEL_LO: DomainPrototype(label=LABEL_LO,
                                      peaks=((30.0, 0.250, 1.0), (90.0, 0.250, 0.9))),
            LABEL_HI: DomainPrototype(label=LABEL_HI,
                                      peaks=((5.0, 0.320, 1.0), (65.0, 0.320, 0.9))),
        },
        provenance="测试用", confirmed_by="test",
        w_angle=1.0, w_period=1.0, match_tol=0.2,
        ambiguity_margin=0.05, mixed_coverage_min=0.6,
    )


# ── 假上下文 ─────────────────────────────────────────────────────────────────

class FakeCtx:
    """按技能名派活。``AssessDomainPhase`` 的回包由**上一次 ScanAt 的坐标**决定 ——
    二分的整个意义就是「下一个位置取决于上一帧的判定」,一个与位置无关的脚本验不了
    这条链。"""

    def __init__(self, verdict_fn=None, fail_scan_at=None):
        self.calls: list[tuple[str, dict]] = []
        self.run_id = "test-domain-bisect"
        self.state = None
        self.last_xy: tuple[float, float] = (0.0, 0.0)
        self._frames = 0
        self._verdict_fn = verdict_fn or phase_by_x(BOUNDARY_X)
        self._fail_scan_at = set(fail_scan_at or ())

    def run(self, skill_name, params, version=None):
        self.calls.append((skill_name, dict(params)))
        if skill_name == "ScanAt":
            self.last_xy = (float(params["center_x_m"]), float(params["center_y_m"]))
            if self.count("ScanAt") in self._fail_scan_at:
                return SkillResult(skill_name=skill_name, success=False,
                                   error="scripted scan failure")
            return SkillResult(skill_name=skill_name, success=True, data={"ok": True})
        if skill_name == "GetLatestScanFile":
            self._frames += 1
            return SkillResult(skill_name=skill_name, success=True,
                               data={"path": f"/tmp/bisect_{self._frames}.sxm"})
        if skill_name == "AssessDomainPhase":
            return SkillResult(skill_name=skill_name, success=True,
                               data=self._verdict_fn(self.last_xy, params))
        raise AssertionError(
            f"二分只该调 ScanAt / GetLatestScanFile / AssessDomainPhase,"
            f"却调了 {skill_name!r} —— 「判不了」时的下一个坐标必须由算法给,"
            f"不许经过任何一个会发明数字的东西。")

    def check_abort(self):
        return False

    def check_halt(self):
        return ""

    def count(self, name):
        return sum(1 for c, _ in self.calls if c == name)

    def scan_positions(self):
        return [(p["center_x_m"], p["center_y_m"])
                for c, p in self.calls if c == "ScanAt"]


def _packet(label=None, verdict=None, reason="", fp=((30.0, 0.25, 1.0),)):
    return {
        "verdict": verdict or (label or "undetermined"),
        "label": label,
        "verdict_reason": reason,
        "next_step": "…",
        "fingerprint": [list(t) for t in fp],
        "scan_angle_deg": 0.0,
        "n_peaks": len(fp), "nm_per_px": 0.0098, "scale": "full",
    }


def phase_by_x(boundary_x_m):
    """一条竖直畴界:x < 边界 ⇒ ``alpha``,x > 边界 ⇒ ``beta``。"""
    def _f(xy, params):
        return _packet(label=(LABEL_LO if xy[0] < boundary_x_m else LABEL_HI))
    return _f


def always_undetermined(reason="no_atomic_phase"):
    """两端判得出、中点一律判不了 —— 「不可判」三级处理的输入。"""
    def _f(xy, params):
        if abs(xy[0] - LO[0]) < 1e-15 and abs(xy[1] - LO[1]) < 1e-15:
            return _packet(label=LABEL_LO)
        if abs(xy[0] - HI[0]) < 1e-15 and abs(xy[1] - HI[1]) < 1e-15:
            return _packet(label=LABEL_HI)
        return _packet(verdict="undetermined", reason=reason)
    return _f


# ── 接线 ─────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def wired(tmp_path, monkeypatch):
    """`map_scope` / 参照系 / 代次全换成显式替身,sidecar 改到 ``tmp_path``。"""
    from mast.core import coord_epoch as CE
    from mast.core import map_scope as MS
    from mast.skills.composite import graph_executor as GE
    from mast.vision import domain_reference as DR

    monkeypatch.setattr(
        MS, "analysis_config",
        lambda state=None, *, safety=None, frame_size_m=None: replace(
            BASE_CFG, frame_size_m=float(frame_size_m or BASE_CFG.frame_size_m)))
    monkeypatch.setattr(MS, "load_markers", lambda **kw: ([], EPOCH, True))
    monkeypatch.setattr(CE, "read_current_epoch", lambda: EPOCH)
    monkeypatch.setattr(DR, "load_reference",
                        lambda v=None, *, sample=None: calibrated_reference())
    monkeypatch.setattr(GE, "_sidecar_dir", lambda: tmp_path)
    yield


def run_bisect(ctx=None, **params):
    ctx = ctx if ctx is not None else FakeCtx()
    p = {"mode": MODE_BISECT, "frame_size_m": ATOMIC_FRAME_M, "pixels": 512,
         "lo_x_m": LO[0], "lo_y_m": LO[1], "hi_x_m": HI[0], "hi_y_m": HI[1],
         "max_iterations": 40}
    p.update(params)
    return ctx, SearchDomainBoundary().execute(ctx, p)


# ═══════════════════════════════════════════════════════════════════════
# 1. 终止条件的下限是**帧宽**,不是压电分辨率(设计 D9)
# ═══════════════════════════════════════════════════════════════════════

def _assert_tolerance_is_the_frame(data):
    assert data["locate_tolerance_m"] == pytest.approx(ATOMIC_FRAME_M), (
        f"终止容差是 {data['locate_tolerance_m']},不是帧宽 {ATOMIC_FRAME_M} —— "
        f"压电还能细三四个数量级,拿它当下限只会让二分空转,而每一次空转都真的"
        f"去扫了一帧")
    assert data["locate_tolerance_clamped"] is True
    why = data["locate_tolerance_reason"]
    assert "同一片表面" in why and "换一套判据" in why, (
        "夹上去的理由必须写明白,否则下一个人只会去调压电")


def test_bisect_stops_at_frame_size_not_piezo_resolution():
    """给一个压电分辨率量级的容差,状态机照样停在帧宽。

    这条同时钉住两件事:那个数被夹上去了(纯函数),以及**真的少扫了 13 帧**
    (端到端)—— 只断言前者的话,一个「算了但没用上」的夹紧看不出来。
    """
    tol, clamped, why = locate_tolerance(ATOMIC_FRAME_M, 1e-12)
    assert (tol, clamped) == (ATOMIC_FRAME_M, True)
    assert "压电" in why

    ctx, result = run_bisect(locate_tolerance_m=1e-12)
    assert result.success
    _assert_tolerance_is_the_frame(result.data)

    b = result.data["bisect"]
    assert b["state"] == STATE_CONVERGED and b["end_reason"] == END_TOLERANCE
    # 640 nm / 2⁷ = 5 nm。7 次收窄 + 2 个端点 = 9 帧。若容差真的用了 1 pm,
    # 要 ceil(log2(640e-9/1e-12)) = 20 次,也就是 22 帧。
    assert b["iterations"] == 7, f"收窄了 {b['iterations']} 次"
    assert ctx.count("ScanAt") == 9
    assert b["gap_m"] <= ATOMIC_FRAME_M


def test_the_converged_boundary_carries_its_uncertainty():
    """收敛给的是**区间中心 + 不确定度**,不是一个光秃秃的坐标。"""
    _, result = run_bisect()
    b = result.data["bisect"]
    bx, by = b["boundary_m"]
    assert abs(bx - BOUNDARY_X) <= ATOMIC_FRAME_M, (
        f"报出来的畴界 {bx * 1e9:.1f} nm 离真边界 {BOUNDARY_X * 1e9:.1f} nm "
        f"超过一个帧宽")
    assert by == pytest.approx(0.0)
    assert b["boundary_uncertainty_m"] == pytest.approx(b["gap_m"] / 2.0)
    assert "±" in (result.summary or "")


def test_an_operator_tolerance_coarser_than_the_frame_is_honoured():
    """比帧宽大的容差是用户的话,照办 —— 夹紧只往一个方向。"""
    tol, clamped, _ = locate_tolerance(ATOMIC_FRAME_M, 40e-9)
    assert (tol, clamped) == (40e-9, False)
    _, result = run_bisect(locate_tolerance_m=40e-9)
    b = result.data["bisect"]
    assert b["state"] == STATE_CONVERGED
    assert b["iterations"] == 4, "640 / 2⁴ = 40 nm"


def test_a_mixed_frame_converges_immediately():
    """两个畴的峰都在这一帧里 ⇒ 畴界就在这一帧内,不必再二分下去。"""
    def _mixed_at_the_middle(xy, params):
        if abs(xy[0]) < 1e-9:
            return _packet(verdict="mixed")
        return _packet(label=(LABEL_LO if xy[0] < 0 else LABEL_HI))

    ctx, result = run_bisect(FakeCtx(verdict_fn=_mixed_at_the_middle))
    b = result.data["bisect"]
    assert b["state"] == STATE_CONVERGED and b["end_reason"] == END_MIXED
    assert b["boundary_m"] == [0.0, 0.0]
    assert b["boundary_uncertainty_m"] == pytest.approx(ATOMIC_FRAME_M / 2)
    assert ctx.count("ScanAt") == 3, "两个端点 + 一个中点就够了"


# ═══════════════════════════════════════════════════════════════════════
# 2. 跨代次(设计 D9b)—— 整份测试里最重要的那两条
# ═══════════════════════════════════════════════════════════════════════

def _assert_bracket_was_invalidated(ctx, result, *, scans_before_bump):
    assert result.success is False, (
        "粗动把整个 bracket 作废了,产物是空的 —— 报成功会让上层拿着一个空结果"
        "往下走,那正是本仓记过的「假成功」")
    b = result.data["bisect"]
    assert b["state"] == STATE_INVALIDATED, (
        f"bracket 的状态是 {b['state']} —— 粗动之后 p_lo / p_hi 指向另一片表面,"
        f"它们之间已经不存在那条畴界。继续二分会扫得很成功,而每一帧都在错的地方")
    assert b["boundary_m"] is None, (
        "作废的 bracket 还给了一个畴界坐标 —— 那是用旧坐标系说的话")
    assert ctx.count("ScanAt") == scans_before_bump, (
        f"代次变了之后还扫了 {ctx.count('ScanAt') - scans_before_bump} 帧")
    err = result.error or ""
    assert "不是暂停" in err, (
        "「作废」与「暂停」的差别不是措辞:暂停意味着回头还能用这两个坐标")
    assert "不做跨代次换算" in err
    stale = [p for p in result.data["point_log"]
             if p["status"] == POINT_EPOCH_STALE]
    assert stale, "作废没有在采样点记录里留下痕迹"
    assert all(p not in result.data["points"] for p in stale), (
        "作废的那个点针尖没去过,不许在地图上留足迹")


def test_epoch_bump_invalidates_the_bracket():
    """二分期间坐标代次改变时，立即作废旧区间，不在新表面继续解释旧坐标。"""
    from mast.core import coord_epoch as CE

    seen = {"n": 0}

    def _epoch():
        seen["n"] += 1
        # 1 = 准备时盖章,2/3 = 两个端点之前的复核,4 = 第一个中点之前 —— 之后粗动。
        return EPOCH if seen["n"] <= 4 else EPOCH + 1

    CE.read_current_epoch = _epoch
    try:
        ctx, result = run_bisect()
    finally:
        CE.read_current_epoch = lambda: EPOCH
    # 两个端点 + 第一个中点扫完了,第二个中点之前撞上代次变化。
    _assert_bracket_was_invalidated(ctx, result, scans_before_bump=3)


def test_bracket_endpoints_must_share_one_epoch():
    """**两端必须同代次才允许二分**(设计 D9b 第 1 条)。

    两端分属粗动前后时,它们之间根本不存在一段连续的表面 —— 「取中点」这个动作没有
    定义,而算出来的那个中点是一个看上去完全正常的米坐标。
    """
    assert check_bracket_epochs(3, 3, 3) is None
    assert check_bracket_epochs(None, None, None) is None, (
        "两端都没盖章 ⇒ 没有代次保护,但那不是「陈旧」")
    split = check_bracket_epochs(2, 3, 3)
    assert split and "不同的坐标代次" in split and "不做跨代次换算" in split
    stale = check_bracket_epochs(2, 2, 3)
    assert stale and "重新找一对端点" in stale
    half = check_bracket_epochs(None, 3, 3)
    assert half and "分不清" in half, "一端有章一端没有 = 分不清,分不清时不许二分"

    # 端到端:记录里的一对端点跨代次 ⇒ 拒绝,一帧都不发。
    markers = [
        _row(LO, role=ROLE_LO, verdict=LABEL_LO, epoch=EPOCH - 1, bid="b1"),
        _row(HI, role=ROLE_HI, verdict=LABEL_HI, epoch=EPOCH, bid="b1"),
    ]
    ctx, result = _run_with_markers(markers, bracket_id="b1")
    assert result.success is False
    assert result.data["refusal"]["code"] == REFUSE_BRACKET_EPOCH_SPLIT
    assert ctx.calls == [], "跨代次的 bracket 还发了扫描"


def test_a_stale_pair_is_refused_even_though_both_ends_agree():
    """两端互相同代次、但都是**上一代**的 —— 同样不许二分。

    这一条与上一条是两个失败模式:「两端互相对不上」和「两端一致但整对过期」。
    只查前者的话,一次粗动之后整对旧端点会原样通过。
    """
    markers = [
        _row(LO, role=ROLE_LO, verdict=LABEL_LO, epoch=EPOCH - 1, bid="b1"),
        _row(HI, role=ROLE_HI, verdict=LABEL_HI, epoch=EPOCH - 1, bid="b1"),
    ]
    ctx, result = _run_with_markers(markers, bracket_id="b1")
    assert result.success is False
    assert result.data["refusal"]["code"] == REFUSE_BRACKET_EPOCH_SPLIT
    assert ctx.calls == []


# ═══════════════════════════════════════════════════════════════════════
# 3. 跨站点:步 + 不确定度,**永远不是米坐标**(设计 D9b 第 3 条)
# ═══════════════════════════════════════════════════════════════════════

def _assert_no_metre_field(report):
    """报告里不许有**带数的**米制字段。

    判据看的是「键名像米 **且** 值是个数」:一句解释米制为什么不行的话
    (``why_not_metres``)当然可以在,而 ``position_m: 3.2e-7`` 不行 —— 危险的
    从来不是那个词,是那个数。
    """
    metric = [k for k, v in report.items()
              if (k.endswith("_m") or k.endswith("_metres") or k.endswith("_nm"))
              and isinstance(v, (int, float)) and not isinstance(v, bool)]
    assert not metric, (
        f"跨站点报告里出现了带数的米制字段 {metric} —— 步数乘一个标称步长得到的数,"
        f"和真坐标长得一模一样,而它是编的。开环步长随温度差五倍。")
    assert not any(isinstance(v, (list, tuple))
                   and k.endswith(("_m", "_metres", "_nm"))
                   for k, v in report.items()), "米制坐标对也不行"


def test_cross_site_boundary_reports_steps_not_metres():
    report = cross_site_downgrade(3, 4, uncertainty_steps=12,
                                  label_lo=LABEL_LO, label_hi=LABEL_HI)
    _assert_no_metre_field(report)
    assert report["between_sites"] == [3, 4]
    assert report["uncertainty_steps"] == 12
    assert report["localised"] is False, (
        "「观察到」和「定位到」必须是两句话")
    assert "半步长" in report["next_step"], "降级也要给出**可执行**的下一步"
    assert "顺序铺" in report["next_step"], (
        "粗动记账对间距的要求随 gap_moves 增长 —— 跳着铺会被拒")

    # 端到端:二分模式下 allow_coarse_move=True 是拒绝,而且拒绝里带着诚实形态。
    ctx, result = run_bisect(allow_coarse_move=True)
    assert result.success is False
    refusal = result.data["refusal"]
    assert refusal["code"] == REFUSE_CROSS_SITE_BISECT
    assert ctx.calls == []
    _assert_no_metre_field(refusal["cross_site_form"])
    assert "没有定义" in refusal["detail"]


# ═══════════════════════════════════════════════════════════════════════
# 4. 二分落点不走 pick_next_position(设计陷阱 4)
# ═══════════════════════════════════════════════════════════════════════

def test_midpoint_is_not_rejected_by_reuse_overlap():
    """二分收尾时,中点与两个端点帧的重叠**真的**会到 ≥30% —— 那条路会把它丢掉。

    造的是二分**最后一步**的真实几何:区间已经收到一个帧宽,两端各扫过一帧,
    中点的帧与它们各压一半。这不是一个人为构造的极端,而是每一次成功的二分
    必然要走的最后一格。

    两半:先证明 ``pick_next_position`` **确实**会拒掉这个坐标(否则这条测试在钉
    一个不存在的危险),再证明二分自己的落点闸放它过去。
    """
    from mast.io import map_analysis as MA
    from mast.io.exp_map import MapMarker

    # 压电半程收到 150 nm,栅格才分辨得出 5 nm 的帧(每帧至少 8 个格)。
    cfg = replace(BASE_CFG, piezo_half_range_m=150e-9,
                  frame_size_m=ATOMIC_FRAME_M)
    lo, hi = (-ATOMIC_FRAME_M / 2, 0.0), (ATOMIC_FRAME_M / 2, 0.0)
    mid = midpoint(lo, hi)
    scanned = [
        MapMarker(kind="scan", x_m=x, y_m=y, w_m=ATOMIC_FRAME_M,
                  h_m=ATOMIC_FRAME_M, status="done")
        for x, y in (lo, hi)
    ]
    covered, _ = MA.rasterize(scanned, cfg)
    overlap = MA._frame_overlap_frac(covered, MA._grid_axis(cfg),
                                     MA._grid_axis(cfg), mid[0], mid[1],
                                     ATOMIC_FRAME_M / 2)
    assert overlap >= cfg.reuse_overlap_frac, (
        f"中点与两端的重叠只有 {overlap:.0%},没到 {cfg.reuse_overlap_frac:.0%} —— "
        f"这条测试钉的危险不存在了,去看 reuse_overlap_frac 是不是改了")
    picked, left = MA.pick_next_position(
        scanned, cfg, covered=covered, circles=[],
        candidates=[(mid[0], mid[1], 0)])
    assert picked is None and left == 0, (
        "reuse_overlap_frac 没有拒掉这个候选 —— 那这条测试钉的危险不存在了,"
        "去看 pick_next_position 是不是改了语义")

    assert probe_position_problem(mid[0], mid[1], cfg=cfg, circles=[]) is None, (
        "二分的落点闸把中点拒了 —— 「已扫 ≠ 不可用」,二分要的正是回到两个"
        "扫过的地方中间再扫一帧")


def test_bisect_never_calls_the_route_planner():
    """结构闸门:二分这条路上 ``pick_next_position*`` 一次都不许被调到。

    它不是「效率问题」:那条路的用途是**找没扫过的地方**,而二分要的正好相反。
    共用一个函数的后果是二分永远排不出下一个点,而且零报错 —— 它只会说
    「这片表面在当前策略下没有位置了」,听起来像表面用完了。
    """
    from mast.io import map_analysis as MA

    def _boom(*a, **kw):
        raise AssertionError("二分调了 pick_next_position —— 见设计陷阱 4")

    orig1, orig2 = MA.pick_next_position, MA.pick_next_positions
    MA.pick_next_position, MA.pick_next_positions = _boom, _boom
    try:
        _, result = run_bisect()
    finally:
        MA.pick_next_position, MA.pick_next_positions = orig1, orig2
    assert result.success and result.data["bisect"]["state"] == STATE_CONVERGED


def test_a_probe_outside_the_piezo_range_is_refused_not_clamped():
    """落点闸查的是硬边界。**不夹紧** —— 夹紧会把中点悄悄挪到别处。"""
    cfg = replace(BASE_CFG, frame_size_m=ATOMIC_FRAME_M)
    far = cfg.piezo_half_range_m       # 帧的一半会伸出去
    problem = probe_position_problem(far, 0.0, cfg=cfg, circles=[])
    assert problem and "压电" in problem


def test_a_probe_on_a_keep_out_disc_is_refused():
    from mast.io.map_analysis import AvoidCircle

    cfg = replace(BASE_CFG, frame_size_m=ATOMIC_FRAME_M)
    circle = AvoidCircle(0.0, 0.0, 50e-9, "pulse", "扎过针")
    assert probe_position_problem(0.0, 0.0, cfg=cfg, circles=[circle])
    assert probe_position_problem(1e-6, 0.0, cfg=cfg, circles=[circle]) is None


# ═══════════════════════════════════════════════════════════════════════
# 5. 「判不了」的下一个坐标由**算法**给(设计 D9 三级处理)
# ═══════════════════════════════════════════════════════════════════════

def _assert_offsets_are_geometric(positions, *, lo, hi, frame):
    """偏移点必须严格垂直于 lo→hi、幅度是整数个帧宽、左右交替。"""
    mid = midpoint(lo, hi)
    axis = (hi[0] - lo[0], hi[1] - lo[1])
    norm = math.hypot(*axis)
    for k, (x, y) in enumerate(positions, start=1):
        dx, dy = x - mid[0], y - mid[1]
        along = (dx * axis[0] + dy * axis[1]) / norm
        assert abs(along) < 1e-15, (
            f"第 {k} 个偏移点沿轴移动了 {along * 1e9:.3f} nm —— 沿轴偏移等于换了一个"
            f"中点,二分的不变式当场就散了。偏移必须**垂直**。")
        assert math.hypot(dx, dy) == pytest.approx(((k + 1) // 2) * frame), (
            f"第 {k} 个偏移点的幅度不是 {((k + 1) // 2)} 个帧宽")
    sides = [math.copysign(1.0, (x - mid[0]) * (-axis[1]) / norm
                           + (y - mid[1]) * axis[0] / norm)
             for x, y in positions]
    assert sides == [1.0 if k % 2 else -1.0
                     for k in range(1, len(positions) + 1)], "左右没有交替"


def test_undetermined_offset_points_come_from_the_algorithm():
    """三级处理里没有一个数字来自模型(设计 D9)。

    ① 同点重采一帧 ② 沿垂直方向偏移一个帧宽、左右交替 ③ 预算用尽 ⇒ 诚实
    ``ABANDONED(undetermined)``。这条测试同时是一道**结构闸门**:``FakeCtx`` 对
    任何第四个技能名直接抛 —— 「判不了,下一个点扫哪儿」是几何问题,答案由 lo、hi、
    帧宽三个数唯一确定,一旦有东西去问模型要坐标,这里当场红。
    """
    # 纯函数:确定性 + 几何形状。
    p1 = offset_probe(LO, HI, frame_size_m=ATOMIC_FRAME_M, attempt=1)
    assert p1 == offset_probe(LO, HI, frame_size_m=ATOMIC_FRAME_M, attempt=1), (
        "同样的输入给了两个不同的点 —— 那就不是算法给的")
    _assert_offsets_are_geometric(
        [offset_probe(LO, HI, frame_size_m=ATOMIC_FRAME_M, attempt=k)
         for k in (1, 2, 3, 4)], lo=LO, hi=HI, frame=ATOMIC_FRAME_M)

    ctx, result = run_bisect(FakeCtx(verdict_fn=always_undetermined()),
                             offset_budget=2)
    b = result.data["bisect"]
    assert result.success, "诚实的 ABANDONED 是一个答案,不是一次失败"
    assert b["state"] == STATE_ABANDONED and b["end_reason"] == END_UNDETERMINED
    assert b["boundary_m"] is None, "判不了还报了一个畴界坐标"
    assert b["offsets_used"] == 2 and b["iterations"] == 0

    scans = ctx.scan_positions()
    assert scans[0] == LO and scans[1] == HI
    mid = midpoint(LO, HI)
    assert scans[2] == mid and scans[3] == mid, (
        "第二级来得太早 —— 判不了的第一级是**同点重采一帧**,"
        "一帧的噪声不该判死一个位置")
    _assert_offsets_are_geometric(scans[4:], lo=LO, hi=HI, frame=ATOMIC_FRAME_M)
    assert len(scans) == 6

    roles = [p["meta"]["role"] for p in result.data["point_log"]]
    assert roles == [ROLE_LO, ROLE_HI, ROLE_MID, ROLE_MID,
                     ROLE_OFFSET, ROLE_OFFSET]


def test_a_zero_offset_budget_abandons_right_after_the_resample():
    ctx, result = run_bisect(FakeCtx(verdict_fn=always_undetermined()),
                             offset_budget=0)
    assert result.data["bisect"]["end_reason"] == END_UNDETERMINED
    assert ctx.count("ScanAt") == 4, "两个端点 + 中点 + 重采一帧"


def test_a_third_label_is_a_contradiction_not_an_undetermined():
    """中点判成了第三个相 —— 这是**证据矛盾**,不该被归进最近的一簇。

    归簇会二分出一条根本不存在的边界,而且每一步都「成功」。无人值守时走保守分支
    (放弃这个 bracket),不猜,也不半夜叫人。
    """
    def _third(xy, params):
        if abs(xy[0] - LO[0]) < 1e-15:
            return _packet(label=LABEL_LO)
        if abs(xy[0] - HI[0]) < 1e-15:
            return _packet(label=LABEL_HI)
        return _packet(label="gamma")

    _, result = run_bisect(FakeCtx(verdict_fn=_third))
    b = result.data["bisect"]
    assert b["state"] == STATE_ABANDONED
    assert b["end_reason"] == SDB.END_THIRD_LABEL
    assert b["boundary_m"] is None
    assert "证据矛盾" in b["note"] and "由人看" in b["note"]


def test_an_unreachable_probe_falls_back_to_the_offsets_not_to_a_guess():
    """中点压在避让圈上 ⇒ 走同一条几何降级路,而且记的是「针尖没去」。

    「采不到」和「采到了但判不了」的下一步一样(换偏移点),但**记录必须分开**:
    前者针尖根本没去过,把它记成一次判定就是往实验记录里写没发生过的事。
    """
    from mast.io.map_analysis import AvoidCircle

    mid = midpoint(LO, HI)
    ctx = FakeCtx()
    orig = SDB.SearchDomainBoundary._prepare_bisect

    def _with_a_crater(self, context, params, out):
        orig(self, context, params, out)
        if not out.get("refusal"):
            out["avoid_circles"] = [AvoidCircle(mid[0], mid[1], 20e-9, "pulse",
                                                "扎过针")]
    SDB.SearchDomainBoundary._prepare_bisect = _with_a_crater
    try:
        _, result = run_bisect(ctx, offset_budget=1)
    finally:
        SDB.SearchDomainBoundary._prepare_bisect = orig

    log = result.data["point_log"]
    unreachable = [p for p in log if p["status"] == POINT_UNREACHABLE]
    assert unreachable, "落在避让圈上的中点没有被记下来"
    assert all(p not in result.data["points"] for p in unreachable), (
        "针尖没去过的位置不许在地图上留足迹")
    assert "避让区" in (unreachable[0]["error"] or "")
    assert mid not in ctx.scan_positions(), "对着一个避让圈发了扫描"


# ═══════════════════════════════════════════════════════════════════════
# 6. 状态从 marker 重建,不存游标(设计 D9 末条)
# ═══════════════════════════════════════════════════════════════════════

def _row(xy, *, role, verdict, epoch=EPOCH, bid="b1", ref="v001"):
    """库里那一行的形状(``storage.get_markers`` 给的 dict)。"""
    return {
        "kind": "scan", "x_m": xy[0], "y_m": xy[1],
        "w_m": ATOMIC_FRAME_M, "h_m": ATOMIC_FRAME_M, "coord_epoch": epoch,
        "meta": {"bracket_id": bid, "role": role, "verdict": verdict,
                 "reference_version": ref, "domain_search_id": "s1"},
    }


def _run_with_markers(rows, **params):
    """把 ``load_markers`` 换成给定的一批,再跑一次二分(= 重启之后那一次)。

    换的是 ``MapMarker`` 对象,不是库里那一行的 dict —— 真的 ``load_markers``
    给的就是对象,而下游的 ``build_avoid_circles`` 只吃对象。用 dict 顶替会让这条
    测试在一条现实中不存在的路上通过。
    """
    from mast.core import map_scope as MS
    from mast.io.exp_map import markers_from_rows

    markers = markers_from_rows(list(rows))
    orig = MS.load_markers
    MS.load_markers = lambda **kw: (
        list(markers) if kw.get("all_epochs")
        else [m for m in markers if m.coord_epoch == EPOCH], EPOCH, True)
    try:
        return run_bisect(lo_x_m=None, lo_y_m=None, hi_x_m=None, hi_y_m=None,
                          **params)
    finally:
        MS.load_markers = orig


def test_rebuild_replays_the_interval_from_the_records():
    """重放而不是读一个存下来的区间 —— 区间**由记录本身定义**。"""
    mid1 = midpoint(LO, HI)
    rows = [
        _row(LO, role=ROLE_LO, verdict=LABEL_LO),
        _row(HI, role=ROLE_HI, verdict=LABEL_HI),
        _row(mid1, role=ROLE_MID, verdict=LABEL_LO),      # 中点像 lo ⇒ lo 前移
    ]
    built = rebuild_brackets(rows, current_epoch=EPOCH)["b1"]
    assert built["problems"] == []
    assert built["lo_m"] == list(mid1) and built["hi_m"] == list(HI)
    assert built["iterations"] == 1
    assert built["gap_m"] == pytest.approx(bracket_gap_m(mid1, HI))
    assert built["lo_label"] == LABEL_LO and built["hi_label"] == LABEL_HI


def test_rebuild_says_so_when_the_recorded_midpoint_does_not_match():
    """记录里的中点与重算的对不上 ⇒ 说出来,不静默采信任何一边。"""
    rows = [
        _row(LO, role=ROLE_LO, verdict=LABEL_LO),
        _row(HI, role=ROLE_HI, verdict=LABEL_HI),
        _row((123e-9, 0.0), role=ROLE_MID, verdict=LABEL_LO),
    ]
    built = rebuild_brackets(rows, current_epoch=EPOCH)["b1"]
    assert built["replay_notes"], "中点规则被改过,而重建一声不吭"
    assert "对不上" in built["replay_notes"][0]


def test_a_bracket_whose_ends_are_the_same_phase_is_not_a_pair():
    rows = [_row(LO, role=ROLE_LO, verdict=LABEL_LO),
            _row(HI, role=ROLE_HI, verdict=LABEL_LO)]
    built = rebuild_brackets(rows, current_epoch=EPOCH)["b1"]
    assert built["end_reason"] == END_NOT_A_PAIR
    ctx, result = _run_with_markers(rows, bracket_id="b1")
    assert result.success is False
    assert result.data["refusal"]["code"] == REFUSE_BRACKET_NOT_A_PAIR
    assert ctx.calls == []


def test_an_undetermined_endpoint_is_not_a_pair_either():
    """没有参照系时每个点都是 undetermined ⇒ 永远凑不出一对。"""
    rows = [_row(LO, role=ROLE_LO, verdict="undetermined"),
            _row(HI, role=ROLE_HI, verdict=LABEL_HI)]
    built = rebuild_brackets(rows, current_epoch=EPOCH)["b1"]
    assert built["end_reason"] == END_NOT_A_PAIR
    assert "建参照系" in built["problems"][0]


def _assert_resumed_from_the_narrowed_interval(ctx, result, *, lo, hi):
    """重启之后的第一帧必须落在**收窄后**区间的中点上。"""
    first = ctx.scan_positions()[0]
    want = midpoint(lo, hi)
    assert first == pytest.approx(want), (
        f"重启之后第一帧扫在 {first[0] * 1e9:.1f} nm,而收窄后区间的中点是 "
        f"{want[0] * 1e9:.1f} nm —— 状态没有从 marker 重建,前面几帧白扫了")
    assert result.data["bracket_source"] == "markers"


def test_state_rebuilds_from_markers_after_restart(tmp_path):
    """**重启之后从 marker 把 bracket 接着跑下去**(设计 D9 末条)。

    不存游标的理由:composite 的 sidecar 按 ``(name, run_id)`` 分键,而重启就是一个
    新 run_id —— 靠 sidecar 续跑等于没有续跑。这条测试走的是**真的**记录路径
    (技能 → recorder → sqlite → 读回 → 重建),因为记录链整条都是 fire-and-forget:
    字段名拼错不会报错,只会表现成「重建出来的区间一直是原始区间」。
    """
    from mast.core import map_scope as MS
    from mast.core.runtime import CoreRuntime
    from mast.io.exp_map import markers_from_rows
    from mast.logging.storage import ExperimentStorage

    # ① 第一次运行,只允许收窄 3 次就停(相当于跑到一半被打断)。
    ctx1, first = run_bisect(max_iterations=3)
    b1 = first.data["bisect"]
    assert b1["state"] == STATE_ABANDONED and b1["iterations"] == 3
    lo1, hi1 = tuple(b1["lo_m"]), tuple(b1["hi_m"])
    assert (lo1, hi1) != (LO, HI), "区间根本没动,后面验不了任何东西"

    # ② 走真的 recorder 落库。先垫两条 coarse_move,让 log_marker 盖的章 = EPOCH。
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    for _ in range(EPOCH):
        st.log_marker(kind="coarse_move", x_m=0.0, y_m=0.0, experiment_id=eid)
    assert st.current_epoch(eid) == EPOCH
    rt = SimpleNamespace(
        _storage=st, _registry=None, _state=None,
        _experiment_log=SimpleNamespace(current_experiment_id=eid,
                                        current_sample_id=None))
    CoreRuntime._record_map_marker(rt, {
        "skill": "SearchDomainBoundary", "success": True,
        "params": {"mode": MODE_BISECT}, "data": first.data})
    rows = st.get_markers(experiment_id=eid)
    bid = first.data["bracket_id"]
    rebuilt = rebuild_brackets(markers_from_rows(rows), current_epoch=EPOCH)
    assert bid in rebuilt, (
        f"库里重建不出 bracket {bid!r} —— marker 的 meta.bracket_id / meta.role "
        f"没落到库里。记录链是 fire-and-forget:拼错字段名不会报错。")
    assert rebuilt[bid]["problems"] == []
    assert tuple(rebuilt[bid]["lo_m"]) == pytest.approx(lo1)
    assert tuple(rebuilt[bid]["hi_m"]) == pytest.approx(hi1)

    # ③ 「重启」:全新的技能实例,只给 bracket_id,状态从库里读。
    orig = MS.load_markers
    MS.load_markers = lambda **kw: (markers_from_rows(rows), EPOCH, True)
    try:
        ctx2, second = run_bisect(bracket_id=bid, lo_x_m=None, lo_y_m=None,
                                  hi_x_m=None, hi_y_m=None)
    finally:
        MS.load_markers = orig
    assert second.success
    _assert_resumed_from_the_narrowed_interval(ctx2, second, lo=lo1, hi=hi1)
    assert second.data["bisect"]["state"] == STATE_CONVERGED
    assert second.data["bisect"]["iterations"] == 7, (
        "收窄次数是**累计**的(重放出来的 3 次 + 这一轮的 4 次)—— 重启把计数清零"
        "会让 max_iterations 变成「每次重启都重新发一份预算」")
    assert ctx2.count("ScanAt") == 4, (
        f"重启之后又扫了 {ctx2.count('ScanAt')} 帧 —— 接着跑只该剩 4 次收窄,"
        f"从头来是 7 次;两个端点也不该重扫,它们的判定已经在记录里")


def test_an_unknown_bracket_id_is_refused_with_the_known_ones():
    rows = [_row(LO, role=ROLE_LO, verdict=LABEL_LO, bid="b1"),
            _row(HI, role=ROLE_HI, verdict=LABEL_HI, bid="b1")]
    ctx, result = _run_with_markers(rows, bracket_id="b9")
    assert result.success is False
    assert result.data["refusal"]["code"] == REFUSE_BRACKET_NOT_FOUND
    assert "b1" in result.data["refusal"]["detail"], "拒绝要说得出有哪些可选"
    assert ctx.calls == []


def test_survey_seed_markers_do_not_become_brackets():
    """普查的种子点 ``bracket_id=None`` —— 不许被重建成一个 bracket。"""
    seed = _row(LO, role="seed", verdict="undetermined", bid=None)
    assert rebuild_brackets([seed], current_epoch=EPOCH) == {}


# ═══════════════════════════════════════════════════════════════════════
# 7. 参数与前置(拒绝 = 这件事没做成)
# ═══════════════════════════════════════════════════════════════════════

def test_bisect_without_a_calibrated_reference_is_refused():
    """没有参照系时每个点都是 undetermined ⇒ 二分开不了头。**明说**这一条。

    让它表现成「这一对不成立」会把人送去换点,而真正该做的是先建参照系。
    """
    from mast.vision import domain_reference as DR

    orig = DR.load_reference
    DR.load_reference = lambda v=None, *, sample=None: None
    try:
        ctx, result = run_bisect()
    finally:
        DR.load_reference = orig
    assert result.success is False
    assert result.data["refusal"]["code"] == REFUSE_NO_REFERENCE
    assert ctx.calls == []
    assert "未标定" in result.data["refusal"]["detail"]


def test_neither_a_bracket_id_nor_coordinates_is_refused():
    ctx, result = run_bisect(lo_x_m=None, lo_y_m=None, hi_x_m=None, hi_y_m=None)
    assert result.success is False
    assert result.data["refusal"]["code"] == REFUSE_NO_BRACKET
    assert ctx.calls == []


def test_half_a_coordinate_pair_is_refused_not_completed():
    ctx, result = run_bisect(hi_x_m=None, hi_y_m=None)
    assert result.success is False
    assert result.data["refusal"]["code"] == REFUSE_NO_BRACKET
    assert "编一个点" in result.data["refusal"]["detail"]


def test_endpoints_closer_than_the_tolerance_are_refused():
    ctx, result = run_bisect(hi_x_m=LO[0] + 1e-9, hi_y_m=LO[1])
    assert result.success is False
    assert result.data["refusal"]["code"] == REFUSE_BRACKET_NOT_A_PAIR
    assert ctx.calls == []


def test_two_endpoints_of_the_same_phase_abandon_instead_of_bisecting():
    """显式坐标的两端先各扫一帧;判出来是同一个相 ⇒ 诚实放弃,不硬找。"""
    ctx, result = run_bisect(FakeCtx(verdict_fn=lambda xy, p: _packet(label=LABEL_LO)))
    b = result.data["bisect"]
    assert result.success
    assert b["state"] == STATE_ABANDONED and b["end_reason"] == END_NOT_A_PAIR
    assert ctx.count("ScanAt") == 2, "两端就判出来了,还接着二分"
    assert b["boundary_m"] is None


def test_dry_run_issues_no_scan_and_admits_what_it_cannot_know():
    ctx, result = run_bisect(dry_run=True)
    assert result.success and ctx.calls == []
    assert result.data["dry_run"] is True
    assert result.data["planned_probes"][0]["role"] == ROLE_MID
    assert result.data["planned_probes"][0]["x_m"] == pytest.approx(0.0)
    assert result.data["iterations_upper_bound"] == 7
    assert result.data["points"] == [], "排练不许在地图上留足迹"
    assert [p["status"] for p in result.data["point_log"]] == [POINT_PLANNED]
    assert "只算得出**第一个**落点" in (result.summary or ""), (
        "排练报一条完整路径出来会让人以为整条路已经验过了")


def test_bisect_points_carry_the_bracket_and_role_in_their_meta():
    """marker 的 ``meta`` 是二分状态唯一的落脚点(设计 §3.4)。"""
    _, result = run_bisect(max_iterations=2)
    log = result.data["point_log"]
    assert [p["meta"]["role"] for p in log[:2]] == [ROLE_LO, ROLE_HI]
    for p in log:
        assert p["meta"]["bracket_id"], "二分的点没有 bracket_id,重建不出来"
        assert p["meta"]["coord_epoch"] == EPOCH
        assert p["meta"]["point_index"] == p["index"]
    assert all(p["status"] == POINT_ASSESSED for p in log)
    assert len({p["index"] for p in log}) == len(log), "编号撞了"


def test_a_scan_failure_mid_bisect_is_not_reported_as_undetermined():
    """扫不成 ⇒ 走同一条几何降级路,但**终态理由不一样**。

    「一帧都没采成」的下一步是查针尖/反馈;「采到了但判不了」的下一步是看
    verdict_reason(尺度门、帧角、原子相)。共用一个码会把人送错方向 —— 而两条路
    看到的现象都是「二分没找到畴界」。
    """
    ctx, result = run_bisect(FakeCtx(fail_scan_at={3, 4, 5}), offset_budget=3)
    b = result.data["bisect"]
    assert b["state"] == STATE_ABANDONED
    assert b["end_reason"] == SDB.END_NOT_SAMPLED, (
        f"没采成被记成了 {b['end_reason']}")
    assert b["end_reason"] != END_UNDETERMINED
    assert "查针尖" in (b["note"] or ""), b["note"]
    assert ctx.count("ScanAt") <= 5


# ═══════════════════════════════════════════════════════════════════════
# 8. 变异验证 —— 先证明变异已应用,再证明测试红了
# ═══════════════════════════════════════════════════════════════════════

def test_mutation_dropping_the_epoch_check_turns_the_negative_test_red():
    """**去掉 bracket 的代次校验** ⇒ 最重要的那条负向测试必须红(设计 §5.6)。

    这是整条 S3 里最该被变异证明的一处:它的失败模式不是报错,是二分继续用两个已经
    指向另一片表面的坐标,一路「成功」地收敛出一个精确到纳米的假畴界。
    """
    from mast.core import coord_epoch as CE

    orig = SearchDomainBoundary.__dict__["_epoch_ok"]     # 拿 staticmethod 本体
    SearchDomainBoundary._epoch_ok = staticmethod(lambda prep: None)
    seen = {"n": 0}

    def _epoch():
        seen["n"] += 1
        return EPOCH if seen["n"] <= 4 else EPOCH + 1

    CE.read_current_epoch = _epoch
    try:
        assert SearchDomainBoundary._epoch_ok({}) is None      # 变异已应用
        ctx, result = run_bisect()
        assert result.success, "变异之后它照样跑完,而且每一帧都「成功」了"
        assert result.data["bisect"]["state"] == STATE_CONVERGED, (
            "变异之后它收敛出了一个畴界 —— 用的是粗动之前的坐标")
        with pytest.raises(AssertionError):
            _assert_bracket_was_invalidated(ctx, result, scans_before_bump=3)
    finally:
        SearchDomainBoundary._epoch_ok = orig
        CE.read_current_epoch = lambda: EPOCH


def test_mutation_dropping_the_endpoint_epoch_check_turns_its_test_red():
    """把 :func:`check_bracket_epochs` 变成永远放行 ⇒ 跨代次的一对端点被二分。"""
    orig = SDB.check_bracket_epochs
    SDB.check_bracket_epochs = lambda lo, hi, current: None
    try:
        assert SDB.check_bracket_epochs(2, 3, 3) is None       # 变异已应用
        markers = [
            _row(LO, role=ROLE_LO, verdict=LABEL_LO, epoch=EPOCH - 1, bid="b1"),
            _row(HI, role=ROLE_HI, verdict=LABEL_HI, epoch=EPOCH, bid="b1"),
        ]
        ctx, result = _run_with_markers(markers, bracket_id="b1")
        assert result.success, "变异之后它照样二分了一对跨代次的端点"
        assert ctx.count("ScanAt") > 0
    finally:
        SDB.check_bracket_epochs = orig


def test_mutation_not_clamping_the_tolerance_turns_its_test_red():
    """把容差的夹紧去掉 ⇒ 压电分辨率那个数被当真,二分空转到预算耗尽。"""
    orig = SDB.locate_tolerance
    SDB.locate_tolerance = lambda frame, want=None: (
        (float(want), False, "变异:照单全收") if want else orig(frame, want))
    try:
        assert SDB.locate_tolerance(ATOMIC_FRAME_M, 1e-12)[0] == 1e-12  # 变异已应用
        _, result = run_bisect(locate_tolerance_m=1e-12)
        with pytest.raises(AssertionError):
            _assert_tolerance_is_the_frame(result.data)
        assert result.data["bisect"]["iterations"] == 20, (
            "多扫的这 13 帧就是「拿压电分辨率当下限」的代价 —— 每一帧都在比较"
            "同一片表面和它自己")
    finally:
        SDB.locate_tolerance = orig


def test_mutation_offsetting_along_the_axis_turns_its_test_red():
    """把偏移方向从**垂直**改成**沿轴** ⇒ 沿轴位置被改掉,不变式散掉。"""
    orig = SDB.offset_probe

    def _along(lo, hi, *, frame_size_m, attempt):
        mx, my = midpoint(lo, hi)
        dx, dy = hi[0] - lo[0], hi[1] - lo[1]
        n = math.hypot(dx, dy) or 1.0
        k = max(1, int(attempt))
        mag = ((k + 1) // 2) * frame_size_m * (1.0 if k % 2 else -1.0)
        return (mx + mag * dx / n, my + mag * dy / n)

    SDB.offset_probe = _along
    try:
        moved = SDB.offset_probe(LO, HI, frame_size_m=ATOMIC_FRAME_M, attempt=1)
        assert moved[0] != midpoint(LO, HI)[0]                 # 变异已应用
        ctx, _ = run_bisect(FakeCtx(verdict_fn=always_undetermined()),
                            offset_budget=2)
        with pytest.raises(AssertionError):
            _assert_offsets_are_geometric(ctx.scan_positions()[4:], lo=LO, hi=HI,
                                          frame=ATOMIC_FRAME_M)
    finally:
        SDB.offset_probe = orig


def test_mutation_applying_reuse_overlap_to_the_probe_turns_its_test_red():
    """把 ``reuse_overlap_frac`` 那条规则搬到二分的落点闸上(设计陷阱 4 的形状)。

    后果不是报错:中点每次都被判成「不可用」,二分走完偏移预算然后诚实地
    ``ABANDONED`` —— 报告说的是「判不了」,而真正发生的是**一帧都没扫成**。
    """
    orig = SDB.probe_position_problem
    SDB.probe_position_problem = lambda x, y, *, cfg, circles=(): (
        "变异:这个落点与已扫区重叠 ≥30%,按 reuse_overlap_frac 拒掉")
    try:
        assert SDB.probe_position_problem(0.0, 0.0, cfg=BASE_CFG)  # 变异已应用
        ctx, result = run_bisect(offset_budget=1)
        assert result.data["bisect"]["state"] == STATE_ABANDONED
        assert result.data["bisect"]["boundary_m"] is None
        assert ctx.count("ScanAt") == 2, (
            "变异之后除了两个端点一帧都没扫成 —— 而报告只会说「判不了」")
    finally:
        SDB.probe_position_problem = orig


def test_mutation_dropping_role_and_bracket_id_from_meta_turns_rebuild_red():
    """把 ``role`` / ``bracket_id`` 从 marker meta 里拿掉 ⇒ 重建不出任何东西。

    失败模式是**静默**的:技能自己的返回体里那两个字段一直好好的,只有从库里读回来
    的时候才发现重建结果是空的 —— 而「空的」在下一次运行里表现成「从头开始」,
    看上去完全正常。
    """
    orig = SearchDomainBoundary._point_record

    def _stripped(self, *a, **kw):
        rec = orig(self, *a, **kw)
        rec["meta"].pop("role", None)
        rec["meta"].pop("bracket_id", None)
        return rec

    SearchDomainBoundary._point_record = _stripped
    try:
        _, result = run_bisect(max_iterations=2)
        assert "role" not in result.data["point_log"][0]["meta"]   # 变异已应用
        rows = [{"kind": "scan", "x_m": p["center_x_m"], "y_m": p["center_y_m"],
                 "coord_epoch": EPOCH, "meta": p["meta"]}
                for p in result.data["points"]]
        assert rebuild_brackets(rows, current_epoch=EPOCH) == {}, (
            "meta 里没有 bracket_id 了,却还重建出了 bracket —— "
            "那说明重建在读别的东西")
    finally:
        SearchDomainBoundary._point_record = orig


# ═══════════════════════════════════════════════════════════════════════
# 9. 状态机本身(纯逻辑,零硬件)
# ═══════════════════════════════════════════════════════════════════════

def _machine(**kw):
    kw.setdefault("bracket_id", "b1")
    kw.setdefault("lo", LO)
    kw.setdefault("hi", HI)
    kw.setdefault("lo_label", LABEL_LO)
    kw.setdefault("hi_label", LABEL_HI)
    kw.setdefault("tolerance_m", ATOMIC_FRAME_M)
    kw.setdefault("frame_size_m", ATOMIC_FRAME_M)
    return BisectMachine(**kw)


def test_the_machine_narrows_on_the_midpoint_not_on_the_offset_coordinate():
    """偏移点判出来的 label 收窄的是**中点**那一刀。

    拿偏移点的坐标当新端点,区间会歪掉 —— 而且歪得看不出来:它照样在收敛,只是
    收敛到一条不是畴界的线上。
    """
    m = _machine()
    p = m.plan_probe()
    m.record(p, verdict="undetermined")          # 判不了
    p2 = m.plan_probe()
    m.record(p2, verdict="undetermined")         # 同点重采,还是判不了
    p3 = m.plan_probe()
    assert p3.role == ROLE_OFFSET and p3.y_m != 0.0
    m.record(p3, verdict=LABEL_LO, label=LABEL_LO)
    assert m.lo == midpoint(LO, HI), "收窄用的是偏移点的坐标,不是中点"
    assert m.hi == HI and m.iterations == 1


def test_the_machine_is_terminal_only_in_plan_probe():
    """「什么时候停」只有一个真源 —— 两个真源迟早会不一致。"""
    m = _machine(max_iterations=1)
    p = m.plan_probe()
    m.record(p, verdict=LABEL_LO, label=LABEL_LO)
    assert not m.finished, "record 自己判了停"
    assert m.plan_probe() is None and m.state == STATE_ABANDONED
    assert m.end_reason == SDB.END_MAX_ITERATIONS


def test_invalidate_clears_a_boundary_it_had_already_found():
    """作废要连已经算出来的边界一起清掉 —— 那是用旧坐标系说的话。"""
    m = _machine(tolerance_m=1e-6)
    assert m.plan_probe() is None and m.state == STATE_CONVERGED
    assert m.boundary is not None
    m.invalidate("粗动了")
    assert m.state == STATE_INVALIDATED
    assert m.boundary is None and m.boundary_uncertainty_m is None
