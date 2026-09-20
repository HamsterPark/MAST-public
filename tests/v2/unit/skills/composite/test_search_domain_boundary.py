"""``SearchDomainBoundary`` —— 畴界普查(``mode="survey"``)。

S3 畴搜索设计 §5.4/§5.5 里与 survey 有关的那几条。这里盯的**不是**指纹算得对不对
(那是 ``tests/v2/unit/vision/test_domain_phase*.py`` 的活),而是这一层四件最容易
悄悄写错、而且写错了**零报错**的事:

1. 网格用的可用区被中心区压成 500 nm 见方 —— 比一条畴界的间距还小,于是「九个点
   全是同一个畴」变成一句必然的废话,而每一步都「成功」了。
2. 一帧都判不了的尺度下照样把图扫出来,再把「判不了」读成「这里没有畴」。
3. 帧角读不到被折叠成 0° —— 同一个畴在两种帧角下报成两个畴。
4. 没有参照系时凭空出一个 label —— 代码自己发明「这是 A 相」。

跑:
    .venv-v2-py313/Scripts/python.exe -m pytest \
      tests/v2/unit/skills/composite/test_search_domain_boundary.py -q
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

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from mast.core.types import SkillResult  # noqa: E402
from mast.io.map_analysis import AnalysisConfig  # noqa: E402
from mast.skills.composite import search_domain_boundary as SDB  # noqa: E402
from mast.skills.composite.search_domain_boundary import (  # noqa: E402
    ALL_POINT_STATUS,
    CLUSTER_NO_SYMMETRY,
    MODE_BISECT,
    MODE_SURVEY,
    POINT_ASSESSED,
    POINT_EPOCH_STALE,
    POINT_PLANNED,
    POINT_SCAN_FAILED,
    POINT_SKIPPED,
    POINT_STALE_FRAME,
    REFUSE_COARSE_NOT_YET,
    REFUSE_SCALE,
    ROLE_SEED,
    SearchDomainBoundary,
    cluster_two,
    default_grid_pitch_m,
    release_center_zone,
    scale_refusal,
)

FAIL = object()

#: 这台仪器有 XY 粗动 ⇒ ``map_scope`` 会给出中心区 500 nm + 脉冲避让 500 nm 的
#: 那一对。**正是要被解除的那一份配置。**
BASE_CFG = AnalysisConfig(
    piezo_half_range_m=1.5e-6,
    frame_size_m=100e-9,          # 巡览帧 —— 网格要换成原子帧
    strategy="center_first",
    has_xy_coarse_motion=True,
    center_zone_side_m=500e-9,
    pulse_r_m=500e-9,
    point_spacing_factor=1.2,
    ring_width_factor=1.2,
)

#: 解除中心区之后可用的半边长(边缘余量 6%)。
R_EFF = 1.5e-6 * 0.94
#: 不解除中心区时可用的半边长 —— 250 nm。整张网格会被压进这里面。
R_ZONED = 250e-9

ATOMIC_FRAME_M = 5e-9
EPOCH = 3


# ── 假上下文 ─────────────────────────────────────────────────────────────────

class FakeCtx:
    """按技能名派活的假上下文(与 ``test_verify_atomic_resolution`` 同款)。"""

    def __init__(self, script=None):
        self.script = dict(script or {})
        self.calls: list[tuple[str, dict]] = []
        self._n: dict[str, int] = {}
        self.run_id = "test-domain-survey"
        self.state = None

    def run(self, skill_name, params, version=None):
        self.calls.append((skill_name, dict(params)))
        n = self._n.get(skill_name, 0)
        self._n[skill_name] = n + 1
        fn = self.script.get(skill_name, {})
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


def frame_at(prefix="/tmp/survey"):
    """``GetLatestScanFile``:每次给一个**新**路径(真机上每帧一个文件)。"""
    return lambda params, n: {"path": f"{prefix}_{n}.sxm", "age_s": 3.0}


def assessed(*, verdict="undetermined", reason="no_reference", label=None,
             fingerprint=((30.0, 0.25, 1.0), (90.0, 0.25, 0.8)),
             angle=0.0):
    """``AssessDomainPhase`` 的回包(键与那个技能的 ``data`` 一致)。"""
    def _f(params, n):
        return {
            "scan_path": params.get("scan_path"),
            "verdict": verdict, "label": label, "verdict_reason": reason,
            "next_step": "…", "fingerprint": [list(p) for p in fingerprint],
            "scan_angle_deg": angle, "n_peaks": len(fingerprint),
            "nm_per_px": 0.0098, "scale": "full",
        }
    return _f


def assessed_varying(specs):
    """按第 n 次调用给不同的回包 —— 用来造「两簇」。"""
    def _f(params, n):
        kw = dict(specs[min(n, len(specs) - 1)])
        return assessed(**kw)(params, n)
    return _f


DEFAULT_SCRIPT = {
    "ScanAt": {"ok": True},
    "GetLatestScanFile": frame_at(),
    "AssessDomainPhase": assessed(),
}


# ── 接线:把这一层与实验记录 / 仪器档案隔开 ──────────────────────────────────

@pytest.fixture(autouse=True)
def wired(tmp_path, monkeypatch):
    """`map_scope` / 参照系 / 代次全部换成显式替身。

    真存储与真仪器档案都不参与:测试要陈述的前提必须写在测试里,而不是碰巧等于
    这台开发机的档案值。sidecar 也改到 ``tmp_path`` —— 测试不往用户目录写东西。
    """
    from mast.core import coord_epoch as CE
    from mast.core import map_scope as MS
    from mast.skills.composite import graph_executor as GE
    from mast.vision import domain_reference as DR

    monkeypatch.setattr(
        MS, "analysis_config",
        lambda state=None, *, safety=None, frame_size_m=None: BASE_CFG)
    monkeypatch.setattr(MS, "load_markers", lambda **kw: ([], EPOCH, True))
    monkeypatch.setattr(CE, "read_current_epoch", lambda: EPOCH)
    monkeypatch.setattr(DR, "load_reference", lambda v=None, *, sample=None: None)
    monkeypatch.setattr(GE, "_sidecar_dir", lambda: tmp_path)
    yield


def run_survey(script=None, **params):
    ctx = FakeCtx(script if script is not None else DEFAULT_SCRIPT)
    p = {"mode": MODE_SURVEY, "frame_size_m": ATOMIC_FRAME_M,
         "pixels": 512, "max_points": 4}
    p.update(params)
    result = SearchDomainBoundary().execute(ctx, p)
    return ctx, result


# ═══════════════════════════════════════════════════════════════════════
# 1. 网格:中心区必须解除(设计陷阱 2),脉冲半径一个字都不许动(陷阱 3)
# ═══════════════════════════════════════════════════════════════════════

def _assert_centre_zone_is_released(cfg):
    assert cfg.center_zone_side_m is None, (
        "中心区没解除 —— 网格会被压进 500 nm 见方,比一条畴界的间距还小")
    assert cfg.effective_half_range_m == pytest.approx(R_EFF, rel=1e-9), (
        f"可用半边长该是 {R_EFF * 1e9:.0f} nm(整个压电范围),"
        f"不是中心区那 {R_ZONED * 1e9:.0f} nm")


def test_center_zone_is_released_for_the_grid():
    cfg, why = release_center_zone(BASE_CFG, frame_size_m=ATOMIC_FRAME_M)
    _assert_centre_zone_is_released(cfg)
    assert cfg.frame_size_m == ATOMIC_FRAME_M, "网格铺的是原子帧,不是巡览帧"
    assert "中心区" in why and "微米" in why, "为什么解除中心区必须写进理由句"
    assert "脉冲避让半径" in why, "还要说清楚**没有**顺手动脉冲半径"


def test_pulse_radius_is_not_touched_when_the_centre_zone_is_released():
    """陷阱 3:那两个 500 nm 是刻意配成一对的,只能解一边。

    反向改(把脉冲半径调小)会把「一发就被迫换区」那条设计拆散,而且方向是更不
    保守的那一边 —— 避让圈让网格少几个点是对的代价。
    """
    cfg, _ = release_center_zone(BASE_CFG, frame_size_m=ATOMIC_FRAME_M)
    for field in ("pulse_r_m", "crash_r_m", "tip_shape_r_m", "approach_r_m",
                  "approach_damages", "reuse_overlap_frac", "edge_margin_frac"):
        assert getattr(cfg, field) == getattr(BASE_CFG, field), (
            f"{field} 被顺手改了 —— 解除中心区只该动中心区")


def test_default_pitch_comes_from_the_released_half_range():
    """兜底间距 = 可用半径 / 4。顺序承重:先解除中心区,再拿半径算。"""
    cfg, _ = release_center_zone(BASE_CFG, frame_size_m=ATOMIC_FRAME_M)
    pitch = cfg.frame_size_m * cfg.point_spacing_factor
    assert pitch == pytest.approx(R_EFF / 4, rel=1e-9)
    assert pitch > R_ZONED, "兜底间距比中心区还小 —— 说明半径是在解除之前算的"
    assert default_grid_pitch_m(BASE_CFG) == pytest.approx(R_ZONED / 4), (
        "这是反例:拿没解除的配置去算,兜底间距只有 62 nm")


def test_explicit_pitch_wins_over_the_fallback():
    cfg, why = release_center_zone(BASE_CFG, frame_size_m=ATOMIC_FRAME_M,
                                   grid_pitch_m=800e-9)
    assert cfg.frame_size_m * cfg.point_spacing_factor == pytest.approx(800e-9)
    assert cfg.ring_width_factor == cfg.point_spacing_factor, (
        "只拉开环上的点距而不拉开环间距,只是把「挨着排」换个方向")
    assert "800 nm" in why


def _assert_grid_reaches_beyond_the_centre_zone(result):
    pts = result.data["planned_positions"]
    reach = max(max(abs(p["x_m"]), abs(p["y_m"])) for p in pts)
    assert reach > R_ZONED, (
        f"整张网格都落在 ±{R_ZONED * 1e9:.0f} nm 里(最远 {reach * 1e9:.0f} nm)"
        f" —— 中心区没有真的被解除,而每一步都会「成功」")


def test_survey_grid_reaches_beyond_the_centre_zone():
    """端到端:这是「解除中心区」唯一能被观察到的后果。"""
    _, result = run_survey(dry_run=True, max_points=9)
    assert result.success
    _assert_grid_reaches_beyond_the_centre_zone(result)


# ═══════════════════════════════════════════════════════════════════════
# 2. 尺度预检:判不了的图一帧都不发(设计陷阱 10)
# ═══════════════════════════════════════════════════════════════════════

def test_scale_refusal_names_two_computable_ways_out():
    assert scale_refusal(ATOMIC_FRAME_M, 512) is None, "5 nm / 512 px 是满权重档"
    ref = scale_refusal(50e-9, 256)
    assert ref is not None and ref["code"] == REFUSE_SCALE
    assert ref["nm_per_px"] == pytest.approx(50.0 / 256, rel=1e-6)
    assert len(ref["alternatives"]) == 2, "两条**算得出来**的路,不是「建议调整参数」"
    assert ref["frame_size_m"] == 50e-9 and ref["pixels"] == 256, (
        "原样回传才证明这是一次拒绝,不是一次悄悄的缩帧")


def test_scale_precheck_refuses_before_any_scan_is_issued():
    ctx, result = run_survey(frame_size_m=50e-9, pixels=256)
    assert result.success is False
    assert REFUSE_SCALE in (result.error or "")
    assert ctx.count("ScanAt") == 0, (
        "扫了一张判不了的图 —— 花的不只是机时,流程还会把「判不了」读成「这里没有畴」")
    assert result.data["refusal"]["code"] == REFUSE_SCALE


def test_the_transition_band_is_refused_too():
    """0.02–0.05 nm/px 是降级测量,不是满权重档 —— 普查的每一帧都得是满权重。"""
    ref = scale_refusal(10e-9, 256)          # 0.039 nm/px
    assert ref is not None and ref["scale"] == "reduced"


# ═══════════════════════════════════════════════════════════════════════
# 3. 「排在下一轮」的那条必须显式拒绝,不许悄悄降级
#
# ⚠️ 这里曾经还有一条 ``test_bisect_mode_is_refused_not_silently_downgraded``:
#    二分那时还没有实现,拒绝就是当时正确的行为。2026-08-15 二分落地之后那条
#    拒绝没有了,测试也随之删掉 —— 它盯的是一个已经不存在的状态。二分自己的
#    测试在 ``test_search_domain_boundary_bisect.py``,那里连 ``MODE_BISECT``
#    的端到端行为一起盯。
# ═══════════════════════════════════════════════════════════════════════

def test_bisect_mode_no_longer_refuses_but_still_needs_a_reference():
    """二分不再是「没实现」,但没有标定过的参照系照样开不了头。

    区别是承重的:「这一版没有」的下一步是等下一版,「没有参照系」的下一步是
    **现在就去建一个**(先跑 survey、人确认两簇)。同一个拒绝码盖住两件事,
    人只会照着错的那一条走。
    """
    ctx, result = run_survey(mode=MODE_BISECT)
    assert result.success is False
    code = result.data["refusal"]["code"]
    assert code == SDB.REFUSE_NO_REFERENCE, f"拒绝码变成了 {code}"
    assert ctx.calls == [], "拒绝了还发命令"
    assert MODE_SURVEY in str(result.data["refusal"]["alternatives"])


def test_cross_site_survey_is_refused_when_asked_for():
    """跨站点的输出形态是「站点 k 与 k+1 之间 ±N 步」,不是米坐标 —— 另一件事。"""
    ctx, result = run_survey(allow_coarse_move=True)
    assert result.success is False
    assert result.data["refusal"]["code"] == REFUSE_COARSE_NOT_YET
    assert ctx.calls == []


def test_unknown_mode_is_refused():
    _, result = run_survey(mode="scan_everything")
    assert result.success is False
    assert result.data["refusal"]["alternatives"], "拒绝必须说得出能做什么"


# ═══════════════════════════════════════════════════════════════════════
# 4. dry_run:完整状态机,零扫描(照 RelocateCoarseXY 的先例)
# ═══════════════════════════════════════════════════════════════════════

def test_dry_run_issues_no_scan():
    ctx, result = run_survey(dry_run=True, max_points=6)
    assert result.success and ctx.calls == [], "dry_run 发了命令"
    assert result.data["dry_run"] is True
    assert len(result.data["point_log"]) == 6
    assert {p["status"] for p in result.data["point_log"]} == {POINT_PLANNED}
    # 排练也要把该算的都算完 —— 否则它验不了任何东西。
    assert result.data["nm_per_px"] == pytest.approx(ATOMIC_FRAME_M * 1e9 / 512)
    assert result.data["center_zone_reason"]
    assert result.data["coord_epoch"] == EPOCH
    assert result.data["estimated_total_s"] >= 0.0


def test_a_rehearsal_leaves_no_footprint_on_the_map():
    """``points`` 是地图的素材(recorder 按它落 marker)—— 排练一个都不许进。

    一次没发生过的扫描画在实验记录的地图上,后果不只是难看:找干净地方的那条
    路径靠它避开用过的位置,一个凭空的足迹会让流程永远绕开一块好表面。
    """
    _, result = run_survey(dry_run=True, max_points=6)
    assert result.data["points"] == []
    assert result.data["points_planned"] == 6 and result.data["points_visited"] == 0
    assert "不留任何足迹" in (result.summary or "")


# ═══════════════════════════════════════════════════════════════════════
# 5. 没有参照系 ⇒ 全 no_reference,一个 label 都不许出(设计 D3 / 验收 R6)
# ═══════════════════════════════════════════════════════════════════════

def _assert_no_label_without_a_reference(result):
    assert result.success, "全部 undetermined(no_reference) 是正确产物,不是失败"
    data = result.data
    assert data["labels_emitted"] == [], (
        f"没有标定过的参照系却出了 label {data['labels_emitted']} —— "
        f"有硬编码的畴名漏进来了")
    assert data["verdict_counts"] == {"undetermined": len(data["points"])}
    assert set(data["undetermined_reasons"]) == {"no_reference"}
    for p in data["points"]:
        assert p["label_out"] is None
        assert p["meta"]["verdict"] == "undetermined"
        assert p["meta"]["reference_version"] is None
    assert "no_reference" in (result.summary or "")


def test_no_reference_never_yields_a_label():
    _, result = run_survey(max_points=3)
    _assert_no_label_without_a_reference(result)
    assert all(p["status"] == POINT_ASSESSED for p in result.data["points"])
    assert all(p["fingerprint"] for p in result.data["points"]), (
        "拒判的时候指纹照报 —— 那正是这一轮的产物")


# ═══════════════════════════════════════════════════════════════════════
# 6. 帧角读不到就是读不到(设计 D4 / 陷阱 1)
# ═══════════════════════════════════════════════════════════════════════

def _assert_unknown_angle_is_not_zero(result):
    for p in result.data["points"]:
        assert p["scan_angle_deg"] is None, (
            "帧角读不到被折叠成一个数 —— 同一个畴会在两种帧角下报成两个畴")
        assert p["meta"]["scan_angle_deg"] is None
        assert p["meta"]["scan_angle_known"] is False, (
            "落库那一层会把 None 的键整个丢掉 —— 没有这个布尔,"
            "「读不到角度」和「这是旧格式 marker」在库里长得一模一样")
        assert p["verdict_reason"] == "unknown_frame_angle"


def test_unknown_frame_angle_is_undetermined_not_zero():
    script = dict(DEFAULT_SCRIPT)
    script["AssessDomainPhase"] = assessed(
        reason="unknown_frame_angle", angle=None, fingerprint=())
    _, result = run_survey(script, max_points=2)
    assert result.success
    _assert_unknown_angle_is_not_zero(result)
    assert any("0°" in w or "0 °" in w for w in result.data["warnings"]), (
        "读不到角度的点数要说出来,并说明**没有**按 0° 处理")


def test_mixed_frame_angles_are_flagged():
    """同一轮里帧角不止一个 ⇒ xy 各向异性会让同一个畴报成两个 —— 先说出来。"""
    script = dict(DEFAULT_SCRIPT)
    script["AssessDomainPhase"] = assessed_varying(
        [{"angle": 0.0}, {"angle": 30.0}])
    _, result = run_survey(script, max_points=2)
    assert result.data["scan_angles_deg"] == [0.0, 30.0]
    assert any("帧角不止一个" in w for w in result.data["warnings"])


# ═══════════════════════════════════════════════════════════════════════
# 7. 采样点记录 = marker 的落库素材(设计 §3.4)
# ═══════════════════════════════════════════════════════════════════════

META_KEYS = {
    "domain_search_id", "bracket_id", "role", "point_index", "point_status",
    "fingerprint", "scan_angle_deg", "scan_angle_known", "verdict",
    "verdict_reason", "reference_version", "coord_epoch",
}


def test_point_record_carries_the_marker_meta_schema():
    _, result = run_survey(max_points=2)
    for i, p in enumerate(result.data["points"], start=1):
        # recorder 认的那几个键(_marker_subrecords 只吃这些形状)
        assert p["center_x_m"] is not None and p["center_y_m"] is not None
        assert p["width_m"] == ATOMIC_FRAME_M == p["height_m"]
        assert p["index"] == i and p["sxm_path"]
        meta = p["meta"]
        assert set(meta) == META_KEYS, "meta 的键集合与设计 §3.4 的表对不上"
        assert meta["role"] == ROLE_SEED and meta["bracket_id"] is None, (
            "survey 的点全是种子点;bracket 是二分的概念,下一轮才有")
        assert meta["domain_search_id"].startswith("domain-survey-")
        assert meta["coord_epoch"] == EPOCH
        assert meta["fingerprint"] and all(len(t) == 3 for t in meta["fingerprint"])
        assert meta["point_status"] in ALL_POINT_STATUS


def test_a_point_without_a_verdict_does_not_pretend_to_have_one():
    """「没测」和「测出来判不了」必须是两句话。"""
    script = dict(DEFAULT_SCRIPT)
    script["ScanAt"] = FAIL
    _, result = run_survey(script, max_points=3)
    failed = [p for p in result.data["points"] if p["status"] != POINT_ASSESSED]
    assert failed, "脚本让每一帧都失败,却没有一个失败的点"
    for p in failed:
        assert p["verdict"] is None and p["verdict_reason"] is None
        assert p["fingerprint"] == [] and p["meta"]["scan_angle_deg"] is None
        assert p["success"] is False


def test_a_reused_frame_path_is_not_evidence_for_the_next_point():
    """「借最新文件伪造历史」:上一个点的 .sxm 被当成这一个点的证据。

    后果不是报错,是**两个点的指纹一模一样** —— 而那个「一样」是假的,
    正好会让普查得出「这一片全是同一个畴」。
    """
    script = dict(DEFAULT_SCRIPT)
    script["GetLatestScanFile"] = lambda params, n: {"path": "/tmp/same.sxm"}
    _, result = run_survey(script, max_points=3)
    _assert_reused_frame_is_refused(result)


def test_consecutive_scan_failures_stop_the_survey():
    """两帧连着扫不成就收尾 —— 后面几十帧每一帧都是几分钟机时。"""
    script = dict(DEFAULT_SCRIPT)
    script["ScanAt"] = FAIL
    ctx, result = run_survey(script, max_points=8)
    assert ctx.count("ScanAt") == SDB.MAX_CONSECUTIVE_SCAN_FAILURES
    statuses = [p["status"] for p in result.data["point_log"]]
    assert statuses[:2] == [POINT_SCAN_FAILED] * 2
    assert set(statuses[2:]) == {POINT_SKIPPED}
    assert result.data["stopped_early"]
    assert result.data["points_visited"] == 2, (
        "只有真的发过扫描的那两个点进地图;剩下 6 个从没去过")


def test_a_coarse_move_mid_survey_stops_it(monkeypatch):
    """普查跑到一半发生粗动 ⇒ 立刻停,**不继续用旧坐标**。

    这是整条流程里最重要的负向一步:粗动之后同一个 ``(x, y)`` 指的是**另一片表面**,
    而剩下的点会照样扫得很成功。拒绝是唯一的处置 —— 既不夹紧,也不换算
    (跨代次换算在本仓有意不存在:开环步长随温度差五倍,换算出来的坐标看上去
    和真坐标一样,而它是编的)。
    """
    from mast.core import coord_epoch as CE

    seen = {"n": 0}

    def _epoch():
        seen["n"] += 1
        # 第 1 次是普查盖章,第 2 次是第一个点的复核 —— 之后「发生了一次粗动」。
        return EPOCH if seen["n"] <= 2 else EPOCH + 1

    monkeypatch.setattr(CE, "read_current_epoch", _epoch)
    ctx, result = run_survey(max_points=3)
    assert ctx.count("ScanAt") == 1, "代次变了还继续扫"
    assert [p["status"] for p in result.data["point_log"]] == \
        [POINT_ASSESSED, POINT_EPOCH_STALE, POINT_SKIPPED]
    stale = result.data["point_log"][1]["error"] or ""
    assert "重新规划" in stale and "不做跨代次换算" in stale
    assert result.data["points_visited"] == 1, (
        "作废的点和跳过的点都没去过 —— 不许在地图上留足迹")


def test_unreadable_epoch_is_not_a_stale_epoch():
    """「查不到」不是「陈旧」—— 那会让读不到记录变成一道解不开的闸。"""
    from mast.core import coord_epoch as CE

    orig = CE.read_current_epoch
    try:
        CE.read_current_epoch = lambda: None
        ctx, result = run_survey(max_points=2)
    finally:
        CE.read_current_epoch = orig
    assert result.success and ctx.count("ScanAt") == 2
    assert any("代次保护" in w for w in result.data["warnings"])


def test_missing_map_history_is_reported_not_assumed_clean():
    """空表与「这片表面确实干净」在几何上分不开 —— 必须说出来。"""
    from mast.core import map_scope as MS

    orig = MS.load_markers
    try:
        MS.load_markers = lambda **kw: ([], 0, False)
        _, result = run_survey(dry_run=True, max_points=2)
    finally:
        MS.load_markers = orig
    assert result.data["map_history_available"] is False
    assert any("一个坑都不知道" in w for w in result.data["warnings"])


# ═══════════════════════════════════════════════════════════════════════
# 8. 聚类报告(设计 D10 第 2 步)
# ═══════════════════════════════════════════════════════════════════════

FP_A = [[30.0, 0.250, 1.0], [90.0, 0.250, 0.9]]
FP_B = [[5.0, 0.320, 1.0], [65.0, 0.320, 0.9]]


def _items(*fps):
    return [{"index": i + 1, "fingerprint": f, "frame": f"/tmp/f{i + 1}.sxm"}
            for i, f in enumerate(fps)]


def test_clustering_needs_a_symmetry_the_code_cannot_guess():
    """``symmetry_deg`` 是**样品事实**。猜一个 60 出来,聚类会照常出结果、
    照常看起来合理,而它量的是一个没人给过的假设。"""
    rep = cluster_two(_items(FP_A, FP_B), symmetry_deg=None)
    assert rep.available is False and rep.reason == CLUSTER_NO_SYMMETRY
    assert rep.clusters == [] and rep.separation_ratio is None
    assert "样品事实" in rep.next_step, "判不了要给出**能做的一件事**"


def test_cluster_report_has_the_fields_a_human_needs():
    rep = cluster_two(_items(FP_A, FP_A, FP_B, FP_B), symmetry_deg=60.0)
    assert rep.available and rep.n_comparable == 4
    assert [c["name"] for c in rep.clusters] == ["cluster_1", "cluster_2"], (
        "簇不是畴的名字 —— 没有人确认过之前,代码里出现一个畴名就已经越界了")
    assert sorted(c["size"] for c in rep.clusters) == [2, 2]
    for c in rep.clusters:
        assert c["representative_frame"].endswith(".sxm")
        assert c["fingerprint"] and all(len(t) == 3 for t in c["fingerprint"])
        assert c["intra_max"] == pytest.approx(0.0, abs=1e-9)
    assert rep.inter_min is not None and rep.inter_min > 0
    assert rep.symmetry_deg == 60.0
    assert rep.weights_source == "uncalibrated_equal_weights", (
        "没有参照系时 w_θ/w_T 是未标定的 —— 报告要说明这一点")


def test_separation_ratio_is_none_when_the_denominator_has_no_sample():
    """D6:分母比分子重要。两个单点簇之间的距离再大也说明不了分离度。"""
    rep = cluster_two(_items(FP_A, FP_B), symmetry_deg=60.0)
    assert rep.available and rep.inter_min is not None
    assert rep.separation_ratio is None, "拿一个没有样本的分母凑出了一个比值"
    assert "分母" in rep.ratio_reason


def test_separation_ratio_is_a_number_once_the_denominator_exists():
    noisy = [[31.0, 0.253, 1.0], [91.0, 0.253, 0.9]]
    rep = cluster_two(_items(FP_A, noisy, FP_B), symmetry_deg=60.0)
    assert rep.available and rep.separation_ratio is not None
    assert rep.separation_ratio > 1.0, "簇间该比簇内远,否则这两簇分不开"


def test_clustering_is_skipped_with_one_comparable_fingerprint():
    rep = cluster_two(_items(FP_A), symmetry_deg=60.0)
    assert rep.available is False and rep.reason == "too_few_comparable_fingerprints"
    assert rep.next_step


def test_survey_reports_the_clustering_end_to_end():
    script = dict(DEFAULT_SCRIPT)
    script["AssessDomainPhase"] = assessed_varying([
        {"fingerprint": tuple(map(tuple, FP_A))},
        {"fingerprint": tuple(map(tuple, FP_A))},
        {"fingerprint": tuple(map(tuple, FP_B))},
        {"fingerprint": tuple(map(tuple, FP_B))},
    ])
    _, result = run_survey(script, max_points=4, symmetry_deg=60.0)
    cl = result.data["clustering"]
    assert cl["available"] and cl["symmetry_source"] == "parameter"
    assert len(cl["clusters"]) == 2
    assert "cluster_1" in (result.summary or "")
    _assert_no_label_without_a_reference(result)


def test_without_symmetry_the_survey_still_reports_every_fingerprint():
    _, result = run_survey(max_points=3)
    assert result.data["clustering"]["reason"] == CLUSTER_NO_SYMMETRY
    assert all(p["fingerprint"] for p in result.data["points"]), (
        "聚类做不了不该把指纹一起吞掉 —— 指纹本身就是这一轮的产物")


# ═══════════════════════════════════════════════════════════════════════
# 9. 落库行为验证(设计陷阱 16)—— 造输入看输出,不 grep 落点
# ═══════════════════════════════════════════════════════════════════════

def _survey_payload():
    """把一次真的 survey 结果包成 recorder 收的那种 payload。"""
    _, result = run_survey(max_points=2)
    assert result.success
    return {"skill": "SearchDomainBoundary", "success": True,
            "params": {"mode": MODE_SURVEY}, "data": result.data}, result


def test_storage_round_trips_the_fingerprint_meta(tmp_path):
    """先证明**数据库这一层不是缺口**:meta 交给它,它就存得下、取得回。

    没有这一条,下面那条红的时候没法区分「recorder 没转发」与「meta 存不进去」。
    """
    from mast.logging.storage import ExperimentStorage

    payload, _ = _survey_payload()
    point = payload["data"]["points"][0]
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    st.log_marker(kind="scan", x_m=point["center_x_m"], y_m=point["center_y_m"],
                  w_m=point["width_m"], h_m=point["height_m"],
                  skill_name="SearchDomainBoundary", experiment_id=eid,
                  meta=dict(point["meta"]))
    rows = st.get_markers(experiment_id=eid)
    assert len(rows) == 1
    meta = rows[0]["meta"]
    assert meta["fingerprint"], "meta.fingerprint 落库之后是空的"
    assert all(len(t) == 3 for t in meta["fingerprint"])
    assert meta["role"] == ROLE_SEED
    assert meta["domain_search_id"].startswith("domain-survey-")


def _assert_fingerprint_meta_landed(rows, points):
    """库里那几行到底长什么样 —— 断言体只有这一份,变异验证要调它。"""
    assert len(rows) == len(points), "每个采样点该落一条 kind='scan' 的 marker"
    assert all(r["kind"] == "scan" for r in rows), "不许为采样点新增 marker kind"
    for r in rows:
        meta = r["meta"]
        assert meta.get("fingerprint"), "meta.fingerprint 是空的"
        assert all(len(t) == 3 for t in meta["fingerprint"])
        assert meta.get("role") == ROLE_SEED
        assert meta.get("domain_search_id", "").startswith("domain-survey-")
        assert meta.get("verdict") == "undetermined"
        assert meta.get("verdict_reason") == "no_reference"
        assert meta.get("coord_epoch") == EPOCH
        # 记录层自己的记账没有被技能的 meta 挤掉。
        assert meta.get("pos_src") == "region_record"
        assert meta.get("region") == meta["point_index"]
    # 足迹是真的落在各个网格点上,不是都堆在同一处。
    assert len({(r["x_m"], r["y_m"]) for r in rows}) == len(rows)
    assert all(r["w_m"] == pytest.approx(ATOMIC_FRAME_M) for r in rows)


def test_marker_meta_fingerprint_reaches_the_database_through_the_recorder(tmp_path):
    """陷阱 16:造 payload 走**真的**记录路径,断言库里落了什么。

    这条曾经是 ``xfail(strict=True)``:落库链路断在两处 —— ``classify_skill``
    认不出这个 composite 的名字(分诊在读 ``data`` 之前就 return),而
    ``_marker_subrecords`` 把 ``meta`` 整个丢掉。两处都 fire-and-forget,症状是
    ``meta.fingerprint`` 永远是 ``None``、零报错。2026-08-14 两处都补了
    (recorder 加第四条准入 + 转发 ``meta``/``kind``),标记随之摘掉。

    **不 grep 落点**:上一次同类修复的教训是「按预期落点 grep 找不到 ≠ 没实现」。
    """
    from mast.core.runtime import CoreRuntime
    from mast.logging.storage import ExperimentStorage

    payload, _ = _survey_payload()
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    rt = SimpleNamespace(
        _storage=st, _registry=None, _state=None,
        _experiment_log=SimpleNamespace(current_experiment_id=eid,
                                        current_sample_id=None))
    CoreRuntime._record_map_marker(rt, payload)

    _assert_fingerprint_meta_landed(st.get_markers(experiment_id=eid),
                                    payload["data"]["points"])


def _record(st, eid, skill, data, category=None):
    from mast.core.runtime import CoreRuntime

    CoreRuntime._record_map_marker(
        SimpleNamespace(_storage=st, _registry=None, _state=None,
                        _experiment_log=SimpleNamespace(
                            current_experiment_id=eid, current_sample_id=None)),
        {"skill": skill, "success": True, "params": {}, "data": data})
    return st.get_markers(experiment_id=eid)


def _points_payload(n=2):
    return {"points": [
        {"index": i + 1, "center_x_m": i * 1e-7, "center_y_m": 0.0,
         "width_m": ATOMIC_FRAME_M, "height_m": ATOMIC_FRAME_M,
         "kind": "scan", "meta": {"fingerprint": [[30.0, 0.25, 1.0]]}}
        for i in range(n)]}


def _assert_leaves_no_footprint(st, eid, skill):
    """断言体只有一份 —— 正向测试与变异验证共用。"""
    rows = _record(st, eid, skill, _points_payload())
    assert rows == [], (
        f"{skill} 不可能在表面上留下事件,却画出了 {len(rows)} 个足迹 —— "
        f"「我报了坐标」不是准入证")


@pytest.mark.parametrize("skill", [
    "AnalyzeScanImage",      # Analyze* —— 只读动词
    "plot_scan",             # snake_case —— 桥接进来的 agent 工具,不碰仪器
    "AssessDomainPhase",     # Assess* —— 拿已有数据打个分
    "ScanIntelSelfCheck",    # *SelfCheck —— 只读自检
    "OpticalStageMove",      # 光学台 —— 另一台仪器,不该出现在 STM 表面地图上
])
def test_a_skill_that_cannot_touch_the_surface_leaves_no_footprint(tmp_path, skill):
    """「我报了坐标」不是准入证。

    逐点定位记录这条准入**绕过了** ``classify_skill``,所以它必须自己守住那条排除。
    否则一个 ``analyze_*`` 报出来的坐标(从数据里读的,不是针尖去过的地方)会变成
    地图上的足迹 —— 而找干净地方的路径靠地图避开用过的位置,一个凭空的坑会让流程
    永远绕开一块好表面。本仓 2026-08 刚因为同一件事修过三个技能。
    """
    from mast.logging.storage import ExperimentStorage

    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    _assert_leaves_no_footprint(st, eid, skill)


def test_an_analysis_category_skill_leaves_no_footprint_either(tmp_path):
    """名字看着像仪器技能,但注册表说它是 ANALYSIS —— 类别是不会被名字绕过的那条。"""
    from mast.core.runtime import CoreRuntime
    from mast.core.types import SkillCategory
    from mast.logging.storage import ExperimentStorage

    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    reg = SimpleNamespace(
        get=lambda name: object(),
        _get_metadata=lambda cls: SimpleNamespace(
            category=SkillCategory.ANALYSIS))
    CoreRuntime._record_map_marker(
        SimpleNamespace(_storage=st, _registry=reg, _state=None,
                        _experiment_log=SimpleNamespace(
                            current_experiment_id=eid, current_sample_id=None)),
        {"skill": "SearchDomainBoundary", "success": True, "params": {},
         "data": _points_payload()})
    assert st.get_markers(experiment_id=eid) == []


def test_an_unknown_declared_kind_falls_back_instead_of_creating_one(tmp_path):
    """子记录声明的 kind 走白名单。

    marker kind 是**双端镜像**的(后端 ``KIND_STYLE`` ↔ 前端配色/图例)。让子记录
    随便声明一个新 kind,失败模式是地图上**静默变灰** —— 不报错,只是那批点看不出
    是什么。所以认不出来就退回按技能名分类,绝不照单全收。
    """
    from mast.io.exp_map import KIND_STYLE
    from mast.logging.storage import ExperimentStorage

    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    data = _points_payload(1)
    data["points"][0]["kind"] = "domain_sample"      # 一个不存在的 kind
    rows = _record(st, eid, "SearchDomainBoundary", data)
    assert len(rows) == 1
    assert rows[0]["kind"] in KIND_STYLE, "凭空造出了一个前端不认识的 kind"
    assert rows[0]["kind"] == "scan"
    assert rows[0]["meta"]["fingerprint"], "退回 kind 的同时把 meta 也丢了"


def test_an_unreadable_frame_angle_survives_the_round_trip_as_unknown(tmp_path):
    """``scan_angle_deg=None`` 落库时会被整个剔掉 —— 靠 ``scan_angle_known`` 认。

    ``_log_one_marker`` 过滤掉 meta 里所有 ``None`` 值。没有那个布尔,「角度读不到」
    与「这是旧格式 marker」在库里长得一模一样,而它们的下一步完全不同。
    """
    from mast.core.runtime import CoreRuntime
    from mast.logging.storage import ExperimentStorage

    script = dict(DEFAULT_SCRIPT)
    script["AssessDomainPhase"] = assessed(
        reason="unknown_frame_angle", angle=None, fingerprint=())
    _, result = run_survey(script, max_points=2)
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    CoreRuntime._record_map_marker(
        SimpleNamespace(_storage=st, _registry=None, _state=None,
                        _experiment_log=SimpleNamespace(
                            current_experiment_id=eid, current_sample_id=None)),
        {"skill": "SearchDomainBoundary", "success": True,
         "params": {"mode": MODE_SURVEY}, "data": result.data})

    rows = st.get_markers(experiment_id=eid)
    assert rows, "点没落库"
    for r in rows:
        assert "scan_angle_deg" not in r["meta"], (
            "None 值会被落库层剔掉 —— 这里出现它说明它被折叠成了一个数")
        assert r["meta"]["scan_angle_known"] is False
        assert r["meta"]["verdict_reason"] == "unknown_frame_angle"


# ═══════════════════════════════════════════════════════════════════════
# 10. 变异验证 —— 先证明变异已应用,再证明测试红了
# ═══════════════════════════════════════════════════════════════════════

def test_mutation_not_releasing_the_centre_zone_turns_the_grid_red(monkeypatch):
    """不解除中心区 ⇒ 整张网格塌进 ±250 nm。"""
    from dataclasses import replace as _replace

    def _keep_zone(base, *, frame_size_m, grid_pitch_m=None):
        cfg, why = release_center_zone(base, frame_size_m=frame_size_m,
                                       grid_pitch_m=grid_pitch_m)
        return _replace(cfg, center_zone_side_m=base.center_zone_side_m), why

    monkeypatch.setattr(SDB, "release_center_zone", _keep_zone)
    cfg, _ = SDB.release_center_zone(BASE_CFG, frame_size_m=ATOMIC_FRAME_M)
    assert cfg.center_zone_side_m == 500e-9              # 变异已应用
    with pytest.raises(AssertionError):
        _assert_centre_zone_is_released(cfg)
    _, result = run_survey(dry_run=True, max_points=9)
    with pytest.raises(AssertionError):
        _assert_grid_reaches_beyond_the_centre_zone(result)


def test_mutation_folding_unknown_angle_to_zero_turns_its_test_red(monkeypatch):
    """把「读不到」折叠成 0.0 —— 零报错,而同一个畴会报成两个。"""
    orig = SearchDomainBoundary._point_record

    def _folded(self, *a, **kw):
        rec = orig(self, *a, **kw)
        if rec["scan_angle_deg"] is None:
            rec["scan_angle_deg"] = 0.0
            rec["meta"]["scan_angle_deg"] = 0.0
            rec["meta"]["scan_angle_known"] = True
        return rec

    monkeypatch.setattr(SearchDomainBoundary, "_point_record", _folded)
    script = dict(DEFAULT_SCRIPT)
    script["AssessDomainPhase"] = assessed(
        reason="unknown_frame_angle", angle=None, fingerprint=())
    _, result = run_survey(script, max_points=2)
    assert result.data["points"][0]["scan_angle_deg"] == 0.0   # 变异已应用
    with pytest.raises(AssertionError):
        _assert_unknown_angle_is_not_zero(result)


def test_mutation_defaulting_a_label_without_a_reference_turns_its_test_red(
        monkeypatch):
    """``undetermined → label`` 的兜底 —— 代码自己发明了「这是 A 相」。"""
    orig = SearchDomainBoundary._point_record

    def _labelled(self, *a, **kw):
        rec = orig(self, *a, **kw)
        if rec["status"] == POINT_ASSESSED and not rec["label_out"]:
            rec["label_out"] = "A"
            rec["verdict"] = "A"
            rec["meta"]["verdict"] = "A"
        return rec

    monkeypatch.setattr(SearchDomainBoundary, "_point_record", _labelled)
    _, result = run_survey(max_points=3)
    assert result.data["labels_emitted"] == ["A"]              # 变异已应用
    with pytest.raises(AssertionError):
        _assert_no_label_without_a_reference(result)


def _assert_reused_frame_is_refused(result):
    statuses = [p["status"] for p in result.data["points"]]
    assert statuses[0] == POINT_ASSESSED
    assert statuses[1:] == [POINT_STALE_FRAME] * (len(statuses) - 1), (
        "同一张 .sxm 被当成了两个点的证据 —— 两个点的指纹于是一模一样,"
        "而那个「一样」是假的,普查会得出「这一片全是同一个畴」")
    assert result.data["points_assessed"] == 1


def test_mutation_dropping_the_meta_forwarding_turns_its_test_red(tmp_path,
                                                                  monkeypatch):
    """把 recorder 的 ``meta`` 转发去掉 —— 也就是**修复之前**的那个行为。

    这是整条链上最该被变异证明的一处:它的失败模式不是报错,是 `meta.fingerprint`
    永远是 `None`,而技能自己的返回体里那个字段一直好好的。所以「有一条测试」不够,
    要证明那条测试**真的在看落库的那一行**。
    """
    from mast.core import runtime as RT
    from mast.logging.storage import ExperimentStorage

    orig = RT._marker_subrecords

    def _no_meta(data):
        out = orig(data)
        for r in out:
            r["meta"] = None            # 变异:回到「meta 被整个丢掉」
        return out

    monkeypatch.setattr(RT, "_marker_subrecords", _no_meta)
    probe = RT._marker_subrecords(_points_payload(1))
    assert probe and probe[0]["meta"] is None                  # 变异已应用

    payload, _ = _survey_payload()
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    rows = _record(st, eid, "SearchDomainBoundary", payload["data"])
    assert rows, "变异只该去掉 meta,不该把 marker 一起弄没"
    with pytest.raises(AssertionError):
        _assert_fingerprint_meta_landed(rows, payload["data"]["points"])


def test_mutation_letting_a_readonly_skill_through_turns_its_test_red(tmp_path,
                                                                      monkeypatch):
    """拆掉 ``can_be_positioned`` 这道闸 ⇒ 只读技能报了坐标就能画足迹。

    这是「绕过一个分类函数就同时绕过了它的排除」那个洞 —— 它是修复过程中**新开**
    的,所以尤其要证明补丁真的在挡。
    """
    from mast.io import exp_map as EM
    from mast.logging.storage import ExperimentStorage

    monkeypatch.setattr(EM, "can_be_positioned", lambda skill, category=None: True)
    assert EM.can_be_positioned("plot_scan") is True           # 变异已应用

    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    with pytest.raises(AssertionError):
        _assert_leaves_no_footprint(st, eid, "plot_scan")


def test_mutation_accepting_a_reused_frame_turns_its_test_red(monkeypatch):
    """拆掉「同一个 .sxm 不能当两个点的证据」这道闸。"""
    monkeypatch.setattr(SearchDomainBoundary, "_frame_is_reused",
                        staticmethod(lambda path, seen: False))
    assert SearchDomainBoundary._frame_is_reused("x", {"x"}) is False  # 变异已应用
    script = dict(DEFAULT_SCRIPT)
    script["GetLatestScanFile"] = lambda params, n: {"path": "/tmp/same.sxm"}
    _, result = run_survey(script, max_points=3)
    with pytest.raises(AssertionError):
        _assert_reused_frame_is_refused(result)
