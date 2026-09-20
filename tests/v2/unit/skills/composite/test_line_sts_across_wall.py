"""LineSTSAcrossWall 薄壳 —— S4 STS 设计 D24/D25、§3.4 的消费侧。

壳的职责只有四件:按 **id** 取几何、三道开跑前的闸、时间账、把坐标交给逐点引擎。
所以这一组测试盯的全是「它有没有在该拒绝的地方拒绝」和「它有没有把该说的话说出来」,
几何本身在 ``tests/v2/unit/core/test_sts_line_plan.py`` 里测。

一条线谱是一晚上。这里每一条拒绝都对应一种「跑完了、数据也齐、结论是空的」失败。
"""

from __future__ import annotations

# ── path bootstrap ──
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

import json
from dataclasses import dataclass, field

import pytest

from mast.core.types import SkillResult
from mast.io.exp_map import MapMarker
from mast.skills.composite.graph_executor import CompositeProgress
from mast.skills.composite.line_sts_across_wall import (
    POINT_ENGINE_SKILL,
    LineSTSAcrossWall,
)

NM = 1e-9
SEARCH = "ds-001"
BRACKET = "br-1"
EPOCH = 7


@pytest.fixture(autouse=True)
def _isolated_project_root(tmp_path, monkeypatch):
    """执行器会往 ``<project_root>/experiments/`` 写 sidecar —— 别写进真目录。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))


# ── 假件 ────────────────────────────────────────────────────────────────────


@dataclass
class FakeCtx:
    engine_result: SkillResult | None = None
    run_log: list = field(default_factory=list)
    emitted: list = field(default_factory=list)
    flushes: int = 0

    def run(self, skill_name: str, params: dict, *a, **kw) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if self.engine_result is not None:
            return self.engine_result
        return SkillResult(skill_name=skill_name, success=True, data={})

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(CompositeProgress.from_dict(progress.to_dict()))

    def get_progress(self, name: str):
        return None

    def checkpoint_flush(self) -> None:
        self.flushes += 1


class FakeRef:
    def __init__(self, confirmed_by: str = "operator"):
        self.confirmed_by = confirmed_by
        self.version = "v001"


def marker(role: str, x_nm: float, y_nm: float, *, verdict: str,
           bracket_id: str = BRACKET, ref: "str | None" = "v001",
           scan_angle_deg=None, ts: str = "", **meta_extra) -> MapMarker:
    meta = {"domain_search_id": SEARCH, "bracket_id": bracket_id, "role": role,
            "verdict": verdict, "reference_version": ref,
            "scan_angle_deg": scan_angle_deg}
    meta.update(meta_extra)
    return MapMarker(kind="scan", x_m=x_nm * NM, y_m=y_nm * NM,
                     w_m=5 * NM, h_m=5 * NM, meta=meta, coord_epoch=EPOCH,
                     timestamp=ts or f"2026-08-14T20:00:{abs(int(x_nm)) % 60:02d}")


def install_map(monkeypatch, markers, *, epoch: int = EPOCH,
                available: bool = True) -> None:
    import mast.core.map_scope as map_scope
    monkeypatch.setattr(map_scope, "load_markers",
                        lambda **kw: (list(markers), epoch, available))


def install_reference(monkeypatch, ref) -> None:
    import mast.vision.domain_reference as dr
    monkeypatch.setattr(dr, "load_reference", lambda *a, **kw: ref)


def a_bracket(**kw) -> list[MapMarker]:
    """一个正常的 bracket:lo 在 −4 nm、hi 在 +4 nm,判定不同。跨度 8 nm。"""
    return [marker("lo", -4, 0, verdict="A", **kw),
            marker("hi", 4, 0, verdict="B", **kw)]


def run(monkeypatch, params: dict, *, markers=None, ref=FakeRef(),
        engine_result=None, **map_kw):
    install_map(monkeypatch, a_bracket() if markers is None else markers, **map_kw)
    install_reference(monkeypatch, ref)
    ctx = FakeCtx(engine_result=engine_result)
    base = {"domain_search_id": SEARCH, "bracket_id": BRACKET,
            "expected_coord_epoch": EPOCH, "dry_run": True,
            "fine_spacing_nm": 2.0, "coarse_spacing_nm": 5.0,
            "fine_half_width_nm": 3.0, "line_half_length_nm": 20.0}
    base.update(params)
    return LineSTSAcrossWall().execute(ctx, base), ctx


# ── 按 id 取几何 ────────────────────────────────────────────────────────────


def test_ids_alone_produce_the_geometry():
    """说一个 id,不说四个浮点数 —— 原点/轴/不确定度全从标记里读出来。"""
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {})
    assert res.success, res.error
    d = res.data
    assert d["input_form"] == "bracket"
    assert d["origin_x_m"] == pytest.approx(0.0, abs=1e-15)
    assert d["axis_unit_x"] == pytest.approx(1.0)
    assert d["uncertainty_m"] == pytest.approx(8 * NM)
    assert d["domain_search_id"] == SEARCH and d["bracket_id"] == BRACKET
    assert d["reference_confirmed_by"] == "operator"


def test_the_max_survives_end_to_end():
    """壳里也要看得见:spec 的 ±3 nm 被 bracket 的 8 nm 跨度撑开(D22/陷阱 24)。"""
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {})
    assert res.data["fine_half_width_m"] == pytest.approx(8 * NM)
    assert res.data["fine_half_width_source"] == "uncertainty"
    assert "fine_window_widened_by_uncertainty" in res.data["warnings"]
    fine = [p for p in res.data["planned_points"] if p["zone"] == "fine"]
    assert max(abs(p["axis_s_m"]) for p in fine) == pytest.approx(8 * NM)


def test_axis_comes_from_the_bracket_not_from_the_scan_angle():
    """``scan_angle_deg`` 是 ``null`` 时**不按 0 处理**,也从不参与定轴(§3.4)。"""
    markers = [marker("lo", 0, -4, verdict="A", scan_angle_deg=None),
               marker("hi", 0, 4, verdict="B", scan_angle_deg=None)]
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, markers=markers)
    assert res.success, res.error
    # 轴是 +y(90°)。若有人把读不到的帧角当 0 拿去定轴,这里会变成 0°。
    assert res.data["axis_angle_deg"] == pytest.approx(90.0)
    assert res.data["scan_angle_deg"] is None
    assert res.data["scan_angle_deg"] != 0


def test_the_result_says_the_axis_is_not_the_wall_normal():
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {})
    assert res.data["axis_is_not_the_wall_normal"] is True
    assert not [k for k in res.data if k.startswith("normal") or "normal_" in k]


# ── 三道闸 ──────────────────────────────────────────────────────────────────


def test_undetermined_verdict_is_refused():
    """「判不了」不等于「在畴 X 里」 —— 不跑一晚上去赌。"""
    markers = [marker("lo", -4, 0, verdict="undetermined"),
               marker("hi", 4, 0, verdict="B")]
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, markers=markers)
    assert not res.success
    assert res.data["refusal_code"] == "verdict_undetermined"


def test_empty_verdict_is_refused_too():
    """读不到判定和判不了是同一件事:都不是一个 label。"""
    markers = [marker("lo", -4, 0, verdict=""), marker("hi", 4, 0, verdict="B")]
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, markers=markers)
    assert res.data["refusal_code"] == "verdict_undetermined"


def test_both_ends_with_the_same_label_is_not_a_bracket():
    """两端判定相同 ⇒ 中间没有确立畴界,跑出来只会是一族一模一样的谱。"""
    markers = [marker("lo", -4, 0, verdict="A"), marker("hi", 4, 0, verdict="A")]
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, markers=markers)
    assert res.data["refusal_code"] == "bracket_verdicts_agree"


def test_unconfirmed_reference_is_refused():
    """``confirmed_by`` 为空 ⇒ 拒绝(§3.4 的人工确认闸)。"""
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, ref=FakeRef(confirmed_by=""))
    assert not res.success
    assert res.data["refusal_code"] == "reference_not_confirmed"


def test_missing_reference_file_is_refused():
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, ref=None)
    assert res.data["refusal_code"] == "reference_not_found"


def test_unrecorded_reference_version_is_refused():
    """标记里没记按哪一版判的 ⇒ 无法确认它被人看过。「读不到」不是「确认过」。"""
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, markers=a_bracket(ref=None))
    assert res.data["refusal_code"] == "reference_version_unknown"


def test_two_ends_judged_by_different_references_are_refused():
    markers = [marker("lo", -4, 0, verdict="A", ref="v001"),
               marker("hi", 4, 0, verdict="B", ref="v002")]
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, markers=markers)
    assert res.data["refusal_code"] == "reference_version_conflict"


def test_the_gate_can_be_opened_but_it_leaves_a_trace():
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {"require_confirmed_reference": False},
                     ref=FakeRef(confirmed_by=""))
    assert res.success, res.error
    assert res.data["reference_confirmation_bypassed"] is True
    assert res.data["reference_confirmed_by"] is None


def test_unreadable_map_refuses_instead_of_passing():
    """``available=False`` ⇒ 拒绝。这条闸的全部意义就是不让「不知道」放行。"""
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, available=False)
    assert not res.success
    assert res.data["refusal_code"] == "markers_unavailable"


def test_coord_epoch_mismatch_refuses():
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {"expected_coord_epoch": EPOCH - 1})
    assert res.data["refusal_code"] == "coord_epoch_mismatch"
    assert res.data["current_coord_epoch"] == EPOCH


def test_cross_site_boundary_is_refused_with_the_remedy():
    """只有粗动步数、没有米坐标 ⇒ 拒绝,并说得出「收进单个站点」(D25/陷阱 25)。"""
    markers = [marker("lo", -4, 0, verdict="A", uncertainty_steps=3),
               marker("hi", 4, 0, verdict="B", uncertainty_steps=3)]
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, markers=markers)
    assert res.data["refusal_code"] == "cross_site_no_metres"
    assert "收进单个站点" in res.error


def test_no_input_at_all_is_refused():
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {"domain_search_id": "", "bracket_id": ""})
    assert res.data["refusal_code"] == "no_input"


def test_explicit_geometry_without_uncertainty_is_refused():
    """显式几何漏了不确定度 ⇒ 拒绝,而不是悄悄按 0 把精细窗缩回 spec 值。"""
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {"domain_search_id": "", "origin_x_m": 0.0,
                          "origin_y_m": 0.0, "axis_deg": 0.0})
    assert res.data["refusal_code"] == "missing_uncertainty"


def test_ambiguous_bracket_end_is_refused_not_guessed():
    markers = [marker("lo", -4, 0, verdict="A", ts="t"),
               marker("lo", -6, 0, verdict="A", ts="t"),
               marker("hi", 4, 0, verdict="B")]
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, markers=markers)
    assert res.data["refusal_code"] == "bracket_ambiguous"


def test_latest_end_wins_when_timestamps_can_order_them():
    """二分每走一步换掉一端 ⇒ 当前 bracket = 各自最新的那条。"""
    markers = [marker("lo", -20, 0, verdict="A", ts="2026-08-14T20:00:00"),
               marker("lo", -4, 0, verdict="A", ts="2026-08-14T21:00:00"),
               marker("hi", 4, 0, verdict="B", ts="2026-08-14T21:10:00")]
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {}, markers=markers)
    assert res.success, res.error
    assert res.data["uncertainty_m"] == pytest.approx(8 * NM)


# ── 时间账(D24)───────────────────────────────────────────────────────────


def test_the_time_estimate_is_in_the_result_and_in_the_summary():
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {"per_point_acquire_s": 25.0})
    assert res.data["estimated_duration_s"] is not None
    assert res.data["estimated_duration_s"] > 0.0
    assert res.data["time_budget"]["move_total_s"] > 0.0
    assert "分钟" in res.data["estimated_duration_note"]
    assert "分钟" in (res.summary or "")


def test_unknown_acquisition_time_is_reported_as_unknown_not_as_zero():
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {})
    assert res.data["estimated_duration_s"] is None
    assert "未知" in res.data["estimated_duration_note"]
    assert res.data["time_budget"]["unknown"] == ["acquire_s_per_point"]


# ── 交给逐点引擎 ────────────────────────────────────────────────────────────


def test_dry_run_touches_nothing():
    with pytest.MonkeyPatch.context() as mp:
        res, ctx = run(mp, {})
    assert res.success and res.data["dry_run"] is True
    assert res.data["acquired"] is False
    assert ctx.run_log == []


def test_the_real_run_hands_the_points_to_the_point_engine_by_name():
    engine = SkillResult(
        skill_name=POINT_ENGINE_SKILL, success=True,
        data={"points": [], "n_keep": 0, "coord_epoch": EPOCH})
    with pytest.MonkeyPatch.context() as mp:
        res, ctx = run(mp, {"dry_run": False, "condition": "overview",
                            "settle_s": 2.0},
                       engine_result=engine)
    assert res.success, res.error
    assert [name for name, _ in ctx.run_log] == [POINT_ENGINE_SKILL]
    sent = ctx.run_log[0][1]
    assert sent["expected_coord_epoch"] == EPOCH
    # 采谱条件是一个**组名**,不是六个数(2026-08-15,S4 STS 设计 D19)。本壳一个
    # 数都不发明也不转发 —— 组名坏掉/未标定由逐点引擎当场拒绝并说清楚。
    assert sent["condition"] == "overview"
    assert sent["settle_s"] == 2.0
    assert not ({"stab_bias_v", "stab_setpoint_a", "sweep_start_v", "sweep_end_v",
                 "num_points", "lockin_preset"} & set(sent)), (
        "显式数值又漏回去了 —— 那六个格子每一个都会变成真实的硬件动作，"
        "摆在工具表里等于请调用方填数")
    positions = json.loads(sent["positions"])
    # D23:交给引擎的表已经是采集顺序(|s| 升序),引擎按列表顺序走。
    assert [p["order"] for p in positions] == list(range(len(positions)))
    assert [abs(p["axis_s_m"]) for p in positions] == sorted(
        abs(p["axis_s_m"]) for p in positions)


def test_unset_engine_params_are_not_shadowed_by_our_own_defaults():
    """没给的项一个都不传下去。

    引擎那边「省略 = 不改仪器上现有的设置」,所以替它填一个默认不是保守 ——
    是替用户改掉他自己配好的稳定条件,而他不会知道是谁改的。
    """
    with pytest.MonkeyPatch.context() as mp:
        res, ctx = run(mp, {"dry_run": False})
    sent = ctx.run_log[0][1]
    assert set(sent) == {"positions", "expected_coord_epoch"}


def test_every_forwarded_name_is_a_real_engine_parameter():
    """转发的参数名必须与引擎的参数表逐字对上。

    引擎不认得的键会被安静丢掉 —— 「我明明设了稳定偏压」与「它根本没收到」长得
    一模一样(设计初稿里的 ``condition`` 组名就是这样一个不存在的参数)。这条测试
    是那个静默的唯一出声处;引擎改名时它当场红。
    """
    mod = pytest.importorskip(
        "mast.skills.composite.spectroscopy_at_positions",
        reason="逐点引擎尚未落地时,本壳的几何与闸门仍然可测")
    engine_params = {p.name for p in
                     mod.SpectroscopyAtPositions().metadata().parameters}
    from mast.skills.composite.line_sts_across_wall import _ENGINE_PASSTHROUGH

    unknown = [n for n in _ENGINE_PASSTHROUGH if n not in engine_params]
    assert not unknown, f"这些名字引擎不认识,会被安静丢掉: {unknown}"
    # 壳自己声明的参数里,凡是与引擎同名的都必须真的转发出去,否则就是接了不读。
    shell_params = {p.name for p in LineSTSAcrossWall().metadata().parameters}
    shared = (shell_params & engine_params) - {"expected_coord_epoch", "positions"}
    assert shared <= set(_ENGINE_PASSTHROUGH), (
        f"壳声明了但没转发: {sorted(shared - set(_ENGINE_PASSTHROUGH))}")


def test_missing_engine_rows_are_marked_no_result_not_zipped_forward():
    """引擎少回几条时不去 zip:少的如实标 ``no_result``,``success`` 显式为假。"""
    engine = SkillResult(
        skill_name=POINT_ENGINE_SKILL, success=True,
        data={"points": [{"index": 1, "success": True, "status": "ok",
                          "path": "/tmp/p000.dat", "verdict": "keep",
                          "x_m": 0.0, "y_m": 0.0}]})
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {"dry_run": False}, engine_result=engine)
    pts = res.data["points"]
    assert len(pts) > 1
    assert pts[0]["status"] == "ok" and pts[0]["success"] is True
    for row in pts[1:]:
        assert row["status"] == "no_result"
        assert row["success"] is False        # 显式,不是缺省
    # 几何字段仍在每一行上,采集顺序才对得上事后的漂移账。
    assert all("axis_s_m" in row and "order" in row for row in pts)


def test_engine_coordinates_are_recorded_but_do_not_overwrite_the_geometry():
    engine = SkillResult(
        skill_name=POINT_ENGINE_SKILL, success=True,
        data={"points": [{"index": 1, "success": True, "status": "ok",
                          "x_m": 999.0, "y_m": 888.0}]})
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {"dry_run": False}, engine_result=engine)
    row = res.data["points"][0]
    assert row["engine_x_m"] == 999.0
    assert row["x_m"] != 999.0        # 坐标以几何为准,不符是要看得见的事


def test_the_real_engine_accepts_our_positions_payload():
    """用**引擎自己的解析器**验一遍,而不是拿假件证明假件能通过。

    「用自己编的键名去证明键名存在,只有『假警报』和『碰巧对』两种结局」——
    这条测试把两者都排除掉:JSON 由本壳生成,由引擎的 ``_parse_positions`` 判。
    """
    mod = pytest.importorskip("mast.skills.composite.spectroscopy_at_positions")
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {})
    payload = json.dumps(res.data["planned_points"], ensure_ascii=False)
    parsed, err = mod.SpectroscopyAtPositions()._parse_positions(
        {"positions": payload})
    assert err == "", err
    planned = res.data["planned_points"]
    assert len(parsed) == len(planned)
    assert parsed[0]["x_m"] == pytest.approx(planned[0]["x_m"])


def test_acquired_runs_use_the_points_key():
    """D17:N 个不同 xy ⇒ 真跑过的那次列表键名必须是 ``points``(记录层只认这个)。"""
    engine = SkillResult(skill_name=POINT_ENGINE_SKILL, success=True,
                         data={"points": []})
    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {"dry_run": False}, engine_result=engine)
    assert isinstance(res.data["points"], list) and res.data["points"]
    assert "spectra" not in res.data


def test_a_dry_run_puts_nothing_on_the_experiment_map():
    """dry_run 一步都没动过针尖 ⇒ 地图上一个足迹都不能有。

    记录层只认 ``regions``/``points``,而且**缺 `success` 字段时按 True 处理** ——
    把点位表放进 ``points`` 会让一次「只是算给人看」的运行在地图上留下 N 个
    「已完成」的谱学点。用记录层自己的函数验,不是看键名眼熟。
    """
    from mast.core.runtime import _marker_subrecords

    with pytest.MonkeyPatch.context() as mp:
        res, _ = run(mp, {})
    assert res.data["dry_run"] is True
    assert res.data["planned_points"]
    assert _marker_subrecords(res.data) == []


def test_metadata_is_confirm_gated_and_declares_the_precondition():
    md = LineSTSAcrossWall().metadata()
    assert md.safety_level.name == "CONFIRM"
    assert "z_controller_on" in md.preconditions
    assert md.name == "LineSTSAcrossWall"
