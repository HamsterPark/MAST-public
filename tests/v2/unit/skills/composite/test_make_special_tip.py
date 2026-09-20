"""使用独立的合成结果验证特殊针尖流程的状态转换。"""
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

from mast.core.sample_facts import SubstrateFacts  # noqa: E402
from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite import make_special_tip as MST  # noqa: E402

FAIL = object()


class FakeCtx:
    """按技能名派活的假上下文（与 test_prepare_noble_tip 同款）。"""

    def __init__(self, script=None, *, abort_after=None):
        self.script = dict(script or {})
        self.calls: list[tuple[str, dict]] = []
        self._n: dict[str, int] = {}
        self._abort_after = abort_after
        self.run_id = "test-special-tip"

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
        return (self._abort_after is not None
                and len(self.calls) >= self._abort_after)

    def check_halt(self):
        return ""

    def safe_call(self, method, *args, role="main", allow_on_abort=False):
        class _R:
            error = ""
            return_value = ("", b"", [0.0])
        return _R()

    def count(self, name):
        return sum(1 for c, _ in self.calls if c == name)

    def params_for(self, name):
        return [p for c, p in self.calls if c == name]


# ── 脚本片段 ────────────────────────────────────────────────────────────────

def spot(dx_nm=50.0):
    def _f(params, n):
        return {"x_m": (n + 1) * dx_nm * 1e-9, "y_m": 0.0,
                "distance_m": dx_nm * 1e-9, "map_known": True}
    return _f


NO_SPOT = lambda params, n: FAIL           # noqa: E731

SCAN_FILE = {"path": "/tmp/frame.sxm"}
DAT_FILE = {"path": "/tmp/spec.dat"}


def poke(verdicts):
    def _f(params, n):
        v = verdicts[min(n, len(verdicts) - 1)]
        return {"indent": {"verdict": v,
                           "delta_m": (0.3e-9 if v == "cluster"
                                       else -0.2e-9 if v == "tip_changed_or_pit"
                                       else 0.0)}}
    return _f


def cluster(*, round_=True, peaks=1):
    return lambda params, n: {"equivalent_axis_ratio": 0.9 if round_ else 0.2,
                              "is_round": round_, "n_components": peaks}


def onset(passed_at_round=1, onset_v=-0.49):
    """AssessShockleyOnset：第 N 次调用起通过。"""
    def _f(params, n):
        ok = (n + 1) >= passed_at_round
        return {"passed": ok, "onset_v": onset_v if ok else -0.30,
                "reasons": [] if ok else ["onset_out_of_window"],
                "warnings": []}
    return _f


def atomic(passed_at_cycle=1, *, reasons=("no_lattice_peak",)):
    def _f(params, n):
        ok = (n + 1) >= passed_at_cycle
        return {"passed": ok, "period_fast_axis_nm": 0.249 if ok else None,
                "angular_concentration": 500.0 if ok else 2.1,
                "reasons": [] if ok else list(reasons),
                "nm_per_px": 0.0195}
    return _f


def _base_script(**over):
    s = {
        "FindCleanSpot": spot(),
        "MoveToXY": {},
        "TipShapeWithReadback": poke(["cluster"]),
        "AssessClusterRoundness": cluster(),
        "ScanAt": {},
        "SaveScan": SCAN_FILE,
        "GetLatestScanFile": SCAN_FILE,
        "FindFlatRegion": {"center_x_m": 1e-8, "center_y_m": 0.0},
        "ConfigureScan": {}, "StartScan": {}, "StopScan": {},
        "SetBias": {}, "SetSetpoint": {}, "ZControllerOnOff": {},
        "ConfigureLockIn": {}, "ConfigureSTS": {},
        "AcquireSTS": DAT_FILE,
        "BiasWiggle": {"flips_executed": 12},
    }
    s.update(over)
    return s


@pytest.fixture
def au111(monkeypatch):
    """当前样品是 Au(111) —— 两个配方的默认场景。"""
    facts = SubstrateFacts(
        available=True, material="Au(111)", type_id="clean_metal",
        source="sample_record", surface_state_onset_v=-0.49,
        nearest_neighbor_nm=0.2884, row_spacing_nm=0.2498,
        step_height_nm=0.2355)
    monkeypatch.setattr(MST, "resolve_substrate", lambda *a, **k: facts,
                        raising=False)
    import mast.core.sample_facts as SF
    monkeypatch.setattr(SF, "resolve_substrate", lambda *a, **k: facts)
    return facts


def _run(skill_cls, ctx, **params):
    return skill_cls().execute(ctx, params)


# ══ 配方 1:做 STS 的金属性针尖 ═══════════════════════════════════════════════

def test_spectroscopy_tip_ready_on_first_round(au111):
    ctx = FakeCtx(_base_script(AssessShockleyOnset=onset(1)))
    res = _run(MST.MakeSpectroscopyTip, ctx)
    assert res.success, res.error
    assert res.data["outcome"] == "sts_tip_ready"
    assert ctx.count("AcquireSTS") == 1
    assert ctx.count("TipShapeWithReadback") >= 1, "该先扎针再测谱"


def test_spectroscopy_tip_retries_until_the_onset_appears(au111):
    """不达标就再扎再测 —— 判据「否则就再扎再测，直到完成」。"""
    ctx = FakeCtx(_base_script(AssessShockleyOnset=onset(3)))
    res = _run(MST.MakeSpectroscopyTip, ctx, max_rounds=4)
    assert res.data["outcome"] == "sts_tip_ready"
    assert ctx.count("AcquireSTS") == 3
    assert ctx.count("ConfigureLockIn") >= 3


def test_spectroscopy_tip_gives_up_after_max_rounds(au111):
    ctx = FakeCtx(_base_script(AssessShockleyOnset=onset(99)))
    res = _run(MST.MakeSpectroscopyTip, ctx, max_rounds=2)
    assert res.success, "计划跑完了就是 success —— 针尖够不够好写在 outcome 里"
    assert res.data["outcome"] == "sts_rounds_exhausted"
    assert ctx.count("AcquireSTS") == 2
    assert "换针" in res.data["summary_cn"] or "PrepareNobleTip" in res.data["summary_cn"]


# ── 测谱之前先验落点是不是干净台面──────────

def test_a_dirty_surface_stops_the_flow_and_says_it_is_not_about_the_tip(au111):
    """整片表面都找不到台面 ⇒ 停手，而且结论要明说**不是针尖**。

    这一半本来就是对的（``poke_phase`` 内部的 ``flat_poke_sites`` 先拦下了），
    钉住它是因为**它才是要害**：把位置问题记到针尖头上，就会一直扎下去，
    而每一次都在消耗针尖。
    """
    ctx = FakeCtx(_base_script(FindFlatRegion={},          # 哪儿都找不到无台阶窗
                               AssessShockleyOnset=onset(99)))
    res = _run(MST.MakeSpectroscopyTip, ctx, max_rounds=5)
    assert res.success, res.error
    assert res.data["outcome"] in ("surface_spent", "sts_no_clean_terrace"),         res.data["outcome"]
    assert ctx.count("AcquireSTS") == 0, "台面都没有就去测谱了"
    text = res.data["summary_cn"]
    assert "不是针尖" in text, text


def test_the_spectrum_spot_goes_through_the_flat_terrace_check(au111):
    """测谱点要走验台面那一套，不能只拿地图给的坐标就测。

    此前这一步**不存在**：``_relocate`` 只问地图要一个「没标记过」的坐标。
    谱落在台阶边 / 吸附物上时肖克利台阶本来就不该出现，而循环会把它算成
    「针尖还不够金属性」接着扎。
    """
    ctx = FakeCtx(_base_script(AssessShockleyOnset=onset(1)))
    res = _run(MST.MakeSpectroscopyTip, ctx)
    assert res.data["outcome"] == "sts_tip_ready"
    terr = [p for p in res.data["phases"] if p.get("phase") == "terrace"]
    assert terr, "报告里没有一条台面校验的记录 —— 那一步没跑"
    assert "已验过" in str(terr[-1].get("reason") or ""), terr[-1]


def test_a_split_tip_is_reported_as_a_tip_problem_not_a_surface_one(au111):
    """台阶重影 ⇒ 找不到台面的原因**在针尖**，换地方无效，要打脉冲。

    这一态是复用 ``flat_poke_sites`` 白捡的 —— 作者搭的那版会把它误判成
    「表面脏」然后停手，而正确答案是打脉冲。
    """
    assert "sts_split_tip" in MST._OUTCOME_CN
    text = MST._OUTCOME_CN["sts_split_tip"]
    assert "针尖" in text and "换位置无效" in text, text
    assert "脉冲" in text, "没说该做什么 —— 一个说不出下一步的结论等于没有结论"


def test_a_clean_terrace_lets_the_round_proceed_to_the_spectrum(au111):
    """台面验过了就照常测谱 —— 这道新闸不许把正常路挡住。"""
    ctx = FakeCtx(_base_script(AssessShockleyOnset=onset(1)))
    res = _run(MST.MakeSpectroscopyTip, ctx)
    assert res.data["outcome"] == "sts_tip_ready"
    assert ctx.count("AcquireSTS") == 1
    assert ctx.count("FindFlatRegion") >= 1, "没验台面就直接测谱了"


def test_lockin_modulation_is_always_switched_off_at_the_end(au111):
    """**每一条退出路径**都要关调制。留在成像态上，后面每张图都带着抖动。"""
    for script, kw in (
        (_base_script(AssessShockleyOnset=onset(1)), {}),
        (_base_script(AssessShockleyOnset=onset(99)), {"max_rounds": 2}),
    ):
        ctx = FakeCtx(script)
        _run(MST.MakeSpectroscopyTip, ctx, **kw)
        offs = [p for p in ctx.params_for("ConfigureLockIn")
                if p.get("mod_on") is False]
        assert offs, "退出时没有关掉 lock-in 调制"


def test_refuses_before_plunging_when_the_substrate_is_unknown(monkeypatch):
    """定不出衬底就不要开始扎针 —— 扎完才发现判据无从建立是白白消耗针尖。"""
    unknown = SubstrateFacts(available=False, source="unknown",
                             reason="读不到当前样品记录")
    import mast.core.sample_facts as SF
    monkeypatch.setattr(SF, "resolve_substrate", lambda *a, **k: unknown)
    ctx = FakeCtx(_base_script())
    res = _run(MST.MakeSpectroscopyTip, ctx)
    assert res.data["outcome"] == "sts_no_substrate"
    assert ctx.count("TipShapeWithReadback") == 0, "定不出判据却已经动手扎针了"
    assert ctx.count("AcquireSTS") == 0


def test_refuses_on_a_substrate_without_a_surface_state(monkeypatch):
    """Pt(111) 没有肖克利表面态 —— 这不是数据缺失，是物理事实。"""
    pt = SubstrateFacts(available=True, material="Pt(111)", type_id="clean_metal",
                        source="sample_record", surface_state_onset_v=None,
                        reason="Pt(111) 没有肖克利表面态")
    import mast.core.sample_facts as SF
    monkeypatch.setattr(SF, "resolve_substrate", lambda *a, **k: pt)
    ctx = FakeCtx(_base_script())
    res = _run(MST.MakeSpectroscopyTip, ctx)
    assert res.data["outcome"] == "sts_no_surface_state"
    assert ctx.count("TipShapeWithReadback") == 0


def test_sweep_window_follows_the_substrate(au111):
    """谱窗口按 onset 相对定义 —— 换衬底自动跟随，不写死 −0.8..0.3。"""
    ctx = FakeCtx(_base_script(AssessShockleyOnset=onset(1)))
    _run(MST.MakeSpectroscopyTip, ctx)
    cfg = ctx.params_for("ConfigureSTS")[0]
    assert cfg["start_v"] < -0.49 < cfg["end_v"], (
        f"窗口 {cfg['start_v']}..{cfg['end_v']} 没把 onset 包在里面")
    assert cfg["start_v"] == pytest.approx(-0.80, abs=0.02)


def test_spectrum_without_a_saved_path_stops_and_says_why(au111):
    """谱跑完了但没落盘 —— 判不了，但这不是针尖的问题，别算成一次失败的锻造。"""
    ctx = FakeCtx(_base_script(AcquireSTS={}, AssessShockleyOnset=onset(1)))
    res = _run(MST.MakeSpectroscopyTip, ctx)
    assert res.data["outcome"] == "sts_spectrum_unavailable"
    assert "自动保存" in res.data["summary_cn"]


def test_surface_spent_is_reported_not_retried(au111):
    ctx = FakeCtx(_base_script(FindCleanSpot=NO_SPOT))
    res = _run(MST.MakeSpectroscopyTip, ctx)
    assert res.data["outcome"] == "surface_spent"
    assert "换区" in res.data["summary_cn"]


def test_shallow_plunge_is_shallower_than_the_base_conditioning_flow(au111):
    """金属性针尖不一定是最尖的针尖 —— 所以扎得比 PrepareNobleTip 浅。"""
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE
    from mast.core.special_tip_workflow import SPECTROSCOPY_TIP

    assert SPECTROSCOPY_TIP.poke_depth_nm < NOBLE_METAL_BASELINE.poke_depth_nm
    assert SPECTROSCOPY_TIP.critical_repeat_n <= NOBLE_METAL_BASELINE.critical_repeat_n


# ══ 配方 2:原子分辨针尖 ══════════════════════════════════════════════════════

def test_atomic_tip_ready_on_first_cycle(au111):
    ctx = FakeCtx(_base_script(AssessAtomicPhase=atomic(1)))
    res = _run(MST.MakeAtomicResolutionTip, ctx)
    assert res.success, res.error
    assert res.data["outcome"] == "atomic_tip_ready"


def test_by_default_it_keeps_trying_after_the_first_pass(au111):
    """反复搜索时保留质量最好的候选图。"""
    ctx = FakeCtx(_base_script(AssessAtomicPhase=atomic(1)))
    res = _run(MST.MakeAtomicResolutionTip, ctx, max_cycles=4)
    assert ctx.count("BiasWiggle") > 1, "第一次通过就停了 —— 图会累积，不该停"
    best = res.data.get("best_frame")
    assert best is not None and best.get("scan_path"), res.data
    assert any(p.get("phase") == "best" for p in res.data["phases"])


def test_stop_at_first_pass_is_available_for_when_the_tip_is_the_product(au111):
    """要的是**针尖处在好状态**（后面还要接着做别的）时，第一次通过就该停。

    两个目标的最优停机点不一样，所以这是一个参数，不是一个默认。
    """
    ctx = FakeCtx(_base_script(AssessAtomicPhase=atomic(1)))
    res = _run(MST.MakeAtomicResolutionTip, ctx, stop_at_first_pass=True)
    assert res.data["outcome"] == "atomic_tip_ready"
    assert ctx.count("BiasWiggle") == 1


def test_the_reason_for_stopping_does_not_erase_what_was_obtained(au111):
    """跑满轮数 ≠ 没拿到。**停下来的理由**和**有没有拿到东西**是两件事。

    改成「通过之后接着试」之后，正常成功的路径也会走到 for-else ——
    第一版就在那里把 outcome 写成了 cycles_exhausted，而报告里明明躺着
    一张 passed 的确认帧。
    """
    ctx = FakeCtx(_base_script(AssessAtomicPhase=atomic(1)))
    res = _run(MST.MakeAtomicResolutionTip, ctx, max_cycles=3)
    assert res.data["outcome"] == "atomic_tip_ready", res.data["outcome"]
    assert res.data["best_frame"] is not None


def test_wiggle_happens_inside_a_sacrificial_frame_then_a_clean_frame_is_judged(au111):
    """扰动打在牺牲帧里，判据用紧接着的干净帧 —— 顺序不能反。

    扰动期间帧内对比度逐行突变，同帧评估等于拿被污染的证据判针尖。
    """
    ctx = FakeCtx(_base_script(AssessAtomicPhase=atomic(1)))
    _run(MST.MakeAtomicResolutionTip, ctx)
    names = [c for c, _ in ctx.calls]
    i_start = names.index("StartScan")
    i_wig = names.index("BiasWiggle")
    i_stop = names.index("StopScan")
    i_eval = names.index("ScanAt", i_stop)
    i_assess = names.index("AssessAtomicPhase")
    assert i_start < i_wig < i_stop < i_eval < i_assess, (
        f"顺序不对: {names}")


def test_atomic_tip_falls_back_to_plunging_every_few_cycles(au111):
    """「实在搞不出来就退回扎针尖法里面去扎两下，回来再搞」。"""
    ctx = FakeCtx(_base_script(AssessAtomicPhase=atomic(99)))
    res = _run(MST.MakeAtomicResolutionTip, ctx,
               max_cycles=4, cycles_per_fallback=2, fallback_budget=2)
    assert res.data["outcome"] == "atomic_cycles_exhausted"
    assert ctx.count("BiasWiggle") == 4
    # 第 2、4 轮各回退一次；回退用的是扎针。
    assert ctx.count("TipShapeWithReadback") >= 2


def test_fallback_budget_is_respected(au111):
    ctx = FakeCtx(_base_script(AssessAtomicPhase=atomic(99)))
    _run(MST.MakeAtomicResolutionTip, ctx,
         max_cycles=6, cycles_per_fallback=1, fallback_budget=1)
    pokes = ctx.count("TipShapeWithReadback")
    ctx2 = FakeCtx(_base_script(AssessAtomicPhase=atomic(99)))
    _run(MST.MakeAtomicResolutionTip, ctx2,
         max_cycles=6, cycles_per_fallback=1, fallback_budget=0)
    assert ctx2.count("TipShapeWithReadback") == 0
    assert pokes > 0


def test_unjudgeable_frame_stops_instead_of_wiggling_more(au111):
    """「判不了」≠「没有原子相」。

    接着扰动只会白白消耗针尖，而且报告会把一个尺度配置错误说成针尖不行。
    """
    ctx = FakeCtx(_base_script(
        AssessAtomicPhase=atomic(99, reasons=("scale_gate",))))
    res = _run(MST.MakeAtomicResolutionTip, ctx, max_cycles=5)
    assert res.data["outcome"] == "atomic_scale_misconfigured"
    assert ctx.count("BiasWiggle") == 1, "判不了之后还在接着扰动"
    assert "不等于没有原子相" in res.data["summary_cn"]


def test_bad_frame_geometry_is_refused_before_any_hardware_action(au111):
    """帧参数判不出原子相时，一条硬件命令都不该发。"""
    ctx = FakeCtx(_base_script())
    res = _run(MST.MakeAtomicResolutionTip, ctx,
               eval_frame_nm=50.0, eval_pixels=64)     # 0.78 nm/px，远超尺度门
    assert res.data["outcome"] == "atomic_scale_misconfigured"
    assert ctx.calls == [], f"被拒绝的配置却发了命令: {ctx.calls}"


def test_evaluation_frame_uses_the_operator_recipe_values(au111):
    """20 mV / 500 pA / 5 nm —— 要求的配方值要原样传下去。"""
    ctx = FakeCtx(_base_script(AssessAtomicPhase=atomic(1)))
    _run(MST.MakeAtomicResolutionTip, ctx)
    evals = [p for p in ctx.params_for("ScanAt") if "pixels" in p]
    assert evals, "没有一张带显式像素数的评估帧"
    ev = evals[0]
    assert ev["bias_v"] == pytest.approx(0.020)
    assert ev["setpoint_a"] == pytest.approx(500e-12)
    assert ev["size_m"] == pytest.approx(5e-9)
    assert ev["pixels"] == 256


def test_wiggle_envelope_is_passed_through_to_the_primitive(au111):
    ctx = FakeCtx(_base_script(AssessAtomicPhase=atomic(1)))
    _run(MST.MakeAtomicResolutionTip, ctx)
    wig = ctx.params_for("BiasWiggle")[0]
    assert wig["base_bias_v"] == pytest.approx(0.020)
    assert wig["wiggle_upper_v"] == pytest.approx(0.020)
    assert wig["wiggle_lower_v"] > 0, "扰动下限必须大于 0（零附近会撞针）"
    assert wig["burst_s"] <= 10.0
    assert wig["slew_rate_v_per_s"] <= 2.0


def test_no_flat_area_is_reported(au111):
    ctx = FakeCtx(_base_script(FindCleanSpot=NO_SPOT))
    res = _run(MST.MakeAtomicResolutionTip, ctx)
    assert res.data["outcome"] == "surface_spent"


# ══ 共用契约 ═════════════════════════════════════════════════════════════════

def test_both_recipes_declare_tip_shaping(au111):
    for cls in (MST.MakeSpectroscopyTip, MST.MakeAtomicResolutionTip):
        meta = cls().metadata()
        assert "tip_shaping" in meta.capabilities
        assert meta.safety_level.name == "CONFIRM"


def test_skill_names_are_covered_by_both_halt_exemptions():
    """技能名同时要被两张豁免表覆盖，否则配方会把自己中止掉。

    * 电流监控那半边：扰动期间电流本来就会跳；
    * 视觉那半边：刚扎完针去扫小图必然判到「针尖突变」。
    """
    from mast.core.tip_intent import is_tip_work
    from mast.monitoring.service import SUPPRESS_SKILL_PATTERNS

    for name in ("MakeSpectroscopyTip", "MakeAtomicResolutionTip", "BiasWiggle"):
        assert is_tip_work(name), f"{name} 不在 tip_intent.TIP_WORK_PATTERNS 里"
        assert any(p in name.lower() for p in SUPPRESS_SKILL_PATTERNS), (
            f"{name} 不在 monitoring 的 SUPPRESS_SKILL_PATTERNS 里")


def test_plunges_are_recorded_as_damage_on_the_map():
    """扎出的坑必须生成避让圆，否则下一次 FindCleanSpot 会把它当干净表面。"""
    from mast.io.exp_map import _SKILL_KIND_RULES

    def kind(name):
        low = name.lower()
        for needles, k in _SKILL_KIND_RULES:
            if any(nd in low for nd in needles):
                return k
        return ""

    assert kind("MakeSpectroscopyTip") == "tip_shape", (
        "含 'spectr' 会被 sts 规则抢走 —— 扎针损伤会被记成一个谱学点")
    assert kind("MakeAtomicResolutionTip") == "tip_shape"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ── 发证之前：这一帧本身算不算数 ────────────────────────────────────────────

def _atomic_with_halves(first, second, passed=True):
    """整帧判「过」，但两半是 (first, second) —— 模拟帧内针尖突变。"""
    def _f(params, n):
        return {"passed": passed, "period_fast_axis_nm": 0.249,
                "angular_concentration": 500.0,
                "half_concentrations": [first, second],
                "reasons": [], "nm_per_px": 0.0195}
    return _f


def test_a_frame_whose_tip_changed_midway_does_not_certify_the_tip(au111):
    """帧在采集中变化时不得确认质量通过。"""
    ctx = FakeCtx(_base_script(AssessAtomicPhase=_atomic_with_halves(800.0, 8.0)))
    res = _run(MST.MakeAtomicResolutionTip, ctx, max_cycles=2)
    assert res.data["outcome"] != "atomic_tip_ready", (
        "针尖在这一帧中途变了，却发了证")
    hits = [p for p in res.data.get("phases", []) if p.get("mid_frame_tip_change")]
    assert hits, "没记下这一帧是因为帧内突变被否的"
    assert hits[0]["mid_frame_tip_change"]["ratio"] == pytest.approx(0.01)


def test_an_uneven_but_intact_frame_still_certifies(au111):
    """独立合成两半衬度不同但完整的帧，验证衬度不均不能直接当成突变。"""
    ctx = FakeCtx(_base_script(AssessAtomicPhase=_atomic_with_halves(800.0, 160.0)))
    res = _run(MST.MakeAtomicResolutionTip, ctx, stop_at_first_pass=True)
    assert res.data["outcome"] == "atomic_tip_ready", res.data.get("phases")


def test_a_verdict_without_halves_is_not_treated_as_a_tip_change(au111):
    """判据没给两半的数（老版本 / 行太少）⇒ **照旧发证**，不当成突变。

    「读不到」不是「出问题了」—— 这条本仓踩过很多次。
    """
    ctx = FakeCtx(_base_script(AssessAtomicPhase=atomic(1)))   # 不带 half_concentrations
    res = _run(MST.MakeAtomicResolutionTip, ctx, stop_at_first_pass=True)
    assert res.data["outcome"] == "atomic_tip_ready", res.data.get("phases")
