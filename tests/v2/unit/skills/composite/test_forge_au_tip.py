"""ForgeAuTip 外环的每一条分支 —— 尤其是它**不肯**说成功的那些。

用假 ExecutionContext 驱动真正的 GraphExecutor(与 test_prepare_noble_tip 同一套
手法),所以测的是流程的决策:什么时候换位、什么时候停、停下来那句话说的是不是
实话。

## 这个文件最重要的一半是「不谎报」

一条会自己跑一个多小时、中途不问人的流程,最贵的失败模式不是跑不完,是**跑完了
说好了**。下一步会拿一根没修好的针去做实验,而报告说它是好的。所以:预算耗尽是
失败、验收量不出来不算通过、被停下要说是被停下。
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

from mast.core.types import SafetyLevel, SkillResult  # noqa: E402
from mast.skills.composite.forge_au_tip import (  # noqa: E402
    ForgeAuTip,
    _looks_like_operator_stop,
)

#: 让某个技能失败的哨兵(与 test_prepare_noble_tip 同一约定:不能用 None,
#: 那和「script 里没配这个技能」撞了)。
FAIL = object()


class FakeCtx:
    """按技能名派活的假上下文。"""

    def __init__(self, script=None, *, abort_after=None):
        self.script = dict(script or {})
        self.calls: list[tuple[str, dict]] = []
        self._n: dict[str, int] = {}
        self._abort_after = abort_after
        self.run_id = "forge-test"

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

    def safe_call(self, method, *args, role="main"):
        class _R:
            error = ""
            return_value = ("", b"", [0.0])
            method = ""
            args = ()
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


NO_SPOT = lambda params, n: FAIL            # noqa: E731 — 表面用完了


def pulse_ok(dz_nm=40.0):
    def _f(params, n):
        return {"step": {"direction": "up", "delta_m": dz_nm * 1e-9}}
    return _f


def pulse_dead():
    """永远打不动 —— 用来逼出 verify 不过关的路径。"""
    def _f(params, n):
        return {"step": {"direction": "none", "delta_m": 0.0}}
    return _f


def prescan(passed=True, similarity=0.95):
    def _f(params, n):
        return {"tip_ready": passed, "similarity": similarity}
    return _f


PRESCAN_UNREADABLE = lambda params, n: {"tip_ready": False}   # noqa: E731 — sim 缺失


def poke_cluster():
    """一次深扎出簇 → 单峰且圆 → 转临界 → 立刻跳变 → 反复扎达标。"""
    def _f(params, n):
        return {"indent": {"verdict": "cluster", "delta_m": 0.3e-9}}
    return _f


def cluster(round_=True, peaks=1):
    def _f(params, n):
        return {"equivalent_axis_ratio": 0.9 if round_ else 0.2,
                "is_round": round_, "n_components": peaks}
    return _f


def sharpness(verdict="sharp", has_step=True):
    """锐度替身返回判据定义的字段。"""
    from mast.skills.builtins.tip_sharpness import SHARPNESS_VERDICTS
    assert verdict in SHARPNESS_VERDICTS, (
        f"替身在编一个 AssessTipSharpness 说不出的 verdict: {verdict!r}。"
        f"真实词汇表 = {SHARPNESS_VERDICTS}。"
        f"要新增一个态,先在技能里加,再来这里用。")

    def _f(params, n):
        return {"verdict": verdict, "has_step": has_step,
                "edge_resolution_nm": 0.4}
    return _f


def _base_script(**over):
    """一条能一路走到验收的脚本;各测试按需覆盖某一环。"""
    s = {
        "FindCleanSpot": spot(),
        "MoveToXY": {},
        "SetBias": {},
        "SetSetpoint": {},
        "BiasPulseWithReadback": pulse_ok(),
        "PreScanCheck": prescan(True),
        "ScanAt": {},
        "SaveScan": {"path": "C:/tmp/forge.sxm"},
        "GetLatestScanFile": {"path": "C:/tmp/forge.sxm"},
        "AnalyzeFrameTilt": {"step_dominated": False, "surface_rms_m": 5e-11},
        # 合成成功回包的残差必须落在 FindFlatRegion 所接受的范围内。
        "FindFlatRegion": {"center_x_m": 1e-8, "center_y_m": 0.0, "rms_m": 8e-12},
        "AutoTilt": {"action": "applied", "skipped": False},
        "TipShapeWithReadback": poke_cluster(),
        "AssessClusterRoundness": cluster(),
        "AssessTipSharpness": sharpness(),
        "ZControllerOnOff": {},
        "RelocateCoarseXY": {"direction": "x+", "steps": 300},
    }
    s.update(over)
    return s


@pytest.fixture(autouse=True)
def _fast_and_offline(monkeypatch):
    """预算调小(测试要快),并且把粗动大地图换成可控的替身。

    真实 provider 会去读实验记录 —— 单元测试里那是「读不到」,而读不到时外环
    的正确行为是**不换位**,那样就测不到换位路径了。"""
    from mast.core import coarse_map_provider
    from mast.io import coarse_map as cm

    monkeypatch.setattr(coarse_map_provider, "markers_and_config",
                        lambda: ([], cm.CoarseMapConfig()), raising=False)
    monkeypatch.setattr(
        cm, "plan_relocation",
        lambda sites, cfg, steps=None: (
            cm.RelocationPlan(axis="x", direction="x+", steps=300,
                              lands_at=(300, 0), clearance_steps=90.0,
                              reason="test"), ""),
        raising=False)
    # 针尖包络 / Tip Shaper 前置在无硬件下应放行(它们各有自己的测试)。
    monkeypatch.setattr(
        "mast.skills.composite.forge_au_tip._tip_shaper_preflight",
        lambda executor, **kw: False, raising=False)
    monkeypatch.setattr(
        "mast.skills.composite.forge_au_tip._qplus_blocked",
        lambda executor, name, params: False, raising=False)


def _run(script, **params):
    ctx = FakeCtx(script)
    p = {"max_sites": 2, "max_rounds_per_site": 2}
    p.update(params)
    return ForgeAuTip().execute(ctx, p), ctx


# ══════════════════════════════════════════════════════════════════════
# metadata / 门控
# ══════════════════════════════════════════════════════════════════════

def test_metadata_declares_both_capabilities_or_safe_mode_is_a_no_op():
    """不填 capability 的声明式流程会让 SAFE 形同虚设(safe_mode_tip_override)。

    这条外环既打脉冲又扎针 —— 少声明任何一个,SAFE 模式下它就能跑起来。"""
    meta = ForgeAuTip().metadata()
    assert "bias_pulse" in meta.capabilities
    assert "tip_shaping" in meta.capabilities
    assert meta.safety_level is SafetyLevel.CONFIRM   # hooks 强制:字段不可省


def test_safe_mode_refuses_it():
    """SAFE 下这条流程必须被拒 —— 走的是真的门,不是这里重写一遍判据。"""
    from mast.core.safety import is_electrical_pulse, is_tip_shaping

    caps = ForgeAuTip().metadata().capabilities
    assert is_electrical_pulse("ForgeAuTip", {}, caps)
    assert is_tip_shaping("ForgeAuTip", {}, caps)


def test_the_name_is_in_both_suppression_tables():
    """两张表都按**最外层**技能名匹配 —— 外环不在表里,内层子步骤在表里也没用。

    少了 tip_intent 那张:外环自己为看修针效果扫出来的评估图,会让视觉判定把外环
    从中间掐断。少了 monitoring 那张:几十发脉冲、几十次扎针会被读成持续的电流
    异常,整晚刷屏。"""
    from mast.core.tip_intent import is_tip_work
    from mast.monitoring.service import SUPPRESS_SKILL_PATTERNS

    name = ForgeAuTip().metadata().name
    assert is_tip_work(name), f"{name} 不在 tip_intent.TIP_WORK_PATTERNS 里"
    assert any(p in name.lower() for p in SUPPRESS_SKILL_PATTERNS), (
        f"{name} 不在 monitoring 的 SUPPRESS_SKILL_PATTERNS 里")


def test_it_is_discoverable_as_a_skill():
    """注册不到 = 写了等于没写(「注册≠跑过」的另一半)。"""
    from mast.agents.instrument_control.tools import discover_instrument_skills

    names = {m.name for m in discover_instrument_skills().list_skills()}
    assert "ForgeAuTip" in names


# ══════════════════════════════════════════════════════════════════════
# 快扫档
# ══════════════════════════════════════════════════════════════════════

def test_the_forge_scans_are_small_and_do_not_touch_the_operator_table():
    """外环内的评估图走 forge 快档;贵金属流程表本身不能被改。"""
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE as base
    from mast.skills.composite.forge_au_tip import _WF, _forge_wf

    wf = _forge_wf({})
    # 2026-08-14:verify 视野 20 → 100 nm(它现在兼任找台阶),于是**大于**出厂表
    # 的 50。原来那条 ``<= base.verify_scan_nm`` 表达的是「快档的图更小更快」——
    # 而快慢由**线时**决定,不由视野决定(帧时 = 行数 × 线时 × 2,视野不在式子里;
    # 本仓栽过两次「窄条能更快走完」的假注释)。真正不能松的是右边那半:
    # **出厂表本身不许被快档动过**,下面单独钉。
    assert wf.verify_scan_nm == base.forge_scan_nm
    assert wf.step_scan_nm == base.forge_step_scan_nm < base.step_scan_nm
    assert wf.cluster_scan_nm == base.forge_cluster_scan_nm
    assert wf.scan_timeout_s == base.forge_scan_timeout_s
    # ⭐ 「不碰出厂表」—— 这半条比「图更小」重要得多,别的流程还在读这些数。
    assert base.verify_scan_nm == 50.0
    assert base.step_scan_nm == 200.0
    assert base.verify_pixels == 128
    assert base.verify_line_time_s is None, "出厂表的 verify 线时必须留空=走档位表"
    # 判据没被快档动过 —— 快是关于「看多快」,不是关于「什么算好」。
    assert wf.fwdbwd_threshold == base.fwdbwd_threshold
    assert wf.min_axis_ratio == base.min_axis_ratio
    # 出厂表本身不受影响(下一个调用方读到的还是成像用的数)。
    assert base.verify_scan_nm == 50.0 and base.step_scan_nm == 200.0


def test_the_forge_timeout_has_margin_for_the_configured_frame():
    """默认预算覆盖由配置计算的帧时；实际解析工作点另有端到端测试。"""
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE as base
    from mast.core.scan_policy import estimate_scan_seconds

    configured_frame_s = estimate_scan_seconds(base.forge_step_pixels, base.forge_step_line_time_s)
    assert configured_frame_s == pytest.approx(76.8)
    assert base.forge_scan_timeout_s >= 2 * configured_frame_s


# ══════════════════════════════════════════════════════════════════════
# 正常路径
# ══════════════════════════════════════════════════════════════════════

def test_verified_and_accepted_finishes_at_the_first_site():
    res, ctx = _run(_base_script())
    assert res.success, res.error
    assert res.data["outcome"] == "ready"
    assert res.data["sites_worked"] == 1
    assert ctx.count("RelocateCoarseXY") == 0, "第一站就好了,不该再换位"
    assert "达标" in res.data["summary_cn"]


def test_a_spent_surface_relocates_without_asking():
    """「果断」的机制化:表面用完 → 直接换位,没有询问环节。"""
    calls = {"n": 0}

    def spot_then_none(params, n):
        calls["n"] += 1
        # 第一站前两次给点(建结用),之后就没地方了。
        return FAIL if calls["n"] > 1 else {
            "x_m": 5e-8, "y_m": 0.0, "distance_m": 5e-8, "map_known": True}

    res, ctx = _run(_base_script(FindCleanSpot=spot_then_none))
    assert ctx.count("RelocateCoarseXY") >= 1, "表面用完了却没换位"
    reloc = ctx.params_for("RelocateCoarseXY")[0]
    assert reloc["reapproach"] is True, "换完不进针,下一站什么都做不了"
    assert reloc["axis"] == "x" and reloc["steps"] == 300
    assert not res.success
    assert "surface_spent" in str(res.data.get("sites"))


def test_rounds_exhausted_relocates_then_fails_honestly():
    """两站都修不出来 → 如实失败,并给出每站战绩。"""
    res, ctx = _run(_base_script(BiasPulseWithReadback=pulse_dead(),
                                 PreScanCheck=prescan(False, 0.2)))
    assert not res.success
    assert res.data["outcome"] == "sites_exhausted"
    assert res.data["sites_worked"] == 2
    assert ctx.count("RelocateCoarseXY") == 1, "两个站点之间应该只换一次位"
    summary = res.data["summary_cn"]
    assert "未达标" in summary
    assert "站点 #1" in summary and "站点 #2" in summary


def test_budget_exhaustion_is_never_reported_as_success():
    """最贵的失败模式:跑完了预算,报告说好了。"""
    res, _ = _run(_base_script(BiasPulseWithReadback=pulse_dead(),
                               PreScanCheck=prescan(False, 0.2)),
                  max_sites=1)
    assert res.success is False
    assert res.data["outcome"] != "ready"
    assert "未达标" in res.data["summary_cn"]
    # 失败的原因要写进 error,不只是藏在 data 里 —— agent 读的是 error。
    assert "未达标" in (res.error or "")


# ══════════════════════════════════════════════════════════════════════
# 验收:三态,不是两态
# ══════════════════════════════════════════════════════════════════════

def test_an_unmeasurable_sharpness_is_not_a_pass():
    """无法测量时既不能判通过，也不能判失败；最终报告应保留未知状态。"""
    res, _ = _run(_base_script(AssessTipSharpness=sharpness(has_step=False)),
                  max_sites=1)
    accept = res.data["sites"][0]["accept"]
    assert accept["passed"] is False
    assert accept["measurable"] is False
    assert "不是不合格" in accept["reason"] or "没法验收" in accept["reason"]

    # 没验过的事,报告里不许说它验过了。
    summary = str(res.data.get("summary_cn") or "")
    assert "锐度通过验收" not in summary, (
        f"锐度根本没量出来,结案词却说它通过了验收:\n{summary}")


def test_a_failed_sharpness_verdict_is_not_a_pass_either():
    res, _ = _run(_base_script(AssessTipSharpness=sharpness(verdict="blunt")),
                  max_sites=1)
    assert not res.success
    assert res.data["sites"][0]["accept"]["measurable"] is True
    assert res.data["sites"][0]["accept"]["passed"] is False


def test_inconclusive_verify_moves_on_rather_than_pulsing_harder():
    """读不到正反扫描线 ≠ 针尖不好。

    当成不好会让流程接着打脉冲 —— 拿一根可能好好的针尖去撞一个测量问题。"""
    res, ctx = _run(_base_script(PreScanCheck=PRESCAN_UNREADABLE), max_sites=1)
    assert not res.success
    assert res.data["sites"][0]["outcome"] == "verify_inconclusive"
    # 只打了第一轮的脉冲,没有因为「验证没过」就再来一轮。
    assert res.data["sites"][0]["rounds"] == 1


# ══════════════════════════════════════════════════════════════════════
# 换位失败 / 预算
# ══════════════════════════════════════════════════════════════════════

def test_no_relocation_advice_stops_instead_of_walking_blind():
    """大地图说无处可去 = 粗动预算到头。硬走一步是不可撤销的。"""
    from mast.io import coarse_map as cm

    import mast.skills.composite.forge_au_tip as mod
    orig = mod.ForgeAuTip._relocation_plan
    try:
        mod.ForgeAuTip._relocation_plan = lambda self, params: None
        res, ctx = _run(_base_script(FindCleanSpot=NO_SPOT))
    finally:
        mod.ForgeAuTip._relocation_plan = orig
    assert not res.success
    assert res.data["outcome"] == "coarse_budget_exhausted"
    assert ctx.count("RelocateCoarseXY") == 0
    assert cm is not None


def test_an_unreadable_coarse_map_does_not_walk_blind():
    """「不知道去哪」与「去哪都行」是两回事。这台机器没有横向位置反馈。"""
    from mast.core import coarse_map_provider

    skill = ForgeAuTip()
    orig = coarse_map_provider.markers_and_config
    try:
        coarse_map_provider.markers_and_config = lambda: (None, None)
        assert skill._relocation_plan({}) is None
    finally:
        coarse_map_provider.markers_and_config = orig
    assert "不猜着走" in skill._last_map_note


def test_a_failed_relocation_stops_the_loop_and_says_so():
    res, ctx = _run(_base_script(FindCleanSpot=NO_SPOT, RelocateCoarseXY=FAIL))
    assert not res.success
    assert res.data["outcome"] == "relocate_failed"
    assert ctx.count("RelocateCoarseXY") == 1, "换位失败后不许再试一次别的站点"


# ══════════════════════════════════════════════════════════════════════
# 电流监控 CRITICAL —— 外环每轮开工前自己看一眼(2026-08-10)
#
# 起因:2026-08-08 一条 CRITICAL saturation 之后 agent 又扫了 13 分钟,
# 那一轮的对照实验结论作废。这条流程比那更脆弱 —— 它一轮一轮地扎、验、扫,
# 针尖中途崩了没人告诉它,它会把剩下的轮次全部跑在废数据上,而且每轮还要再扎一次。
# ══════════════════════════════════════════════════════════════════════

SAT_ZH = ("隧道电流持续贴轨饱和(段内 100% 的样本达到满量程)"
          "——疑似撞针或前置放大器过载,建议停止扫描并检查针尖。")


def _one_critical(monkeypatch, *, after_calls: int = 0):
    """让 ``_critical_since`` 在第 ``after_calls+1`` 次被问时报一条 CRITICAL。"""
    seen = {"n": 0}

    def fake(watermark):
        seen["n"] += 1
        if seen["n"] > after_calls:
            return [{"id": 3861, "ts": 1.0, "level": "critical",
                     "rule": "saturation", "summary_zh": SAT_ZH, "acked": 0}]
        return []

    monkeypatch.setattr("mast.skills.composite.forge_au_tip._critical_since",
                        fake, raising=True)
    return seen


# ── 瞬变 vs 持续:修针的本职签名不能把修针自己停掉 ────────────────────

class TestOnlySustainedStopsTipWork:
    """修针过程产生的瞬变与持续异常必须分别处理，避免所有 CRITICAL 一律中止。"""

    def test_giant_spike_does_not_stop_tip_work(self):
        """瞬变类 = 本职动作的签名,不是事故。"""
        from mast.skills.composite.forge_au_tip import _halts_tip_work

        assert _halts_tip_work("giant_spike") is False

    def test_sustained_rails_do_stop_tip_work(self):
        """持续类 = 针压进表面出不来 / 信号链死了。修针不该造成持续贴轨。"""
        from mast.skills.composite.forge_au_tip import _halts_tip_work

        assert _halts_tip_work("saturation") is True
        assert _halts_tip_work("freeze") is True

    def test_an_unclassified_new_rule_does_NOT_stop_tip_work(self):
        """**极性在这里是反的,而且反的是对的。**

        投递层的原则是「没分类的新规则默认送达 —— 忘了分类的后果是吵不是哑」。
        这里必须反过来:一条新加的 CRITICAL 规则**不该有权静默地把修针功能废掉**。
        持续型贴轨仍然被另外两道网接着(看门狗真退针 + 投递送到 agent 眼前)。
        """
        from mast.skills.composite.forge_au_tip import _halts_tip_work

        assert _halts_tip_work("some_brand_new_critical_rule") is False
        assert _halts_tip_work("") is False
        assert _halts_tip_work(None) is False

    def test_the_split_comes_from_tip_intent_not_a_second_copy(self):
        """判据必须是 ``core.tip_intent`` 那一份 —— 别新写第二份。

        ⑰-C2 特意把定义搬进 core,理由是「让**会停仪器的那一方**去一个只写措辞的
        模块里 import 判据,方向是反的」。这条对着**产出方**核,不抄字面量。
        """
        from mast.core.tip_intent import (
            SUSTAINED_PHYSICAL_SIGNALS, TRANSIENT_PHYSICAL_SIGNALS,
        )
        from mast.monitoring.alerts import CRIT_RULES, _CRIT_SIGNAL
        from mast.skills.composite.forge_au_tip import _halts_tip_work

        # 每一条 CRIT 规则的去向都由 tip_intent 的两张表决定,一条不漏。
        for rule in CRIT_RULES:
            signal = _CRIT_SIGNAL[rule]
            assert _halts_tip_work(rule) is (signal in SUSTAINED_PHYSICAL_SIGNALS)
        # 而且这两张表确实是互斥的 —— 否则上面那条断言可以两边都成立。
        assert not (SUSTAINED_PHYSICAL_SIGNALS & TRANSIENT_PHYSICAL_SIGNALS)

    def test_a_spike_storm_does_not_stop_the_outer_loop(self, monkeypatch):
        """端到端:库里塞满 giant_spike,外环必须跑完,不是停在第 2 轮。"""
        import mast.skills.composite.forge_au_tip as F

        def only_spikes(_since, limit=20):
            return [{"id": i, "ts": 1.0, "level": "critical",
                     "rule": "giant_spike", "summary_zh": "巨幅瞬变"}
                    for i in range(37)]

        class _S:
            critical_alerts_since = staticmethod(only_spikes)

        monkeypatch.setattr("mast.monitoring.store.get_store_if_exists",
                            lambda: _S(), raising=True)
        assert F._critical_since(0.0) == [], "37 条 giant_spike 把修针停手了"
        res, _ = _run(_base_script())
        assert res.data["outcome"] == "ready", "脉冲自己的签名把外环打死了"


def test_a_critical_before_the_first_round_stops_before_touching_the_tip(monkeypatch):
    """开工前就有 CRITICAL:一发脉冲、一次扎针都不许打出去。"""
    _one_critical(monkeypatch)
    res, ctx = _run(_base_script())
    assert not res.success
    assert res.data["outcome"] == "critical_alert"
    assert ctx.count("BiasPulseWithReadback") == 0, "带着 CRITICAL 还打了脉冲"
    assert ctx.count("TipShapeWithReadback") == 0, "带着 CRITICAL 还扎了针"


def test_a_critical_mid_run_stops_the_remaining_rounds(monkeypatch):
    """第一轮跑完之后崩掉:剩下的轮次**不许**继续跑在废数据上。"""
    # 第一次问(第 1 轮开工前)干净,第二次问(第 2 轮开工前)报 CRITICAL。
    _one_critical(monkeypatch, after_calls=1)
    res, ctx = _run(_base_script(BiasPulseWithReadback=pulse_dead(),
                                 PreScanCheck=prescan(False, 0.2)),
                    max_sites=2, max_rounds_per_site=3)
    assert res.data["outcome"] == "critical_alert"

    # 用**轮数**而不是脉冲发数来钉:一轮 pulse_phase 本来就会打十几发,
    # 拿发数断言等于把「一轮打几发」这个无关的实现细节焊进这条测试。
    site = res.data["sites"][0]
    assert site["rounds"] == 1, "第 1 轮跑完就该停,不该进第 2 轮"
    assert site["critical_alert"]["round_would_have_been"] == 2
    pulses_in_round_1 = ctx.count("BiasPulseWithReadback")
    assert pulses_in_round_1 > 0, "第 1 轮本来就该正常打完"

    # 反证:没有 CRITICAL 时,同一份脚本会跑满 3 轮 —— 说明上面停住的是 CRITICAL,
    # 不是别的什么东西恰好也让它停在第 1 轮。
    #
    # 必须**重新**打桩:上面那个 fake 带着自己的调用计数,直接再跑一遍会立刻
    # 命中 CRITICAL,于是这条反证会把「两次都停了」误读成「对照成立」。
    monkeypatch.setattr("mast.skills.composite.forge_au_tip._critical_since",
                        lambda watermark: [], raising=True)
    res2, ctx2 = _run(_base_script(BiasPulseWithReadback=pulse_dead(),
                                   PreScanCheck=prescan(False, 0.2)),
                      max_sites=2, max_rounds_per_site=3)
    assert res2.data["sites"][0]["rounds"] == 3
    assert ctx2.count("BiasPulseWithReadback") > pulses_in_round_1


def test_the_watermark_is_one_per_outer_loop_not_one_per_site(monkeypatch):
    """水位线必须**整条外环一个**,跨站不重建。

    每站重建会让两段最危险的时间掉进缝里:第一站的第 1 轮(窗口 ≈0),
    以及**站与站之间的换位 + 重新进针** —— 撞针最容易发生在重新进针那一下。

    ⚠️ 这条测的是**调用点**,不是 ``_CriticalWatch`` 自己。第一版写成了直接
    构造 watch 来测,于是「把 watch 挪进每站」的变异**没能把它弄红** ——
    测了原语、没测可达性,正是本仓反复出事的那个形状。
    """
    import mast.skills.composite.forge_au_tip as F

    # 注入一个**计数钟**:每读一次 +1。于是「谁消耗了一次读数」变得可数,
    # 而「窗口被重置」就表现为水位线序列里**跳号**——不用靠时间差和 epsilon。
    class _Clock:
        def __init__(self):
            self.n = 0

        def time(self):
            self.n += 1
            return float(self.n)

    monkeypatch.setattr(F, "_time", _Clock(), raising=True)

    asked: list[float] = []

    def spy(w):
        asked.append(w)
        return []

    monkeypatch.setattr(F, "_critical_since", spy, raising=True)

    # 两站:第一站表面用完 → 换位 → 第二站。
    calls = {"n": 0}

    def spot_then_none(params, n):
        calls["n"] += 1
        return FAIL if calls["n"] > 1 else {
            "x_m": 5e-8, "y_m": 0.0, "distance_m": 5e-8, "map_known": True}

    _run(_base_script(FindCleanSpot=spot_then_none), max_sites=2)

    assert len(asked) >= 2, "外环里查得太少,覆盖不了跨站"
    # 一个 watch 贯穿全程 ⇒ 每次查用的水位线正是**上一次查完那一刻**存下的值。
    #
    # ⚠️ 2026-08-14 判据从「连号」改成「**严格递增**」。原来的连号断言隐含了
    # 「查的次数是固定的」这个假设,而那一天精修相加了途中查(``should_stop``
    # 每次扎针前问一句)—— 查得更勤,序号自然跳。跳号本身不是缺陷,
    # **回退**才是:水位线只要往回走,中间那段就没人看了。
    #
    # 这条测试真正要防的是「有人在中途重建了 watch」,而重建的特征是水位线
    # **变小或重置**,不是跳号。跨站那一段(换位 + 重新进针,撞针最容易发生的
    # 地方)只要没被跳过就行 —— 严格递增保证了这一点。
    assert all(b > a for a, b in zip(asked, asked[1:])), (
        f"水位线回退了 —— 有人在中途重建了 watch,退回去的那一段会被重复看/漏看"
        f":{asked}")
    assert asked[0] == pytest.approx(2.0), (
        f"水位线不是在外环开工那一刻建的(首次查用的是 {asked[0]})—— "
        "建晚了的话,第一站第一轮的窗口宽度就是 0")


def test_the_refine_phase_is_covered_too(monkeypatch):
    """精修前也要查:它是唯一会反复深扎的一段,带着贴住表面的针进去最贵。

    原来这道查只在 ``for rnd`` 里,``poke_phase`` 与验收完全不经过它。
    """
    # 第 1 轮开工前干净 → 大修/验证通过 → **精修开始前**那一次命中。
    _one_critical(monkeypatch, after_calls=1)
    res, ctx = _run(_base_script())
    assert res.data["outcome"] == "critical_alert"
    assert res.data["sites"][0]["critical_alert"]["caught_at"] == "精修开始前"
    assert ctx.count("TipShapeWithReadback") == 0, "带着 CRITICAL 还去精修深扎了"


def test_a_critical_does_not_relocate_to_another_site(monkeypatch):
    """CRITICAL 是**整条流程**的停止理由,不是「这一站不行」。

    换到下一站再扎一次,是在一根可能已经压进表面的针上继续动手 —— 而换位本身
    (RelocateCoarseXY)也要用这根针。
    """
    _one_critical(monkeypatch)
    res, ctx = _run(_base_script(), max_sites=3)
    assert res.data["outcome"] == "critical_alert"
    assert ctx.count("RelocateCoarseXY") == 0, "带着 CRITICAL 还换位再扎"
    assert res.data["sites_worked"] == 1


def test_the_critical_report_carries_the_alert_own_words(monkeypatch):
    """报告要带告警自己的话 ——「外环中止」四个字对用户没有信息量。"""
    _one_critical(monkeypatch)
    res, _ = _run(_base_script())
    summary = res.data["summary_cn"]
    assert "saturation" in summary
    assert "疑似撞针或前置放大器过载" in summary
    # **不能**建议提高预算:停手的理由不是预算不够。
    assert "提高预算" not in summary
    assert "先查针尖和信号链" in summary


def test_an_unreadable_monitor_does_not_block_the_run(monkeypatch):
    """监控库读不到(纯离线 / numpy 缺席)时照常开工 —— 这是额外一道网,不是唯一一道。"""
    import mast.skills.composite.forge_au_tip as F

    def boom():
        raise RuntimeError("no monitoring store")

    monkeypatch.setattr("mast.monitoring.store.get_store_if_exists", boom,
                        raising=True)
    assert F._critical_since(0.0) == []
    res, _ = _run(_base_script())
    assert res.data["outcome"] == "ready"


def test_the_selfcheck_never_creates_the_monitoring_db(monkeypatch):
    """只读的自检不许有副作用 —— ``get_store()`` 会**建库建目录**。

    在测试里那意味着写进用户真实的 experiments 目录,而 conftest 对这个库没有
    像 wishlist / 文献注册表那样的 autouse 守卫。本仓「测试污染真实数据」已四次。
    """
    import mast.monitoring.store as S

    called = {"creating": 0}
    monkeypatch.setattr(S, "get_store",
                        lambda: called.__setitem__("creating",
                                                   called["creating"] + 1),
                        raising=True)
    monkeypatch.setattr(S, "get_store_if_exists", lambda: None, raising=True)

    import mast.skills.composite.forge_au_tip as F
    assert F._critical_since(0.0) == []
    assert called["creating"] == 0, "自检调用了会建库的 get_store()"


# ══════════════════════════════════════════════════════════════════════
# stopped_early —— 用户按了停止是事实,不是判断
# ══════════════════════════════════════════════════════════════════════

def test_the_stop_marker_matches_what_the_scan_composites_actually_write():
    """跨模块的停止标记应完整传递。"""
    from mast.skills.composite.graph_executor import CompositeProgress
    from mast.skills.composite.scan_at import ScanAt

    progress = CompositeProgress(composite_name="ScanAt", total_steps=1)
    progress.partial_data["wait_stopped_early"] = True
    progress.partial_data["scan_lines_done"] = 9
    progress.partial_data["scan_lines_total"] = 256
    ok, reason = ScanAt()._decide_outcome(True, progress, {})
    assert ok is False
    assert _looks_like_operator_stop(reason), (
        f"ScanAt 的中途停止文案已经变了,ForgeAuTip 认不出来了:{reason!r}")


def test_an_ordinary_failure_is_not_called_an_operator_stop():
    assert not _looks_like_operator_stop("扫描在超时时间内没有完成")
    assert not _looks_like_operator_stop("")


def test_a_stopped_scan_aborts_the_outer_loop_and_is_reported_as_such():
    """§2.24:用户按停是事实,外环照此中止,而且要说清是被停的。"""
    script = _base_script(
        PreScanCheck=lambda params, n: FAIL)
    ctx = FakeCtx(script)
    skill = ForgeAuTip()
    # 让 PreScanCheck 的失败原文带上真正的「中途停止」措辞。
    def _prescan_stopped(params, n):
        return FAIL
    ctx.script["PreScanCheck"] = _prescan_stopped
    orig_run = ctx.run

    def run_with_stop_text(name, params, version=None):
        res = orig_run(name, params, version)
        if name == "PreScanCheck":
            res.error = "预扫描中途停止 (9/256 行) —— 这条线没扫完,测不出针尖状态。"
        return res

    ctx.run = run_with_stop_text
    res = skill.execute(ctx, {"max_sites": 2, "max_rounds_per_site": 2})
    assert not res.success
    assert res.data["outcome"] == "stopped_early"
    assert "用户" in res.data["summary_cn"]
    assert ctx.count("RelocateCoarseXY") == 0, "被停下时不该继续换位再干"


# ══════════════════════════════════════════════════════════════════════
# 战绩记账
# ══════════════════════════════════════════════════════════════════════

def test_the_record_survives_an_abort_midway():
    """被中止时,报告里必须还有「刚才做了什么」。

    战绩攒到最后再写 = 一被打断就什么都没有,而被打断恰恰是最需要知道刚才做了
    什么的时候。"""
    ctx = FakeCtx(_base_script(), abort_after=6)
    res = ForgeAuTip().execute(ctx, {"max_sites": 2, "max_rounds_per_site": 2})
    assert not res.success
    assert res.data.get("sites"), "中止后战绩全丢了"


def test_the_summary_does_not_claim_an_unmeasured_state():
    """收尾报文来自实时回读而非计划值；读取失败必须如实报告。"""
    res, _ctx = _run(_base_script(BiasPulseWithReadback=pulse_dead(),
                                  PreScanCheck=prescan(False, 0.2)),
                     max_sites=1)
    s = res.data["summary_cn"]
    assert "实测回读" in s, s
    # 计划值不许出现在报文里(FakeCtx 的回读一律给 0,与流程表的 0.05 V / 1000 pA
    # 都不同 —— 所以「照抄计划」在这里是可分辨的)。
    assert "0.05 V" not in s, s
    assert "1000 pA" not in s, s
    # 不许出现无条件的「仪器留在 X」这种陈述句。
    assert "仪器**留在" not in s, s
    assert "仪器留在" not in s, s


def test_summary_lists_every_site_it_touched():
    res, _ = _run(_base_script(BiasPulseWithReadback=pulse_dead(),
                               PreScanCheck=prescan(False, 0.3)))
    sites = res.data["sites"]
    worked = [s for s in sites if s.get("site")]
    assert len(worked) == 2
    for s in worked:
        assert s["rounds"] == 2, "每站都该把轮数用满才换位"
        assert s["outcome"] == "verify_exhausted"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


# ══════════════════════════════════════════════════════════════════════
# 缺陷⑮(2026-08-06 首演):覆盖值到不了门控 —— 因为它根本没被声明过
# ══════════════════════════════════════════════════════════════════════
#
# 首演实况:agent 连试 poke_depth_nm 0.5 / 0.4,门控每次都回同一句「超出 W-qPlus
# 0.5 nm 包络」,而句子里的数字是流程表的 **1.5**。
#
# 症状看起来像「执行层认覆盖、门控层读表」(两处不同源)。**实际根因不是那个**:
# ForgeAuTip 的 ParameterSpec 里**根本没有** poke_depth_nm ⇒ 工具 schema 里没有这个
# 字段 ⇒ pydantic 默认 extra=ignore ⇒ 模型传的值被**静默丢弃** ⇒ 执行层和门控层
# **都**回落到流程表。两处不是不同源,是两处都没拿到。
#
# 这种失败从两端都看不出来:调用方以为自己传了,技能以为没人传。而 `_forge_wf` 当时
# 从 params 里读 22 个键,其中 16 个是这样的**死读** —— 读起来像可调项,调它没有任何
# 效果。


def test_every_key_the_workflow_reads_is_a_declared_parameter():
    """**死读一个都不许有。**

    `_forge_wf` 从 params 里读的每个键,都必须是 ForgeAuTip 声明过的参数 —— 否则
    模型传了会被工具层静默丢掉,而值一路回落到流程表。加读一个键就必须同时声明它。
    """
    from mast.skills.composite.forge_au_tip import _WF_PARAM_KEYS

    declared = {p.name for p in ForgeAuTip().metadata().parameters}
    dead = sorted(set(_WF_PARAM_KEYS) - declared)
    assert not dead, f"读了但没声明(模型传了会被静默丢弃):{dead}"


def test_the_poke_depth_override_reaches_the_tool_schema():
    """探针有效性 + 根因那一条:这个键必须出现在**模型看得见的** schema 里。

    只断言 metadata 里有是不够的 —— 中间隔着一层 args_schema,而当初断在那一层的。
    """
    from mast.agents._shared.skill_adapter import wrap_skill

    tool = wrap_skill(ForgeAuTip, lambda: None)
    fields = tool.args_schema.model_fields
    assert "poke_depth_nm" in fields, sorted(fields)
    # 物理量在本仓是以**字符串**进 schema 的(防指数丢失的那套机制),
    # 由 _coerce_si_params 转回 float。这里连带钉住那条链没断。
    from mast.agents._shared.skill_adapter import _coerce_si_params

    kw, errs = _coerce_si_params(ForgeAuTip().metadata(), {"poke_depth_nm": "0.5"})
    assert not errs and kw["poke_depth_nm"] == pytest.approx(0.5)


def test_the_gate_judges_the_depth_that_will_actually_run():
    """传 0.5 用 0.5 判(过)、传 2.0 用 2.0 判(拒)、留空用表值判。

    门控的取值路径必须和执行的取值路径是同一条 —— 否则拒绝理由里的数字不是用户
    传的那个,而会照着一个不存在的约束改参数,来回改了两次都没用。
    """
    from mast.core import tip_state
    from mast.skills.composite.forge_au_tip import _WF, _forge_wf

    from mast.core.tip_conditioning_policy import resolve_policy

    prev = tip_state.get_current_tip()
    try:
        facts = {"material": "W", "form": "qplus",
                 "id": "t-gate", "name": "gate-test"}
        tip_state.set_current_tip(facts)
        # 探针值**跟着包络走**,不写死:qPlus 的下压包络曾从 0.5 nm 抬到 5 nm,
        # 而这条测试原来的 0.5/2.0 两个探针一个变成「远低于」、
        # 一个变成「合法」—— 断言仍然全绿,测的却已经不是拒绝路径了。
        limit_nm = float(resolve_policy(facts)["max_poke_depth_m"]) * 1e9
        ok_nm, over_nm = limit_nm * 0.5, limit_nm * 1.2
        assert ForgeAuTip().validate_params({"poke_depth_nm": ok_nm}) == []
        deep = ForgeAuTip().validate_params({"poke_depth_nm": over_nm})
        assert deep and f"{over_nm * 1e-9:.3e}" in deep[0], deep
        # 执行侧拿到的是同一个数(同源)。
        assert _forge_wf({"poke_depth_nm": ok_nm}).poke_depth_nm == pytest.approx(ok_nm)
        # 留空:出厂默认在这根针的包络内(2026-08-10 起成立),所以对账不动它。
        assert _forge_wf({}).poke_depth_nm == pytest.approx(_WF.poke_depth_nm)
    finally:
        tip_state.set_current_tip(prev)


def test_an_unreadable_envelope_does_not_block_the_run():
    """包络读不到不是拒绝的理由 —— 没登记针尖时不该把修针卡死。

    fail-open 在这里是对的:包络是**额外**的保护,不是执行的前提;而技能自己的
    ParameterSpec 边界仍然在。
    """
    from mast.core import tip_state

    prev = tip_state.get_current_tip()
    try:
        tip_state.set_current_tip(None)
        assert ForgeAuTip().validate_params({"poke_depth_nm": 2.0}) == []
    finally:
        tip_state.set_current_tip(prev)


# ══════════════════════════════════════════════════════════════════════
# 起伏弃权门:**分叉本身**才是这条要钉的东西(2026-08-10)
# ══════════════════════════════════════════════════════════════════════

def prescan_abstain():
    """PreScanCheck 弃权:数据**读到了**,只是这块地方太平,判据用不上。

    这不是 ``PRESCAN_UNREADABLE``(那条是「读不到」)—— 两者都让 ``similarity``
    为 ``None``,但指向完全不同的下一步:那条查通信,这条**换个有形貌的地方重量**。
    """
    def _f(params, n):
        return {"tip_ready": None, "similarity": None,
                "recommendation": "inconclusive",
                "corrugation_rms_m": 8.3e-12,
                "abstain_reason": (
                    "去趋势起伏只有 8.3 pm,低于下限 15.0 pm —— 没有形貌就没有可"
                    "相关的信号。**这是弃权,不是「针尖不合格」**。")}
    return _f


def test_low_corrugation_relocates_instead_of_pulsing():
    """用合成对照验证起伏不足时换区，而明确不合格时才继续处理；检查实际分支与脉冲次数。"""
    res_abstain, ctx_abstain = _run(_base_script(PreScanCheck=prescan_abstain()))
    res_bad, ctx_bad = _run(_base_script(PreScanCheck=prescan(False, 0.2)))

    n_abstain = ctx_abstain.count("BiasPulseWithReadback")
    n_bad = ctx_bad.count("BiasPulseWithReadback")

    assert n_abstain < n_bad, (
        f"弃权和判「坏」走了同一条路(各 {n_abstain} / {n_bad} 次脉冲)—— "
        "「判不了」被当成了「针尖坏」,而那意味着拿脉冲去打一根可能好好的针。")
    # 2026-08-17:弃权那一侧从 **2 发降到 1 发**,而且那 1 发是**故意的**。
    #
    #   · 站点内环改成「到站先验」⇒ 判不了时不再白打一批脉冲才发现;
    #   · 但**第一站的那一发是强制的** —— 要求:「我认为修针尖的第一个脉冲
    #     一定要是强制的。」他发起修针,就是因为他认为针尖需要修,不该让一个
    #     阈值 0.80 的重合度判据把他的判断否掉。
    #
    # 所以 1 = 第一站强制那一发;第二站(粗动换区来的)先验之后一发没打。
    # 这条测试的论证仍然成立,而且更清楚了:
    #     误判「坏」→ 打 4 发 …… **不可逆**
    #     弃权     → 打 1 发(强制那发)后换地方重测 …… 可逆
    assert n_abstain == 1 and n_bad == 4, (
        f"脉冲次数变了(弃权 {n_abstain}、判坏 {n_bad});"
        "如果是流程有意改动请更新这条,但先确认弃权那一侧**仍然更少**。")

    # 走的是 inconclusive 那条腿,而且**每一站**都是
    assert res_abstain.data["outcome"] == "sites_exhausted"
    assert all(s.get("outcome") == "verify_inconclusive"
               for s in res_abstain.data["sites"] if s.get("site")), \
        res_abstain.data["sites"]
    # 换位确实发生了(「换个地方重量」不能只是嘴上说)
    assert ctx_abstain.count("RelocateCoarseXY") == 1

    # 而且**不谎报**:弃权不是通过
    assert res_abstain.success is False
    assert "未达标" in res_abstain.data["summary_cn"]


def test_the_abstain_reason_survives_to_the_report():
    """「为什么判不了」要能读到 —— 「太平」和「读不到」指向不同的下一步。

    这条挡的是「把三种判不了折叠成同一句话」那个回归(prescan_check.py 已经因为
    把三种读失败折叠成一个 None 栽过一次)。
    """
    res, _ = _run(_base_script(PreScanCheck=prescan_abstain()), max_sites=1)
    verify = [p for s in res.data["sites"] if s.get("site")
              for p in s.get("phases", []) if p.get("phase") == "verify"]
    assert verify, res.data["sites"]
    assert "起伏" in verify[0]["reason"], verify[0]
    assert "读不到" not in verify[0]["reason"], (
        "起伏不足被说成了「读不到正反扫描线数据」—— 数据读得好好的,"
        "而这两句话指向的下一步完全不同。")


def prescan_unmeasurable():
    """PreScanCheck 弃权:图没扫完,裁完仍然没有可比较的区域。

    与 ``prescan_abstain``(太平)、``PRESCAN_UNREADABLE``(读不到)并列的第三种
    「判不了」—— 三种都让 ``similarity`` 为 ``None``,下一步各不相同。
    """
    def _f(params, n):
        return {"tip_ready": None, "similarity": None,
                "recommendation": "inconclusive",
                "rows_total": 256, "rows_acquired": 1,
                "abstain_reason": (
                    "这一帧只有 1/256 行是扫完整的 —— 这一帧还没有可比较的区域。"
                    "**这是弃权,不是「针尖不合格」**。")}
    return _f


def test_a_half_scanned_frame_relocates_instead_of_pulsing():
    """不完整帧先裁到已扫行；剩余数据不足时弃权，不把 NaN 比较的 False 当成坏针尖。"""
    res_unmeas, ctx_unmeas = _run(_base_script(PreScanCheck=prescan_unmeasurable()))
    res_bad, ctx_bad = _run(_base_script(PreScanCheck=prescan(False, 0.2)))

    n_unmeas = ctx_unmeas.count("BiasPulseWithReadback")
    n_bad = ctx_bad.count("BiasPulseWithReadback")
    assert n_unmeas < n_bad, (
        f"「没扫完」和「针尖坏」走了同一条路(各 {n_unmeas} / {n_bad} 次脉冲)—— "
        "一张被中断的扫描触发了不可逆动作。")
    assert all(s.get("outcome") == "verify_inconclusive"
               for s in res_unmeas.data["sites"] if s.get("site")), \
        res_unmeas.data["sites"]
    assert ctx_unmeas.count("RelocateCoarseXY") == 1
    assert res_unmeas.success is False        # 弃权不是通过


# ── 到站先验(2026-08-17) ────────────────────────────────────────────────────

def test_the_first_pulse_of_a_run_is_mandatory():
    """**修针的第一发脉冲一定要打** —— 判据说针尖好也照打。

    「我认为修针尖的第一个脉冲一定要是强制的。」

    到站先验(见下一条)是为了解决「粗动换区都会自动带一个 pulse」,而那件事的
    前提是**针尖状态已经在上一站量过了**。整条流程刚开工没有这个前提:他发起
    修针,就是因为他认为针尖需要修 —— 让一个阈值 0.80 的重合度判据把他的判断
    否掉,是「未标定的阈值否决人」的又一次。
    """
    res, ctx = _run(_base_script(PreScanCheck=prescan(True, 0.99)))
    assert ctx.count("BiasPulseWithReadback") >= 1, (
        "判据说针尖好,第一站就一发不打了 —— 第一发是强制的")
    site1 = (res.data.get("sites") or [{}])[0]
    assert site1.get("pulse_skipped_reason") is None
    assert site1.get("forced_repair"), (
        "第一站没打先验、却没在记录里说明为什么 —— 那会被读成「判据放行了」")


def test_a_good_tip_costs_no_pulses_from_the_second_site_onward():
    """后续站点的验证已通过时跳过脉冲，直接进入台阶与精修阶段。"""
    # 弃权脚本会在第一站打完强制那发之后判不了 → 换区 → 第二站先验。
    # 全程只有第一站那一发:第二站的先验确实把脉冲挡住了。
    res, ctx = _run(_base_script(PreScanCheck=prescan_abstain()))
    assert ctx.count("RelocateCoarseXY") == 1, "没走到第二站,这条就没测到东西"
    assert ctx.count("BiasPulseWithReadback") == 1, (
        f"第二站又打脉冲了(共 {ctx.count('BiasPulseWithReadback')} 发)—— "
        "到站先验没生效")


def test_force_repair_overrides_the_entry_check():
    """``force_repair=true`` ⇒ 不问判据,先大修。

    判据看的是正反扫描线重合度(阈值 0.80),它**量不到**的针尖毛病会让到站先验
    放行。用户知道针尖不好而判据说好时,逃生门比判据管用 ——
    「未标定的阈值没资格否决人」的同一条纪律。
    """
    res, ctx = _run(_base_script(PreScanCheck=prescan_abstain()),
                    force_repair=True)
    # 弃权脚本会换到第二站;``force_repair`` 让第二站也照打,
    # 所以脉冲数比不设它时多(不设时第二站先验之后一发不打)。
    assert ctx.count("BiasPulseWithReadback") > 1, (
        "force_repair=true 却仍然被到站先验拦下了 —— 逃生门没接上")
