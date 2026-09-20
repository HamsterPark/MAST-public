"""原子像验证区分不合格与无法判断。"""
from __future__ import annotations

import inspect
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

from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite import verify_atomic_resolution as VAR  # noqa: E402
from mast.skills.composite.verify_atomic_resolution import (  # noqa: E402
    ALL_REMEDIES,
    ALL_VERDICTS,
    REASON_VERDICT,
    VERDICT_ABSENT,
    VERDICT_RESOLVED,
    VERDICT_UNDECIDABLE,
    VerifyAtomicResolution,
    atomic_scale_reject,
    classify_reasons,
    evaluate_frame_admission,
)
from mast.vision.atomic_phase import ALL_REASONS  # noqa: E402

# 源码级断言一律走它,不用 ``inspect.getsource``(2026-08-15)——
# ``not in`` + ``getsource`` 是假绿组合,理由见该模块 docstring 与下面那条测试。
from tests.v2.srcref import source_of  # noqa: E402

FAIL = object()


class FakeCtx:
    """按技能名派活的假上下文(与 test_make_special_tip 同款)。"""

    def __init__(self, script=None):
        self.script = dict(script or {})
        self.calls: list[tuple[str, dict]] = []
        self._n: dict[str, int] = {}
        self.run_id = "test-verify-atomic"

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


# ── 脚本片段 ──────────────────────────────────────────────────────────────────

GOOD_METRICS = {
    "nan_frac": 0.0,
    "rowcorr_median": 0.85,
    "fb_instability": 0.10,
    "artifacts": {"bad_row_frac": 0.02},
    "atomic": {"passed": True, "reasons": []},
}


def analyze(metrics=None, profile="synthetic-test-profile", provenance="独立合成测试配置"):
    m = GOOD_METRICS if metrics is None else metrics
    return lambda params, n: {
        "scan_path": params.get("scan_path"),
        "metrics": dict(m),
        "plan": {"threshold_profile": profile,
                 "threshold_provenance": provenance},
    }


def assessed(*, passed=False, reasons=(), nm_per_px=0.0098, scale="full",
             period=0.249, conc=1500.0, sharp=22.0, snr=30.0):
    def _f(params, n):
        return {
            "passed": passed, "scale": scale, "nm_per_px": nm_per_px,
            "period_nm": period, "period_fast_axis_nm": period,
            "snr": snr, "angular_concentration": conc, "order_ratio": 0.21,
            "fft_sharpness": sharp, "expected_a_nm": params.get("expected_a_nm"),
            "slow_axis_trusted": False,
            "reasons": list(reasons), "warnings": [],
        }
    return _f


def run_skill(script, **params):
    ctx = FakeCtx(script)
    p = {"scan_path": "/tmp/f0001.sxm"}
    p.update(params)
    result = VerifyAtomicResolution().execute(ctx, p)
    return ctx, result


# ══════════════════════════════════════════════════════════════════════
# 三态映射:穷举是硬要求
# ══════════════════════════════════════════════════════════════════════

def test_every_reason_word_has_a_home():
    """映射表对 ``atomic_phase.ALL_REASONS`` **穷举**。

    不能硬编码一串字符串再逐条断言 —— 那样判据新增出局词时这里会静默漏判,
    而漏判的默认落点多半是「没有原子分辨」(最坏的那一档)。
    """
    assert set(REASON_VERDICT) == set(ALL_REASONS), (
        f"判据有而映射表没有: {set(ALL_REASONS) - set(REASON_VERDICT)};"
        f"映射表有而判据没有: {set(REASON_VERDICT) - set(ALL_REASONS)}")


@pytest.mark.parametrize("reason", sorted(ALL_REASONS))
def test_each_reason_maps_to_a_legal_verdict_and_remedy(reason):
    """每个出局词一条用例:态在闭集里,而且「判不了」必须给得出下一步。"""
    verdict, remedy = REASON_VERDICT[reason]
    assert verdict in ALL_VERDICTS
    if verdict == VERDICT_UNDECIDABLE:
        assert remedy in ALL_REMEDIES, f"{reason} 判不了却说不出下一步"
    else:
        assert verdict == VERDICT_ABSENT and remedy is None

    got = classify_reasons(False, [reason])
    assert got["verdict"] == verdict
    assert got["remedy"] == remedy


def test_scale_reduced_is_undecidable_not_absent():
    """**本设计最容易被写反的一条。**

    ``scale_reduced`` 在判据里被塞进 ``reasons`` 让 ``passed=False``,长得和
    「没过」一模一样 —— 但判据模块自己写的是「在那个尺度上『有原子相』这句话的
    证据强度撑不住一次针尖验收」。**是证据不足,不是没有。**

    归错的后果:5.12-10 nm 这一整段(出厂 `atomic` 档的上半段就落在这里)会稳定地
    报「这个偏压上没有原子分辨」—— 一句彻头彻尾的假话。
    """
    got = classify_reasons(False, ["scale_reduced"])
    assert got["verdict"] == VERDICT_UNDECIDABLE
    assert got["remedy"] == "shrink_field"


def test_scale_reduced_wins_over_criterion_words():
    """过渡带上判据同时报了「不是晶格」—— 仍然是**判不了**,不是「没有」。

    这是 5.12-10 nm 上最常见的组合:尺度不够时判据当然容易说「不是晶格」,
    但那个结论在那个尺度上没有说的资格。判不了压倒没有。
    """
    got = classify_reasons(
        False, ["no_lattice_peak", "not_a_lattice", "scale_reduced"])
    assert got["verdict"] == VERDICT_UNDECIDABLE
    assert got["remedy"] == "shrink_field"
    assert got["absent_reasons"] == ["no_lattice_peak", "not_a_lattice"]


def test_scale_gate_and_unknown_pixel_size_are_two_different_undecidables():
    """两种判不了,补救不同:一个去缩视野,一个去查 .sxm 头解析与几何。

    ``make_special_tip`` 现在只判了 ``scale_gate`` 一个词 —— S2 这一版把两个都判上。
    """
    a = classify_reasons(False, ["scale_gate"])
    b = classify_reasons(False, ["unknown_pixel_size"])
    assert a["verdict"] == b["verdict"] == VERDICT_UNDECIDABLE
    assert a["remedy"] == "shrink_field"
    assert b["remedy"] == "fix_pixel_scale"
    assert a["remedy"] != b["remedy"]


def test_pure_criterion_words_are_absent():
    got = classify_reasons(False, ["no_lattice_peak", "fft_not_sharp"])
    assert got["verdict"] == VERDICT_ABSENT
    assert got["remedy"] is None


def test_passed_with_no_reasons_is_resolved():
    got = classify_reasons(True, [])
    assert got["verdict"] == VERDICT_RESOLVED
    assert got["remedy"] is None


def test_an_unknown_reason_word_falls_to_undecidable_not_absent():
    """兜底档是「判不了」而**不是**「没有」。

    映射表对 ALL_REASONS 穷举,所以这一档在测试通过时不可达 —— 它防的是判据
    新增出局词而映射表没跟上的那一天。那一天最不该发生的事,就是拿一个没人认识
    的词去下「没有原子分辨」的结论,然后接着扰动针尖。
    """
    got = classify_reasons(False, ["a_word_from_the_future"])
    assert got["verdict"] == VERDICT_UNDECIDABLE
    assert got["remedy"] == "ask_operator"
    assert got["unmapped_reasons"] == ["a_word_from_the_future"]


def test_a_self_contradictory_criterion_result_is_undecidable():
    """``passed=True`` 却带着出局词 —— 不替它圆场,判不了。"""
    got = classify_reasons(True, ["not_a_lattice"])
    assert got["verdict"] == VERDICT_UNDECIDABLE


# ══════════════════════════════════════════════════════════════════════
# 端到端:三态 + 依据
# ══════════════════════════════════════════════════════════════════════

def test_resolved_frame_reports_the_fast_axis_period():
    ctx, res = run_skill({"AnalyzeScanImage": analyze(),
                          "AssessAtomicPhase": assessed(passed=True)})
    assert res.success
    assert res.data["verdict"] == VERDICT_RESOLVED
    assert res.data["remedy"] is None
    assert res.data["period_fast_axis_nm"] == pytest.approx(0.249)
    assert "快扫方向周期" in res.summary


def test_absent_frame_counts_as_a_real_answer():
    ctx, res = run_skill({
        "AnalyzeScanImage": analyze(),
        "AssessAtomicPhase": assessed(passed=False, reasons=["not_a_lattice"])})
    assert res.data["verdict"] == VERDICT_ABSENT
    assert res.data["remedy"] is None


def test_coarse_frame_says_cannot_judge_not_no_atoms():
    """粗帧上判不了。summary 必须**说出来**这不等于「没有」。"""
    ctx, res = run_skill({
        "AnalyzeScanImage": analyze(),
        "AssessAtomicPhase": assessed(passed=False, reasons=["scale_gate"],
                                      nm_per_px=0.195, scale="off")})
    assert res.data["verdict"] == VERDICT_UNDECIDABLE
    assert res.data["remedy"] == "shrink_field"
    assert "不等于" in res.summary


def test_criterion_failure_is_undecidable_not_absent():
    """判据跑不起来 = 判不了。**「读不到」不是「没有」。**"""
    ctx, res = run_skill({"AnalyzeScanImage": analyze(),
                          "AssessAtomicPhase": FAIL})
    assert res.data["verdict"] == VERDICT_UNDECIDABLE
    assert res.data["remedy"] == "rescan_frame"


def test_verdict_always_carries_its_gated_and_ungated_criteria():
    """没有这两栏,verdict 就是一句没依据的话。

    读的人要能分清:这个 "absent" 是三条判据都看过了,还是尺度门根本没让判据开口。
    """
    ctx, res = run_skill({"AnalyzeScanImage": analyze(),
                          "AssessAtomicPhase": assessed(passed=True)})
    gated = {c["name"] for c in res.data["gated_criteria"]}
    ungated = {c["name"] for c in res.data["ungated_criteria"]}
    assert {"angular_concentration", "fft_sharpness", "band_peak_snr",
            "scale_gate"} <= gated
    # 诊断量必须在**不参与裁决**那一栏,而且要写明为什么
    assert "order_ratio" in ungated
    assert all(c.get("why_not_gated") for c in res.data["ungated_criteria"])
    # 同一个名字不许两边都在
    assert not (gated & ungated) - {"expected_a_nm_comparison"}


def test_profile_provenance_travels_with_the_verdict():
    """阈值出身必填 —— 一个没有 provenance 的阈值不该出现在结论里。"""
    ctx, res = run_skill({
        "AnalyzeScanImage": analyze(profile="p-x", provenance="2026-08-02 13 帧"),
        "AssessAtomicPhase": assessed(passed=True)})
    assert res.data["profile_name"] == "p-x"
    assert res.data["profile_provenance"] == "2026-08-02 13 帧"
    assert "2026-08-02 13 帧" in res.summary


# ══════════════════════════════════════════════════════════════════════
# 帧准入:闸在裁决之前
# ══════════════════════════════════════════════════════════════════════

def test_a_broken_frame_is_never_judged_for_atoms():
    """残帧 / 噪声帧上**不跑判据**。

    在自己刚制造的坑上做判定 —— 那个 "absent" 说的是这一帧坏了,不是这个偏压上
    没有原子对比度。
    """
    bad = dict(GOOD_METRICS, rowcorr_median=0.05)
    ctx, res = run_skill({"AnalyzeScanImage": analyze(bad),
                          "AssessAtomicPhase": assessed(passed=False,
                                                        reasons=["not_a_lattice"])})
    assert res.data["verdict"] == VERDICT_UNDECIDABLE
    assert res.data["remedy"] == "rescan_frame"
    assert ctx.count("AssessAtomicPhase") == 0, "残帧上判据被跑了"


def test_unreadable_admission_is_not_a_pass():
    """帧准入指标读不到 ⇒ 判不了。**读不到 ≠ 干净。**"""
    ctx, res = run_skill({"AnalyzeScanImage": FAIL,
                          "AssessAtomicPhase": assessed(passed=True)})
    assert res.data["verdict"] == VERDICT_UNDECIDABLE
    assert res.data["remedy"] == "rescan_frame"
    assert ctx.count("AssessAtomicPhase") == 0


def test_admission_can_be_turned_off_explicitly():
    bad = dict(GOOD_METRICS, rowcorr_median=0.05)
    ctx, res = run_skill({"AnalyzeScanImage": analyze(bad),
                          "AssessAtomicPhase": assessed(passed=True)},
                         require_frame_admission=False)
    assert res.data["verdict"] == VERDICT_RESOLVED
    assert res.data["frame_admission"]["passed"] is False, "关闸不等于把不合格改成合格"
    assert res.data["frame_admission"]["required"] is False


def test_bad_row_frac_is_reported_but_never_gates():
    """``bad_row_frac_annotate`` 出厂是 0.0 —— 那是**标注**线不是准入线。

    拿它当硬闸会让几乎每张真实帧都过不了准入(有一行坏就出局)。字段标签会说谎,
    这里按它的实际用途读它:报数,不设闸。
    """
    from mast.vision.scan_prep_thresholds import resolve

    th = resolve(None)
    assert th.bad_row_frac_annotate == 0.0, "前提变了,重看这条闸"
    metrics = dict(GOOD_METRICS, artifacts={"bad_row_frac": 0.30})
    adm = evaluate_frame_admission(metrics, th)
    assert adm["passed"] is True
    assert adm["bad_row_frac"] == pytest.approx(0.30)
    assert "bad_row_frac" not in {c["name"] for c in adm["gated_criteria"]}


def test_single_direction_frame_is_not_punished_for_having_no_backward():
    """单向帧没有反扫可比 —— 那是**不适用**,不是**读不到**。"""
    from mast.vision.scan_prep_thresholds import resolve

    metrics = dict(GOOD_METRICS, fb_instability=None)
    adm = evaluate_frame_admission(metrics, resolve(None))
    assert adm["passed"] is True
    fb = next(c for c in adm["gated_criteria"] if c["name"] == "fb_instability")
    assert fb["applicable"] is False and fb["passed"] is None


def test_missing_rowcorr_is_unreadable_not_clean():
    from mast.vision.scan_prep_thresholds import resolve

    metrics = dict(GOOD_METRICS, rowcorr_median=float("nan"))
    adm = evaluate_frame_admission(metrics, resolve(None))
    assert adm["passed"] is False
    assert "rowcorr_median" in adm["unreadable"]


# ══════════════════════════════════════════════════════════════════════
# 两条预处理的对照:说出来,但不抢方向盘
# ══════════════════════════════════════════════════════════════════════

def test_disagreement_is_recorded_and_said_but_does_not_change_the_verdict():
    """裁决永远以 ``AssessAtomicPhase`` 为准;对照路只是一条免费的自检信号。"""
    metrics = dict(GOOD_METRICS, atomic={"passed": False,
                                         "reasons": ["not_a_lattice"]})
    ctx, res = run_skill({"AnalyzeScanImage": analyze(metrics),
                          "AssessAtomicPhase": assessed(passed=True)})
    assert res.data["verdict"] == VERDICT_RESOLVED, "对照路抢了方向盘"
    assert res.data["cross_check"]["agrees"] is False
    assert res.success is True, "不一致不该阻断"
    assert "两条预处理给出不同结论" in res.summary


def test_no_cross_check_result_is_none_not_disagreement():
    """对照路没算出来 ⇒ ``agrees=None``。**「读不到」不是「不一致」。**"""
    metrics = dict(GOOD_METRICS)
    metrics.pop("atomic")
    ctx, res = run_skill({"AnalyzeScanImage": analyze(metrics),
                          "AssessAtomicPhase": assessed(passed=True)})
    assert res.data["cross_check"]["agrees"] is None
    assert res.data["cross_check"]["analyze_scan_image_passed"] is None


# ══════════════════════════════════════════════════════════════════════
# 被否掉的方案钉成测试
# ══════════════════════════════════════════════════════════════════════

def test_live_buffer_input_is_refused():
    """实时缓冲不能被无条件视为存盘帧；未确认可比性的输入必须拒绝。"""
    names = {p.name for p in VerifyAtomicResolution().metadata().parameters}
    assert not (names & {"buffer", "frame_data", "image", "array",
                         "use_buffer", "allow_buffer", "channel_index"}), \
        f"参数表里出现了缓冲类入口: {names}"
    scan_path = next(p for p in VerifyAtomicResolution().metadata().parameters
                     if p.name == "scan_path")
    assert scan_path.required is True

    for bad in ("", "/tmp/frame.dat", "buffer://0", "/tmp/frame.png"):
        ctx, res = run_skill({"AnalyzeScanImage": analyze(),
                              "AssessAtomicPhase": assessed(passed=True)},
                             scan_path=bad)
        assert res.data["verdict"] == VERDICT_UNDECIDABLE, bad
        assert ctx.count("AssessAtomicPhase") == 0, bad


def test_peak_strength_metrics_never_enter_the_decision_path():
    """AST 检查从当前源文件定位节点，避免依赖固定行号。"""
    banned = ("fine_periodic_snr", "fft_quality", "has_lattice")
    decision_path = (
        VAR.classify_reasons,
        VAR.VerifyAtomicResolution.plan_dynamic,
        VAR.VerifyAtomicResolution._finish_from_criterion,
        VAR.VerifyAtomicResolution._finish,
        VAR.VerifyAtomicResolution.aggregate,
        VAR.evaluate_frame_admission,
    )
    for fn in decision_path:
        src = source_of(fn)
        # 自检先行:证明取到的确实是**这个**函数的源码。取错/取空时下面那 3 条
        # ``not in`` 恒真,而恒真的断言与守得住的断言在报告里长得一模一样。
        assert f"def {fn.__name__}" in src, (
            f"取到的不是 {fn.__qualname__} 的源码(前 80 字:{src[:80]!r})—— "
            "下面那几条 not in 会因此恒真,先修取源")
        for name in banned:
            assert name not in src, f"{fn.__qualname__} 引用了 {name}"

    # 返回的数据里也不该出现 —— 转发出去,下一个人就会拿它当依据。
    ctx, res = run_skill({"AnalyzeScanImage": analyze(),
                          "AssessAtomicPhase": assessed(passed=True)})
    flat = repr(res.data)
    for name in banned:
        assert name not in flat, f"返回数据里泄漏了 {name}"


def test_scale_rejection_never_rewrites_the_frame():
    """**被否方案 3**:静默缩帧以过尺度门。

    帧宽是用户的意图,偷偷改掉等于换了被测对象。所以这是一条**拒绝**:
    ``size_m`` / ``pixels`` 原样回传,由人决定走哪条路;``alternatives`` 给两条
    **算得出来的具体**路,不是「建议调整参数」这种废话。
    """
    rej = atomic_scale_reject(50e-9, 256)
    assert rej is not None
    assert rej["code"] == "atomic_scale_unreachable"
    assert rej["size_m"] == 50e-9, "视野被改写了"
    assert rej["pixels"] == 256
    assert len(rej["alternatives"]) == 2
    assert any("2501 px" in a for a in rej["alternatives"])
    assert any("5.12 nm" in a for a in rej["alternatives"])
    # 满权重档不产生拒绝
    assert atomic_scale_reject(5e-9, 512) is None


def test_scale_rejection_also_fires_in_the_transition_band():
    """过渡带也是拒绝 —— 「有结论但证据不足」不该被当成可以下发。"""
    rej = atomic_scale_reject(10e-9, 256)
    assert rej is not None and rej["scale"] == "reduced"
    assert rej["size_m"] == 10e-9


def test_more_pixels_alternative_says_to_pay_in_line_time():
    """加像素那条路必须写明「线时同比加大」。

    提像素不提线时会让 ``nm/px`` 变好看而每像素驻留砍半 —— 尺度门是被骗过去的,
    不是真的过了。一条不带这句话的建议会**直接教人去骗闸**。
    """
    rej = atomic_scale_reject(50e-9, 256)
    more_px = next(a for a in rej["alternatives"] if "px 以上" in a)
    assert "线时" in more_px and "驻留" in more_px


# ══════════════════════════════════════════════════════════════════════
# expected_a_nm:默认不传 = 比对**真的**关闭
# ══════════════════════════════════════════════════════════════════════

def test_default_explicitly_disables_the_lattice_comparison():
    """「默认不传 = 比对关闭」只有**显式传 0** 才成立。

    ``AssessAtomicPhase`` 的参数缺省时会去从当前样品**推断**一个晶格常数 ——
    不传 ≠ 关闭,不传 = 让别处替你填。而那个值一进判据就是一道**下界严**的硬闸:
    填小了会把真原子分辨判成不合格,而上界是松的 —— 两个方向的代价不对称。

    (这是「静态默认遮蔽条件默认」的同一形状:兜底值合理得让人看不出兜底发生了。)
    """
    ctx, res = run_skill({"AnalyzeScanImage": analyze(),
                          "AssessAtomicPhase": assessed(passed=True)})
    sent = ctx.params_for("AssessAtomicPhase")[0]
    assert "expected_a_nm" in sent, "没传 ⇒ 判据会去推断一个晶格常数出来"
    assert sent["expected_a_nm"] == 0.0


def test_an_explicit_lattice_constant_is_forwarded():
    ctx, res = run_skill({"AnalyzeScanImage": analyze(),
                          "AssessAtomicPhase": assessed(passed=True)},
                         expected_a_nm=0.2494)
    assert ctx.params_for("AssessAtomicPhase")[0]["expected_a_nm"] == 0.2494


def test_thresholds_are_forwarded_not_left_to_defaults():
    """阈值由本壳显式传入 —— 验收闸与修针闸用同一次调用,阈值不会两边漂。"""
    ctx, res = run_skill({"AnalyzeScanImage": analyze(),
                          "AssessAtomicPhase": assessed(passed=True)},
                         snr_min=5.0, concentration_min=30.0, sharpness_min=9.0)
    sent = ctx.params_for("AssessAtomicPhase")[0]
    assert sent["snr_min"] == 5.0
    assert sent["concentration_min"] == 30.0
    assert sent["sharpness_min"] == 9.0


def test_allow_reduced_scale_defaults_off_and_is_forwarded():
    ctx, _ = run_skill({"AnalyzeScanImage": analyze(),
                        "AssessAtomicPhase": assessed(passed=True)})
    assert ctx.params_for("AssessAtomicPhase")[0]["allow_reduced_scale"] is False
    ctx2, _ = run_skill({"AnalyzeScanImage": analyze(),
                         "AssessAtomicPhase": assessed(passed=True)},
                        allow_reduced_scale=True)
    assert ctx2.params_for("AssessAtomicPhase")[0]["allow_reduced_scale"] is True


# ══════════════════════════════════════════════════════════════════════
# 只读 + 元数据
# ══════════════════════════════════════════════════════════════════════

def test_skill_is_read_only_analysis():
    md = VerifyAtomicResolution().metadata()
    assert md.safety_level.value == "auto"
    assert md.category.value == "analysis"


def test_skill_is_registered():
    from mast.skills import composite

    assert composite.VerifyAtomicResolution is VerifyAtomicResolution
    assert "VerifyAtomicResolution" in composite.__all__

    from mast.core.registry import registered_skill_names

    assert "VerifyAtomicResolution" in registered_skill_names(), \
        "导出了但派不到 —— 「已经写好了」是生产方的话"


def test_the_sub_skill_names_it_dispatches_to_actually_exist():
    """调度使用的技能名必须存在。"""
    from mast.core.registry import registered_skill_names

    names = registered_skill_names()
    for sub in ("AnalyzeScanImage", "AssessAtomicPhase"):
        assert sub in names, f"裁决壳要派给 {sub},但注册表里没有它"


def test_it_only_calls_read_only_sub_skills():
    """裁决壳不碰硬件 —— 它只读一张已经存在的文件。"""
    ctx, _ = run_skill({"AnalyzeScanImage": analyze(),
                        "AssessAtomicPhase": assessed(passed=True)})
    assert {name for name, _ in ctx.calls} <= {"AnalyzeScanImage",
                                               "AssessAtomicPhase"}


def test_sample_pointer_mismatch_is_a_hint_not_a_gate():
    """两个「现在是什么样品」的指针对不上时:说出来,**不阻断**。"""
    ctx, res = run_skill({"AnalyzeScanImage": analyze(),
                          "AssessAtomicPhase": assessed(passed=True)})
    assert res.success is True
    assert "sample_pointer_agrees" in res.data
    # 读不到样品记录时是 None(不知道),**不是** False(不一致)
    assert res.data["sample_pointer_agrees"] in (None, True, False)


# ── 补救方向不许写反 ────────────────────────────────────────────────────────
#
# ``too_few_periods``（视野太小，装不下足够多的周期）与 ``scale_gate``
# （像素太粗，看不清周期）都是「判不了」，但补救**方向相反**：前者要**扩**视野，
# 后者要**缩**。写反了不会报错，只会让流程一路缩到更判不了 —— 那正是这张表
# 自己在注释里警告过的失效方式（``scale_reduced`` 曾经差点被归成 absent）。

def test_too_few_periods_asks_to_widen_the_field_not_shrink_it():
    verdict, remedy = REASON_VERDICT["too_few_periods"]
    assert verdict == VERDICT_UNDECIDABLE
    assert remedy == VAR.REMEDY_WIDEN_FIELD
    assert remedy != VAR.REMEDY_SHRINK_FIELD, (
        "视野太小装不下周期，缩视野只会更判不了 —— 方向写反了")


def test_the_two_scale_remedies_point_opposite_ways():
    """反证：``scale_gate`` 仍然是「缩」，新词没有把老词一起改掉。"""
    assert REASON_VERDICT["scale_gate"][1] == VAR.REMEDY_SHRINK_FIELD
    assert REASON_VERDICT["scale_reduced"][1] == VAR.REMEDY_SHRINK_FIELD


def test_too_few_periods_does_not_count_against_the_failure_budget():
    """判不了不计入失败预算 —— 否则一串「装不下」会把偏压判成不可用。"""
    got = classify_reasons(False, ["too_few_periods"])
    assert got["verdict"] == VERDICT_UNDECIDABLE
    assert got["remedy"] == "widen_field"


# ── 出局词的中文措辞：也要穷举 ──────────────────────────────────────────────
#
# 2026-08-24。这一边此前**根本不存在** —— `_why()` 直接 join 英文 snake_case，
# 于是用户读到的是「这一帧上没有原子分辨: peaks_not_one_lattice、
# fast_axis_no_peak」。16 个词全是这样。
#
# 加了措辞表就必须同时加这道闸门，否则下一个加出局词的人只会补 REASON_VERDICT
# （那边有闸门会红），措辞这边静默漏掉，而漏掉的表现是「露出一个英文词」——
# 没有任何测试会为此变红。**加一个键永远是双边动作。**


def test_reason_zh_is_exhaustive():
    from mast.skills.composite.verify_atomic_resolution import REASON_ZH

    assert set(REASON_ZH) == set(ALL_REASONS), (
        f"判据有而措辞表没有: {set(ALL_REASONS) - set(REASON_ZH)};"
        f"措辞表有而判据没有: {set(REASON_ZH) - set(ALL_REASONS)}")


@pytest.mark.parametrize("reason", sorted(ALL_REASONS))
def test_every_reason_word_reads_as_chinese(reason):
    """每个词都要有中文，而且**不许把英文词原样当中文**。"""
    from mast.skills.composite.verify_atomic_resolution import REASON_ZH

    zh = REASON_ZH[reason]
    assert zh and zh != reason, "%s 的措辞还是它自己" % reason
    assert not zh.isascii(), "%s 的措辞里一个汉字都没有: %r" % (reason, zh)


def test_an_unknown_word_comes_through_verbatim_not_invented():
    """查不到就**原样返回**。

    编一句比露出英文词坏得多：用户会照着那句话去处置，而那句话与判据实际
    看到的东西无关。露出英文词至少是可查的。
    """
    from mast.skills.composite.verify_atomic_resolution import reason_zh

    assert reason_zh("a_word_nobody_has_seen") == "a_word_nobody_has_seen"


def test_the_absent_sentence_is_readable():
    """终点验收：拼出来的那句话里不许还留着 snake_case。"""
    from mast.skills.composite.verify_atomic_resolution import (
        VerifyAtomicResolution, classify_reasons)

    cls = classify_reasons(False, ["peaks_are_ridges", "fast_axis_no_peak"])
    why = VerifyAtomicResolution._why(cls, {"nm_per_px": 0.019})
    assert "peaks_are_ridges" not in why and "fast_axis_no_peak" not in why, (
        "英文词漏进了给人看的句子: %s" % why)
    assert "脊" in why and "逐行谱" in why, why
