"""整平流程传递声明参数，并正确解释两类返回状态。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.chat import narration_templates as T  # noqa: E402
from mast.skills.composite import _tip_phases as P  # noqa: E402
from mast.skills.composite.auto_tilt import AutoTilt  # noqa: E402


# ── ① 调用方发的参数，被调方必须收得下 ──────────────────────────────────


class _FakeProgress:
    def __init__(self):
        self.partial_data: dict = {}
        self.failed_reasons: dict = {}


class _FakeExecutor:
    """只提供 ``_tip_phases`` 的三个 helper 用到的东西。"""

    def __init__(self, sub_results: "dict | None" = None):
        self.sub_results = dict(sub_results or {})
        self.progress = _FakeProgress()

    def set_partial(self, *_a, **_kw):
        pass


class _Res:
    def __init__(self, data):
        self.data = data


def _drain_level_step(window_side_m=3.5e-8, rms_m=4e-11):
    """跑 ``_level_on_terrace``，把它 yield 出来的那个 CompositeStep 抓下来。"""
    ex = _FakeExecutor()
    gen = P._level_on_terrace(
        ex, MagicMock(), step_prefix="D",
        d={"window_side_m": window_side_m, "rms_m": rms_m})
    step = next(gen)
    return ex, gen, step


def test_the_params_d_phase_sends_are_params_autotilt_declares():
    """**这一条就是那个 bug。**

    ``AutoTilt`` 拒掉的不是一个值，是一个**键名** —— 而键名对不上是调用方
    和被调方之间唯一一种「两边分别都对」的错法。
    """
    _ex, _gen, step = _drain_level_step()
    assert step.skill_name == "AutoTilt"
    errors = AutoTilt().validate_params(dict(step.params))
    assert errors == [], (
        f"D 相发给 AutoTilt 的参数过不了它自己的校验：{errors}\n"
        f"发的是 {sorted(step.params)}；"
        f"AutoTilt 声明的是 "
        f"{sorted(p.name for p in AutoTilt().metadata().parameters)}")


def test_the_terrace_window_is_what_autotilt_judges_against():
    """传入 AutoTilt 的台阶宽度应来自当前测量。"""
    _ex, _gen, step = _drain_level_step(window_side_m=3.5e-8)
    assert step.params.get("next_frame_m") == pytest.approx(3.5e-8)


def test_no_key_is_sent_with_a_none_value():
    """未指定的参数不应以 None 键下发。"""
    _ex, _gen, step = _drain_level_step(window_side_m=None, rms_m=None)
    assert all(v is not None for v in step.params.values()), step.params
    assert AutoTilt().validate_params(dict(step.params)) == []


# ── ② 回包的键名：读的和写的必须是同一批 ────────────────────────────────


def _autotilt_data(outcome: str, **extra) -> dict:
    """照 ``AutoTilt.execute`` 里 ``report()`` 的形状造一份回包。"""
    data = {"outcome": outcome, "reason": extra.pop("reason", "")}
    data.update(extra)
    return data


@pytest.mark.parametrize("outcome,done,skipped,no_action", [
    ("applied", True, False, False),
    ("no_action_needed", False, False, True),
    ("skipped", False, True, False),
    ("failed", False, False, False),
])
def test_tilt_readout_tells_the_four_outcomes_apart(outcome, done, skipped,
                                                    no_action):
    """四个结局指向四个不同的下一步，两态装不下它们。

    尤其 ``skipped``（没标定：**没做**）与 ``failed``（做了没做成）：前者要去跑
    ``TiltCalibrate``，后者要去查标定为什么失效。
    """
    ex = _FakeExecutor({"D:tilt": _Res(_autotilt_data(outcome))})
    t = P._tilt_readout(ex, "D:tilt")
    assert (t["done"], t["skipped"], t.get("no_action")) == (done, skipped,
                                                             no_action)


def test_tilt_readout_finds_the_z_span_where_autotilt_actually_puts_it():
    """落差在 ``before["z_span_m"]`` / ``after["z_span_m"]``，**不是**两个平铺的键。

    读错名字不会报错 —— 它让每次调平都念成「(落差没记下来)」，
    看起来像仪器没给数，其实是我们没去取。
    """
    ex = _FakeExecutor({"D:tilt": _Res(_autotilt_data(
        "applied", before={"z_span_m": 8e-9}, after={"z_span_m": 2e-9}))})
    t = P._tilt_readout(ex, "D:tilt")
    assert t["before_m"] == pytest.approx(8e-9)
    assert t["after_m"] == pytest.approx(2e-9)


def test_a_step_that_never_ran_is_not_reported_as_skipped():
    """步骤失败（``sub_results`` 里根本没有它）时，**「读不到」不是一个结局**。

    退到失败原文，绝不编一个原因，更不许说成「跳过」。
    """
    ex = _FakeExecutor()
    ex.progress.failed_reasons["D:tilt"] = "Unknown parameter: 'scan_path'"
    t = P._tilt_readout(ex, "D:tilt")
    assert t["outcome"] == ""
    assert (t["done"], t["skipped"]) == (False, False)
    assert "scan_path" in t["detail"]


# ── ③ 旁白：四个结局四句话 ──────────────────────────────────────────────


def _level_sentence(**data) -> str:
    r = T.render("level_result", data)
    assert r is not None
    return r.text


def test_a_skip_is_never_read_out_as_a_failure():
    """「没做」与「没做成」是两句话。

    08-18 之前判「跳过」读的是一个**不存在的字段**，于是每一次「没做」都被念成
    「没做成」—— 一个前置条件缺失被报成了一次失败。
    """
    s = _level_sentence(done=False, skipped=True, no_action=False,
                        reason="这台仪器还没做过倾斜响应标定")
    assert "执行失败" not in s
    assert "未执行" in s and "标定" in s


def test_nothing_to_do_is_not_read_out_as_done():
    """``no_action_needed`` = 本来就够平，**一个字没改**。

    念成「调平做了」是在报告一件没发生的事。
    """
    s = _level_sentence(done=False, no_action=True, skipped=False)
    assert "无需执行" in s
    assert "调平**已执行**" not in s


def test_a_real_level_reads_out_both_numbers():
    """真调平了就要报出落差，前后两个数都要。"""
    s = _level_sentence(done=True, skipped=False, no_action=False,
                        before_m=8e-9, after_m=2e-9)
    assert "已执行" in s and "落差没记下来" not in s
    assert "8 nm" in s and "2 nm" in s


def test_a_failure_without_a_reason_says_so():
    """说不出原因就明说没记下来。

    原来那一支输出「调平没做成。」—— 一句读起来像结论、实际不含任何信息的话。
    """
    s = _level_sentence(done=False, skipped=False, no_action=False, reason="")
    assert "执行失败" in s and "原因没记下来" in s


def test_the_tone_follows_the_verdict():
    """配色不许和句子说两件事：没做 / 没做成都要显眼，不用做是好消息。"""
    def tone(**d):
        r = T.render("level_result", d)
        assert r is not None
        return r.tone

    assert tone(done=True) == "good"
    assert tone(done=False, no_action=True) == "good"
    assert tone(done=False, skipped=True) == "warn"
    assert tone(done=False, skipped=False, no_action=False) == "warn"


# ── ④ 正反扫描线重合度:B 相唯一的判决,此前零旁白 ────────────────────────


def _fwdbwd_sentence(**data):
    r = T.render("fwd_bwd_result", data)
    assert r is not None
    return r.text, r.tone


def test_a_missing_verdict_is_never_read_out_as_a_verdict():
    """**读不到 ≠ 不通过。**

    三态的最后一支是「不重合」,所以任何一次「两个键都没传到」都会念出一句
    「针尖还没到基本态,回去打脉冲」—— 一个**根本没做过的判决**,而且它指向的
    是打脉冲。本仓 `read_failure_folded_into_a_value` 那一型:
    「读不到」被折叠成一个具体的、合理得没人会去核的值。
    """
    text, _ = _fwdbwd_sentence()
    assert "判决没记下来" in text
    assert "不重合" not in text and "打脉冲" not in text

    # 只有读数、没有判决 —— 一样不许判。读数照报(它是真的量到的)。
    text, _ = _fwdbwd_sentence(similarity=0.6)
    assert "判决没记下来" in text and "0.600" in text


def test_a_real_failure_still_says_go_pulse():
    """判据真的说了「不通过」时,那句话必须照说 —— 上一条不能把它一起吞掉。

    (防「假值念错」的守卫把真值一起吞了,本仓 `gate_checks_form_not_output`。)
    """
    text, tone = _fwdbwd_sentence(passed=False, inconclusive=False,
                                  similarity=0.612, threshold=0.80)
    assert "不重合" in text and "打脉冲" in text
    assert "0.612" in text and "0.800" in text
    assert tone == "warn"


def test_inconclusive_is_not_a_tip_verdict():
    """「判不了」既不是通过也不是不通过 —— 而且它**不配色成警告**。"""
    text, tone = _fwdbwd_sentence(inconclusive=True, reason="这块地方起伏不足")
    assert "判不了" in text
    assert "这不等于针尖没问题" in text
    assert "打脉冲" not in text
    assert tone == "info"


def test_a_pass_reads_out_both_numbers_and_is_good_news():
    text, tone = _fwdbwd_sentence(passed=True, inconclusive=False,
                                  similarity=0.875, threshold=0.80)
    assert "重合" in text and "0.875" in text and "0.800" in text
    assert tone == "good"


# ── ⑤ 多针尖:三态全说,配色跟着判决走 ────────────────────────────────────


@pytest.mark.parametrize("verdict,tone,must_say", [
    # 三种判定状态分别使用固定的专业术语与提示颜色。
    ("split", "warn", "多针尖"),
    ("single", "good", "单针尖"),
    ("undecidable", "info", "判不了"),
])
def test_multi_tip_speaks_in_all_three_states(verdict, tone, must_say):
    """质量判据保留通过、失败与未知三种状态。"""
    r = T.render("step_split", {"result": {
        "verdict": verdict, "score": 0.125, "score_threshold": 0.2,
        "reason": "视野太小", "levels_pm": [-100, 100, 300]}})
    assert r is not None
    assert must_say in r.text
    assert r.tone == tone, f"{verdict} 被画成了 {r.tone}"


def test_the_single_tip_sentence_carries_the_margin():
    """判成「单尖」时也要报出那个数 —— 它离阈值有多近才是有用的信息。"""
    r = T.render("step_split", {"result": {
        "verdict": "single", "score": 0.125, "score_threshold": 0.2}})
    assert r is not None
    assert "0.125" in r.text and "0.200" in r.text


# ── ⑥ 多针尖一票否决 ──────────────────────────


def test_a_split_tip_is_not_ready_even_when_the_lines_overlap():
    """重合度过了 ≠ 针尖修好了。

    两条判据看的是**两件事**:一根劈成两个顶点的针尖照样可以来回重复 ——
    两趟画出的是同一对重影,所以重合度量不到它。
    """
    text, tone = _fwdbwd_sentence(passed=False, inconclusive=False,
                                  fwd_bwd_ok=True, split_tip=True,
                                  similarity=0.875, threshold=0.80)
    assert "一票否决" in text and "打脉冲" in text
    assert "0.875" in text, "重合度照报 —— 它确实过了,只是不够"
    assert tone == "warn"


def test_the_veto_sentence_is_not_shown_when_there_is_no_split():
    """没劈就不许说否决 —— 否则每一次通过都挂着一句吓人的话。"""
    text, _ = _fwdbwd_sentence(passed=True, inconclusive=False,
                               fwd_bwd_ok=True, split_tip=False,
                               similarity=0.912, threshold=0.80)
    assert "否决" not in text
    assert "过了这道基本判据" in text


def test_undecidable_multi_tip_does_not_veto():
    """**只有 `split` 否决,「判不了」不否决。**

    本仓 `uncalibrated_threshold_has_no_veto`:未标定的阈值没资格否决人。
    这条判据标定过(零重叠、6/6 零漏),所以它有资格 —— 但资格只覆盖它**说得出话**
    的那一态。一个「这张图看不出来」不该把一根可能好好的针推去挨脉冲。

    调用侧的 ``out["split_tip"] = (verdict == "split")`` 就是这条不变量,
    这里从句子侧再钉一次:`split_tip` 假 ⇒ 通过。
    """
    text, tone = _fwdbwd_sentence(passed=True, inconclusive=False,
                                  fwd_bwd_ok=True, split_tip=False,
                                  similarity=0.9, threshold=0.80)
    assert tone == "good" and "否决" not in text


def test_a_question_that_was_never_asked_says_so():
    """验证图存不下来时,多针尖那一问**根本没被问** —— 而否决因此不生效。

    沉默在这里最危险:`step_split` 那张卡片不会出现,用户看到的是「重合度过了」
    然后放行 —— 和「问了,是单尖」长得**一模一样**。
    """
    r = T.render("step_split", {"result": {
        "verdict": "undecidable",
        "reason": "验证图没存下来、拿不到文件,这一轮没能问"}})
    assert r is not None
    assert "判不了" in r.text and "没存下来" in r.text
    # 「判不了」不是坏消息也不是好消息 —— 别染成警告。
    assert r.tone == "info"


def test_verify_phase_sets_the_veto_flag_from_the_verdict_not_from_truthiness():
    """`split_tip` 必须由 **`verdict == "split"`** 派生,不是 `bool(verdict)`。

    `verdict` 三态全是**非空字符串**(`split` / `single` / `undecidable`),
    所以 `bool(verdict)` 恒真 —— 写成那样的话每一次验证都会被否决,而且
    「一票否决」那句话每次都出现,看起来完全正常。
    """
    import re

    src = Path(P.__file__).read_text(encoding="utf-8")
    m = re.search(r'out\["split_tip"\]\s*=\s*(.+)', src)
    assert m, "找不到 split_tip 的赋值 —— 这条闸门在空跑"
    expr = m.group(1)
    assert '== "split"' in expr, (
        f"split_tip 不是从 verdict 等值派生的:{expr!r}")


# 技能报告与旁白共用 AutoTilt 的实际返回键，避免消费者字段错位。


def _tilt_result_sentence(data: dict) -> str:
    """把一份 AutoTilt 回包按**旁白真正收到的形状**渲染出来。

    形状取自 ``graph_executor._narrate_step_result``:
    ``_narrate(kind, skill=…, params=…, result=data)`` —— 所以是 ``{"result": data}``。
    直接调 ``Template.render(...)`` 会绕开这一层包装,而绕开它正是
    ``poke_indent`` 30/30 走 fallback 却十条单测全绿的原因。
    """
    r = T.render("auto_tilt_result", {"result": data})
    assert r is not None
    return r.text


def test_the_levelling_result_speaks_on_a_real_autotilt_payload():
    """**这一条就是那个 bug。** 真形状的回包必须说出话,不许走 fallback。"""
    data = _autotilt_data("applied",
                          before={"z_span_m": 1.8e-9, "measured_slope_deg": 0.021},
                          after={"z_span_m": 3e-10}, iterations=2)
    r = T.render("auto_tilt_result", {"result": data})
    assert r is not None
    assert r.text != T.TEMPLATES["auto_tilt_result"].fallback, (
        f"真形状的回包仍然说的是 fallback 那句:{r.text}")
    assert "已执行" in r.text
    # 「效果」要有前后两个数 —— 只说「做了」等于没说。
    assert "1.8 nm" in r.text and "300 pm" in r.text, r.text


def test_the_narration_and_the_readout_agree_on_every_outcome():
    """旁白与 ``_tilt_readout`` 是同一批键的两个消费方,结论必须一致。

    钉这条是因为「只改一个消费方」在这个函数上**已经发生过两次**
    (08-18 改了 readout 没改旁白;更早两相各抄一份)。
    """
    for outcome in ("applied", "no_action_needed", "skipped", "failed",
                    "rolled_back"):
        data = _autotilt_data(outcome, before={"z_span_m": 5e-9})
        ex = _FakeExecutor({"D:tilt": _Res(data)})
        t = P._tilt_readout(ex, "D:tilt")
        said = _tilt_result_sentence(data)
        good = bool(t["done"] or t.get("no_action"))
        assert ("已执行" in said or "无需执行" in said) is good, (
            f"{outcome}: readout 说 leveled={good},旁白说的却是「{said}」")
        assert "没记下来" not in said, f"{outcome} 走了 fallback:{said}"


def test_a_rolled_back_run_is_not_read_as_a_plain_failure():
    """回滚与执行失败是不同状态。"""
    rolled = _tilt_result_sentence(_autotilt_data(
        "rolled_back", detail="第 1 轮后残余 Z 占用 8.0 nm,未降到上一轮的 70% 以下"))
    failed = _tilt_result_sentence(_autotilt_data(
        "failed", detail="3 轮后残余 Z 占用 1.2 nm 仍高于验收阈"))
    assert rolled != failed
    assert "写回" in rolled
    assert "写回" not in failed


def test_the_failure_sentence_still_carries_a_number():
    """没做成也要报读数 —— 「做了没做成」和「本来就多斜」靠的就是这个数分开。

    要求:调平时的操作和效果宜记入旁白。
    """
    said = _tilt_result_sentence(_autotilt_data(
        "rolled_back", before={"z_span_m": 10.0e-9}, trigger_z_span_m=1e-9,
        detail="第 1 轮后残余 Z 占用 8.0 nm"))
    assert "10 nm" in said, said
    assert "触发阈" in said and "1 nm" in said, said


def test_the_unknown_outcome_is_never_read_out_as_a_verdict():
    """读不到 outcome ⇒ 明说没记下来,**绝不猜**,更不许说成「没做成」。"""
    said = _tilt_result_sentence({})
    assert "没记下来" in said
    assert "已执行" not in said and "未执行" not in said and "失败" not in said


def test_the_recorded_facts_are_paths_that_actually_resolve():
    """报告字段必须来自实际返回值的对应路径。"""
    data = _autotilt_data("applied",
                          before={"z_span_m": 1.8e-9, "measured_slope_deg": 0.021},
                          after={"z_span_m": 3e-10}, iterations=2,
                          trigger_z_span_m=1e-9, accept_z_span_m=5e-10)
    r = T.render("auto_tilt_result", {"result": data})
    assert r is not None
    for path in ("result.outcome", "result.before.z_span_m",
                 "result.after.z_span_m", "result.iterations"):
        assert path in r.facts, f"facts 里没有 {path} —— 这句话的数对不回去"


def test_the_ghost_keys_are_really_absent_from_autotilt():
    """闸门自检:那四个凭记忆写出来的键,``auto_tilt.py`` 里一个都没有。

    没有这一条,上面几条只是「换了一批键名」;有了它,才说明**换对了方向**。
    (这就是那份夹具当年该做而没做的事 —— 它是照设计文档写的,不是照源码。)
    """
    src = (Path(_MASTV2_ROOT) / "mast" / "skills" / "composite"
           / "auto_tilt.py").read_text(encoding="utf-8")
    for ghost in ('"z_span_before_m"', '"z_span_after_m"', '"action"'):
        assert ghost not in src, (
            f"{ghost} 现在真的在 AutoTilt 的回包里了 —— 那这几条测试的前提变了")
    # ``after`` 是 ``report(..., after={...})`` 的关键字实参,不是字典字面量的
    # 键 —— 找 ``"after"`` 会找不到。**这一条自己就踩过一次**:一个按错形状
    # 去找的检查,报的是「这个键没了」,而真相是「它从来不长那个样子」。
    for real in ('"outcome"', '"z_span_m"', '"measured_slope_deg"',
                 '"before": before', 'after={', '"trigger_z_span_m"'):
        assert real in src, f"{real} 不在 auto_tilt.py 里了 —— 旁白该跟着改"


# ── ⑤ C 相的调平也要说话 ────────────────────────────────────
#
# 「旁白说『调平是否执行』没记下来 —— 调平做没做、效果如何,
# 旁白里看不到。」
#
# 除了模板读错键(上面那一组),还有一处**结构性缺口**:D 相的
# ``_level_on_terrace`` 08-17 就发 level_begin / level_result 了,而 C 相 ——
# 这一相的名字就叫 ``level_phase`` —— 从头到尾**一条都没发**,结论只进
# ``out["tilt"]`` 那份报文。两相各改一半,又是「一侧改了,另一侧没跟上」。


class _WF:
    """``NobleTipWorkflow`` 里 ``level_phase`` 真正读到的那几个字段。

    不用 ``MagicMock``:属性访问会返回 Mock,而 ``float(mock) > float(mock)``
    这一类比较会静默走到另一条分支上去 —— 本仓「替身让断言变空」的同形。
    """
    step_scan_nm = 100.0
    step_fallback_nm = 0.0          # 0 ⇒ 不走回退分支
    step_pixels = 128
    step_line_time_s = 0.15
    flat_region_nm = 35.0
    forge_scan_nm = 200.0


def _drain_level_phase(monkeypatch, tilt_data, *, flat_found=True):
    """跑一遍 ``level_phase``,把它发出的旁白收下来。

    ``reuse=`` 给一张现成的图 ⇒ 不走「自己扫一张」那条分支(那条要 _relocate,
    与本条测试要证的事无关)。
    """
    said: list = []
    monkeypatch.setattr(P, "_say", lambda kind, /, **d: said.append((kind, d)))
    monkeypatch.setattr(P, "_step_split_look",
                        lambda *_a, **_kw: {"verdict": "single"})

    ex = _FakeExecutor()
    gen = P.level_phase(ex, _WF(), prefix="C",
                        reuse=("D:/fake/scan_0001.sxm", 1e-7, -1e-7))
    # 一步步喂:每 yield 一个 CompositeStep,就把那一步的"结果"放进 sub_results。
    replies = {
        "C:steps": {"step_dominated": True, "surface_rms_m": 9e-12,
                    "tilt_valid": True, "tilt_deg": 0.02},
        "C:flat": ({"center_x_m": 1.1e-7, "center_y_m": -1.1e-7,
                    "window_side_m": 3.5e-8, "rms_m": 8e-12,
                    "windows_checked": 12} if flat_found else {}),
        "C:tilt": tilt_data,
    }
    steps = []
    try:
        step = next(gen)
        while True:
            steps.append(step)
            data = replies.get(step.step_id)
            if data is not None:
                ex.sub_results[step.step_id] = _Res(data)
            step = gen.send(None)
    except StopIteration as stop:
        out = stop.value
    return said, steps, out


def test_the_c_phase_levelling_now_speaks(monkeypatch):
    """**这一条就是那个缺口。** C 相调平前后各要说一句。"""
    said, steps, out = _drain_level_phase(
        monkeypatch,
        _autotilt_data("applied", before={"z_span_m": 1.8e-9},
                       after={"z_span_m": 3e-10}))
    kinds = [k for k, _ in said]
    assert "level_begin" in kinds, f"C 相调平前一句话都没说:{kinds}"
    assert "level_result" in kinds, f"C 相调平后一句话都没说:{kinds}"
    # 顺序:先说要做,再说结果 —— 反过来就成了「事后通报」。
    assert kinds.index("level_begin") < kinds.index("level_result")
    assert out["leveled"] is True


def test_the_c_phase_begin_carries_the_window_it_really_sends(monkeypatch):
    """开工那句里的窗,必须是**真正下发给 AutoTilt 的那个** ``next_frame_m``。

    这两个数一旦各取各的,句子就会在「我们用 35 nm 判」和实际用 39 nm 之间
    悄悄分家 —— 而 AutoTilt 的触发判据正是拿这个尺寸算的。
    """
    said, steps, _out = _drain_level_phase(
        monkeypatch, _autotilt_data("no_action_needed"))
    tilt_step = [s for s in steps if s.step_id == "C:tilt"][0]
    begin = dict([d for k, d in said if k == "level_begin"][0])
    assert begin["window_nm"] == pytest.approx(
        float(tilt_step.params["next_frame_m"]) * 1e9)


def test_the_c_phase_result_reports_the_same_verdict_as_the_report(monkeypatch):
    """旁白说的和报文写的必须是同一个结论 —— 两份读法就是两个真源。"""
    for outcome in ("applied", "no_action_needed", "skipped", "failed"):
        said, _steps, out = _drain_level_phase(
            monkeypatch, _autotilt_data(outcome, before={"z_span_m": 5e-9}))
        res = dict([d for k, d in said if k == "level_result"][0])
        leveled = bool(res["done"] or res.get("no_action"))
        assert leveled is bool(out["leveled"]), (
            f"{outcome}: 报文说 leveled={out['leveled']},旁白说 {leveled}")


def test_nothing_is_said_when_the_phase_never_reaches_levelling(monkeypatch):
    """找不到平区就退出的那条路上**不许**说「调平已执行」。

    一句凭空出现的「调平做完了」比不说更坏 —— 它是一件没发生的事。
    """
    said, _steps, out = _drain_level_phase(
        monkeypatch, _autotilt_data("applied"), flat_found=False)
    assert "level_begin" not in [k for k, _ in said]
    assert "level_result" not in [k for k, _ in said]
    assert out.get("leveled") is False


# 步骤失败时保留可读 detail，与短错误码同时返回。


def _step_failed_sentence(**data) -> str:
    r = T.render("step_failed", data)
    assert r is not None
    return r.text


def test_a_failed_step_carries_the_skills_own_sentence():
    """**这一条就是那个缺口。** detail 里的数字必须出现在旁白里。"""
    said = _step_failed_sentence(
        skill="AutoTilt", step_id="C:tilt", continued=True,
        reason="AutoTilt failed: rolled_back: diverged",
        detail="第 1 轮后残余 Z 占用 8.0 nm,未降到上一轮 10.0 nm 的 70% 以下")
    assert "8.0 nm" in said, f"技能说的那句被丢了:{said}"
    assert "diverged" in said, "短码也要留着 —— 它是检索用的"
    assert "继续往下走" in said


def test_no_detail_reads_exactly_like_before():
    """没有 detail 时逐字保持原样 —— 绝大多数技能不给这个字段。"""
    without = _step_failed_sentence(skill="ScanAt", reason="超时", continued=False)
    empty = _step_failed_sentence(skill="ScanAt", reason="超时", continued=False,
                                  detail="")
    assert without == empty
    assert "——" not in without


def test_a_detail_already_inside_the_reason_is_not_said_twice():
    """有些技能把同一句同时放进 error 和 detail —— 不许念两遍。"""
    said = _step_failed_sentence(
        skill="AutoTilt", continued=True,
        reason="AutoTilt failed: 残余 Z 占用 8.0 nm", detail="残余 Z 占用 8.0 nm")
    assert said.count("8.0 nm") == 1, said


def test_the_executor_actually_hands_the_detail_over():
    """接线:``graph_executor`` 失败那一支必须把 ``result.data['detail']`` 传下去。

    模板自测证明不了发出方发了 —— 这正是 ``poke_indent`` 30/30 走 fallback 却
    十条单测全绿的那个形状。
    """
    import ast
    import inspect

    from mast.skills.composite import graph_executor as GE

    sig = inspect.signature(GE.GraphExecutor._narrate_failure)
    assert "detail" in sig.parameters, "_narrate_failure 收不下 detail"
    sig2 = inspect.signature(GE.GraphExecutor._handle_failure)
    assert "detail" in sig2.parameters, "_handle_failure 收不下 detail"

    tree = ast.parse(Path(GE.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "_narrate_failure"]
    assert calls, "找不到 _narrate_failure 的调用点"
    assert all(any(kw.arg == "detail" for kw in c.keywords) for c in calls), (
        "有一处 _narrate_failure 没传 detail —— 那条路径上技能的原话仍会被丢掉")
    # 而 detail 必须真的来自失败结果的 data,不是凭空造的常量。
    src = Path(GE.__file__).read_text(encoding="utf-8")
    assert 'get("detail")' in src, "detail 不是从 result.data 里取的"


def test_a_small_tilt_is_never_printed_as_zero():
    """0.0004° 不许印成 "0.000°" —— 那读起来像「正好是零」。

    与「把 0.3 nm 报成 0」同一条:一个被抹成零的读数,比不报这个读数更坏,
    因为它看起来是个结论。
    """
    said = _tilt_result_sentence(_autotilt_data(
        "no_action_needed", before={"z_span_m": 7.6e-12,
                                    "measured_slope_deg": 0.0004}))
    # ⚠️ 判据是「印出来的那个数不是零」,**不是**「句子里没有子串 0.000」——
    # 后者会被 "0.0004" 自己命中(第一版就是这么写的,当场红了)。
    assert "0.0004" in said, said
    assert "0.000°" not in said, f"倾斜被抹成了零:{said}"
