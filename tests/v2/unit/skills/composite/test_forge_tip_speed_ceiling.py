"""验证每一帧的线时由几何与速度上限派生；软件上限不等于仪器标定。"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.core import scan_policy  # noqa: E402
from mast.core.noble_tip_workflow import (  # noqa: E402
    _BOUNDS,
    NOBLE_METAL_BASELINE,
    forge_line_time_s,
    forge_speed_notes,
    resolve,
)
from mast.skills.composite.forge_au_tip import _forge_wf  # noqa: E402
from mast.skills.composite.scan_at import ScanAt  # noqa: E402

# 复用端到端上下文与 autouse fixture，避免绕过生产入口或继承其他测试的针尖状态。
from tests.v2.unit.skills.composite.test_forge_au_tip import (  # noqa: E402
    _base_script,
    _fast_and_offline,  # noqa: F401 —— autouse fixture,导入即生效,别删
    _run,
)

# 速度界限取自当前配置表。
_CONFIGURED_SPEED_UPPER = _BOUNDS["forge_v_tip_nm_s"][1]


# ══════════════════════════════════════════════════════════════════════
# 主验收:端到端,每一帧
# ══════════════════════════════════════════════════════════════════════

def test_forge_line_time_never_exceeds_tip_speed_ceiling():
    """跑完一遍外环,**每一次** ScanAt 解析出来的针尖速度都不超过上限。

    走的是 ``ScanAt._resolve``(→ ``scan_resolver`` 的 explicit 优先级链),
    不是外环递出来的那个 dict —— 「参数被传出去了」证明不了下游不会丢掉它。
    """
    _res, ctx = _run(_base_script(), max_sites=1)
    wf = _forge_wf({})
    ceiling = wf.forge_v_tip_nm_s
    seen = 0
    for p in ctx.params_for("ScanAt"):
        resolved = ScanAt()._resolve(p)
        nm = float(p["size_m"]) * 1e9
        lt = float(resolved.configure_scan["line_time_s"])
        speed = nm / lt
        assert speed <= ceiling * 1.001, (
            f"{nm:.0f} nm 的图针尖 {speed:.0f} nm/s,超过 forge 速度上限 "
            f"{ceiling:g} nm/s —— 派生没生效,或者有一张图绕过了 scan_at_params")
        seen += 1
    assert seen >= 3, f"只检了 {seen} 次 ScanAt,自检失败(探针没盖到)"


def test_no_forge_frame_reaches_the_configured_exclusive_upper_bound():
    """任意尺寸的扫描帧均须遵守配置速度界限。"""
    _res, ctx = _run(_base_script(), max_sites=1)
    for p in ctx.params_for("ScanAt"):
        resolved = ScanAt()._resolve(p)
        nm = float(p["size_m"]) * 1e9
        speed = nm / float(resolved.configure_scan["line_time_s"])
        assert speed < _CONFIGURED_SPEED_UPPER, (
            f"{nm:.0f} nm 的图针尖 {speed:.0f} nm/s 达到或超过配置上界 "
            f"{_CONFIGURED_SPEED_UPPER:g} nm/s")


def test_the_acceptance_frame_is_the_one_this_fix_was_about():
    """确认端到端探针实际覆盖了配置中的验收图，而不是只遍历空调用列表。"""
    _res, ctx = _run(_base_script(), max_sites=1)
    wf = _forge_wf({})
    acceptance = [p for p in ctx.params_for("ScanAt")
                  if round(float(p["size_m"]) * 1e9) == round(wf.step_scan_nm)
                  and int(p.get("pixels") or 0) == int(wf.step_pixels)]
    assert acceptance, (
        f"一次 {wf.step_scan_nm:.0f} nm / {wf.step_pixels} px 的图都没扫到 —— "
        "这组测试没有盖到验收图,先修探针再看结论")
    for p in acceptance:
        lt = float(p["line_time_s"])
        # 改前这里是 0.15 s ⇒ 667 nm/s。
        assert wf.step_scan_nm / lt <= wf.forge_v_tip_nm_s * 1.001
        assert lt > NOBLE_METAL_BASELINE.forge_step_line_time_s, (
            "验收图仍在用流程表里那个 0.15 s —— 派生没接上")


# ══════════════════════════════════════════════════════════════════════
# 上限的来源:不许与它的出处漂移
# ══════════════════════════════════════════════════════════════════════

def test_the_default_ceiling_agrees_with_the_configured_reference_frame():
    """默认流程的参考帧几何、线时和速度上限应保持一致；这些是软件配置，不是仪器标定。"""
    wf = NOBLE_METAL_BASELINE
    pinned = wf.forge_scan_nm / wf.forge_verify_line_time_s
    assert pinned == pytest.approx(wf.forge_v_tip_nm_s, rel=1e-3), (
        f"上限 {wf.forge_v_tip_nm_s} nm/s 与它的出处("
        f"{wf.forge_scan_nm:g} nm / {wf.forge_verify_line_time_s:g} s = "
        f"{pinned:.2f} nm/s)对不上 —— 两个数在表达同一件事,不能各走各的")
    # 而那一帧是 5 分钟(判据),这是上一环的出处。
    assert scan_policy.estimate_scan_seconds(
        wf.forge_verify_pixels, wf.forge_verify_line_time_s) == pytest.approx(
        300.0, rel=1e-3), "那一帧不再是 5 分钟 —— 上限的整条出处链断了"
    # 默认速度必须严格低于可配置上界。
    assert wf.forge_v_tip_nm_s < _CONFIGURED_SPEED_UPPER


def test_configured_speed_bounds_are_applied_without_clamping(monkeypatch):
    """独立合成配置验证开上界、闭下界、合法值通过和越界回退；不代表仪器标定。"""
    monkeypatch.setitem(_BOUNDS, "forge_v_tip_nm_s", (20.0, 250.0))
    baseline = NOBLE_METAL_BASELINE.forge_v_tip_nm_s
    for accepted in (20.0, 90.0, 249.0):
        assert accepted != baseline
        assert resolve({"forge_v_tip_nm_s": accepted}).forge_v_tip_nm_s == pytest.approx(accepted)
    for rejected in (19.0, 250.0, 375.0):
        assert rejected != baseline
        assert resolve({"forge_v_tip_nm_s": rejected}).forge_v_tip_nm_s == pytest.approx(baseline)


def test_the_configured_upper_bound_is_exclusive():
    """当前配置上界及其以上均须拒绝；内部合法值和闭下界仍须接受。"""
    base = NOBLE_METAL_BASELINE.forge_v_tip_nm_s
    for rejected in (_CONFIGURED_SPEED_UPPER, _CONFIGURED_SPEED_UPPER + 1e-9,
                     _CONFIGURED_SPEED_UPPER * 1.1):
        assert resolve({"forge_v_tip_nm_s": rejected}).forge_v_tip_nm_s == (
            pytest.approx(base)), (
            f"超出配置上限的 {rejected} nm/s 被接受了")
    # 上界拒绝而紧邻上界的合法值应接受，防止守卫恒拒。
    just_under = _CONFIGURED_SPEED_UPPER - 1.0
    assert resolve({"forge_v_tip_nm_s": just_under}).forge_v_tip_nm_s == (
        pytest.approx(just_under)), "合法值也被拒了 —— 闸门恒拒,上一条测了个寂寞"
    # 而下界仍是闭的 —— 只有这一个字段的**上界**是开区间,别把它当成全表新规矩。
    lo, _hi = _BOUNDS["forge_v_tip_nm_s"]
    assert resolve({"forge_v_tip_nm_s": lo}).forge_v_tip_nm_s == pytest.approx(lo)


# ══════════════════════════════════════════════════════════════════════
# 派生函数本身的三条规矩
# ══════════════════════════════════════════════════════════════════════

def test_the_ceiling_only_slows_down_never_speeds_up():
    """速度上限不能把已经更慢的工作点主动提速。"""
    wf = _forge_wf({})
    lt, why = forge_line_time_s(wf, wf.cluster_scan_nm, wf.cluster_line_time_s)
    assert lt == pytest.approx(wf.cluster_line_time_s), "簇图被上限改动了"
    assert why is None, f"没超速却出了说明:{why}"
    # 自检:这条测试不是恒真的 —— 同一个函数在超速的那张图上确实会动手。
    fast, why_fast = forge_line_time_s(wf, wf.step_scan_nm,
                                       NOBLE_METAL_BASELINE.forge_step_line_time_s)
    assert fast > NOBLE_METAL_BASELINE.forge_step_line_time_s and why_fast


def test_none_line_time_stays_none():
    """None 保持未指定语义，让下游档位解析仍可生效。"""
    assert NOBLE_METAL_BASELINE.step_line_time_s is None
    lt, why = forge_line_time_s(NOBLE_METAL_BASELINE,
                                NOBLE_METAL_BASELINE.step_scan_nm, None)
    assert lt is None and why is None, (
        f"「不下发」被派生成了一个具体的值 {lt} —— 档位表那条路从此到不了")


def test_an_operator_pinned_line_time_is_not_clamped_but_does_say_so():
    """显式线时保持原值，超出软件上限时必须报告；未显式指定时仍应派生限速。"""
    wf = _forge_wf({"forge_line_time_s": 0.05})     # 100 nm ⇒ 2000 nm/s
    lt, why = forge_line_time_s(wf, wf.step_scan_nm, wf.step_line_time_s)
    assert lt == pytest.approx(0.05), "用户钉的数被夹紧了"
    assert why and "未夹紧" in why and "2000" in why, (
        f"超速却没有出声,或者没把算出来的 nm/s 说出来:{why!r}")
    # 而**没钉**的时候必须夹 —— 否则上一行就成了「谁都不夹」。
    lt2, why2 = forge_line_time_s(_forge_wf({}), 100.0, 0.05)
    assert lt2 > 0.05 and why2


def test_the_verify_frame_is_capped_too_even_though_it_skips_scan_at():
    """验证帧经过 PreScanCheck 时仍应遵守上限。"""
    wf = _forge_wf({"forge_scan_nm": 200.0})
    naive = wf.verify_scan_nm / wf.verify_line_time_s
    assert naive > wf.forge_v_tip_nm_s, (
        f"探针失效:200 nm 上验证帧只有 {naive:.0f} nm/s,没超上限,测不到限速")

    _res, ctx = _run(_base_script(), max_sites=1, forge_scan_nm=200.0)
    calls = ctx.params_for("PreScanCheck")
    assert calls, "一次 PreScanCheck 都没跑到,探针没盖到验证帧"
    for p in calls:
        nm = float(p["width_m"]) * 1e9
        lt = p.get("line_time_s")
        assert lt is not None, (
            "验证帧没带每线时间 —— 会落回档位表(而档位表是他的**成像**档)")
        speed = nm / float(lt)
        assert speed <= wf.forge_v_tip_nm_s * 1.001, (
            f"验证帧 {nm:.0f} nm 针尖 {speed:.0f} nm/s,超过上限 "
            f"{wf.forge_v_tip_nm_s:g} nm/s —— PreScanCheck 那条路没接限速")


# ══════════════════════════════════════════════════════════════════════
# 让这个数出声
# ══════════════════════════════════════════════════════════════════════

def test_every_frame_speed_reaches_the_report():
    """报告中的速度应与实际下发的每一帧一致。"""
    res, _ctx = _run(_base_script(), max_sites=1)
    speed_notes = res.data.get("scan_speed_notes") or []
    assert len(speed_notes) >= 5, (
        f"只有 {len(speed_notes)} 条速度说明,五张评估图没说全:{speed_notes}")
    for label in ("验证帧", "验收图", "回退图", "找台面图", "簇图"):
        assert any(label in n for n in speed_notes), (
            f"报告里没有「{label}」的速度:{speed_notes}")
    for n in speed_notes:
        assert "⇒ 针尖" in n, f"这条没把速度算出来:{n!r}"
    # 而且要真的印进那句给人读的话里 —— 存进 data 但没人念,等于没说。
    summary = str(res.data.get("summary_cn") or "")
    assert "针尖横向速度" in summary and "171 nm/s" in summary, (
        f"速度没进结案报文:{summary[:400]}")


def test_the_speed_notes_are_not_labelled_as_envelope_adjustments():
    """**钉住一个差点被造出来的标签谎言。**

    第一版把速度说明拼进 ``envelope_notes``,而 ``_summary`` 给那个列表里**每一条**
    都加前缀「参数按当前针尖的安全包络调整过」—— 于是五条速度里那四条**根本没被
    调整过**的帧会被印成「调整过」。用户会照着那句话去调一个没在生效的数。
    本仓管这个叫「字段标签会说谎」。

    两个键从此分开:``envelope_notes`` 只装真的被包络改过的,``scan_speed_notes``
    是陈述。
    """
    res, _ctx = _run(_base_script(), max_sites=1)
    env = res.data.get("envelope_notes") or []
    for n in env:
        assert "⇒ 针尖" not in str(n), (
            f"速度说明混进了包络说明,会被印成「按安全包络调整过」:{n!r}")
    summary = str(res.data.get("summary_cn") or "")
    for line in summary.splitlines():
        if "⇒ 针尖" in line:
            assert "安全包络调整过" not in line, (
                f"这一行把速度陈述印成了包络调整:{line!r}")


def test_the_reported_fallback_speed_is_the_one_actually_sent():
    """显式覆盖的行时间应同时体现在下发参数和报告中。"""
    pinned_lt = 2.0                                   # 测试显式参数覆盖。
    _res, ctx = _run(_base_script(), max_sites=1, forge_line_time_s=pinned_lt)
    wf = _forge_wf({"forge_line_time_s": pinned_lt})
    fb_calls = [p for p in ctx.params_for("ScanAt")
                if round(float(p["size_m"]) * 1e9) == round(wf.step_fallback_nm)]
    assert fb_calls, "回退图没被扫到,探针没盖到这条路"
    sent = float(fb_calls[0]["line_time_s"])
    # 他钉的数照跑,不缩放也不夹紧。
    assert sent == pytest.approx(pinned_lt), (
        f"用户钉的 {pinned_lt} s/线被改成了 {sent} —— 「一律」成了假话")
    note = next(n for n in forge_speed_notes(wf) if "回退图" in n)
    said = wf.step_fallback_nm / sent
    assert f"{said:.0f} nm/s" in note, (
        f"报告说的和真正下发的对不上:下发 {sent} s/线 ⇒ {said:.0f} nm/s,"
        f"而报告写的是 {note!r}")


def test_a_frame_whose_speed_cannot_be_computed_says_so_instead_of_guessing():
    """线时走档位表时,**如实说算不出来**,不折叠成一个具体的 nm/s。

    「读不到」被折叠成一个具体的值,本仓一天记过四次。这一层看不见档位表的解析
    结果,所以它对这类帧的正确答案是「不知道」,不是任何一个数字。
    """
    notes = forge_speed_notes(NOBLE_METAL_BASELINE)   # 出厂表:step/cluster 都是 None
    unknown = [n for n in notes if "算不出来" in n]
    assert unknown, f"走档位表的帧被报成了一个具体速度:{notes}"
    for n in unknown:
        assert "⇒ 针尖" not in n, f"说了算不出来又给了个数:{n!r}"


# ══════════════════════════════════════════════════════════════════════
# 派生改了几何 ⇒ 等待预算必须跟着重算
# ══════════════════════════════════════════════════════════════════════

def test_the_wait_budget_still_holds_the_biggest_derived_frame():
    """线时一变长,最大的那一帧就变了 —— ``forge_scan_timeout_s`` 必须跟上。

    「两个各自有理由的数字凑起来就是一次必然失败的扫描」,这条流程栽过一次:
    「断言为真、被测系统是坏的:真正会跑的帧是 512-614 s,预算 180 s 必然判死。」

    派生把回退图从 154 s 拉到 600 s,而 ``scan_at_params`` 的派生预算
    600×1.3+30 = 810 s **只有 1.35 倍**,兜不住 —— 所以表里那个数必须自己抬上去。
    """
    _res, ctx = _run(_base_script(), max_sites=1)
    biggest = 0.0
    for p in ctx.params_for("ScanAt"):
        est = ScanAt()._resolve(p).estimated_scan_s
        budget = ScanAt._wait_timeout_s(p, est)
        assert budget >= 2.0 * est, (
            f"{round(float(p['size_m']) * 1e9)} nm:帧要 {est:.0f} s,"
            f"预算只有 {budget:.0f} s")
        biggest = max(biggest, est)
    # 自检:探针真的盖到了那张变大的帧(否则这条在「回退图没被扫」时恒绿)。
    assert biggest > 200.0, (
        f"最大的一帧只有 {biggest:.0f} s —— 派生后的大帧没被扫到,先修探针")
    assert NOBLE_METAL_BASELINE.forge_scan_timeout_s >= 2.0 * biggest, (
        f"最大的一帧 {biggest:.0f} s,而 forge_scan_timeout_s 只有 "
        f"{NOBLE_METAL_BASELINE.forge_scan_timeout_s:.0f} s")
