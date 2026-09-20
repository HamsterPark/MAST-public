"""省略线时时查同一档位表，显式输入优先；宽度缺失走明确回退，等待预算随几何派生。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core.scan_policy import get_tier_for_size  # noqa: E402
from mast.skills.composite.prescan_check import (  # noqa: E402
    DEFAULT_LINE_TIME_S,
    DEFAULT_WAIT_TIMEOUT_S,
    PreScanCheck,
    resolve_line_time_s,
)


# ── 1. 默认来自档位表 ──────────────────────────────────────────────────────

@pytest.mark.parametrize("width_nm", [1.0, 5.0, 50.0, 100.0, 300.0, 2000.0])
def test_default_line_time_is_the_tier_line_time(width_nm):
    """各个尺度都要跟表一致 —— 只测 50 nm 会让「表只有一档接对了」也通过。"""
    w = width_nm * 1e-9
    want = float(get_tier_for_size(w)["line_time_s"])
    got = resolve_line_time_s(w)
    assert got == want, (
        f"{width_nm} nm 的每线时间是 {got} s,而档位表说 {want} s —— "
        "两个数一旦能不一致,预扫描就不再是在预测同一次扫描了")


def test_the_real_machine_case_is_ten_times_slower_than_before():
    """预扫描行时间跟随当前尺度档位。"""
    lt = resolve_line_time_s(50e-9)
    assert lt == 1.0, f"50 nm 应落 highres 档(1.0 s/线),实际 {lt}"
    tip_speed_nm_s = 50.0 / lt
    assert tip_speed_nm_s == 50.0
    assert 50.0 / DEFAULT_LINE_TIME_S > 400.0, (
        "旧常数下的针尖速度应该仍然是那个刮针的量级 —— "
        "这一条在解释上一条为什么重要;它变了说明常数被人动过")


# ── 2. 显式优先 ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("explicit", [0.02, 0.5, 2.0, 30.0])
def test_explicit_line_time_always_wins(explicit):
    """用户逐字说过的值压过任何表 —— 与 scan_at_params / scan_resolver 同惯例。

    **包括比档位表更快的值**:这条流程有它自己的判断,系统不替他否决,
    只保证他没说话时默认落在安全的那一侧。
    """
    assert resolve_line_time_s(50e-9, explicit) == explicit


# ── 3. 退化输入 ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_width", [None, 0.0, -1.0, "junk", float("nan"),
                                       float("inf")])
def test_unknown_width_falls_back_and_never_raises(bad_width):
    """宽度读不出来 ⇒ 没有尺寸也就没有档,只能退回常数。**但不许崩。**"""
    assert resolve_line_time_s(bad_width) == DEFAULT_LINE_TIME_S


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, "junk", float("nan")])
def test_junk_explicit_falls_through_to_the_tier(bad):
    """显式值是垃圾 ⇒ 当作没给,查表。**不要拿它去除宽度**(0 会除零)。"""
    assert resolve_line_time_s(50e-9, bad) == float(get_tier_for_size(50e-9)["line_time_s"])


# ── 4. 预算必须跟着走 ──────────────────────────────────────────────────────

class _Ctx:
    """模拟含行数的完整 ScanBuffer 协议信封。"""

    def __init__(self, lines: int = 256):
        self._lines = lines

    def buffer_reply(self):
        return ("", b"", (2, (0, 1), self._lines, self._lines))

    def safe_call(self, method, *args, **kw):
        from mast.core.types import NanonisCallRecord
        if method == "Scan_BufferGet":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=self.buffer_reply())
        return NanonisCallRecord(method=method, args=args)


def _assert_stub_parses(ctx: _Ctx) -> None:
    """替身必须真的能被产品代码解析出行数。

    没有这一条,任何一次「桩形状写错」都会退化成对出厂常数的测试 ——
    绿的,而且测的不是被测的那件事。
    """
    from mast.io.nanonis_files import parse_buffer_get
    parsed = parse_buffer_get(ctx.buffer_reply())
    assert parsed is not None, "替身的 Scan_BufferGet 回包解析不出来 —— 桩错了,不是代码错了"
    assert parsed.get("lines") == ctx._lines


def _wait_s_of(plan):
    """``WaitScanComplete`` 收的是 **``timeout_ms``**,不是秒。

    第一版把它写成 ``timeout_s`` —— 取不到就 ``or 0``,于是断言拿 0 去比,
    测试红了而**代码是对的**。`or` 兜底把「字段名写错」伪装成「值不对」;
    所以这里改成:找不到就明确报错,不许悄悄退成 0。
    """
    for st in plan:
        if st.skill_name == "WaitScanComplete":
            ms = st.params.get("timeout_ms")
            assert ms is not None, f"WaitScanComplete 的参数变了: {st.params}"
            return float(ms) / 1000.0
    raise AssertionError("计划里没有 WaitScanComplete —— 这条测试测错了对象")


def test_an_explicit_budget_is_a_floor_not_a_ceiling():
    """forge 传的 300 s **不能**把预算钉死在扫描时长以下。

    这是本次改动最容易漏的那半:旧实现是 ``if wait_timeout_s is None`` ——
    调用方一给值,派生就整个关掉。而 forge 正好给了 300 s(理由是「PreScanCheck
    的出厂预算是 15 s」,那句话 2026-08-10 起就不成立了)。线时从 0.1 改到 1.0
    之后帧时 52 s → 512 s,**每一次 verify 都会超时**。
    """
    ctx = _Ctx(256)
    _assert_stub_parses(ctx)
    sk = PreScanCheck()
    params = {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9,
              "wait_timeout_s": 300.0}
    sk._derived_wait_s = sk._derived_timeout_s(
        ctx, resolve_line_time_s(params["width_m"], params.get("line_time_s")))
    assert sk._derived_wait_s != DEFAULT_WAIT_TIMEOUT_S, (
        "派生值恰好等于出厂盲估常数 —— 多半是解析失败被吞了,"
        "这条测试会变成在测兜底路径")
    wait = _wait_s_of(sk.plan(params))
    assert wait > 512.0, (
        f"预算 {wait} s 短于 256×1.0×2 = 512 s 的帧时 —— "
        "一个短于扫描本身的预算不表达「我只愿意等这么久」,它表达「保证失败」")
    assert wait >= 300.0, "显式值仍然必须是下限(用户想等更久时照办)"


def test_a_larger_explicit_budget_still_wins():
    """反过来也要成立,否则上一条会被一个「永远用派生值」的实现骗过。"""
    ctx = _Ctx(256)
    _assert_stub_parses(ctx)
    sk = PreScanCheck()
    params = {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9,
              "wait_timeout_s": 9000.0}
    sk._derived_wait_s = sk._derived_timeout_s(ctx, 1.0)
    assert _wait_s_of(sk.plan(params)) == 9000.0


def test_the_tip_speed_actually_sent_to_hardware_is_the_tier_speed():
    """**下发给硬件的那个数**必须是档位表的速度 —— 不是计划里某个中间量。

    问题不是「参数不对」,是「针尖被刮坏了」,而刮针的直接原因是
    ``SetScanSpeed`` 里那个 **m/s**。所以这条断言钉在 ``fwd_speed`` 上:
    50 nm / 1.0 s = **5e-8 m/s = 50 nm/s**(旧值 4.88e-7 = 488 nm/s)。

    线速度以前是字面量 ``width * 10``(= width / 0.1),每线时间在两处各写了
    一遍。两处一旦能不一致,下发给硬件的速度就和预算算的不是同一次扫描 ——
    所以 ``fwd_line_time`` 与 ``fwd_speed`` **一起**钉。
    """
    plan = PreScanCheck().plan({"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9})
    sp = next(s.params for s in plan if s.skill_name == "SetScanSpeed")
    assert sp["fwd_line_time"] == 1.0 and sp["bwd_line_time"] == 1.0
    for k in ("fwd_speed", "bwd_speed"):
        assert abs(float(sp[k]) - 5e-8) < 1e-18, (
            f"{k} = {sp[k]} m/s,应为 5e-8(50 nm/s)。"
            f"旧值是 4.88e-7(488 nm/s),那正是把针刮坏的那个速度。")


def test_the_frame_is_square():
    """顺带守住 6.2.12 那条:高 = 宽,不是 width × 0.05 的细条。

    钉在这里是因为它和线时是**同一段计划**里的两个数,而两次问题反馈
    (细条看不出信息、太快了刮针)指的是同一次扫描。
    """
    plan = PreScanCheck().plan({"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9})
    cfg = next(s for s in plan if s.skill_name == "ConfigureScan")
    assert cfg.params["height_m"] == cfg.params["width_m"] == 50e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

# 从 schema 开始验证省略参数的路径，确保默认值注入不会绕过按宽度选择线时间。
def _through_the_tool_schema(**model_args):
    """模型只填这几个参数 ⇒ 技能最终看到的 params。**生产路径。**"""
    from mast.agents._shared.skill_adapter import (
        _coerce_si_params,
        _schema_from_metadata,
    )
    meta = PreScanCheck().metadata()
    kw = _schema_from_metadata(meta)(**model_args).model_dump()
    kw.pop("tool_call_id", None)
    out, errs = _coerce_si_params(meta, kw)
    assert not errs, f"SI 转换报错: {errs}"
    return out


def test_omitting_line_time_really_reaches_the_tier_table():
    """未指定行时间时，工具适配入口必须保留 None。"""
    p = _through_the_tool_schema(center_x_m="0n", center_y_m="0n", width_m="50n")
    assert p.get("line_time_s") is None, (
        f"没传 line_time_s,却拿到 {p.get('line_time_s')!r} —— "
        "ParameterSpec 又有 default 了。它会让「显式优先」分支恒命中,"
        "档位表永远查不到,针尖以 488 nm/s 刮过表面。")
    assert resolve_line_time_s(p["width_m"], p.get("line_time_s")) == 1.0


def test_omitting_the_budget_really_reaches_the_derivation():
    """未指定预算时由扫描几何推导。"""
    p = _through_the_tool_schema(center_x_m="0n", center_y_m="0n", width_m="50n")
    assert p.get("wait_timeout_s") is None, (
        f"没传 wait_timeout_s,却拿到 {p.get('wait_timeout_s')!r} —— "
        "派生预算那条路会再一次一步都走不到")


def test_the_speed_sent_to_hardware_through_the_real_path():
    """端到端:模型的三个参数 → 下发给硬件的 m/s。

    钉在 `fwd_speed` 上,因为那才是**刮不刮针**的那个数。
    """
    p = _through_the_tool_schema(center_x_m="0n", center_y_m="0n", width_m="50n")
    plan = PreScanCheck().plan(p)
    sp = next(s.params for s in plan if s.skill_name == "SetScanSpeed")
    assert sp["fwd_line_time"] == 1.0
    assert abs(float(sp["fwd_speed"]) - 5e-8) < 1e-18, (
        f"下发 {sp['fwd_speed']} m/s;出厂档位表下应为 5e-8(50 nm/s)。"
        f"4.88e-7(488 nm/s)是真机上把针刮坏的那个值。")


def test_an_explicit_value_still_wins_through_the_real_path():
    """用户逐字说过的值仍然压过档位表 —— 修 default 不能顺手关掉这条。"""
    p = _through_the_tool_schema(center_x_m="0n", center_y_m="0n",
                                 width_m="50n", line_time_s="0.25")
    assert p.get("line_time_s") == pytest.approx(0.25)
    plan = PreScanCheck().plan(p)
    sp = next(s.params for s in plan if s.skill_name == "SetScanSpeed")
    assert sp["fwd_line_time"] == pytest.approx(0.25)


def test_no_default_shadows_a_code_path_that_tests_for_omission():
    """结构闸门:**代码里判过 `is None` 的参数,不许有 default。**

    单独钉住 line_time_s / wait_timeout_s 只防得住这两个;下一个加参数的人不会
    想到这一层。但判据也不能写成「任何可选参数都不许有 default」——
    ``quality_threshold=0.8`` 就是一个**正当**的默认值:0.8 是它的语义,代码里
    没有任何地方问过「调用方到底传没传」。

    ⚠️ 第一版正是写成了那个宽判据,当场误报了 quality_threshold。
    **一条会误报的闸门迟早被人加豁免、再被删掉**,那时它防的东西就没人守了
    (今天早些时候粗动地图那条闸门也栽过同一次)。

    所以判据是**这两件事同时成立**:
      ① 这个可选参数带着一个非 None 的 default;
      ② 模块源码里存在对它的「没传吗」判断 —— 即 ``params.get("<名>")`` 出现在
         一个 ``is None`` 比较里,或被交给 ``resolve_line_time_s`` 这类
         「None = 去派生」的解析器。

    两条同时成立 ⇒ 那个 default 让 ② 那段代码**永远走不到**,而它看起来还在。
    """
    import inspect
    import re

    from mast.skills.composite import prescan_check as mod

    src = inspect.getsource(mod)
    offenders = []
    for spec in PreScanCheck().metadata().parameters:
        if getattr(spec, "required", True):
            continue
        if getattr(spec, "default", None) is None:
            continue
        n = re.escape(spec.name)
        asks_if_omitted = (
            re.search(rf'params\.get\(\s*["\']{n}["\']\s*\)\s*is None', src)
            or re.search(rf'resolve_\w+\([^)]*["\']{n}["\']', src)
            or re.search(rf'["\']{n}["\']\s*\)\s*\)\s*is None', src)
        )
        if asks_if_omitted:
            offenders.append(spec.name)

    assert not offenders, (
        f"这些可选参数带着 default,而代码里又在问「调用方传了吗」: {offenders}。"
        "pydantic 会替调用方填上那个 default,于是那个判断**永远为假** —— "
        "PreScanCheck 因为这一点在 6.2.13 上把针尖以 488 nm/s 刮了一遍,"
        "而 2026-08-10 的「预算从几何派生」也是这样被架空的。")


def test_that_gate_has_discriminating_power():
    """上一条闸门不能是恒真的 —— 造一个反例必须被它抓到。

    没有这一条,``asks_if_omitted`` 的正则哪怕永远匹配不上,闸门也会一直绿。
    """
    import re
    fake_src = 'if params.get("line_time_s") is None:\n    pass\n'
    assert re.search(r'params\.get\(\s*["\']line_time_s["\']\s*\)\s*is None',
                     fake_src), "正则连这个明显的反例都抓不到"
    # 而一个只被读、从不被问「传没传」的参数不该命中
    benign = 'thr = float(params.get("quality_threshold") or 0.8)\n'
    assert not re.search(r'params\.get\(\s*["\']quality_threshold["\']\s*\)\s*is None',
                         benign)


# ── 6. verify 自己的分辨率(2026-08-13,用户选的方案)─────────────────────
#
# 每线时间修好之后,一帧 verify 从 52 秒变成 34 分钟(512 行 × 2.0 s × 2),
# 一次 forge 光 verify 就 5.1 小时。用户选择「给 verify 单独一个更小的帧」。
#
# ⚠️ 最初给出的数字是**错的**:曾以为「改小到 20 nm,一帧能降到 ~5 分钟」。
# **帧时 = 行数 × 每线时间 × 2,尺寸不在式子里。** 改小视野一秒都不省 ——
# 那正是本仓那条已经骗过一次的假注释(「窄条能更快走完」)的同一个错误,
# 同一天又重复了一次。
#
# 两个旋钮是独立的:
#   针尖安全 = 视野 / 每线时间   (20 nm / 2.0 s = 10 nm/s,比 50 nm 时的 25 更稳)
#   耗时     = 行数 × 每线时间   (128 行 ⇒ 8.5 分钟,而不是 34 分钟)
#
# 所以这一组钉的是**那个区分**,不只是两个新数值。

def test_a_smaller_frame_alone_does_not_make_the_scan_shorter():
    """反例钉子:只改视野、不改行数 ⇒ **一秒都不省**。

    没有这一条,下一个人(或我)会再一次把「改小」当成「变快」。
    """
    wide = _through_the_tool_schema(center_x_m="0n", center_y_m="0n", width_m="50n")
    narrow = _through_the_tool_schema(center_x_m="0n", center_y_m="0n", width_m="20n")
    wide_plan, narrow_plan = PreScanCheck().plan(wide), PreScanCheck().plan(narrow)
    assert _wait_s_of(wide_plan) == _wait_s_of(narrow_plan), (
        "改小视野让预算变了 —— 那说明有人以为尺寸进了帧时公式。"
        "帧时 = 行数 × 每线时间 × 2。")
    # 变的是**针尖速度**,那才是改小视野买到的东西。
    sw = next(s.params for s in wide_plan if s.skill_name == "SetScanSpeed")
    sn = next(s.params for s in narrow_plan if s.skill_name == "SetScanSpeed")
    assert sn["fwd_speed"] < sw["fwd_speed"], "改小视野没有让针尖更慢 —— 那它就白改了"


def test_pixels_is_the_only_knob_that_shortens_the_frame():
    """给了 `pixels` ⇒ 预算按它算,而且**真的**下发一条 SetScanBuffer。"""
    p128 = _through_the_tool_schema(center_x_m="0n", center_y_m="0n",
                                    width_m="20n", pixels=128)
    plan = PreScanCheck().plan(p128)
    buf = next((s.params for s in plan if s.skill_name == "SetScanBuffer"), None)
    assert buf == {"pixels": 128, "lines": 128}, (
        f"没有下发分辨率({buf}) —— 那 verify 还是会继承上一次的 512 行")
    # 顺序:必须排在 ConfigureScan 之后(它内部走 Scan_BufferSet(ch,0,0))
    ids = [s.step_id for s in plan]
    assert ids.index("set_buffer") > ids.index("configure")
    # 预算按 128 行算,不是按盲估的 512
    inherit = PreScanCheck().plan(
        _through_the_tool_schema(center_x_m="0n", center_y_m="0n", width_m="20n"))
    assert _wait_s_of(plan) < _wait_s_of(inherit), (
        "设了 128 行,预算却还是按盲估的最坏行数算 —— "
        "两条路给出两个数,迟早有人拿错的那个去解释一次超时")


def test_omitting_pixels_keeps_the_historical_inherit_behaviour():
    """未指定像素数时跳过写入，不采用含义不明的零值。"""
    plan = PreScanCheck().plan(
        _through_the_tool_schema(center_x_m="0n", center_y_m="0n", width_m="20n"))
    assert not any(s.skill_name == "SetScanBuffer" for s in plan)


def test_forge_verify_uses_the_cheap_frame():
    """forge 的 verify 必须真的用上这两个数 —— 否则改了流程表等于没改。

    「生产方接上了、消费方不存在」是本仓反复栽的形状;这里两侧一起钉。

    ⚠️ **视野那个数要看 `forge_scan_nm`,不是 `verify_scan_nm`。**
    `_forge_wf` 用前者覆写后者,所以 forge 根本不读出厂表里那一个 ——
    第一版把 20 nm 改在了出厂表上,forge 那侧一点没变,而
    `test_the_forge_scans_are_small_and_do_not_touch_the_operator_table`
    (它钉的是「快档不许改出厂表」)当场红了。**改对了地方,才是改对了。**
    """
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE as base
    from mast.skills.composite.forge_au_tip import _forge_wf

    wf = _forge_wf({})          # forge 真正会用的那一份
    # 验证图还承担寻找台阶的用途，因此视野取流程表的配置。
    assert wf.verify_scan_nm == 100.0, (
        f"forge 的验证视野是 {wf.verify_scan_nm} nm —— 它来自 forge_scan_nm,"
        "改出厂表的 verify_scan_nm 到不了这里")
    # 256 而不是 128:这张图要找台阶,0.781 nm/px 太粗;256 px ⇒ 0.391 nm/px,
    # 正是 2026-08-14 对照实验用的分辨率。
    assert wf.verify_pixels == 256
    # 线时也必须到位 —— 它是 2026-08-14 补上的最后一个缺口(此前 verify 靠
    # 用户的成像档位表,他一改帧时就跟着变而无人管)。
    assert wf.verify_line_time_s == pytest.approx(0.586)
    frame_s = wf.verify_pixels * wf.verify_line_time_s * 2
    assert 290.0 <= frame_s <= 310.0, f"verify 帧时 {frame_s:.0f} s,不是 5 分钟"
    # 出厂表本身不许被快档动过(别的流程还在读它)
    assert base.verify_scan_nm == 50.0
    src = (Path(_MASTV2_ROOT) / "mast" / "skills" / "composite"
           / "_tip_phases.py").read_text(encoding="utf-8")
    head, _, tail = src.partition('skill_name="PreScanCheck"')
    assert tail, "verify 不再走 PreScanCheck 了 —— 这条测试的前提没了"
    assert "verify_pixels" in tail[:1400], (
        "verify 没有把 wf.verify_pixels 传下去 —— 那它还是会继承 512 行,"
        "一帧 34 分钟,一次 forge 光 verify 就 5 小时")
