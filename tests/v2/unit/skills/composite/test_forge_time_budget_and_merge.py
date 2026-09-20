"""验证时间预算、验证帧与调平帧复用、脉冲结果四态及未读到测量值时的 None 语义。

软件默认值作为配置契约核对，不构成仪器标定或实验结果。"""
from __future__ import annotations

import inspect

import pytest

# 源码断言通过 source_of 重读并按 AST 定位，避免文件行号变化造成错位切片。
from tests.v2.srcref import source_of


@pytest.fixture(autouse=True)
def _dont_touch_the_tip_registry(monkeypatch):
    """隔离针尖存储，仅验证参数接线，防止模块状态污染其他测试。"""
    import mast.core.noble_tip_workflow as m

    monkeypatch.setattr(m, "_tip_policy_now", lambda: None, raising=True)


# ── 1. 时间预算 ──────────────────────────────────────────────────────────

def test_counting_budgets_have_no_default():
    """``max_sites`` / ``max_rounds_per_site`` **不许有 default**。

    有了 default,「没传」就变成「传了 3」,而这一层区分的正是
    「由时间预算管」和「用户明确要限制站点数」。同一个坑的原形见
    ``prescan_check.py:96-102``(一个 ``default=0.1`` 让查档位表那行永远到不了)。
    """
    from mast.skills.composite.forge_au_tip import ForgeAuTip

    specs = {p.name: p for p in ForgeAuTip().metadata().parameters}
    for name in ("max_sites", "max_rounds_per_site", "time_budget_h"):
        assert name in specs, f"{name} 没有声明 —— 传了会被静默丢弃(死读)"
        assert getattr(specs[name], "default", None) is None, (
            f"{name} 有了 default,「没传」就不再是「没传」")


def test_default_budget_is_twelve_hours():
    from mast.skills.composite.forge_au_tip import _DEFAULT_BUDGET_H

    assert _DEFAULT_BUDGET_H == 12.0


def test_hard_caps_are_failsafes_not_budgets():
    """失控保险要**远高于**任何真实需求 —— 否则它就成了一个偷偷生效的预算。"""
    from mast.skills.composite.forge_au_tip import (
        _HARD_ROUND_CAP, _HARD_SITE_CAP, _IDLE_SITE_LIMIT)

    assert _HARD_SITE_CAP >= 100
    assert _HARD_ROUND_CAP >= 50
    assert 1 <= _IDLE_SITE_LIMIT <= 10


@pytest.mark.parametrize("phases,expected", [
    ([], False),
    ([{"phase": "pulse", "fired": 0}], False),
    ([{"phase": "pulse", "fired": 3}], True),
    ([{"phase": "verify", "similarity": None}], False),
    ([{"phase": "verify", "similarity": 0.43}], True),
    ([{"phase": "poke", "pokes": 2}], True),
    ([{"phase": "pulse", "fired": 0}, {"phase": "verify", "similarity": 0.9}], True),
])
def test_idle_detection_counts_measurements_not_steps(phases, expected):
    """空转判据看的是「测到了东西」,不是「跑完了步骤」。

    一串立刻失败的步骤不算进展 —— 那正是空转的样子。
    """
    from mast.skills.composite.forge_au_tip import _site_measured_something

    assert _site_measured_something({"phases": phases}) is expected


@pytest.mark.parametrize("outcome", ["time_budget_exhausted", "spinning", "hard_cap"])
def test_new_outcomes_never_suggest_raising_the_budget(outcome):
    """这三种停手都**不该**建议加大预算 —— 那是听起来很自然的错建议。

    时间到 = 请人来看;空转 = 没在修不是没修好;硬顶 = bug 信号。
    """
    from mast.skills.composite.forge_au_tip import _OUTCOME_CN, _summary

    assert outcome in _OUTCOME_CN

    class _P:
        aborted = False
        aborted_reason = ""
        partial_data: dict = {}

    text = _summary(outcome, [{"site": 1, "outcome": outcome,
                               "phases": [{"fired": 5}], "rounds": 1}], _P())
    assert "提高时间预算" not in text, f"{outcome} 不该建议加大预算:{text[:120]}"
    assert "提高预算" not in text, f"{outcome} 不该建议加大预算:{text[:120]}"


def test_budget_start_is_stashed_once_so_resume_accumulates():
    """起算时刻用 ``set_partial_default`` —— 续跑累加,不重新起算。

    每次 resume 都从零起算的话,反复被打断的跑可以无限进行,预算等于没有。
    """
    src = source_of(
        __import__("mast.skills.composite.forge_au_tip", fromlist=["x"])
        .ForgeAuTip.plan_dynamic)
    assert 'set_partial_default("started_at"' in src, (
        "起算时刻不再用 set_partial_default —— 确认续跑时它不会被重置")


# ── 2. verify + level 合并 ───────────────────────────────────────────────

def test_verify_frame_is_100nm_and_five_minutes():
    """100 nm / 256 px / 0.586 s ⇒ 5 min/帧、0.391 nm/px、针尖 171 nm/s。"""
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE as W, _BOUNDS

    assert W.forge_scan_nm == 100.0
    assert W.forge_verify_pixels == 256
    frame_s = W.forge_verify_pixels * W.forge_verify_line_time_s * 2
    assert 290.0 <= frame_s <= 310.0, f"帧时 {frame_s:.0f} s,不是 5 分钟"
    speed = W.forge_scan_nm / W.forge_verify_line_time_s
    assert speed < _BOUNDS["forge_v_tip_nm_s"][1], (
        f"针尖 {speed:.0f} nm/s 超过当前声明的速度界限")
    assert W.forge_scan_nm / W.forge_verify_pixels == pytest.approx(0.391, abs=0.01)


def test_forge_wires_verify_line_time_through():
    """新字段真的到得了 ``verify_line_time_s`` —— 死读比不读更坏。"""
    from mast.skills.composite.forge_au_tip import _forge_wf

    wf = _forge_wf({})
    assert wf.verify_scan_nm == 100.0
    assert wf.verify_pixels == 256
    assert wf.verify_line_time_s == pytest.approx(0.586)
    assert wf.step_fallback_nm == 200.0


def test_operator_override_still_wins_over_the_new_defaults():
    """``forge_pixels`` / ``forge_line_time_s`` 的「一律」不能有例外。

    曾经栽过一次:约定是「扫图一律 N px」,而 verify 收不到 ——
    「一律」这个词自己成了假话。新加的 verify 线时不能把这个缺口带回来。
    """
    from mast.skills.composite.forge_au_tip import _forge_wf

    wf = _forge_wf({"forge_pixels": 512, "forge_line_time_s": 2.0})
    assert wf.verify_pixels == 512
    assert wf.verify_line_time_s == 2.0
    assert wf.step_pixels == 512
    assert wf.cluster_line_time_s == 2.0


def test_fallback_frame_keeps_the_tip_speed():
    """回退行时间随尺寸变化，保持配置的速度限制。"""
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE as W, _BOUNDS
    from mast.skills.composite._tip_phases import level_phase

    naive = W.forge_step_fallback_nm / W.forge_step_line_time_s
    assert naive > W.forge_v_tip_nm_s, (
        "前提变了:回退图沿用 step 线时已经不超标 —— "
        "这条测试的理由消失了,去确认 forge_step_line_time_s 是不是被改小了")

    scaled = (W.forge_step_line_time_s
              * W.forge_step_fallback_nm / W.forge_step_scan_nm)
    assert (W.forge_step_fallback_nm / scaled
            == pytest.approx(W.forge_step_scan_nm / W.forge_step_line_time_s)), \
        "缩放之后针尖速度应当与 100 nm 那张完全相同"

    src = source_of(level_phase)
    assert "float(wf.step_scan_nm)" in src and "fb_line_time" in src, (
        "回退图不再按视野缩放线时 —— 针尖速度会跟着视野翻倍")


def test_level_phase_accepts_a_reused_frame():
    from mast.skills.composite._tip_phases import level_phase

    assert "reuse" in inspect.signature(level_phase).parameters, (
        "level_phase 不再收 reuse —— 合并没了,verify 那张 100 nm 图会被白扫")


def test_verify_phase_hands_out_its_scan_path():
    src = source_of(
        __import__("mast.skills.composite._tip_phases", fromlist=["x"]).verify_phase)
    assert '"scan_path"' in src, "verify 不再交出图的路径 ⇒ level 无从复用"


def test_acceptance_frame_size_does_not_follow_the_fallback():
    """验收图使用固定流程视野，不随回退视野变化；源码定位使用 source_of。"""
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE as W, _BOUNDS

    assert W.forge_step_scan_nm == 100.0
    assert W.forge_step_fallback_nm > W.forge_step_scan_nm

    src = source_of(
        __import__("mast.skills.composite.forge_au_tip", fromlist=["x"])
        .ForgeAuTip._accept)
    # 自检:先证明我们真的拿到了 ``_accept`` 的源码,再让 ``not in`` 说话。
    # 没有这一行,一个取错/取空的 src 会让下面那半**恒真** —— 而恒真的断言
    # 与守得住的断言在测试报告里长得一模一样。
    assert "def _accept" in src, (
        f"取到的不是 _accept 的源码(前 80 字:{src[:80]!r})—— "
        "下面那条 not in 会因此恒真,先修取源")
    assert "wf.step_scan_nm" in src and "fallback" not in src, (
        "验收图开始跟着回退视野走了 —— 阈值会失去意义")


# ── 3. 脉冲四态 ──────────────────────────────────────────────────────────

def test_pulse_treats_down_as_success_and_unreadable_separately():
    """四态各归各位:up/down → 满意;none → 没效果;读不到 → **单独记账**。

    「20 发全是 no_effect」说的是「脉冲打不动这根针」,
    「20 发全是 unreadable」说的是「Z 读回来有问题,去查信号链」——
    两句话指向完全不同的下一步,混在一起就都问不出来了。

    ── 为什么这里**必须**用 ``source_of`` 而不是 ``inspect.getsource``──

    本函数最后那条断言(``'... or 0.0' not in src``)守的是**本仓最反复出事的那一条
    规矩**:「读不到」不许被折叠成一个具体的值 —— 记账上一天之内出现过四次。

    而它自己恰好是**最容易悄悄失效**的形状。``inspect.getsource`` 按 import 那一刻
    记下的行号去切**当前磁盘上**的文件;共用工作树上别人往文件上方插几行,它返回的
    就是一段错位切片 —— **错位切片里当然不会有那个字符串,于是断言必然通过**。

    ⚠️ **守得越贵的规矩,它的绊线失效时越没人会发现** —— 因为没有人会去查一条绿着
    的测试。假红会被当成回归追一轮(吵,但有人查);假绿一声不响,而它守的恰恰是
    这一条。

    ``source_of`` 重读文件并用 ``ast`` 按名定位,源码与节点来自同一次读取,错位在
    结构上不可能;文件被写到一半会抛 ``SyntaxError``,那是**看得见**的失败。
    """
    src = source_of(
        __import__("mast.skills.composite._tip_phases", fromlist=["x"]).pulse_phase)

    # 自检必须在 ``not in`` **之前**:先证明拿到的确实是 pulse_phase 的源码。
    # 取错或取空时,下面那条 ``not in`` 恒真 —— 而那正是这次要根除的形状。
    assert "def pulse_phase" in src, (
        f"取到的不是 pulse_phase 的源码(前 80 字:{src[:80]!r})—— "
        "最后那条 not in 会因此恒真,先修取源")
    assert 'direction in ("up", "down")' in src, "down 还是不算成功"
    assert "abs(dz_nm)" in src, "还在用带符号的 dz 比阈值"
    assert '"unreadable"' in src, "「读不到」没有自己的记账"
    assert 'float(step.get("delta_m") or 0.0)' not in src, (
        "``or 0.0`` 又回来了 —— 那会把「读不到」折叠成「测到了,是零」")


#: 「读不到就是读不到」的**测量量**键名 —— 折成 0 就是撒谎。
#:
#: ⚠️ **计数器不在这里,而且是故意的**:``pulse_phase`` 里有三处
#: ``int(out.get("unreadable") or 0)``,那是「这一跑还没记过这个数」⇒ 0 次,
#: **折得对**。测量量折成 0 是把「没测到」说成「测到了,是零」,那才是本仓
#: 一天记过四次的那条。两者形状相同、语义相反,所以只能按**量**分,不能按写法分。
_MEASUREMENT_KEYS = ("delta_m",)


def _folded_to_zero(src: str) -> list[str]:
    """源码里把**测量量**折成 0 的地方(``X or 0`` / ``X or 0.0``)。

    结构判据:``BoolOp(Or)`` 的**末项**是数值 0 字面量,且左侧读的是测量量键名。
    """
    import ast

    out = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)):
            continue
        last = node.values[-1]
        if not (isinstance(last, ast.Constant)
                and isinstance(last.value, (int, float))
                and not isinstance(last.value, bool)
                and last.value == 0):
            continue
        head = ast.unparse(node.values[0])
        if any(k in head for k in _MEASUREMENT_KEYS):
            out.append(f"{node.lineno}: {ast.unparse(node)}")
    return out


def test_no_measurement_is_folded_to_zero_in_the_pulse_path():
    """守的是**语义**,不是那一个写法。

    上一条测试里那句 ``'float(step.get("delta_m") or 0.0)' not in src`` 是**字符串
    匹配** —— 它守的是**一种写法**:改成 ``or 0``、换个行、多个空格、去掉 ``float()``
    外壳,全都绕得过去,而语义一模一样(把「读不到」折叠成「测到了,是零」)。

    这里按**结构**判:``BoolOp(Or)`` 末项是数值 0 字面量,且左侧读的是测量量。
    两条并存不是重复 —— 字符串那条钉住**历史上真实回归过的那一行**(它是事故的
    原文),这条钉住**那一类**。

    ⚠️ **仍然守不住的**(说出来的缺口不是缺口):
    ``if x is None else 0``、``dict.get(k, 0)`` 的默认值形式、
    以及换一个不在 ``_MEASUREMENT_KEYS`` 里的键名。要更宽就得把「哪些量是测量值」
    这件事变成代码里的一等公民(例如给读数统一走一个 ``reading()`` 取值口),
    那超出这一轮。
    """
    from mast.skills.composite import _tip_phases as T

    src = source_of(T.pulse_phase)
    assert "def pulse_phase" in src, "取源坏了,下面的结论都不作数"

    # 自检①:检测器不是恒空的 —— 喂它一段确凿的违规,必须命中。
    probe = 'def f(step):\n    return float(step.get("delta_m") or 0.0)\n'
    assert _folded_to_zero(probe), "检测器抓不到确凿的违规 —— 判据失去区分力,换判法"
    # 自检②:检测器不是恒真的 —— 计数器那三处合法写法**不许**被算进来。
    counter = 'def f(out):\n    return int(out.get("unreadable") or 0)\n'
    assert not _folded_to_zero(counter), (
        "计数器的 or 0 被当成违规了 —— 这条会把三处正确代码判红")

    hits = _folded_to_zero(src)
    assert not hits, (
        f"pulse_phase 里把测量量折成 0:{hits} —— "
        "那会把「读不到」说成「测到了,是零」")


def test_pulse_success_threshold_is_documented_as_unvalidated_for_down():
    """down 沿用 up 的 20 nm **没有数据支持**,这件事必须留在字段注释里。

    ⚠️ 这里的 ``inspect.getsource(m)`` 收的是**模块**不是函数,**故意保留**:
    读的是整个文件,不存在「按旧行号切错位置」那回事(只怕读到写了一半的中间态,
    而那会以 ``SyntaxError`` 或明显的内容缺失暴露出来)。同一档的还有
    ``test_cross_point_tip_check.py`` 的整文件 ``read_text``。
    **别顺手把它一起改成 ``source_of``** —— 多改一处不如少改一处正确,
    而且 ``source_of`` 按 ``__qualname__`` 在 AST 上找名字,模块根本喂不进去。
    """
    import mast.core.noble_tip_workflow as m

    src = inspect.getsource(m)
    i = src.find("pulse_success_dz_nm: float")
    assert i > 0
    head = src[max(0, i - 1400):i]
    assert "down" in head and ("没有数据支持" in head or "未标定" in head), (
        "字段注释里没说 down 方向的阈值是借用的 —— "
        "一个借来的数看起来和一个标定过的数一模一样")
