"""扎针技能:**不显式给参数时,针尖上不许出现一个高压**。

## 这个文件为什么单独存在(读之前先读这一段)

一组缺陷合起来的效果是:MAST 每次扎针都无条件在结上打 **3 V**,而安全门放行、
回包不说。用户范本要求的是 qPlus 扎针前把偏压降到 **20 mV**(否则音叉起振、
每次扎针都在毁针)—— 3 V 是它的 **150 倍**。

而当时**所有相关测试都是绿的**。原因不是漏测,是**校验交给了不会犯这个错的那
一方**:

* 钉「bias 缺省 = 当前成像偏压」的三个测试(``test_tip_shaper_readback.py``)
  全部直接调 ``.execute()`` —— 恰好是修法**生效**的那条路;
* 工具路径的测试要么显式传 ``bias_v``,要么只断言 ``executed_skills``
  (全默认调用,却什么都不看)。

真正出事的是 **LangChain 工具路径**:``skill_adapter._schema_from_metadata`` 把
``ParameterSpec.default`` 灌进 pydantic ``Field``,模型省略参数时 pydantic 会把
那个默认值**物化**成实参。于是技能里那段「没给就去读成像偏压」永远走不到 ——
它看到的从来不是 ``None``,而是 3.0。

**所以这个文件里的每一条都必须走 ``tool.invoke`` + 空参 ToolCall**(模型省略可选
参数时的真实形状),断言的是**下发给 Nanonis 的实参**,不是我们配置里写了什么。
用 ``tool.func(...)`` / ``skill.execute(...)`` 改写其中任何一条,都会把这个文件变回
它要防的那种测试。

## 两条路都要钉

同一个洞有两个入口,只堵一个 = 修法在另一条路上仍是死代码:

* **agent 路**:pydantic 默认抢先(上面那段);
* **composite 路**:``core/tip_conditioning_policy.py`` 的方案表抢先 ——
  ``apply_tip_policy`` 在技能读偏压**之前**跑,表里写着 3.0 就填 3.0。

跑:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/builtins/test_tip_shaper_hot_default.py -q
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

from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core import tip_state
from mast.core.safety import is_electrical_pulse
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.tip_shaper import TipShape
from mast.skills.builtins.tip_shaper_readback import TipShapeWithReadback

#: 用户范本的成像偏压(qPlus 扎针前必须降到这个量级)。
IMAGING_BIAS_V = 0.02

#: 「显著偏压」的判据线。范本是 20 mV;1 V 已经是它的 50 倍,任何缺省调用碰到
#: 这条线都算 MAST 自己发明了一个电压。
HOT_V = 1.0

#: ``TipShaper_PropsSet`` 的 11 个实参里,哪几个槽是**电压**。
#: (Switch_Off_Delay, Change_Bias, **Bias_V**, Tip_Lift_m, Lift_Time_1_s,
#:  **Bias_Lift_V**, Bias_Settling_Time_s, Lift_Height_m, Lift_Time_2_s,
#:  End_Wait_Time_s, Restore_Feedback) —— 与厂商协议逐条核对过。
SLOT_CHANGE_BIAS = 1
SLOT_BIAS_V = 2
SLOT_TIP_LIFT = 3
SLOT_BIAS_LIFT_V = 5
SLOT_LIFT_HEIGHT = 7
VOLTAGE_SLOTS = (SLOT_BIAS_V, SLOT_BIAS_LIFT_V)

_TIP_SHAPING = frozenset({"tip_shaping"})


# ── 一个只会回话、不会开机器的 context ─────────────────────────────────────

@dataclass
class FakeCtx:
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    bias_v: float = IMAGING_BIAS_V

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        canned: dict[str, Any] = {
            "Bias_Get": [self.bias_v],
            "Current_Get": [1.0e-10],
            "ZCtrl_ZPosGet": [1.0e-9],
            "FolMe_XYPosGet": [1.0e-9, 2.0e-9],
            "TipShaper_PropsSet": [],
            "TipShaper_Start": [],
        }
        if method not in canned:
            return NanonisCallRecord(method=method, args=args,
                                     error=f"unmocked: {method}")
        return NanonisCallRecord(method=method, args=args,
                                 return_value=("", b"", canned[method]))


def _tool_and_ctxs(skill_cls):
    """返回 ``(tool, ctxs)``;``ctxs[-1]`` 是最后一次调用用的 context。"""
    ctxs: list[FakeCtx] = []

    def provider() -> FakeCtx:
        ctx = FakeCtx()
        ctxs.append(ctx)
        return ctx

    return wrap_skill(skill_cls, provider), ctxs


def _tool_call(tool, args: dict | None = None):
    """**真正的** LangChain 工具路径:一个 ToolCall,走 args_schema 校验与默认物化。

    ⚠️ 不要换成 ``tool.func(**kwargs)`` —— 那正好绕过 pydantic,也就绕过了本文件
    要钉的那个默认值。
    """
    return tool.invoke({"name": tool.name, "args": dict(args or {}),
                        "id": "pin-1", "type": "tool_call"})


def _props_args(ctx: FakeCtx) -> tuple:
    sets = [a for m, a in ctx.calls if m == "TipShaper_PropsSet"]
    assert len(sets) == 1, f"期望正好一次 PropsSet,实际 {ctx.calls}"
    return sets[0]


@pytest.fixture(autouse=True)
def _no_tip_registered():
    """未登记针尖 = 方案表落到 ``_DEFAULT`` 那一档 —— 正是曾经填 3.0 的那一档。"""
    tip_state.set_current_tip(None)
    yield
    tip_state.set_current_tip(None)


@pytest.fixture(autouse=True)
def _fast_readback(readback_clock):
    """采集循环用假时钟(见本目录 conftest),否则每条测试要烧掉近一秒墙钟。"""
    return readback_clock


BOTH_SKILLS = pytest.mark.parametrize(
    "skill_cls", [TipShape, TipShapeWithReadback],
    ids=["TipShape", "TipShapeWithReadback"])


# ── ① 缺省调用不许打出高压(工具路径) ──────────────────────────────────────

@BOTH_SKILLS
def test_empty_tool_call_puts_no_volt_on_the_tip(skill_cls):
    """未指定的工作偏压使用回读值。"""
    tool, ctxs = _tool_and_ctxs(skill_cls)
    _tool_call(tool)                       # ← 空参 ToolCall,一个参数都不给
    args = _props_args(ctxs[-1])

    hot = [(i, args[i]) for i in VOLTAGE_SLOTS if abs(float(args[i])) >= HOT_V]
    assert not hot, (
        f"{skill_cls.__name__} 缺省调用下发了高压 {hot} —— 实参 {args}。"
        f"用户范本是 {IMAGING_BIAS_V} V;没人要求过的电压不该由默认值发明。")


@BOTH_SKILLS
def test_empty_tool_call_follows_the_imaging_bias_on_both_bias_fields(skill_cls):
    """不只是「不高」,而是**跟随刚读到的成像偏压** —— 两个 bias 字段都要。

    ``bias_lift_v`` 单列一条,因为 2026-08-10 的修法只修了 ``bias_v``,
    **孪生兄弟没修**,而它才是无条件施加的那一个。
    """
    tool, ctxs = _tool_and_ctxs(skill_cls)
    _tool_call(tool)
    ctx = ctxs[-1]
    args = _props_args(ctx)

    assert "Bias_Get" in [m for m, _ in ctx.calls], "根本没去读成像偏压"
    assert float(args[SLOT_BIAS_V]) == pytest.approx(IMAGING_BIAS_V)
    assert float(args[SLOT_BIAS_LIFT_V]) == pytest.approx(IMAGING_BIAS_V), (
        "Bias Lift 没跟上 —— 它是**无条件施加**的那一个,"
        "change_bias 关掉也拦不住它")


@BOTH_SKILLS
def test_the_reply_states_the_bias_that_was_actually_applied(skill_cls):
    """回包要说 ``bias_lift_v``。

    以前回包只写 ``bias_v``:关掉 change_bias 的调用方拿到一份「没有电压」的回执,
    而针尖上刚过了 3 V。**回包不说**是这组缺陷能活这么久的一半原因。
    """
    tool, ctxs = _tool_and_ctxs(skill_cls)
    _tool_call(tool)
    args = _props_args(ctxs[-1])
    # 回包里的值必须等于**真正下发的那个**,不是我们希望它是什么。
    assert float(args[SLOT_BIAS_LIFT_V]) == pytest.approx(IMAGING_BIAS_V)


# ── ③ change_bias 缺省 = 关 ─────────────────────────────────────────────────

@BOTH_SKILLS
def test_change_bias_defaults_to_off_on_the_tool_path(skill_cls):
    """裸技能的默认要与工作流层的判断一致。

    ``composite/_tip_phases.py`` 自己拒绝走 change-bias 那条路:「TipShaper 的
    change-bias 只能**阶跃**改偏压…那一次阶跃本身就是一记冲量」。默认却是 True,
    等于工作流说「这条路会毁针」而技能说「这是缺省」。
    编码:1=True,2=False。
    """
    tool, ctxs = _tool_and_ctxs(skill_cls)
    _tool_call(tool)
    args = _props_args(ctxs[-1])
    assert args[SLOT_CHANGE_BIAS] == 2, (
        f"change_bias 缺省仍是开(实参 {args}) —— 与工作流层的判断相反")


# ── ④ lift_height 规则下沉 ──────────────────────────────────────────────────

@BOTH_SKILLS
def test_lift_height_defaults_to_minus_tip_lift_on_the_tool_path(skill_cls):
    """压进去多少就抬回来多少。

    这条规则原本只活在三个 composite 调用方里(``_tip_phases.py`` /
    ``shape_tip_on_surface.py`` / ``builtin_composites.py``),而**裸技能默认 0.0**:
    模型只给 ``tip_lift_m`` 时,针压下去就不抬了。规则住在调用方而不是被调方。
    """
    tool, ctxs = _tool_and_ctxs(skill_cls)
    _tool_call(tool, {"tip_lift_m": "-2n"})
    args = _props_args(ctxs[-1])
    assert float(args[SLOT_TIP_LIFT]) == pytest.approx(-2e-9)
    assert float(args[SLOT_LIFT_HEIGHT]) == pytest.approx(2e-9), (
        f"第二段斜坡没有抬回来(实参 {args})")


@BOTH_SKILLS
def test_an_explicit_zero_lift_height_is_still_honoured(skill_cls):
    """「压下去不抬」是一个合法意图 —— 只是不该是**缺省**意图。"""
    tool, ctxs = _tool_and_ctxs(skill_cls)
    _tool_call(tool, {"tip_lift_m": "-2n", "lift_height_m": "0n"})
    args = _props_args(ctxs[-1])
    assert float(args[SLOT_LIFT_HEIGHT]) == pytest.approx(0.0)


# ── ② 安全门要看 bias_lift_v ────────────────────────────────────────────────

def test_gate_calls_a_default_poke_an_electrical_pulse():
    """「只给 ``tip_lift_m``」的缺省调用 = 电学脉冲。

    这一条与 ③ 是**一对**:change_bias 缺省改成 False 之后,旧判据「一见
    ``change_bias=False`` 就短路返回 False」会让**每一次缺省扎针**都被判成非电学
    脉冲。单独做 ③ 等于亲手在安全门上开一个洞。
    """
    assert is_electrical_pulse(
        "TipShape", {"tip_lift_m": -1e-9}, _TIP_SHAPING) is True


def test_gate_is_not_disarmed_by_change_bias_false():
    """``change_bias=False`` **不等于不加电**。

    厂商同一句话里一个带条件一个不带:「Bias (V) … **if Change Bias is True**」/
    「Bias Lift (V) … applied **just after the first Z ramping**」。
    旧判据在 ``core/safety.py`` 一见 ``change_bias=False`` 就 return False,
    **从不看 bias_lift_v** —— SAFE/SEMI 判成「非电学脉冲」放行,硬件照打 3 V。
    """
    assert is_electrical_pulse(
        "TipShape", {"change_bias": False, "bias_lift_v": 3.0},
        _TIP_SHAPING) is True


def test_gate_is_not_disarmed_by_zeroing_only_bias_v():
    """把 ``bias_v`` 归零从来没有解除过 Bias Lift —— 它们是两个字段。"""
    assert is_electrical_pulse(
        "TipShape", {"change_bias": True, "bias_v": 0, "bias_lift_v": 3.0},
        _TIP_SHAPING) is True


def test_gate_says_no_only_when_both_biases_are_explicitly_zero():
    """判据不是「一律 True」—— 真的纯机械下压仍要判为非电学脉冲,
    否则这道门就没有分辨力,下一个人会有理由把它删掉。"""
    assert is_electrical_pulse(
        "TipShape", {"change_bias": False, "bias_v": 0, "bias_lift_v": 0},
        _TIP_SHAPING) is False


def test_gate_treats_an_omitted_lift_bias_as_live():
    """没说 ≠ 零。缺省时的实际取值由技能在运行时解析,门看不见 —— fail-closed。"""
    assert is_electrical_pulse(
        "TipShape", {"change_bias": False}, _TIP_SHAPING) is True


# ── 第二条路:方案表不许再发明一个 shaper 偏压 ──────────────────────────────

@pytest.mark.parametrize("tip", [
    None,
    {"id": "q1", "name": "qPlus #1", "material": "PtIr",
     "fabrication": "cut", "form": "qplus"},
], ids=["unregistered", "qplus"])
def test_the_policy_table_no_longer_invents_a_shaper_bias(tip):
    """方案表不填扎针偏压默认值，交给技能回读当前偏压；包络上限仍然独立生效。"""
    from mast.core.tip_conditioning_resolver import resolve_conditioning

    tip_state.set_current_tip(tip)
    plan = resolve_conditioning(("shaper_bias_v", "shaper_lift_v"))
    assert "shaper_bias_v" not in plan.params, plan.params
    assert "shaper_lift_v" not in plan.params, plan.params


def test_a_registered_wire_tip_keeps_its_owned_policy_value():
    """有主的、写了理由的每种针的值**保留** —— 删的是无主 stub,不是整张表。

    钉这条是因为「顺手把方案表清空」是一个看起来很像修复的过度修法。
    """
    from mast.core.tip_conditioning_resolver import resolve_conditioning

    tip_state.set_current_tip({"id": "w1", "name": "W-etched #1", "material": "W",
                               "fabrication": "etched", "form": "stm_wire"})
    plan = resolve_conditioning(("shaper_bias_v",))
    assert plan.params.get("shaper_bias_v") == 4.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
