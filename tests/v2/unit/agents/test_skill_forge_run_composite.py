"""执行:走的必须是**本来那条路**,外加这条路上唯一缺的那道闸。

``run_composite`` 存在的理由很窄:刚保存的技能本轮还不在工具表里(工具表建图时
冻结),它是那座桥。它**不是**第四个执行收口 —— 现场 ``wrap_skill`` 再分发,子步
照旧走 ``ExecutionContext.run``。这一组钉的是:

* 它真的经 wrap_skill 出来(返回 Command、tool_call_id 对得上、子步走 ctx.run);
* abort / 非组合 / 关闭名单 三道守卫;
* **SAFE 模式闸** —— 这是这条路上唯一新开的口子。模式闸只活在 ``safety_mw``
  里,而那道中间件对没有 skill_metadata 的工具直接放行,``ExecutionContext.run``
  里又没有模式闸。不补的话 ``run_composite`` 就是一条 SAFE 旁路。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from _forge_fixtures import (  # noqa: E402
    StubCtx,
    call,
    fresh_registry,
    spec_with,
    store,
    tools,
    two_step_spec,
)


def _kit(tmp_path, ctx=None):
    ctx = ctx or StubCtx()
    reg = fresh_registry()
    st = store(tmp_path)
    return ctx, reg, tools(reg=reg, ctx=ctx, st=st)


def _save(ts, spec, **kw):
    return call(ts["save_composite"], spec_json=json.dumps(spec), **kw)


def _run(ts, name, params=None, tcid="tc-1"):
    return ts["run_composite"].func(name=name,
                                    params_json=json.dumps(params or {}),
                                    tool_call_id=tcid)


@pytest.fixture
def mode(monkeypatch):
    """Bind the live operating-mode source for the duration of one test."""
    from mast.core import operating_mode as om
    box = {"v": None}
    monkeypatch.setattr(om, "_SOURCE", lambda: box["v"])
    return box


# ── 走的是本来那条路 ────────────────────────────────────────────────

def test_running_a_forged_composite_goes_through_the_normal_skill_adapter(tmp_path):
    ctx, _reg, ts = _kit(tmp_path)
    assert _save(ts, two_step_spec())["ok"] is True
    res = _run(ts, "ScanThenCheck", {"x_m": 0.0, "y_m": 0.0})

    upd = getattr(res, "update", None)
    assert isinstance(upd, dict), f"没拿到 Command,拿到的是 {type(res).__name__}"
    assert upd["executed_skills"] == ["ScanThenCheck"]
    assert "composite_progress" in upd
    # ToolNode 靠 tool_call_id 校验 Command —— 对不上的话整个 update 不会落进
    # state,而那是一种**看起来成功了**的失败。
    assert upd["messages"][0].tool_call_id == "tc-1"


def test_the_sub_steps_go_through_execution_context_run(tmp_path):
    """子步走 ctx.run 就是走全部硬闸、包络与样品闸。StubCtx 的 safe_call 会炸。"""
    ctx, _reg, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    _run(ts, "ScanThenCheck", {"x_m": 0.0, "y_m": 0.0})
    assert [n for n, _ in ctx.ran] == ["ScanAt", "AssessImageQuality"]
    # 工作流参数被解析成了真实数值,而不是把 {"$expr": ...} 原样递下去。
    assert ctx.ran[0][1]["center_x_m"] == 0.0


def test_a_failing_sub_step_is_reported_not_swallowed(tmp_path):
    ctx, _reg, ts = _kit(tmp_path, ctx=StubCtx(fail={"AssessImageQuality"}))
    _save(ts, two_step_spec())
    res = _run(ts, "ScanThenCheck", {"x_m": 0.0, "y_m": 0.0})
    assert "AssessImageQuality" in str(res)
    assert "failed" in str(res) or "失败" in str(res)


def test_an_out_of_envelope_parameter_is_refused_before_anything_runs(tmp_path):
    """包络是**真约束**,不是文案。

    ``_schema_from_metadata`` 把 min/max 写成 pydantic 的 ge/le,所以分发必须走
    ``.invoke()``(带 schema 校验)而不是 ``.func()``(直调,绕过它)。这条测试是
    那个选择的钉子:改回直调,它就红。
    """
    ctx, _reg, ts = _kit(tmp_path)
    spec = two_step_spec("BoundedScan")
    spec["params"] = [{"name": "x_m", "type": "number", "required": True},
                      {"name": "y_m", "type": "number", "required": True}]
    _save(ts, spec)
    # size_m 在 spec 里是字面量,所以越界要从工作流自己的参数进 —— 换个角度:
    # 直接给一个类型对不上的必填参数,pydantic 必须在 execute 之前拦住。
    res = ts["run_composite"].func(name="BoundedScan",
                                   params_json='{"x_m": "不是数字", "y_m": 0.0}',
                                   tool_call_id="tc-x")
    assert ctx.ran == [], "参数校验没拦住,子步已经跑了"
    assert "precondition_failed" in str(res)


# ── 守卫 ────────────────────────────────────────────────────────────

def test_an_active_abort_refuses_before_any_sub_step(tmp_path):
    """中止状态下拒绝开始新的仪器动作 —— 这道闸在 wrap_skill 里,复用即可。"""
    ctx, _reg, ts = _kit(tmp_path, ctx=StubCtx(aborted=True))
    _save(ts, two_step_spec())
    res = _run(ts, "ScanThenCheck", {"x_m": 0.0, "y_m": 0.0})
    assert ctx.ran == []
    assert "aborted" in str(res)


def test_a_plain_skill_cannot_be_dispatched_through_run_composite(tmp_path):
    """普通技能在工具表里有自己的入口(带参数说明与范围提示),按名放开只多一条
    绕过那些说明的路。"""
    ctx, _reg, ts = _kit(tmp_path)
    res = _run(ts, "GetBias")
    assert ctx.ran == []
    assert "不是组合技能" in str(res)


def test_an_unknown_name_says_so_instead_of_crashing(tmp_path):
    _ctx, _reg, ts = _kit(tmp_path)
    assert "注册表里没有" in str(_run(ts, "NoSuchThing"))


def test_a_composite_referencing_a_disabled_skill_is_refused_at_run_time(tmp_path, monkeypatch):
    """双重检查:起草期拦一次,执行期再拦一次。

    两处都要有,因为它们防的是不同的时间:名单是**运行期可变的**(用户在【高级】
    页关掉一条能力),而 spec 是**保存时**校验的。一份保存时合法的 composite,在
    能力被关掉之后必须停止可用 —— 否则关闭就成了「只对新技能生效」。
    """
    ctx, _reg, ts = _kit(tmp_path)
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": "GetCurrent", "params": {}},
    ], name="LaterDisabled")
    assert _save(ts, spec)["ok"] is True          # 保存时它还是开着的

    import mast.agents._shared.skill_forge_tools as mod
    monkeypatch.setattr(mod, "_disabled_names", lambda: frozenset({"GetCurrent"}))
    res = _run(ts, "LaterDisabled")
    assert ctx.ran == [], "关掉之后还是跑了"
    assert "GetCurrent" in str(res)


# ── SAFE / SEMI 模式闸(这条路唯一新开的口子) ────────────────────

def test_safe_mode_refuses_a_composite_that_contains_a_pulse(tmp_path, mode):
    """SAFE 的契约是「专注实验、不修针」—— 包一层组合不会改变它。

    这是本模块唯一自己新增的闸门。它绕不过去的理由要**结构性**地成立:能力标签
    是 ``_inherited_capabilities`` 从子步并集来的,所以一份含 TipPulse 的 spec
    自己就带着 ``bias_pulse``,不需要在这里重走一遍树。
    """
    from mast.core.types import OperatingMode
    ctx, _reg, ts = _kit(tmp_path)
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": "TipPulse",
         "params": {"pulse_v": 3.0, "duration_s": 0.1, "count": 1}},
    ], name="SafePulser", safety_level="confirm")
    assert _save(ts, spec)["ok"] is True

    mode["v"] = OperatingMode.SAFE
    res = _run(ts, "SafePulser")
    assert ctx.ran == [], "SAFE 模式下组合技能里的脉冲还是打出去了"
    assert "safe_mode_tip_processing_blocked" in str(res)


def test_auto_mode_runs_the_same_composite(tmp_path, mode):
    """闸门要有边界:AUTO 下同一份 composite 必须照跑,否则这就不是模式闸而是禁令。"""
    from mast.core.types import OperatingMode
    ctx, _reg, ts = _kit(tmp_path)
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": "TipPulse",
         "params": {"pulse_v": 3.0, "duration_s": 0.1, "count": 1}},
    ], name="AutoPulser", safety_level="confirm")
    _save(ts, spec)
    mode["v"] = OperatingMode.AUTO
    _run(ts, "AutoPulser")
    assert [n for n, _ in ctx.ran] == ["GetBias", "TipPulse"]


def test_safe_mode_does_not_block_a_composite_that_never_touches_the_tip(tmp_path, mode):
    """SAFE 是「不修针」,不是「什么都不做」。"""
    from mast.core.types import OperatingMode
    ctx, _reg, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    mode["v"] = OperatingMode.SAFE
    _run(ts, "ScanThenCheck", {"x_m": 0.0, "y_m": 0.0})
    assert [n for n, _ in ctx.ran] == ["ScanAt", "AssessImageQuality"]


def test_unreadable_metadata_refuses_instead_of_running_ungated(tmp_path, monkeypatch):
    """「读不到元数据」**不是**「没有限制」。

    模式闸和 DANGEROUS 判据都读 metadata;meta=None 时两者都会静静放行,而那
    正是本仓一天之内记过四次的形状(「读不到」被折叠成一个具体的答案)。
    这条路实践中基本不可达(wrap_skill 也读 metadata),钉住它是为了让
    「不可达」写下来,而不是靠运气。
    """
    import mast.agents._shared.skill_forge_tools as mod
    ctx, _reg, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    monkeypatch.setattr(mod, "_meta_of", lambda reg, cls: None)
    res = _run(ts, "ScanThenCheck", {"x_m": 0.0, "y_m": 0.0})
    assert ctx.ran == [], "元数据读不到却照跑了 —— 那一路上没有任何闸门在看"
    assert "读不到" in str(res)


def test_an_unreadable_mode_does_not_wedge_the_tool(tmp_path, mode):
    """「读不到模式」不是「拒绝执行」——那会让一次读取故障停掉整台机器。"""
    ctx, _reg, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    mode["v"] = None
    _run(ts, "ScanThenCheck", {"x_m": 0.0, "y_m": 0.0})
    assert len(ctx.ran) == 2


# ── DANGEROUS 的那一行台账 ───────────────────────────────────────

def test_a_dangerous_composite_runs_and_leaves_a_notice(tmp_path, monkeypatch):
    """⑰ 之后 DANGEROUS 照跑,但**必须**留一行。

    顶层这一次本来是 ``AutoApprovalNoticeMiddleware`` 发的,而它同样看
    ``skill_metadata`` —— 走 run_composite 就没人发了。少的那一行正是用户事后
    问「今晚它自己批了什么」时要查的东西。
    """
    import mast.agents._shared.skill_forge_tools as mod
    notices: list[tuple] = []
    monkeypatch.setattr(mod, "_notice",
                        lambda name, meta, params: notices.append((name, meta)))
    ctx, reg, ts = _kit(tmp_path)
    from mast.core.types import SafetyLevel
    from mast.agents._shared.skill_forge_tools import _disabled_names
    off = _disabled_names()
    victim = next(m.name for m in reg.list_skills()
                  if m.safety_level is SafetyLevel.DANGEROUS
                  and not m.parameters and m.name not in off)
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": victim, "params": {}},
    ], name="LoudlyDangerous", safety_level="dangerous")
    assert _save(ts, spec)["ok"] is True
    _run(ts, "LoudlyDangerous")
    assert [n for n, _ in ctx.ran] == ["GetBias", victim], "DANGEROUS 被拦住了"
    assert [n for n, _ in notices] == ["LoudlyDangerous"], "跑了但没留台账"


def test_the_notice_predicate_is_the_shared_one(tmp_path):
    """判据单源:``would_have_asked`` 对一个 DANGEROUS composite 必须说「会」。"""
    from mast.core.auto_approval import would_have_asked
    from mast.core.types import SafetyLevel, SkillMetadata
    meta = SkillMetadata(name="X", description="d",
                         safety_level=SafetyLevel.DANGEROUS)
    assert would_have_asked(meta, tool_name="X", args={}) is not None
