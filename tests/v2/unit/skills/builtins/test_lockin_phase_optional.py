"""相位未显式指定时不写寄存器，默认 None 不能被表单或方案替换为零。"""
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

from mast.skills.builtins.lockin import ConfigureLockIn  # noqa: E402

_BASE = {"modulator": 1, "mod_on": True, "frequency_hz": 973.0,
         "amplitude_v": 0.02}


class _Ctx:
    """上下文替身模拟参数锁定。"""

    def __init__(self, locked: bool = False):
        self.locked = locked
        self.calls: list[str] = []

    def safe_call(self, method, *args, **kw):
        self.calls.append(method)
        err = ("Access Denied" if (self.locked and method == "LockIn_ModPhasSet")
               else "")

        class _R:
            error = err
            return_value = ("", b"", [0.0])

        _R.method, _R.args = method, args
        return _R()


def test_phase_has_no_declared_default():
    """守卫是 ``is not None``,所以任何默认值都会把它变成 no-op。"""
    spec = next(p for p in ConfigureLockIn().metadata().parameters
                if p.name == "phase_deg")
    assert spec.default is None, (
        "phase_deg 又有默认值了 —— 那个值会让「没人要求改相位」变成「写 0°」")
    assert spec.required is False


# 通过实际工具适配器验证参数传递。


def _tool_calls(**kwargs) -> tuple[list[str], object]:
    """经真实工具路径调一次,返回 (TCP 调用序列, 结果)。

    结果文本一并返回,因为「调用序列是空的」也能让 ``not in`` 断言通过 ——
    第一版就撞上了(参数名写错 → precondition_failed → 一次**空过**)。凡是断言
    「某调用没发生」,都要同时断言「该发生的发生了」。

    ``kwargs`` 里给 ``None`` 表示**整个键不传**,不是「传一个 None」。这两件事在
    pydantic 下不同:显式 None 直接落成 None,而**省略**才会去取字段默认值 ——
    也就是这一整节要测的那条路。第二版又在这里栽了一次:用 ``amplitude_v=None``
    去测「省略」,于是把默认值物化那一步整个绕开了,变异时测试照样绿。
    """
    from mast.agents._shared.skill_adapter import wrap_skill

    ctx = _Ctx()
    tool = wrap_skill(ConfigureLockIn, lambda: ctx)
    args = {"mod_on": True, "frequency_hz": "973", "amplitude_v": "0.02"}
    args.update(kwargs)
    args = {k: v for k, v in args.items() if v is not None}   # None = 不传这个键
    res = tool.func(tool_call_id="t-1", state={}, **args)
    return ctx.calls, res


def test_the_materialised_default_is_none_on_the_real_path():
    """模型看到的 schema 里,phase_deg 的默认值必须是 None 而不是 0.0。

    这是「不传就不发」在**真实路径**上唯一的支点:pydantic 字段默认值一旦是 0.0,
    execute 收到的就永远是 0.0,下游那句 ``if phase_deg is not None`` 再正确也没用。
    """
    from mast.agents._shared.skill_adapter import wrap_skill

    tool = wrap_skill(ConfigureLockIn, lambda: _Ctx())
    assert tool.args_schema.model_fields["phase_deg"].default is None, (
        "wrap_skill 把一个非 None 默认值物化进去了 —— 「没传」已经变成「传了」")


def test_not_asking_for_a_phase_emits_no_modphasset_through_the_tool_path():
    """未指定相位时不得调用 ModPhasSet。"""
    calls, res = _tool_calls()
    assert "LockIn_ModPhasSet" not in calls, (
        f"不传相位仍然发了 modphasset:{calls}")
    assert "LockIn_ModPhasFreqSet" in calls and "LockIn_ModAmpSet" in calls


def test_a_locked_modulator_phase_does_not_break_the_tool_path():
    """修改其他参数时不应额外写入相位。"""
    from mast.agents._shared.skill_adapter import wrap_skill

    ctx = _Ctx(locked=True)
    tool = wrap_skill(ConfigureLockIn, lambda: ctx)
    res = tool.func(tool_call_id="t-1", state={}, mod_on=True,
                    frequency_hz="973", amplitude_v="0.02")
    assert "LockIn_ModPhasFreqSet" in ctx.calls, "什么都没跑,这条断言是空的"
    assert "LockIn_ModPhasSet" not in ctx.calls
    text = str(getattr(res, "update", res))
    assert "Access Denied" not in text, f"相位锁把不相干的配置也弄失败了:{text}"


# 声明的 schema 与执行适配器分别验证，防止默认值在入口被改写。


def test_omitting_the_amplitude_does_not_write_an_amplitude():
    """未指定振幅时不得调用 ModAmpSet。"""
    calls, _ = _tool_calls(amplitude_v=None)
    assert "LockIn_ModOnOffSet" in calls, "什么都没跑,这条断言是空的"
    assert "LockIn_ModAmpSet" not in calls, (
        f"省略幅度却写了幅度:{calls}")


def test_a_declared_default_does_not_reach_execute():
    """直接执行与工具适配入口都须保留可选参数的 None 语义。"""
    from mast.agents._shared.skill_adapter import wrap_skill

    seen: list[dict] = []

    class _Spy(ConfigureLockIn):
        def execute(self, context, params):
            seen.append(dict(params))
            return super().execute(context, params)

    tool = wrap_skill(_Spy, lambda: _Ctx())
    tool.func(tool_call_id="t-1", state={}, mod_on=True, frequency_hz="973")
    assert seen, "execute 没被调到"
    assert "amplitude_v" not in seen[0] or seen[0]["amplitude_v"] is None, (
        f"声明默认值被塞进了 params:{seen[0]}")


def test_an_explicit_zero_amplitude_still_reaches_the_hardware():
    """守卫写成 ``>= 0`` 是有原因的:开调制前把幅度显式压到 0 V 是正当操作。

    修「省略 = 别写」不能顺手把「显式 0」也堵死 —— 那是把一个真实用例换成另一个 bug。
    """
    calls, _ = _tool_calls(amplitude_v="0")
    assert "LockIn_ModAmpSet" in calls


def test_omitting_the_frequency_does_not_write_a_frequency():
    calls, _ = _tool_calls(frequency_hz=None)
    assert "LockIn_ModOnOffSet" in calls
    assert "LockIn_ModPhasFreqSet" not in calls


def test_the_schema_advertises_no_default_for_a_write_gated_parameter():
    """JSON schema 中未指定的可选参数默认值为 None。"""
    for spec in ConfigureLockIn().metadata().parameters:
        if spec.required:
            continue
        assert spec.default is None, (
            f"{spec.name} 声明了默认值 {spec.default!r} —— 它会出现在模型读到的 "
            "schema 里,而模型会照着传,于是「省略」变成「显式传了这个值」")


def test_the_declared_default_is_what_the_model_actually_sees():
    """把「default 会进模型读到的 schema」这一步也钉住 —— 上一条的前提。

    否则上一条只是一句关于 dataclass 字段的断言,与模型看到什么无关。
    """
    from mast.agents._shared.skill_adapter import wrap_skill

    tool = wrap_skill(ConfigureLockIn, lambda: _Ctx())
    props = tool.args_schema.model_json_schema().get("properties") or {}
    for name in ("phase_deg", "amplitude_v", "frequency_hz"):
        assert props[name].get("default", None) is None, (
            f"模型读到的 schema 里 {name} 仍带默认值:{props[name].get('default')}")


def test_not_asking_for_a_phase_writes_no_phase():
    ctx = _Ctx()
    res = ConfigureLockIn().execute(ctx, dict(_BASE))
    assert res.success
    assert "LockIn_ModPhasSet" not in ctx.calls
    # 而且不许在报告里凭空说一个相位。
    assert res.data["phase_written"] is False
    assert res.data["phase_deg"] is None


def test_a_locked_phase_does_not_break_a_run_that_never_wanted_it():
    """相位锁着的机器上,不碰相位的调用必须照常成功。

    这是一种真实场景:锁相位是为了保护标定,不是为了让 lock-in 配置
    整个不可用。"""
    ctx = _Ctx(locked=True)
    res = ConfigureLockIn().execute(ctx, dict(_BASE))
    assert res.success, res.error
    assert "LockIn_ModPhasFreqSet" in ctx.calls and "LockIn_ModAmpSet" in ctx.calls


def test_explicitly_asking_for_a_phase_still_writes_it():
    """降级不能变成「相位再也改不了」—— 明确要求时照旧写。"""
    ctx = _Ctx()
    res = ConfigureLockIn().execute(ctx, dict(_BASE, phase_deg=12.5))
    assert res.success
    assert "LockIn_ModPhasSet" in ctx.calls
    assert res.data["phase_written"] is True
    assert res.data["phase_deg"] == 12.5


def test_an_explicit_phase_on_a_locked_rig_fails_loudly():
    """要求写、写不了 —— 必须失败,不能悄悄当成写好了。"""
    ctx = _Ctx(locked=True)
    res = ConfigureLockIn().execute(ctx, dict(_BASE, phase_deg=12.5))
    assert not res.success
    assert "Access Denied" in (res.error or "")


# ══════════════════════════════════════════════════════════════════════
# 报告层:同一句谎的另外两个载体(amplitude / frequency)
# ══════════════════════════════════════════════════════════════════════
#
# 相位那条修的是「报了一个没写过的值」,而 amplitude/frequency 一直照着
# ``params.get(..., 0.0)`` 报 —— 省略幅度的调用,结果里写着「幅度 0 V」。0 V 在这
# 里不是一个无害的占位:它读起来正好是「调制等于没开」,而真实幅度原封不动地留在
# 硬件里。判据统一成**写没写**,不是**传没传**:0 Hz 传了也不会下发(无效调制频
# 率,守卫是 ``> 0``),那种情况报「已写」同样是谎。


def test_omitting_the_amplitude_reports_no_amplitude():
    ctx = _Ctx()
    res = ConfigureLockIn().execute(ctx, {"mod_on": True, "frequency_hz": 973.0})
    assert res.success
    assert "LockIn_ModAmpSet" not in ctx.calls, "探针无效:这一路其实写了幅度"
    assert res.data["amplitude_v"] is None
    assert res.data["amplitude_written"] is False


def test_omitting_the_frequency_reports_no_frequency():
    ctx = _Ctx()
    res = ConfigureLockIn().execute(ctx, {"mod_on": True, "amplitude_v": 0.02})
    assert res.success
    assert "LockIn_ModPhasFreqSet" not in ctx.calls, "探针无效:这一路其实写了频率"
    assert res.data["frequency_hz"] is None
    assert res.data["frequency_written"] is False


def test_a_written_amplitude_is_echoed_with_its_flag():
    ctx = _Ctx()
    res = ConfigureLockIn().execute(ctx, {"mod_on": True, "amplitude_v": 0.02})
    assert res.success
    assert res.data["amplitude_v"] == 0.02
    assert res.data["amplitude_written"] is True


def test_a_written_frequency_is_echoed_with_its_flag():
    ctx = _Ctx()
    res = ConfigureLockIn().execute(ctx, {"mod_on": True, "frequency_hz": 973.0})
    assert res.success
    assert res.data["frequency_hz"] == 973.0
    assert res.data["frequency_written"] is True


def test_each_parameter_is_reported_on_its_own():
    """一个写了一个没写 —— 两侧的标志必须各说各的。

    共用一个「有没有传参数」的标志会让「设了幅度、没动频率」报成两个都设了。
    """
    only_amp = ConfigureLockIn().execute(_Ctx(), {"mod_on": True, "amplitude_v": 0.02})
    assert only_amp.data["amplitude_written"] is True
    assert only_amp.data["frequency_written"] is False
    assert only_amp.data["frequency_hz"] is None

    only_freq = ConfigureLockIn().execute(_Ctx(), {"mod_on": True, "frequency_hz": 973.0})
    assert only_freq.data["frequency_written"] is True
    assert only_freq.data["amplitude_written"] is False
    assert only_freq.data["amplitude_v"] is None


def test_an_explicit_zero_amplitude_is_reported_as_written():
    """显式 0 V 是真下发的一次写(守卫 ``>= 0`` 就是为它留的)。

    用 falsy 判断代替 ``写没写`` 会把这一次真实的写吞成「没写」—— 而「幅度确实
    被压到 0 了」和「幅度没被碰过」是用户必须分得开的两件事。
    """
    ctx = _Ctx()
    res = ConfigureLockIn().execute(ctx, {"mod_on": True, "amplitude_v": 0.0})
    assert "LockIn_ModAmpSet" in ctx.calls, "探针无效:显式 0 根本没下发"
    assert res.data["amplitude_v"] == 0.0
    assert res.data["amplitude_written"] is True


def test_a_frequency_of_zero_is_never_reported_as_written():
    """传了 ≠ 写了:0 Hz 被守卫跳过(无效调制频率),报告不能说它写了。

    这是把标志绑在**实际下发**而不是 ``is not None`` 上的理由;改成后者这条会红。
    """
    ctx = _Ctx()
    res = ConfigureLockIn().execute(ctx, {"mod_on": True, "frequency_hz": 0.0})
    assert res.success
    assert "LockIn_ModPhasFreqSet" not in ctx.calls, "探针无效:0 Hz 居然下发了"
    assert res.data["frequency_written"] is False
    assert res.data["frequency_hz"] is None


def test_the_tool_path_result_never_claims_an_amplitude_it_did_not_write():
    """模型真正读到的那份文本里,不许出现一个没被写过的幅度。

    ConfigureLockIn 没有 ``summary``,所以适配器把整个 ``data`` 直接当摘要送进
    ToolMessage —— 谎报的那个 0.0 是**模型逐字读到**的,不只是记录里的一个字段。
    """
    calls, res = _tool_calls(amplitude_v=None)
    assert "LockIn_ModAmpSet" not in calls, "探针无效:这一路其实写了幅度"
    text = res.update["messages"][0].content
    assert "'amplitude_written': False" in text, f"标志没到模型手里:{text}"
    assert "'amplitude_v': 0.0" not in text, f"结果里报了一个没写过的幅度:{text}"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
