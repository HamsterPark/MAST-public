"""v2 unit tests for mast.skills.builtins.imaging.

Skills covered (7):
  StartScan (CONFIRM), StopScan (AUTO), ConfigureScan (CONFIRM),
  SetScanSpeed (CONFIRM), GetScanXYPosition (AUTO),
  ScanBackgroundPaste (CONFIRM), ScanBackgroundDelete (CONFIRM).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_imaging.py -x -v
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
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.imaging import (
    ConfigureScan,
    GetScanXYPosition,
    ScanBackgroundDelete,
    ScanBackgroundPaste,
    SetScanSpeed,
    StartScan,
    StopScan,
)


# ── FakeCtx ──────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def make_provider(canned: dict[str, Any] | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# ── Shape tests ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("skill_cls,expected_name,expected_danger", [
    (StartScan, "StartScan", "CONFIRM"),
    (StopScan, "StopScan", "AUTO"),
    (ConfigureScan, "ConfigureScan", "CONFIRM"),
    (SetScanSpeed, "SetScanSpeed", "CONFIRM"),
    (GetScanXYPosition, "GetScanXYPosition", "AUTO"),
    (ScanBackgroundPaste, "ScanBackgroundPaste", "CONFIRM"),
    (ScanBackgroundDelete, "ScanBackgroundDelete", "CONFIRM"),
])
def test_imaging_skill_shape(skill_cls, expected_name, expected_danger):
    tool = wrap_skill(skill_cls, make_provider())
    assert tool.name == expected_name
    assert tool.metadata["danger_level"] == expected_danger
    assert tool.metadata["skill_source"].endswith(".imaging")


def test_configure_scan_required_fields():
    tool = wrap_skill(ConfigureScan, make_provider())
    fields = tool.args_schema.model_fields
    for req in ("center_x_m", "center_y_m", "width_m", "height_m"):
        assert req in fields
        assert fields[req].is_required()
    assert not fields["angle_deg"].is_required()
    assert not fields["channels"].is_required()


def test_set_scan_speed_required_fields():
    tool = wrap_skill(SetScanSpeed, make_provider())
    fields = tool.args_schema.model_fields
    for req in ("fwd_speed", "bwd_speed", "fwd_line_time", "bwd_line_time"):
        assert req in fields
        assert fields[req].is_required()
    assert not fields["keep_const"].is_required()


def test_start_scan_has_no_params():
    """StartScan has no parameters — args_schema has no required fields."""
    tool = wrap_skill(StartScan, make_provider())
    fields = tool.args_schema.model_fields
    required = [k for k, v in fields.items() if v.is_required()]
    assert required == []


# ── Execution tests ───────────────────────────────────────────────────────────

#: 一台**能回答「连续扫描开着吗」**的替身机器,回答是「关」。
#:
#: 这两条测试问的是「StartScan 会不会发 Scan_Action(0, …)」,所以前提必须成立:
#: 2026-08-19 起,读不到 continuous 状态的机器**根本不会被发起扫描**(见
#: imaging.StartScan._continuous_gate)。原来这里的 ``return_value: None`` 表示
#: 「回包读不懂」,于是这两条测的其实是那道闸门,不是它们标题上写的那件事。
_PROPS_OFF = [0, b"", [0, 1, 0, "au111", "", ["Bias", "Current"]]]
_CANNED_STARTABLE = {
    "Scan_PropsGet": {"return_value": _PROPS_OFF},
    "Scan_PropsSet": {"return_value": None},
    "Scan_Action": {"return_value": None},
}


def test_start_scan_executes():
    tool = wrap_skill(StartScan, make_provider(_CANNED_STARTABLE))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["StartScan"]


def test_start_scan_calls_scan_action_0():
    """StartScan should call Scan_Action with action=0 (start)."""
    canned = _CANNED_STARTABLE
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(StartScan, capturing_provider)
    _invoke(tool)
    ctx = instances[-1]
    action_calls = [c for c in ctx.calls if c[0] == "Scan_Action"]
    assert len(action_calls) >= 1
    assert action_calls[-1][1][0] == 0  # action=0 means start


def test_stop_scan_executes():
    canned = {"Scan_Action": {"return_value": None}}
    tool = wrap_skill(StopScan, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["StopScan"]


def test_stop_scan_calls_action_1():
    """StopScan should call Scan_Action with action=1."""
    canned = {"Scan_Action": {"return_value": None}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(StopScan, capturing_provider)
    _invoke(tool)
    ctx = instances[-1]
    calls = [c for c in ctx.calls if c[0] == "Scan_Action"]
    assert calls[0][1][0] == 1  # action=1 means stop


def test_configure_scan_executes():
    canned = {
        "Scan_FrameSet": {"return_value": None},
        "Signals_NamesGet": {"return_value": None},
        "Scan_SpeedSet": {"return_value": None},
    }
    tool = wrap_skill(ConfigureScan, make_provider(canned))
    result = _invoke(tool, center_x_m=0.0, center_y_m=0.0, width_m=100e-9, height_m=100e-9)
    assert result.update["executed_skills"] == ["ConfigureScan"]


def test_set_scan_speed_executes():
    canned = {"Scan_SpeedSet": {"return_value": None}}
    tool = wrap_skill(SetScanSpeed, make_provider(canned))
    result = _invoke(
        tool,
        fwd_speed=500e-9,
        bwd_speed=500e-9,
        fwd_line_time=0.1,
        bwd_line_time=0.1,
    )
    assert result.update["executed_skills"] == ["SetScanSpeed"]


def test_get_scan_xy_position_executes():
    # return_value is the real Nanonis triple (error_string, raw_bytes, Variables);
    # Scan.XYPosGet ResponseTypes = ["f", "f"] → Variables == [X_m, Y_m].
    canned = {"Scan_XYPosGet": {"return_value": ["", b"", [10e-9, 20e-9]]}}
    tool = wrap_skill(GetScanXYPosition, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetScanXYPosition"]


def test_scan_background_paste_executes():
    # Scan.BackgroundPaste ResponseTypes = ["I"] → Variables == [Timed_out?].
    canned = {"Scan_BackgroundPaste": {"return_value": ["", b"", [0]]}}
    tool = wrap_skill(ScanBackgroundPaste, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["ScanBackgroundPaste"]


def test_scan_background_delete_executes():
    # Scan.BackgroundDelete ResponseTypes = ["I"] → Variables == [Timed_out?].
    canned = {"Scan_BackgroundDelete": {"return_value": ["", b"", [0]]}}
    tool = wrap_skill(ScanBackgroundDelete, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["ScanBackgroundDelete"]


def test_start_scan_error_propagates():
    canned = {
        "Scan_PropsGet": {"return_value": None},
        "Scan_PropsSet": {"return_value": None},
        "Scan_Action": {"error": "scan hardware fault"},
    }
    tool = wrap_skill(StartScan, make_provider(canned))
    result = _invoke(tool)
    assert result.update["messages"][0].status == "error"


def test_configure_scan_frame_set_error():
    canned = {"Scan_FrameSet": {"error": "piezo out of range"}}
    tool = wrap_skill(ConfigureScan, make_provider(canned))
    result = _invoke(tool, center_x_m=0.0, center_y_m=0.0, width_m=100e-9, height_m=100e-9)
    assert result.update["messages"][0].status == "error"


# ── Triple-parsing white-box tests (real Nanonis return triples) ───────────────
#
# nanonis_spm methods return [error_string, raw_bytes, Variables]. The actual
# decoded data lives in Variables (parsed[2][i]); parsed[0] is the (empty on
# success) error string and parsed[1] is the raw byte buffer. Reading parsed[0]
# instead of parsed[2][0] was the bug class fixed here: success=True but the
# returned scalar was always the empty error string → bool("")==False, so e.g.
# a real timeout was reported as "did not time out".


def _exec(skill_cls, canned, **params):
    """Call the raw skill.execute() against a FakeCtx so we can assert on the
    returned SkillResult.data (the tool wrapper hides .data)."""
    ctx = FakeCtx(canned=canned or {})
    return skill_cls().execute(ctx, params)


def test_get_scan_xy_position_reads_variables_not_error_string():
    # Scan.XYPosGet ResponseTypes = ["f", "f"] → Variables == [X_m, Y_m].
    canned = {"Scan_XYPosGet": {"return_value": ["", b"\x00\x00", [12.5e-9, -7.0e-9]]}}
    res = _exec(GetScanXYPosition, canned)
    assert res.success
    assert res.data["x_m"] == pytest.approx(12.5e-9)
    assert res.data["y_m"] == pytest.approx(-7.0e-9)


def test_scan_background_paste_reports_real_timeout():
    # Timed_out? == 1 → must surface as True. Reading parsed[0] (="") would have
    # made this always False.
    canned = {"Scan_BackgroundPaste": {"return_value": ["", b"", [1]]}}
    res = _exec(ScanBackgroundPaste, canned)
    assert res.success
    assert res.data["timed_out"] is True


def test_scan_background_paste_reports_no_timeout():
    canned = {"Scan_BackgroundPaste": {"return_value": ["", b"", [0]]}}
    res = _exec(ScanBackgroundPaste, canned)
    assert res.success
    assert res.data["timed_out"] is False


def test_scan_background_delete_reports_real_timeout():
    canned = {"Scan_BackgroundDelete": {"return_value": ["", b"", [1]]}}
    res = _exec(ScanBackgroundDelete, canned, delete_all=True)
    assert res.success
    assert res.data["timed_out"] is True
    assert res.data["delete_all"] is True


def test_scan_background_delete_reports_no_timeout():
    canned = {"Scan_BackgroundDelete": {"return_value": ["", b"", [0]]}}
    res = _exec(ScanBackgroundDelete, canned)
    assert res.success
    assert res.data["timed_out"] is False


def test_configure_scan_signals_namesget_triple():
    # Signals.NamesGet ResponseTypes = ["i", "i", "*+c"] →
    # Variables == [size, count, [name0, name1, ...]]. ConfigureScan must find
    # the nested names array and resolve channel indexes from it.
    names = ["Z (m)", "Current (A)", "Bias (V)"]
    canned = {
        "Scan_FrameSet": {"return_value": ["", b"", []]},
        # ConfigureScan 读回当前角度以「保持不变」；读不到就拒绝配置帧
        # （v6.1.3）。这些替身以前没有 Scan_FrameGet，于是一直悄悄走
        # 假定 0° 的那条路 —— 测试从未覆盖过这次读取。
        "Scan_FrameGet": {"return_value": ["", b"", [0.0, 0.0, 1e-7, 1e-7, 30.0]]},
        "Signals_NamesGet": {"return_value": ["", b"", [64, len(names), names]]},
        "Scan_BufferSet": {"return_value": ["", b"", []]},
        "Scan_SpeedSet": {"return_value": ["", b"", []]},
    }
    ctx = FakeCtx(canned=canned)
    res = ConfigureScan().execute(
        ctx,
        {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 100e-9,
         "height_m": 100e-9, "channels": "Z,Current"},
    )
    assert res.success
    buf_calls = [c for c in ctx.calls if c[0] == "Scan_BufferSet"]
    assert len(buf_calls) == 1
    # Z (idx 0) and Current (idx 1) resolved from the names array.
    assert buf_calls[0][1][0] == [0, 1]


# ── ConfigureScan speed-override behaviour ─────────────────────────────────────


def test_configure_scan_sets_speed_by_default():
    canned = {
        "Scan_FrameSet": {"return_value": ["", b"", []]},
        # ConfigureScan 读回当前角度以「保持不变」；读不到就拒绝配置帧
        # （v6.1.3）。这些替身以前没有 Scan_FrameGet，于是一直悄悄走
        # 假定 0° 的那条路 —— 测试从未覆盖过这次读取。
        "Scan_FrameGet": {"return_value": ["", b"", [0.0, 0.0, 1e-7, 1e-7, 30.0]]},
        "Signals_NamesGet": {"return_value": ["", b"", [0, 0, []]]},
        "Scan_SpeedSet": {"return_value": ["", b"", []]},
    }
    ctx = FakeCtx(canned=canned)
    res = ConfigureScan().execute(
        ctx,
        {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 100e-9, "height_m": 100e-9},
    )
    assert res.success
    speed_calls = [c for c in ctx.calls if c[0] == "Scan_SpeedSet"]
    assert len(speed_calls) == 1
    # 默认行时间由尺度档位解析，不能退回固定常量。
    from mast.core.scan_policy import get_tier_for_size
    tier_lt = float(get_tier_for_size(100e-9)["line_time_s"])
    fwd, bwd, fwd_t, bwd_t, keep_const, ratio = speed_calls[0][1]
    assert fwd_t == pytest.approx(tier_lt), "没查档位表，回到硬编码默认了"
    assert fwd == pytest.approx(100e-9 / tier_lt)
    assert bwd == pytest.approx(100e-9 / tier_lt)
    assert fwd < 1e-6, "1000 nm/s 扫 100 nm 的图 —— 这是刮针尖的那个速度"
    # keep time-per-line constant (=2), not "no change" (=0).
    assert keep_const == 2
    assert res.data["scan_speed_set"] is True
    assert res.data["linear_speed_m_s"] == pytest.approx(100e-9 / tier_lt)


def test_configure_scan_honours_explicit_line_time():
    canned = {
        "Scan_FrameSet": {"return_value": ["", b"", []]},
        # 替身回显最近设置的帧，供读回校验核对。
        "Scan_FrameGet": {"return_value": ["", b"", [0.0, 0.0, 200e-9, 200e-9, 30.0]]},
        "Signals_NamesGet": {"return_value": ["", b"", [0, 0, []]]},
        "Scan_SpeedSet": {"return_value": ["", b"", []]},
    }
    ctx = FakeCtx(canned=canned)
    res = ConfigureScan().execute(
        ctx,
        {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 200e-9,
         "height_m": 200e-9, "line_time_s": 0.5},
    )
    assert res.success
    speed_calls = [c for c in ctx.calls if c[0] == "Scan_SpeedSet"]
    fwd, bwd, fwd_t, bwd_t, keep_const, ratio = speed_calls[0][1]
    # 200e-9 / 0.5 = 4e-7 m/s
    assert fwd == pytest.approx(4e-7)
    assert fwd_t == pytest.approx(0.5)
    assert res.data["line_time_s"] == pytest.approx(0.5)


def test_configure_scan_can_skip_speed_override():
    """set_scan_speed=False must NOT touch the scan speed (no silent rewrite)."""
    canned = {
        "Scan_FrameSet": {"return_value": ["", b"", []]},
        # ConfigureScan 读回当前角度以「保持不变」；读不到就拒绝配置帧
        # （v6.1.3）。这些替身以前没有 Scan_FrameGet，于是一直悄悄走
        # 假定 0° 的那条路 —— 测试从未覆盖过这次读取。
        "Scan_FrameGet": {"return_value": ["", b"", [0.0, 0.0, 1e-7, 1e-7, 30.0]]},
        "Signals_NamesGet": {"return_value": ["", b"", [0, 0, []]]},
        "Scan_SpeedSet": {"return_value": ["", b"", []]},
    }
    ctx = FakeCtx(canned=canned)
    res = ConfigureScan().execute(
        ctx,
        {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 100e-9,
         "height_m": 100e-9, "set_scan_speed": False},
    )
    assert res.success
    speed_calls = [c for c in ctx.calls if c[0] == "Scan_SpeedSet"]
    assert speed_calls == []
    assert res.data["scan_speed_set"] is False
    assert res.data["line_time_s"] is None


def test_configure_scan_set_scan_speed_param_declared():
    """The speed side-effect must be a declared, controllable parameter."""
    tool = wrap_skill(ConfigureScan, make_provider())
    fields = tool.args_schema.model_fields
    assert "set_scan_speed" in fields
    assert "line_time_s" in fields
    assert not fields["set_scan_speed"].is_required()
    assert not fields["line_time_s"].is_required()


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


# 角度读不到时应保留未知状态，不能默认为零。

_CFG = {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 100e-9,
        "height_m": 100e-9, "set_scan_speed": False}


def _no_frameget_before_frameset(ctx):
    """FrameSet 之前有没有读过帧。

    2026-08-28：ConfigureScan 在 FrameSet **之后**加了一次读回校验
    （仪器把框夹到量程内时，上层原来完全看不出来）。所以「FrameGet 一次
    都不调」这个断言过强了 —— 它会把那道新闸门也一并禁掉。
    原意是「别靠读来决定写什么」，精确的写法是**读不能发生在写之前**。
    """
    for m, _ in ctx.calls:
        if m == "Scan_FrameSet":
            return True
        if m == "Scan_FrameGet":
            return False
    return True


def _cfg_ctx(frame_get):
    return FakeCtx(canned={
        "Scan_FrameSet": {"return_value": ["", b"", []]},
        "Scan_BufferSet": {"return_value": ["", b"", []]},
        "Signals_NamesGet": {"return_value": ["", b"", [64, 1, ["Z (m)"]]]},
        "Scan_FrameGet": frame_get,
    })


def test_angle_read_error_record_refuses_path_2():
    """设置失败时保留拒绝原因，不能声称已执行写入。"""
    ctx = _cfg_ctx({"error": "TCP timeout"})
    res = ConfigureScan().execute(ctx, dict(_CFG))
    assert not res.success
    assert "扫描角度" in (res.error or "")
    assert not [c for c in ctx.calls if c[0] == "Scan_FrameSet"], "编了个角度写进硬件"


def test_angle_unparseable_reply_refuses_path_3():
    """无声路径：回包不是序列 → vals=None → if 不成立 → 旧代码停在 0.0。"""
    ctx = _cfg_ctx({"return_value": None})
    res = ConfigureScan().execute(ctx, dict(_CFG))
    assert not res.success
    assert not [c for c in ctx.calls if c[0] == "Scan_FrameSet"]


def test_angle_short_reply_refuses_path_4():
    """无声路径：Variables 短于 5 → 取不到 vals[4] → 旧代码停在 0.0。"""
    ctx = _cfg_ctx({"return_value": ["", b"", [0.0, 0.0]]})
    res = ConfigureScan().execute(ctx, dict(_CFG))
    assert not res.success
    assert not [c for c in ctx.calls if c[0] == "Scan_FrameSet"]


def test_angle_non_finite_refuses():
    """NaN 是个合法 float，会一路被写进 Scan_FrameSet。"""
    ctx = _cfg_ctx({"return_value": ["", b"", [0.0, 0.0, 1e-7, 1e-7, float("nan")]]})
    res = ConfigureScan().execute(ctx, dict(_CFG))
    assert not res.success
    assert not [c for c in ctx.calls if c[0] == "Scan_FrameSet"]


def test_a_readable_angle_is_preserved_not_zeroed():
    """对照组，也是这条功能本身：读到 30° 就要写 30°，不是 0°。"""
    ctx = _cfg_ctx({"return_value": ["", b"", [0.0, 0.0, 1e-7, 1e-7, 30.0]]})
    res = ConfigureScan().execute(ctx, dict(_CFG))
    assert res.success
    fs = next(c for c in ctx.calls if c[0] == "Scan_FrameSet")
    assert fs[1][4] == 30.0, f"角度没被保持：{fs[1]}"


def test_an_explicit_zero_angle_is_still_allowed():
    """显式传入的零值必须保留。"""
    # 模拟帧回读并保留调用次序。
    ctx = _cfg_ctx({"return_value": ["", b"", [0.0, 0.0, 100e-9, 100e-9, 0.0]]})
    res = ConfigureScan().execute(ctx, dict(_CFG, angle_deg=0.0))
    assert res.success
    assert _no_frameget_before_frameset(ctx), "显式角度不该先去读角度"
    fs = next(c for c in ctx.calls if c[0] == "Scan_FrameSet")
    assert fs[1][4] == 0.0


# 一次启动对应一帧；显式关闭 continuous，避免继承界面的连续扫描状态。

from mast.skills.builtins.imaging import (  # noqa: E402
    _GET_ON,
    _SET_NO_CHANGE,
    _SET_OFF,
    _scan_props_continuous,
)


def _props_reply(continuous: int, series: str = "au111") -> list:
    """一条 Scan_PropsGet 回包,形状同解析侧所见:``(err, raw, values)``,
    ``values = [continuous, bouncy, autosave, series, comment, [modules...]]``。
    **GET** 编码:continuous 0=关、1=开。"""
    return [0, b"", [continuous, 1, 0, series, "", ["Bias", "Current"]]]


@dataclass
class SeqCtx(FakeCtx):
    """一台会记住自己被写过没有的替身。

    「写之前」和「写之后」能不一致,是回读检查唯一可能失败的方式 —— 两次都发同一个
    罐头的话,这条检查在两个世界里都绿。

    状态**按 Scan_PropsSet 翻面,不按第几次调用**翻。第一版是按调用次数发答案的,
    当场就被自己咬了:没有活动实验时 StartScan 会为了取序列名**再读一次**
    Scan_PropsGet,于是"第二次"落在了写入之前。照仪器建模就不会有这个问题 ——
    替身要模仿的是那台机器的状态,不是原先以为的调用顺序。
    """

    props_before: Any = None
    props_after: Any = None
    _written: bool = False

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        if method == "Scan_PropsGet" and self.props_before is not None:
            self.calls.append((method, args))
            reply = self.props_after if self._written else self.props_before
            return NanonisCallRecord(method=method, args=args, return_value=reply)
        if method == "Scan_PropsSet":
            self._written = True
        return super().safe_call(method, *args, role=role)


def _start_ctx(before=None, after=None) -> SeqCtx:
    return SeqCtx(
        canned={"Scan_PropsGet": {"return_value": None},
                "Scan_PropsSet": {"return_value": None},
                "Scan_Action": {"return_value": None}},
        props_before=before,
        props_after=after if after is not None else before,
    )


def test_start_scan_sets_continuous_off_it_does_not_leave_it_alone():
    """首位实参必须是 2(关),不能是 0。

    这条钉的不是「我们换了个数」,是**那个 0 会重新长回来** —— 它看起来就该是
    「关」,而且在 autosave 那一列 0 确实是「不改」、1 才是「All」,同一行里三个
    位置三套含义,回到 0 是最省事也最像对的改法。
    """
    ctx = _start_ctx(_props_reply(1), _props_reply(0))
    StartScan().execute(ctx, {})
    props_set = [c for c in ctx.calls if c[0] == "Scan_PropsSet"]
    assert len(props_set) == 1
    assert props_set[0][1][0] == _SET_OFF == 2, "continuous 必须写「关」(2),不是「不改」(0)"
    assert props_set[0][1][2] == 1, "autosave 仍是 All(SET 表里的 1)"


def test_start_scan_reports_a_refused_write_as_refused():
    """机器不认这次写入 ⇒ ``continuous_scan_still_on`` 为真。

    少了第二次读,「我们把它关了」和「它是关的」就是同一句话 —— 而这次散架的
    正是这两者之间。
    """
    ctx = _start_ctx(_props_reply(1), _props_reply(1))
    result = StartScan().execute(ctx, {})
    assert result.data["continuous_scan_before"] == 1
    assert result.data["continuous_scan_after"] == 1
    assert result.data["continuous_scan_still_on"] is True


def test_start_scan_accepted_write_is_not_reported_as_still_on():
    ctx = _start_ctx(_props_reply(1), _props_reply(0))
    result = StartScan().execute(ctx, {})
    assert result.data["continuous_scan_before"] == 1
    assert result.data["continuous_scan_after"] == 0
    assert result.data["continuous_scan_still_on"] is False


def test_start_scan_unreadable_flag_is_not_reported_as_off():
    """continuous 回读使用三态：开启、关闭或无法判断。"""
    ctx = _start_ctx()          # 罐头 Scan_PropsGet 回 None:解析不出标志
    result = StartScan().execute(ctx, {})
    assert result.data["continuous_scan_before"] is None
    assert result.data["continuous_scan_after"] is None
    assert result.data["continuous_scan_still_on"] is None, (
        "读不到被报成了「没开着」—— 故障折叠成了正常")


def test_start_scan_reports_false_only_when_it_actually_read_off():
    """读到了、而且是关的 —— 这时候才允许说 False。

    与上一条合起来钉住三态：None(读不到) / False(读到关) / True(读到开)。
    只测其中一个的话，「永远返回 None」也能过。
    """
    ctx = _start_ctx(_props_reply(0), _props_reply(0))
    result = StartScan().execute(ctx, {})
    assert result.data["continuous_scan_after"] == 0
    assert result.data["continuous_scan_still_on"] is False


def test_set_and_get_do_not_share_one_encoding_table():
    """钉住两张表不同 —— 把它们合并成一张是显然的「简化」,而且是错的。

    写「关」是 2,读「关」是 0,读「开」是 1。若拿 SET 表去比对回读值,一台开着
    连续扫描的机器(读回 1)会被判成「不是 2……嗯,那就是写进去了」—— 一条在两个
    世界里都通过的检查。
    """
    assert (_SET_OFF, _SET_NO_CHANGE, _GET_ON) == (2, 0, 1)
    assert _SET_OFF != _GET_ON
    assert _scan_props_continuous(_props_reply(1)) == _GET_ON
    assert _scan_props_continuous(_props_reply(0)) == 0
    assert _scan_props_continuous(None) is None


# continuous 读回失败时不得继续启动单帧扫描。

from mast.skills.builtins.imaging import (  # noqa: E402
    _GET_OFF,
    _continuous_state,
    _scan_props_module_count,
    _scan_props_series_name,
)


def _full_props_reply(continuous: int = 0, modules=("Bias", "Current"),
                      series: str = "au111", comment: str = "") -> list:
    """完整的 ScanProps 回包包含多个长度字段，不能误把字符串数组长度当成载荷。"""
    mods = [str(m) for m in modules]
    return ["", b"", [
        continuous, 0, 0,                                  # 0,1,2
        len(series), series,                               # 3,4
        len(comment), comment,                             # 5,6
        sum(4 + len(m) for m in mods), len(mods), mods,    # 7,8,9
        len(mods), [1] * len(mods),                        # 10,11
        len(mods), 1, [f"{m}>P (V)" for m in mods],        # 12,13,14  ← 参数表
        0,                                                 # 15
    ]]


def test_a_value_outside_the_get_table_is_not_an_answer():
    """GET 回读只接受定义的状态值，不沿用 SET 的占位语义。"""
    assert _continuous_state(_GET_OFF) is False
    assert _continuous_state(_GET_ON) is True
    assert _continuous_state(None) is None
    assert _continuous_state(_SET_OFF) is None, "SET 表的 2 被当成了 GET 表的答案"
    assert _continuous_state(7) is None


def test_the_module_list_is_read_by_position_not_by_shape():
    """真回包里有**两个**全字符串数组:模块名(第 9 位)和参数表(第 14 位)。

    「在回包里找第一个全字符串的 list」这条启发式,在模块名非空时**碰巧**对 ——
    它靠的是模块名排在参数表前面。模块名为空时它就不再碰巧了:它会把**参数名**
    交给 ``Scan_PropsSet`` 当模块名写回去。那正好落在这个文件反复钉的那件事上
    ——「MAST 发明出来的模块名」有两种下场,都很坏。

    所以这条构造的正是那个分歧点:声明 0 个模块、参数表非空。
    按位置读 ⇒ ``[]``;按形状找 ⇒ ``["Bias>P (V)"]``。
    """
    reply = _full_props_reply(modules=("Bias",))
    reply[2][8] = 0        # 声明:0 个模块
    reply[2][9] = []       # 数组:空
    # 第 14 位的参数表照旧非空 —— 启发式会一路走到它。
    assert reply[2][14] == ["Bias>P (V)"]

    from mast.skills.builtins.imaging import _scan_props_modules as _by_shape
    assert _by_shape(reply) == ["Bias>P (V)"], "前提没成立:启发式并没有走到参数表"
    assert _scan_props_module_count(reply) == 0

    ctx = _start_ctx(reply, reply)
    res = StartScan().execute(ctx, {})
    written = list([c for c in ctx.calls if c[0] == "Scan_PropsSet"][0][1][5])
    assert written == [], f"把参数名当模块名写回去了:{written}"
    assert res.data["module_names_count_declared"] == 0


def test_a_normal_reply_round_trips_the_operators_list_verbatim():
    """完整 16 字段回包上的对照组:读到几个就原样写回几个,顺序不变。"""
    reply = _full_props_reply(
        modules=("Bias", "Z-Controller", "Piezo Configuration"))
    assert _scan_props_module_count(reply) == 3
    ctx = _start_ctx(reply, reply)
    res = StartScan().execute(ctx, {})
    written = list([c for c in ctx.calls if c[0] == "Scan_PropsSet"][0][1][5])
    assert written == ["Bias", "Z-Controller", "Piezo Configuration"], written
    assert res.data["module_names_count_declared"] == 3
    assert res.data["module_names_source"] == "read"


def test_a_declared_count_that_disagrees_with_the_array_is_not_trusted():
    """两个字段对不上 = 这份回包错位了。**判成不知道,不去修它。**

    一次错位的解析没有理由让「声明的个数」和「解出来的数组」保持一致,所以这个
    交叉检查是便宜且真实的。修它(比如截断到较短的那个)会把一份坏数据变成一份
    看起来正常的坏数据 —— 然后原样写回仪器。
    """
    reply = _full_props_reply(modules=("Bias", "Current"))
    reply[2][8] = 5                      # 声明 5 个,数组里只有 2 个
    assert _scan_props_module_count(reply) is None


def test_zero_modules_is_a_fact_not_a_failure():
    """**「用户一个模块都没选」和「问不出来」不是同一件事。**

    以前只有两条:非空 ⇒ read,其余 ⇒ unchanged —— 于是前者被并进了后者,而它
    恰恰是**唯一**一种不必读清单也能安全下发 ``Scan_PropsSet`` 的情形:空数组
    无论按「清空」还是按「不改」解释,作用在一份本来就空的清单上都是空操作。
    并进去的代价:一台「零模块」的机器永远关不掉 continuous。

    要推翻这条需要什么:证明 Nanonis 对 ``Modules names number = 0`` 还有第三种
    反应(既不是清空也不是不改)。手册对这个字段没有定义任何哨兵值。
    """
    reply = _full_props_reply(modules=())
    assert _scan_props_module_count(reply) == 0
    ctx = _start_ctx(reply, reply)
    res = StartScan().execute(ctx, {})
    sets = [c for c in ctx.calls if c[0] == "Scan_PropsSet"]
    assert len(sets) == 1, "确证零个模块时仍然不敢下发 —— 那是把「零」当成了「不知道」"
    assert list(sets[0][1][5]) == []
    assert res.data["module_names_source"] == "read_empty"
    assert res.data["scan_props_written"] is True
    assert res.success


def test_an_empty_array_is_only_written_when_the_rig_declared_zero():
    """反向对照,而且是这一整段的安全边界。

    读不到清单 ⇒ 数组也是空的 ⇒ 如果这里按「空就写空」办,就正好把用户手配的
    11 个模块清空 —— 也就是 2026-08-04 那个顾虑本身。区分只能来自仪器**自己声明
    的个数**,不能来自数组的长度。
    """
    ctx = _start_ctx()                    # 回包读不懂:count 无从谈起
    res = StartScan().execute(ctx, {})
    assert not [c for c in ctx.calls if c[0] == "Scan_PropsSet"], "把空表写回去了"
    assert res.data["module_names_source"] == "unchanged"
    assert res.data["module_names_count_declared"] is None


# ── 闸门:不发起一次停不下来的扫描 ────────────────────────────────────────────

def test_a_scan_that_will_not_stop_is_not_started():
    """回读确认 continuous 仍开启时，不得发 Scan_Action。"""
    ctx = _start_ctx(_props_reply(1), _props_reply(1))    # 写了也没关掉
    res = StartScan().execute(ctx, {})
    assert not res.success
    assert not [c for c in ctx.calls if c[0] == "Scan_Action"]
    assert res.data["continuous_scan_still_on"] is True
    assert res.data["scan_running"] is False


def test_the_refusal_does_not_claim_a_write_that_never_happened():
    """拒绝消息必须区分写入被拒绝与根本没有写入。"""
    partial = [0, b"", [1, 1, 0, "au111", ""]]      # 有标志,没有模块名数组
    ctx = _start_ctx(partial, partial)
    res = StartScan().execute(ctx, {})
    assert not res.success
    assert res.data["continuous_scan_still_on"] is True
    assert res.data["scan_props_written"] is False
    assert "根本没能写" in res.error, res.error
    assert "回读仍然是" not in res.error, "报了一次没发生过的写入:\n" + res.error


def test_a_confirmed_off_scan_is_started_normally():
    """对照组,也是这条闸门唯一的放行条件:**回读说「关」**。

    少了这一条,「永远拒绝」也能让上面那条变绿。
    """
    ctx = _start_ctx(_props_reply(1), _props_reply(0))
    res = StartScan().execute(ctx, {})
    assert res.success
    assert [c for c in ctx.calls if c[0] == "Scan_Action"]
    assert res.data["continuous_scan_still_on"] is False
    assert res.data["continuous_scan_override_used"] is False


def test_the_gate_does_not_infer_off_from_before_plus_write():
    """**不从「之前是关的 + 我写了关」推出「现在是关的」。**

    那是绕着缺失的证据讲道理,而证据是可以去拿的(见下一条的重试)。这条钉住
    闸门只有一个子句:回读说关才放行。多一个子句就多一个在两个世界里都通过的
    分支 —— 本文件上方 ``test_start_scan_reports_a_refused_write_as_refused``
    记的就是那种检查。
    """
    unreadable = ["", b"", 42]                  # 回包到了,但 body 读不懂
    ctx = _start_ctx(_props_reply(0), unreadable)
    res = StartScan().execute(ctx, {})
    assert not res.success
    assert res.data["continuous_scan_before"] == 0
    assert res.data["continuous_scan_still_on"] is None


# ── 重试:先去拿证据,再谈拒绝 ────────────────────────────────────────────────

@dataclass
class FlakyPropsCtx(FakeCtx):
    """前 ``fail_first`` 次 ``Scan_PropsGet`` 失败,之后正常。"""

    fail_first: int = 1
    reply: Any = None
    _seen: int = 0

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        if method == "Scan_PropsGet":
            self.calls.append((method, args))
            self._seen += 1
            if self._seen <= self.fail_first:
                return NanonisCallRecord(method=method, args=args,
                                         error="TCP timeout")
            return NanonisCallRecord(method=method, args=args,
                                     return_value=self.reply)
        return super().safe_call(method, *args, role=role)


def test_a_transient_read_failure_is_retried_not_converted_into_a_refusal():
    """只读失败允许有限重试；确定性的解析错误不能造成循环。"""
    ctx = FlakyPropsCtx(
        canned={"Scan_PropsSet": {"return_value": None},
                "Scan_Action": {"return_value": None}},
        fail_first=1, reply=_props_reply(0))
    res = StartScan().execute(ctx, {})
    assert res.success, res.error
    assert [c for c in ctx.calls if c[0] == "Scan_Action"]


def test_the_retry_is_bounded():
    """一台一直读不出来的机器不会被无限重问 —— 每次读最多两发。"""
    ctx = FlakyPropsCtx(
        canned={"Scan_PropsSet": {"return_value": None},
                "Scan_Action": {"return_value": None}},
        fail_first=99, reply=None)
    res = StartScan().execute(ctx, {})
    assert not res.success
    gets = [c for c in ctx.calls if c[0] == "Scan_PropsGet"]
    # 读清单 2 发 + 回读 2 发。多一发都说明重试上限漏了。
    assert len(gets) == 4, gets


# ── 序列名:一次读,一个答案 ──────────────────────────────────────────────────

def test_the_series_name_comes_from_the_reply_we_already_read():
    """原来取序列名要**再发一次** ``Scan_PropsGet``。两发可以给出不同答案,而分歧
    不是无害的:取清单那发成功(于是这一整发会下出去)、取名字那发失败时,
    basename 变成空串,而空的 Series name 会把用户配的文件名前缀打回
    ``unnamed####``。

    一次读、一个答案 —— 顺便少一个 TCP 往返。
    """
    reply = _full_props_reply(series="operator_prefix_")
    assert _scan_props_series_name(reply) == "operator_prefix_"
    ctx = _start_ctx(reply, reply)
    StartScan().execute(ctx, {})
    written = [c for c in ctx.calls if c[0] == "Scan_PropsSet"][0][1]
    assert written[3] == "operator_prefix_", f"序列名被改成了 {written[3]!r}"
    # 读清单 1 发 + 回读 1 发。第三发就是那个已经被删掉的「再读一次取名字」。
    assert len([c for c in ctx.calls if c[0] == "Scan_PropsGet"]) == 2


def test_configure_scan_line_time_comes_from_the_tier_table_per_size():
    """不同尺寸必须落到不同的档 —— 一个常数默认在这里是查不出来的。

    只测一个尺寸的话，「查表」和「碰巧等于那个常数」区分不开。8 nm（atomic）
    与 100 nm（highres）在出厂表里线时间不同，两个一起测才能证明它真的在查。
    """
    from mast.core.scan_policy import get_tier_for_size

    seen = {}
    for size in (8e-9, 100e-9):
        canned = {
            "Scan_FrameSet": {"return_value": ["", b"", []]},
            "Scan_FrameGet": {"return_value": ["", b"", [0.0, 0.0, size, size, 0.0]]},
            "Signals_NamesGet": {"return_value": ["", b"", [0, 0, []]]},
            "Scan_SpeedSet": {"return_value": ["", b"", []]},
        }
        ctx = FakeCtx(canned=canned)
        res = ConfigureScan().execute(ctx, {
            "center_x_m": 0.0, "center_y_m": 0.0,
            "width_m": size, "height_m": size,
        })
        assert res.success
        call = [c for c in ctx.calls if c[0] == "Scan_SpeedSet"][0]
        seen[size] = call[1][2]          # fwd_line_time
        assert seen[size] == pytest.approx(float(get_tier_for_size(size)["line_time_s"]))
    assert seen[8e-9] != seen[100e-9], "两个尺寸拿到同一个线时间 —— 那不是查表"


def test_configure_scan_reports_where_the_line_time_came_from():
    """「这个速度是谁定的」必须可查：显式 / 档位表 / 兜底。

    一个光秃秃的 1.2 决定不了它来自档位表还是来自某人手打，而这两者在事故
    复盘时是完全不同的两件事。
    """
    canned = {
        "Scan_FrameSet": {"return_value": ["", b"", []]},
        "Scan_FrameGet": {"return_value": ["", b"", [0.0, 0.0, 8e-9, 8e-9, 0.0]]},
        "Signals_NamesGet": {"return_value": ["", b"", [0, 0, []]]},
        "Scan_SpeedSet": {"return_value": ["", b"", []]},
    }
    res = ConfigureScan().execute(FakeCtx(canned=canned), {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 8e-9, "height_m": 8e-9})
    assert res.data.get("line_time_source", "").startswith("tier:")

    res2 = ConfigureScan().execute(FakeCtx(canned=canned), {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 8e-9, "height_m": 8e-9,
        "line_time_s": 2.5})
    assert res2.data.get("line_time_source") == "explicit"
    assert res2.data.get("line_time_s") == pytest.approx(2.5)
