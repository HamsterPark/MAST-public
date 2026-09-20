"""Z 软限位读写均须暴露启用状态。

限值回读正确不代表保护已启用；禁用和读不到状态时分别提示，
并遵守调用方明确的 enable 参数。夹具使用独立构造的范围与控制器参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from mast.skills.builtins.instrument_limits import SetZLimits
from mast.skills.builtins.readback import GetZControllerState, _as_bool


@dataclass
class _Rec:
    method: str = ""
    args: tuple = ()
    return_value: Any = None
    error: str = ""


class _Ctx:
    """safe_call 按 verb 查表；未列出的 verb 报错（= 该键读不到）。"""

    def __init__(self, table: dict[str, Any], errors: tuple[str, ...] = ()):
        self.table = table
        self.errors = set(errors)
        self.calls: list[str] = []

    def safe_call(self, method: str, *args, **kw) -> _Rec:
        self.calls.append(method)
        if method in self.errors:
            return _Rec(method=method, args=args, error="TCP timeout")
        return _Rec(method=method, args=args,
                    return_value=("", b"", list(self.table.get(method, [0]))))


def _rig_table() -> dict[str, list]:
    return {
        "ZCtrl_OnOffGet": [0],
        "ZCtrl_StatusGet": [1],
        "ZCtrl_SetpntGet": [1e-10],
        "ZCtrl_GainGet": [2e-12, 2.0, 1e-12],
        "ZCtrl_ZPosGet": [4e-7],
        "ZCtrl_LimitsGet": [4e-7, -4e-7],
        "ZCtrl_LimitsEnabledGet": [0],          # ← 关着
        "ZCtrl_TipLiftGet": [0.0],
        "ZCtrl_WithdrawRateGet": [1e-7],
        "ZCtrl_SwitchOffDelayGet": [0.0],
        "ZCtrl_HomePropsGet": [0, 0.0],
    }


# ── 读侧 ────────────────────────────────────────────────────────────────


def test_disabled_limits_produce_a_warning_the_agent_can_see():
    ctx = _Ctx(_rig_table())
    res = GetZControllerState().execute(ctx, {})

    assert res.data["z_limits_enabled"] is False
    warning = res.data.get("z_limits_warning", "")
    assert "未启用" in warning
    assert "SetZLimits" in warning, "告警要告诉 agent 怎么修，而不只是抱怨"


def test_enabled_limits_produce_no_warning():
    table = _rig_table()
    table["ZCtrl_LimitsEnabledGet"] = [1]
    res = GetZControllerState().execute(_Ctx(table), {})

    assert res.data["z_limits_enabled"] is True
    assert "z_limits_warning" not in res.data


def test_unreadable_flag_is_not_reported_as_disabled():
    """「问不到」和「关着」不是一回事。

    把前者当后者会天天喊狼来了；把后者当前者会漏掉真正该说的那次。
    """
    ctx = _Ctx(_rig_table(), errors=("ZCtrl_LimitsEnabledGet",))
    res = GetZControllerState().execute(ctx, {})

    assert res.data["z_limits_enabled"] is None
    assert "z_limits_warning" not in res.data
    assert "z_limits_enabled" in res.data["_unreadable"]


@pytest.mark.parametrize("raw,expected", [
    (0, False), (1, True), ([0], False), ([1], True),
    (("", b"\x00", [0]), False), (("", b"\x01", [1]), True),
    (None, None), ("", None), ([], None),
])
def test_as_bool_covers_every_wire_shape(raw, expected):
    assert _as_bool(raw) is expected


# ── 写侧 ────────────────────────────────────────────────────────────────


class _WriteCtx(_Ctx):
    """SetZLimits 用：LimitsSet 之后 LimitsGet 回读到新值。"""

    def __init__(self, enabled_after: int):
        super().__init__({
            "ZCtrl_LimitsGet": [4e-7, -4e-7],
            "ZCtrl_LimitsEnabledGet": [enabled_after],
            "ZCtrl_LimitsSet": [0],
            "ZCtrl_LimitsEnabledSet": [0],
        })
        self.written: list[tuple] = []

    def safe_call(self, method: str, *args, **kw) -> _Rec:
        if method == "ZCtrl_LimitsSet":
            self.written.append(args)
            self.table["ZCtrl_LimitsGet"] = [args[0], args[1]]
        return super().safe_call(method, *args, **kw)


def test_writing_limits_without_enabling_says_they_are_inert():
    """以前这里返回一句欢快的成功，而那次写入什么也没做。"""
    ctx = _WriteCtx(enabled_after=0)
    res = SetZLimits().execute(
        ctx, {"z_high_limit_m": 2e-7, "z_low_limit_m": -2e-7, "enable": False})

    assert res.success
    assert res.data["enabled"] is False
    assert "未启用" in res.summary
    assert "enable=true" in res.summary
    # 明确不自动启用：调用方写了 enable=false。
    assert "ZCtrl_LimitsEnabledSet" not in ctx.calls


def test_enabling_path_does_not_grow_a_redundant_readback():
    ctx = _WriteCtx(enabled_after=1)
    res = SetZLimits().execute(
        ctx, {"z_high_limit_m": 2e-7, "z_low_limit_m": -2e-7, "enable": True})

    assert res.success
    assert res.data["enabled"] is True
    assert "未启用" not in res.summary
    assert "ZCtrl_LimitsEnabledSet" in ctx.calls
    assert "ZCtrl_LimitsEnabledGet" not in ctx.calls


def test_unreadable_enable_state_is_admitted_not_assumed():
    ctx = _WriteCtx(enabled_after=0)
    ctx.errors.add("ZCtrl_LimitsEnabledGet")
    res = SetZLimits().execute(
        ctx, {"z_high_limit_m": 2e-7, "z_low_limit_m": -2e-7, "enable": False})

    assert res.data["enabled"] is None
    assert "未能读回启用状态" in res.summary
