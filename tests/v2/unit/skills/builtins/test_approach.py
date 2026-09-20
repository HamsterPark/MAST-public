"""v2 unit tests for mast.skills.builtins.approach.

Skills covered: AutoApproach (DANGEROUS), WithdrawTip (CONFIRM),
               GetAutoApproachStatus (AUTO) — 3 skills total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_approach.py -x -v
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
from mast.skills.builtins.approach import (
    AutoApproach,
    GetAutoApproachStatus,
    WithdrawTip,
)


# ── FakeCtx ──────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    # If set, AutoApproach_OnOffGet reports running=True for the first N polls
    # then running=False (module reached the setpoint). Lets the composite's
    # polling wait phase complete deterministically.
    oog_running_polls: int | None = None
    _oog_count: int = 0

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method == "AutoApproach_OnOffGet" and self.oog_running_polls is not None:
            self._oog_count += 1
            still = self._oog_count <= self.oog_running_polls
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [1 if still else 0]))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def make_provider(canned: dict[str, Any] | None = None,
                  oog_running_polls: int | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned, oog_running_polls=oog_running_polls)


@pytest.fixture(autouse=True)
def _fast_approach_wait(monkeypatch):
    """Keep the AutoApproach wait-phase poll loop fast in unit tests."""
    monkeypatch.setattr(AutoApproach, "_poll_interval_s", 0.01, raising=False)
    monkeypatch.setattr(AutoApproach, "_grace_s", 0.05, raising=False)
    monkeypatch.setattr(AutoApproach, "_DEFAULT_WAIT_TIMEOUT_S", 5.0, raising=False)


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_auto_approach_shape():
    tool = wrap_skill(AutoApproach, make_provider())
    assert tool.name == "AutoApproach"
    # v0.3.22: downgraded from DANGEROUS to AUTO — Nanonis 内置硬件安全足够，
    # 软件 confirm 框反而卡死了进针流程。
    assert tool.metadata["danger_level"] == "AUTO"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_withdraw_tip_shape():
    tool = wrap_skill(WithdrawTip, make_provider())
    assert tool.name == "WithdrawTip"
    assert tool.metadata["danger_level"] == "CONFIRM"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_get_auto_approach_status_shape():
    tool = wrap_skill(GetAutoApproachStatus, make_provider())
    assert tool.name == "GetAutoApproachStatus"
    assert tool.metadata["danger_level"] == "AUTO"


def test_skill_source_points_to_approach_module():
    tool = wrap_skill(AutoApproach, make_provider())
    assert tool.metadata["skill_source"].endswith(".approach")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_auto_approach_executes():
    canned = {
        "AutoApproach_Open": {"return_value": ("", b"", [])},
        "AutoApproach_OnOffSet": {"return_value": ("", b"", [])},
    }
    # oog_running_polls=1 → module runs one poll then reaches the setpoint, so
    # the polling wait phase completes with success.
    tool = wrap_skill(AutoApproach, make_provider(canned, oog_running_polls=1))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["AutoApproach"]


def test_auto_approach_calls_both_methods():
    canned = {
        "AutoApproach_Open": {"return_value": ("", b"", [])},
        "AutoApproach_OnOffSet": {"return_value": ("", b"", [])},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned, oog_running_polls=1)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(AutoApproach, capturing_provider)
    _invoke(tool)
    last_ctx = instances[-1]
    methods = [c[0] for c in last_ctx.calls]
    assert "AutoApproach_Open" in methods
    assert "AutoApproach_OnOffSet" in methods


def test_withdraw_tip_executes():
    canned = {"ZCtrl_Withdraw": {"return_value": ("", b"", [])}}
    tool = wrap_skill(WithdrawTip, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["WithdrawTip"]


def test_get_auto_approach_status_executes():
    canned = {"AutoApproach_OnOffGet": {"return_value": [1]}}
    tool = wrap_skill(GetAutoApproachStatus, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["GetAutoApproachStatus"]


# ── Triplet-fixture regression tests (real Nanonis return shape) ───────────────
#
# AutoApproach.OnOffGet ResponseTypes=["H"] -> return_value is the
# (error_string, raw_bytes, parsed_list) triplet with Status at parsed[2][0].
# The old code read parsed[0] (the empty error string) so it ALWAYS reported
# "not running" on real hardware. These assert the parsed status, not just that
# the skill ran.

def _run_status(canned_rv):
    """Execute GetAutoApproachStatus directly against a FakeCtx, return data."""
    ctx = FakeCtx(canned={"AutoApproach_OnOffGet": {"return_value": canned_rv}})
    res = GetAutoApproachStatus().execute(ctx, {})
    assert res.success
    return res.data


def test_get_auto_approach_status_running_real_triplet():
    # Status=1 (running) lives at parsed[2][0].
    assert _run_status(("", b"\x00\x01", [1]))["running"] is True


def test_get_auto_approach_status_off_real_triplet():
    # Status=0 (off) at parsed[2][0]; previously parsed[0]="" also yielded
    # False, so this case never exposed the bug — the running case did.
    assert _run_status(("", b"\x00\x00", [0]))["running"] is False


def test_get_auto_approach_status_does_not_read_error_string():
    # Even with a non-empty (but here harmless) header the running flag must
    # come from parsed[2][0], not from truthiness of parsed[0].
    assert _run_status(("", b"", [1]))["running"] is True
    assert _run_status(("", b"", [0]))["running"] is False


def test_auto_approach_wait_phase_parses_running_from_triplet():
    # The composite's _phase_wait_complete / _phase_verify_status must read the
    # Status from parsed[2][0]. The module reports running for 2 polls (Status=1
    # triplets) then stops (Status=0) — the polling wait phase must SEE the
    # running Status (else it would hit the "did not start" path), CONFIRM the
    # tunnelling current (stopped ≠ reached setpoint since 2026-07-10 #42),
    # and complete, reporting final running=False.
    canned = {
        "AutoApproach_Open": {"return_value": ("", b"", [])},
        "AutoApproach_OnOffSet": {"return_value": ("", b"", [])},
        # tunnelling confirmed: |I| = setpoint = 0.5 nA
        "Current_Get": {"return_value": ("", b"", [5e-10])},
        "ZCtrl_SetpntGet": {"return_value": ("", b"", [5e-10])},
    }
    tool = wrap_skill(AutoApproach, make_provider(canned, oog_running_polls=2))
    result = _invoke(tool)
    content = result.update["messages"][0].content
    # Reached the setpoint → final running False; and it completed (did not fail
    # with a "did not start" error, which proves the running Status was parsed).
    assert "'running': False" in content
    assert "did not start" not in content


def test_auto_approach_stop_without_current_fails():
    """2026-07-10 the module stopping is NOT proof of engagement —
    it also stops when the coarse range is exhausted (Z at limit). With the
    current stuck at noise level (0.17 pA vs 500 pA setpoint, the exact field
    values) the wait phase must FAIL loudly instead of declaring 进针成功."""
    canned = {
        "AutoApproach_Open": {"return_value": ("", b"", [])},
        "AutoApproach_OnOffSet": {"return_value": ("", b"", [])},
        "Current_Get": {"return_value": ("", b"", [-1.7e-13])},
        "ZCtrl_SetpntGet": {"return_value": ("", b"", [5e-10])},
    }
    tool = wrap_skill(AutoApproach, make_provider(canned, oog_running_polls=2))
    result = _invoke(tool)
    content = result.update["messages"][0].content
    # Wording changed 2026-08-05 (settle judgement). The CONTRACT is unchanged
    # and is what is pinned: a stopped module with the current at noise fails,
    # and the failure quotes what it measured.
    assert "稳定地" in content and "未进入隧穿" in content
    assert "0.17 pA" in content and "500.00 pA" in content
    assert "failed" in content.lower()


def test_auto_approach_noise_floor_blocks_lowered_setpoint():
    """2026-07-10 '调低电流假装进到针了': with the setpoint lowered
    to 0.2 pA, amplifier noise (0.17 pA ≈ 85% of it) would pass a relative-only
    check. The absolute 1 pA noise floor must still reject it."""
    canned = {
        "AutoApproach_Open": {"return_value": ("", b"", [])},
        "AutoApproach_OnOffSet": {"return_value": ("", b"", [])},
        "Current_Get": {"return_value": ("", b"", [-1.7e-13])},
        "ZCtrl_SetpntGet": {"return_value": ("", b"", [2e-13])},  # gamed setpoint
    }
    tool = wrap_skill(AutoApproach, make_provider(canned, oog_running_polls=2))
    result = _invoke(tool)
    content = result.update["messages"][0].content
    assert "稳定地" in content and "未进入隧穿" in content
    # The bar printed is the NOISE FLOOR, not 50% of the gamed setpoint — that
    # is the whole point of #75 and it is now visible in the message.
    assert "1.00 pA" in content


# ── 「状态读不懂」不许折成「模块没在跑」（普查 A4，2026-08-15）──────────────
#
# `_parse_running` 原来在每条不认识的路上都 `return False`。最关键的一条是
# `("", b"", [])` —— 一个 **parsed 列表为空**的回包，也就是**上游解析失败**的
# 样子（`nanonis_spm` 的 `+*c` parser 每次 pip 后都要 patch，见记忆）。
# `bool([])` = False = 「模块没在跑」。
#
# 这个位置有前科：`GetAutoApproachStatus` 上面那段注释记着它**已经在真机上
# 静默报过一次「未在运行」**（当时是读错了下标，读的是 parsed[0]）。下标早修
# 好了，但「读不懂 ⇒ 没在跑」这条一直在。

from mast.skills.builtins.approach import WaitProgress, _parse_running  # noqa: E402


@pytest.mark.parametrize("rv,why", [
    pytest.param(("", b"", []), "parsed 列表是空的 = 上游没解出来", id="空-parsed"),
    pytest.param(None, "根本没有 return_value", id="None"),
    pytest.param("garbage", "不是序列", id="字符串"),
    pytest.param((), "空元组", id="空元组"),
    pytest.param(("", b"", ["x"]), "状态位不是数", id="状态位不是数"),
])
def test_unreadable_status_is_none_not_false(rv, why):
    assert _parse_running(rv) is None, why


@pytest.mark.parametrize("rv,expected", [
    (("", b"", [1]), True),
    (("", b"", [0]), False),
    (("", b"", [(1,)]), True),   # 数值数组字段是一串 1-元组（真机形状）
    (("", b"", [(0,)]), False),
    (1, True),
    (0, False),
])
def test_a_readable_status_still_answers_plainly(rv, expected):
    """反向对照：**该答的时候要答。**

    没有这一条，上面那组可以被一个「永远回 None」的实现满足 —— 那会把每一次
    正常的状态查询都变成一次失败，比原来的缺陷更吵也更没用。

    ## 这条测试是怎么找出第二个 bug 的（留成记录）

    上面那两行 1-元组参数，是我写这条反向对照时**顺手列进去、期待它绿**的 ——
    结果 `("", b"", [(0,)])` 红了，回的是 `True`。因为 `bool((0,))` 为真：
    一个「已停止」的回包被读成「还在跑」，**方向和这一轮要修的 A4 正好相反**。

    我不是看出来的，是这条反向对照撞出来的。反向对照的常规作用是「证明我没
    把闸门修成永远拒绝」，它多数时候修前修后都绿、看着像白写；这一次它自己
    找出了一个谁都没在找的缺陷。**下次仍然要写。**

    （那条修复本身是防御性的、未在真机观测过 —— 理由与克制都写在
    `_parse_running` 的 docstring 里，不在这儿重复。）
    """
    assert _parse_running(rv) is expected


def test_get_status_refuses_instead_of_reporting_not_running():
    """agent 直接可调的那个口：读不懂就别说「没在跑」。"""
    ctx = FakeCtx(canned={"AutoApproach_OnOffGet": {"return_value": ("", b"", [])}})
    res = GetAutoApproachStatus().execute(ctx, {})
    assert res.success is False
    assert res.data.get("running") is None, "不许在失败时还塞一个 running=False"


def test_get_status_still_reports_an_honest_zero():
    """反向对照：一个**真的**读到的 0 必须仍然说得出口。"""
    ctx = FakeCtx(canned={"AutoApproach_OnOffGet": {"return_value": ("", b"", [0])}})
    res = GetAutoApproachStatus().execute(ctx, {})
    assert res.success is True
    assert res.data["running"] is False


# ── 等待循环：保守语义不变，但 None 与真的 0 要分得开 ──────────────────────
#
# `_confirm_stopped` 读不到时回 True 是**写下过理由的决定**（approach.py
# 「不推翻已经读到的那个 0」，否则一条坏链路会让等待循环永远结束不了）。
# 那条不动。要改的是：它现在分不出「复读读到 0」和「复读读不懂」，而
# `status_flap_n` 这个唯一的可见计数只统计前者。

def _confirm(rv):
    skill = AutoApproach()
    skill._call_log = []
    prog = WaitProgress()
    ctx = FakeCtx(canned={"AutoApproach_OnOffGet": {"return_value": rv}})
    return skill._confirm_stopped(ctx, prog), prog


def test_an_unreadable_recheck_is_counted_not_silently_believed():
    stopped, prog = _confirm(("", b"", []))
    assert stopped is True, "保守语义不变：不推翻已经读到的那个 0"
    assert prog.status_unreadable_n == 1, "但它必须留下痕迹"
    assert prog.status_flap_n == 0


def test_a_real_zero_recheck_is_not_counted_as_unreadable():
    stopped, prog = _confirm(("", b"", [0]))
    assert stopped is True
    assert prog.status_unreadable_n == 0


def test_a_recheck_that_says_still_running_still_flaps():
    stopped, prog = _confirm(("", b"", [1]))
    assert stopped is False
    assert prog.status_flap_n == 1
    assert prog.status_unreadable_n == 0


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
