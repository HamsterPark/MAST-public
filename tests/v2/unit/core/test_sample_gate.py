"""样品门控：产数据的操作要样品，安全操作永远放行。

设计文档：docs/v2/design/experiment_folder_persistence.md §11

**豁免清单是本文件的重点。** 一个因为"没选样品"而按不动的急停按钮，比一堆
没归属的 .sxm 危险得多；所以这里的每一条豁免断言都是安全断言，不是洁癖。
"""

from __future__ import annotations

import pytest

from mast.core.sample_gate import (
    check_sample_scope,
    requires_sample,
    sample_gate_message,
)


class _Meta:
    """最小 duck-type 的 skill metadata。"""

    def __init__(self, name="X", tags=(), category="write", capabilities=()):
        self.name = name
        self.tags = tuple(tags)
        self.category = category
        self.capabilities = tuple(capabilities)


class _Log:
    def __init__(self, eid=None, sid=None):
        self.current_experiment_id = eid
        self.current_sample_id = sid


# ── 必须放行（安全 / 只读） ───────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "StopScan", "StopSTS", "StopMotor", "StopAutoApproach", "StopFolMe",
    "SafeRetract", "EmergencyRetract", "WithdrawTip",
])
def test_remedies_are_never_gated(name):
    """补救动作在任何情况下都要能跑 —— 它们存在的意义就是出事时立刻可用。"""
    assert requires_sample(_Meta(name=name)) is False


@pytest.mark.parametrize("tag", ["retract", "emergency", "withdraw", "safety", "stop"])
def test_safety_tags_are_never_gated(tag):
    assert requires_sample(_Meta(name="Whatever", tags=(tag,))) is False


@pytest.mark.parametrize("category", ["read", "analysis", "READ", "ANALYSIS"])
def test_reads_are_never_gated(category):
    """所有 Get*/Assess* 永远放行：只读操作不产生需要归属的数据。"""
    assert requires_sample(_Meta(name="GetCurrent", category=category)) is False


def test_scan_tagged_read_still_passes():
    """一个既带 scan 又是 READ 的技能仍然放行 —— READ 判定在产数据判定之前。"""
    assert requires_sample(
        _Meta(name="GetScanStatus", tags=("scan",), category="read")) is False


def test_unclassifiable_skill_is_allowed():
    """★ fail-open：分类不出来就放行。

    与 instrument_lock.needs_token 的 fail-closed 方向刻意相反。那边失败模式是
    两条链路同时驱动仪器（危险），这边失败模式是一条记录没归属（记账损失）。
    """
    assert requires_sample(_Meta(name="SomeNewSkill", tags=(), category="write")) is False


@pytest.mark.parametrize("name", ["SetBias", "SetSetpoint", "ZCtrlOn", "MotorMove"])
def test_plain_writes_are_not_gated(name):
    """改仪器参数不产生数据文件，不该被样品门控挡住。"""
    assert requires_sample(_Meta(name=name)) is False


# ── 必须拦截（产数据） ────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "StartScan", "SaveScan", "AcquireSTS", "GridSpectroscopy", "TipPulse",
])
def test_data_producing_names_are_gated(name):
    assert requires_sample(_Meta(name=name)) is True


@pytest.mark.parametrize("tag", [
    "scan", "spectroscopy", "grid", "datalog", "lithography", "manipulation",
])
def test_data_producing_tags_are_gated(tag):
    assert requires_sample(_Meta(name="Custom", tags=(tag,))) is True


# ── check_sample_scope 的整体行为 ─────────────────────────────────────

def test_gate_passes_when_sample_selected():
    assert check_sample_scope(_Meta("StartScan"), "StartScan", _Log("e1", "s1")) is None


def test_gate_blocks_when_no_sample():
    msg = check_sample_scope(_Meta("StartScan"), "StartScan", _Log("e1", None))
    assert msg and "no_active_sample" in msg
    assert "start_sample" in msg               # 给模型指出自救工具
    assert "Do NOT retry this call unchanged" in msg   # 防重试循环


def test_gate_message_differs_without_experiment():
    msg = check_sample_scope(_Meta("StartScan"), "StartScan", _Log(None, None))
    assert msg and "no_active_experiment" in msg
    assert "start_experiment" in msg


def test_gate_allows_everything_without_a_log():
    """记录系统缺席（无头 / 测试）时一律放行 —— 门控是记账约束，
    不该在记录系统本身没接线时把仪器锁死。"""
    assert check_sample_scope(_Meta("StartScan"), "StartScan", None) is None


def test_gate_never_raises_on_broken_log():
    class _Broken:
        @property
        def current_experiment_id(self):
            raise RuntimeError("boom")

        @property
        def current_sample_id(self):
            raise RuntimeError("boom")

    assert check_sample_scope(_Meta("StartScan"), "StartScan", _Broken()) is None


def test_remedy_passes_even_with_no_scope_at_all():
    """★ 最重要的一条：没有实验、没有样品，退针照样能跑。"""
    for name in ("SafeRetract", "EmergencyRetract", "WithdrawTip", "StopScan"):
        assert check_sample_scope(_Meta(name), name, _Log(None, None)) is None


def test_message_is_chinese_and_actionable():
    msg = sample_gate_message("StartScan", has_experiment=True)
    assert "样品" in msg and "StartScan" in msg
    assert "出路" in msg
