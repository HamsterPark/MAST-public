"""失败步骤只能写回经过回读验证的硬件事实。

显式声明的 _verified_state 在失败时也必须更新缓存；未声明的意图值不能写回。
本文件分别验证状态补丁、预扫描早停的实际返回以及 ExecutionContext.run 的完整路径。
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core.state import (  # noqa: E402
    VERIFIED_STATE_KEY,
    patch_state_from_result,
    patch_verified_state,
)


class _Cache:
    def __init__(self, **kw):
        self.scan_running = kw.get("scan_running")
        self.bias_v = kw.get("bias_v")


class _State:
    """够用的 InstrumentState 替身 —— 只保留 apply_patch 的白名单语义。"""

    _PATCHABLE = frozenset({"scan_running", "bias_v"})

    def __init__(self, **kw):
        self._cache = _Cache(**kw)

    def apply_patch(self, **fields):
        for k, v in fields.items():
            if v is not None and k in self._PATCHABLE:
                setattr(self._cache, k, v)


# ── 1. 声明过的事实,失败也写回 ─────────────────────────────────────────────
def test_a_verified_fact_is_written_back_even_when_the_skill_failed():
    st = _State(scan_running=True)
    failed_data = {
        "error_detail": "预扫描中途停止 (0/64 行)",
        VERIFIED_STATE_KEY: {"scan_running": False},
    }
    patch_verified_state(st, failed_data, what="PreScanCheck")
    assert st._cache.scan_running is False, (
        "技能明说「我回读验证过扫描已停」,而缓存还是 True —— "
        "下一步的 scan_not_running 前置会被这个陈值挡住。")


# ── 2. 没声明的字段一个都不写 ──────────────────────────────────────────────
def test_an_unverified_intent_value_is_not_written_back_on_failure():
    """失败的 SetBias 照样在 data 里带着目标 bias_v —— 那是**意图**,不是回读。

    这一条是上一条的边界:开一条失败也写的通道,就必须证明它**窄**。
    """
    st = _State(bias_v=0.5)
    failed_data = {"bias_v": 3.0, "error_detail": "SetBias 失败"}
    patch_verified_state(st, failed_data, what="SetBias")
    assert st._cache.bias_v == 0.5, (
        "一个失败技能的意图值被写进了状态缓存 —— 那是把虚构当成事实,"
        "比留一个陈值更坏。")


def test_the_success_path_still_writes_the_whole_data_block():
    """成功路径的语义一个字没改 —— 这次加的是**另一条**通道,不是改这一条。"""
    st = _State(bias_v=0.5, scan_running=True)
    patch_state_from_result(st, {"bias_v": 1.0, "scan_running": False},
                            what="SetBias")
    assert st._cache.bias_v == 1.0
    assert st._cache.scan_running is False


def test_a_declaration_that_is_not_a_dict_is_ignored():
    st = _State(scan_running=True)
    for junk in (None, "scan_running=False", [], 0, {"": None}):
        patch_verified_state(st, {VERIFIED_STATE_KEY: junk}, what="x")
    assert st._cache.scan_running is True


# ── 3. 端到端:真正会跑的那条代码接上了没有 ────────────────────────────────
def test_prescan_early_stop_actually_declares_the_scan_stopped():
    """执行早停分支，验证已确认的停止状态能够更新缓存。"""
    from mast.skills.composite.prescan_check import PreScanCheck
    from tests.v2.unit.skills.composite.test_prescan_check_graph import FakeCtx

    ctx = FakeCtx(wait_lines_done=3, wait_lines_total=64)
    result = PreScanCheck().execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 12e-9,
    })
    assert result.success is False
    assert result.data[VERIFIED_STATE_KEY] == {"scan_running": False}
    assert result.data["tip_ready"] is None
    assert result.data["similarity"] is None
    assert "Scan_Action" not in [method for method, _ in ctx.safe_call_log]

    st = _State(scan_running=True, bias_v=0.5)
    patch_verified_state(st, result.data, what="PreScanCheck")
    assert st._cache.scan_running is False
    assert st._cache.bias_v == 0.5

# ── 4. ⭐ 走**真** ExecutionContext.run 的那一条 ──────────────────────────
def test_the_runtime_path_really_writes_a_failed_skills_verified_fact():
    """真 `ExecutionContext.run` + 一个**失败**的技能 ⇒ 缓存必须被更新。

    ⚠️ 这一条是变异验证逼出来的:把 `execution_context.py` 里那行回写删掉之后,
    本文件原来的 4 条测试**一条都不红** —— 它们要么直接调 `patch_verified_state`,
    要么读源码,**没有一条经过真正会跑的那条路**。
    「闸门测的不是真正会跑的那条代码」是本仓反复栽的形状,这次差点又栽一遍。
    """
    from mast.core.execution_context import ExecutionContext
    from mast.core.registry import SkillRegistry
    from mast.core.types import (
        SafetyLevel, SkillCategory, SkillMetadata, SkillResult,
    )
    from mast.skills.base import BaseSkill

    class _FailsButSawTheHardware(BaseSkill):
        """失败返回,但它**回读验证过**扫描已经停了 —— PreScanCheck 早停路径的形状。"""

        def metadata(self):
            return SkillMetadata(
                name="_FailsButSawTheHardware", version="1.0.0",
                category=SkillCategory.READ, safety_level=SafetyLevel.AUTO,
                description="test double", estimated_duration_s=0.0,
                composition_level=0, tags=["test"])

        def execute(self, context, params):
            return SkillResult(
                skill_name="_FailsButSawTheHardware", success=False,
                error="预扫描中途停止 (0/64 行)",
                data={"bias_v": 9.99,            # 意图值:不许被写回
                      VERIFIED_STATE_KEY: {"scan_running": False}})

    reg = SkillRegistry()
    reg.register(_FailsButSawTheHardware)

    class _Pool:
        def safe_call(self, m, *a, role="main"):
            from mast.core.types import NanonisCallRecord
            return NanonisCallRecord(method=m, args=a)

    st = _State(scan_running=True, bias_v=0.5)
    ctx = ExecutionContext(pool=_Pool(), state=st, registry=reg)
    res = ctx.run("_FailsButSawTheHardware", {})

    assert res.success is False, "前提没成立:这个替身本该失败"
    assert st._cache.scan_running is False, (
        "失败技能声明过的已验证事实没有写回缓存，后续步骤会读取陈旧状态。")
    assert st._cache.bias_v == 0.5, (
        "失败技能 data 里的**意图值**被写进了缓存 —— 通道开得太宽了。")
