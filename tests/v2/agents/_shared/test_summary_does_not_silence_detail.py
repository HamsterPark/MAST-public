"""技能的摘要和诊断数据都必须穿过工具边界。相同失败结果带摘要时，不得比无摘要时丢失更多详情。"""
from __future__ import annotations

# ── path bootstrap (robust walk-up) ──
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
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
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from operator import add

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import SafetyLevel, SkillCategory, SkillMetadata


# ── 真机那一份回包的形状(数字取自真机 AutoTilt 回包) ──────────────────────────

AUTOTILT_DETAIL = ("第 1 轮后残余 Z 占用 4.21e-08 m,未降到上一轮 5.09e-08 m "
                   "的 70% 以下")
AUTOTILT_DATA: dict[str, Any] = {
    "outcome": "rolled_back",
    "reason": "diverged",
    "detail": AUTOTILT_DETAIL,
    "matrix_g": [[-0.9610, 0.1640], [-0.2670, -0.9862]],
    "history": [{
        "iteration": 1,
        "slope_in_deg": [0.3012, -0.2988],
        "delta_tilt_deg": [0.2404, 0.3752],
        "n_sub_steps": 1,
        "residual_slope_vec_deg": [0.2871, -0.2803],
        "truncated_at_limit": False,
    }],
}
AUTOTILT_SUMMARY = "AutoTilt: rolled_back(diverged)"


@dataclass
class _Result:
    success: bool = False
    data: dict = field(default_factory=dict)
    error: str = ""
    summary: "str | None" = None


def _skill(name: str, result: _Result):
    class _S:
        def metadata(self):
            return SkillMetadata(
                name=name, description="d",
                category=SkillCategory.READ,
                safety_level=SafetyLevel.AUTO,
                parameters=[],
            )

        def validate_params(self, kwargs):
            return []

        def execute(self, ctx, params):
            return result

    return _S


class _ToolState(TypedDict):
    messages: Annotated[list, add_messages]
    executed_skills: Annotated[list, add]
    scan_paths: Annotated[list, add]


def _content(name: str, result: _Result) -> str:
    """把技能推过**真的** ToolNode,取回 agent 实际读到的那段文本。

    不直接调 ``tool.func``:边界正是被测对象,绕过它就等于假设它是对的。
    """
    tool = wrap_skill(_skill(name, result), lambda: object())
    g = StateGraph(_ToolState)
    g.add_node("tools", ToolNode([tool]))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    ai = AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "tc"}])
    out = g.compile().invoke(
        {"messages": [ai], "executed_skills": [], "scan_paths": []})
    msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert len(msgs) == 1
    return msgs[0].content


# ══════════════════════════════════════════════════════════════════════════
# 一、真机那一份:结论和支撑它的东西必须一起过边界
# ══════════════════════════════════════════════════════════════════════════

def test_the_autotilt_shape_carries_its_own_explanation():
    """`diverged` 这个名字不许是唯一穿过来的东西。"""
    body = _content("AutoTilt", _Result(
        success=False, data=dict(AUTOTILT_DATA), summary=AUTOTILT_SUMMARY))

    assert AUTOTILT_SUMMARY in body, "摘要本身还是要在"
    # 那两个绝对 span —— 「发散」与「降得不够快」正是靠它们分开的。
    assert "4.21e-08" in body and "5.09e-08" in body
    # 逐轮向量:方向类的失效只有向量看得出来(附十八)。
    assert "slope_in_deg" in body and "residual_slope_vec_deg" in body
    assert "truncated_at_limit" in body, "环路有没有施加它算出来的量"


# ══════════════════════════════════════════════════════════════════════════
# 二、承重条款:**设了 summary 不许因此带得更少**
# ══════════════════════════════════════════════════════════════════════════

def test_writing_a_summary_does_not_silence_the_diagnosis():
    """诱因本身要被移除,不是靠提醒作者「记得把数塞进摘要」。

    ⚠️ 这一条是和上一条**分开**写的:上一条在「两边都带」和「两边都不带」
    之间分得开,却分不出诱因还在不在。要证明的是**两条路都通**。
    """
    with_summary = _content("WithSummary", _Result(
        success=False, data=dict(AUTOTILT_DATA), summary=AUTOTILT_SUMMARY))
    without_summary = _content("NoSummary", _Result(
        success=False, data=dict(AUTOTILT_DATA), summary=None,
        error="rolled_back: diverged"))

    for key in ("4.21e-08", "5.09e-08", "residual_slope_vec_deg"):
        assert key in with_summary, f"设了摘要就丢了 {key}"
        assert key in without_summary, f"没设摘要也丢了 {key}"


def test_a_failure_without_data_still_reads_cleanly():
    """没有 data 的失败不该多出一行空壳。"""
    body = _content("Bare", _Result(success=False, data={}, summary="Bare: nope"))
    assert body.strip() == "Bare: nope"


# ══════════════════════════════════════════════════════════════════════════
# 三、成功路径:带 `detail`,但**不**无差别地把整个 data 附上
# ══════════════════════════════════════════════════════════════════════════

def test_success_carries_detail_when_the_skill_wrote_one():
    body = _content("OkWithDetail", _Result(
        success=True, data={"detail": "半径由扫描框推导: 80 nm", "z_m": 1.5e-9},
        summary="倾斜 0.13°"))
    assert "倾斜 0.13°" in body
    assert "半径由扫描框推导" in body


@pytest.mark.parametrize("data", [
    {"z_m": 1.5e-9, "noise_floor_m": 1.5e-11},          # 有 data,无 detail
    {"detail": "   "},                                   # detail 是空白
])
def test_success_without_a_detail_stays_short(data):
    """成功路径不许因为这次改动整体变长 —— 那是另一个方向的错误。

    每一次工具返回都变长会挤占上下文、并让提示缓存反复失效。
    这一条是上面那条的**代价上限**。
    """
    body = _content("OkQuiet", _Result(success=True, data=data, summary="ok"))
    assert body == "ok"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
