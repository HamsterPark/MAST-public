"""SPIKE（P2-F）：langgraph 1.1 工具/节点体内 interrupt() 的传播与重放语义。

这不是功能测试，是把 human 节点设计建立其上的三条语义钉死：
  1. node 体内（深层普通函数调用栈中）raise 的 interrupt() 会以
     '__interrupt__' 冒泡到 invoke 结果；
  2. Command(resume=...) 续跑时该 node 从头整体重放，interrupt() 此时返回
     决议值；
  3. 重放意味着 interrupt 之前的副作用会执行两次 —— 这就是 composite 步进度
     必须带外持久化（sidecar）的证明：否则 human 节点前已执行的仪器动作会
     在 resume 时真实地重复执行。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_interrupt_spike.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class _S(TypedDict, total=False):
    answer: str
    done: bool


def _deep_helper(side_effects: list) -> str:
    """模拟 composite 解释器深层调用栈里的 human 节点。"""
    side_effects.append("pre-interrupt-step")      # ≈ interrupt 前的仪器动作
    decision = interrupt({"question": "继续吗？", "routes": ["yes", "no"]})
    side_effects.append("post-interrupt-step")
    return str(decision)


def test_interrupt_bubbles_and_replays_node_from_top():
    side_effects: list = []

    def node(state: _S) -> _S:
        ans = _deep_helper(side_effects)           # 深层栈内 interrupt
        return {"answer": ans, "done": True}

    g = StateGraph(_S)
    g.add_node("work", node)
    g.add_edge(START, "work")
    g.add_edge("work", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "spike-1"}}

    # 1) 第一次 invoke：interrupt 冒泡，不抛异常、不执行 post 段
    out = app.invoke({}, cfg)
    assert "__interrupt__" in out
    intr = out["__interrupt__"][0]
    assert intr.value["question"] == "继续吗？"
    assert side_effects == ["pre-interrupt-step"]
    assert "done" not in (app.get_state(cfg).values or {})

    # 2) resume：node 从头整体重放（pre 段第二次执行！），interrupt 返回决议
    out2 = app.invoke(Command(resume="yes"), cfg)
    assert out2.get("answer") == "yes" and out2.get("done") is True
    # 3) 重放语义钉死：pre 段执行了两次 —— sidecar 带外持久化的必要性证明
    assert side_effects == ["pre-interrupt-step",
                            "pre-interrupt-step",
                            "post-interrupt-step"]


def test_interrupt_outside_graph_raises_cleanly():
    """图运行时之外调用 interrupt() 抛错（human 节点在 GUI 手动路径需降级）。"""
    with pytest.raises(Exception):
        interrupt({"question": "x"})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
