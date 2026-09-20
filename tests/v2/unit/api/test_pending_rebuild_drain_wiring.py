"""排队的 agent 工具表重建，**谁来补做**。

`CoreRuntime.request_agent_rebuild` 在任务运行中把重建排进队列而不是丢掉
（机制本身由 `tests/v2/unit/admin/test_override_hot_reload.py` 覆盖）。
但一个只会排队、没人来取的队列，比原来那句「暂不重建」更糟 —— 它多了一层
「我已经处理了」的错觉。

所以这份测试盯的是**接线**：

1. `_run_task_stream` 的 finally 里有一处 drain，且它必须在**所有权检查内**
   —— 一条晚退的旧流不能替刚接手的新 run 重建图（这是 2026-07-10 #41 那条
   race 的同一个形状）。
2. run-task 入口在认领 task slot **之前**有一处 `sync=True` 的 drain。
   这条是保底：客户端硬断连 / 流崩了会跳过 finally，那时只剩它。
   它买到的不变式也更强 —— 不是「总会补上」，而是**任何任务都不会在陈旧
   工具表上开跑**。

为什么用 AST 而不是跑一遍真流程：`_run_task_stream` 要一个编译好的
orchestrator 图、一个 BufferService 和一条真 SSE 流。这里要证明的不是它跑得
对，而是**两个调用点在不在该在的位置** —— 那是结构问题，用结构闸门答。
（本仓的教训：「每页各自记得」人肉找不齐，先写结构闸门。）
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = (Path(__file__).resolve().parents[4]
       / "MASTv2" / "mast" / "api" / "routes" / "orchestrator.py")


@pytest.fixture(scope="module")
def stream_fn() -> ast.FunctionDef:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_run_task_stream"):
            return node
    pytest.fail("_run_task_stream 没了 —— 排队重建的两个补做点要重接")


def _drain_calls(node) -> list[ast.Call]:
    """所有 `<x>(sync=...)`/`<x>()` 形式的 drain 调用（经 getattr 拿到的那个）。"""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and \
                n.func.id == "_drain":
            out.append(n)
    return out


def _mentions_drain_name(node) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and n.value == "drain_pending_agent_rebuild":
            return True
    return False


def test_both_drain_points_exist(stream_fn):
    """两处都要在。少一处就退回「有时候补做」。"""
    lookups = [n for n in ast.walk(stream_fn)
               if isinstance(n, ast.Call) and _mentions_drain_name(n)]
    assert len(lookups) == 2, (
        f"期望两处 drain_pending_agent_rebuild 接线（run-task 入口 + finally），"
        f"实际 {len(lookups)} 处"
    )


def test_entry_drain_is_synchronous(stream_fn):
    """入口那处必须 `sync=True`。

    异步的话任务会和重建赛跑，「不在陈旧工具表上开跑」这个不变式就没了 ——
    而那正是这一处存在的全部理由（finally 那处已经覆盖了「事后补做」）。
    """
    syncs = [c for c in _drain_calls(stream_fn)
             if any(kw.arg == "sync" and getattr(kw.value, "value", None) is True
                    for kw in c.keywords)]
    assert len(syncs) == 1, "入口 drain 必须且只需一处 sync=True"


def test_finally_drain_sits_inside_the_ownership_check(stream_fn):
    """finally 里那处必须在 `_released_the_slot` 的 if 内。

    反例（这就是为什么这条测试存在）：把 drain 提到 if 外面看起来更简洁，
    但一条**晚退的旧流**会在新 run 已经接手 slot 之后重建图 —— 把新任务的
    工具表在它脚下换掉。这与 `st["task"].get("id") == task_id` 那道检查防的是同一件事。
    """
    guarded = []
    for n in ast.walk(stream_fn):
        if not isinstance(n, ast.If):
            continue
        test_src = ast.dump(n.test)
        if "_released_the_slot" not in test_src:
            continue
        for inner in ast.walk(n):
            if isinstance(inner, ast.Call) and _mentions_drain_name(inner):
                guarded.append(inner)
    assert guarded, (
        "finally 里的 drain 不在 _released_the_slot 守卫内 —— "
        "晚退的旧流会替刚接手的新 run 重建图"
    )


def test_entry_drain_precedes_the_slot_claim(stream_fn):
    """入口 drain 必须在认领 slot **之前**。

    之后再 drain，就等于「先开跑、再换工具表」—— 正好是要避免的那件事。
    """
    drain_line = min(c.lineno for c in _drain_calls(stream_fn)
                     if any(kw.arg == "sync" for kw in c.keywords))
    claim_lines = [
        n.lineno for n in ast.walk(stream_fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Subscript)
                and isinstance(t.slice, ast.Constant) and t.slice.value == "task"
                for t in n.targets)
    ]
    assert claim_lines, 'st["task"] = {...} 的认领点没找到 —— 结构变了，这条要重写'
    assert drain_line < min(claim_lines), (
        f"sync drain 在第 {drain_line} 行，而 slot 认领在第 {min(claim_lines)} 行 —— "
        "顺序反了，任务会先开跑"
    )
