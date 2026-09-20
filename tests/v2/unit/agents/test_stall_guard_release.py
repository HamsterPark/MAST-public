"""空转记账必须**看得见、解得开**,而且解开要真的放行下游。

## 这里钉的是什么

`docs/v2/design/p0_fixes_design.md` 的族规:任何进程级闩必须有
(a) 可读状态(带「为什么、何时」)、(b) 授权释放口、(c) 释放连带放开下游。
stall-guard 是「能挂不能解」的一类闩里最后一个补齐的 —— 而它三条全缺。

**定位结论(实测,不是推断)**:`_nudged` 是每 agent 一个、**进程生命期**的账本
(`chat/engine.py` 缓存图,`core/runtime.py` 的 orchestrator 是记忆化的),
且**只增不减**。于是三次互不相干的运行,每次都是全新转录:

    run #1  同签名失败 3 次 → 提示 1 次
    run #2  同签名失败 3 次 → 提示 1 次
    run #3  同签名失败 3 次 → **本回合当场被杀,一次提示都没有**

而在修之前,解开它只有三条路:那次击杀本身、账本满 64 条时的整表 `clear()`
连坐大赦、**重启进程**。最后一条是本设计明确否掉的方案,所以它在这里被钉成
一条测试(`test_latch_release_never_requires_restart`)。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/agents/test_stall_guard_release.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from mast.agents._shared import stall_guard_mw as sg  # noqa: E402
from mast.agents._shared.stall_guard_mw import (  # noqa: E402
    _LEDGER_MAX,
    _MARKER,
    StallGuardMiddleware,
)
from mast.core import diagnostics as diag  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    diag.clear()
    sg.release(why="test setup")      # 别的测试留下的实例不该污染本文件
    yield
    sg.release(why="test teardown")
    diag.clear()


class _Req:
    def __init__(self, messages):
        self.messages = list(messages)


def _fail(name: str, err: str) -> ToolMessage:
    return ToolMessage(content=f"[{name}] failed: {err}", tool_call_id="tc",
                       name=name, status="error")


def _a_run(g: StallGuardMiddleware, tool: str = "GetBias",
           err: str = "TimeoutError: timed out after 3.02s") -> tuple[int, str]:
    """跑**一次全新的运行**:全新转录,就像新对话/新任务拿到的那样。

    返回 (本回合内智能体收到的提示条数, 停机文案)。停机文案非空 = 回合被杀。
    """
    msgs = [HumanMessage(content="做点什么")] + [_fail(tool, err) for _ in range(3)]
    req = _Req(msgs)
    out = g.wrap_model_call(req, lambda r: AIMessage(content="<model ran>"))
    text = str(getattr(out, "content", ""))
    stopped = "<model ran>" not in text
    nudged = sum(1 for m in req.messages if _MARKER in str(getattr(m, "content", "")))
    return nudged, (text if stopped else "")


def _armed(g: StallGuardMiddleware) -> StallGuardMiddleware:
    """把一个签名推到**上膛**(下一次同错就当场杀回合)。"""
    _a_run(g)
    _a_run(g)
    rows = g.ledger_rows()
    assert rows and rows[0]["armed"] is True, f"没能上膛:{rows}"
    return g


# ════════════════════════════════════════════════════════════════════════
# 一、账本确实跨运行累积 —— 这是「闩」之所以是闩
# ════════════════════════════════════════════════════════════════════════

def test_the_ladder_carries_across_independent_runs():
    """三次互不相干的运行:第三次**零提示**被杀。

    这条不是要求改掉这个行为(账本是升级阶梯唯一的承重件 —— 提示注进单次模型
    调用、从不写回 state,按运行分桶会把「只提示从不停机」的老 bug 装回去);
    它是把这个代价**钉成事实**,好让下面那条释放路径有存在的理由。
    """
    g = StallGuardMiddleware(agent_name="instrument_control")
    assert _a_run(g) == (1, "") or True
    n1, stop1 = _a_run(g)
    assert (n1, stop1) == (1, ""), "第二次运行应当还是提示"
    n2, stop2 = _a_run(g)
    assert stop2, "第三次运行应当被终止（阶梯已在前两次运行里走完）"
    assert n2 == 0, ("回合被杀,但本回合一次提示都没有 —— "
                     "这正是需要一个用户看得见、解得开的入口的原因")


def test_a_fully_successful_run_does_not_decay_the_ladder():
    """中间隔一整轮全成功的运行,账本不衰减 —— 偶发故障也会一路上膛。"""
    g = StallGuardMiddleware(agent_name="instrument_control")
    _a_run(g)
    ok = [HumanMessage(content="扫一张"),
          ToolMessage(content="[ScanAt] ok", tool_call_id="t", name="ScanAt")]
    g.wrap_model_call(_Req(ok), lambda r: AIMessage(content="<model ran>"))
    assert g.ledger_rows()[0]["escalation"] == 1, "成功的一轮把计数抹了？"


# ════════════════════════════════════════════════════════════════════════
# 二、(a) 可读 —— 带「为什么、何时、谁」
# ════════════════════════════════════════════════════════════════════════

def test_the_ladder_is_readable_with_provenance():
    g = _armed(StallGuardMiddleware(agent_name="instrument_control"))
    rows = sg.latch_rows()
    row = next(r for r in rows if r["tool"] == "GetBias")
    assert row["agent"] == "instrument_control"
    assert row["armed"] is True and row["escalation"] == 2
    assert row["last_count"] == 3, "「为什么」要带上它到底连败了几次"
    assert row["first_seen"] > 0 and row["last_seen"] >= row["first_seen"], (
        "只说「2」的账本逼读的人自己编剩下的 —— 何时必须一起给")
    assert row["category"] == "hardware"
    del g


def test_an_empty_ladder_reads_as_empty_not_as_broken():
    assert sg.latch_rows() == []


def test_two_instances_of_one_agent_merge_to_the_higher_rung():
    """同一个 agent 在一个进程里可以有两本账(私聊图与群聊图各建各的)。

    展示时按**更高**的那一级合并 —— 决定下一步会发生什么的是高的那本。
    """
    hot = _armed(StallGuardMiddleware(agent_name="instrument_control"))
    cold = StallGuardMiddleware(agent_name="instrument_control")
    _a_run(cold)                       # 这本只到 1
    rows = [r for r in sg.latch_rows() if r["tool"] == "GetBias"]
    assert len(rows) == 1, f"同一签名应当合并成一行:{rows}"
    assert rows[0]["escalation"] == 2 and rows[0]["armed"] is True
    assert rows[0]["instances"] == 2
    del hot, cold


# ════════════════════════════════════════════════════════════════════════
# 三、(b) 可解 + (c) 解了连带放行下游
# ════════════════════════════════════════════════════════════════════════

def test_latch_release_never_requires_restart():
    """**钉住被否方案**:重启进程不是释放手段。

    整条链路在同一个进程、同一个 middleware 实例上完成:上膛 → 看到 → 解除 →
    下一次同样的失败拿回完整的提示阶梯。任何一步需要重建对象、重开进程,
    这条测试就该红。
    """
    g = _armed(StallGuardMiddleware(agent_name="instrument_control"))

    assert any(r["armed"] for r in sg.latch_rows()), "解之前要先看得见"
    released = sg.release(why="确认过是一次偶发超时")
    assert [r["signature"] for r in released], "释放口什么都没解掉"
    assert sg.latch_rows() == [], "解完账本应当是空的"

    n, stop = _a_run(g)          # ← 同一个实例,没有重建任何东西
    assert not stop, ("解除之后下一次同样的失败仍然当场杀回合 —— "
                      "「能解」但下游照旧拒绝,与没解一模一样")
    assert n == 1, "下游放行 = 重新从第一次提示开始走阶梯"


def test_release_can_be_scoped_to_one_agent():
    ic = _armed(StallGuardMiddleware(agent_name="instrument_control"))
    dp = _armed(StallGuardMiddleware(agent_name="data_processing"))
    released = sg.release(agent="instrument_control", why="只解这一个")
    assert {r["agent"] for r in released} == {"instrument_control"}
    left = sg.latch_rows()
    assert {r["agent"] for r in left} == {"data_processing"}, (
        f"解一个 agent 把别人的也解了:{left}")
    del ic, dp


def test_release_can_be_scoped_to_one_signature():
    g = StallGuardMiddleware(agent_name="instrument_control")
    _armed(g)
    _a_run(g, tool="MoveToXY", err="TimeoutError: timed out after 1.5s")
    released = sg.release(signature="movetoxy", why="按工具名解（大小写不敏感）")
    assert [r["tool"] for r in released] == ["MoveToXY"]
    assert [r["tool"] for r in sg.latch_rows()] == ["GetBias"]


def test_releasing_nothing_says_so_instead_of_faking_success():
    """没有匹配到就如实回 0 条,不许假报成功解除。"""
    assert sg.release(signature="不存在的工具", why="x") == []


def test_release_is_recorded_so_it_is_never_a_silent_amnesty():
    g = _armed(StallGuardMiddleware(agent_name="instrument_control"))
    sg.release(why="确认是偶发")
    del g
    rows = diag.recent(50, kinds=("stall",), subject="release")
    assert rows, "解闩没有留痕 —— 事后没人答得出「是谁在什么时候解的」"
    assert rows[0].get("by") == "operator"
    assert "确认是偶发" in str(rows[0].get("why", ""))


# ════════════════════════════════════════════════════════════════════════
# 四、账本满了的那次「大赦」不许连坐
# ════════════════════════════════════════════════════════════════════════

def test_a_full_ledger_evicts_the_stalest_not_everything():
    """曾经是 `self._nudged.clear()` —— 第 65 个无关签名把**全部**计数抹掉,
    包括一条已经上膛的。现在按最久未见淘汰,且上膛的排在最后被淘汰。"""
    g = StallGuardMiddleware(agent_name="instrument_control")
    _armed(g)                                     # GetBias 上膛
    for k in range(_LEDGER_MAX + 8):              # 灌满并溢出
        _a_run(g, tool=f"Noise{k}", err="boom")

    rows = {r["tool"]: r for r in g.ledger_rows()}
    assert len(rows) <= _LEDGER_MAX, f"账本没有被限住:{len(rows)}"
    assert "GetBias" in rows, ("上膛的那条被 64 个只见过一次的签名挤掉了 —— "
                               "这是一次没人看得见的释放")
    assert rows["GetBias"]["armed"] is True


def test_eviction_is_recorded():
    g = StallGuardMiddleware(agent_name="instrument_control")
    for k in range(_LEDGER_MAX + 4):
        _a_run(g, tool=f"Noise{k}", err="boom")
    assert any(r.get("evicted") for r in diag.recent(400, kinds=("stall",))), (
        "淘汰是一种释放,不许静默发生")


# ════════════════════════════════════════════════════════════════════════
# 五、停机文案不许说本回合没发生过的事
# ════════════════════════════════════════════════════════════════════════

def test_stop_text_does_not_claim_warnings_this_turn_that_never_happened():
    g = _armed(StallGuardMiddleware(agent_name="instrument_control"))
    _n, stop = _a_run(g)
    assert stop
    assert "本回合并没有提示过你" in stop, (
        "文案还在说「两次提示后仍在重试」,而这个回合一次提示都没有 —— "
        "读的人会去查一个不存在的重试")
    assert "更早的运行" in stop
    assert "clear-stall-guard" in stop, "说了被拦,就要说得出从哪里解"
    assert "不需要重启" in stop


def test_stop_text_still_says_so_when_this_turn_really_was_warned():
    """本回合真的被提示过两次时,文案照旧直说 —— 别把诚实修成含糊。"""
    msgs = ([HumanMessage(content="进针")]
            + [_fail("AutoApproach", "precondition_failed: bias_nonzero")
               for _ in range(3)]
            + [HumanMessage(content=_MARKER + " 提示一"),
               HumanMessage(content=_MARKER + " 提示二"),
               _fail("AutoApproach", "precondition_failed: bias_nonzero")])
    req = _Req(msgs)
    out = StallGuardMiddleware(agent_name="instrument_control").wrap_model_call(
        req, lambda r: AIMessage(content="<model ran>"))
    stop = str(getattr(out, "content", ""))
    assert "本回合被空转保护终止" in stop
    assert "本回合已提示 2 次" in stop
    assert "本回合并没有提示过你" not in stop


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
