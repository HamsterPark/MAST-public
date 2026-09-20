"""CRITICAL 告警到底有没有**进到模型手里** —— 端到端,走真的编译图。

## 为什么这一组不能只测中间件

2026-08-08 的失败不是「中间件写错了」,是**没有任何东西证明那条链路是通的**。
⑰(2026-08-08)割掉打断链时,模块 docstring 写下的替代方案是:

> 事件本身**一个字节都没少**:照发射、照进 buffer 的 event_journal、照进
> ``state["event_refs"]``、``ReadHardwareEvents`` 照读得到、面板照看得到。

这句话里有**一半从写下那天起就不成立**:``state["event_refs"]`` 全树**零读者**;
而另一半(``ReadHardwareEvents``)是**拉取式**的 —— 它证明的是「查得到」,
不是「会知道」。两者的差别正好是那 13 分钟。

所以这一组测的是**可观测行为**,不是字段被写了:

> 一条 CRITICAL 落库 → agent 随后发起模型调用 →
> **断言模型在第一次调用时收到的文本里就有它**。

「有没有 delivered_agent 这个字段」答不了「agent 到底看见没看见」——
KNOWN_ISSUES §2.22 已经为这个形状付过一次学费(「一个零可达调用方的原语,
从测试外面看和一个能用的逃生门一模一样」),`ack_alert` 是第二次,
`state["event_refs"]` 是第三次。这一组存在就是为了不让 ``delivered_agent`` 成为第四次。

## 走的是**生产接线**

中间件在 ``graph.build()`` 里是**不带任何注入**构造的
(``AlertDeliveryMiddleware()``),所以它走默认路径 ``get_store_if_exists()``。
这里用 ``set_store_for_test`` 把那个单例指到 tmp 库 —— 于是测的是真的那条路,
而不是一条只有测试才走的旁路。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/instrument_control/test_alert_delivery_reaches_the_model.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found above " + str(Path(__file__).resolve()))


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

import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from mast.agents.instrument_control.graph import build  # noqa: E402
from mast.core.types import NanonisCallRecord  # noqa: E402
from mast.monitoring.store import (  # noqa: E402
    CurrentMonitorStore, set_store_for_test,
)

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of  # noqa: E402

#: 现场那条 CRITICAL 的正文(``alerts.summarize_zh("saturation", ...)`` 的产物)。
SAT_ZH = ("隧道电流持续贴轨饱和(段内 100% 的样本达到满量程)"
          "——疑似撞针或前置放大器过载,建议停止扫描并检查针尖。")

#: 模型每次被调用时收到的消息,按调用顺序。
_SEEN: list[list] = []


class _CapturingModel(GenericFakeChatModel):
    """记录每次调用收到的 messages —— 「模型看见了什么」是这一组唯一的判据。"""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _SEEN.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager,
                                 **kwargs)


@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        return NanonisCallRecord(method=method, args=args,
                                 error=f"unmocked: {method}")


@pytest.fixture()
def store(tmp_path):
    """把**生产单例**指到 tmp —— 中间件走的就是这条默认路径。"""
    s = CurrentMonitorStore(tmp_path / "monitor.sqlite", tmp_path)
    set_store_for_test(s)
    _SEEN.clear()
    yield s
    set_store_for_test(None)
    s.close()
    _SEEN.clear()


def _agent(n_turns: int = 3):
    llm = _CapturingModel(messages=iter(
        [AIMessage(content=f"回合 {i}") for i in range(n_turns)]))
    ctx = FakeCtx()
    return build(buf=None, context_provider=(lambda: ctx), model=llm,
                 checkpointer=InMemorySaver(), enable_hitl=False)


def _text_of(call_idx: int) -> str:
    return "\n".join(str(getattr(m, "content", "")) for m in _SEEN[call_idx])


# ══════════════════════════════════════════════════════════════════════
# 验收:复现 2026-08-08 那一次
# ══════════════════════════════════════════════════════════════════════

def test_a_critical_reaches_the_model_on_the_very_next_call(store):
    """**这条就是那次事故的验收。**

    CRITICAL 在 T 时刻入表 → agent 随后发起模型调用 →
    第一次调用收到的文本里就必须有它。不是「查得到」,是「送到了」。
    """
    store.add_alert(ts=time.time() - 5, level="critical", rule="saturation",
                    summary_zh=SAT_ZH, emitted_buffer=True)

    agent = _agent()
    agent.invoke({"messages": [("user", "继续扫描第 3 个区域")]},
                 config={"configurable": {"thread_id": "t-1"}})

    assert _SEEN, "模型一次都没被调用 —— 这条测试什么也没证明"
    first = _text_of(0)
    assert "saturation" in first, "CRITICAL 没有进到模型第一次调用的上下文里"
    assert "建议停止扫描并检查针尖" in first
    assert "先处置" in first


def test_delivered_agent_is_actually_written_by_the_production_path(store):
    """``delivered_agent`` 不是第四个「有原语没人调」。

    注意这里**没有注入任何东西** —— 中间件在 graph.build() 里是裸构造的,
    走的是 ``get_store_if_exists()``。这条断言证明那条默认路径真的会落账。
    """
    aid = store.add_alert(ts=time.time() - 5, level="critical", rule="saturation",
                          summary_zh=SAT_ZH)
    agent = _agent()
    agent.invoke({"messages": [("user", "继续")]},
                 config={"configurable": {"thread_id": "t-2"}})

    rows, _ = store.alerts_query(limit=10)
    row = next(r for r in rows if r["id"] == aid)
    assert row["delivered_agent"] == 1, (
        "agent 看过了,但那一列还是 0 —— 于是「看没看见」这个问题又没人能回答")


def test_no_alerts_means_the_model_gets_nothing_extra(store):
    """没有未处理 CRITICAL 时注入块必须是空的。

    否则每轮都塞东西,agent 会学会忽略它 —— 那等于没接。
    """
    agent = _agent()
    agent.invoke({"messages": [("user", "继续")]},
                 config={"configurable": {"thread_id": "t-3"}})
    assert _SEEN
    assert "电流监控告警" not in _text_of(0)


def test_the_same_critical_is_not_re_injected_on_later_calls(store):
    """同一条不会每轮重复注入(确认语义)。"""
    store.add_alert(ts=time.time() - 5, level="critical", rule="saturation",
                    summary_zh=SAT_ZH)
    agent = _agent()
    for i in range(2):
        agent.invoke({"messages": [("user", f"第 {i} 步")]},
                     config={"configurable": {"thread_id": f"t-4-{i}"}})
    assert len(_SEEN) >= 2
    assert "saturation" in _text_of(0)
    assert "saturation" not in _text_of(1)


def test_the_middleware_is_actually_wired_into_the_built_agent():
    """**可达性**,不是「原语能用」。

    §2.22 的教训逐字:``reset_all_gates`` / ``resolve_all_gates`` 的旧测试
    「**自己直接调用**那两个重开函数……没有任何东西断言有人会调用它」。
    所以这里断言的是「build() 真的把它挂上去了」,而不是「这个类能工作」。
    """
    from mast.agents._shared.alert_delivery_mw import AlertDeliveryMiddleware

    agent = _agent()
    found = any(isinstance(m, AlertDeliveryMiddleware)
                for m in getattr(agent, "middleware", None) or [])
    if not found:
        # 编译后的图不一定把 middleware 列表挂在同一个属性上;退回到构建源
        # (仍然是**可达性**判据:build() 里有没有把它接上去)。
        import inspect
        src = source_of(build)
        found = "AlertDeliveryMiddleware()" in src
    assert found, "IC agent 没有挂 AlertDeliveryMiddleware —— 投递整条不可达"
