"""RequestReplyReadbackMiddleware (⑦ 稳版): auto-inject operator answers.

A blocked agent that stopped has no turn on which to poll check_my_requests, so
the operator's answer sat unread (agent 「无记忆/无匹配」). This middleware hands
the answer over automatically on the agent's next turn — the framework providing
certain info, not the LLM rediscovering it. Pins: it injects the path/note into
the system message, marks it delivered exactly once, and stays stable across the
sub-steps of one turn (prompt-cache safe).

2026-08-24：落点从 system 消息换成**最后一条 human 消息**。答复的内容逐轮不同，
而 Anthropic 的 cache 断点就打在 system 末尾 —— 把逐轮变的东西放在那里，整段
system 加上全部历史每一轮都 miss。这些测试因此同时钉两件事：块**注进去了**，
以及 system **逐字节没变**。后一条是旧版没有的。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from mast.agents._shared.request_readback_mw import (
    RequestReplyReadbackMiddleware,
    make_request_readback_middleware,
)
from mast.wishlist import get_board, post_agent_request, reset_default_board, resolve_agent_request


@pytest.fixture(autouse=True)
def _isolated_board(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    reset_default_board()
    yield
    reset_default_board()


class _Req:
    """Minimal stand-in for a middleware ModelRequest."""

    def __init__(self, human_text: str, system=None):
        self.messages = [HumanMessage(content=human_text)]
        self.system_message = system


def _answered(rid_msg="请提供 Au111 图路径", path=r"D:\Data\Au111.sxm", note=""):
    rid = post_agent_request("data_processing", rid_msg, kind="info")["id"]
    resolve_agent_request(rid, "done", path=path, note=note)
    return rid


def _sys_text(req) -> str:
    sm = req.system_message
    return sm.content if isinstance(sm, SystemMessage) else (str(sm) if sm else "")


def _human_text(req) -> str:
    """最后一条 human 消息的文本 —— 块现在落在这里。"""
    msgs = list(getattr(req, "messages", None) or [])
    for m in reversed(msgs):
        kind = getattr(m, "type", None) or type(m).__name__
        if str(kind).lower() in ("human", "humanmessage", "user"):
            c = getattr(m, "content", "")
            return c if isinstance(c, str) else str(c)
    return ""


# ── the core behaviour ────────────────────────────────────────────────────────
def test_answer_is_injected_onto_the_last_human_message():
    rid = _answered(path=r"D:\Data\Au111.sxm", note="就是这张")
    mw = RequestReplyReadbackMiddleware()
    req = mw._apply(_Req("继续"))
    text = _human_text(req)
    assert rid in text
    assert r"D:\Data\Au111.sxm" in text
    assert "就是这张" in text
    # 原来的用户提问还在 —— 是**追加**，不是替换。
    assert "继续" in text


def test_the_system_message_is_left_untouched():
    """这条是落点搬家的**目的**，不是副作用。

    system 一变，Anthropic 的 cache 断点（打在 system 末尾）后面的全部前缀
    ——system 本身加上整段历史——每一轮都 miss。
    """
    _answered(path=r"D:\Data\Au111.sxm")
    mw = RequestReplyReadbackMiddleware()
    req = mw._apply(_Req("继续", system=SystemMessage(content="ORIGINAL-SYS")))
    assert _sys_text(req) == "ORIGINAL-SYS"
    assert r"D:\Data\Au111.sxm" in _human_text(req)


def test_falls_back_to_the_system_message_when_there_is_no_human_turn():
    """没有 human 消息可挂时退回 system —— 答复送达压过一次 cache 命中。"""
    _answered(path=r"D:\Data\Au111.sxm")
    mw = RequestReplyReadbackMiddleware()
    req = SimpleNamespace(messages=[], system_message=SystemMessage(content="SYS"))
    out = mw._apply(req)
    assert r"D:\Data\Au111.sxm" in _sys_text(out)


def test_answer_is_marked_delivered_exactly_once():
    _answered()
    mw = RequestReplyReadbackMiddleware()
    mw._apply(_Req("继续"))
    # consumed: nothing undelivered remains on the board
    assert get_board().resolved_requests_for("", undelivered_only=True) == []


def test_block_is_stable_across_substeps_of_one_turn():
    """Same human message (a ReAct loop's sub-steps) → same injected block, so the
    system message does not thrash the prompt cache within a turn."""
    _answered(path="D:\\x.sxm")
    mw = RequestReplyReadbackMiddleware()
    first = _human_text(mw._apply(_Req("继续")))
    second = _human_text(mw._apply(_Req("继续")))   # same turn key
    assert first == second
    assert "D:\\x.sxm" in first


def test_next_turn_does_not_reinject_a_delivered_answer():
    _answered(path="D:\\x.sxm")
    mw = RequestReplyReadbackMiddleware()
    assert "D:\\x.sxm" in _human_text(mw._apply(_Req("继续")))
    # a genuinely new turn (different human text) recomputes → already delivered
    req2 = mw._apply(_Req("扫描下一个区域"))
    assert "D:\\x.sxm" not in _human_text(req2)
    assert req2.system_message is None or "D:\\x.sxm" not in _sys_text(req2)


def test_no_answer_leaves_the_request_untouched():
    mw = RequestReplyReadbackMiddleware()
    req = mw._apply(_Req("继续"))
    assert req.system_message is None


def test_appends_to_an_existing_human_message():
    _answered(path="D:\\x.sxm")
    mw = RequestReplyReadbackMiddleware()
    req = mw._apply(_Req("原始提问", system=SystemMessage(content="ORIGINAL-SYS")))
    text = _human_text(req)
    assert "原始提问" in text and "D:\\x.sxm" in text
    assert _sys_text(req) == "ORIGINAL-SYS"


def test_wrap_model_call_passes_modified_request_to_handler():
    _answered(path="D:\\x.sxm")
    mw = make_request_readback_middleware()
    seen = {}

    def handler(req):
        seen["human"] = _human_text(req)
        return "ok"

    out = mw.wrap_model_call(_Req("继续"), handler)
    assert out == "ok"
    assert "D:\\x.sxm" in seen["human"]


def test_pending_not_yet_answered_is_not_injected():
    post_agent_request("data_processing", "路径?", kind="info")  # posted, NOT resolved
    mw = RequestReplyReadbackMiddleware()
    assert mw._apply(_Req("继续")).system_message is None


# ── wiring: the middleware reaches BOTH 群聊 and 私聊 (评审: 私聊路径也命中) ──
def test_wired_into_chat_agent_middleware():
    """``_chat_agent_middleware`` is the ONE list shared by the orchestrator agents
    (群聊) and the private-chat agents (私聊). Asserting the readback middleware is
    in it proves the 私聊 main chat is covered — the gap the route-level supervisor
    injection could not reach."""
    from mast.core.runtime import CoreRuntime

    stub = SimpleNamespace(_settings=None, _cognition=None,
                           _chat_experiment_id=lambda: None)
    mws = CoreRuntime._chat_agent_middleware(stub, "instrument_control")
    assert any(type(m).__name__ == "RequestReplyReadbackMiddleware" for m in mws), (
        "readback middleware missing from _chat_agent_middleware → 私聊 not covered")


# ── route hint + middleware together deliver EXACTLY once (no double-deliver) ──
def test_route_hint_is_read_only_middleware_delivers_once():
    """The route hint (supervisor routing) must NOT consume the answer; the agent
    middleware is the single place that delivers + marks it."""
    from mast.api.routes.orchestrator import _resume_lead_messages

    rid = _answered(path="D:\\once.sxm")
    app = SimpleNamespace(_experiment_log=None, _storage=None, _plan_store=None)

    # route: names the request but leaves it undelivered (read-only)
    lead = _resume_lead_messages(app, "继续")
    assert rid in "\n".join(getattr(m, "content", "") for m in lead)
    assert get_board().resolved_requests_for("", undelivered_only=True), (
        "route hint must not consume the answer")

    # middleware: delivers the path + marks delivered
    mw = RequestReplyReadbackMiddleware()
    assert "D:\\once.sxm" in _human_text(mw._apply(_Req("继续")))
    assert get_board().resolved_requests_for("", undelivered_only=True) == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
