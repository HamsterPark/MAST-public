"""Agent→Wishlist tools: request_user_action + report_upgrade_idea (idea #3).

report_upgrade_idea lets an agent log a capability gap / upgrade idea to the
shared Wishlist (kind="upgrade") for the next upgrade — non-blocking.
"""
from __future__ import annotations

import pytest

from mast.agents._shared.request_tools import make_request_tools
from mast.wishlist import (
    get_board,
    list_agent_requests,
    post_agent_request,
    reset_default_board,
    resolve_agent_request,
)


@pytest.fixture(autouse=True)
def _isolated_board(tmp_path, monkeypatch):
    """Redirect the shared board into tmp so posting/resolving never writes the
    operator's REAL 心愿单 (enforced by the conftest guard)."""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    reset_default_board()
    yield
    reset_default_board()


def _tools(agent_id="instrument_control"):
    return make_request_tools(lambda: {"agent_id": agent_id, "experiment_id": None})


def test_both_tools_present():
    names = [t.name for t in _tools()]
    assert "request_user_action" in names
    assert "report_upgrade_idea" in names


# ── check_request_reply: a NON-destructive single-request poll (feedback ⑦) ────
# The one-way gap: the agent asked for a path, the operator answered on the board,
# and the blocked agent — with no turn on which to poll check_my_requests — never
# saw it. check_request_reply lets the agent poll ONE request it posted, as many
# times as it likes, without consuming the answer.

def test_check_request_reply_is_present():
    assert "check_request_reply" in [t.name for t in _tools()]


def test_check_request_reply_pending_says_wait():
    reset_default_board()
    rid = post_agent_request("data_processing", "请提供 Au111 图路径", kind="info")["id"]
    poll = next(t for t in _tools() if t.name == "check_request_reply")
    out = poll.invoke({"request_id": rid})
    assert "待处理" in out and rid in out


def test_check_request_reply_returns_the_answered_path():
    reset_default_board()
    rid = post_agent_request("data_processing", "请提供 Au111 图路径", kind="info")["id"]
    resolve_agent_request(rid, "done", path=r"D:\Data\Au111.sxm", note="就是这张")
    poll = next(t for t in _tools() if t.name == "check_request_reply")
    out = poll.invoke({"request_id": rid})
    assert "已完成" in out
    assert r"D:\Data\Au111.sxm" in out
    assert "就是这张" in out


def test_check_request_reply_is_non_destructive():
    """Polling must NOT consume the answer — a repeated poll returns the same
    thing, and the consuming reader (check_my_requests) still delivers it."""
    reset_default_board()
    rid = post_agent_request("data_processing", "路径?", kind="info")["id"]
    resolve_agent_request(rid, "done", path="D:\\x.sxm")
    tools = {t.name: t for t in _tools("data_processing")}
    # poll twice — both see it, and it stays undelivered (poll does not consume)
    assert "D:\\x.sxm" in tools["check_request_reply"].invoke({"request_id": rid})
    assert "D:\\x.sxm" in tools["check_request_reply"].invoke({"request_id": rid})
    assert get_board().resolved_requests_for("", undelivered_only=True), (
        "the poll must not mark the answer delivered")
    # check_my_requests (the consuming reader) still sees it, then consumes it
    assert "D:\\x.sxm" in tools["check_my_requests"].invoke({})
    assert not get_board().resolved_requests_for("", undelivered_only=True)


def test_check_request_reply_unknown_id():
    reset_default_board()
    poll = next(t for t in _tools() if t.name == "check_request_reply")
    out = poll.invoke({"request_id": "r-999"})
    assert "未找到" in out


def test_report_upgrade_idea_posts_upgrade_kind():
    reset_default_board()
    upg = next(t for t in _tools() if t.name == "report_upgrade_idea")
    out = upg.invoke({"message": "[仪器] 需要一个自动对中样品的 skill"})
    assert "心愿单" in out
    reqs = list_agent_requests()
    assert any(r.get("kind") == "upgrade" for r in reqs)


def test_request_user_action_still_works():
    reset_default_board()
    act = next(t for t in _tools("literature") if t.name == "request_user_action")
    out = act.invoke({"message": "[文献] 请上传论文全文 PDF", "kind": "upload"})
    assert "请求" in out
    reqs = list_agent_requests()
    assert any(r.get("kind") == "upload" for r in reqs)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
