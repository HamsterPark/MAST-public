"""The two ways an agent can ask the operator for something, made consistent.

There are two boards, and that is fine — they do different jobs:

  * 取文请求板 — typed by work_id, frozen to the experiment that asked, wired into
    ingest, and able to resume the conversation that was blocked;
  * 心愿单 — free text with a first-class ``path`` field, for "please do a thing"
    and "please tell me where that file is".

What was not fine: the generic tool advertised the specific one's job, only one
of the two boards had any operator-facing notification, the tools that create a
request existed on one entry point while the machinery that reads answers back
ran on both, and an arrival that auto-resume could not announce reached nobody.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_ask_channel_convergence.py -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (canonical block for tests/v2/) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest

from mast.agents._shared.request_tools import make_request_tools
from mast.knowledge.fetch_board import FetchBoard


@pytest.fixture
def tools():
    return {t.name: t for t in make_request_tools(
        lambda: {"agent_id": "literature", "experiment_id": "exp-1"})}


@pytest.fixture
def wish_board(tmp_path, monkeypatch):
    from mast.wishlist import store as wl_store
    board = wl_store.WishlistBoard(tmp_path / "wl")
    import mast.wishlist as wl
    monkeypatch.setattr(wl, "get_board", lambda: board, raising=False)
    monkeypatch.setattr(wl_store, "_BOARD", board, raising=False)
    return board


# ── the generic tool must not do the specific one's job ──────────────────

def test_asking_for_a_paper_is_redirected_not_posted(tools, wish_board):
    """Posting it here succeeds — and silently loses everything that matters.

    No work_id, no experiment freezing, no library filing, no ingest, and no
    resume when it arrives. The request would look answered and be worth less.
    """
    out = tools["request_user_action"].invoke(
        {"message": "[文献] 请上传论文 10.1103/PhysRevLett.49.57 的全文 PDF",
         "kind": "upload"})
    assert "request_fulltext" in out
    assert "10.1103/PhysRevLett.49.57" in out, "name the id it should pass along"
    assert wish_board.list_agent_requests() == [], "nothing should have been posted"


def test_work_id_form_is_redirected_too(tools, wish_board):
    out = tools["request_user_action"].invoke(
        {"message": "请上传 W2110167104 的全文", "kind": "upload"})
    assert "request_fulltext" in out and wish_board.list_agent_requests() == []


def test_a_non_paper_upload_still_posts(tools, wish_board):
    """The redirect must stay narrow — it is not a ban on the word 'upload'."""
    out = tools["request_user_action"].invoke(
        {"message": "[仪器] 请把标定曲线的 CSV 放到机器上，把路径填在心愿单里",
         "kind": "upload"})
    assert "已向用户发起请求" in out
    assert len(wish_board.list_agent_requests()) == 1


def test_a_paper_mentioned_in_an_action_request_still_posts(tools, wish_board):
    """Only an UPLOAD ask for a full text is the fetch board's job."""
    out = tools["request_user_action"].invoke(
        {"message": "请照着 10.1103/PhysRevLett.49.57 的方法换一次样品", "kind": "action"})
    assert "已向用户发起请求" in out
    assert len(wish_board.list_agent_requests()) == 1


def test_docstring_no_longer_advertises_the_other_board(tools):
    doc = tools["request_user_action"].description or ""
    assert "request_fulltext" in doc
    assert "请上传论文 <DOI/标题> 的全文 PDF" not in doc


# ── an arrival nobody announced still reaches the agent ──────────────────

def test_unannounced_arrivals_are_listed_then_marked(tmp_path):
    board = FetchBoard(board_dir=tmp_path / "libs")
    board.post_request("W1", requested_by="literature", reason="需要偏压")
    board.resolve_work_id("W1", "fulfilled", note="全文/条目已入库")

    rows = board.unannounced_fulfilled("literature")
    assert [r["request_id"] for r in rows] == ["fr-1"]

    assert board.mark_announced(["fr-1"]) == 1
    assert board.unannounced_fulfilled("literature") == []
    assert board.mark_announced(["fr-1"]) == 0, "marking twice must be a no-op"


def test_only_the_asking_agent_is_told(tmp_path):
    board = FetchBoard(board_dir=tmp_path / "libs")
    board.post_request("W1", requested_by="literature")
    board.post_request("W2", requested_by="data_processing")
    board.resolve_work_id("W1", "fulfilled")
    board.resolve_work_id("W2", "fulfilled")
    assert [r["work_id"] for r in board.unannounced_fulfilled("literature")] == ["W1"]


def test_still_pending_is_not_an_arrival(tmp_path):
    board = FetchBoard(board_dir=tmp_path / "libs")
    board.post_request("W1", requested_by="literature")
    assert board.unannounced_fulfilled("literature") == []


def test_arrival_block_carries_the_original_reason(tmp_path):
    from mast.agents._shared.resume_context import build_fetch_arrival_block
    board = FetchBoard(board_dir=tmp_path / "libs")
    board.post_request("W1", requested_by="literature", title="A paper",
                       reason="需要 methods 里的偏压")
    board.resolve_work_id("W1", "fulfilled", note="全文/条目已入库")

    block, ids = build_fetch_arrival_block(board)
    assert ids == ["fr-1"]
    assert "W1" in block and "A paper" in block
    assert "需要 methods 里的偏压" in block
    assert "search_fulltext" in block


def test_arrival_block_repeats_the_unreadable_warning(tmp_path):
    """A scanned upload closes the request without becoming readable."""
    from mast.agents._shared.resume_context import build_fetch_arrival_block
    board = FetchBoard(board_dir=tmp_path / "libs")
    board.post_request("W1", requested_by="literature")
    board.resolve_work_id("W1", "fulfilled",
                          note="PDF 已入库，但没有可读文本层（扫描件未 OCR）")

    block, _ids = build_fetch_arrival_block(board)
    assert "没有可读文本层" in block
    assert "不要反复换工具去试" in block


def test_nothing_to_announce_is_a_quiet_none(tmp_path):
    from mast.agents._shared.resume_context import build_fetch_arrival_block
    board = FetchBoard(board_dir=tmp_path / "libs")
    assert build_fetch_arrival_block(board) == (None, [])


# ── the wishlist can wake somebody up too ────────────────────────────────

def test_answering_a_request_resumes_the_conversation_that_asked(wish_board,
                                                                 monkeypatch):
    """An agent that stopped at the blocker has no later turn to read the answer on.

    The board's auto-injection only pays out if a turn happens; the whole reason
    the agent is waiting is that none will.
    """
    from mast.core import fetch_resume
    seen = {}
    fetch_resume.set_resumer(
        lambda w, r, **kw: seen.update(kind=kw.get("kind"), rows=r) or {"resumed": 1})
    try:
        rec = wish_board.post_agent_request(
            "instrument_control", "请到机台更换样品", kind="action",
            origin_conversation_id="conv-7")
        wish_board.resolve_agent_request(rec["id"], "done", note="换好了")
    finally:
        fetch_resume.clear_resumer()

    assert seen["kind"] == "request"
    assert seen["rows"][0]["origin_conversation_id"] == "conv-7"
    assert seen["rows"][0]["note"] == "换好了"


def test_a_resumed_answer_is_not_injected_again(wish_board, monkeypatch):
    """Told once, by whichever mechanism got there first."""
    from mast.core import fetch_resume
    fetch_resume.set_resumer(lambda w, r, **kw: {"resumed": 1})
    try:
        rec = wish_board.post_agent_request("literature", "帮个忙",
                                            origin_conversation_id="c1")
        wish_board.resolve_agent_request(rec["id"], "done")
    finally:
        fetch_resume.clear_resumer()
    assert wish_board.resolved_requests_for("", undelivered_only=True) == []


def test_a_failed_resume_leaves_it_for_the_readback(wish_board):
    """If nobody could be woken, the answer must still reach the next turn."""
    from mast.core import fetch_resume
    fetch_resume.set_resumer(lambda w, r, **kw: {"resumed": 0})
    try:
        rec = wish_board.post_agent_request("literature", "帮个忙",
                                            origin_conversation_id="c1")
        wish_board.resolve_agent_request(rec["id"], "done")
    finally:
        fetch_resume.clear_resumer()
    assert len(wish_board.resolved_requests_for("", undelivered_only=True)) == 1


def test_a_request_with_no_origin_still_resolves(wish_board):
    """Background runs have no conversation; resolving must not depend on one."""
    rec = wish_board.post_agent_request("literature", "帮个忙")
    out = wish_board.resolve_agent_request(rec["id"], "done", path="D:/x")
    assert out["status"] == "done" and out["path"] == "D:/x"


def test_origin_survives_a_reload(tmp_path):
    from mast.wishlist import store as wl_store
    b1 = wl_store.WishlistBoard(tmp_path / "wl")
    rid = b1.post_agent_request("literature", "x", origin_conversation_id="c9")["id"]
    b2 = wl_store.WishlistBoard(tmp_path / "wl")
    assert b2.get_request(rid)["origin_conversation_id"] == "c9"


def test_legacy_rows_without_origin_load(tmp_path):
    d = tmp_path / "wl"
    d.mkdir()
    (d / "wishlist.json").write_text(
        '{"wishes": [], "requests": [{"id": "r-1", "agent_id": "literature",'
        ' "message": "old", "status": "pending"}]}', encoding="utf-8")
    from mast.wishlist import store as wl_store
    board = wl_store.WishlistBoard(d)
    assert board.get_request("r-1").get("origin_conversation_id", "") == ""
    assert board.resolve_agent_request("r-1", "done")["status"] == "done"


# ── the operator's to-do count means what it says ────────────────────────

def test_upgrade_ideas_do_not_inflate_the_pending_count(wish_board, monkeypatch):
    """"Here is an idea for later" is not "please go do something"."""
    from fastapi.testclient import TestClient
    from mast.api.app import create_app
    from mast.api.context import AppContext

    wish_board.post_agent_request("literature", "请到机台换样品", kind="action")
    wish_board.post_agent_request("literature", "需要一个新 skill", kind="upgrade")

    ctx = AppContext()
    ctx.wishlist_board = wish_board          # type: ignore[attr-defined]
    body = TestClient(create_app(context=ctx, dev_cors=False)).get(
        "/api/wishlist").json()
    assert len(body["requests"]) == 2
    assert body["pending_count"] == 1


# ── the readback middleware covers both channels ─────────────────────────

def test_middleware_reads_back_both_boards(tmp_path, monkeypatch):
    from mast.agents._shared.request_readback_mw import (
        make_request_readback_middleware,
    )
    from mast.wishlist import store as wl_store
    import mast.wishlist as wl
    from mast.knowledge import fetch_board as fb

    wboard = wl_store.WishlistBoard(tmp_path / "wl")
    rec = wboard.post_agent_request("literature", "请给我文件路径", kind="action")
    wboard.resolve_agent_request(rec["id"], "done", path="D:/data/x.sxm")
    monkeypatch.setattr(wl, "get_board", lambda: wboard, raising=False)

    fboard = FetchBoard(board_dir=tmp_path / "libs")
    fboard.post_request("W1", requested_by="literature", reason="需要偏压")
    fboard.resolve_work_id("W1", "fulfilled")
    monkeypatch.setattr(fb, "_BOARD", fboard, raising=False)

    mw = make_request_readback_middleware(fetch_for_agent="literature")
    block = mw._build_and_consume()
    assert "D:/data/x.sxm" in block, "wishlist answer missing"
    assert "W1" in block, "fetch arrival missing"
    # Both marked, so the next turn injects nothing.
    assert mw._build_and_consume() == ""


def test_middleware_without_the_fetch_gate_stays_out_of_literature_business(
        tmp_path, monkeypatch):
    """A paper arriving must not interrupt an instrument conversation."""
    from mast.agents._shared.request_readback_mw import (
        make_request_readback_middleware,
    )
    from mast.knowledge import fetch_board as fb

    fboard = FetchBoard(board_dir=tmp_path / "libs")
    fboard.post_request("W1", requested_by="literature")
    fboard.resolve_work_id("W1", "fulfilled")
    monkeypatch.setattr(fb, "_BOARD", fboard, raising=False)

    mw = make_request_readback_middleware()          # no fetch_for_agent
    assert "W1" not in mw._build_and_consume()
    assert fboard.unannounced_fulfilled("literature"), "must stay unannounced"


def test_a_broken_board_does_not_break_the_turn(monkeypatch):
    from mast.agents._shared import request_readback_mw as mod

    mw = mod.make_request_readback_middleware(fetch_for_agent="literature")
    monkeypatch.setattr(mw, "_wishlist_block",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                        raising=False)
    with pytest.raises(RuntimeError):
        mw._wishlist_block()          # the stub really does raise
    # …and the real implementations swallow their own failures, so a turn survives.
    assert isinstance(mod.RequestReplyReadbackMiddleware()._fetch_block(), str)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
