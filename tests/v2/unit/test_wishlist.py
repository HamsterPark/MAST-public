"""Wishlist board + agent→user request tool + render helpers."""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2 = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2:
    while _MASTV2 in sys.path:
        sys.path.remove(_MASTV2)
    sys.path.insert(0, _MASTV2)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.wishlist.store import WishlistBoard  # noqa: E402


def _board(tmp_path) -> WishlistBoard:
    return WishlistBoard(board_dir=str(tmp_path))


def test_wish_lifecycle(tmp_path):
    b = _board(tmp_path)
    w = b.add_wish("支持批量 STS", category="功能建议", client_version="2.2.2")
    assert w["status"] == "queued" and w["id"] == "w-1"
    b.mark_wish_sent(w["id"], ok=True, server_ack="fb-x")
    assert b.list_wishes()[0]["status"] == "sent"
    # failure path records the error, stays listed
    w2 = b.add_wish("第二个愿望")
    b.mark_wish_sent(w2["id"], ok=False, error="no-server")
    got = {x["id"]: x for x in b.list_wishes()}
    assert got[w2["id"]]["status"] == "failed" and "no-server" in got[w2["id"]]["error"]


def test_empty_wish_rejected(tmp_path):
    b = _board(tmp_path)
    assert b.add_wish("   ").get("error")
    assert b.list_wishes() == []


def test_agent_request_lifecycle(tmp_path):
    b = _board(tmp_path)
    r = b.post_agent_request("literature", "[文献] 请上传 X 的全文", kind="upload")
    assert r["status"] == "pending" and b.pending_request_count() == 1
    # dedup: identical open request returns the same record
    r2 = b.post_agent_request("literature", "[文献] 请上传 X 的全文", kind="upload")
    assert r2["id"] == r["id"] and b.pending_request_count() == 1
    b.resolve_agent_request(r["id"], "done", note="已上传")
    assert b.pending_request_count() == 0
    assert b.list_agent_requests()[0]["status"] == "done"


def test_request_invalid_status(tmp_path):
    b = _board(tmp_path)
    r = b.post_agent_request("instrument", "[仪器] 请进针")
    assert b.resolve_agent_request(r["id"], "bogus").get("error")


def test_persistence_roundtrip(tmp_path):
    b = _board(tmp_path)
    b.add_wish("persist me")
    b.post_agent_request("data_processing", "[数据] 请确认")
    # new board over the same dir re-reads from disk
    b2 = _board(tmp_path)
    assert len(b2.list_wishes()) == 1
    assert len(b2.list_agent_requests()) == 1


def test_request_tool_posts_to_board(tmp_path, monkeypatch):
    import mast.wishlist.store as S

    # Was `monkeypatch.setattr(S, "_DEFAULT_DIR", tmp_path)` — patching a
    # module-level CONSTANT. That worked, but it was the only isolation that did:
    # every OTHER wishlist test used `monkeypatch.setenv("MAST2_PROJECT_ROOT", …)`
    # and the constant ignored the env entirely, so those tests silently wrote into
    # the operator's real board. The store now resolves lazily through
    # project_root(), so the env override is real — use it, like everywhere else.
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    S.reset_default_board()
    from mast.agents._shared.request_tools import make_request_tools
    tool = make_request_tools(lambda: {"agent_id": "instrument", "experiment_id": None})[0]
    out = tool.invoke({"message": "[仪器] 请手动进针", "kind": "action"})
    assert "已向用户发起请求" in out
    assert S.get_board().pending_request_count() == 1
    S.reset_default_board()


def test_render_helpers_smoke(tmp_path):
    from mast.webui.wishlist_panel import (
        render_wishes_html, render_requests_html, open_request_choices,
    )
    b = _board(tmp_path)
    b.add_wish("w")
    r = b.post_agent_request("literature", "req")
    assert "心愿单" not in render_wishes_html(b.list_wishes())  # no crash; has content
    assert "literature" in render_requests_html(b.list_agent_requests())
    choices = open_request_choices(b.list_agent_requests())
    assert choices and choices[0][1] == r["id"]
    assert render_wishes_html([]).strip()  # empty-state renders
    assert render_requests_html([]).strip()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
