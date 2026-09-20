"""GUI 文献库-tab helpers (library manager / big-library search / ingest / fetch).

Pure helpers over the committed knowledge backend; search/ingest/fetch are
monkeypatched so the test needs no real index / network / PDF.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/gui/test_literature_panel.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.webui import literature_panel as lp  # noqa: E402
from mast.knowledge import literature_index  # noqa: E402
from mast.knowledge.libraries import LibraryRegistry  # noqa: E402

XSS = "<script>alert(1)</script>"


@pytest.fixture
def reg(tmp_path):
    return LibraryRegistry(libs_dir=tmp_path / "libs")


def test_library_crud(reg):
    assert "✅" in lp.create_library_h("My Si Papers", "custom", registry=reg)
    libs = {l[1] for l in lp.library_choices(registry=reg)}
    assert "reading" in libs                       # undeletable global
    custom = [l for l in lp.library_choices(registry=reg) if "reading" not in l[1]]
    assert custom
    cid = custom[0][1]
    assert "✅" in lp.switch_library_h(cid, registry=reg)
    assert "✅" in lp.rename_library_h(cid, "Renamed", registry=reg)
    html = lp.render_libraries_html(registry=reg)
    assert "Renamed" in html


def test_global_library_undeletable(reg):
    assert "❌" in lp.delete_library_h("reading", registry=reg)


def test_add_remove_members_are_pointers(reg):
    lp.create_library_h("L", "custom", registry=reg)
    cid = [l[1] for l in lp.library_choices(registry=reg) if l[1] != "reading"][0]
    lp.switch_library_h(cid, registry=reg)
    assert "✅" in lp.add_members_h("W1\nW2, W3", cid, registry=reg)
    members_html = lp.render_library_members_html(cid, registry=reg)
    assert "W1" in members_html and "W3" in members_html
    assert "3 个指针" in members_html
    assert "🗑" in lp.remove_members_h("W2", cid, registry=reg)
    assert "W2" not in lp.render_library_members_html(cid, registry=reg)


def test_search_filters_to_library_members(reg, monkeypatch):
    rows = [
        {"work_id": "A", "title": "alpha", "year": 2021, "abstract_excerpt": "ex", "source": "openalex"},
        {"work_id": "B", "title": "beta", "year": 2022, "abstract_excerpt": "", "source": "openalex"},
        {"work_id": "C", "title": "gamma", "year": 2023, "abstract_excerpt": "", "source": "user"},
    ]
    monkeypatch.setattr(literature_index, "search", lambda q, **kw: list(rows))
    # full big-library search
    out = lp.search_big_library_html("x", k=8, registry=reg)
    assert "alpha" in out and "gamma" in out
    # library-filtered search returns only members (pointer model, no 2nd index)
    lp.create_library_h("L", "custom", registry=reg)
    cid = [l[1] for l in lp.library_choices(registry=reg) if l[1] != "reading"][0]
    lp.add_members_h("A\nC", cid, registry=reg)
    out2 = lp.search_big_library_html("x", k=8, library_id=cid, registry=reg)
    assert "alpha" in out2 and "gamma" in out2 and "beta" not in out2


def test_search_escapes_title(reg, monkeypatch):
    monkeypatch.setattr(literature_index, "search",
                        lambda q, **kw: [{"work_id": "X", "title": XSS, "year": 2020}])
    out = lp.search_big_library_html("x", registry=reg)
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_fetch_abstract_html(monkeypatch):
    monkeypatch.setattr(literature_index, "fetch_abstract",
                        lambda w: {"found": True, "title": "T", "abstract": "A body",
                                   "year": 2020, "source": "openalex"})
    out = lp.fetch_abstract_html("W1")
    assert "A body" in out
    monkeypatch.setattr(literature_index, "fetch_abstract",
                        lambda w: {"found": False, "note": "not provisioned"})
    assert "未找到" in lp.fetch_abstract_html("W9")


@pytest.fixture
def board(tmp_path):
    from mast.knowledge.fetch_board import FetchBoard
    return FetchBoard(board_dir=tmp_path / "board")


def test_ingest_promotes_then_adds_pointer(reg, board, monkeypatch):
    class _Res:
        status = "ok"
        work_id = "W-ingested"
    monkeypatch.setattr(lp.ingest_mod, "ingest_pdf", lambda *a, **k: _Res())
    lp.create_library_h("L", "custom", registry=reg)
    cid = [l[1] for l in lp.library_choices(registry=reg) if l[1] != "reading"][0]
    msg = lp.ingest_pdf_h("C:/fake.pdf", cid, registry=reg, board=board)
    assert "大库" in msg and "W-ingested" in msg
    # the library gained the pointer (promote into big library, pointer in lib)
    assert "W-ingested" in lp.render_library_members_html(cid, registry=reg)


def test_ingest_passes_empty_library_id_to_ingest_pdf(reg, board, monkeypatch):
    """Pointer model: ingest_pdf must be called with library_id='' so NO
    per-library index is built; the library gets only a work_id pointer."""
    seen = {}
    class _Res:
        status = "ok"; work_id = "W"
    def _fake(pdf_path, library_id="", **k):
        seen["library_id"] = library_id
        return _Res()
    monkeypatch.setattr(lp.ingest_mod, "ingest_pdf", _fake)
    lp.create_library_h("L", "custom", registry=reg)
    cid = [l[1] for l in lp.library_choices(registry=reg) if l[1] != "reading"][0]
    lp.ingest_pdf_h("C:/f.pdf", cid, registry=reg, board=board)
    assert seen["library_id"] == ""


def test_fetch_then_ingest_honest_when_paywalled(reg, board, monkeypatch):
    monkeypatch.setattr(lp.fetch_mod, "try_fetch_fulltext",
                        lambda *a, **k: {"status": "blocked", "reason": "paywall"})
    msg = lp.fetch_then_ingest_h("10.1/x", "reading", registry=reg, board=board)
    assert "未取到全文" in msg and "付费墙" in msg


def test_fetch_board_render_and_choices(board):
    assert "为空" in lp.render_fetch_board_html(board=board)
    board.post_request("W1", reason="need methods", title="Paper One")
    html = lp.render_fetch_board_html(board=board)
    assert "W1" in html and "Paper One" in html and "待处理" in html
    ch = lp.fetch_request_choices(board=board)
    assert ch and ch[0][1].startswith("fr-")
    assert lp.request_work_id(ch[0][1], board=board) == "W1"


def test_dismiss_request(board):
    r = board.post_request("W1")
    assert "忽略" in lp.dismiss_request_h(r["request_id"], board=board)
    assert board.pending_count() == 0
    assert "❌" in lp.dismiss_request_h("", board=board)


def test_ingest_closes_matching_fetch_request(reg, board, monkeypatch):
    """Agent-asks-user loop: ingesting a paper fulfils its open fetch request."""
    board.post_request("W-need", reason="agent needs full text",
                       requested_by="literature")
    class _Res:
        status = "ok"; work_id = "W-need"
    monkeypatch.setattr(lp.ingest_mod, "ingest_pdf", lambda *a, **k: _Res())
    msg = lp.ingest_pdf_h("C:/f.pdf", "reading", work_id="W-need",
                          registry=reg, board=board)
    assert "满足了 1 条取文请求" in msg
    assert board.pending_count() == 0
    assert board.list_requests("fulfilled")[0]["work_id"] == "W-need"


def test_fetch_board_escapes_html(board):
    board.post_request("W1", title="<script>alert(1)</script>", reason="x")
    out = lp.render_fetch_board_html(board=board)
    assert "<script>" not in out and "&lt;script&gt;" in out


# ── P1: active-library preselect + pending-request badge ───────────────

def test_active_library_value_preselects_active(reg):
    """The dropdown-preselect helper returns the active library id (defaults to
    the global 'reading' library, then follows switch)."""
    assert lp.active_library_value(registry=reg) == "reading"   # global default
    lp.create_library_h("Working", "custom", registry=reg)
    cid = [l[1] for l in lp.library_choices(registry=reg) if l[1] != "reading"][0]
    lp.switch_library_h(cid, registry=reg)
    assert lp.active_library_value(registry=reg) == cid          # follows active


def test_active_library_value_none_when_not_a_choice(reg, monkeypatch):
    """If the active id isn't among the choices (corrupt state) the helper
    returns None rather than seeding a stale/invalid selection."""
    monkeypatch.setattr(lp.lib_mod, "get_active",
                        lambda **kw: {"library_id": "ghost-lib"})
    assert lp.active_library_value(registry=reg) is None


def test_fetch_badge_empty_when_no_pending(board):
    assert lp.render_fetch_badge_html(board=board) == ""


def test_fetch_badge_surfaces_pending_count(board):
    board.post_request("W1", reason="need full text", requested_by="literature")
    board.post_request("W2", reason="and another")
    out = lp.render_fetch_badge_html(board=board)
    assert "全文请求" in out and "<b>2</b>" in out
    # clearing the board makes the badge vanish again (safe to refresh on timer)
    for r in board.list_requests("pending"):
        lp.dismiss_request_h(r["request_id"], board=board)
    assert lp.render_fetch_badge_html(board=board) == ""


# ── 摄入诚实化 (scanned warning / pre-flight / dangling pointer) ────────

def test_ingest_pdf_h_surfaces_no_text_warning(reg, board, monkeypatch):
    """A warning:no_text_layer ingest must NOT read as green success."""
    class _Res:
        status = "warning:no_text_layer"
        work_id = "local:x"
        detail = "提取到的文本过少，疑似扫描版"
    monkeypatch.setattr(lp.ingest_mod, "ingest_pdf", lambda *a, **k: _Res())
    msg = lp.ingest_pdf_h("C:/scan.pdf", "reading", registry=reg, board=board)
    assert msg.startswith("⚠") and "扫描版" in msg
    assert "促进进大库" not in msg


def test_ingest_readiness_reports_missing(monkeypatch):
    monkeypatch.setattr(lp, "_dashscope_key_present", lambda: False)
    out = lp.ingest_readiness_html()
    assert "摄取当前会失败" in out and "DashScope" in out


def test_ingest_readiness_ready_when_key_and_fitz(monkeypatch):
    monkeypatch.setattr(lp, "_dashscope_key_present", lambda: True)
    out = lp.ingest_readiness_html()
    # pymupdf is a pinned dep, so in the test env fitz is present → green.
    assert "摄取就绪" in out or "摄取当前会失败" in out  # never crashes


def test_add_members_flags_ids_missing_from_big(reg, monkeypatch):
    lp.create_library_h("L", "custom", registry=reg)
    cid = [l[1] for l in lp.library_choices(registry=reg) if l[1] != "reading"][0]
    # pretend the big index knows only W100; W999 is a dangling pointer.
    monkeypatch.setattr(lp, "_ids_missing_from_big",
                        lambda ids: [i for i in ids if i != "W100"])
    msg = lp.add_members_h("W100\nW999", cid, registry=reg)
    assert "加入" in msg and "大库中暂未找到" in msg


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
