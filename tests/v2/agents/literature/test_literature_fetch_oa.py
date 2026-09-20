"""The literature agent fetching an open-access full text by itself.

Before this the agent could only ASK for a full text (``request_fulltext`` →ain
operator). The open-access machinery existed but only the GUI could trigger it,
so a paper with a freely downloadable PDF still went on a human's to-do list.

Every test here injects: no network, no real index, no real registry.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/literature/test_literature_fetch_oa.py -q
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

import types

import pytest

from mast.agents.literature import tools as littools
from mast.agents.literature.tools import fetch_fulltext_oa


class _IngestResult:
    def __init__(self, status="ingested", work_id="W1", title="A paper",
                 n_chunks=7, slug_dir="/papers/w1_slug", detail=""):
        self.status = status
        self.work_id = work_id
        self.title = title
        self.n_chunks = n_chunks
        self.slug_dir = slug_dir
        self.detail = detail


def _patch_fetch(monkeypatch, result: dict, seen: dict | None = None):
    mod = types.ModuleType("mast.knowledge.fetch")

    def _try(doi_or_url, **kw):
        if seen is not None:
            seen.update(doi=doi_or_url, kw=kw)
        return result

    mod.try_fetch_fulltext = _try  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mast.knowledge.fetch", mod)
    import mast.knowledge as pkg
    monkeypatch.setattr(pkg, "fetch", mod, raising=False)


def _patch_ingest(monkeypatch, result, seen: dict | None = None):
    import mast.knowledge.ingest as ing

    def _ingest(pdf_path, **kw):
        if seen is not None:
            seen.update(pdf_path=pdf_path, kw=kw)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(ing, "ingest_pdf", _ingest)


def _patch_curate(monkeypatch, out: dict, seen: dict | None = None):
    import mast.knowledge.fulfilment as ful

    def _curate(work_id, library_id="", **kw):
        if seen is not None:
            seen.update(work_id=work_id, kw=kw)
        return out

    monkeypatch.setattr(ful, "curate_ingested_fulltext", _curate)


# ── the happy path ───────────────────────────────────────────────────────

def test_success_ingests_curates_and_points_at_the_next_tool(monkeypatch):
    fetch_seen, ingest_seen, curate_seen = {}, {}, {}
    _patch_fetch(monkeypatch, {"status": "ok_pdf", "pdf_path": "/tmp/p.pdf"},
                 fetch_seen)
    _patch_ingest(monkeypatch, _IngestResult(), ingest_seen)
    _patch_curate(monkeypatch, {"pointer_library_id": "exp_lib",
                                "pointer_library_source": "experiment",
                                "fulfilled_requests": 2}, curate_seen)

    out = fetch_fulltext_oa.invoke(
        {"doi_or_url": "10.1103/PhysRevLett.1", "work_id": "W1"})

    assert "已通过开源渠道获取全文并入库" in out
    assert "W1" in out and "A paper" in out and "7" in out
    assert "exp_lib" in out
    assert "关闭了取文板上 2 条请求" in out
    assert "read_paper_section" in out          # tells the agent what to do next

    # The OA resolver must be switched on, and the fetch must be time-bounded.
    assert fetch_seen["kw"]["auto_oa"] is True
    assert fetch_seen["kw"]["timeout"] <= 30
    # Ingest binds the identity we already know, and files under the agent's name.
    assert ingest_seen["kw"]["work_id"] == "W1"
    assert ingest_seen["kw"]["source"] == "agent"
    assert curate_seen["kw"]["added_by"] == "agent"
    assert curate_seen["kw"]["slug"] == "w1_slug"


def test_success_without_prior_work_id_uses_the_resolved_one(monkeypatch):
    _patch_fetch(monkeypatch, {"status": "ok_pdf", "pdf_path": "/tmp/p.pdf"})
    _patch_ingest(monkeypatch, _IngestResult(work_id="local:abc123"))
    seen = {}
    _patch_curate(monkeypatch, {}, seen)
    out = fetch_fulltext_oa.invoke({"doi_or_url": "10.1/x"})
    assert "local:abc123" in out and seen["work_id"] == "local:abc123"


def test_curation_failure_does_not_lose_the_paper(monkeypatch):
    """The PDF is ingested; a registry hiccup must not read as a failed fetch."""
    _patch_fetch(monkeypatch, {"status": "ok_pdf", "pdf_path": "/tmp/p.pdf"})
    _patch_ingest(monkeypatch, _IngestResult())
    import mast.knowledge.fulfilment as ful

    def _boom(*_a, **_k):
        raise RuntimeError("registry down")

    monkeypatch.setattr(ful, "curate_ingested_fulltext", _boom)
    out = fetch_fulltext_oa.invoke({"doi_or_url": "10.1/x", "work_id": "W1"})
    assert "已通过开源渠道获取全文并入库" in out


# ── downloaded but not ingestable ────────────────────────────────────────

def test_ingest_error_reports_where_the_pdf_landed(monkeypatch):
    _patch_fetch(monkeypatch, {"status": "ok_pdf", "pdf_path": "/tmp/here.pdf"})
    _patch_ingest(monkeypatch, _IngestResult(
        status="error:numpy_or_pandas_missing", detail="numpy 缺失"))
    out = fetch_fulltext_oa.invoke({"doi_or_url": "10.1/x", "work_id": "W1"})
    assert "/tmp/here.pdf" in out and "numpy 缺失" in out
    assert "继续用摘要" in out


def test_ingest_raising_is_caught(monkeypatch):
    _patch_fetch(monkeypatch, {"status": "ok_pdf", "pdf_path": "/tmp/here.pdf"})
    _patch_ingest(monkeypatch, RuntimeError("boom"))
    out = fetch_fulltext_oa.invoke({"doi_or_url": "10.1/x"})
    assert "/tmp/here.pdf" in out and "boom" in out


# ── the honest refusals ──────────────────────────────────────────────────

@pytest.mark.parametrize("status", ["ok_metadata", "unavailable"])
def test_no_oa_copy_hands_off_to_request_fulltext(monkeypatch, status):
    _patch_fetch(monkeypatch, {"status": status, "message": "仅有落地页"})
    out = fetch_fulltext_oa.invoke({"doi_or_url": "10.1/x", "work_id": "W9"})
    assert "request_fulltext" in out and "W9" in out
    assert "继续用摘要" in out


def test_paywalled_says_it_will_not_bypass(monkeypatch):
    _patch_fetch(monkeypatch, {"status": "blocked", "message": "付费墙"})
    out = fetch_fulltext_oa.invoke({"doi_or_url": "10.1/x", "work_id": "W9"})
    assert "不绕过" in out and "request_fulltext" in out


def test_network_error_says_do_not_retry(monkeypatch):
    _patch_fetch(monkeypatch, {"status": "error", "message": "连接超时"})
    out = fetch_fulltext_oa.invoke({"doi_or_url": "10.1/x"})
    assert "连接超时" in out and "不要反复重试" in out


def test_empty_input_asks_for_a_doi():
    out = fetch_fulltext_oa.invoke({"doi_or_url": "  "})
    assert "需要 doi_or_url" in out


def test_fetch_module_raising_never_escapes(monkeypatch):
    mod = types.ModuleType("mast.knowledge.fetch")

    def _boom(*_a, **_k):
        raise RuntimeError("kaput")

    mod.try_fetch_fulltext = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mast.knowledge.fetch", mod)
    out = fetch_fulltext_oa.invoke({"doi_or_url": "10.1/x"})
    assert out.startswith("fetch_fulltext_oa failed:") and "kaput" in out


# ── availability gating ──────────────────────────────────────────────────

def test_missing_httpx_marks_the_tool_unavailable(monkeypatch):
    from mast.knowledge import fetch as fetch_mod
    monkeypatch.setattr(fetch_mod, "httpx", None, raising=False)
    assert "fetch_fulltext_oa" in littools.tool_availability()


def test_placeholder_accepts_the_real_call_shape():
    """A placeholder that rejects the arguments restarts the retry loop it ends."""
    ph = littools._make_unavailable_tool("fetch_fulltext_oa", "httpx 不可用")
    out = ph.invoke({"doi_or_url": "10.1/x", "work_id": "W1"})
    assert "当前不可用" in out and "已知约束" in out


def test_tool_is_registered_and_gateable():
    names = [getattr(t, "name", "") for t in littools.build_tools(None)]
    assert "fetch_fulltext_oa" in names
    # In AGENT_TOOLS (not LIBRARY_TOOLS) so the placeholder swap can reach it.
    assert "fetch_fulltext_oa" in [getattr(t, "name", "") for t in littools.AGENT_TOOLS]
    swapped = littools.build_tools(None, unavailable={"fetch_fulltext_oa": "no httpx"})
    tool = next(t for t in swapped if getattr(t, "name", "") == "fetch_fulltext_oa")
    assert "不可用" in (tool.description or "")


# ── request_fulltext records who asked ───────────────────────────────────

def test_request_fulltext_records_the_asking_conversation(monkeypatch, tmp_path):
    from mast.knowledge import fetch_board as board_mod
    from mast.knowledge.fetch_board import FetchBoard
    from mast.core import turn_context

    board = FetchBoard(board_dir=tmp_path / "libs")
    monkeypatch.setattr(board_mod, "_BOARD", board, raising=False)

    turn_context.set_turn(conversation_id="conv-42", run_id="r1")
    try:
        littools.request_fulltext.invoke(
            {"work_id": "W3", "reason": "需要 methods 的偏压"})
    finally:
        turn_context.clear_turn()

    rec = board.open_requests("W3")[0]
    assert rec["origin_conversation_id"] == "conv-42"
    assert rec["reason"] == "需要 methods 的偏压"


def test_request_fulltext_without_a_turn_still_posts(monkeypatch, tmp_path):
    """A background run has no conversation — the ask still reaches the board."""
    from mast.knowledge import fetch_board as board_mod
    from mast.knowledge.fetch_board import FetchBoard
    from mast.core import turn_context

    board = FetchBoard(board_dir=tmp_path / "libs")
    monkeypatch.setattr(board_mod, "_BOARD", board, raising=False)
    turn_context.clear_turn()
    littools.request_fulltext.invoke({"work_id": "W4"})
    assert board.open_requests("W4")[0]["origin_conversation_id"] == ""


# ── prompt wiring ────────────────────────────────────────────────────────

def test_prompt_puts_self_fetch_before_asking():
    from mast.agents.literature.prompts import SYSTEM_PROMPT
    assert "fetch_fulltext_oa" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.index("fetch_fulltext_oa") < SYSTEM_PROMPT.index("request_fulltext")
    assert "先" in SYSTEM_PROMPT.split("fetch_fulltext_oa")[1][:120]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
