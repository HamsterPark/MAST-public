"""``load_document`` / ``list_documents`` — the readers the artifact channel needs.

These tools were added on 2026-07-30 to close a defect: the artifact channel had
shipped the day before telling every document consumer, in its own system prompt,
to call ``load_document(doc_id)`` — and that tool existed nowhere in the tree.

Every test here calls the tool for real. That is the standing lesson from the
2026-07-29 docx round: a tool was added to ``AGENT_TOOLS``, the renderer and the
HTTP endpoint were both green, and the module could not be imported at all
because of a stray newline inside an f-string. Neither green suite touched that
import. **A new agent tool needs a test that actually invokes it.**
"""
from __future__ import annotations

# ── path bootstrap ──────────────────────────────────────────────────────────
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        c = p / "MASTv2"
        if c.is_dir():
            return str(c)
        p = p.parent
    raise RuntimeError("MASTv2 not found above " + str(Path(__file__).resolve()))


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.agents._shared.document_tools import (  # noqa: E402
    DOCUMENT_TOOLS,
    list_documents,
    load_document,
)


def _save(text: str, *, kind: str = "literature_report", title: str = "T",
          doc_id: str = "") -> object:
    from mast.documents import store
    res = store().save(text=text, kind=kind, title=title, doc_id=doc_id,
                       created_by="test")
    assert res.ok, getattr(res, "error", "save failed")
    return res


class TestLoadDocument:
    def test_reads_back_what_was_saved(self, documents_root):
        res = _save("# 综述\n偏压取 -1.5~+1.5 V。", title="NiI2 综述")
        out = load_document.invoke({"doc_id": res.doc_id})
        assert "偏压取 -1.5~+1.5 V" in out
        assert res.doc_id in out
        assert "NiI2 综述" in out
        assert "literature_report" in out

    def test_a_specific_version_is_readable(self, documents_root):
        r1 = _save("第一版正文", title="报告")
        _save("第二版正文", title="报告", doc_id=r1.doc_id)
        latest = load_document.invoke({"doc_id": r1.doc_id})
        assert "第二版正文" in latest
        first = load_document.invoke({"doc_id": r1.doc_id, "version": 1})
        assert "第一版正文" in first
        assert "第二版正文" not in first

    def test_an_unknown_id_is_honest_and_does_not_invent(self, documents_root):
        out = load_document.invoke({"doc_id": "01NOSUCHDOCIDATALL000000"})
        assert "找不到" in out
        assert "list_documents" in out, "a dead end must name the way out"

    def test_an_empty_id_explains_where_ids_come_from(self, documents_root):
        out = load_document.invoke({"doc_id": "  "})
        assert "doc_id" in out and "list_documents" in out

    def test_a_long_body_is_truncated_and_says_so(self, documents_root):
        res = _save("x" * 9000, title="长报告")
        out = load_document.invoke({"doc_id": res.doc_id})
        assert "截断" in out, "silent truncation would misrepresent the document"
        assert len(out) < 9000

    def test_it_never_raises_at_the_model(self, monkeypatch, documents_root):
        """A tool that raises kills the turn; it must return a string instead."""
        import mast.documents as _documents

        def _boom():
            raise RuntimeError("store exploded")

        monkeypatch.setattr(_documents, "store", _boom)
        out = load_document.invoke({"doc_id": "whatever"})
        assert isinstance(out, str) and "failed" in out


class TestListDocuments:
    def test_lists_what_exists_with_ids_that_load(self, documents_root):
        a = _save("正文 A", title="报告 A")
        b = _save("正文 B", kind="review", title="评审 B")
        out = list_documents.invoke({})
        assert a.doc_id in out and b.doc_id in out
        assert "报告 A" in out and "评审 B" in out
        # the ids it hands out must actually open
        assert "正文 A" in load_document.invoke({"doc_id": a.doc_id})

    def test_kind_filter_narrows(self, documents_root):
        a = _save("正文 A", title="报告 A")
        b = _save("正文 B", kind="review", title="评审 B")
        out = list_documents.invoke({"kind": "review"})
        assert b.doc_id in out
        assert a.doc_id not in out

    def test_empty_says_confirmed_empty_not_maybe(self, documents_root):
        """"There are no documents" and "the query failed" are different facts and
        must not read alike — the same rule ``class_status`` follows with its
        ``known`` flag."""
        out = list_documents.invoke({})
        assert "没有" in out
        assert "不是查询失败" in out

    def test_a_read_failure_does_not_claim_emptiness(self, monkeypatch, documents_root):
        import mast.documents as _documents

        def _boom():
            raise RuntimeError("nope")

        monkeypatch.setattr(_documents, "store", _boom)
        out = list_documents.invoke({})
        assert "读不到" in out
        assert "这不等于" in out, "a failed query must not be reported as 'no documents'"


class TestWiring:
    def test_both_tools_are_exported_for_every_agent(self):
        names = {t.name for t in DOCUMENT_TOOLS}
        assert names == {"load_document", "list_documents"}

    @pytest.mark.parametrize("attach_point", [
        "group",     # runtime._build_orchestrator_impl → memory_tools
        "private",   # runtime._chat_agent_tools
    ])
    def test_runtime_attaches_them_on_both_chat_paths(self, attach_point):
        """Both paths matter: a private-chat agent is shown the same artifact block
        (including its OWN last product's doc_id) so it needs the same reader."""
        import inspect

        import mast.core.runtime as rt
        src = inspect.getsource(rt)
        assert src.count("from mast.agents._shared.document_tools import DOCUMENT_TOOLS") >= 2, (
            "document tools are not attached on both the group and private paths")
