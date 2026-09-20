"""The seam that wakes a conversation when its requested paper arrives.

``core/fetch_resume`` is deliberately dumb: a process-wide registry plus a text
builder. The runtime registers the thing that can actually drive a conversation;
everything below it (``knowledge.fulfilment``) just calls ``notify_fulfilled``
and carries on regardless of the answer.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_fetch_resume.py -q
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

from mast.core import fetch_resume


@pytest.fixture(autouse=True)
def _clear_resumer():
    """The resumer is a process-wide singleton — exactly the shape that leaks.

    A resumer left registered by one test would let it drive the next test's
    fake engine, which is how "my test passes alone but not in the suite" starts.
    """
    fetch_resume.clear_resumer()
    yield
    fetch_resume.clear_resumer()


def _req(**kw):
    base = {"request_id": "fr-1", "work_id": "W1", "title": "",
            "reason": "", "origin_conversation_id": "c1"}
    base.update(kw)
    return base


# ── registry ─────────────────────────────────────────────────────────────

def test_no_resumer_is_a_noop():
    out = fetch_resume.notify_fulfilled("W1", [_req()])
    assert out["resumed"] == 0 and "no resumer" in out["detail"]


def test_registered_resumer_receives_arguments():
    seen = {}

    def _fake(work_id, requests, *, kind="fetch", exclude_conversation_id=""):
        seen.update(work_id=work_id, requests=requests, kind=kind,
                    exclude=exclude_conversation_id)
        return {"resumed": 1}

    fetch_resume.set_resumer(_fake)
    out = fetch_resume.notify_fulfilled(
        "W7", [_req(work_id="W7")], exclude_conversation_id="c9")
    assert out["resumed"] == 1
    assert seen["work_id"] == "W7" and seen["exclude"] == "c9"
    assert seen["kind"] == "fetch"
    assert seen["requests"][0]["request_id"] == "fr-1"


def test_wishlist_answers_use_the_same_seam():
    """One reason to wake somebody up, whichever board the answer landed on."""
    seen = {}

    def _fake(work_id, requests, *, kind="fetch", exclude_conversation_id=""):
        seen.update(work_id=work_id, kind=kind, n=len(requests))
        return {"resumed": 1}

    fetch_resume.set_resumer(_fake)
    out = fetch_resume.notify_request_answered(
        [{"id": "r-4", "status": "done", "path": "D:/x.sxm"}])
    assert out["resumed"] == 1
    assert seen == {"work_id": "", "kind": "request", "n": 1}


def test_answer_instruction_hands_over_the_path_verbatim():
    """Storing the path as a field is pointless if the agent must parse it back out."""
    txt = fetch_resume.build_answer_instruction([
        {"id": "r-4", "status": "done", "message": "请给我上次那张图的路径",
         "path": "D:/data/2026-08-01/scan_012.sxm", "note": "在 D 盘"}])
    assert "r-4" in txt and "请给我上次那张图的路径" in txt
    assert "D:/data/2026-08-01/scan_012.sxm" in txt
    assert "在 D 盘" in txt
    assert "不要就此展开新的工作" in txt


def test_answer_instruction_marks_a_dismissal_as_dropped():
    txt = fetch_resume.build_answer_instruction([
        {"id": "r-5", "status": "dismissed", "message": "请换样品"}])
    assert "已忽略" in txt and "就不要再做了" in txt


def test_resumer_exception_is_swallowed():
    """A failed wake-up must never turn a completed ingest into an error."""
    def _boom(*_a, **_k):
        raise RuntimeError("engine exploded")

    fetch_resume.set_resumer(_boom)
    out = fetch_resume.notify_fulfilled("W1", [_req()])
    assert out["resumed"] == 0 and "resumer failed" in out["detail"]


def test_non_dict_return_is_tolerated():
    fetch_resume.set_resumer(lambda *a, **k: "surprise")
    assert fetch_resume.notify_fulfilled("W1", [_req()])["resumed"] == 0


def test_empty_request_list_short_circuits():
    called = []
    fetch_resume.set_resumer(lambda *a, **k: called.append(1) or {"resumed": 1})
    assert fetch_resume.notify_fulfilled("W1", [])["resumed"] == 0
    assert not called


def test_set_and_clear_are_idempotent():
    assert not fetch_resume.has_resumer()
    fetch_resume.set_resumer(lambda *a, **k: {"resumed": 0})
    fetch_resume.set_resumer(lambda *a, **k: {"resumed": 0})
    assert fetch_resume.has_resumer()
    fetch_resume.clear_resumer()
    fetch_resume.clear_resumer()
    assert not fetch_resume.has_resumer()


# ── instruction text ─────────────────────────────────────────────────────

def test_instruction_carries_identity_and_original_reason():
    txt = fetch_resume.build_resume_instruction([
        _req(request_id="fr-3", work_id="W123", title="Au(111) herringbone",
             reason="需要 methods 里的偏压参数")])
    assert "fr-3" in txt and "W123" in txt
    assert "Au(111) herringbone" in txt
    assert "需要 methods 里的偏压参数" in txt


def test_instruction_aggregates_several_papers_into_one_turn():
    txt = fetch_resume.build_resume_instruction([
        _req(request_id="fr-1", work_id="W1"),
        _req(request_id="fr-2", work_id="W2"),
    ])
    assert "fr-1" in txt and "fr-2" in txt
    assert txt.count("【取文请求已满足") == 1


def test_instruction_does_not_authorise_a_new_survey():
    """Being handed a paper resumes the blocked task; it is not a fresh mandate."""
    txt = fetch_resume.build_resume_instruction([_req()])
    assert "不要就此展开新的调研" in txt
    assert "继续当时因为缺全文而搁置的那件事" in txt


def test_instruction_survives_sparse_rows():
    txt = fetch_resume.build_resume_instruction([{}])
    assert "【取文请求已满足" in txt  # no crash on a row with nothing in it


def test_unreadable_upload_is_announced_as_unreadable():
    """A scanned PDF ingests fine and is readable by nobody.

    Told "the full text has arrived", the agent works through every reading tool
    hunting for text that is not there and stops only at the recursion limit —
    seen on a live run before this branch existed.
    """
    txt = fetch_resume.build_resume_instruction([
        _req(request_id="fr-9", work_id="W9") | {"fulltext_readable": False}])
    assert "没有可读的文本层" in txt
    assert "不要反复换工具去试" in txt
    assert "read_paper_section" not in txt   # do not point at a tool that cannot work


def test_readable_and_unreadable_are_reported_separately():
    txt = fetch_resume.build_resume_instruction([
        _req(request_id="fr-1", work_id="W1") | {"fulltext_readable": True},
        _req(request_id="fr-2", work_id="W2") | {"fulltext_readable": False},
    ])
    good, bad = txt.index("全文已入库"), txt.index("没有可读的文本层")
    assert txt.index("fr-1") < bad and txt.index("fr-2") > good
    assert "read_paper_section" in txt   # still offered, for the readable one


def test_missing_flag_defaults_to_readable():
    """Old rows (and the manual-entry path) carry no flag; assume readable."""
    assert "全文已入库" in fetch_resume.build_resume_instruction([_req()])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
