"""The SI-attachment seam: browser upload → knowledge.attachments.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/api/test_literature_si.py -q
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
from fastapi.testclient import TestClient

from mast.api.app import create_app


def _client() -> TestClient:
    return TestClient(create_app(dev_cors=False))


@pytest.fixture
def paper(tmp_path, monkeypatch):
    """An ingested paper the supplement can hang off."""
    papers = tmp_path / "papers"
    (papers / "W1").mkdir(parents=True)
    (papers / "W1" / "source.pdf").write_bytes(b"%PDF-1.4 main")
    monkeypatch.setenv("MAST_PAPERS_DIR", str(papers))
    return papers / "W1"


def _upload(work_id="W1", name="si.pdf", body=b"%PDF-1.4 si", **data):
    return _client().post(
        "/api/literature/attach-si",
        files={"file": (name, body, "application/pdf")},
        data={"work_id": work_id, **data})


def test_attach_si_stores_the_file(paper, monkeypatch):
    from mast.knowledge import attachments as att
    monkeypatch.setattr(att, "_extract_text",
                        lambda p, **kw: ("Supplementary Note 3. bias -0.8 V", False))

    r = _upload(label="Supplementary Note 3")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["slug"] == "W1"
    assert body["file"].endswith(".pdf") and body["n_chars"] > 0
    assert (paper / "attachments" / body["file"]).is_file()


def test_attach_si_rejects_non_pdf(paper):
    r = _upload(name="notes.txt", body=b"hello")
    assert r.status_code == 200
    assert r.json()["ok"] is False and "PDF" in r.json()["error"]


def test_attach_si_requires_work_id(paper):
    r = _upload(work_id="")
    assert r.status_code == 200 and r.json()["ok"] is False


def test_attach_si_before_the_paper_exists_explains_why(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_PAPERS_DIR", str(tmp_path / "papers"))
    r = _upload(work_id="W_nothing")
    assert r.status_code == 200
    assert r.json()["ok"] is False and "正文" in r.json()["error"]


def test_attach_si_reports_a_duplicate_without_a_second_copy(paper, monkeypatch):
    from mast.knowledge import attachments as att
    monkeypatch.setattr(att, "_extract_text", lambda p, **kw: ("text here", False))
    _upload()
    r = _upload()
    assert r.json()["duplicate"] is True
    assert len(list((paper / "attachments").glob("*.pdf"))) == 1


def test_attach_si_degrades_when_backend_missing(paper, monkeypatch):
    """A dead backend is a typed degraded body, never a 500."""
    import mast.knowledge as pkg
    # A None entry in sys.modules is how the import machinery reports "this
    # module is unavailable"; patching __import__ would miss the call, because
    # `from mast.knowledge import attachments` imports the PACKAGE and then
    # takes an attribute off it.
    monkeypatch.setitem(sys.modules, "mast.knowledge.attachments", None)
    monkeypatch.delattr(pkg, "attachments", raising=False)

    r = _upload()
    assert r.status_code == 200
    assert r.json()["degraded"] is True and r.json()["ok"] is False


def test_list_attachments_round_trip(paper, monkeypatch):
    from mast.knowledge import attachments as att
    monkeypatch.setattr(att, "_extract_text", lambda p, **kw: ("body", False))
    _upload(label="SI-1")

    r = _client().get("/api/literature/attachments", params={"work_id": "W1"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False and body["slug"] == "W1"
    assert len(body["attachments"]) == 1
    assert body["attachments"][0]["label"] == "SI-1"
    assert body["attachments"][0]["registered"] is True


def test_list_attachments_empty_is_not_degraded(paper):
    r = _client().get("/api/literature/attachments", params={"work_id": "W1"})
    assert r.json() == {"attachments": [], "slug": "W1", "degraded": False}


def test_list_attachments_requires_work_id(paper):
    assert _client().get("/api/literature/attachments").status_code == 422


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
