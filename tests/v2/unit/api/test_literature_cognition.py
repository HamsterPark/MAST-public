"""Wave-A parity rebuild — literature_cognition API contract tests.

Guards the typed seam that re-exposes the literature ingest/fetch/board controls
and the cognition phase-summaries list that the Gradio→TS SPA rewrite dropped
(core logic still lives in ``mast.knowledge.*`` / ``mast.memory.sharding``).

Two layers, per house style:
  * STANDALONE (bare AppContext, no live core) — every endpoint returns 200 with
    a valid empty/degraded body (``degraded=True`` / ``ok=False``), NEVER 500.
  * WIRED / monkeypatched — exercise the real relay against monkeypatched core
    fns (no DashScope / no real network / no 50k index) and a real PhaseManager
    on a temp SQLite DB.

The router is mounted on a throwaway app exactly as the brief prescribes.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.literature_cognition import router


# ── clients ───────────────────────────────────────────────────────────
def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx if ctx is not None else AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


# ── ingest: degraded + relay ──────────────────────────────────────────
def test_ingest_relays_result(monkeypatch) -> None:
    from mast.knowledge import ingest as ingest_mod

    class _Res:
        def to_dict(self):
            return {
                "work_id": "local:abc", "source": "user_pdf", "n_chunks": 4,
                "slug_dir": "/papers/x", "promoted": True, "status": "ingested",
                "doi": "10.x/1", "title": "T", "sha256": "deadbeef",
                "detail": "", "library_id": "",
            }

    def fake_ingest(pdf_path, **kw):
        assert pdf_path == "/tmp/x.pdf"
        # the panel/seam promotes (empty library_id → pointer model)
        assert kw.get("library_id") == ""
        return _Res()

    monkeypatch.setattr(ingest_mod, "ingest_pdf", fake_ingest)
    r = _client().post("/api/literature/ingest", json={"pdf_path": "/tmp/x.pdf"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["work_id"] == "local:abc"
    assert body["status"] == "ingested"
    assert body["promoted"] is True and body["n_chunks"] == 4


def test_ingest_relays_ocr_used(monkeypatch) -> None:
    """A scanned-PDF ingest recovered via OCR surfaces ``ocr_used`` in the body."""
    from mast.knowledge import ingest as ingest_mod

    class _Res:
        def to_dict(self):
            return {
                "work_id": "local:ocr", "source": "user_pdf", "n_chunks": 3,
                "slug_dir": "/papers/ocr", "promoted": True, "status": "ingested",
                "doi": "", "title": "Scanned", "sha256": "beef",
                "detail": "扫描件，已用 qwen-vl-ocr 提取文本", "library_id": "",
                "ocr_used": True,
            }

    monkeypatch.setattr(ingest_mod, "ingest_pdf", lambda *a, **k: _Res())
    r = _client().post("/api/literature/ingest", json={"pdf_path": "/tmp/scan.pdf"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["ocr_used"] is True


def test_ingest_error_status_is_not_ok(monkeypatch) -> None:
    from mast.knowledge import ingest as ingest_mod

    class _Res:
        def to_dict(self):
            return {"work_id": "", "source": "user_pdf", "n_chunks": 0,
                    "slug_dir": "", "promoted": False,
                    "status": "error:pdf_not_found", "detail": "nope",
                    "doi": "", "title": "", "sha256": "", "library_id": ""}

    monkeypatch.setattr(ingest_mod, "ingest_pdf", lambda *a, **k: _Res())
    r = _client().post("/api/literature/ingest", json={"pdf_path": "/no.pdf"})
    assert r.status_code == 200
    body = r.json()
    # the core degraded gracefully (error:* status) → ok False, NOT degraded/500
    assert body["ok"] is False and body["degraded"] is False
    assert body["status"] == "error:pdf_not_found"
    assert body["detail"] == "nope"


def test_ingest_core_raises_degrades(monkeypatch) -> None:
    from mast.knowledge import ingest as ingest_mod

    def boom(*a, **k):
        raise RuntimeError("explode")

    monkeypatch.setattr(ingest_mod, "ingest_pdf", boom)
    r = _client().post("/api/literature/ingest", json={"pdf_path": "/x.pdf"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True


def test_ingest_missing_path_422() -> None:
    # pdf_path is required → request validation (not a 500)
    r = _client().post("/api/literature/ingest", json={})
    assert r.status_code == 422


# ── fetch: degraded + relay ───────────────────────────────────────────
def test_fetch_relays_blocked(monkeypatch) -> None:
    from mast.knowledge import fetch as fetch_mod

    def fake_fetch(doi_or_url, *, oa_url="", auto_oa=True):
        assert auto_oa is True  # the panel passes auto_oa=True
        return {
            "status": "blocked", "pdf_path": None, "metadata": {"doi": "10.x/1"},
            "message": "付费墙", "url": "https://doi.org/10.x/1",
            "source": "user_url",
        }

    monkeypatch.setattr(fetch_mod, "try_fetch_fulltext", fake_fetch)
    r = _client().post("/api/literature/fetch", json={"doi_or_url": "10.x/1"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["status"] == "blocked"
    assert body["pdf_path"] is None
    assert body["metadata"]["doi"] == "10.x/1"


def test_fetch_relays_ok_pdf(monkeypatch) -> None:
    from mast.knowledge import fetch as fetch_mod

    monkeypatch.setattr(
        fetch_mod, "try_fetch_fulltext",
        lambda *a, **k: {"status": "ok_pdf", "pdf_path": "/papers/y/source.pdf",
                         "metadata": None, "message": "ok", "url": "u",
                         "source": "user_url"},
    )
    r = _client().post(
        "/api/literature/fetch",
        json={"doi_or_url": "https://doi.org/10.x/2", "auto_oa": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok_pdf"
    assert body["pdf_path"] == "/papers/y/source.pdf"


def test_fetch_core_raises_degrades(monkeypatch) -> None:
    from mast.knowledge import fetch as fetch_mod

    def boom(*a, **k):
        raise RuntimeError("net")

    monkeypatch.setattr(fetch_mod, "try_fetch_fulltext", boom)
    r = _client().post("/api/literature/fetch", json={"doi_or_url": "x"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "error" and body["degraded"] is True


# ── fetch-board ───────────────────────────────────────────────────────
def test_fetch_board_real_isolated(monkeypatch, tmp_path) -> None:
    """Use a real (isolated) FetchBoard so list/pending go through the kept fn."""
    from mast.knowledge import fetch_board as board_mod

    board = board_mod.FetchBoard(board_dir=str(tmp_path))
    board.post_request("W1", doi="10.x/1", title="P1", reason="need fulltext")
    board.post_request("W2", title="P2")

    monkeypatch.setattr(board_mod, "list_requests",
                        lambda status=None: board.list_requests(status))
    monkeypatch.setattr(board_mod, "pending_count", lambda: board.pending_count())

    r = _client().get("/api/literature/fetch-board")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["count"] == 2 and body["pending_count"] == 2
    wids = {e["work_id"] for e in body["requests"]}
    assert wids == {"W1", "W2"}
    e1 = next(e for e in body["requests"] if e["work_id"] == "W1")
    assert e1["status"] == "pending" and e1["doi"] == "10.x/1"
    assert e1["reason"] == "need fulltext" and e1["request_id"]


def test_fetch_board_status_filter(monkeypatch, tmp_path) -> None:
    from mast.knowledge import fetch_board as board_mod

    board = board_mod.FetchBoard(board_dir=str(tmp_path))
    rec = board.post_request("W1")
    board.resolve(rec["request_id"], "fulfilled", note="uploaded")

    monkeypatch.setattr(board_mod, "list_requests",
                        lambda status=None: board.list_requests(status))
    monkeypatch.setattr(board_mod, "pending_count", lambda: board.pending_count())

    r = _client().get("/api/literature/fetch-board", params={"status": "fulfilled"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["requests"][0]["status"] == "fulfilled"
    assert body["pending_count"] == 0


def test_fetch_board_read_raises_degrades(monkeypatch) -> None:
    from mast.knowledge import fetch_board as board_mod

    def boom(*a, **k):
        raise RuntimeError("disk")

    monkeypatch.setattr(board_mod, "list_requests", boom)
    r = _client().get("/api/literature/fetch-board")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["requests"] == [] and body["count"] == 0


# ── ingest-status probe ───────────────────────────────────────────────
def test_ingest_status_probe() -> None:
    # In the test venv numpy+pandas are present → available True, never 500.
    r = _client().get("/api/literature/ingest-status")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert isinstance(body["available"], bool)
    assert isinstance(body["can_extract_pdf"], bool)
    assert isinstance(body["can_fetch"], bool)
    assert isinstance(body["has_embedder_key"], bool)
    assert isinstance(body["can_ocr"], bool)  # scanned-PDF OCR readiness
    assert body["detail"]


# ── cognition phases ──────────────────────────────────────────────────
def test_phases_degrades_unwired() -> None:
    r = _client().get("/api/cognition/phases")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["phases"] == [] and body["count"] == 0


class _StorageStub:
    """Minimal ExperimentStorage shape: exposes a ``_db_path`` so the phases
    route can build a PhaseManager off the shared DB file."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path


def _wired_ctx(tmp_path):
    from mast.memory.sharding import PhaseManager

    db = str(tmp_path / "exp.db")
    pm = PhaseManager(db)  # creates the conversation_phase table
    ctx = AppContext()
    ctx._experiment_storage = _StorageStub(db)  # noqa: SLF001
    return ctx, pm


def test_phases_wired_lists_open_and_closed(tmp_path) -> None:
    ctx, pm = _wired_ctx(tmp_path)

    # one closed (summarised) phase + one still-open phase
    p1 = pm.start_phase(title="Setup")
    pm.end_phase()  # closes p1 (rule-based summary), opens nothing
    pm.start_phase(title="Scanning")  # still open (ended_msg_id is None)

    r = _client(ctx).get("/api/cognition/phases")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["count"] == 2
    by_idx = {p["phase_index"]: p for p in body["phases"]}
    assert by_idx[p1["phase_index"]]["open"] is False
    # the second phase is open
    open_phases = [p for p in body["phases"] if p["open"]]
    assert len(open_phases) == 1
    assert open_phases[0]["title"] == "Scanning"


def test_phases_wired_empty_not_degraded(tmp_path) -> None:
    ctx, _ = _wired_ctx(tmp_path)
    r = _client(ctx).get("/api/cognition/phases")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["phases"] == [] and body["count"] == 0


def test_phases_experiment_scoped(tmp_path) -> None:
    ctx, pm = _wired_ctx(tmp_path)
    pm.start_phase(title="exp-a phase", experiment_id="exp-a")
    pm.start_phase(title="global phase")  # experiment_id None

    r = _client(ctx).get("/api/cognition/phases", params={"experiment_id": "exp-a"})
    assert r.status_code == 200
    body = r.json()
    assert body["experiment_id"] == "exp-a"
    assert body["count"] == 1
    assert body["phases"][0]["title"] == "exp-a phase"


# ── upload-pdf (multipart): browser file → ingest + pointer  ─────
def _patch_pointer(monkeypatch):
    """Stub the post-ingest curation (add pointer + close board) so the endpoint
    tests don't touch the real registry / board."""
    from mast.knowledge import libraries as lib_mod
    from mast.knowledge import fetch_board as board_mod

    monkeypatch.setattr(
        lib_mod, "add_members",
        lambda ids, **kw: {"library_id": kw.get("library_id") or "reading",
                           "added": list(ids)},
    )
    monkeypatch.setattr(board_mod, "resolve_work_id", lambda *a, **k: 1)


def test_upload_pdf_ingests_and_adds_pointer(monkeypatch) -> None:
    from mast.knowledge import ingest as ingest_mod

    captured = {}

    class _Res:
        def to_dict(self):
            return {"work_id": "local:up1", "source": "user_pdf", "n_chunks": 3,
                    "slug_dir": "/papers/up1", "promoted": True, "status": "ingested",
                    "doi": "", "title": "Uploaded", "sha256": "abc", "detail": "",
                    "library_id": ""}

    def fake_ingest(pdf_path, **kw):
        captured["path"] = pdf_path
        captured["library_id"] = kw.get("library_id")
        return _Res()

    monkeypatch.setattr(ingest_mod, "ingest_pdf", fake_ingest)
    _patch_pointer(monkeypatch)

    r = _client().post(
        "/api/literature/upload-pdf",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"library_id": "mylib", "promote": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["work_id"] == "local:up1"
    assert body["pointer_added"] is True
    assert body["pointer_library_id"] == "mylib"
    assert body["fulfilled_requests"] == 1
    # ingest was handed a real temp path (not the original name) + pointer model
    assert captured["path"].endswith(".pdf")
    assert captured["library_id"] == ""  # promote-only, pointer added separately


def test_server_path_ingest_closes_the_board(monkeypatch) -> None:
    """The server-path ingest is a fulfilment route too — it must close the board.

    It never ran the curation tail, so an operator who satisfied a request through
    the "PDF 摄取（服务器路径）" control (or by ingesting a freshly fetched PDF) left
    the request sitting at 'pending' — the agent's ask still looked unanswered
    after they had answered it.
    """
    from mast.knowledge import ingest as ingest_mod

    class _Res:
        def to_dict(self):
            return {"work_id": "W77", "source": "user_pdf", "n_chunks": 5,
                    "slug_dir": "/papers/w77_slug", "promoted": True,
                    "status": "ingested", "doi": "", "title": "T", "sha256": "s",
                    "detail": "", "library_id": ""}

    monkeypatch.setattr(ingest_mod, "ingest_pdf", lambda *a, **k: _Res())

    closed = {}
    from mast.knowledge import libraries as lib_mod
    from mast.knowledge import fetch_board as board_mod
    monkeypatch.setattr(lib_mod, "add_members",
                        lambda ids, **kw: {"library_id": kw.get("library_id") or "reading"})
    monkeypatch.setattr(board_mod, "open_requests", lambda *a, **k: [])
    monkeypatch.setattr(
        board_mod, "resolve_work_id",
        lambda wid, status, **kw: closed.update(work_id=wid, status=status) or 1)

    r = _client().post("/api/literature/ingest",
                       json={"pdf_path": "/tmp/x.pdf", "promote": True})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert closed == {"work_id": "W77", "status": "fulfilled"}
    assert body["fulfilled_requests"] == 1
    assert body["pointer_added"] is True


def test_server_path_ingest_failure_leaves_board_alone(monkeypatch) -> None:
    from mast.knowledge import ingest as ingest_mod
    from mast.knowledge import fetch_board as board_mod

    class _Res:
        def to_dict(self):
            return {"work_id": "", "status": "error:no_text_layer"}

    monkeypatch.setattr(ingest_mod, "ingest_pdf", lambda *a, **k: _Res())
    calls = []
    monkeypatch.setattr(board_mod, "resolve_work_id",
                        lambda *a, **k: calls.append(1) or 1)

    r = _client().post("/api/literature/ingest",
                       json={"pdf_path": "/tmp/x.pdf", "promote": True})
    assert r.json()["ok"] is False
    assert calls == []


def test_upload_pdf_notifies_the_asker(monkeypatch) -> None:
    """End-to-end through route → curate → fetch_resume."""
    from mast.knowledge import ingest as ingest_mod
    from mast.knowledge import fetch_board as board_mod
    from mast.core import fetch_resume

    class _Res:
        def to_dict(self):
            return {"work_id": "W5", "source": "user_pdf", "n_chunks": 1,
                    "slug_dir": "/papers/w5", "promoted": True, "status": "ingested",
                    "doi": "", "title": "T", "sha256": "s", "detail": "",
                    "library_id": ""}

    monkeypatch.setattr(ingest_mod, "ingest_pdf", lambda *a, **k: _Res())
    _patch_pointer(monkeypatch)
    monkeypatch.setattr(
        board_mod, "open_requests",
        lambda *a, **k: [{"request_id": "fr-2", "work_id": "W5",
                          "origin_conversation_id": "conv-1", "reason": "need bias"}])

    seen = {}
    fetch_resume.set_resumer(
        lambda w, r, **kw: seen.update(work_id=w, n=len(r)) or {"resumed": 1})
    try:
        r = _client().post(
            "/api/literature/upload-pdf",
            files={"file": ("p.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"promote": "true"})
    finally:
        fetch_resume.clear_resumer()
    assert r.json()["ok"] is True
    assert seen == {"work_id": "W5", "n": 1}


def test_upload_pdf_rejects_non_pdf(monkeypatch) -> None:
    r = _client().post(
        "/api/literature/upload-pdf",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["status"] == "error:not_a_pdf"


def test_upload_pdf_error_status_no_pointer(monkeypatch) -> None:
    from mast.knowledge import ingest as ingest_mod

    class _Res:
        def to_dict(self):
            return {"work_id": "", "status": "warning:no_text_layer",
                    "detail": "scanned", "promoted": False}

    called = {"pointer": False}

    def _no_pointer(*a, **k):
        called["pointer"] = True

    monkeypatch.setattr(ingest_mod, "ingest_pdf", lambda *a, **k: _Res())
    from mast.knowledge import libraries as lib_mod
    monkeypatch.setattr(lib_mod, "add_members", _no_pointer)

    r = _client().post(
        "/api/literature/upload-pdf",
        files={"file": ("scan.pdf", b"%PDF junk", "application/pdf")},
        data={},
    )
    body = r.json()
    assert body["ok"] is False
    assert body["status"] == "warning:no_text_layer"
    assert called["pointer"] is False  # a non-ok ingest must NOT add a pointer


# ── manual-entry: typed metadata → entry + pointer  ──────────────
def test_manual_entry_creates_and_adds_pointer(monkeypatch) -> None:
    from mast.knowledge import ingest as ingest_mod

    captured = {}

    class _Res:
        def to_dict(self):
            return {"work_id": "local:man1", "source": "user_manual",
                    "promoted": False, "status": "ingested_no_embed",
                    "doi": "10.9/z", "title": "Manual", "detail": "已登记"}

    def fake_manual(**kw):
        captured.update(kw)
        return _Res()

    monkeypatch.setattr(ingest_mod, "ingest_manual", fake_manual)
    _patch_pointer(monkeypatch)

    r = _client().post(
        "/api/literature/manual-entry",
        json={"title": "Manual", "abstract": "typed abstract", "authors": "A. Bee",
              "year": "2021", "doi": "10.9/z", "library_id": "mylib"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["work_id"] == "local:man1"
    assert body["status"] == "ingested_no_embed"
    assert body["pointer_added"] is True and body["pointer_library_id"] == "mylib"
    # the relay forwarded the typed fields
    assert captured["title"] == "Manual" and captured["authors"] == "A. Bee"


def test_manual_entry_empty_is_rejected() -> None:
    r = _client().post("/api/literature/manual-entry", json={"title": "", "abstract": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["status"] == "error:empty"


def test_manual_entry_core_raises_degrades(monkeypatch) -> None:
    from mast.knowledge import ingest as ingest_mod

    def boom(**k):
        raise RuntimeError("explode")

    monkeypatch.setattr(ingest_mod, "ingest_manual", boom)
    r = _client().post("/api/literature/manual-entry", json={"title": "x"})
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True


# ── openapi: paths registered ─────────────────────────────────────────
def test_openapi_paths_present() -> None:
    spec = _client().get("/openapi.json").json()
    paths = set(spec.get("paths", {}))
    assert {
        "/api/literature/ingest",
        "/api/literature/upload-pdf",
        "/api/literature/manual-entry",
        "/api/literature/fetch",
        "/api/literature/fetch-board",
        "/api/literature/ingest-status",
        "/api/cognition/phases",
    } <= paths
