"""Domain G — literature libraries + semantic search API contract tests.

Guards the typed seam for the 文献库 panel:
  - every endpoint returns its defined status with the schema-shaped body;
  - STANDALONE (no live core wired) every endpoint degrades empty-not-broken
    (``degraded=True`` / ``ok=False``), NEVER 500;
  - a WIRED context exercises real library CRUD against an isolated registry,
    and search/abstract against monkeypatched core functions (no 50k index).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.literature import router


# ── clients ───────────────────────────────────────────────────────────
def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx if ctx is not None else AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture()
def wired(tmp_path) -> AppContext:
    """A context with a live, ISOLATED library registry (its own tmp JSON dir)
    plus the wiring flag the route gate reads."""
    from mast.knowledge.libraries import LibraryRegistry

    ctx = AppContext()
    ctx.literature_wired = True  # type: ignore[attr-defined]
    ctx.library_registry = LibraryRegistry(libs_dir=str(tmp_path / "libs"))  # type: ignore[attr-defined]
    return ctx


# ── standalone (unwired) degradation ──────────────────────────────────
def test_libraries_degrades_unwired() -> None:
    r = _client().get("/api/literature/libraries")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["libraries"] == [] and body["count"] == 0
    assert body["active_library_id"] is None


def test_create_degrades_unwired() -> None:
    r = _client().post("/api/literature/libraries", json={"name": "x"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True


def test_patch_degrades_unwired() -> None:
    r = _client().patch("/api/literature/libraries/foo", json={"new_name": "y"})
    assert r.status_code == 200
    assert r.json()["degraded"] is True


def test_delete_degrades_unwired() -> None:
    r = _client().delete("/api/literature/libraries/foo")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


def test_add_members_degrades_unwired() -> None:
    r = _client().post(
        "/api/literature/libraries/foo/members", json={"work_ids": ["W123"]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert body["added"] == []


def test_search_degrades_unwired() -> None:
    r = _client().post("/api/literature/search", json={"query": "graphene"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["hits"] == [] and body["count"] == 0


def test_abstract_degrades_unwired() -> None:
    r = _client().get("/api/literature/abstract/W42")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["found"] is False
    assert body["work_id"] == "W42"


# ── wired: library CRUD lifecycle ─────────────────────────────────────
def test_libraries_list_wired_has_global(wired: AppContext) -> None:
    r = _client(wired).get("/api/literature/libraries")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    ids = {x["library_id"] for x in body["libraries"]}
    assert "reading" in ids  # the undeletable global reading library
    assert body["active_library_id"] == "reading"
    # member_count/scope shape present
    glob = next(x for x in body["libraries"] if x["library_id"] == "reading")
    assert glob["scope"] == "global"
    assert glob["member_count"] == 0


def test_create_rename_delete_wired(wired: AppContext) -> None:
    c = _client(wired)
    # create
    r = c.post("/api/literature/libraries", json={"name": "My Papers", "scope": "custom"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    lib_id = body["library"]["library_id"]
    assert body["library"]["name"] == "My Papers"
    # rename
    r = c.patch(f"/api/literature/libraries/{lib_id}", json={"new_name": "Renamed"})
    assert r.status_code == 200
    assert r.json()["library"]["name"] == "Renamed"
    # delete
    r = c.delete(f"/api/literature/libraries/{lib_id}")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_create_empty_name_rejected(wired: AppContext) -> None:
    r = _client(wired).post("/api/literature/libraries", json={"name": "   "})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is False


def test_delete_global_rejected_not_500(wired: AppContext) -> None:
    # the global reading library is undeletable → LibraryError surfaces as ok=False
    r = _client(wired).delete("/api/literature/libraries/reading")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is False


def test_add_members_wired(wired: AppContext) -> None:
    c = _client(wired)
    r = c.post("/api/literature/libraries", json={"name": "Lib"})
    lib_id = r.json()["library"]["library_id"]
    r = c.post(
        f"/api/literature/libraries/{lib_id}/members",
        json={"work_ids": ["W1", "W2", "W1"], "reason": "seed"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert set(body["added"]) == {"W1", "W2"}
    assert body["skipped"] == ["W1"]  # dup upsert
    assert body["member_count"] == 2
    assert body["library_id"] == lib_id


def test_add_members_empty_rejected(wired: AppContext) -> None:
    r = _client(wired).post(
        "/api/literature/libraries/reading/members", json={"work_ids": []}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is False


# ── wired: search (monkeypatched core, no real index) ─────────────────
def test_search_semantic_wired(monkeypatch, wired: AppContext) -> None:
    from mast.knowledge import literature_index

    def fake_search(query, k=20, **kw):
        return [
            {
                "work_id": "W1", "title": "Graphene on Ir", "year": 2020,
                "journal": "PRL", "doi": "10.x/1", "cited": 12, "score": 0.91,
                "source": "openalex", "abstract_excerpt": "an excerpt",
            },
            {
                "work_id": "W2", "title": "STM of MoS2", "year": 2021,
                "journal": "Nature", "doi": "", "cited": 3, "score": 0.80,
                "source": "openalex", "abstract_excerpt": "",
            },
        ][:k]

    monkeypatch.setattr(literature_index, "search", fake_search)
    r = _client(wired).post("/api/literature/search", json={"query": "graphene", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["count"] == 2
    h0 = body["hits"][0]
    assert h0["work_id"] == "W1" and h0["title"] == "Graphene on Ir"
    assert h0["year"] == 2020 and h0["cited"] == 12
    assert h0["retrieval"] == "semantic"


def test_search_keyword_fallback_flags_degraded(monkeypatch, wired: AppContext) -> None:
    from mast.knowledge import literature_index

    def fake_search(query, k=20, **kw):
        # mimic the keyword fallback: rows carry degraded=True / retrieval=keyword
        return [{
            "work_id": "W9", "title": "kw hit", "year": 2019, "journal": "",
            "doi": "", "cited": 0, "score": 2.0, "source": "openalex",
            "degraded": True, "retrieval": "keyword",
        }]

    monkeypatch.setattr(literature_index, "search", fake_search)
    r = _client(wired).post("/api/literature/search", json={"query": "kw"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True  # surfaced from per-row flag
    assert body["hits"][0]["retrieval"] == "keyword"


def test_search_library_filtered(monkeypatch, wired: AppContext) -> None:
    from mast.knowledge import literature_index

    c = _client(wired)
    lib_id = c.post("/api/literature/libraries", json={"name": "L"}).json()[
        "library"
    ]["library_id"]
    c.post(f"/api/literature/libraries/{lib_id}/members", json={"work_ids": ["W2"]})

    def fake_search(query, k=20, **kw):
        return [
            {"work_id": "W1", "title": "a", "year": 0, "journal": "", "doi": "",
             "cited": 0, "score": 0.5, "source": ""},
            {"work_id": "W2", "title": "b", "year": 0, "journal": "", "doi": "",
             "cited": 0, "score": 0.4, "source": ""},
        ]

    monkeypatch.setattr(literature_index, "search", fake_search)
    r = c.post("/api/literature/search", json={"query": "q", "library_id": lib_id})
    assert r.status_code == 200
    body = r.json()
    # filtered to the library's member pointer-set (only W2)
    assert [h["work_id"] for h in body["hits"]] == ["W2"]


def test_search_library_filtered_canonical_mixed_forms(monkeypatch, wired: AppContext) -> None:
    """Regression : members store the CANONICAL bare id while search
    results carry the URL form. The filter must compare canonically or a
    library-scoped search would return nothing."""
    from mast.knowledge import literature_index

    c = _client(wired)
    lib_id = c.post("/api/literature/libraries", json={"name": "L"}).json()[
        "library"
    ]["library_id"]
    # add via the URL form → stored bare as "W2912345678"
    c.post(
        f"/api/literature/libraries/{lib_id}/members",
        json={"work_ids": ["https://openalex.org/W2912345678"]},
    )

    def fake_search(query, k=20, **kw):
        # search surfaces the URL form (as the real metadata.parquet does)
        return [
            {"work_id": "https://openalex.org/W2912345678", "title": "match",
             "year": 0, "journal": "", "doi": "", "cited": 0, "score": 0.9, "source": ""},
            {"work_id": "https://openalex.org/W9999999999", "title": "other",
             "year": 0, "journal": "", "doi": "", "cited": 0, "score": 0.4, "source": ""},
        ]

    monkeypatch.setattr(literature_index, "search", fake_search)
    r = c.post("/api/literature/search", json={"query": "q", "library_id": lib_id})
    assert r.status_code == 200
    hits = r.json()["hits"]
    # the URL-form result matched the bare-stored member → exactly one hit
    assert [h["title"] for h in hits] == ["match"]


def test_search_blank_query_not_degraded(wired: AppContext) -> None:
    r = _client(wired).post("/api/literature/search", json={"query": "   "})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"] == [] and body["degraded"] is False


def test_search_core_raises_degrades(monkeypatch, wired: AppContext) -> None:
    from mast.knowledge import literature_index

    def boom(*a, **k):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(literature_index, "search", boom)
    r = _client(wired).post("/api/literature/search", json={"query": "x"})
    assert r.status_code == 200
    assert r.json()["degraded"] is True


# ── wired: abstract (monkeypatched core) ──────────────────────────────
def test_abstract_found_wired(monkeypatch, wired: AppContext) -> None:
    from mast.knowledge import literature_index

    def fake_fetch(work_id):
        return {
            "work_id": work_id, "found": True, "title": "T",
            "abstract": "full abstract", "first_author": "A. Author",
            "year": "2020", "journal": "PRL", "source": "openalex",
            "cited_by_count": 7,
        }

    monkeypatch.setattr(literature_index, "fetch_abstract", fake_fetch)
    r = _client(wired).get("/api/literature/abstract/W1")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False and body["found"] is True
    assert body["abstract"] == "full abstract"
    assert body["cited_by_count"] == 7
    assert body["year"] == "2020"


def test_abstract_not_found_wired(monkeypatch, wired: AppContext) -> None:
    from mast.knowledge import literature_index

    def fake_fetch(work_id):
        return {"work_id": work_id, "found": False, "abstract": "",
                "note": "literature index not provisioned"}

    monkeypatch.setattr(literature_index, "fetch_abstract", fake_fetch)
    r = _client(wired).get("/api/literature/abstract/Wxxx")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False and body["found"] is False
    assert body["note"] == "literature index not provisioned"


def test_abstract_blank_id_wired(wired: AppContext) -> None:
    # blank id after the wiring gate → found False, not degraded, not 500
    r = _client(wired).get("/api/literature/abstract/%20")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is False and body["degraded"] is False


# ── openapi: paths + schemas registered ───────────────────────────────
def test_openapi_paths_present(wired: AppContext) -> None:
    spec = _client(wired).get("/openapi.json").json()
    paths = set(spec.get("paths", {}))
    assert {
        "/api/literature/libraries",
        "/api/literature/libraries/{library_id}",
        "/api/literature/libraries/{library_id}/members",
        "/api/literature/search",
        "/api/literature/abstract/{work_id}",
    } <= paths
