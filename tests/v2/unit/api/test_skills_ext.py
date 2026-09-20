"""Contract tests for Domain C of the typed API seam (skills detail +
composites + encyclopedia).

Every endpoint must return its declared status with a schema-shaped body, and
every degraded path must be empty-not-broken (``degraded=True``, never a 500).
The router is tested via a throwaway app (integration wires it into
``mast.api.app`` separately).
"""

from __future__ import annotations

import builtins

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.skills_ext import router


def _client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture(autouse=True)
def _isolated(seeded_composite_store):
    """Every test in this module writes through the REAL ``composite_store()``
    unless the store is redirected — and the clone/restore round-trips below DO
    write. A full suite run used to bump the operator's ``BatchRegionsScan`` from
    v46 to v47 and leave two junk files in ``_history/``. It surfaced only because
    ``git checkout`` refused to switch branches over the dirty file.

    ``seeded_composite_store`` (tests/v2/conftest.py) points the store at a tmp
    COPY of the real specs — so the round-trips still exercise real composites
    with real version history; they just do it somewhere disposable. (A bare tmp
    dir would NOT work: these tests ``pytest.skip`` on an empty store, so
    isolating them that way would silently stop them testing anything.)

    Sibling files — test_builder / test_version_store / test_composite_panel_* —
    have isolated since they were written. This one never got the fixture."""
    return seeded_composite_store


@pytest.fixture()
def client() -> TestClient:
    return _client()


# ── /api/skills/{name} ───────────────────────────────────────────────────────


def test_skill_card_degrades_unwired(client: TestClient) -> None:
    # No registry wired in the throwaway ctx → empty-but-not-broken card.
    r = client.get("/api/skills/SetBias")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["found"] is False
    assert body["name"] == "SetBias"
    assert body["parameters"] == []
    assert body["tags"] == []


def test_skill_card_real_when_registry_wired(client: TestClient) -> None:
    # Wire a sentinel registry so the handler takes the live-catalog branch,
    # then point builder_api.get_catalog at a fake card.
    import mast.webui.builder_api as ba

    client.app.state.ctx._skill_registry = object()
    fake = {
        "cards": {
            "SetBias": {
                "name": "SetBias",
                "zh": "设置偏压",
                "category": "BIAS",
                "safety": "confirm",
                "level": 0,
                "source": "builtin",
                "source_zh": "内置",
                "tags": ["bias"],
                "domain": "偏压与电流",
                "version": "1.0",
                "description": "set the bias voltage",
                "description_zh": "设置偏压电压",
                "parameters": [
                    {"name": "bias_v", "type": "float", "description": "",
                     "unit": "V", "required": True, "default": None,
                     "min": -10.0, "max": 10.0, "allowed_values": None}
                ],
                "preconditions": [],
                "postconditions": [],
                "estimated_duration_s": 0.5,
                "rollback_skill": None,
                "extra": {},
            }
        }
    }
    orig = ba.get_catalog
    ba.get_catalog = lambda: fake  # type: ignore[assignment]
    try:
        r = client.get("/api/skills/SetBias")
        assert r.status_code == 200
        body = r.json()
        assert body["degraded"] is False
        assert body["found"] is True
        assert body["name"] == "SetBias"
        assert body["zh"] == "设置偏压"
        assert body["parameters"][0]["name"] == "bias_v"
        assert body["parameters"][0]["max"] == 10.0
    finally:
        ba.get_catalog = orig  # type: ignore[assignment]


def test_skill_card_unknown_name_when_wired(client: TestClient) -> None:
    import mast.webui.builder_api as ba

    client.app.state.ctx._skill_registry = object()
    orig = ba.get_catalog
    ba.get_catalog = lambda: {"cards": {}}  # type: ignore[assignment]
    try:
        r = client.get("/api/skills/NoSuchSkill")
        assert r.status_code == 200
        body = r.json()
        # Registry IS wired but skill is unknown → not found, not degraded.
        assert body["found"] is False
        assert body["degraded"] is False
    finally:
        ba.get_catalog = orig  # type: ignore[assignment]


# ── /api/composites ──────────────────────────────────────────────────────────


def test_composites_list_real(client: TestClient) -> None:
    # File-backed store works standalone — real (non-degraded) data.
    r = client.get("/api/composites")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert isinstance(body["composites"], list)
    assert body["count"] == len(body["composites"])
    for c in body["composites"]:
        assert set(c) >= {"name", "version", "description", "safety_level",
                          "n_nodes", "tags"}


def test_composites_list_degrades_on_backend_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "mast.webui.composite_panel":
            raise RuntimeError("store unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.get("/api/composites")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["composites"] == [] and body["count"] == 0


def test_composite_get_found(client: TestClient) -> None:
    names = [c["name"] for c in client.get("/api/composites").json()["composites"]]
    if not names:
        pytest.skip("no composites stored in this checkout")
    name = names[0]
    r = client.get(f"/api/composites/{name}")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["found"] is True
    assert body["name"] == name
    assert isinstance(body["spec"], dict) and body["spec"].get("name") == name
    assert isinstance(body["versions"], list)
    if body["versions"]:
        assert set(body["versions"][0]) >= {"version", "saved_at", "n_nodes",
                                            "description"}


def test_composite_get_unknown(client: TestClient) -> None:
    r = client.get("/api/composites/__no_such_composite__")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is False
    assert body["degraded"] is False
    assert body["spec"] is None


def test_composite_versions(client: TestClient) -> None:
    names = [c["name"] for c in client.get("/api/composites").json()["composites"]]
    if not names:
        pytest.skip("no composites stored in this checkout")
    name = names[0]
    r = client.get(f"/api/composites/{name}/versions")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["degraded"] is False
    assert isinstance(body["versions"], list)


def test_composite_versions_unknown(client: TestClient) -> None:
    r = client.get("/api/composites/__no_such_composite__/versions")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is False
    assert body["versions"] == []


# ── /api/composites/{name}/validate ──────────────────────────────────────────


def test_validate_returns_report(client: TestClient) -> None:
    spec = {"name": "X", "description": "d", "safety_level": "confirm",
            "params": [], "nodes": [], "tags": []}
    r = client.post("/api/composites/X/validate", json={"spec": spec})
    assert r.status_code == 200
    body = r.json()
    # Shape is always present; ok may be True (empty spec parses) or carry
    # problems depending on the validator — both are valid contract outcomes.
    assert "ok" in body and isinstance(body["problems"], list)
    assert isinstance(body["steps"], list)
    assert isinstance(body["degraded"], bool)


def test_validate_empty_spec(client: TestClient) -> None:
    r = client.post("/api/composites/X/validate", json={})
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body and isinstance(body["problems"], list)


def test_validate_degrades_on_backend_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "mast.webui.builder_api":
            raise RuntimeError("validator unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.post("/api/composites/X/validate",
                    json={"spec": {"name": "X", "nodes": []}})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["ok"] is False
    assert body["problems"]


# ── /api/composites/{name}/clone ─────────────────────────────────────────────


def test_clone_missing_new_name(client: TestClient) -> None:
    r = client.post("/api/composites/Foo/clone", json={"new_name": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "missing_new_name"
    assert body["degraded"] is False


def test_clone_unknown_source(client: TestClient) -> None:
    # Source does not exist → typed error, not a 500.
    r = client.post("/api/composites/__no_such_src__/clone",
                    json={"new_name": "__tmp_clone_target__"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"]


def test_clone_degrades_on_backend_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "mast.webui.composite_panel":
            raise RuntimeError("store unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.post("/api/composites/Foo/clone", json={"new_name": "Bar"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True


def test_clone_roundtrip(client: TestClient) -> None:
    names = [c["name"] for c in client.get("/api/composites").json()["composites"]]
    if not names:
        pytest.skip("no composites stored to clone from")
    src = names[0]
    target = "__test_clone_skills_ext__"
    # Ensure a clean slate first.
    from mast.webui.composite_panel import composite_store
    store = composite_store()
    try:
        if store.exists(target):
            store.delete(target)
        r = client.post(f"/api/composites/{src}/clone",
                        json={"new_name": target, "author": "pytest"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["name"] == target
        # Version numbering is history-backed (the version store retains prior
        # _history/<name>.vN.json even after delete of the active spec), so a
        # re-run does not necessarily reset to 1 — assert a valid version, not ==1.
        assert body["version"] >= 1
        assert body["degraded"] is False
    finally:
        try:
            if store.exists(target):
                store.delete(target)
        except Exception:
            pass


# ── /api/composites/{name}/restore/{version} ─────────────────────────────────


def test_restore_unknown(client: TestClient) -> None:
    r = client.post("/api/composites/__no_such__/restore/1")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"]


def test_restore_degrades_on_backend_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "mast.webui.composite_panel":
            raise RuntimeError("store unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.post("/api/composites/Foo/restore/1")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True


def test_restore_roundtrip(client: TestClient) -> None:
    names = [c["name"] for c in client.get("/api/composites").json()["composites"]]
    if not names:
        pytest.skip("no composites stored to restore")
    name = names[0]
    versions = client.get(f"/api/composites/{name}/versions").json()["versions"]
    if not versions:
        pytest.skip("composite has no version history")
    v = versions[-1]["version"]  # oldest available
    before = client.get(f"/api/composites/{name}").json()["spec"]["version"]
    r = client.post(f"/api/composites/{name}/restore/{v}")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    # Restore writes forward as a NEW version (non-destructive).
    assert body["version"] > before


# ── /api/encyclopedia/* ──────────────────────────────────────────────────────


def test_encyclopedia_domains_real(client: TestClient) -> None:
    r = client.get("/api/encyclopedia/domains")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["count"] == len(body["domains"]) > 0
    ids = {d["id"] for d in body["domains"]}
    assert {"bias_current", "scanning", "spectroscopy"} <= ids
    d0 = next(d for d in body["domains"] if d["id"] == "bias_current")
    assert "SetBias" in d0["skills"]
    assert d0["agent"] == "instrument_control"


def test_encyclopedia_domains_degrades(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "mast.webui.encyclopedia":
            raise RuntimeError("encyclopedia unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.get("/api/encyclopedia/domains")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["domains"] == [] and body["count"] == 0


def test_encyclopedia_intent_mapping_real(client: TestClient) -> None:
    r = client.get("/api/encyclopedia/intent-mapping")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["count"] == len(body["mapping"]) > 0
    e0 = body["mapping"][0]
    assert set(e0) >= {"keywords", "skill", "note"}


def test_encyclopedia_intent_mapping_degrades(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "mast.webui.encyclopedia":
            raise RuntimeError("encyclopedia unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.get("/api/encyclopedia/intent-mapping")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["mapping"] == []


# ── OpenAPI contract ─────────────────────────────────────────────────────────


def test_openapi_paths(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    paths = set(spec.get("paths", {}))
    assert {
        "/api/skills/{name}",
        "/api/composites",
        "/api/composites/{name}",
        "/api/composites/{name}/versions",
        "/api/composites/{name}/validate",
        "/api/composites/{name}/clone",
        "/api/composites/{name}/restore/{version}",
        "/api/encyclopedia/domains",
        "/api/encyclopedia/intent-mapping",
    } <= paths
