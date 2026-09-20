"""Contract tests for domain ``guidance_ext`` (decision trees / recipes /
templates — the three guidance sub-tabs beyond per-skill annotations).

The router is NOT yet mounted in mast.api.app (integration wires that), so
each test builds a throwaway FastAPI app and includes the router under /api.

Guarantees asserted:
  * every endpoint returns 200 + a body matching its response_model;
  * with no live core wired (standalone AppContext) every registry-backed
    endpoint DEGRADES — empty-but-valid, ``degraded: true``, never 500 — while
    still surfacing the code defaults on GET;
  * an unknown kind degrades cleanly (no 404 / 500);
  * when a real ConfigOverrideRegistry IS wired (pointed at a tmp dir) the
    per-kind write persists + hot-reloads and the read reflects it (effective =
    default ⊕ override), proving the passthrough; empty payload resets it.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.guidance_ext import router

KINDS = ["decision_trees", "recipes", "templates"]


def _client(user_root: str | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext(user_root=user_root)
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _client_with_registry(tmp_path) -> tuple[TestClient, object]:
    """Throwaway app with a REAL registry pointed at a tmp dir (leader-style
    wiring: attribute set on ctx)."""
    from mast.admin.override_store import ConfigOverrideRegistry

    reg = ConfigOverrideRegistry(overrides_dir=tmp_path / "overrides")
    app = FastAPI()
    ctx = AppContext(user_root=str(tmp_path))
    ctx.override_registry = reg
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app), reg


@pytest.fixture()
def client() -> TestClient:
    return _client()


# ── Standalone degradation (no registry wired) ───────────────────────────────
@pytest.mark.parametrize("kind", KINDS)
def test_get_degrades_but_surfaces_defaults(client: TestClient, kind: str) -> None:
    r = client.get(f"/api/admin/guidance-extra/{kind}")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == kind
    assert body["degraded"] is True
    assert body["has_override"] is False
    assert body["writable"] is True
    # Code defaults still surface (the constants exist in skill_guidance).
    assert body["data"] is not None


@pytest.mark.parametrize("kind", KINDS)
def test_post_degrades_without_registry(client: TestClient, kind: str) -> None:
    r = client.post(f"/api/admin/guidance-extra/{kind}", json={"data": {"x": 1}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["kind"] == kind
    assert body["degraded"] is True


def test_unknown_kind_degrades_not_404(client: TestClient) -> None:
    r = client.get("/api/admin/guidance-extra/bogus")
    assert r.status_code == 200
    assert r.json()["degraded"] is True

    w = client.post("/api/admin/guidance-extra/bogus", json={"data": {"x": 1}})
    assert w.status_code == 200
    wb = w.json()
    assert wb["ok"] is False
    assert wb["degraded"] is True


# ── Real registry: roundtrip persists + hot-reloads + merges ─────────────────
def test_decision_trees_roundtrip(tmp_path) -> None:
    c, reg = _client_with_registry(tmp_path)
    payload = {
        "my_tree": {
            "title": "Test Tree",
            "nodes": [{"q": "go?", "options": [["yes", "act"]]}],
        }
    }
    w = c.post("/api/admin/guidance-extra/decision_trees", json={"data": payload})
    assert w.status_code == 200
    wb = w.json()
    assert wb["ok"] is True
    assert wb["degraded"] is False
    # `reloaded` is False because nothing subscribes to the override hot-reload
    # (it was a hardcoded True until 2026-08-03). The write still persisted.
    assert wb["reloaded"] is False
    assert wb["override"] == payload

    r = c.get("/api/admin/guidance-extra/decision_trees")
    rb = r.json()
    assert rb["degraded"] is False
    assert rb["has_override"] is True
    assert rb["override"] == payload
    # Effective = code defaults merged with the new id.
    assert "my_tree" in rb["data"]
    assert rb["data"]["my_tree"]["title"] == "Test Tree"


def test_recipes_roundtrip_list_replace(tmp_path) -> None:
    c, reg = _client_with_registry(tmp_path)
    recipes = [{"name": "R1", "desc": "d", "chain": ["A", "B"], "params": ""}]
    w = c.post("/api/admin/guidance-extra/recipes", json={"data": recipes})
    assert w.json()["ok"] is True

    r = c.get("/api/admin/guidance-extra/recipes")
    rb = r.json()
    assert rb["has_override"] is True
    # List kind = whole-list REPLACE: effective is exactly the override list.
    assert rb["data"] == recipes
    assert rb["override"] == recipes


def test_templates_roundtrip_and_reset(tmp_path) -> None:
    c, reg = _client_with_registry(tmp_path)
    tmpl = {
        "my_tmpl": {
            "name": "T",
            "name_en": "T",
            "description": "",
            "parameters": {"bias": {"min": 0, "max": 1}},
            "skills_chain": ["SetBias"],
        }
    }
    assert c.post("/api/admin/guidance-extra/templates", json={"data": tmpl}).json()["ok"]

    rb = c.get("/api/admin/guidance-extra/templates").json()
    assert rb["has_override"] is True
    assert "my_tmpl" in rb["data"]

    # Empty payload ⇒ reset to defaults (override dropped).
    reset = c.post("/api/admin/guidance-extra/templates", json={"data": {}})
    assert reset.json()["ok"] is True
    after = c.get("/api/admin/guidance-extra/templates").json()
    assert after["has_override"] is False
    assert "my_tmpl" not in (after["data"] or {})


def test_kinds_share_file_independently(tmp_path) -> None:
    """All three kinds live in guidance_overrides.json under distinct keys —
    writing one must not clobber another."""
    c, reg = _client_with_registry(tmp_path)
    c.post("/api/admin/guidance-extra/recipes", json={"data": [{"name": "R"}]})
    c.post(
        "/api/admin/guidance-extra/decision_trees",
        json={"data": {"t": {"title": "x", "nodes": []}}},
    )
    assert c.get("/api/admin/guidance-extra/recipes").json()["has_override"] is True
    assert c.get("/api/admin/guidance-extra/decision_trees").json()["has_override"] is True
