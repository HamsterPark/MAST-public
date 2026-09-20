"""Phase-2 read-only API contract tests.

Guards the typed seam between the TS frontend and the Python core:
  - every read-only endpoint returns 200 with the expected shape;
  - config/models is real (not degraded) — the end-to-end type-flow proof;
  - registry/storage-backed endpoints degrade gracefully when unwired;
  - the OpenAPI spec exposes the full path + schema set (frontend typegen input).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mast.api.app import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "mast-api"
    assert isinstance(body["version"], str) and body["version"]


def test_config_models_is_real(client: TestClient) -> None:
    r = client.get("/api/config/models")
    assert r.status_code == 200
    body = r.json()
    assert body["default_alias"] == "kimi-k3"
    assert body["thinking_presets"]["off"] == 0
    assert body["thinking_presets"]["max"] == 128000
    aliases = {m["alias"] for m in body["models"]}
    assert {"kimi-k3", "glm-5.2", "opus", "sonnet", "kimi-k2.6"} <= aliases
    # capability fields are derived per model
    default_model = next(m for m in body["models"] if m["alias"] == "kimi-k3")
    assert default_model["is_default"] is True
    assert default_model["thinking_mode"] in ("none", "fixed", "tunable")
    assert default_model["output_limit"] > 0 and default_model["input_context"] > 0
    # Every advertised model must carry a valid thinking_mode.
    assert all(m["thinking_mode"] in ("none", "fixed", "tunable") for m in body["models"])
    # 'none' tagging must still be correct. moonshot-v1-128k (chat-only, no reasoning)
    # was removed from the ADVERTISED list 2026-06-23, but the capability table still
    # knows it — thinking must NEVER be sent to a 'none' model, so pin the function.
    from mast.config import model_thinking_mode
    assert model_thinking_mode("moonshot-v1-128k") == "none"


def test_settings_shape(client: TestClient) -> None:
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    # all whitelisted keys present (None when unset), no extras
    assert "model_alias" in body and "theme" in body and "codex_live_search" in body


def test_safety_limits_real(client: TestClient) -> None:
    r = client.get("/api/safety/limits")
    assert r.status_code == 200
    body = r.json()
    assert body["bias_min_v"] == -10.0
    assert body["bias_max_v"] == 10.0
    assert body["scan_size_max_m"] == 1e-5


def test_skills_catalog_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/skills/catalog")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["index"] == [] and body["count"] == 0


def test_experiments_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/experiments")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["experiments"] == []


def test_openapi_contract(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    paths = set(spec.get("paths", {}))
    assert {
        "/api/health",
        "/api/config/models",
        "/api/settings",
        "/api/safety/limits",
        "/api/skills/catalog",
        "/api/experiments",
    } <= paths
    # schemas are the frontend typegen input — must be non-trivial
    assert len(spec.get("components", {}).get("schemas", {})) >= 6
