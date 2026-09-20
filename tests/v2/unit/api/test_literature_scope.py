"""实验专属文献库的两个端点：ensure + copy。

设计文档：``docs/v2/design/document_and_library_management.md`` §4

也钉住 ``GET /literature/libraries`` 的新契约：响应里必须给出 ``effective_library_id``
和 ``effective_source``。前端此前把 ``selectedLibraryId`` 初始化为 ``null`` 且从不读
后端给的活动库 id，而 schema 明写了「``active_library_id`` is the id the UI should
pre-select」—— 契约违约。现在要预选的是**有效库**，因为活动库只是无实验时的兜底指针。
"""

from __future__ import annotations

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
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes import literature as lit_routes
from mast.api.routes import literature_scope
from mast.knowledge import experiment_library as EL
from mast.knowledge import libraries as L
from mast.logging.storage import ExperimentStorage


def _client(*, wired: bool = True) -> TestClient:
    app = FastAPI()
    ctx = AppContext()
    if wired:
        ctx.literature_wired = True
    app.state.ctx = ctx
    app.include_router(literature_scope.router, prefix="/api")
    app.include_router(lit_routes.router, prefix="/api")
    return TestClient(app)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(tmp_path / "experiments"))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "db" / "exp.db"))
    monkeypatch.setenv("MAST_LITERATURE_LIBS_DIR", str(tmp_path / "libs"))
    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
    L.reset_default_registry()
    yield tmp_path
    L.reset_default_registry()


@pytest.fixture()
def experiments(env):
    from mast.agents._shared.data_paths import experiment_db_path
    st = ExperimentStorage(experiment_db_path())
    a = st.create_experiment("Au(111) 形貌与 STS", "看清重构")
    b = st.create_experiment("NbSe2 CDW", "看 CDW")
    return st, a, b


# ── ensure ────────────────────────────────────────────────────────────
def test_ensure_creates_then_is_idempotent(experiments):
    _st, a, _b = experiments
    c = _client()
    r1 = c.post(f"/api/literature/experiments/{a}/library").json()
    assert r1["ok"] and r1["created"] is True
    assert r1["library_id"] == EL.experiment_library_id(a)
    assert r1["experiment_id"] == a
    assert r1["members_path"].endswith("members.jsonl")

    r2 = c.post(f"/api/literature/experiments/{a}/library").json()
    assert r2["ok"] and r2["created"] is False          # ensure, not create
    assert r2["library_id"] == r1["library_id"]


def test_ensure_reports_member_count(experiments):
    _st, a, _b = experiments
    EL.add_members(a, ["W111", "W222"], reason="x")
    out = _client().post(f"/api/literature/experiments/{a}/library").json()
    assert out["member_count"] == 2


def test_ensure_rejects_blank_experiment_id(experiments):
    # An all-whitespace path segment reaches the handler and must be refused
    # honestly rather than minting a library called "exp_00000000".
    out = _client().post("/api/literature/experiments/%20/library").json()
    assert out["ok"] is False and out["degraded"] is False


# ── copy ──────────────────────────────────────────────────────────────
def test_copy_into_another_experiment(experiments):
    _st, a, b = experiments
    EL.add_members(a, ["W111", "W222"], reason="Au 参考")
    out = _client().post(
        f"/api/literature/libraries/{EL.experiment_library_id(a)}/copy",
        json={"to_experiment_id": b},
    ).json()
    assert out["ok"] and sorted(out["copied"]) == ["W111", "W222"]
    assert out["library_id"] == EL.experiment_library_id(b)
    assert {m["work_id"] for m in EL.current_members(b)} == {"W111", "W222"}


def test_copy_reports_skipped_without_overwriting(experiments):
    _st, a, b = experiments
    EL.add_members(a, ["W111"], reason="A 的说法")
    EL.add_members(b, ["W111"], reason="B 自己的批注")
    out = _client().post(
        f"/api/literature/libraries/{EL.experiment_library_id(a)}/copy",
        json={"to_experiment_id": b},
    ).json()
    assert out["ok"] and out["copied"] == [] and out["skipped"] == ["W111"]
    got = {m["work_id"]: m for m in EL.current_members(b)}
    assert got["W111"]["reason"] == "B 自己的批注"


def test_copy_from_a_custom_library(experiments):
    """非实验库也能当源 —— 这是「把攒了几年的收藏挂到一个实验上」的路径。"""
    _st, a, _b = experiments
    cust = L.create_library("Au(111) 收藏")["library_id"]
    L.add_members(["W111", "W222"], cust, reason="收藏", added_by="user")
    out = _client().post(f"/api/literature/libraries/{cust}/copy",
                         json={"to_experiment_id": a}).json()
    assert out["ok"] and sorted(out["copied"]) == ["W111", "W222"]
    assert len(L.get_library(cust)["members"]) == 2       # 源库不动


def test_copy_unknown_library_is_a_domain_error_not_a_500(experiments):
    _st, a, _b = experiments
    out = _client().post("/api/literature/libraries/nope/copy",
                         json={"to_experiment_id": a}).json()
    assert out["ok"] is False and out["degraded"] is False and out["message"]


def test_copy_without_target_and_without_active_experiment_is_refused(experiments):
    _st, a, _b = experiments
    EL.add_members(a, ["W111"])
    out = _client().post(
        f"/api/literature/libraries/{EL.experiment_library_id(a)}/copy",
        json={"to_experiment_id": ""},
    ).json()
    assert out["ok"] is False and "无法确定复制目标" in out["message"]


# ── degraded contract ─────────────────────────────────────────────────
@pytest.mark.parametrize("call", ["ensure", "copy"])
def test_endpoints_never_500(experiments, call):
    _st, a, _b = experiments
    c = _client()
    if call == "ensure":
        r = c.post(f"/api/literature/experiments/{a}/library")
    else:
        r = c.post(f"/api/literature/libraries/{EL.experiment_library_id(a)}/copy",
                   json={"to_experiment_id": a})
    assert r.status_code == 200


# ── the libraries list carries the effective library ──────────────────
def test_library_list_exposes_the_effective_library(experiments):
    st, a, _b = experiments
    st.set_active_scope(a, None, updated_by="test")
    EL.add_members(a, ["W111"], reason="x")
    out = _client().get("/api/literature/libraries").json()
    assert out["effective_library_id"] == EL.experiment_library_id(a)
    assert out["effective_source"] == "experiment"
    row = next(r for r in out["libraries"]
               if r["library_id"] == EL.experiment_library_id(a))
    assert row["scope"] == "experiment"
    assert row["experiment_id"] == a
    # the UI needs a human label, not a bare exp_1a2b3c4d
    assert row["experiment_name"] == "Au(111) 形貌与 STS"


def test_library_list_effective_is_manual_without_an_experiment(experiments):
    st, a, _b = experiments
    st.set_active_scope(a, None, updated_by="test")
    st.set_active_scope(None, None, updated_by="test")
    out = _client().get("/api/literature/libraries").json()
    assert out["effective_library_id"] == "reading"
    assert out["effective_source"] == "manual"


def test_library_list_degrades_without_a_wired_backend(experiments):
    out = _client(wired=False).get("/api/literature/libraries").json()
    assert out["degraded"] is True and out["libraries"] == []
