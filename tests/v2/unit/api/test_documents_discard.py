"""废弃 / 恢复 —— 没有「删除」这个动作。

设计：`docs/v2/design/document_and_library_management.md`（`_discarded` 区）

为什么有这条路：`save()` 在 doc_id 找不到时刻意**另立新文档**而不是报错（丢掉 LLM
写好的整篇内容不可接受）。允许增殖的前提是事后能清理 —— 否则一个忘传 doc_id 的
agent 就能在用户的报告列表里永久留下一串重复。

为什么不是硬删除：`_quarantine` 定下的规矩是「绝不丢字节，宁可事后认领」。一份看着
是垃圾的报告，可能是用户唯一还留着的那一版。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mast.api.app import create_app
from mast.documents import reset_caches, store
from mast.documents.paths import exp_dir_for
from mast.logging.storage import ExperimentStorage
from mast.logging.v2 import manifest as mf


@pytest.fixture()
def env(tmp_path, monkeypatch):
    root = tmp_path / "MAST-Data" / "experiments"
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "db" / "exp.db"))
    root.mkdir(parents=True, exist_ok=True)
    reset_caches()
    yield root
    reset_caches()


@pytest.fixture()
def scoped(env):
    from mast.agents._shared.data_paths import experiment_db_path
    st = ExperimentStorage(experiment_db_path())
    eid = st.create_experiment("Au(111)", "")
    st.set_active_scope(eid, None, updated_by="test")
    d = exp_dir_for(eid, create=True)
    mf.write_experiment_manifest(d, experiment_id=eid, title="Au(111)", dir_name=d.name)
    reset_caches()
    return st, eid


@pytest.fixture()
def client(env):
    return TestClient(create_app(dev_cors=False))


def _titles(payload) -> set[str]:
    return {r["title"] for r in payload["documents"]}


def test_discarded_document_leaves_the_list_but_keeps_its_bytes(scoped, client):
    store().save(text="留着的", kind="experiment_report", title="留着")
    junk = store().save(text="重复产生的垃圾", kind="experiment_report", title="垃圾")

    r = client.post(f"/api/documents/{junk.doc_id}/discard", params={"reason": "重复"})
    assert r.status_code == 200
    assert r.json()["root_kind"] == "discarded"
    assert "未删除" in (r.json().get("detail") or "")

    assert _titles(client.get("/api/documents").json()) == {"留着"}
    assert _titles(client.get("/api/documents",
                              params={"include_discarded": True}).json()) == {"留着", "垃圾"}

    # 正文照样读得出来 —— 一个字节都没删
    got = client.get(f"/api/documents/{junk.doc_id}").json()
    assert got["content"] == "重复产生的垃圾"
    assert got["root_kind"] == "discarded"


def test_restore_puts_it_back(scoped, client):
    _st, eid = scoped
    junk = store().save(text="其实有用", kind="experiment_report", title="误废")
    client.post(f"/api/documents/{junk.doc_id}/discard")

    r = client.post(f"/api/documents/{junk.doc_id}/restore",
                    params={"experiment_id": eid})
    assert r.status_code == 200 and r.json()["root_kind"] == "experiment"
    assert _titles(client.get("/api/documents").json()) == {"误废"}
    assert store().get(junk.doc_id).read_text() == "其实有用"


def test_restore_uses_the_owner_the_document_remembers(scoped, client):
    """不给 experiment_id 时用文档**自己记着的**主归属 —— 那不是猜，是它带着的事实。

    `discard` 刻意不清 `experiment_id`（区由路径表达）。把东西丢进未归属区让用户
    再找一遍原主，才是白扔掉已知信息。
    """
    _st, eid = scoped
    junk = store().save(text="x", kind="experiment_report", title="记得原主")
    client.post(f"/api/documents/{junk.doc_id}/discard")
    r = client.post(f"/api/documents/{junk.doc_id}/restore",
                    params={"experiment_id": ""})
    assert r.status_code == 200
    assert r.json()["root_kind"] == "experiment"
    assert r.json()["experiment_id"] == eid


def test_discarding_an_unknown_document_is_a_404(scoped, client):
    r = client.post("/api/documents/NOSUCHDOC/discard")
    assert r.status_code == 404
    assert r.json().get("detail")
