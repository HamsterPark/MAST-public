"""documents API 在**半接线**进程里的行为。

`AppContext` 没接 `experiment_storage` 时（纯 API 测试、离线脚本、启动早期），
文档读路径必须照样可用 —— 权威是实验文件夹，不是 DB。

这里专门钉住一条曾经悄悄坏掉的列：`_storage()` 拿不到 ctx 就直接返回 None，于是
`experiment_name` **恒为空字符串**，而「这份文档属于哪个实验」正是关联文档那一栏
唯一的信息量（前端会渲染成「属于《》」）。
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
    eid = st.create_experiment("Au(111) 形貌与 STS", "")
    other = st.create_experiment("NbSe2 CDW", "")
    st.set_active_scope(eid, None, updated_by="test")
    for e, title in ((eid, "Au(111) 形貌与 STS"), (other, "NbSe2 CDW")):
        d = exp_dir_for(e, create=True)
        mf.write_experiment_manifest(d, experiment_id=e, title=title, dir_name=d.name)
    reset_caches()
    return st, eid, other


@pytest.fixture()
def client(env):
    """未接线的 app —— ctx 里没有 experiment_storage。"""
    return TestClient(create_app(dev_cors=False))


def test_experiment_name_is_resolved_without_a_wired_context(scoped, client):
    _st, eid, _other = scoped
    store().save(text="正文", kind="experiment_report", title="阶段总结")

    rows = client.get("/api/documents").json()["documents"]
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == eid
    assert rows[0]["experiment_name"] == "Au(111) 形貌与 STS"


def test_related_document_shows_the_owning_experiment_name(scoped, client):
    _st, eid, other = scoped
    res = store().save(text="跨实验论文", kind="paper_draft", title="综合论文")
    store().patch(res.doc_id, related_experiment_ids=[other])

    rows = client.get("/api/documents", params={"experiment_id": other}).json()["documents"]
    assert len(rows) == 1
    assert rows[0]["relation"] == "related"
    # 关联栏渲染的是「属于《X》」—— 名字为空这一栏就没有信息量
    assert rows[0]["experiment_name"] == "Au(111) 形貌与 STS"


def test_document_body_is_readable_with_no_db_index(scoped, client):
    """DB 索引删空也照样能读 —— 正文从来只在文件夹里。"""
    from mast.agents._shared.data_paths import experiment_db_path
    res = store().save(text="# 报告\n\n正文在文件夹里。", kind="experiment_report",
                       title="不依赖 DB")
    st = ExperimentStorage(experiment_db_path())
    with st._connect() as conn:
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM document_versions")
    assert st.list_documents() == []

    got = client.get(f"/api/documents/{res.doc_id}").json()
    assert "正文在文件夹里" in (got.get("content") or "")
    assert client.get("/api/documents").json()["documents"], "列表也不该依赖 DB"
