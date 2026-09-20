"""``/api/artifacts/{id}/{edit,history,diff,export}`` —— **兼容层**。

这五个 URL 是旧对象编辑器（``ArtifactEditor.tsx``）用的，内部已改走
``mast.documents`` store。它们存在的唯一理由是让前端在切到 ``/api/documents``
之前继续工作；切换后整个模块会被删（设计 §5.3 P2）。所以这里断言两件事：

1. **旧 URL 与旧响应形状仍然成立** —— 前端零改动还能用；
2. **语义没有退化** —— 保存 = 新版本、回退不删任何东西、历史与 diff 来自磁盘、
   非文件型产物诚实拒绝并指出它真正存在哪里。

这个文件更早的版本钉的是相反的东西：它断言保存落进
``live_app._agents_api_state["artifact_edits"]``，"orchestrator 会读的那个槽"，
然后 GET 把它回显出来。每次都过 —— 而功能什么都没做。**没有任何 agent、图节点
或 orchestrator 读过那个 dict。** 所以每个测试都以一个 agent 工具收尾
（``paper_review.load_draft`` / ``paper_writing.load_review``），从不以 API 回显收尾。
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.v2.toolcall import tool_call
from mast.api.context import AppContext
from mast.api.routes.artifacts_edit import router
from mast.documents import reset_caches, store
from mast.logging.storage import ExperimentStorage

_DOC_ID_RE = re.compile(r"doc_id\s*=\s*(\S+)")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(tmp_path / "experiments"))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "db" / "mast_experiments.db"))
    # list_existing() 会枚举 figures，真实 data/figures 里任何一次 plot_scan 留下的
    # 图都会混进结果（2026-07-27 实际发生过）。指到 tmp 才隔离得干净。
    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
    monkeypatch.setenv("MAST_DRAFTS_DIR", str(tmp_path / "drafts"))
    monkeypatch.setenv("MAST_REVIEWS_DIR", str(tmp_path / "reviews"))
    st = ExperimentStorage(tmp_path / "db" / "mast_experiments.db")
    eid = st.create_experiment("Au111 形貌与 STS", "测 Au(111) 台阶")
    st.set_active_scope(eid, None, updated_by="test")
    import mast.logging.experiment_log as el
    monkeypatch.setattr(el, "get_active_log", lambda: None)
    reset_caches()
    yield {"storage": st, "eid": eid}
    reset_caches()


@pytest.fixture()
def client(env) -> TestClient:
    app = FastAPI()
    ctx = AppContext()
    ctx.wire(experiment_storage=env["storage"])
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _seed_draft(title: str = "Au111", body: str = "# T\nAGENT ORIGINAL") -> str:
    """让 paper_writing agent 真的存一份报告，返回它的 doc_id。"""
    from mast.agents.paper_writing.tools import save_draft

    out = save_draft.invoke(tool_call(save_draft, {"title": title, "markdown_text": body}))
    m = _DOC_ID_RE.search(str(out))
    assert m, f"save_draft 没有在返回值里给出 doc_id：{out}"
    return m.group(1)


# ════════════════════════════════════════════════════════════════════════
# 核心问题：我改完之后，agent 读到的是我这一版吗？
# ════════════════════════════════════════════════════════════════════════

class TestTheEditReachesTheAgent:
    def test_saved_edit_is_what_the_reviewer_loads(self, client):
        from mast.agents.paper_review.tools import load_draft

        did = _seed_draft()
        r = client.post(f"/api/artifacts/{did}/edit",
                        json={"body": "# T\nOPERATOR CORRECTED THIS"}).json()
        assert r["ok"] is True and r["degraded"] is False

        assert "OPERATOR CORRECTED THIS" in load_draft.invoke({"draft_id": did}), (
            "评审 agent 看不到用户的修改 —— 它去了别的地方，这正是本模块曾经的 bug")

    def test_the_agents_own_version_survives_the_edit(self, client):
        """保存 = 新版本，不是覆盖。因为人碰了文件就销毁 agent 的成果是数据丢失。"""
        did = _seed_draft()
        client.post(f"/api/artifacts/{did}/edit", json={"body": "# T\nEDITED"})

        h = client.get(f"/api/artifacts/{did}/history").json()
        assert h["degraded"] is False
        assert h["count"] == 1, "agent 的原始版本被销毁了"
        assert "AGENT ORIGINAL" in h["entries"][0]["body"]
        assert "EDITED" in h["current"]["body"]

    def test_a_review_edit_is_read_back_by_paper_writing(self, client):
        from mast.agents.paper_review.tools import save_review
        from mast.agents.paper_writing.tools import load_review

        did = _seed_draft()
        out = save_review.invoke(tool_call(save_review, {"target_doc_id": did, "verdict": "REVISE",
                                  "report_markdown": "## Overall Verdict: REVISE\n1. fix X"}))
        m = _DOC_ID_RE.search(str(out))
        assert m, out
        rid = m.group(1)

        r = client.post(f"/api/artifacts/{rid}/edit", json={
            "body": "## Overall Verdict: REVISE\n1. fix X\n2. AND ALSO fix Y"}).json()
        assert r["ok"] is True, r

        assert "AND ALSO fix Y" in load_review.invoke({"draft_name": "latest"})


# ════════════════════════════════════════════════════════════════════════
# 旧 id 形式仍然解析得到
# ════════════════════════════════════════════════════════════════════════

class TestLegacyIdsStillResolve:
    def test_old_kind_stem_id_resolves_via_legacy_stem(self, client):
        """迁移进来的文档带 ``doc.json.legacy_stem``，旧 ``draft:<stem>`` id 照样能开。

        这是兼容层的全部价值：旧对话、旧书签、``ArtifactEditor`` 里存着的 id
        不会一夜之间全部失效。
        """
        res = store().save(text="# T\n迁移进来的正文", kind="experiment_report",
                           title="Au111 报告", created_by="operator",
                           legacy_stem="Au111_report_v003")
        assert res.ok, res.error

        b = client.get("/api/artifacts/draft:Au111_report_v003/export",
                       params={"format": "json"}).json()
        assert b["ok"] is True and "迁移进来的正文" in (b["body"] or "")

    def test_every_id_the_artifact_list_hands_out_opens(self, client):
        """``list_existing`` 给出的每个 id 都必须能在编辑器里打开。

        它曾经给每份草稿都发 ``artifact_id="draft"``：盘上五份草稿时编辑器根本
        说不出要开哪一份，只能落在后端先解析到的那份上。
        """
        from mast.agents._shared.artifacts import list_existing

        ids = [_seed_draft("first", "# a\nbody"),
               _seed_draft("second", "# b\nbody")]
        for did in ids:
            b = client.get(f"/api/artifacts/{did}/export",
                           params={"format": "json"}).json()
            assert b["ok"] is True, f"{did} 解析不到文件"

        rows = [r for r in list_existing() if r["artifact_id"] in ("draft", "review")]
        assert rows, "list_existing 没有枚举出任何文档"
        doc_ids = [r["doc_id"] for r in rows]
        assert len(doc_ids) == len(set(doc_ids)), f"id 撞了：{doc_ids}"
        for did in doc_ids:
            b = client.get(f"/api/artifacts/{did}/export",
                           params={"format": "json"}).json()
            assert b["ok"] is True, f"列表给出的 {did} 在编辑器里打不开"


# ════════════════════════════════════════════════════════════════════════
# 回退 —— 一个什么都不销毁的 undo
# ════════════════════════════════════════════════════════════════════════

class TestRevert:
    def test_revert_takes_the_agent_back_to_the_previous_version(self, client):
        from mast.agents.paper_review.tools import load_draft

        did = _seed_draft()
        client.post(f"/api/artifacts/{did}/edit", json={"body": "# T\nA BAD EDIT"})
        assert "A BAD EDIT" in load_draft.invoke({"draft_id": did})

        r = client.delete(f"/api/artifacts/{did}/edit").json()
        assert r["ok"] is True and r["reverted"] is True

        back = load_draft.invoke({"draft_id": did})
        assert "AGENT ORIGINAL" in back and "A BAD EDIT" not in back

    def test_revert_deletes_nothing(self, client):
        """回退 = 把旧正文重新存成新版本。坏编辑还在盘上 —— undo 绝不能是全 app 里
        唯一会丢数据的按钮。"""
        did = _seed_draft()
        client.post(f"/api/artifacts/{did}/edit", json={"body": "# T\nA BAD EDIT"})
        client.delete(f"/api/artifacts/{did}/edit")

        h = client.get(f"/api/artifacts/{did}/history").json()
        bodies = [e["body"] for e in h["entries"]] + [h["current"]["body"]]
        assert len(bodies) == 3, f"有一个版本被销毁了：{len(bodies)}"
        assert any("A BAD EDIT" in b for b in bodies), "被回退掉的那一版被删了"

    def test_revert_with_no_history_is_an_honest_no_op(self, client):
        did = _seed_draft()
        r = client.delete(f"/api/artifacts/{did}/edit").json()
        assert r["degraded"] is False
        assert r["reverted"] is False
        assert "没有可回退" in (r["detail"] or "")


# ════════════════════════════════════════════════════════════════════════
# 历史 / diff / 导出 —— 来自磁盘，不是一个重启即死的 dict
# ════════════════════════════════════════════════════════════════════════

class TestHistoryDiffExport:
    def test_diff_compares_the_real_previous_version(self, client):
        """旧 diff 比的是 ``task["artifacts"]["produced"]`` —— 全树无人写过的槽 ——
        所以 original 侧永远是空的，每次 diff 都把整篇报成新增。"""
        did = _seed_draft(body="# T\nline one\nline two")
        client.post(f"/api/artifacts/{did}/edit",
                    json={"body": "# T\nline one\nline two CHANGED"})

        d = client.get(f"/api/artifacts/{did}/diff").json()
        assert d["degraded"] is False
        assert d["has_original"] is True and d["changed"] is True
        assert "-line two" in d["diff"] and "+line two CHANGED" in d["diff"]

    def test_diff_of_a_single_version_is_not_all_new(self, client):
        did = _seed_draft()
        d = client.get(f"/api/artifacts/{did}/diff").json()
        assert d["degraded"] is False
        assert d["has_original"] is False
        assert d["changed"] is False, "单个版本被 diff 成了全篇新增"

    def test_history_comes_from_disk_not_a_per_process_dict(self, client):
        """旧实现把历史放在进程内 dict 里，所以每次冷启动都是空的 —— 而它旁边的
        目录里躺着一堆版本。这里换一个新 client（新 app 实例）再读一遍。"""
        did = _seed_draft()
        client.post(f"/api/artifacts/{did}/edit", json={"body": "# T\nSECOND"})

        app = FastAPI()
        ctx = AppContext()
        app.state.ctx = ctx
        app.include_router(router, prefix="/api")
        fresh = TestClient(app)
        h = fresh.get(f"/api/artifacts/{did}/history").json()
        assert h["degraded"] is False and h["count"] == 1
        assert "SECOND" in h["current"]["body"]

    def test_export_streams_the_newest_version(self, client):
        did = _seed_draft()
        client.post(f"/api/artifacts/{did}/edit", json={"body": "# T\nNEWEST"})
        r = client.get(f"/api/artifacts/{did}/export")
        assert r.status_code == 200
        assert "NEWEST" in r.text
        assert "attachment" in r.headers.get("content-disposition", "")

    def test_export_json_shape(self, client):
        did = _seed_draft()
        b = client.get(f"/api/artifacts/{did}/export", params={"format": "json"}).json()
        assert b["ok"] is True and b["degraded"] is False
        assert "AGENT ORIGINAL" in b["body"]
        assert b["source"] == "file"
        assert b["filename"] and b["filename"].endswith("_v001.md"), (
            "文件名要认得出是哪份的哪一版")


# ════════════════════════════════════════════════════════════════════════
# 不可编辑的必须说清 —— 黑洞是从一次假的「好」开始的
# ════════════════════════════════════════════════════════════════════════

class TestNonEditableArtifactsRefuse:
    @pytest.mark.parametrize("artifact_id,store_hint", [
        ("experiment_records", "mast_experiments.db"),
        ("vision_buffer", "vision_buffer"),
        ("scan_files", "Nanonis"),
        ("literature_library", "registry.json"),
        ("memory", "cognition"),
    ])
    def test_refuses_and_names_the_real_store(self, client, artifact_id, store_hint):
        """SQLite 库和 .sxm 不可能通过一个 textarea 手改。收下文本再悄悄丢掉正是
        本模块过去干的事 —— 所以拒绝，并告诉用户这东西真正存在哪里。"""
        r = client.post(f"/api/artifacts/{artifact_id}/edit",
                        json={"body": "nice try"}).json()
        assert r["degraded"] is True and r["ok"] is False
        assert store_hint in (r["detail"] or ""), (
            f"拒绝文案没说 {artifact_id} 存在哪里：{r['detail']}")

    def test_an_unknown_id_is_refused_not_silently_accepted(self, client):
        r = client.post("/api/artifacts/not-a-thing/edit", json={"body": "x"}).json()
        assert r["degraded"] is True and r["ok"] is False

    @pytest.mark.parametrize("bad", ["draft:../etc/passwd", "draft:", "bogus:x"])
    def test_traversal_and_malformed_ids_refused(self, client, bad):
        r = client.post(f"/api/artifacts/{bad}/edit", json={"body": "x"})
        assert r.status_code == 200          # 永不 500 / traceback
        assert r.json()["degraded"] is True

    @pytest.mark.parametrize("vague", ["current", "latest"])
    def test_vague_refs_are_refused_by_the_compat_layer(self, client, vague):
        """``resolve_ref`` 会把 ``current`` 解析成「最近更新的那一份」。兼容层的每次
        调用都是写某一份**具体**文档，落到别人的文档上不可接受 —— 想要「当前那一
        份」请用 ``/api/documents``。"""
        _seed_draft()
        r = client.post(f"/api/artifacts/{vague}/edit", json={"body": "x"}).json()
        assert r["degraded"] is True and r["ok"] is False

    def test_empty_body_refused(self, client):
        did = _seed_draft()
        r = client.post(f"/api/artifacts/{did}/edit", json={"body": "  \n "}).json()
        assert r["degraded"] is True
        assert "空" in (r["detail"] or "")
        h = client.get(f"/api/artifacts/{did}/history").json()
        assert h["count"] == 0, "被拒绝的空保存不能留下版本"


# ════════════════════════════════════════════════════════════════════════
# 兼容层不得自己分配版本号
# ════════════════════════════════════════════════════════════════════════

def test_module_does_not_call_next_version_path():
    """``next_version_path`` 是 glob-then-write（无锁、无 ``O_EXCL``）：两个并发写
    入者会算出同一个 ``_v003``，**后写者静默覆盖前者**。这是「永不覆盖」承诺的
    破口（设计陷阱 ①），版本分配只能经 store。"""
    from pathlib import Path

    src = Path(__file__).resolve().parents[4] / "MASTv2/mast/api/routes/artifacts_edit.py"
    body = src.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "next_version_path(" not in code
