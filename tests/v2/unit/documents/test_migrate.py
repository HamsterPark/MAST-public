"""旧全局文档 → 实验文件夹的迁移。幂等，复制导入，不销毁原文件。

设计文档：docs/v2/design/document_and_library_management.md §3.12
"""

from __future__ import annotations


import pytest

from mast.documents import reset_caches, store
from mast.documents.migrate import migrate_all, migrate_legacy_markdown, migrate_plans
from mast.documents.paths import exp_dir_for
from mast.logging.storage import ExperimentStorage
from mast.logging.v2 import manifest as mf
from mast.planning.plan_store import ExperimentPlan, PlanPhase, PlanStore


@pytest.fixture()
def env(tmp_path, monkeypatch):
    root = tmp_path / "MAST-Data" / "experiments"
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "db" / "exp.db"))
    monkeypatch.setenv("MAST_DRAFTS_DIR", str(tmp_path / "data" / "drafts"))
    monkeypatch.setenv("MAST_REVIEWS_DIR", str(tmp_path / "data" / "reviews"))
    root.mkdir(parents=True, exist_ok=True)
    reset_caches()
    yield tmp_path
    reset_caches()


def _experiment(title: str = "旧世界实验") -> tuple[ExperimentStorage, str]:
    from mast.agents._shared.data_paths import experiment_db_path
    st = ExperimentStorage(experiment_db_path())
    eid = st.create_experiment(title, "")
    exp_dir = exp_dir_for(eid, create=True)
    mf.write_experiment_manifest(exp_dir, experiment_id=eid, title=title,
                                 dir_name=exp_dir.name)
    return st, eid


def _old_world_plan_store(monkeypatch, db_path, plans_dir) -> PlanStore:
    """一个**不做文档同步**的 PlanStore —— 这就是迁移前的世界。

    比「先 save 再把 doc_id 抹掉」真实：那样会在磁盘上留下一个孤儿文档，让
    「迁移后应该只有一份计划文档」的断言凭空多出一个。
    """
    monkeypatch.setattr(PlanStore, "_doc_target", lambda self, plan: None)
    return PlanStore(db_path, plans_dir=plans_dir)


def test_legacy_plan_markdown_is_imported_to_its_experiment(env, monkeypatch):
    from mast.agents._shared.data_paths import experiment_db_path
    _st, eid = _experiment()
    legacy = env / "old_plans"
    ps = _old_world_plan_store(monkeypatch, experiment_db_path(), legacy)
    plan = ExperimentPlan(plan_id="", experiment_id=eid, sample_id="",
                          name="旧格式计划", goal="",
                          phases=[PlanPhase(id="x", name="X", steps=[])])
    pid = ps.save(plan)
    md = legacy / f"plan_{pid}.md"
    md.write_text("# 旧格式计划\n目标: 老的\n## Phase 1: X ✅\n- [x] 手写内容\n",
                  encoding="utf-8")

    res = migrate_plans(plan_store=ps)
    assert res["imported"] == 1 and not res["errors"], res
    doc_id = ps.doc_id_for(pid)
    assert doc_id
    entry = store().get(doc_id)
    assert entry.meta.experiment_id == eid
    assert "手写内容" in entry.read_text()
    assert md.is_file(), "复制导入 —— 原文件不动"


def test_plan_migration_is_idempotent(env, monkeypatch):
    from mast.agents._shared.data_paths import experiment_db_path
    _st, eid = _experiment()
    legacy = env / "old_plans"
    ps = _old_world_plan_store(monkeypatch, experiment_db_path(), legacy)
    pid = ps.save(ExperimentPlan(plan_id="", experiment_id=eid, sample_id="",
                                 name="P", goal="",
                                 phases=[PlanPhase(id="x", name="X")]))
    (legacy / f"plan_{pid}.md").write_text("# P\n内容\n", encoding="utf-8")

    first = migrate_plans(plan_store=ps)
    second = migrate_plans(plan_store=ps)
    assert first["imported"] == 1
    assert second["imported"] == 0 and second["skipped"] >= 1
    assert len([e for e in store().list(experiment_id=eid)
                if e.meta.kind == "experiment_plan"]) == 1


def test_plan_without_experiment_goes_to_unfiled(env, monkeypatch):
    from mast.agents._shared.data_paths import experiment_db_path
    legacy = env / "old_plans"
    ps = _old_world_plan_store(monkeypatch, experiment_db_path(), legacy)
    pid = ps.save(ExperimentPlan(plan_id="", experiment_id="", sample_id="",
                                 name="无归属", goal="",
                                 phases=[PlanPhase(id="x", name="X")]))
    (legacy / f"plan_{pid}.md").write_text("# 无归属\n内容\n", encoding="utf-8")
    res = migrate_plans(plan_store=ps)
    assert res["imported"] == 1 and res["unfiled"] == 1
    doc_id = ps.doc_id_for(pid)
    assert store().get(doc_id).meta.root_kind == "unfiled"


def test_draft_version_family_becomes_one_document_with_versions(env):
    """``<stem>_v001/_v002/_v003`` 是同一份文档的历史，导成一个文档的三个版本。

    这是旧命名方案里唯一还能救回来的结构信息 —— 拆成三个独立文档就把历史弄丢了。
    """
    drafts = env / "data" / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    for v, body in ((1, "第一版"), (2, "第二版"), (3, "第三版")):
        (drafts / f"Au111_report_v{v:03d}.md").write_text(body, encoding="utf-8")
    (drafts / "另一份_v001.md").write_text("别的文档", encoding="utf-8")

    res = migrate_legacy_markdown()
    assert res["drafts"] == 2, res
    assert res["versions"] == 4, res

    docs = store().list()
    fam = [d for d in docs if d.meta.legacy_stem == "Au111_report"]
    assert len(fam) == 1
    entry = fam[0]
    assert [v.v for v in entry.versions] == [1, 2, 3]
    assert entry.read_text(1) == "第一版"
    assert entry.read_text() == "第三版", "最新版必须是 v003"


def test_review_family_is_imported_as_review_kind(env):
    reviews = env / "data" / "reviews"
    reviews.mkdir(parents=True, exist_ok=True)
    (reviews / "Au111_report_review_v001.md").write_text(
        "<!-- verdict: REVISE -->\n\n1. 缺误差棒", encoding="utf-8")
    res = migrate_legacy_markdown()
    assert res["reviews"] == 1
    entry = next(d for d in store().list() if d.meta.kind == "review")
    assert "verdict" in entry.read_text()


def test_markdown_migration_is_idempotent(env):
    drafts = env / "data" / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    (drafts / "报告_v001.md").write_text("内容", encoding="utf-8")
    first = migrate_legacy_markdown()
    second = migrate_legacy_markdown()
    assert first["drafts"] == 1
    assert second["drafts"] == 0 and second["skipped"] >= 1
    assert len(store().list()) == 1


def test_dry_run_writes_nothing(env):
    drafts = env / "data" / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    (drafts / "报告_v001.md").write_text("内容", encoding="utf-8")
    res = migrate_legacy_markdown(dry_run=True)
    assert res["drafts"] == 1 and res["dry_run"] is True
    assert store().list() == []


def test_migrate_all_never_raises_on_a_broken_corner(env, monkeypatch):
    """迁移是运维动作 —— 一个角落坏掉不该让整件事炸。"""
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(env / "nope" / "missing.db"))
    out = migrate_all(dry_run=True)
    assert "plans" in out and "markdown" in out


def test_reindex_honours_an_explicit_root(env, monkeypatch, tmp_path):
    """★ M6:``rebuild_from_folders(root=A)`` 必须真的从 A 重建文档,不是从全局配置根。

    ``root`` 从前被**吞掉**:只用来清缓存,随后 ``_scan_all`` 自己从
    ``experiment_paths.experiment_root()`` 解析 —— 于是同一次调用会从 A 重建实验/样品、
    却从另一个根重建文档。签名读起来像支持,实际不支持。
    """
    from mast.logging.v2.reindex import rebuild_from_folders

    _st, eid = _experiment("显式根实验")
    store().save(text="正文", kind="experiment_report", title="显式根文档")

    # 把 env 指到一个**空**的另一个根；显式 root= 仍应指向真正有东西的那个
    real_root = env / "MAST-Data" / "experiments"
    empty_root = tmp_path / "somewhere-else" / "experiments"
    empty_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(empty_root))
    reset_caches()

    fresh = ExperimentStorage(str(tmp_path / "fresh.db"))
    res = rebuild_from_folders(storage=fresh, root=real_root)
    assert res["documents_added"] == 1, (
        f"显式 root 被忽略了（从空根重建出 {res['documents_added']} 份文档）")
    assert not res["errors"], res["errors"]
    assert [r["title"] for r in fresh.list_documents()] == ["显式根文档"]


# ── 幂等的第二把钥匙（2026-08-02 打包审计）─────────────────────────────────

def test_plan_migration_survives_a_failed_doc_id_writeback(env, monkeypatch):
    """★ doc_id 回写失败之后重跑,不能再导一份。

    `migrate_plans` 原来只看 `plans.doc_id`。那一列是循环**末尾**回写的,而回写失败
    只 append 到 errors、**不回滚**已经建好的文档 —— 于是一次半成功的运行会留下
    「文档已存在、plans.doc_id 仍为空」的状态,下一次重跑把同一份计划又导一遍。

    `migrate_legacy_markdown` 从来没有这个洞,因为它认的是 `legacy_stem`
    （文档自己带着的那把钥匙,不依赖任何回写）。这里补上同一把钥匙。
    """
    from contextlib import closing

    from mast.agents._shared.data_paths import experiment_db_path

    _st, eid = _experiment()
    legacy = env / "old_plans"
    ps = _old_world_plan_store(monkeypatch, experiment_db_path(), legacy)
    pid = ps.save(ExperimentPlan(plan_id="", experiment_id=eid, sample_id="",
                                 name="半成功计划", goal="",
                                 phases=[PlanPhase(id="x", name="X")]))
    (legacy / f"plan_{pid}.md").write_text("# P\n内容\n", encoding="utf-8")

    first = migrate_plans(plan_store=ps)
    assert first["imported"] == 1 and not first["errors"], first

    # 模拟「文档建好了,但 doc_id 回写没成功」——把那一列抹回空。
    with closing(ps._connect()) as conn, conn:
        conn.execute("UPDATE plans SET doc_id='' WHERE plan_id=?", (pid,))
    assert not ps.doc_id_for(pid)

    second = migrate_plans(plan_store=ps)

    assert second["imported"] == 0, "legacy_stem 那把钥匙应当认出它已经导过了"
    assert second["skipped"] >= 1
    plans_docs = [e for e in store().list(experiment_id=eid)
                  if e.meta.kind == "experiment_plan"]
    assert len(plans_docs) == 1, f"重复导入了 {len(plans_docs)} 份"


def test_both_migration_modules_expose_a_runnable_entry_point():
    """零调用方的模块 = 谁也跑不了的模块（2026-08-02 审计第 2 条）。

    两个迁移模块都没有进程内的自动触发（那是刻意的:它们会复制大量文件,是运维
    决定而不是启动副作用),所以至少要有一个能跑的入口。
    """
    from mast.documents import migrate as doc_mig
    from mast.logging.v2 import migrate_folders as folder_mig

    assert callable(getattr(doc_mig, "main", None))
    assert callable(getattr(folder_mig, "main", None))
    # --help 必须能自己跑完(argparse 配置错了会在这里抛)
    for mod in (doc_mig, folder_mig):
        with pytest.raises(SystemExit) as exc:
            mod.main(["--help"])
        assert exc.value.code == 0
