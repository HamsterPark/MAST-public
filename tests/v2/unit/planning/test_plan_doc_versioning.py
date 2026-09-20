"""计划的「定义修订才是版本，进度是事件 + 视图」。

设计文档：docs/v2/design/document_and_library_management.md §3.7

旧实现只有一个 ``plan_<id>.md``，``save()`` 和 ``update_progress()`` 都往它裸
``write_text`` —— **推进一个阶段就吃掉上一份计划**，写到一半崩就是半个文件。
"""

from __future__ import annotations


import pytest

from mast.documents import reset_caches, store
from mast.documents.paths import exp_dir_for
from mast.logging.storage import ExperimentStorage
from mast.logging.v2 import manifest as mf
from mast.planning.plan_store import (
    ExperimentPlan,
    PlanPhase,
    PlanStatus,
    PlanStore,
)


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
def scoped(env, tmp_path):
    from mast.agents._shared.data_paths import experiment_db_path
    st = ExperimentStorage(experiment_db_path())
    eid = st.create_experiment("过夜 STS 实验", "测计划版本化")
    st.set_active_scope(eid, None, updated_by="test")
    exp_dir = exp_dir_for(eid, create=True)
    mf.write_experiment_manifest(exp_dir, experiment_id=eid,
                                 title="过夜 STS 实验", dir_name=exp_dir.name)
    ps = PlanStore(experiment_db_path(), plans_dir=tmp_path / "legacy_plans")
    return st, eid, ps


def _plan(eid: str, extra_phase: bool = False) -> ExperimentPlan:
    phases = [
        PlanPhase(id="approach", name="进针", steps=[{"skill": "ApproachTip", "params": {}}]),
        PlanPhase(id="scan", name="扫图", steps=[{"skill": "StartScan", "params": {"n": 512}}]),
    ]
    if extra_phase:
        phases.append(PlanPhase(id="withdraw", name="退针",
                                steps=[{"skill": "Withdraw", "params": {}}]))
    return ExperimentPlan(plan_id="", experiment_id=eid, sample_id="",
                          name="过夜 STS 扫描", goal="拿谱", phases=phases,
                          status=PlanStatus.RUNNING)


def test_save_creates_definition_doc_in_experiment_folder(scoped):
    _st, eid, ps = scoped
    pid = ps.save(_plan(eid))
    doc_id = ps.doc_id_for(pid)
    assert doc_id, "plan 行必须记下它的文档 id"
    entry = store().get(doc_id)
    assert entry is not None
    assert entry.meta.kind == "experiment_plan"
    assert entry.meta.experiment_id == eid
    assert entry.latest_version == 1
    assert entry.dir.parent.name == "plans"


def test_definition_text_carries_no_progress_markers(scoped):
    """定义里不能有进度痕迹，否则每次推进都会产生一个新版本。"""
    _st, eid, ps = scoped
    pid = ps.save(_plan(eid))
    text = store().get(ps.doc_id_for(pid)).read_text()
    for marker in ("当前步骤", "✅", "🔄", "[x]", "状态:"):
        assert marker not in text


def test_progress_updates_do_not_create_versions(scoped):
    """一个 10 阶段计划推完约 30 次进度写；每次发版本就是版本爆炸。"""
    _st, eid, ps = scoped
    pid = ps.save(_plan(eid))
    doc_id = ps.doc_id_for(pid)

    ps.update_progress(pid, 0, 1, "done")
    ps.update_progress(pid, 1, 0, "running")
    ps.update_progress(pid, 1, 1, "running")

    entry = store().get(doc_id)
    assert entry.latest_version == 1, "进度不是版本"
    events = store().read_events(doc_id, "progress.jsonl")
    assert len(events) == 3
    assert all(e["op"] == "progress" for e in events)
    assert (entry.dir / "progress.md").is_file()


def test_status_change_is_an_event_with_its_note(scoped):
    _st, eid, ps = scoped
    pid = ps.save(_plan(eid))
    doc_id = ps.doc_id_for(pid)
    ps.update_status(pid, PlanStatus.PAUSED, notes="等液氦")
    events = store().read_events(doc_id, "progress.jsonl")
    assert events[-1]["op"] == "status"
    assert events[-1]["plan_status"] == "paused"
    assert events[-1]["notes"] == "等液氦"
    assert store().get(doc_id).latest_version == 1


def test_unchanged_definition_does_not_bump_version(scoped):
    _st, eid, ps = scoped
    pid = ps.save(_plan(eid))
    doc_id = ps.doc_id_for(pid)
    ps.save(ps.load(pid))
    ps.save(ps.load(pid))
    assert store().get(doc_id).latest_version == 1


def test_real_definition_change_bumps_version_and_keeps_the_old_one(scoped):
    _st, eid, ps = scoped
    pid = ps.save(_plan(eid))
    doc_id = ps.doc_id_for(pid)

    revised = ps.load(pid)
    revised.phases.append(PlanPhase(id="withdraw", name="退针",
                                    steps=[{"skill": "Withdraw", "params": {}}]))
    ps.save(revised)

    entry = store().get(doc_id)
    assert entry.latest_version == 2
    assert "退针" in entry.read_text()
    assert "退针" not in entry.read_text(1), "v001 不可被改写"


def test_save_keeps_doc_id_across_insert_or_replace(scoped):
    """``INSERT OR REPLACE`` 会把列清单里没写的列清成 NULL。

    漏掉 ``doc_id`` 的后果不是报错而是静默失忆：每次 save 都以为这个计划还没有
    文档，于是**另立一份新文档**，同一个计划在 plans/ 里越攒越多。
    """
    _st, eid, ps = scoped
    pid = ps.save(_plan(eid))
    first = ps.doc_id_for(pid)
    for _ in range(3):
        ps.save(ps.load(pid))
    assert ps.doc_id_for(pid) == first
    plan_docs = [e for e in store().list(experiment_id=eid)
                 if e.meta.kind == "experiment_plan"]
    assert len(plan_docs) == 1, f"计划文档被重复创建了：{len(plan_docs)} 份"


def test_plan_without_experiment_falls_back_locally_and_writes_atomically(env, tmp_path):
    """没有实验归属就不碰实验文件夹（否则用假 id 的测试会往真实数据根里写）。"""
    from mast.agents._shared.data_paths import experiment_db_path
    legacy = tmp_path / "legacy_plans"
    ps = PlanStore(experiment_db_path(), plans_dir=legacy)
    plan = ExperimentPlan(plan_id="", experiment_id="", sample_id="", name="孤儿计划",
                          goal="", phases=[PlanPhase(id="a", name="A", steps=[])])
    pid = ps.save(plan)
    assert (legacy / f"plan_{pid}.md").is_file()
    assert ps.doc_id_for(pid) is None
    assert not list(legacy.glob("*.part")), "原子替换不该留下 .part"


def test_plan_pointing_at_a_missing_experiment_stays_local(env, tmp_path):
    """实验行不存在（测试里很常见的假 id）→ 同样退回本地，不落 _unfiled。"""
    from mast.agents._shared.data_paths import experiment_db_path
    legacy = tmp_path / "legacy_plans"
    ps = PlanStore(experiment_db_path(), plans_dir=legacy)
    plan = ExperimentPlan(plan_id="", experiment_id="does-not-exist", sample_id="",
                          name="假 id", goal="", phases=[PlanPhase(id="a", name="A")])
    pid = ps.save(plan)
    assert (legacy / f"plan_{pid}.md").is_file()
    assert ps.doc_id_for(pid) is None


def test_markdown_path_resolves_to_the_definition_version(scoped):
    _st, eid, ps = scoped
    pid = ps.save(_plan(eid))
    p = ps.markdown_path(pid)
    assert p.name == "v001.md"
    assert p.is_file()
