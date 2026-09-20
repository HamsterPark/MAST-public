"""Artifact status must join class ids with produced file kinds, enumerate each storage backend, and reject edits addressed only to a class id."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from tests.v2.toolcall import tool_call
from mast.agents._shared.artifacts import (  # noqa: E402
    ARTIFACTS,
    TOOL_ACCESS,
    agent_tool_names,
    class_status,
    list_existing,
)


@pytest.fixture()
def isolated(tmp_path, monkeypatch, documents_root):
    """Point every store ``class_status()`` probes at tmp_path.

    One redirect per store, and missing one does not fail loudly — it silently
    reads the developer's real data. (That is how this fixture's ancestor in
    test_artifacts_edit.py started failing: the day someone first ran plot_scan
    for real, their figures showed up in an unrelated id test.) The two
    baseline assertions at the bottom are the tripwire for the next one.

    The literature registry needs a different lever: its directory is a
    module-level constant off a private ``_find_repo_root()`` walk, so it
    ignores the project-root env var entirely — the same shape as the wishlist
    board that quietly collected 25 junk rows in the operator's real file (see
    tests/v2/conftest.py). This probe only READS, so nothing is corrupted, but
    it cannot be redirected either; the process-wide singleton has to be
    replaced instead.

    Documents needed a third lever again: the ``documents_root`` fixture, which
    sets ``MAST_EXPERIMENT_ROOT``. ``MAST2_PROJECT_ROOT`` does not reach it — the
    experiment folder lives outside the project on purpose (its lifetime is longer
    than the software's), so the day save_draft started writing there this fixture
    was suddenly reading and WRITING the operator's real data."""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MAST_DRAFTS_DIR", str(tmp_path / "drafts"))
    monkeypatch.setenv("MAST_REVIEWS_DIR", str(tmp_path / "reviews"))
    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "experiments" / "db.sqlite"))

    from mast.core import scan_registry
    from mast.knowledge import libraries

    monkeypatch.setattr(libraries, "_default_registry",
                        libraries.LibraryRegistry(tmp_path / "literature_libs"),
                        raising=False)
    scan_registry.clear()

    assert list_existing() == [], "a file store leaked in — redirect it here"
    leaked = [s.artifact.id for s in class_status() if s.produced]
    assert leaked == [], f"a non-file store leaked in: {leaked}"
    return tmp_path


def _status_by_id(*args):
    return {s.artifact.id: s for s in class_status(*args)}


# ════════════════════════════════════════════════════════════════════════
# #16 — a class must be able to STOP saying 待产出
# ════════════════════════════════════════════════════════════════════════

class TestAClassCanReportItselfProduced:
    def test_every_class_is_reported_not_just_the_file_backed_ones(self, isolated):
        """Every registered class, always. The UI cannot show a state for a class
        the backend never mentions — and four of them were never mentioned.

        Counted against ``ARTIFACTS`` rather than a literal (it was ``== 9`` until
        literature_report was added on 2026-07-29): the point of the assertion is
        "no class is dropped", and a hard-coded count only restates the registry's
        length while making every future addition look like a regression."""
        got = _status_by_id()
        assert set(got) == {a.id for a in ARTIFACTS}
        assert len(got) == len(ARTIFACTS)

    def test_a_fresh_workspace_says_nothing_is_produced(self, isolated):
        """The honest baseline. 待产出 is correct HERE — the bug was that it was
        also reported on a machine with 77 experiment records."""
        for s in class_status():
            assert s.produced is False, f"{s.artifact.id} claims产出 on empty stores"
            assert s.known is True, f"{s.artifact.id} should be readable-and-empty"

    def test_saving_a_draft_flips_the_draft_class_to_produced(self, isolated):
        """THE regression. Before: this class stayed 待产出 no matter what, because
        the check compared 'draft:T_v001' against 'draft'."""
        assert _status_by_id()["draft"].produced is False

        from mast.agents.paper_writing.tools import save_draft

        save_draft.invoke(tool_call(save_draft, {"title": "T", "markdown_text": "# T\nbody"}))

        after = _status_by_id()["draft"]
        assert after.produced is True
        assert after.count == 1
        assert after.known is True

    def test_an_experiment_record_flips_a_NON_FILE_class(self, isolated):
        """实验记录 lives in a SQLite table, not a directory. Nothing enumerated
        it, so it was one of the four classes that could never leave 待产出 even
        after the id-space bug was fixed."""
        from mast.agents._shared.data_paths import experiment_db_path
        from mast.logging.storage import ExperimentStorage

        assert _status_by_id()["experiment_records"].produced is False

        db = experiment_db_path()
        db.parent.mkdir(parents=True, exist_ok=True)
        ExperimentStorage(db).create_experiment("Au(111) 测试", "goal")

        after = _status_by_id()["experiment_records"]
        assert after.produced is True, "a real experiment row still reads 待产出"
        assert after.count == 1
        assert "1" in after.detail

    def test_a_saved_plan_is_enumerated_as_an_openable_file(self, isolated):
        """实验方案 renders plan_<id>.md beside the experiment DB. It had no
        enumerator, so 7 plans on disk still read 尚未产出."""
        from mast.agents._shared.artifacts import plans_dir

        assert _status_by_id()["experiment_plan"].produced is False
        plans_dir().mkdir(parents=True, exist_ok=True)
        (plans_dir() / "plan_abc123.md").write_text("# plan", encoding="utf-8")

        st = _status_by_id()["experiment_plan"]
        assert st.produced is True and st.count == 1

        row = next(r for r in list_existing() if r["artifact_id"] == "experiment_plan")
        assert Path(row["path"]).is_file(), "an enumerated artifact must be openable"
        assert row["editable"] is False, "a plan is rendered by PlanStore, not typed"

    def test_the_count_is_the_number_of_things_not_of_stores(self, isolated):
        """文献库 counts MEMBERSHIP, not libraries. LibraryRegistry always
        materialises a GLOBAL library, so counting libraries would report
        '1 · 已产出' on a virgin install before anything had been done."""
        st = _status_by_id()["literature_library"]
        assert st.count == 0 and st.produced is False, (
            f"virgin registry reports produced: {st.detail}")


# ════════════════════════════════════════════════════════════════════════
# "I cannot tell" must never be rendered as "there is nothing"
# ════════════════════════════════════════════════════════════════════════

class TestUnreadableIsNotZero:
    def test_an_unreadable_store_is_known_false_not_count_zero(self, isolated, monkeypatch):
        """A locked/corrupt DB must degrade to 无法读取. Reporting it as 0 would
        put the panel right back to claiming 待产出 about a full store — the
        original complaint, with a different cause."""
        import mast.agents._shared.artifacts as A

        def _boom():
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setitem(A._STORE_PROBES, "experiment_records", _boom)
        st = _status_by_id()["experiment_records"]
        assert st.known is False
        assert st.produced is False   # unknown is not a claim of production
        assert "无法读取" in st.detail

    def test_probing_a_missing_db_does_not_create_it(self, isolated):
        """The probe opens read-only on purpose: sqlite3.connect happily creates
        the file, which would turn 'no database yet' into 'a database with 0
        rows' AND leave a real empty file behind to make it permanent."""
        from mast.agents._shared.artifacts import _sqlite_count
        from mast.agents._shared.data_paths import experiment_db_path

        db = experiment_db_path()
        assert not db.exists()
        assert _sqlite_count(db, "SELECT COUNT(*) FROM experiments") == 0
        assert not db.exists(), "a status probe created the store it asked about"

    def test_a_db_without_the_table_is_a_real_zero(self, isolated):
        """An existing DB whose writer has never run genuinely holds nothing —
        that IS zero, not unknown."""
        from mast.agents._shared.artifacts import _sqlite_count
        from mast.agents._shared.data_paths import experiment_db_path

        db = experiment_db_path()
        db.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(db).close()
        assert _sqlite_count(db, "SELECT COUNT(*) FROM experiments") == 0


# ════════════════════════════════════════════════════════════════════════
# The id-space confusion that caused #16 — pinned so it cannot come back
# ════════════════════════════════════════════════════════════════════════

class TestIdSpaces:
    def test_a_class_id_is_never_a_file_id(self, isolated):
        """The join the UI used (produced FILE ids ∩ CLASS ids) is empty BY
        CONSTRUCTION. Any code matching those two together is broken."""
        from mast.agents.paper_writing.tools import save_draft

        save_draft.invoke(tool_call(save_draft, {"title": "T", "markdown_text": "# T\nbody"}))
        file_ids = {r["doc_id"] for r in list_existing()}
        class_ids = {a.id for a in ARTIFACTS}
        assert file_ids, "need at least one produced artifact for this to mean anything"
        assert file_ids & class_ids == set()

    def test_joining_on_kind_is_what_works(self, isolated):
        """…and this is the join that does: a row's artifact_id IS its class."""
        from mast.agents.paper_writing.tools import save_draft

        save_draft.invoke(tool_call(save_draft, {"title": "T", "markdown_text": "# T\nbody"}))
        rows = list_existing()
        assert {r["artifact_id"] for r in rows} <= {a.id for a in ARTIFACTS}
        assert any(r["artifact_id"] == "draft" for r in rows)


# ════════════════════════════════════════════════════════════════════════
# #29 — grouping needs the class to be a real, complete partition
# ════════════════════════════════════════════════════════════════════════

class TestGroupingIsWellFormed:
    def test_scan_files_is_flagged_high_volume(self):
        """The pile that made the list unreadable. The UI starts it collapsed;
        WHICH class floods is a fact about the artifact, so it lives here."""
        hv = {a.id for a in ARTIFACTS if a.high_volume}
        assert hv == {"scan_files"}

    def test_every_produced_artifact_lands_in_exactly_one_group(self, isolated):
        """Grouping must not lose rows. Every enumerated artifact's class has to
        be one of the nine, or it would vanish from a grouped list."""
        from mast.agents.paper_writing.tools import save_draft
        from mast.core import scan_registry

        save_draft.invoke(tool_call(save_draft, {"title": "T", "markdown_text": "# T\nbody"}))
        sxm = isolated / "synthetic_sample-001_0005.sxm"
        sxm.write_bytes(b"sxm")
        scan_registry.record_scan_path(str(sxm))

        rows = list_existing()
        classes = {a.id for a in ARTIFACTS}
        for r in rows:
            assert r["artifact_id"] in classes, f"{r['doc_id']} has no group to sit in"
        counted = sum(s.count for s in class_status(rows)
                      if s.artifact.id in {"draft", "scan_files"})
        assert counted == len(rows) == 2

    def test_sxm_rows_are_not_editable(self, isolated):
        """#29 showed 「打开编辑器」 on every .sxm row. The backend refuses those,
        so the button could only ever open a degraded editor."""
        from mast.core import scan_registry

        sxm = isolated / "synthetic_sample-001_0005.sxm"
        sxm.write_bytes(b"sxm")
        scan_registry.record_scan_path(str(sxm))
        row = next(r for r in list_existing() if r["artifact_id"] == "scan_files")
        assert row["editable"] is False


# ════════════════════════════════════════════════════════════════════════
# #16 — the 编辑/填充 button's target never worked
# ════════════════════════════════════════════════════════════════════════

def test_the_old_seed_button_target_is_still_refused(tmp_path, monkeypatch):
    """「编辑/填充」 posted the bare CLASS id. Every class id is refused: the
    editable ones for lacking a filename, the rest for not being text at all.
    So the button could not produce anything — exactly the same dead end
    as before, in one line of code."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mast.api.context import AppContext
    from mast.api.routes.artifacts_edit import router

    monkeypatch.setenv("MAST_DRAFTS_DIR", str(tmp_path / "drafts"))
    monkeypatch.setenv("MAST_REVIEWS_DIR", str(tmp_path / "reviews"))
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    client = TestClient(app)

    for art in ARTIFACTS:
        r = client.post(f"/api/artifacts/{art.id}/edit", json={"body": "seed text"})
        assert r.status_code == 200                      # degrade, never a 500
        body = r.json()
        assert body["degraded"] is True and body["ok"] is False, (
            f"the class id {art.id!r} was accepted as an edit target — "
            "that write goes nowhere")
        assert body["detail"], "a refusal must say why"


# ════════════════════════════════════════════════════════════════════════
# The dead keys found on the way: memory edges named tools that don't exist
# ════════════════════════════════════════════════════════════════════════

class TestMemoryEdgesNameRealTools:
    def test_the_ghost_names_are_gone(self):
        """``remember_insight`` / ``recall_insights`` appeared ONLY in
        artifacts.py — as a TOOL_ACCESS key and, fatally, in the hard-coded set
        agent_tool_names() injected into every agent. The guard test compared
        the map against that set, i.e. against itself, so it passed."""
        held = agent_tool_names()
        everywhere = set().union(*held.values()) if held else set()
        assert "remember_insight" not in everywhere
        assert "recall_insights" not in everywhere
        assert "remember_insight" not in TOOL_ACCESS
        assert "recall_insights" not in TOOL_ACCESS

    def test_memory_edges_use_the_names_the_factory_really_registers(self):
        """Derived from the factory, so renaming a memory tool breaks this test
        instead of silently un-drawing the edge."""
        from mast.agents._shared.memory_tools import make_memory_tools

        real = {t.name for t in make_memory_tools(lambda: {})}
        declared = {k for k, (art, _) in TOOL_ACCESS.items() if art == "memory"}
        assert declared <= real, f"TOOL_ACCESS invents memory tools: {declared - real}"
        assert "memory_write" in declared, "the memory WRITE edge went missing"

    def test_agent_tool_names_does_not_hard_code_shared_tool_names(self):
        """The shared factories are CALLED, not transcribed. A transcription is a
        second table, and this one had already drifted into fiction.

        Split into two halves on 2026-08-21 because the two grants are not the
        same grant: MEMORY really does go to every agent (the orchestrator's
        ``_shared()`` hands it out unconditionally), while the BUFFER trio only
        goes to agents that take a ``buf``. Asserting one rule for both was true
        by coincidence — every agent happened to take a buffer — right up until
        one didn't, and at that point the honest fix is to state the two rules,
        not to hand an agent tools it has no use for so a test stays green.
        """
        from mast.agents._shared.artifacts import NO_BUFFER_AGENTS
        from mast.agents._shared.buffer_tools import make_buffer_tools
        from mast.agents._shared.memory_tools import make_memory_tools

        memory = {t.name for t in make_memory_tools(lambda: {})}
        buffer = {t.name for t in make_buffer_tools(None)}
        held = agent_tool_names()
        for agent, names in held.items():
            assert memory <= names, f"{agent} lost the memory tools: {memory - names}"
            if agent in NO_BUFFER_AGENTS:
                continue
            assert buffer <= names, f"{agent} lost the buffer tools: {buffer - names}"

    def test_the_no_buffer_exception_is_true_of_the_real_agent(self):
        """An exception that only agrees with itself is worthless.

        ``NO_BUFFER_AGENTS`` suppresses an edge; this checks the suppression
        matches reality by calling the agent's OWN assembly. If someone later
        gives that agent buffer tools, the name has to come off the list — and
        until it does, the artifact graph would be under-reporting a real edge.
        """
        import importlib

        from mast.agents._shared.artifacts import NO_BUFFER_AGENTS
        from mast.agents._shared.buffer_tools import make_buffer_tools

        buffer = {t.name for t in make_buffer_tools(None)}
        assert buffer, "buffer factory produced nothing — this gate is idling"
        for agent in NO_BUFFER_AGENTS:
            mod = importlib.import_module(f"mast.agents.{agent}.tools")
            own = {getattr(t, "name", "") for t in mod.build_tools(None)}
            assert not (own & buffer), (
                f"{agent} is listed as holding no buffer tools but its build_tools "
                f"returns {sorted(own & buffer)}")


# ════════════════════════════════════════════════════════════════════════
# The endpoint the panel actually reads
# ════════════════════════════════════════════════════════════════════════

def test_api_artifacts_returns_a_group_for_every_class(isolated):
    """The panel can only render a class the payload mentions. It used to get
    only produced FILES and had to invent the class states itself — which is
    where it went wrong."""
    from fastapi.testclient import TestClient

    from mast.api.app import create_app

    from mast.agents.paper_writing.tools import save_draft

    save_draft.invoke(tool_call(save_draft, {"title": "T", "markdown_text": "# T\nbody"}))

    client = TestClient(create_app())
    body = client.get("/api/artifacts").json()
    assert body["degraded"] is False

    groups = {g["artifact_id"]: g for g in body["groups"]}
    assert set(groups) == {a.id for a in ARTIFACTS}
    assert groups["draft"]["produced"] is True and groups["draft"]["count"] == 1
    assert groups["review"]["produced"] is False
    assert groups["scan_files"]["high_volume"] is True
    # every group carries the honest sentence the UI shows on its header
    for g in groups.values():
        assert g["detail"], f"{g['artifact_id']} has no explanation of its state"
