"""Readiness (admission) + the environment-survey tool — W1 of wakeup scheduling.

``docs/v2/design/wakeup_scheduling.md`` §2/§3.1. Two things land here:

  * ``artifact_channel.readiness()`` — the deterministic, NO-LLM pre-filter that
    answers "can dispatching this agent right now produce anything but fiction?";
  * ``environment_tools.survey_environment()`` — the agent-facing "what is in this
    experiment right now", which is the reverse of what the artifact channel does
    (it PUSHES each agent its own CONSUMES row at dispatch time and nothing could
    ask the question the other way round).

The invariants worth breaking a build over:

  * ``REQUIRES`` stays TINY. A hard dependency must hold unconditionally; a
    doubtful entry builds a rule that cannot express the rule it stands in for and
    then enforces the wrong one, and the failure mode is a permanently parked
    agent rather than a visible error.
  * ``unknown`` never collapses into ``missing``. "There is no analysis" is a
    reason to wait; "the analysis store could not be read" is a reason to say so.
    This is the same distinction ``ClassStatus.known`` exists for, and it has to
    survive all the way into the text the model reads.
  * the closed ``waiting_for`` set is DERIVED from ``CARRIED_FIELDS``, because a
    fourth hand-maintained copy of those names is one that will drift — and a
    ``waiting_for`` naming a non-channel is a park that can never wake.
  * the survey NEVER pads an empty listing with an example (2026-07-27) and never
    prints a pointer the reader cannot redeem (the ``load_document`` incident).
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import time  # noqa: E402

import pytest  # noqa: E402

from mast.agents._shared import artifact_channel as ac  # noqa: E402
from mast.agents._shared import environment_tools as et  # noqa: E402
from mast.agents._shared.artifacts import ARTIFACT_BY_ID, ClassStatus  # noqa: E402


# ════════════════════════════════════════════════════════════════════
# The tables themselves
# ════════════════════════════════════════════════════════════════════

class TestRequiresStaysMinimal:
    def test_only_the_two_unconditional_hard_dependencies(self):
        """If this fails because an entry was ADDED, the question to answer first is
        "is it true even on the very first hop of a brand-new experiment?" — for
        every candidate so far the answer was no, which is why they belong to the
        model's judgement instead."""
        assert set(ac.REQUIRES) == {"paper_review", "data_processing"}
        assert ac.REQUIRES["paper_review"] == ("draft",)
        assert ac.REQUIRES["data_processing"] == ("last_scan",)

    def test_paper_writing_has_no_hard_dependency(self):
        """The obvious-looking wrong entry: PW revising needs a draft, PW drafting
        does not. Conditional ⇒ not a hard dependency."""
        assert "paper_writing" not in ac.REQUIRES

    def test_experiment_design_has_no_hard_dependency(self):
        """Designing against a survey is better; designing from an operator's
        dictated goal is legitimate and must not be blocked."""
        assert "experiment_design" not in ac.REQUIRES

    def test_every_named_field_is_a_real_channel(self):
        for agent, fields in list(ac.REQUIRES.items()) + list(ac.PREFERS.items()):
            for f in fields:
                assert f in ac.CARRIED_FIELDS, \
                    f"{agent} depends on {f!r}, which is not a channel field"

    def test_every_named_agent_is_real(self):
        for agent in list(ac.REQUIRES) + list(ac.PREFERS):
            assert agent in ac.CONSUMES, f"{agent!r} is not an agent"

    def test_hard_and_soft_never_name_the_same_field(self):
        for agent in set(ac.REQUIRES) & set(ac.PREFERS):
            assert not (set(ac.REQUIRES[agent]) & set(ac.PREFERS[agent])), \
                f"{agent}: a field cannot be both required and merely preferred"


class TestWaitableFieldsIsDerived:
    def test_derived_from_carried_fields_not_hand_written(self):
        """A hand-maintained fourth copy of these names WILL drift, and a
        ``waiting_for`` value that names no real channel can never be matched
        against a future arrival — a park that never wakes."""
        assert set(ac.WAITABLE_FIELDS) <= set(ac.CARRIED_FIELDS)
        # everything except the one deliberate exclusion
        assert set(ac.WAITABLE_FIELDS) == set(ac.CARRIED_FIELDS) - {"scan_id"}

    def test_scan_id_is_excluded_because_it_never_arrives_on_its_own(self):
        assert "scan_id" not in ac.WAITABLE_FIELDS

    def test_every_requires_and_prefers_field_is_waitable(self):
        """Otherwise an agent could be parked for something it can never be told
        has arrived."""
        for fields in list(ac.REQUIRES.values()) + list(ac.PREFERS.values()):
            for f in fields:
                assert f in ac.WAITABLE_FIELDS, f"{f} is depended on but not waitable"


# ════════════════════════════════════════════════════════════════════
# readiness() over an in-run state
# ════════════════════════════════════════════════════════════════════

def _doc(doc_id="rpt__ab12"):
    return ac.doc_ref(doc_id=doc_id, version=1, kind="literature_report",
                      title="t", summary="s")


@pytest.fixture()
def empty_disk(monkeypatch):
    """Every artifact class readable and EMPTY.

    Required for any test about the state path, because a state miss deliberately
    falls through to disk (see ``TestStateAbsenceIsNotProofOfAbsence``). Without
    pinning disk these tests would read the developer's real artifacts directory and
    pass or fail depending on what happens to be in it.
    """
    rows = [ClassStatus(artifact=a, count=0, known=True, detail="empty")
            for a in ARTIFACT_BY_ID.values()]
    monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                        lambda *a, **k: rows)
    return rows


class TestReadinessFromState:
    def test_paper_review_without_a_draft_has_a_hard_miss(self, empty_disk):
        r = ac.readiness("paper_review", {})
        assert r["missing_hard"] == ["draft"]

    def test_paper_review_with_a_draft_has_no_hard_miss(self, empty_disk):
        r = ac.readiness("paper_review", {"draft": _doc("draft__x")})
        assert r["missing_hard"] == []
        assert "draft" in r["present"]

    def test_data_processing_without_a_scan_has_a_hard_miss(self, empty_disk):
        assert ac.readiness("data_processing", {})["missing_hard"] == ["last_scan"]

    def test_soft_misses_are_reported_separately_from_hard(self, empty_disk):
        r = ac.readiness("paper_review", {"draft": _doc("draft__x")})
        assert r["missing_hard"] == []
        assert set(r["missing_soft"]) == {"analysis", "literature_report"}

    def test_an_agent_with_no_declared_needs_is_always_ready(self, empty_disk):
        r = ac.readiness("literature", {})
        assert r["missing_hard"] == [] and r["missing_soft"] == []

    def test_unknown_agent_is_not_an_error(self, empty_disk):
        r = ac.readiness("nope", {})
        assert r["missing_hard"] == [] and r["missing_soft"] == []

    def test_emptiness_agrees_with_carried_from(self, empty_disk):
        """`readiness` and `carried_from` must not disagree about what counts as a
        product, or an agent could be dispatched for a field the handoff will not
        carry (or parked for one it would have)."""
        for empty in ({}, [], "", None):
            state = {"draft": empty}
            assert ac.readiness("paper_review", state)["missing_hard"] == ["draft"]
            assert "draft" not in ac.carried_from(state)


class TestStateAbsenceIsNotProofOfAbsence:
    """The correction that the first integration run forced (2026-07-30).

    A field carried in ``state`` proves the product exists. Its ABSENCE proves
    nothing: the artifact channel only populates state when an agent hands off
    WITHIN the current run, so a fresh run over an experiment folder full of drafts
    starts with an empty state. Treating that as "there is no draft" would park the
    reviewer while its input sat on disk — the whole mechanism turning into a
    deadlock over a fact it never checked.
    """

    def test_a_draft_on_disk_satisfies_an_empty_state(self, monkeypatch):
        monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                            lambda *a, **k: _status(draft=1, figures=0,
                                                    literature_report=0))
        r = ac.readiness("paper_review", {})   # state knows nothing
        assert r["missing_hard"] == [], \
            "an existing on-disk draft was reported as missing because this run's " \
            "state had not been seeded with it"
        assert "draft" in r["present"]

    def test_a_scan_on_disk_satisfies_an_empty_state(self, monkeypatch):
        monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                            lambda *a, **k: _status(scan_files=17))
        assert ac.readiness("data_processing", {})["missing_hard"] == []

    def test_state_still_wins_when_it_HAS_the_field(self, monkeypatch):
        """State is the fresher of the two, so a present field needs no disk read at
        all — and must not be overturned by a disk probe that says empty."""
        monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                            lambda *a, **k: _status(draft=0, figures=0,
                                                    literature_report=0))
        r = ac.readiness("paper_review", {"draft": _doc("draft__x")})
        assert "draft" in r["present"] and r["missing_hard"] == []

    def test_only_disk_can_turn_a_miss_into_a_confirmed_absence(self, monkeypatch):
        """State-absent + disk-unreadable = unknown, NOT missing. Parking on that
        would be parking on an assumption."""
        monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                            lambda *a, **k: _status(draft=None, figures=0,
                                                    literature_report=0))
        r = ac.readiness("paper_review", {})
        assert r["unknown"] == ["draft"] and r["missing_hard"] == []


# ════════════════════════════════════════════════════════════════════
# readiness() with NO state — the idle path
# ════════════════════════════════════════════════════════════════════

def _status(**counts) -> list[ClassStatus]:
    """ClassStatus rows for the real registry. Pass ``id=count`` for a readable
    store, or ``id=None`` for one that could not be read."""
    out = []
    for cid, n in counts.items():
        art = ARTIFACT_BY_ID[cid]
        out.append(ClassStatus(artifact=art, count=(n or 0), known=n is not None,
                               detail="test"))
    return out


class TestReadinessFromDisk:
    def test_disk_answers_when_there_is_no_state(self, monkeypatch):
        """The idle case is the whole reason this path exists: when nothing is awake
        there IS no state to consult, and disk was always the authority anyway
        (state only ever carried pointers into it)."""
        monkeypatch.setattr(ac, "class_status", None, raising=False)
        monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                            lambda *a, **k: _status(draft=2))
        r = ac.readiness("paper_review")
        assert "draft" in r["present"]
        assert r["missing_hard"] == []

    def test_a_confirmed_empty_store_is_a_miss_not_an_unknown(self, monkeypatch):
        # All three classes paper_review consults must be present in the status
        # rows: a class the probe does not report on is genuinely unjudgeable, and
        # this test is about the readable-but-empty case specifically.
        monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                            lambda *a, **k: _status(draft=0, figures=0,
                                                    literature_report=0))
        r = ac.readiness("paper_review")
        assert r["missing_hard"] == ["draft"]
        assert r["unknown"] == []

    def test_an_unreadable_store_is_unknown_not_a_miss(self, monkeypatch):
        """THE distinction. Reporting an outage as "no draft exists" would park the
        reviewer on a false premise, and nothing afterwards would reveal it."""
        monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                            lambda *a, **k: _status(draft=None, figures=0,
                                                    literature_report=0))
        r = ac.readiness("paper_review")
        assert r["unknown"] == ["draft"]
        assert r["missing_hard"] == [], \
            "an unreadable store was reported as a confirmed absence"

    def test_a_class_the_probe_omits_entirely_is_unknown(self, monkeypatch):
        """Not merely tolerated — REQUIRED. If a class is missing from the status
        report we know nothing about it, and guessing "absent" would park an agent
        on an assumption."""
        monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                            lambda *a, **k: _status(literature_report=1))
        r = ac.readiness("paper_review")
        assert "draft" in r["unknown"]
        assert r["missing_hard"] == []

    def test_a_crashing_probe_is_unknown_not_a_miss(self, monkeypatch):
        def _boom(*_a, **_k):
            raise RuntimeError("disk gone")

        monkeypatch.setattr("mast.agents._shared.artifacts.class_status", _boom)
        r = ac.readiness("data_processing")
        assert r["unknown"] == ["last_scan"] and r["missing_hard"] == []

    def test_every_mapped_artifact_class_id_really_exists(self):
        """A mistyped class id would resolve to nothing and silently report every
        field as unknown — indistinguishable from a broken store."""
        for field, ids in ac._FIELD_TO_ARTIFACT_CLASS.items():
            for cid in ids:
                assert cid in ARTIFACT_BY_ID, \
                    f"{field} maps to artifact class {cid!r}, which does not exist"

    def test_every_waitable_field_has_a_disk_answer(self):
        """Otherwise a park on that field is permanently `unknown` on the idle path
        — it could never be told its wait was over."""
        for f in ac.WAITABLE_FIELDS:
            assert f in ac._FIELD_TO_ARTIFACT_CLASS, \
                f"{f} is waitable but has no disk-side check"


class TestProducerNaming:
    def test_every_waitable_field_names_its_producer(self):
        """"You are missing some information" earns an equally vague answer back."""
        for f in ac.WAITABLE_FIELDS:
            assert ac.producer_of(f), f"{f} has no named producer"

    def test_producers_are_real_agents(self):
        for f in ac.WAITABLE_FIELDS:
            assert ac.producer_of(f) in ac.CONSUMES

    def test_every_waitable_field_has_a_human_label(self):
        for f in ac.WAITABLE_FIELDS:
            assert ac.field_label(f) and ac.field_label(f) != f or f == "scan_id"


# ════════════════════════════════════════════════════════════════════
# survey_environment() — invoked for real, per the "注册 ≠ 跑过" rule
# ════════════════════════════════════════════════════════════════════

def _patch_env(monkeypatch, *, existing, status):
    monkeypatch.setattr("mast.agents._shared.artifacts.list_existing",
                        lambda *a, **k: existing)
    monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                        lambda *a, **k: status)


class TestSurveyEnvironmentTool:
    def test_it_is_actually_invokable(self):
        """Registered ≠ runs. This repo shipped a tool list whose module could not
        even be imported (an f-string with a real newline) while the registration
        and endpoint tests were all green."""
        out = et.survey_environment.invoke({})
        assert isinstance(out, str) and out

    def test_empty_environment_says_empty_and_invents_nothing(self, monkeypatch):
        _patch_env(monkeypatch, existing=[],
                   status=_status(draft=0, review=0, literature_report=0))
        out = et.survey_environment.invoke({})
        assert "❌" in out
        assert "尚无" in out
        assert "确实" in out, "an empty environment must say so explicitly"
        # No fabricated pointers of any kind.
        assert "doc_id=" not in out
        for fake in ("rpt__", "draft__", "例如", "示例"):
            assert fake not in out, f"the empty survey contains {fake!r}"

    def test_unreadable_and_empty_use_different_symbols_and_wording(self, monkeypatch):
        """The rule this tool exists to honour. If these two ever render the same,
        a database outage reads as an empty experiment and the model will write a
        confident report about having found nothing."""
        _patch_env(monkeypatch, existing=[],
                   status=_status(draft=0, review=None))
        out = et.survey_environment.invoke({})
        assert "❌" in out and "⚠️" in out
        assert "尚无" in out
        assert "读不到" in out
        assert "不是「没有」" in out or "不等于" in out, \
            "the survey must state that unreadable is not the same as absent"

    def test_produced_classes_list_their_newest_items(self, monkeypatch):
        now = time.time()
        rows = [
            {"artifact_id": "draft", "doc_id": "draft__aa", "name": "稿件 A",
             "path": "/x/a.md", "modified_at": now, "bytes": 10},
            {"artifact_id": "draft", "doc_id": "draft__bb", "name": "稿件 B",
             "path": "/x/b.md", "modified_at": now - 600, "bytes": 10},
        ]
        _patch_env(monkeypatch, existing=rows, status=_status(draft=2))
        out = et.survey_environment.invoke({})
        assert "✅" in out and "2 项" in out
        assert "稿件 A" in out and "draft__aa" in out
        # newest first
        assert out.index("稿件 A") < out.index("稿件 B")

    def test_it_says_how_to_obtain_each_id(self, monkeypatch):
        """A line that says a document exists but not how to open it is a dead end
        — and the channel already paid for advertising an unreachable capability."""
        rows = [{"artifact_id": "literature_report", "doc_id": "rpt__z",
                 "name": "综述", "path": "/x/v1.md", "modified_at": time.time(),
                 "bytes": 1}]
        _patch_env(monkeypatch, existing=rows, status=_status(literature_report=1))
        out = et.survey_environment.invoke({})
        assert "load_document" in out and "list_documents" in out

    def test_scan_files_get_a_path_not_a_synthesised_doc_id(self, monkeypatch):
        """``list_existing`` synthesises ``doc_id`` as "<class>:<stem>" for
        non-document classes. Printing that would hand the model an id
        ``load_document`` cannot resolve — exactly the failure the readback table
        shipped with for a day."""
        rows = [{"artifact_id": "scan_files", "doc_id": "scan_files:img001",
                 "name": "img001.sxm", "path": "/s/img001.sxm",
                 "modified_at": time.time(), "bytes": 1}]
        _patch_env(monkeypatch, existing=rows, status=_status(scan_files=1))
        out = et.survey_environment.invoke({})
        assert "scan_files:img001" not in out, \
            "a synthesised doc_id was advertised as if load_document could open it"
        assert "/s/img001.sxm" in out

    def test_a_broken_probe_never_reads_as_an_empty_environment(self, monkeypatch):
        def _boom(*_a, **_k):
            raise RuntimeError("nope")

        monkeypatch.setattr("mast.agents._shared.artifacts.list_existing", _boom)
        out = et.survey_environment.invoke({})
        assert "不等于" in out, \
            "a failed query must not be presentable as 'the environment is empty'"

    def test_the_tool_has_a_docstring_the_model_can_act_on(self):
        doc = et.survey_environment.description or ""
        assert "读不到" in doc and "尚无" in doc, \
            "the tool must advertise the empty/unreadable distinction to its caller"


class TestWiredToEveryAgent:
    def test_environment_tools_is_a_flat_list_of_real_tools(self):
        assert et.ENVIRONMENT_TOOLS
        for t in et.ENVIRONMENT_TOOLS:
            assert hasattr(t, "invoke") and getattr(t, "name", "")

    def test_runtime_attaches_it_on_both_paths(self):
        """Group path (memory_tools, which every wired agent receives) and private
        chat. Attaching in one place and not the other is how 私聊 and 群聊 drift."""
        import inspect

        from mast.core import runtime as rt
        src = inspect.getsource(rt)
        assert src.count("ENVIRONMENT_TOOLS") >= 2, \
            "survey_environment reaches only one of {group, private chat}"
