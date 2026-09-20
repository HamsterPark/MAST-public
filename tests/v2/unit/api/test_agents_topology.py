"""Contract tests for the agents_topology domain (parity rebuild Wave A).

Per the house test rule the router under test is NOT yet included in
``mast.api.app`` (integration wires that); we mount it on a throwaway
FastAPI app with a fresh AppContext. We assert:

  * every endpoint returns 200 + its schema-shaped body (NEVER 500);
  * standalone (no live agents runtime wired) paths degrade — empty, never
    broken — while the static registry-derived fields still resolve;
  * a fake live runtime wired via ad-hoc ``ctx`` hooks
    (``agents_snapshot`` / ``agents_interrupts`` / ``agents_artifacts``) drives
    the snapshot / interrupts / artifacts endpoints through their live paths;
  * the artifact permission matrix is static (does not need a live runtime).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.v2.toolcall import tool_call
from mast.api.context import AppContext
from mast.api.routes.agents_topology import router


# ── throwaway apps ─────────────────────────────────────────────────────


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx or AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _live_snapshot() -> dict:
    """A realistic ``/agents/snapshot`` payload (old MASTApp shape)."""
    return {
        "models": {"instrument_control": "kimi-k2.6"},
        "thinking": {"instrument_control": "high (固定)"},
        "holds": {"data_processing": True},
        "threads_index": {"instrument_control": 5, "literature": 2},
        "handoff_events": [
            {"t": 1.0, "kind": "handoff", "text": "[HANDOFF → DP] routing"},
        ],
        "buffer_events": [{"t": 2.0, "kind": "vision", "text": "tip ok"}],
        "interject_count": 1,
        "pending_interrupts": [{"event_id": "e1", "agent_id": "instrument_control"}],
        "interrupt_gating": True,
        "session_active": True,
        "active_agent_id": "instrument_control",
        "artifacts_count": 2,
        "backend": "orchestrator",
        "capabilities": {"interject": True, "hold": True, "abort": True},
        "active_task": {
            "id": "t1",
            "description": "Au(111) Kondo",
            "active": True,
            "final_text": "",
            "error": None,
        },
        "server_time": "14:32:55",
    }


def _live_client() -> TestClient:
    """A client whose ctx carries fake live read-only agents hooks."""
    ctx = AppContext()
    ctx.agents_snapshot = _live_snapshot  # type: ignore[attr-defined]
    ctx.agents_interrupts = lambda: [  # type: ignore[attr-defined]
        {
            "event_id": "e1",
            "kind": "skill_gate",
            "agent_id": "instrument_control",
            "skill": "TipPulse",
            "params": {"voltage": 5},
            "rationale": "DANGEROUS",
            "allowed_decisions": ["approve", "reject", "edit"],
            "thread_id": "th1",
            "weird_extra": 42,
        },
        {"event_id": "e2", "agent_id": "data_processing", "kind": "workflow_human",
         "routes": ["accept", "redo"]},
    ]
    ctx.agents_artifacts = lambda: {  # type: ignore[attr-defined]
        "session_active": True,
        "produced": {
            "prior_art_summary": "47 Au(111) Kondo hits, top 8 selected.",
            "scan_paths": ["/scan_00184.sxm"],
        },
        "edits": {"draft_sections": {"body": "Results §3.2 draft", "t": 1.0}},
    }
    return _client(ctx)


# ── snapshot: standalone (degraded but registry-derived) ───────────────


def test_snapshot_degrades_unwired() -> None:
    r = _client().get("/api/agents/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["backend"] == "none"
    assert body["session_active"] is False
    # The static registry roster still renders even with no live runtime.
    ids = [a["id"] for a in body["agents"]]
    assert "_supervisor" in ids
    assert "instrument_control" in ids
    # 数目派生，不写死：_supervisor + 整条 pipeline + buffer_summarizer。
    # 写死的那个 8 在加第七个 agent 时红了一次，而它本来要拦的是「静态 roster
    # 在没有活 runtime 时也要渲染得出来」—— 那件事和有几个 agent 无关。
    assert len(ids) == len(body["pipeline"]) + 2
    assert set(body["pipeline"]) <= set(ids)
    assert "research_director" in body["pipeline"], (
        "pipeline 里没有科研策划 —— 同一份响应的产物权限那一半已经在报它了")
    # No live session → empty timelines / zero counts, never broken.
    assert body["handoff_events"] == []
    assert body["pending_interrupt_count"] == 0
    assert body["capabilities"]["interject"] is False


# ── snapshot: live path ────────────────────────────────────────────────


def test_snapshot_live_path() -> None:
    r = _live_client().get("/api/agents/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["backend"] == "orchestrator"
    assert body["session_active"] is True
    assert body["active_agent_id"] == "instrument_control"
    assert body["interject_count"] == 1
    assert body["pending_interrupt_count"] == 1
    assert body["interrupt_gating"] is True
    assert body["artifacts_count"] == 2
    assert body["server_time"] == "14:32:55"
    assert body["capabilities"]["interject"] is True
    # threads / holds / active flag fold into the per-agent nodes.
    ic = next(a for a in body["agents"] if a["id"] == "instrument_control")
    assert ic["active"] is True
    assert ic["thread_count"] == 5
    assert ic["model"] == "kimi-k2.6"
    dp = next(a for a in body["agents"] if a["id"] == "data_processing")
    assert dp["held"] is True
    assert body["handoff_events"][0]["kind"] == "handoff"
    assert body["buffer_events"][0]["text"] == "tip ok"
    assert body["active_task"]["id"] == "t1"


def test_snapshot_never_500_on_bad_hook() -> None:
    ctx = AppContext()
    ctx.agents_snapshot = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[attr-defined]
    r = _client(ctx).get("/api/agents/snapshot")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


# ── interrupts ─────────────────────────────────────────────────────────


def test_interrupts_degrade_unwired() -> None:
    r = _client().get("/api/agents/instrument_control/interrupts")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["agent_id"] == "instrument_control"
    assert body["interrupts"] == []
    assert body["count"] == 0


def test_interrupts_filtered_by_agent() -> None:
    r = _live_client().get("/api/agents/instrument_control/interrupts")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["interrupt_gating"] is True
    assert body["count"] == 1
    intr = body["interrupts"][0]
    assert intr["event_id"] == "e1"
    assert intr["skill"] == "TipPulse"
    assert intr["allowed_decisions"] == ["approve", "reject", "edit"]
    # Unknown keys are preserved in extra, never dropped.
    assert intr["extra"]["weird_extra"] == 42


def test_interrupts_roster_returns_all() -> None:
    r = _live_client().get("/api/agents/__all__/interrupts")
    body = r.json()
    assert body["count"] == 2
    r2 = _live_client().get("/api/agents/_supervisor/interrupts")
    assert r2.json()["count"] == 2


# ── artifacts ──────────────────────────────────────────────────────────


def test_artifacts_empty_is_honest_not_degraded(tmp_path, monkeypatch,
                                                documents_root) -> None:
    """Nothing produced yet is a TRUE statement, not a broken one.

    (The old test asserted degraded=True with no live app. /artifacts no longer
    needs a live app: it enumerates the real stores on disk.)"""
    monkeypatch.setenv("MAST_DRAFTS_DIR", str(tmp_path / "drafts"))
    monkeypatch.setenv("MAST_REVIEWS_DIR", str(tmp_path / "reviews"))
    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
    # plans/ joined the enumerated stores on 2026-07-28 (— 实验方案
    # had no enumerator, so it read 待产出 with seven plan_*.md sitting on disk).
    # It lives beside the experiment DB, so this redirects it.
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "experiments" / "db.sqlite"))
    body = _client().get("/api/artifacts").json()
    assert body["degraded"] is False
    assert body["artifacts"] == []
    assert body["count"] == 0
    # …but the CLASS roster is always complete. An empty produced-list is not
    # the same statement as "we have no idea what this system can produce", and
    # the panel needs the second one to render a class at all.
    # (Only the classes this test redirects are asserted empty — the literature
    # registry resolves through a module-level constant that no env var reaches,
    # so on a machine with a real library it is legitimately non-empty here.)
    # Counted against the registry, not a literal: the roster grew to 10 when
    # literature_report was added (2026-07-29), and the assertion is "every class
    # is present", not "there are exactly N of them".
    from mast.agents._shared.artifacts import ARTIFACTS
    groups = {g["artifact_id"]: g for g in body["groups"]}
    assert set(groups) == {a.id for a in ARTIFACTS}
    assert groups["draft"]["produced"] is False
    assert groups["review"]["produced"] is False


def test_artifacts_lists_real_files(tmp_path, monkeypatch, documents_root) -> None:
    """THE regression this endpoint existed to have.

    It used to read ``task["artifacts"]`` — a dict initialised to {} in exactly
    one place and written by NOTHING anywhere in the tree, so the "produced
    artifacts" list could only ever be empty. The old version of THIS TEST fed it
    fake data through a mock, which is why the suite stayed green while the
    feature was dead. No mock now: produce a real artifact, see a real file.
    """
    monkeypatch.setenv("MAST_DRAFTS_DIR", str(tmp_path / "drafts"))
    monkeypatch.setenv("MAST_REVIEWS_DIR", str(tmp_path / "reviews"))
    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "experiments" / "db.sqlite"))

    from mast.agents.paper_writing.tools import save_draft
    save_draft.invoke(tool_call(save_draft, {"title": "Au111 report", "markdown_text": "# T\nbody"}))
    save_draft.invoke(tool_call(save_draft, {"title": "另一份报告", "markdown_text": "# T\n无关的内容"}))

    body = _client().get("/api/artifacts").json()
    assert body["degraded"] is False
    assert body["count"] == 2
    # …and the draft CLASS now says so. This is the #16 fix at the seam: the UI
    # used to work this out itself by subtracting produced FILE ids from CLASS
    # ids — disjoint id spaces, so the answer was permanently "待产出".
    draft_group = next(g for g in body["groups"] if g["artifact_id"] == "draft")
    assert draft_group["produced"] is True and draft_group["count"] == 2
    a = body["artifacts"][0]
    # `id` addresses ONE DOCUMENT; `kind` is the artifact CLASS. Both used to be
    # the class, so every draft carried id="draft" and the editor could not tell
    # two of them apart — it opened whichever the backend happened to resolve
    # first. The id is now the document's own doc_id, not "<class>:<file stem>":
    # the synthesised string changed whenever the title was reworded, so an id
    # already handed to the editor could stop resolving.
    assert a["kind"] == "draft"
    assert a["id"] and ":" not in a["id"]
    ids = [x["id"] for x in body["artifacts"]]
    assert len(set(ids)) == 2, f"two documents collide on one id: {ids}"
    # The preview is a title + version, not "v001.md" — a filename that is the
    # same for every document tells the operator nothing.
    assert any("Au111 report v1" in x["preview"] for x in body["artifacts"])
    assert a["editable"] is True          # a file the operator can really edit
    assert a["bytes"] > 0
    from pathlib import Path as _P
    assert _P(a["path"]).is_file()        # and it is genuinely there


# ── artifact permission matrix (static) ────────────────────────────────


def _perms() -> dict:
    r = _client().get("/api/artifacts/permissions")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    return {p["artifact_id"]: p for p in body["permissions"]}


def test_artifact_flow_is_directed_and_real() -> None:
    """The flow model must describe the ACTUAL pipeline, and it is
    now DERIVED from the agents' real tool lists rather than hand-written.

    Note what the derivation corrected in my own first hand-written pass: I had
    listed paper_writing as a reader of the scan files. It holds no scan tool, so
    the edge was fiction — exactly the disease. The derivation cannot make that
    mistake, because an edge only exists if a tool performs it.
    """
    by_id = _perms()

    scans = by_id["scan_files"]
    assert scans["writers"] == ["instrument_control"]   # SaveScan
    assert scans["readers"] == ["data_processing"]      # load_scan/fft/… — DP only
    assert "literature" not in scans["readers"]
    assert "paper_writing" not in scans["readers"]      # holds no scan tool
    assert scans["store"] and "MASTState" not in scans["store"]

    draft = by_id["draft"]
    assert draft["writers"] == ["paper_writing"]        # save_draft/draft_section
    assert draft["readers"] == ["paper_review"]         # load_draft/check_*

    review = by_id["review"]
    assert review["writers"] == ["paper_review"]        # save_review
    # PW can now read the review back (load_review, added 2026-07-11): the
    # revision loop used to depend on the issue list surviving in the handoff
    # message text, which conversation compaction can summarise away.
    assert review["readers"] == ["paper_writing"]


def test_multi_writer_artifacts_are_expressible() -> None:
    """The case the single-writer model COULD NOT represent, which is exactly
    what the operator flagged: 'some documents can be written by several
    agents'."""
    by_id = _perms()

    figures = by_id["figures"]
    assert set(figures["writers"]) == {"data_processing", "paper_writing"}
    assert figures["multi_writer"] is True

    # Every agent carries the memory tools — so the count is the ROSTER's size,
    # derived rather than transcribed. A literal 6 here does not test "all of
    # them", it tests "there are six of them", and it went red the day a seventh
    # agent (research_director) was wired without anything being wrong.
    from mast.agents._shared.artifacts import PIPELINE

    memory = by_id["memory"]
    assert set(memory["writers"]) == set(PIPELINE)
    assert memory["multi_writer"] is True

    # IC holds the full meta-tool set; XD holds the lifecycle subset — BOTH write
    # the experiment DB.
    records = by_id["experiment_records"]
    assert set(records["writers"]) == {"instrument_control", "experiment_design"}
    assert records["multi_writer"] is True


def test_store_never_points_at_a_dead_state_field() -> None:
    """``store`` exists so a claim can be CHECKED against the disk — so it must
    never name a MASTState slot that is dead.

    The typed-artifact fields (last_scan / experiment_plan / analysis / draft /
    review / tip_status / pending_scan) have ZERO reads and ZERO writes anywhere
    in the tree; their Pydantic models are never instantiated. A first cut of
    this model cited "MASTState.experiment_plan" and "MASTState.analysis" as
    stores — unverifiable claims, which is precisely the dishonesty the field was
    added to prevent.
    """
    dead = ("MASTState.experiment_plan", "MASTState.analysis", "MASTState.draft",
            "MASTState.review", "MASTState.last_scan", "MASTState.tip_status")
    for p in _perms().values():
        for slot in dead:
            assert slot not in (p["store"] or ""), (
                f"{p['artifact_id']} claims a store that does not exist: {slot}")


def test_plan_writer_is_the_agent_that_can_actually_persist_one() -> None:
    """曾经的错配已经修好了(2026-08-14),所以这条测试跟着改守新事实。

    旧的 docstring 记的是一个诚实的缺陷:「create_plan/approve_plan 只有
    instrument_control 拿得到,设计 agent 存不下自己设计的方案,执行 agent 反而能
    —— 把 XD 画成写者是在画愿望不是画系统」。P3 的断链修复让愿望成了系统:XD 现在
    真的持有 ``create_plan``(单一常量 ``DESIGN_TOOL_NAMES``,runtime 与 artifacts
    两个消费者同源)。

    于是 ``experiment_plan`` 现在是**真的两个写者**:XD 起草落 DRAFT,IC 执行时
    推进。仍然只有一个批准者,而且不是 agent —— ``approve_plan`` 不在任何 agent
    的工具面里(测试见 ``test_xd_design_tool_surface``)。
    """
    plan = _perms()["experiment_plan"]
    assert set(plan["writers"]) == {"experiment_design", "instrument_control"}
    assert "plan_store" in plan["store"] or "plans/" in plan["store"]


def test_single_writer_artifacts_are_not_flagged_multi() -> None:
    by_id = _perms()
    assert by_id["draft"]["multi_writer"] is False
    # ``experiment_plan`` 2026-08-14 起是真的多写者(XD 起草 + IC 推进),
    # 所以它从这条测试里移出去 —— 而且要正向断言那件事,否则「多写者标志坏了」
    # 与「这个产物本来就是单写者」在测试里长得一模一样。
    assert by_id["experiment_plan"]["multi_writer"] is True


def test_vision_buffer_has_no_agent_writer() -> None:
    """'Agents never write the buffer' is a standing invariant — the flow model
    has to be able to say so (writers=[]), which a one-writer-per-artifact table
    literally could not."""
    vb = _perms()["vision_buffer"]
    assert vb["writers"] == []
    assert "instrument_control" in vb["readers"]
    assert vb["multi_writer"] is False


def test_ask_user_question_survives_the_typed_round_trip() -> None:
    """``ask`` is a typed field, not a stray key in ``extra``.

    The polling clients (private-chat + per-agent modals) learn about an
    ask_user question ONLY from this endpoint, so if the structured question
    were dropped here they would render an approval box with no options — the
    operator could see that a question exists but not what it asks."""
    from mast.api.routes.agents_topology import _to_interrupt

    row = {
        "event_id": "ask1",
        "kind": "ask_user",
        "agent_id": "literature",
        "skill": "向用户提问",
        "params": {"question": "先扫哪个区域？"},
        "rationale": "先扫哪个区域？",
        "allowed_decisions": ["answer"],
        "thread_id": "th9",
        "ask": {
            "question": "先扫哪个区域？",
            "header": "区域选择",
            "options": [{"label": "A 区", "description": "缺陷密集"},
                        {"label": "B 区", "description": "平坦台面"}],
            "multi_select": False,
            "allow_custom": True,
            "timeout_action": "continue",
        },
        "lg_id": "lg-7",
    }
    model = _to_interrupt(row)
    assert model.kind == "ask_user"
    assert model.allowed_decisions == ["answer"]
    assert model.ask is not None
    assert [o["label"] for o in model.ask["options"]] == ["A 区", "B 区"]
    assert model.ask["timeout_action"] == "continue"
    # unrelated keys still land in extra rather than being dropped
    assert model.extra.get("lg_id") == "lg-7"
    assert "ask" not in model.extra


def test_non_ask_interrupt_has_no_ask_block() -> None:
    from mast.api.routes.agents_topology import _to_interrupt

    model = _to_interrupt({"event_id": "e1", "kind": "dangerous",
                           "allowed_decisions": ["approve", "reject"]})
    assert model.ask is None


# ── 目标闸门关掉的 park 也要看得见（2026-08-28） ─────────────────────

def test_a_park_closed_by_its_goal_still_shows_on_the_panel(tmp_path, monkeypatch):
    """`done_by_goal` 是终态，但**面板必须留得住它**。

    `list_parks(include_terminal=False)` 只留 waiting/woken/expired ——
    第一版就直接用它，于是被唤醒调度器按目标关掉的 park 从面板上悄悄消失：
    一个在等的 agent 某天不见了，而这块面板存在的理由恰恰是「什么都没发生」
    和「它挂了」不能长得一样。

    `goal_check` 同理：后端每 tick 都在算，判不了时那句「读不到什么」必须能
    到人眼前。
    """
    import mast.core.park_board as pb

    monkeypatch.setenv("MAST_PARK_BOARD_DIR", str(tmp_path))
    pb.set_board_for_test(None)
    try:
        b = pb.board()
        alive = b.park("literature", waiting_for=["experiment_plan"],
                       experiment_id="e1", campaign_id="cmp-1")
        closed = b.park("paper_review", waiting_for=["draft"],
                        experiment_id="e2", campaign_id="cmp-2")
        b.note_goal_check(alive["park_id"], verdict="unknown",
                          reason="conducts 库打不开")
        b.mark_done_by_goal(closed["park_id"], reason="判据全满足",
                            campaign_id="cmp-2")

        body = _client().get("/api/agents/pending-activations").json()
        by_agent = {p["agent"]: p for p in body["parks"]}
        assert "paper_review" in by_agent, (
            "被目标关掉的 park 从面板上消失了 —— 用户看到的是一个在等的 "
            "agent 某天不见了")
        assert by_agent["paper_review"]["status"] == "done_by_goal"
        assert by_agent["paper_review"]["campaign_id"] == "cmp-2"
        # 它不该去抢「需要处理」那一栏。
        assert by_agent["paper_review"]["needs_attention"] is False

        assert by_agent["literature"]["goal_check"]["verdict"] == "unknown"
        assert "conducts" in by_agent["literature"]["goal_check"]["reason"]
    finally:
        pb.set_board_for_test(None)
