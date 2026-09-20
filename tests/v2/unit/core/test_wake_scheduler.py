"""Waking parked agents — W4, and the circuit breaker the design did not have.

``docs/v2/design/wakeup_scheduling.md`` §5.4.

The single most important thing in this file is ``TestTheLoopBreaker``. Product-driven
waking creates a cycle none of the existing guards can see:

    paper_writing wakes → writes a draft → wakes paper_review → writes a review →
    wakes paper_writing to revise → …

That is the A→B→A ping-pong agent-to-agent handoff was removed for on 2026-05-30, with
the entry point moved — and worse, because **every wake is a NEW run**, so
``recursion_limit``, ``visit_count`` and the per-run USD budget all reset to zero, and
every step SUCCEEDS, so ``StallGuard`` (which keys off repeated FAILURE signatures) is
structurally blind. The measured $30.66 runaway had exactly this shape: four identical
successful cycles. Every other bound in the system is per-run, so the only place a
bound can live is here.

The rest of the file holds the honesty properties:

* an unreadable ledger does not silently disable the ceiling, and does not stop work
  either — the count-based quota is the bound that survives a billing outage;
* deadlines are swept even when the foreground is busy (a timeout the operator was
  promised must not be postponed by unrelated activity);
* the scheduler never decides while the mainline is deciding;
* nothing is woken WITHOUT asking, and nothing is left silent when its input arrives
  but the machinery to ask is missing — that escalates instead.
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

import time  # noqa: E402

import pytest  # noqa: E402

from mast.core import park_board as pb  # noqa: E402
from mast.core import wake_scheduler as ws  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_PARK_BOARD_DIR", str(tmp_path))
    pb.set_board_for_test(None)
    yield
    pb.set_board_for_test(None)


@pytest.fixture()
def board(tmp_path):
    b = pb.ParkBoard(tmp_path / "pb.json")
    pb.set_board_for_test(b)
    return b


@pytest.fixture()
def versions(monkeypatch):
    """Pin the disk-derived version map so 'something arrived' is deterministic."""
    state = {"draft": (1, 111.0), "analysis": (0, 0.0), "last_scan": (0, 0.0),
             "literature_report": (0, 0.0), "experiment_plan": (0, 0.0),
             "review": (0, 0.0)}
    monkeypatch.setattr(ws, "_artifact_versions", lambda: dict(state))
    return state


class _Spy:
    """Records what the scheduler asked, spawned and escalated."""

    def __init__(self, action="start"):
        self.action = action
        self.asked: list = []
        self.spawned: list = []
        self.escalated: list = []
        self.run_no = 0

    def ask(self, park, arrived, versions):
        self.asked.append((park["park_id"], list(arrived)))
        return {"action": self.action, "waiting_for": [], "reason": "test"}

    def spawn(self, park):
        self.run_no += 1
        self.spawned.append(park["park_id"])
        return f"run-{self.run_no}"

    def escalate(self, kind, park, detail):
        self.escalated.append((kind, (park or {}).get("park_id"), detail))


def _mk(board, agent="paper_review", waiting=("draft",), eid="exp-A"):
    return board.park(agent, waiting_for=list(waiting), experiment_id=eid,
                      instruction="做完它")


def _sched(spy, **kw):
    kw.setdefault("should_wake", lambda: True)
    kw.setdefault("daily_budget_usd", 0.0)     # off unless a test wants it
    kw.setdefault("ask_cooldown_s", 0.0)
    return ws.WakeScheduler(ask=spy.ask, spawn=spy.spawn, escalate=spy.escalate, **kw)


# ════════════════════════════════════════════════════════════════════
# THE LOOP BREAKER — the thing the original design was missing
# ════════════════════════════════════════════════════════════════════

class TestTheLoopBreaker:
    def test_wakes_stop_at_the_daily_per_experiment_quota(self, board, versions):
        """Without this, PW⇄PR wake each other forever. Every wake is a fresh run, so
        recursion_limit / visit_count / the per-run budget all reset, and every step
        succeeds — StallGuard cannot see it. This is the only bound there is."""
        spy = _Spy()
        s = _sched(spy, max_wakes_per_day=2)
        for i in range(6):
            _mk(board, agent=f"agent{i}")
            # each park waits on something that just "arrived"
            versions["draft"] = (1, 100.0 + i)
            s.tick()
        assert len(spy.spawned) == 2, \
            f"the wake quota did not bound the loop: {len(spy.spawned)} wakes"

    def test_hitting_the_quota_escalates_to_the_operator(self, board, versions):
        spy = _Spy()
        s = _sched(spy, max_wakes_per_day=1)
        _mk(board, agent="a1")
        s.tick()
        _mk(board, agent="a2")
        versions["draft"] = (2, 222.0)
        s.tick()
        kinds = [k for k, _p, _d in spy.escalated]
        assert "breaker" in kinds, \
            "the loop was stopped silently — an operator would see work simply cease"

    def test_the_breaker_reports_once_not_every_tick(self, board, versions):
        """An alarm that repeats every minute gets muted, and a muted alarm is the
        same as no alarm."""
        spy = _Spy()
        s = _sched(spy, max_wakes_per_day=0)   # everything blocked
        _mk(board)
        for _ in range(5):
            s.tick()
        breakers = [e for e in spy.escalated if e[0] == "breaker"]
        assert len(breakers) == 1

    def test_zero_means_zero_wakes_not_unlimited(self, board, versions):
        """A safety counter set to 0 by an operator means "do not do this". Reading it
        as "no limit" — the convention the USD ceiling uses, where 0 is what an
        unconfigured setting looks like — would hand them the exact opposite of what
        they asked for."""
        spy = _Spy()
        s = _sched(spy, max_wakes_per_day=0)
        _mk(board)
        s.tick()
        assert spy.spawned == []

    def test_a_negative_quota_disables_the_check(self, board, versions):
        spy = _Spy()
        s = _sched(spy, max_wakes_per_day=-1)
        _mk(board)
        s.tick()
        assert len(spy.spawned) == 1

    def test_the_quota_is_per_experiment(self, board, versions):
        """One experiment exhausting its budget must not stop another's work."""
        spy = _Spy()
        s = _sched(spy, max_wakes_per_day=1)
        _mk(board, agent="a1", eid="exp-A")
        s.tick()
        _mk(board, agent="a2", eid="exp-B")
        versions["draft"] = (3, 333.0)
        s.tick()
        assert len(spy.spawned) == 2

    def test_the_quota_resets_the_next_day(self, board, versions, monkeypatch):
        spy = _Spy()
        s = _sched(spy, max_wakes_per_day=1)
        _mk(board, agent="a1")
        s.tick()
        assert len(spy.spawned) == 1
        # roll the day over
        monkeypatch.setattr("mast.billing.run_meter._local_midnight_ts",
                            lambda *a, **k: time.time() + 86400)
        _mk(board, agent="a2")
        versions["draft"] = (9, 999.0)
        s.tick()
        assert len(spy.spawned) == 2


class TestTheDailySpendCeiling:
    def test_waking_stops_when_the_day_is_spent(self, board, versions, monkeypatch):
        """A per-run ceiling bounds ONE run and cannot see run COUNT — and waking
        multiplies exactly that. Ten woken runs at $8 are $80, each individually
        legal."""
        monkeypatch.setattr("mast.billing.run_meter.daily_spend_usd", lambda: 31.0)
        spy = _Spy()
        s = _sched(spy, daily_budget_usd=30.0)
        _mk(board)
        s.tick()
        assert spy.spawned == []
        assert any(k == "breaker" for k, _p, _d in spy.escalated)

    def test_spending_under_the_ceiling_does_not_block(self, board, versions,
                                                      monkeypatch):
        monkeypatch.setattr("mast.billing.run_meter.daily_spend_usd", lambda: 3.0)
        spy = _Spy()
        s = _sched(spy, daily_budget_usd=30.0)
        _mk(board)
        s.tick()
        assert len(spy.spawned) == 1

    def test_an_unreadable_ledger_does_not_block_work(self, board, versions,
                                                     monkeypatch):
        """None means "unknown". Treating it as over-budget would let a billing
        hiccup stop the system; the COUNT quota is the bound that does not depend on
        billing being readable."""
        monkeypatch.setattr("mast.billing.run_meter.daily_spend_usd", lambda: None)
        spy = _Spy()
        s = _sched(spy, daily_budget_usd=30.0)
        _mk(board)
        s.tick()
        assert len(spy.spawned) == 1

    def test_a_zero_budget_means_off_not_zero_dollars(self, board, versions):
        spy = _Spy()
        s = _sched(spy, daily_budget_usd=0.0)
        _mk(board)
        s.tick()
        assert len(spy.spawned) == 1


# ════════════════════════════════════════════════════════════════════
# Deciding at the right moment
# ════════════════════════════════════════════════════════════════════

class TestWhenItDecides:
    def test_it_does_nothing_while_the_foreground_is_busy(self, board, versions):
        """Two decision streams deciding the same thing is how they contradict each
        other. The mainline supervisor owns the decision whenever it is awake."""
        spy = _Spy()
        s = _sched(spy, should_wake=lambda: False)
        _mk(board)
        out = s.tick()
        assert spy.asked == [] and spy.spawned == []
        assert out["skipped"] == "foreground busy"

    def test_deadlines_are_swept_even_when_the_foreground_is_busy(self, board,
                                                                 versions):
        """A timeout the operator was promised must not be postponed by unrelated
        activity happening to be underway."""
        rec = _mk(board)
        board._items[rec["park_id"]]["deadline_at"] = time.time() - 1
        spy = _Spy()
        s = _sched(spy, should_wake=lambda: False)
        out = s.tick()
        assert out["expired"] == 1
        assert any(k == "expired" for k, _p, _d in spy.escalated)

    def test_an_expired_park_escalates_rather_than_waking(self, board, versions):
        rec = _mk(board)
        board._items[rec["park_id"]]["deadline_at"] = time.time() - 1
        spy = _Spy()
        s = _sched(spy)
        s.tick()
        assert spy.spawned == [], \
            "an expired wait was resumed silently instead of surfacing to a human"

    def test_a_broken_idle_predicate_stops_rather_than_guesses(self, board, versions):
        def _boom():
            raise RuntimeError("no runtime")

        spy = _Spy()
        s = _sched(spy, should_wake=_boom)
        _mk(board)
        out = s.tick()
        assert spy.spawned == [] and "idle predicate failed" in out["skipped"]


# ════════════════════════════════════════════════════════════════════
# What counts as "it arrived"
# ════════════════════════════════════════════════════════════════════

class TestArrivalDetection:
    def test_nothing_present_means_nothing_arrived(self, board, monkeypatch):
        monkeypatch.setattr(ws, "_artifact_versions",
                            lambda: {"draft": (0, 0.0)})
        spy = _Spy()
        s = _sched(spy)
        _mk(board)
        s.tick()
        assert spy.asked == []

    def test_the_same_version_is_not_re_asked(self, board, versions):
        """Presence alone would re-report the same artifact forever, and the agent
        would be asked about the same fact every tick."""
        spy = _Spy(action="wait")   # declining is what keeps the park open
        s = _sched(spy, max_wakes_per_day=99)
        rec = _mk(board)
        board.note_decision(rec["park_id"], woke=False, reason="x",
                            versions={"draft": (1, 111.0)})
        s.tick()
        assert spy.asked == []

    def test_a_new_version_is_asked_about(self, board, versions):
        spy = _Spy()
        s = _sched(spy)
        rec = _mk(board)
        board.note_decision(rec["park_id"], woke=False, reason="x",
                            versions={"draft": (1, 111.0)})
        versions["draft"] = (2, 222.0)     # a new version landed
        s.tick()
        assert [p for p, _a in spy.asked] == [rec["park_id"]]

    def test_unreadable_versions_are_not_treated_as_an_arrival(self, board,
                                                              monkeypatch):
        monkeypatch.setattr(ws, "_artifact_versions", lambda: {})
        spy = _Spy()
        s = _sched(spy)
        _mk(board)
        out = s.tick()
        assert spy.asked == []
        assert "unreadable" in out["skipped"]


class TestAskCooldown:
    def test_a_high_frequency_product_does_not_cause_endless_questions(self, board,
                                                                      monkeypatch):
        """A scan class's "version" is (count, newest mtime), so it changes with every
        saved frame — the version check cannot suppress what genuinely keeps
        changing, and a park waiting on last_scan would be re-asked every tick for a
        whole imaging session."""
        vs = {"last_scan": (1, 1.0)}
        monkeypatch.setattr(ws, "_artifact_versions", lambda: dict(vs))
        # action="wait" is what keeps the park open across ticks — NOT a zero quota,
        # which would block before the question and make this pass vacuously.
        spy = _Spy(action="wait")
        s = _sched(spy, ask_cooldown_s=3600, max_wakes_per_day=99)
        _mk(board, agent="data_processing", waiting=("last_scan",))
        for i in range(5):
            vs["last_scan"] = (i + 2, float(i + 2))   # a new frame every tick
            s.tick()
        assert len(spy.asked) == 1, \
            f"asked {len(spy.asked)} times during one scan session"


# ════════════════════════════════════════════════════════════════════
# Deciding, and recording the decision
# ════════════════════════════════════════════════════════════════════

class TestDecisions:
    def test_waking_spawns_and_marks_the_park(self, board, versions):
        spy = _Spy(action="start")
        s = _sched(spy)
        rec = _mk(board)
        out = s.tick()
        assert out["woken"] == 1
        row = board.get(rec["park_id"])
        assert row["status"] == "woken" and row["woken_run_id"] == "run-1"

    def test_woken_is_not_marked_done(self, board, versions):
        """The run has been STARTED, not finished. It is detached on an InMemorySaver
        and can die taking its work with it; only its products coming back close the
        park."""
        spy = _Spy(action="start")
        _sched(spy).tick() if False else None
        s = _sched(spy)
        rec = _mk(board)
        s.tick()
        assert board.get(rec["park_id"])["status"] != "done"

    def test_declining_is_counted_and_the_park_stays_open(self, board, versions):
        spy = _Spy(action="wait")
        s = _sched(spy)
        rec = _mk(board)
        out = s.tick()
        assert out["declined"] == 1
        row = board.get(rec["park_id"])
        assert row["status"] == "waiting" and row["declines"] == 1

    def test_a_failed_spawn_escalates_instead_of_losing_the_park(self, board,
                                                                versions):
        spy = _Spy(action="start")
        spy.spawn = lambda park: ""      # spawn refuses
        s = _sched(spy)
        rec = _mk(board)
        s.tick()
        assert board.get(rec["park_id"])["status"] == "waiting"
        assert any(k == "spawn_failed" for k, _p, _d in spy.escalated)

    def test_a_raising_ask_does_not_kill_the_pass(self, board, versions):
        def _boom(park, arrived, versions):
            raise RuntimeError("provider down")

        spy = _Spy()
        spy.ask = _boom
        s = _sched(spy)
        _mk(board)
        s.tick()   # must not raise
        assert spy.spawned == []

    def test_no_question_engine_escalates_rather_than_waking_unasked(self, board,
                                                                    versions):
        """The agent chose to wait; waking it without asking overrides a decision it
        was invited to make. Staying silent while its input sits there is the other
        wrong answer. So: escalate."""
        spy = _Spy()
        s = ws.WakeScheduler(ask=None, spawn=spy.spawn, escalate=spy.escalate,
                             should_wake=lambda: True, daily_budget_usd=0.0,
                             ask_cooldown_s=0.0)
        _mk(board)
        s.tick()
        assert spy.spawned == []
        assert any(k == "ready" for k, _p, _d in spy.escalated)

    def test_a_declined_park_can_narrow_what_it_waits_for(self, board, versions):
        class _Narrow(_Spy):
            def ask(self, park, arrived, versions):
                return {"action": "wait", "waiting_for": ["analysis"],
                        "reason": "其实我要的是分析"}

        spy = _Narrow()
        s = _sched(spy)
        rec = _mk(board)
        s.tick()
        assert board.get(rec["park_id"])["waiting_for"] == ["analysis"]


# ════════════════════════════════════════════════════════════════════
# Thread + event plumbing
# ════════════════════════════════════════════════════════════════════

class TestThreading:
    def test_start_and_stop_are_clean(self, board):
        s = ws.WakeScheduler(interval_s=5.0, should_wake=lambda: False)
        s.start()
        assert s._thread is not None and s._thread.daemon
        s.stop()
        assert s._thread is None

    def test_start_is_idempotent(self, board):
        s = ws.WakeScheduler(interval_s=5.0, should_wake=lambda: False)
        s.start()
        first = s._thread
        s.start()
        assert s._thread is first
        s.stop()

    def test_nudge_only_sets_a_flag(self, board):
        """An EventBus subscriber runs SYNCHRONOUSLY on the publishing thread — the
        thread that just saved a document. Anything more than setting an Event here
        would block a document write behind a scheduling decision."""
        s = ws.WakeScheduler(should_wake=lambda: True)
        s.nudge()
        assert s._hint.is_set()

    def test_the_event_subscription_does_not_require_a_runtime(self, board):
        s = ws.WakeScheduler()
        s.subscribe_to_events()   # must not raise


class TestBoardFailuresDegrade:
    def test_an_unavailable_board_skips_rather_than_raising(self, monkeypatch):
        class _Boom:
            def sweep_expired(self):
                raise RuntimeError("gone")

            def open_parks(self):
                raise RuntimeError("gone")

        s = ws.WakeScheduler(board=_Boom(), should_wake=lambda: True)
        out = s.tick()
        assert "cannot list parks" in out["skipped"]

    def test_an_empty_board_is_a_cheap_no_op(self, board, monkeypatch):
        """An install that never enables activation gating pays one no-op tick a
        minute — it must not read every artifact on disk to discover that."""
        called = []
        monkeypatch.setattr(ws, "_artifact_versions",
                            lambda: called.append(1) or {})
        s = ws.WakeScheduler(should_wake=lambda: True)
        s.tick()
        assert called == [], "an empty board still walked the artifact tree"


# ════════════════════════════════════════════════════════════════════
# 目标闸门（2026-08-27）—— 挡的不是「醒太多次」，是「根本不该再醒」
# ════════════════════════════════════════════════════════════════════

class _V:
    """GoalVerdict 的鸭子替身（只用到调度器真正读的那两个字段）。"""

    def __init__(self, verdict, reason="因为"):
        self.verdict = verdict
        self.reason = reason


class TestGoalGate:
    """次数配额挡的是「醒太多次」；这道闸挡的是**根本不该再醒**。

    两者互不替代：一份目标已经达成的 park，再醒一次也是合法的一次唤醒，配额
    看不出问题；而它每醒一次都在花钱。
    """

    def test_a_met_goal_closes_the_park_without_waking(self, board, versions):
        spy = _Spy()
        p = _mk(board)
        s = _sched(spy, goal_check=lambda park: _V("done", "判据全满足"))
        out = s.tick()

        assert spy.spawned == [], "目标已达成还是把它叫醒了"
        assert spy.asked == [], "已达成还去问它要不要醒 —— 那次提问也要花钱"
        assert out["closed_by_goal"] == 1
        row = board.list_parks()[0]
        assert row["status"] == "done_by_goal"
        assert "判据全满足" in (row.get("note") or "")
        assert ("done_by_goal", p["park_id"], "判据全满足") in spy.escalated

    def test_closing_by_goal_does_not_spend_a_wake_quota(self, board, versions):
        """关掉一份不必再醒的等待**不是一次唤醒**。

        算进配额的话，一天里几份已完成的 park 就能把真正需要的那次唤醒挤掉。
        """
        spy = _Spy()
        _mk(board)
        s = _sched(spy, goal_check=lambda park: _V("done"))
        s.tick()
        assert s._wakes == {}, f"关 park 竟然计了配额：{s._wakes}"

    def test_unknown_wakes_exactly_as_before(self, tmp_path, versions):
        """判不了 ⇒ **照旧唤醒**，而且与没有这道闸时逐条相同。

        失败代价不对称：判不了却照醒，最坏是多醒一次（有界，而且就是今天的
        行为）；判不了就不醒，则一次 DB 读失败造出一个永远不醒的 park，只能等
        24 h 超时浮出来 —— 无界，而且正是本设计自己点名「最像静默死亡通道」的
        形状。
        """
        # 两块**各自独立**的板：同一块板跑两趟不行 —— 第一趟会写下
        # ``asked_at_versions``，第二趟的 ``_arrived`` 就什么都看不到了，
        # 于是「两边一样」会在一个假的理由上成立（两边都没醒）。
        spy_a, spy_b = _Spy(), _Spy()
        b_a = pb.ParkBoard(tmp_path / "a.json")
        b_b = pb.ParkBoard(tmp_path / "b.json")
        _mk(b_a)
        _mk(b_b)

        base = _sched(spy_a, board=b_a).tick()
        with_gate = _sched(spy_b, board=b_b,
                           goal_check=lambda park: _V("unknown", "库读不到")).tick()

        assert spy_a.spawned and spy_b.spawned, "两边都该唤醒"
        # park_id 是随机的，比的是「问了几次、问的什么」而不是 id。
        assert [w for _, w in spy_a.asked] == [w for _, w in spy_b.asked]
        assert with_gate == base, f"带闸与不带闸不一致：{with_gate} vs {base}"
        assert with_gate["closed_by_goal"] == 0

    def test_unknown_is_visible_on_the_board(self, board, versions):
        """判不了必须**可见** —— 否则它和「一切正常」长得一样。"""
        spy = _Spy()
        p = _mk(board)
        _sched(spy, goal_check=lambda park: _V("unknown", "conducts 库打不开")).tick()
        gc = board.list_parks()[0]["goal_check"]
        assert gc["verdict"] == "unknown" and "conducts" in gc["reason"]

    def test_not_done_wakes_too(self, board, versions):
        spy = _Spy()
        _mk(board)
        s = _sched(spy, goal_check=lambda park: _V("not_done", "还差分析"))
        s.tick()
        assert spy.spawned, "判据说没做完，那正是该醒的时候"

    def test_a_crashing_goal_check_does_not_kill_the_tick(self, board, versions):
        """判据坏了不许杀掉这一趟 —— 那会让整个唤醒机制随之停摆。"""
        spy = _Spy()
        _mk(board)

        def _boom(park):
            raise RuntimeError("库炸了")

        s = _sched(spy, goal_check=_boom)
        out = s.tick()
        assert spy.spawned and out["closed_by_goal"] == 0

    def test_zero_quota_still_lets_the_goal_gate_close_a_park(self, board, versions):
        """两个 0 的语义不受影响。

        次数 0 = 一次都不许醒（安全计数器）；而「关掉一份不必再醒的等待」既不是
        唤醒也不读设置，所以它照做 —— 熔断日不该把已完成的 park 留到明天。
        """
        spy = _Spy()
        _mk(board)
        s = _sched(spy, max_wakes_per_day=0, goal_check=lambda park: _V("done"))
        out = s.tick()
        assert out["closed_by_goal"] == 1 and spy.spawned == []

    def test_a_busy_foreground_defers_the_goal_gate_too(self, board, versions):
        """「主线在决定时调度器不决定」照旧 —— 关 park 也是一个决定。"""
        spy = _Spy()
        _mk(board)
        s = _sched(spy, should_wake=lambda: False,
                   goal_check=lambda park: _V("done"))
        out = s.tick()
        assert out["skipped"] == "foreground busy"
        assert board.list_parks()[0]["status"] == "waiting"

    # ── 变异 ────────────────────────────────────────────────────────
    def test_mutation_without_the_gate_a_met_goal_still_wakes(self, board, versions):
        """不注入 ``goal_check`` = 这道闸不存在 ⇒ 同一份 park 照样被叫醒。

        与第一条是一对：没有它，那条测试可能绿在「这份 park 本来就不会醒」上。
        """
        spy = _Spy()
        _mk(board)
        _sched(spy).tick()                    # goal_check 缺省 = None
        assert spy.spawned, "拆掉闸门之后它也没醒 —— 上面那条测试没测到东西"


class TestGoalGateStaysOutOfCore:
    def test_the_scheduler_imports_neither_goals_nor_conduct(self):
        """``mast.core.wake_scheduler`` 不许 import ``mast.goals`` / ``mast.conduct``。

        那条线程的整个意义是在系统空闲、没有任何 agent 上下文时也能跑；把判据
        栈拉进去与它的目的相反。判据由 runtime **注入**。
        """
        import ast
        from pathlib import Path

        src = Path(ws.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        bad = []
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.startswith("mast.goals") or m.startswith("mast.conduct"):
                    bad.append(m)
        assert bad == [], f"调度器直接 import 了 {bad} —— 判据应当由 runtime 注入"

    def test_the_scanner_can_actually_see_an_import(self):
        """扫描器自检：它必须在这个文件里看得见**别的** mast import。

        没有这一条，「一个都没有」可能只是扫描器坏了。
        """
        import ast
        from pathlib import Path

        tree = ast.parse(Path(ws.__file__).read_text(encoding="utf-8"))
        seen = [n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("mast.")]
        assert seen, "扫描器在 wake_scheduler 里一个 mast import 都没看见 —— 它坏了"
