"""Durable park-board contracts: waking is not terminal, expiry stays visible until acknowledged, reparking preserves the deadline, and experiment identity is frozen at creation. All tests use isolated temporary stores."""
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

import json  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

from mast.core import park_board as pb  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_board(tmp_path, monkeypatch):
    """Fail-closed isolation: redirect the resolver AND the process singleton.

    Autouse and belt-and-braces on purpose. The four real-data incidents in this
    repo all had the same shape — the test redirected env var A while the store
    resolved path B — so both doors are shut, and the guard below proves it.
    """
    monkeypatch.setenv("MAST_PARK_BOARD_DIR", str(tmp_path))
    pb.set_board_for_test(None)
    yield
    pb.set_board_for_test(None)


@pytest.fixture()
def board(tmp_path):
    return pb.ParkBoard(tmp_path / "park_board.json")


def _park(board, agent="paper_review", **kw):
    kw.setdefault("waiting_for", ["draft"])
    kw.setdefault("experiment_id", "exp-A")
    return board.park(agent, **kw)


# ════════════════════════════════════════════════════════════════════
# Isolation guard — must come first
# ════════════════════════════════════════════════════════════════════

class TestNeverTouchesRealData:
    def test_the_resolver_honours_the_env_override(self, tmp_path):
        p = pb._board_path()
        assert str(tmp_path) in str(p), \
            "the board resolver ignored MAST_PARK_BOARD_DIR — a test run would " \
            "write into the operator's real board"

    def test_the_default_resolver_is_not_frozen_at_import(self, monkeypatch, tmp_path):
        """Resolved per CALL, never captured in a module constant. That pattern
        () means an override set after import is silently ignored, which
        is precisely how a suite starts writing to real paths."""
        monkeypatch.setenv("MAST_PARK_BOARD_DIR", str(tmp_path / "later"))
        assert "later" in str(pb._board_path())

    def test_the_board_lives_in_the_DATA_root_not_the_install_dir(self, monkeypatch,
                                                                 tmp_path):
        """A park has a multi-day lifetime — its whole purpose is to still be there
        when the thing it waits for arrives. The neighbouring boards anchor on
        ``knowledge.paths.base_dir()``, which on a frozen build is the directory
        holding the EXE and whose own docstring calls that tree "assets shipped with
        the exe". An OTA update replaces it. Anchoring a park there would mean an
        update silently deletes every pending activation — the exact silent death
        this subsystem exists to prevent.

        So: same anchor as the billing ledger and the experiment folders.
        """
        monkeypatch.delenv("MAST_PARK_BOARD_DIR", raising=False)
        monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path / "data-root"))
        got = pb._board_path()
        assert "data-root" in str(got), \
            f"the park board resolved to {got} — outside the user data root, so an " \
            "OTA update would delete pending activations"

        from mast._runtime_paths import project_root
        assert Path(project_root()) in got.parents

    def test_the_singleton_is_injectable(self, board):
        pb.set_board_for_test(board)
        assert pb.board() is board


# ════════════════════════════════════════════════════════════════════
# Parking
# ════════════════════════════════════════════════════════════════════

class TestPark:
    def test_a_park_records_everything_the_ui_and_the_next_question_need(self, board):
        rec = _park(board, reason="没有草稿", instruction="评审这篇", hard=True)
        assert rec["agent"] == "paper_review"
        assert rec["waiting_for"] == ["draft"]
        assert rec["status"] == "waiting"
        assert rec["hard"] is True
        assert rec["instruction"] == "评审这篇"
        assert rec["declines"] == 0
        assert rec["deadline_at"] > rec["created_at"]
        assert rec["park_id"].startswith("pk-")

    def test_asked_at_versions_is_per_kind_not_a_scalar(self, board):
        """``waiting_for`` is a LIST, and a document's version is an int while a
        scan's "version" is a count plus an mtime. One number could not say which of
        several waits had moved."""
        rec = _park(board, waiting_for=["draft", "analysis"])
        assert isinstance(rec["asked_at_versions"], dict)

    def test_re_parking_the_same_agent_is_one_wait_not_two(self, board):
        a = _park(board)
        b = _park(board, reason="又一次")
        assert a["park_id"] == b["park_id"]
        assert len(board.list_parks()) == 1

    def test_re_parking_does_not_slide_the_deadline(self, board):
        """Otherwise a bounded wait silently becomes unbounded, and "it has a
        timeout" stops being a true statement about the system."""
        a = _park(board)
        time.sleep(0.01)
        b = _park(board, reason="refresh")
        assert b["deadline_at"] == a["deadline_at"]
        assert b["created_at"] == a["created_at"]

    def test_re_parking_updates_what_it_is_waiting_for(self, board):
        _park(board, waiting_for=["draft"])
        b = _park(board, waiting_for=["draft", "analysis"])
        assert b["waiting_for"] == ["draft", "analysis"]

    def test_two_experiments_get_their_own_parks(self, board):
        """Same agent, different experiments = two independent waits. Merging them
        would wake one experiment's work into the other's data."""
        _park(board, experiment_id="exp-A")
        _park(board, experiment_id="exp-B")
        assert len({r["park_id"] for r in board.list_parks()}) == 2

    def test_experiment_id_is_frozen_at_creation(self, board, monkeypatch):
        """fetch_board's trap ⑯, same hazard with a longer horizon: a park may sit
        for days and be woken while another experiment is active. Filing it into
        whichever happens to be active then is simply the wrong drawer, and nothing
        afterwards would reveal the error."""
        monkeypatch.setattr(pb, "_active_experiment_id", lambda: "exp-A")
        rec = board.park("paper_review", waiting_for=["draft"])
        assert rec["experiment_id"] == "exp-A"
        monkeypatch.setattr(pb, "_active_experiment_id", lambda: "exp-B")
        again = board.get(rec["park_id"])
        assert again["experiment_id"] == "exp-A", \
            "the park was re-attributed to whatever experiment is active NOW"

    def test_a_full_board_refuses_rather_than_growing_without_limit(self, board,
                                                                   monkeypatch):
        monkeypatch.setattr(pb, "MAX_ENTRIES", 2)
        _park(board, "paper_review")
        _park(board, "data_processing")
        out = board.park("paper_writing", waiting_for=["analysis"],
                         experiment_id="exp-A")
        assert out.get("error")

    def test_no_active_experiment_is_not_an_error(self, board, monkeypatch):
        """The board has to keep working with no runtime, no DB and no experiment."""
        monkeypatch.setattr(pb, "_active_experiment_id", lambda: "")
        rec = board.park("paper_review", waiting_for=["draft"])
        assert rec["experiment_id"] == ""


# ════════════════════════════════════════════════════════════════════
# The lifecycle, and why `woken` is not the end
# ════════════════════════════════════════════════════════════════════

class TestLifecycle:
    def test_woken_is_not_terminal(self, board):
        """A woken run is detached on an InMemorySaver — if it dies, its work is
        gone. Treating ``woken`` as the end would leave no record that the chain
        broke, which is a task leaking into silence."""
        rec = _park(board)
        w = board.mark_woken(rec["park_id"], "run-123")
        assert w["status"] == "woken"
        assert w["woken_run_id"] == "run-123"
        assert "woken" in pb.STATUSES and "done" in pb.STATUSES
        assert w["status"] != "done", "woken must not be the terminal state"

    def test_done_is_the_clean_end(self, board):
        rec = _park(board)
        board.mark_woken(rec["park_id"], "run-1")
        d = board.mark_done(rec["park_id"], note="产物已回主线")
        assert d["status"] == "done"

    def test_cancel_is_available_without_pretending_work_happened(self, board):
        rec = _park(board)
        assert board.cancel(rec["park_id"])["status"] == "cancelled"

    def test_mutating_an_unknown_park_is_an_error_not_a_crash(self, board):
        assert board.mark_woken("pk-nope", "r")["error"]
        assert board.mark_done("pk-nope")["error"]
        assert board.acknowledge("pk-nope")["error"]


class TestDecisionsAreRecorded:
    def test_a_decline_is_counted_and_explained(self, board):
        """Without the reason the board shows a park getting older with no
        explanation — the same unreadable state as a hang. And the count feeds the
        next question, because an agent that does not know it has declined three
        times cannot decide better than it did the first."""
        rec = _park(board)
        out = board.note_decision(rec["park_id"], woke=False, reason="分析还没到")
        assert out["declines"] == 1
        assert out["history"][-1]["reason"] == "分析还没到"
        assert out["history"][-1]["woke"] is False

    def test_waking_does_not_increment_declines(self, board):
        rec = _park(board)
        out = board.note_decision(rec["park_id"], woke=True, reason="到位了")
        assert out["declines"] == 0

    def test_versions_are_remembered_so_the_same_state_is_not_re_asked(self, board):
        rec = _park(board)
        out = board.note_decision(rec["park_id"], woke=False, reason="x",
                                  versions={"draft": 3})
        assert out["asked_at_versions"] == {"draft": 3}
        assert out["asked_at"] > 0

    def test_history_is_bounded(self, board):
        """A park polled for days must not grow a file without limit."""
        rec = _park(board)
        for i in range(70):
            board.note_decision(rec["park_id"], woke=False, reason=f"r{i}")
        assert len(board.get(rec["park_id"])["history"]) <= 50


# ════════════════════════════════════════════════════════════════════
# Expiry: escalate, never resume, never disappear
# ════════════════════════════════════════════════════════════════════

class TestExpiry:
    def test_a_past_deadline_expires_on_sweep(self, board):
        rec = board.park("paper_review", waiting_for=["draft"],
                         experiment_id="exp-A", ttl_s=60)
        # ttl is floored at 60s, so move the deadline instead of sleeping.
        board._items[rec["park_id"]]["deadline_at"] = time.time() - 1
        out = board.sweep_expired()
        assert [r["park_id"] for r in out] == [rec["park_id"]]
        assert board.get(rec["park_id"])["status"] == "expired"

    def test_expiry_does_not_resume_the_agent(self, board):
        rec = _park(board)
        board._items[rec["park_id"]]["deadline_at"] = time.time() - 1
        board.sweep_expired()
        assert board.get(rec["park_id"])["status"] == "expired", \
            "an expired wait must NOT quietly become a dispatch — the agent " \
            "declined, and overriding that silently is worse than asking"

    def test_expiry_does_not_delete_the_record(self, board):
        rec = _park(board)
        board._items[rec["park_id"]]["deadline_at"] = time.time() - 1
        board.sweep_expired()
        assert board.get(rec["park_id"]) is not None, \
            "the evidence that a wait timed out is the most useful thing about it"

    def test_an_expired_park_needs_attention_until_acknowledged(self, board):
        """A one-shot SSE frame or transcript line is not a delivery when nobody is
        watching — and nobody watching is the normal case here."""
        rec = _park(board)
        board._items[rec["park_id"]]["deadline_at"] = time.time() - 1
        board.sweep_expired()
        assert [r["park_id"] for r in board.needs_attention()] == [rec["park_id"]]
        board.acknowledge(rec["park_id"])
        assert board.needs_attention() == []

    def test_acknowledging_does_not_resume_or_delete(self, board):
        rec = _park(board)
        board._items[rec["park_id"]]["deadline_at"] = time.time() - 1
        board.sweep_expired()
        board.acknowledge(rec["park_id"])
        row = board.get(rec["park_id"])
        assert row["status"] == "expired" and row["acknowledged_at"] > 0

    def test_sweeping_twice_does_not_double_report(self, board):
        rec = _park(board)
        board._items[rec["park_id"]]["deadline_at"] = time.time() - 1
        assert len(board.sweep_expired()) == 1
        assert board.sweep_expired() == []

    def test_a_ttl_below_the_floor_is_raised_not_accepted(self, board):
        """A 0-second TTL would expire a park before anything could possibly
        arrive — a timeout that guarantees failure is not a timeout."""
        rec = board.park("paper_review", waiting_for=["draft"],
                         experiment_id="exp-A", ttl_s=0)
        assert rec["deadline_at"] - rec["created_at"] >= 60


# ════════════════════════════════════════════════════════════════════
# Persistence
# ════════════════════════════════════════════════════════════════════

class TestPersistence:
    def test_a_park_survives_a_restart(self, tmp_path):
        """The whole reason this board exists rather than living in state."""
        p = tmp_path / "b.json"
        b1 = pb.ParkBoard(p)
        rec = b1.park("paper_review", waiting_for=["draft"], experiment_id="exp-A")
        b2 = pb.ParkBoard(p)
        assert b2.get(rec["park_id"])["agent"] == "paper_review"

    def test_the_write_is_atomic(self, tmp_path):
        """temp+replace, so a crash mid-write leaves the previous board intact rather
        than a truncated file that loads as "no parks"."""
        p = tmp_path / "b.json"
        b = pb.ParkBoard(p)
        b.park("paper_review", waiting_for=["draft"], experiment_id="exp-A")
        assert p.exists()
        assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"
        json.loads(p.read_text(encoding="utf-8"))  # valid JSON

    def test_a_corrupt_board_degrades_to_empty_rather_than_crashing(self, tmp_path):
        p = tmp_path / "b.json"
        p.write_text("{ this is not json", encoding="utf-8")
        b = pb.ParkBoard(p)
        assert b.list_parks() == []

    def test_an_unwritable_path_does_not_raise_at_the_caller(self, tmp_path):
        """Best-effort persistence: failing to save must degrade to a park that does
        not survive the run, never to a failed routing decision."""
        b = pb.ParkBoard(tmp_path / "nested" / "b.json")
        b._path = tmp_path  # a directory — writing here will fail
        rec = b.park("paper_review", waiting_for=["draft"], experiment_id="exp-A")
        assert rec.get("park_id")


# ════════════════════════════════════════════════════════════════════
# Reads the rest of the system depends on
# ════════════════════════════════════════════════════════════════════

class TestReads:
    def test_attention_rows_sort_first(self, board):
        """An expired park is the one thing here that needs a human; it must not sit
        below ten healthy waits."""
        _park(board, "paper_review")
        rec2 = _park(board, "data_processing", waiting_for=["last_scan"])
        board._items[rec2["park_id"]]["deadline_at"] = time.time() - 1
        board.sweep_expired()
        assert board.list_parks()[0]["park_id"] == rec2["park_id"]

    def test_terminal_rows_can_be_excluded(self, board):
        rec = _park(board)
        board.mark_done(rec["park_id"])
        assert board.list_parks(include_terminal=False) == []

    def test_as_state_cache_is_the_shape_pending_activations_expects(self, board):
        rec = _park(board, reason="没有草稿", hard=True)
        cache = board.as_state_cache("exp-A")
        assert cache["paper_review"]["status"] == "waiting"
        assert cache["paper_review"]["waiting_for"] == ["draft"]
        assert cache["paper_review"]["park_id"] == rec["park_id"]
        assert cache["paper_review"]["hard"] is True

    def test_as_state_cache_excludes_finished_parks(self, board):
        """Otherwise a fresh run would inherit a park that is already resolved and
        decline to dispatch an agent that has nothing left to wait for."""
        rec = _park(board)
        board.mark_done(rec["park_id"])
        assert board.as_state_cache("exp-A") == {}

    def test_as_state_cache_is_scoped_to_one_experiment(self, board):
        _park(board, "paper_review", experiment_id="exp-A")
        _park(board, "data_processing", waiting_for=["last_scan"],
              experiment_id="exp-B")
        assert set(board.as_state_cache("exp-A")) == {"paper_review"}

    def test_open_parks_filters_by_status(self, board):
        rec = _park(board)
        board.mark_woken(rec["park_id"], "r1")
        assert board.open_parks("exp-A") == []


# ── campaign_id 在建 park 时冻结（2026-08-27） ──────────────────────

def test_campaign_id_is_frozen_at_park_time(board):
    """空闲进程里没有任何东西能重新推导出这份等待属于哪条纲领。

    v1 实验行的 ``v2_campaign_id`` 只指向 GUI 作用域纲领，而
    research_director 建的纲领与任何实验都没有链接 —— 猜一个的后果是把 A
    纲领的判据套到 B 的 park 上，静默地不唤醒。
    """
    b = board
    rec = _park(b, "literature", waiting_for=["experiment_plan"],
                campaign_id="cmp-1")
    assert rec["campaign_id"] == "cmp-1"
    assert b.open_parks()[0]["campaign_id"] == "cmp-1"


def test_a_park_without_a_campaign_says_so_rather_than_guessing(board):
    rec = _park(board, "literature", waiting_for=["experiment_plan"])
    assert rec["campaign_id"] == ""


def test_refreshing_without_a_campaign_does_not_erase_the_frozen_one(board):
    """重新 park 拿不到 campaign 时不许把已冻结的归属抹掉。

    抹掉之后这份 park 的目标判据从此永远判不了，而且没有任何地方会说一声。
    """
    b = board
    _park(b, "literature", waiting_for=["experiment_plan"], campaign_id="cmp-1")
    _park(b, "literature", waiting_for=["analysis"])        # 同一个 (agent, exp)
    open_ = b.open_parks()
    assert len(open_) == 1, "去重没生效，这条测试没测到刷新那条路"
    assert open_[0]["campaign_id"] == "cmp-1"
    assert open_[0]["waiting_for"] == ["analysis"], "刷新本身要照常生效"


# ── done_by_goal 是终态（2026-08-27） ───────────────────────────────

def test_done_by_goal_is_terminal(board):
    """它不该再出现在任何「还要处理」的清单里。

    与 ``done`` 分开是因为两者回答不同的问题：``done`` 说「醒了、干完了、产物
    回来了」，``done_by_goal`` 说「一次都没醒，而且不用醒了」。合成一个，用户
    就再也答不出「那件事到底是谁做的」。
    """
    p = _park(board)
    board.mark_done_by_goal(p["park_id"], reason="判据全满足", campaign_id="cmp-1")
    assert board.open_parks() == []
    assert board.list_parks(include_terminal=False) == []
    assert board.needs_attention() == []
    row = board.list_parks()[0]
    assert row["status"] == "done_by_goal"
    assert row["goal_campaign_id"] == "cmp-1"


def test_note_goal_check_does_not_rewrite_the_file_when_nothing_changed(board):
    """调度器每 60 s 走一趟。结论没变就不该写盘。

    **直接比文件内容**，不比内存副本 —— 「好好的直到进程结束」对一个靠熬过进程
    的记录是最坏的失效方式，本仓为此付过一次账。
    """
    p = _park(board)
    pid = p["park_id"]
    board.note_goal_check(pid, verdict="not_done", reason="还差分析")
    first = board._path.read_text(encoding="utf-8")

    board.note_goal_check(pid, verdict="not_done", reason="还差分析")
    assert board._path.read_text(encoding="utf-8") == first, "同一结论又写了一次盘"

    board.note_goal_check(pid, verdict="unknown", reason="库读不到")
    assert board._path.read_text(encoding="utf-8") != first, "结论变了却没落盘"
    assert board.list_parks()[0]["goal_check"]["verdict"] == "unknown"


def test_every_status_has_a_label_in_the_panel():
    """`STATUSES` 的每一项都要在前端的标签表里有名字。

    漏一个的后果不是报错，是那一行显示成一个裸的英文状态码 —— 用户看到
    `done_by_goal` 时并不知道它是好事还是出事。三份手抄的名单总有一份会漏，
    所以这里按**后端闭集**去查前端，而不是反过来。
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve()
    while root.parent != root and not (root / "frontend").is_dir():
        root = root.parent
    panel = root / "frontend" / "src" / "components" / "agents" / \
        "PendingActivationsPanel.tsx"
    src = panel.read_text(encoding="utf-8")

    m = re.search(r"const STATUS_LABEL: Record<string, string> = \{(.*?)\};",
                  src, re.S)
    assert m, "前端的 STATUS_LABEL 找不到了 —— 扫描器坏了，不是「都盖住了」"
    labelled = set(re.findall(r"^\s*(\w+):", m.group(1), re.M))
    missing = [s for s in pb.STATUSES if s not in labelled]
    assert missing == [], f"这些状态在面板上没有中文名：{missing}"

    # 扫描器自检：它必须真的看见了几个已知状态，否则「一个都不缺」可能只是
    # 正则没匹配上。
    assert {"waiting", "woken"} <= labelled
