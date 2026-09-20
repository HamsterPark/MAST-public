"""Waking parked agents when the thing they waited for arrives.

``docs/v2/design/wakeup_scheduling.md`` §5.4. The operator's ask:

    "可以等到有了新资料,再被激活再工作。"

Why a thread and not an event handler
-------------------------------------
The obvious design is "subscribe to ARTIFACT_SAVED and wake whoever was waiting".
That cannot work as the primary path, for two reasons that are properties of the
system rather than preferences:

* ``EventBus`` fans out SYNCHRONOUSLY on the publishing thread and swallows
  subscriber exceptions. A subscriber that asked an LLM whether to wake would block
  the thread that just saved a document, for seconds, silently.
* The pre-existing notification path contains ``if not task.get("active"): return`` —
  when no foreground run is active, notifications are DROPPED. Which is exactly the
  situation waking exists for.

So this is a slow poll on its own daemon thread, shaped after ``memory/dreaming.py``
(1 s tick accumulating to an interval, an injected idle predicate, daemon, joinable
``stop()``). Being outside the graph, it MAY block — the no-sleep invariant binds
``agents/**/graph.py`` nodes, and ``runtime._background_run_fn`` already states that
distinction for the same reason. Events are still useful as a hint that shortens
latency; they are not load-bearing.

⚠️ The circuit breaker is not optional
--------------------------------------
Product-driven waking creates a loop the existing guards cannot see. paper_writing
wakes, writes a draft, which wakes paper_review, which writes a review, which wakes
paper_writing to revise… This is the A→B→A ping-pong that agent-to-agent handoff was
removed for on 2026-05-30 — with the entry point moved, and **worse**, because every
wake is a NEW run: ``recursion_limit``, ``visit_count`` and the per-run USD budget all
reset to zero, and every step SUCCEEDS, so ``StallGuard`` (which keys off repeated
FAILURE signatures) is structurally blind to it. The measured $30.66 runaway had
exactly this shape: four identical successful cycles.

Nothing else in the system bounds it. The supervisor's loop guards are per-run; the
budget gate is per-run; ``recursion_limit`` is per-run. The only place a bound can
live is here, so:

  * a per-experiment cap on wakes per day, and
  * a system-wide daily USD ceiling (``billing.run_meter.daily_spend_usd``),

and when either is hit the scheduler STOPS waking and surfaces to the operator rather
than continuing. A wake that cannot happen is visible on the board; a spend loop that
nobody bounded is a bill.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

#: Poll interval. A park's timescale is hours and the artifacts it waits for arrive
#: minutes apart at best, so 60 s costs nothing and removes any dependence on the
#: event path working.
DEFAULT_INTERVAL_S = 60.0

#: Minimum gap between two questions about the SAME park. Without it a park waiting
#: on ``last_scan`` is re-asked every tick while a scan runs, because the "version"
#: of a scan class (count + newest mtime) changes with every saved frame — the
#: version check cannot suppress what genuinely keeps changing. 30 min is well inside
#: a 24 h deadline and bounds the questions per park per day to ~48.
DEFAULT_ASK_COOLDOWN_S = 30 * 60.0

#: Wakes per experiment per day. The loop bound. 6 is above any legitimate pipeline
#: (a six-stage campaign needs at most one wake per stage) and far below a
#: ping-pong, which produces them as fast as runs finish.
#:
#: ⚠️ ``0`` means ZERO WAKES, not "no limit" — use a NEGATIVE value to disable the
#: check. That asymmetry with the USD ceiling below is deliberate and is the safe
#: reading of each: for a spend ceiling, 0 is what an unconfigured setting looks
#: like, so it has to mean "no ceiling"; for a safety counter, an operator who
#: types 0 means "do not do this", and handing them unlimited waking would be the
#: exact opposite of what they asked for.
DEFAULT_MAX_WAKES_PER_DAY = 6

#: System-wide daily USD ceiling across runs. Per-run limits cannot constrain
#: repeated wakeups. The separate wake-count quota also bounds loops when billing
#: data is unavailable; it does not depend on costs becoming readable.
DEFAULT_DAILY_BUDGET_USD = 300.0


def _artifact_versions() -> dict:
    """A comparable "version" per waitable field, from DISK.

    Documents get their latest version number; scan data gets ``(count, newest
    mtime)`` because .sxm files have no version — they accumulate. Both are only ever
    compared for INEQUALITY, so the two shapes need not be commensurable; they only
    need to change when something arrives, and not otherwise.

    Returns ``{}`` when nothing can be read. An empty dict compares unequal to a
    stored non-empty one, so a failed read must not be treated as "something arrived"
    — the caller checks for emptiness rather than diffing blindly.
    """
    out: dict = {}
    try:
        from mast.agents._shared.artifacts import list_existing

        rows = list_existing() or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("wake: cannot read artifacts (%s)", exc)
        return {}
    try:
        from mast.agents._shared.artifact_channel import _FIELD_TO_ARTIFACT_CLASS

        mapping = dict(_FIELD_TO_ARTIFACT_CLASS)
    except Exception as exc:  # noqa: BLE001
        logger.debug("wake: field mapping unavailable (%s)", exc)
        return {}
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(str(r.get("artifact_id") or ""), []).append(r)
    for field, classes in mapping.items():
        count = 0
        newest = 0.0
        for cid in classes:
            for item in by_class.get(cid, ()):
                count += 1
                newest = max(newest, float(item.get("modified_at") or 0.0))
        out[field] = (count, round(newest, 3))
    return out


def _arrived(park: dict, versions: dict) -> list[str]:
    """Which of this park's awaited fields look like they now have something.

    Deliberately conservative: a field counts as arrived only if it is PRESENT now
    (count > 0) and its version differs from the one recorded when the park was last
    asked. Presence alone would re-report the same artifact forever; change alone
    would fire on a deletion.
    """
    asked = park.get("asked_at_versions") or {}
    out: list[str] = []
    for field in (park.get("waiting_for") or []):
        now = versions.get(field)
        if not now or not now[0]:
            continue
        if asked.get(field) != now:
            out.append(field)
    return out


class WakeScheduler:
    """Polls the park board and wakes agents whose inputs have arrived.

    Shaped after :class:`mast.memory.dreaming.DreamingService`; see the module
    docstring for why polling rather than pure event handling.

    Injected collaborators, all optional so this class is testable and so a missing
    runtime degrades to "no waking" rather than a crash:

      * ``should_wake()`` — idle predicate. Wired to "no foreground run active", so
        the scheduler and the mainline supervisor never decide at the same time.
      * ``ask()`` — the wake question. ``(park, arrived, versions) -> {action, ...}``.
      * ``spawn()`` — start the detached run. ``(park) -> run_id``.
      * ``escalate()`` — tell the operator. ``(kind, park|None, detail)``.
      * ``goal_check()`` — 这份 park 服务的目标达成了没有（2026-08-27）。
        ``(park) -> GoalVerdict``。**``None``（缺省）⇒ 这段逻辑整个不存在**，
        代码路径逐字节等于它出现之前。注入而不是直接 import，是为了让本模块
        （``mast.core``）不依赖 ``mast.goals`` / ``mast.conduct`` —— 有一条
        AST 结构测试钉着这句话。
    """

    def __init__(self, *, interval_s: float = DEFAULT_INTERVAL_S,
                 board=None, should_wake=None, ask=None, spawn=None,
                 escalate=None, goal_check=None,
                 max_wakes_per_day: int = DEFAULT_MAX_WAKES_PER_DAY,
                 daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD,
                 ask_cooldown_s: float = DEFAULT_ASK_COOLDOWN_S) -> None:
        self._interval = max(5.0, float(interval_s))
        self._board = board
        self._should_wake = should_wake or (lambda: True)
        self._ask = ask
        self._spawn = spawn
        self._escalate = escalate or (lambda *a, **k: None)
        # None ⇒ 目标闸门不存在。**不给它一个「恒回 not_done」的缺省** ——
        # 那会让「没接上」看起来像「判过了，还没到」。
        self._goal_check = goal_check
        # NOT clamped at 0: negative = disabled, 0 = zero wakes allowed. See the
        # constant's note for why a safety counter must read 0 that way.
        self._max_wakes = int(max_wakes_per_day)
        self._daily_budget = max(0.0, float(daily_budget_usd or 0.0))
        self._cooldown = max(0.0, float(ask_cooldown_s))
        self._running = False
        self._thread: "threading.Thread | None" = None
        #: Wakes granted today, per experiment. Reset when the local day rolls over.
        self._wakes: dict[str, int] = {}
        self._wakes_day: float = 0.0
        #: Set once per day when a breaker trips, so the operator gets ONE escalation
        #: rather than one per tick — an alarm that repeats every minute is an alarm
        #: that gets muted, and a muted alarm is the same as no alarm.
        self._breaker_reported: set[str] = set()
        self._hint = threading.Event()

    # ── board access ─────────────────────────────────────────────────
    def _get_board(self):
        if self._board is not None:
            return self._board
        from mast.core.park_board import board

        return board()

    # ── circuit breakers ─────────────────────────────────────────────
    def _roll_day(self) -> None:
        from mast.billing.run_meter import _local_midnight_ts

        today = _local_midnight_ts()
        if today != self._wakes_day:
            self._wakes_day = today
            self._wakes = {}
            self._breaker_reported = set()

    def budget_blocked(self) -> "str | None":
        """Reason the daily USD ceiling forbids waking right now, or ``None``.

        ``None`` also covers "spend is unreadable": a billing hiccup must not stop the
        system working, in the same direction the per-run gate treats it (see
        ``run_meter``'s three-valued contract). The per-experiment count cap below is
        the bound that does not depend on billing being readable.
        """
        if self._daily_budget <= 0:
            return None
        try:
            from mast.billing.run_meter import daily_spend_usd

            spent = daily_spend_usd()
        except Exception as exc:  # noqa: BLE001
            logger.debug("wake: daily spend unreadable (%s)", exc)
            return None
        if spent is None:
            return None
        if spent >= self._daily_budget:
            return (f"今日全部 run 的花销已达 ${spent:.2f}(上限 ${self._daily_budget:.2f}),"
                    "本日不再自动唤醒")
        return None

    def wake_quota_blocked(self, experiment_id: str) -> "str | None":
        """Reason the per-experiment daily wake count forbids waking, or ``None``.

        THE loop bound. Product-driven waking has no other: every wake is a fresh run,
        so the per-run loop guards, the per-run budget and recursion_limit all reset,
        and each step succeeds so StallGuard cannot see the cycle.

        Negative = disabled. **Zero = zero wakes**, not unlimited.
        """
        if self._max_wakes < 0:
            return None
        self._roll_day()
        used = int(self._wakes.get(experiment_id or "", 0))
        if used >= self._max_wakes:
            if self._max_wakes == 0:
                return "自动唤醒已被设置为 0 次/天(关闭),本实验不会被自动唤醒"
            return (f"本实验今天已自动唤醒 {used} 次(上限 {self._max_wakes}),"
                    "为避免 A→B→A 反复唤醒的死循环,本日不再自动唤醒")
        return None

    def _note_wake(self, experiment_id: str) -> None:
        self._roll_day()
        key = experiment_id or ""
        self._wakes[key] = int(self._wakes.get(key, 0)) + 1

    def _report_breaker(self, key: str, reason: str, park: "dict | None") -> None:
        if key in self._breaker_reported:
            return
        self._breaker_reported.add(key)
        logger.warning("wake scheduler breaker: %s", reason)
        try:
            self._escalate("breaker", park, reason)
        except Exception as exc:  # noqa: BLE001
            logger.debug("wake: escalate failed (%s)", exc)

    # ── goal gate ────────────────────────────────────────────────────
    def _goal_pass(self, board, parks: list, summary: dict) -> list:
        """关掉目标已达成的 park，返回还需要考虑的那些。

        为什么这道闸在这里而不在别处：唤醒环是**唯一**能给「停不下来」放边界的
        地方 —— 每次唤醒都是新 run，``recursion_limit`` / ``visit_count`` /
        per-run 预算全部归零，而且每一步都 SUCCEEDS，所以 StallGuard（按重复
        **失败**签名）结构上看不见成功循环。次数配额限制唤醒总量，
        此处进一步阻止目标已经达成后再次唤醒。

        **不计入配额、不读设置、不花钱。** 关掉一份不必再醒的等待不是一次唤醒。

        ``unknown`` ⇒ **照旧唤醒**。失败代价不对称：判不了却照醒，最坏是对一个
        其实已达成的目标多醒一次 —— 有界（每实验每日次数 + 日金额），而且**就是
        今天的行为**；判不了就不醒，则一次 DB 读失败会造出一个永远不醒的 park，
        只能等 24 h 超时才浮出来，那正是本设计自己点名「最像静默死亡通道」的
        形状，而且无界。同构先例：``budget_blocked`` 在账本读不到时回 ``None``
        （= 不拦）。

        但判不了必须**可见**：写进 park 的 ``goal_check``，面板据此显示
        「目标判据：读不到（原因）」。
        """
        keep: list = []
        for park in parks:
            pid = park.get("park_id") or ""
            try:
                verdict = self._goal_check(park)
            except Exception as exc:  # noqa: BLE001 — 判据坏了不许杀掉这一趟
                logger.debug("wake: goal check failed for %s (%s)", pid, exc)
                keep.append(park)
                continue
            if verdict is None:
                keep.append(park)
                continue
            state = str(getattr(verdict, "verdict", "") or "")
            reason = str(getattr(verdict, "reason", "") or "")
            if state == "done":
                try:
                    board.mark_done_by_goal(
                        pid, reason=reason,
                        campaign_id=str(park.get("campaign_id") or ""))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("wake: cannot close %s by goal (%s)", pid, exc)
                    keep.append(park)
                    continue
                summary["closed_by_goal"] += 1
                logger.info("wake: park %s closed — 目标已达成 (%s)", pid, reason)
                try:
                    self._escalate("done_by_goal", park, reason)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("wake: done_by_goal escalation failed (%s)", exc)
                continue
            # not_done / unknown ⇒ 照常走后面的流程；只在结论**变化**时落一次盘。
            try:
                board.note_goal_check(pid, verdict=state, reason=reason)
            except Exception as exc:  # noqa: BLE001
                logger.debug("wake: cannot note goal check for %s (%s)", pid, exc)
            keep.append(park)
        return keep

    # ── one pass ─────────────────────────────────────────────────────
    def tick(self) -> dict:
        """One pass. Returns a small summary for logs and tests.

        Order matters and is not arbitrary:
          1. expire overdue parks FIRST — a park past its deadline must escalate even
             if a product arrived in the same tick, because the operator asked to be
             told, and quietly waking it instead would discard that;
          2. bail out if the foreground is busy — two decision streams deciding the
             same thing is how they contradict each other;
          2.5. 关掉「目标已经达成」的 park（2026-08-27）。放在熔断**之前**：
             关掉一份不必再醒的等待既不是一次花销也不是一次唤醒，熔断日不该把
             它留到明天。放在 idle 判断**之后**：「主线在决定时调度器不决定」
             照旧 —— 关 park 也是一个决定。
          3. check the breakers BEFORE spending anything, including before the
             deterministic version read;
          4. only then ask, and only about parks whose inputs actually changed.
        """
        summary = {"expired": 0, "asked": 0, "woken": 0, "declined": 0,
                   "closed_by_goal": 0, "skipped": "", }
        try:
            b = self._get_board()
        except Exception as exc:  # noqa: BLE001
            summary["skipped"] = f"board unavailable: {exc}"
            return summary

        # 1. Deadlines. Runs even when the foreground is busy: a timeout the operator
        #    was promised must not be postponed by unrelated activity.
        try:
            for rec in b.sweep_expired():
                summary["expired"] += 1
                try:
                    self._escalate("expired", rec, "等待已超时,需要用户处理")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("wake: expiry escalation failed (%s)", exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("wake: sweep failed (%s)", exc)

        # 2. Never decide while the mainline is deciding.
        try:
            if not self._should_wake():
                summary["skipped"] = "foreground busy"
                return summary
        except Exception as exc:  # noqa: BLE001
            summary["skipped"] = f"idle predicate failed: {exc}"
            return summary

        try:
            parks = b.open_parks()
        except Exception as exc:  # noqa: BLE001
            summary["skipped"] = f"cannot list parks: {exc}"
            return summary
        if not parks:
            return summary

        # 2.5 目标闸门（2026-08-27）。``goal_check`` 没注入 ⇒ 整段跳过。
        if self._goal_check is not None:
            parks = self._goal_pass(b, parks, summary)
            if not parks:
                return summary

        # 3. Breakers, before any spend.
        blocked = self.budget_blocked()
        if blocked:
            self._report_breaker("daily_budget", blocked, None)
            summary["skipped"] = blocked
            return summary

        # 4. Deterministic version read — one disk pass shared by every park.
        versions = _artifact_versions()
        if not versions:
            summary["skipped"] = "artifact versions unreadable"
            return summary

        now = time.time()
        for park in parks:
            pid = park.get("park_id") or ""
            eid = park.get("experiment_id") or ""
            quota = self.wake_quota_blocked(eid)
            if quota:
                self._report_breaker(f"wakes:{eid}", quota, park)
                summary["skipped"] = quota
                continue
            if self._cooldown and (now - float(park.get("asked_at") or 0.0)) < self._cooldown:
                continue
            arrived = _arrived(park, versions)
            if not arrived:
                continue
            if self._ask is None:
                # No question engine wired: the honest move is to escalate, NOT to
                # wake unasked (the agent chose to wait and is owed the question) and
                # NOT to stay silent (its input is here).
                self._escalate("ready", park,
                               f"等的资料({'、'.join(arrived)})已到位,但询问引擎不可用")
                continue
            try:
                decision = self._ask(park, arrived, versions)
            except Exception as exc:  # noqa: BLE001
                logger.debug("wake: ask failed for %s (%s)", pid, exc)
                continue
            summary["asked"] += 1
            woke = str((decision or {}).get("action") or "") == "start"
            reason = str((decision or {}).get("reason") or "")
            try:
                b.note_decision(pid, woke=woke, reason=reason, versions=versions)
            except Exception as exc:  # noqa: BLE001
                logger.debug("wake: cannot record decision for %s (%s)", pid, exc)
            if not woke:
                summary["declined"] += 1
                # A park that keeps declining must also keep waiting for something
                # matchable; the question engine clamps ``waiting_for`` to the closed
                # set, so this can only narrow or re-state it.
                new_wait = list((decision or {}).get("waiting_for") or [])
                if new_wait and new_wait != list(park.get("waiting_for") or []):
                    try:
                        b.park(park.get("agent") or "", waiting_for=new_wait,
                               reason=reason, experiment_id=eid,
                               instruction=park.get("instruction") or "")
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("wake: cannot update wait for %s (%s)", pid, exc)
                continue
            if self._spawn is None:
                self._escalate("ready", park, "该醒了,但后台运行不可用")
                continue
            try:
                run_id = self._spawn(park)
            except Exception as exc:  # noqa: BLE001
                logger.warning("wake: spawn failed for %s: %s", pid, exc)
                self._escalate("spawn_failed", park, f"唤醒失败:{exc}")
                continue
            if not run_id:
                self._escalate("spawn_failed", park, "唤醒失败(未拿到 run_id)")
                continue
            try:
                b.mark_woken(pid, str(run_id))
            except Exception as exc:  # noqa: BLE001
                logger.debug("wake: cannot mark woken %s (%s)", pid, exc)
            self._note_wake(eid)
            summary["woken"] += 1
            logger.info("woke %s (park %s) → run %s; arrived=%s",
                        park.get("agent"), pid, run_id, arrived)
        return summary

    # ── thread ───────────────────────────────────────────────────────
    def nudge(self) -> None:
        """Shorten the wait until the next tick.

        What an ``ARTIFACT_SAVED`` / ``SCAN_COMPLETE`` subscriber calls. It only sets
        an Event — no decision, no I/O, no LLM — because subscribers run
        synchronously on the publishing thread and that thread is in the middle of
        saving a document. The polling loop is what actually decides.
        """
        self._hint.set()

    def subscribe_to_events(self) -> None:
        """Attach ``nudge`` to the artifact/scan events, best-effort."""
        try:
            from mast.core.events import EventBus, EventType

            def _on(event) -> None:
                if getattr(event, "type", None) in (EventType.ARTIFACT_SAVED,
                                                    EventType.SCAN_COMPLETE):
                    self.nudge()

            EventBus.get().subscribe(_on)
            self._on_event = _on          # keep a ref so it can be unsubscribed
        except Exception as exc:  # noqa: BLE001
            logger.debug("wake: event subscription unavailable (%s)", exc)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="WakeScheduler")
        self._thread.start()
        logger.info("WakeScheduler started (interval=%.0fs, max_wakes/day=%d, "
                    "daily_budget=$%.2f)",
                    self._interval, self._max_wakes, self._daily_budget)

    def stop(self) -> None:
        self._running = False
        self._hint.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _loop(self) -> None:
        # Settle before the first pass, then tick on the interval — or earlier if an
        # artifact landed. Waiting on the Event rather than sleeping is what makes a
        # nudge take effect without shortening the poll for everyone.
        while self._running:
            fired = self._hint.wait(timeout=self._interval)
            self._hint.clear()
            if not self._running:
                break
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 — a daemon must not die
                logger.debug("wake tick error: %s", exc)
            if fired:
                # An artifact just landed; a document write may still be finishing its
                # siblings (a figure + its report). A short settle avoids asking about
                # a half-arrived set, and blocking is allowed here — this is not a
                # graph node (same distinction runtime._background_run_fn records).
                time.sleep(1.0)


__all__ = ["WakeScheduler", "DEFAULT_INTERVAL_S", "DEFAULT_MAX_WAKES_PER_DAY",
           "DEFAULT_DAILY_BUDGET_USD", "DEFAULT_ASK_COOLDOWN_S"]
