"""Process-wide arbitration for the ONE physical instrument.

Why this exists (dispatch design audit 2026-07-28, 致命一)
---------------------------------------------------------
Several independent entry points can drive the same Nanonis controller, each
building its own graph / tools / middleware over the **same** ``ConnectionPool``:
the group-chat orchestrator's instrument_control agent, the private chat (the
main chat IS a private instrument_control agent), the signals routes building
their own ``ExecutionContext`` on the same pool, and by now several more.

An exhaustive search of the tree found **no** global instrument lock, busy flag,
experiment lock, serialising queue or single-worker executor. The only lock was
``ConnectionPool``'s per-role one, whose critical section is a *single TCP
command round-trip* — its own comment says so: "or two callers would interleave
bytes on one Nanonis socket". That is byte-level corruption protection, not run
arbitration. The concurrency safety argument on record
(``test_parallel_real_graphs.py:399``: "instrument control is just one agent …
therefore parallelism is completely safe") only ever covered the inside of the
group-chat graph.

So the real behaviour was: the group chat runs an AutoApproach composite while
the operator asks the main chat to change the bias, and the two skill sequences
interleave command by command into one instrument. No refusal, no queue, no
warning. Tailscale remote access (v5.4.0) makes "two entry points live at once"
routine rather than exotic.

What this is
------------
One **re-entrant, process-wide token** taken at SKILL granularity. Not per TCP
command (the pool already does that, and it is the wrong grain), not per run
(too coarse — a run is minutes long and a status read must never queue behind
it).

Design points, each of them load-bearing:

* **Re-entrant per thread.** A composite holds the token and its sub-steps
  re-acquire it on the same thread; ``threading.RLock`` makes that free. Without
  it every composite would deadlock against itself on its first sub-skill.
* **Reads never take it.** READ / ANALYSIS skills touch no instrument state, and
  making the dashboard's 2 s poll queue behind a 10-minute scan would violate
  the UI-never-freezes invariant to buy nothing.
* **Remedies never take it.** A retract or a stop must run *while* someone else
  holds the instrument — that is the whole point of an emergency. Gating the
  fix behind the token held by the thing that needs fixing is a deadlock with a
  broken tip at the end of it.
* **Refuse, do not queue.** Waiting is bounded (:data:`DEFAULT_WAIT_S`) and the
  failure is a clear, non-retryable message naming the current holder. An agent
  told "someone else is driving the instrument" must not spin.

谁在这道闸后面（2026-08-20 改成清单，不再点数）
--------------------------------------------------
这份 docstring 从前的开头是 "Three independent entry points"，`core/executor.py`
写着自己是 "the THIRD driver"，`api/routes/skill_exec.py` 写着自己是「第四个入口」，
而 conduct 的设计文档写着它是「第 5 个」。**四份文档四种数法，实际 owner 字符串
至少八个。** 手工维护的计数必漂，所以这里改成两件可查的东西：

**取令牌的机制只有四个**（新增入口必须是其中之一，否则就是又一次「共用一个
ConnectionPool 而没有仲裁」）：

===================================== ==========================================
`core/executor.py`                     手动 / GUI 按钮（SkillExecutor）
`core/execution_context.py`            一切 ExecutionContext 持有者
                                       （composite 子步 · 技能直调 API ·
                                        信号 API · conduct Director · …）
`agents/_shared/skill_adapter.py`      agent 的工具边界（群聊 IC / 私聊 IC）
本模块的 :func:`hold_for_skill`        定义本身
===================================== ==========================================

**owner 是开放集**，只用来在拒绝消息里说清「现在开车的是谁」。今天在册的有：
群聊任务 / 主聊天·私聊 / 手动·执行器 / 信号采集 API / 技能直调 API /
外部 agent ``ext:<名>``（``/api/ext/v1`` 作业，经 ``api/direct_exec``）/
``conduct:<id>`` / 新仪器初始化对账 / 设置页·Nanonis 保存目录。**加一个新 owner
不需要改这里**；加一条新的取令牌路径需要——而那件事有结构测试钉着
(`tests/v2/unit/core/test_instrument_lock_entry_inventory.py`)。
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

#: How long a caller waits for the token before being refused. Short on
#: purpose: it absorbs brief overlaps (one skill finishing as another starts)
#: without ever letting a queue build up behind a long scan.
DEFAULT_WAIT_S = 5.0

#: Skill metadata tags that mark a remedy — see the module docstring. Kept in
#: sync with ``agents/_shared/buffer_hitl.REMEDY_TAGS`` on purpose: both answer
#: the same question ("may this run while the system is in trouble?").
BYPASS_TAGS: frozenset[str] = frozenset({"retract", "emergency", "withdraw"})

#: Names that are remedies without carrying a tag for it.
BYPASS_NAMES: frozenset[str] = frozenset({
    "StopScan", "StopSTS", "StopMotor", "StopAutoApproach",
    "SafeRetract", "EmergencyRetract", "WithdrawTip",
})


class InstrumentBusy(RuntimeError):
    """Raised when the instrument token could not be taken in time."""

    def __init__(self, holder: dict[str, Any] | None, skill: str, waited_s: float):
        self.holder = holder or {}
        self.skill = skill
        self.waited_s = waited_s
        super().__init__(self.message())

    def message(self) -> str:
        who = self.holder.get("owner") or "另一个入口"
        what = self.holder.get("skill") or "某个仪器动作"
        held = self.holder.get("held_s")
        held_txt = f"，已持有 {held:.0f}s" if isinstance(held, (int, float)) else ""
        return (
            f"仪器正被占用：{who} 正在执行 {what}{held_txt}。"
            f"'{self.skill}' 未执行——同一台 Nanonis 不能被两条链路同时驱动。"
            f"请等对方结束后再试，或先中止对方的运行。"
            f"Do NOT retry immediately — 重试不会让对方更快结束。"
        )


class InstrumentLock:
    """Re-entrant single-holder token over the one instrument."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._meta = threading.Lock()
        self._owner: str = ""
        self._skill: str = ""
        self._since: float = 0.0
        self._depth: int = 0
        self._thread_id: int = 0
        #: 最后一次**完全**放手的时刻(``time.monotonic()``);从未持有过就是 ``0.0``。
        #: 只在 depth 归 0 时写 —— 见 :meth:`release`。
        self._last_released_at: float = 0.0

    # ── introspection ────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any] | None:
        """Who holds it right now, or None. JSON-safe."""
        with self._meta:
            if self._depth <= 0:
                return None
            return {
                "owner": self._owner,
                "skill": self._skill,
                "since": self._since,
                "held_s": max(0.0, time.monotonic() - self._since),
                "depth": self._depth,
                "thread_id": self._thread_id,
            }

    def idle_s(self) -> float | None:
        """令牌**空着**多久了(秒);正被持有时返回 ``None``,从未持有过返回 ``inf``。

        ## 这个方法是给谁用的

        看门狗要回答的问题是「**现在动仪器的是我们还是人**」。
        「持有令牌」回答了一半 —— 但技能与技能之间(agent 在想下一步)令牌是空的,
        而那时把针放在接触位置的仍然是 MAST。这个方法补的就是那半:
        **刚刚还在操作**,和**已经很久没碰过了**,是两件事。

        ## 为什么是令牌,不是「最近发过写命令」

        「最近发过写命令」是这件事的一个**漏的代理**:一次长扫描是
        **持续操作、零写命令** —— MAST 发的是 ``Scan_StatusGet`` 这类读,
        而 2026-08-08 那 13 分钟正是这种。用写命令做判据会把那段判成「人在操作」,
        于是把刚修好的那个失败重新做出来。

        令牌没有这个问题:``hold_for_skill`` 在**最外层技能**上获取,
        **整条 composite 期间一直持有**(子步可重入不改写),所以长扫描期间它是满的。

        而且它不需要「哪些方法算写」这个判据 —— ``needs_token`` 早就做完了那个决定
        (读与解药不取令牌),**这里复用的不是它的代码,是它的事实**。
        监控自己 2 Hz 的读、看门狗自己的读,天然不污染这个数。

        ``monotonic`` 不受系统时钟调整影响。
        """
        with self._meta:
            if self._depth > 0:
                return None
            if self._last_released_at <= 0.0:
                return float("inf")
            return max(0.0, time.monotonic() - self._last_released_at)

    def held_by_this_thread(self) -> bool:
        with self._meta:
            return self._depth > 0 and self._thread_id == threading.get_ident()

    # ── acquire / release ────────────────────────────────────────────

    def acquire(self, *, owner: str, skill: str,
                timeout_s: float = DEFAULT_WAIT_S) -> bool:
        """Take the token. Returns False if it stayed busy for *timeout_s*."""
        t0 = time.monotonic()
        got = self._lock.acquire(timeout=timeout_s) if timeout_s > 0 \
            else self._lock.acquire(blocking=False)
        if not got:
            return False
        with self._meta:
            if self._depth == 0:
                self._owner = owner
                self._skill = skill
                self._since = time.monotonic()
                self._thread_id = threading.get_ident()
            self._depth += 1
        waited = time.monotonic() - t0
        if waited > 0.25:
            logger.info("instrument token acquired by %s for %s after %.2fs wait",
                        owner, skill, waited)
        return True

    def release(self) -> None:
        with self._meta:
            if self._depth > 0:
                self._depth -= 1
                if self._depth == 0:
                    self._owner = ""
                    self._skill = ""
                    self._since = 0.0
                    self._thread_id = 0
                    # ⚠️ 只在**完全**放手时写。令牌是可重入的,composite 的每个子步
                    # 都会 release 一次;在嵌套的那些上写会让「上次放手」变成
                    # 「上一个子步结束」,而那时 MAST 明明还在驱动 —— 这个数就不再
                    # 回答它自己那个问题了。见 :meth:`idle_s`。
                    self._last_released_at = time.monotonic()
        try:
            self._lock.release()
        except RuntimeError:  # pragma: no cover — released by a foreign thread
            logger.warning("instrument token released without being held")

    @contextlib.contextmanager
    def hold(self, *, owner: str, skill: str,
             timeout_s: float = DEFAULT_WAIT_S):
        """Context manager. Raises :class:`InstrumentBusy` rather than queueing."""
        t0 = time.monotonic()
        holder_before = self.snapshot()
        if not self.acquire(owner=owner, skill=skill, timeout_s=timeout_s):
            waited = time.monotonic() - t0
            holder = self.snapshot() or holder_before
            logger.warning(
                "instrument BUSY — refused '%s' for %s after %.1fs (holder=%s)",
                skill, owner, waited, holder)
            try:
                from mast.core.diagnostics import record as _diag

                _diag("instrument_busy", skill,
                      "另一条链路正在驱动仪器，本次动作被拒绝（未下发任何命令）",
                      requester=owner, holder=holder, waited_s=round(waited, 2))
            except Exception:  # noqa: BLE001
                pass
            raise InstrumentBusy(holder, skill, waited)
        try:
            yield
        finally:
            self.release()

    # tests only
    def _reset_for_tests(self) -> None:  # pragma: no cover - test helper
        self.__init__()


_INSTRUMENT_LOCK = InstrumentLock()


def instrument_lock() -> InstrumentLock:
    """The one process-wide token. Every entry point shares this object."""
    return _INSTRUMENT_LOCK


# ─────────────────────────────────────────────────────────────────────
# Which skills take the token
# ─────────────────────────────────────────────────────────────────────

def needs_token(meta: Any, skill_name: str = "") -> bool:
    """True iff running this skill should hold the instrument token.

    False for reads/analysis (they must never queue behind a long write) and for
    the stop/retract remedies (they must run *while* someone else holds it).
    Unknown metadata → True: a skill we cannot classify is assumed to drive the
    instrument, because the fail-open direction here is two chains writing at
    once.
    """
    name = str(skill_name or getattr(meta, "name", "") or "")
    if name in BYPASS_NAMES:
        return False
    category = getattr(getattr(meta, "category", None), "value", None)
    if str(category).lower() in ("read", "analysis"):
        return False
    tags = {str(t).lower() for t in (getattr(meta, "tags", None) or ())}
    if tags & BYPASS_TAGS:
        return False
    return True


@contextlib.contextmanager
def hold_for_skill(meta: Any, skill_name: str, owner: str,
                   timeout_s: float = DEFAULT_WAIT_S):
    """``instrument_lock().hold`` if this skill needs it, else a no-op.

    The single helper every entry point calls, so the "which skills arbitrate"
    rule lives in exactly one place.
    """
    if not needs_token(meta, skill_name):
        yield
        return
    with instrument_lock().hold(owner=owner, skill=skill_name,
                                timeout_s=timeout_s):
        yield


__all__ = [
    "InstrumentBusy",
    "InstrumentLock",
    "instrument_lock",
    "needs_token",
    "hold_for_skill",
    "DEFAULT_WAIT_S",
    "BYPASS_NAMES",
    "BYPASS_TAGS",
]
