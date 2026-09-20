"""StallGuardMiddleware — end a repeated-identical-failure spin, and say what it was.

This is the diagnosis someone had to make themselves, reverse-engineered from
what the system actually said:

    "运行出错：任务步数达到上限——可能某个智能体在预条件或安全门上反复失败而空转。
     已停止本次运行。"

They were right. And this middleware was ALREADY RUNNING when it happened. Three
reasons it did nothing (all fixed here):

  1. **It was wired into instrument_control only.** The other five agents had no
     stall detection at all.
  2. **It only nudged.** It appended one directive and then refused to stack a
     second — so a model that ignored the nudge spun on, unimpeded, until the
     recursion cap killed the whole run with no useful outcome. (Raising the cap
     150→500 made a runaway take LONGER. It is not detection.)
  3. **It only matched byte-identical error text.** ``timed out after 3.02s`` and
     ``timed out after 3.14s`` are the same spin and looked like two different
     failures, so the counter never reached the threshold. Any error carrying a
     number, a timestamp, or a path could not trip it — which is most of them.

Now:
  * every agent gets it;
  * failure signatures are NORMALISED (numbers, timestamps, paths, ids collapsed)
    so a spin is recognised as one;
  * the directive escalates: nudge → nudge again, harder → **stop the turn** with
    a readable conclusion rather than burning the budget;
  * every trip is written to the refusal ledger (``core.diagnostics``), so the
     next time this happens the answer is on disk instead of in the operator's head.

Still conservative where it matters: the guard only ever adds a message or ends a
turn. It never touches the instrument, never rolls anything back, and a false
positive costs one extra instruction.

## 2026-08-15 — 这个闩以前**挂得上、解不掉、也看不见**

`docs/v2/design/p0_fixes_design.md` 修复项 的族规:任何进程级闩必须有 (a) 可读状态
(带「为什么、何时」)、(b) 授权释放口、(c) 释放连带放开下游。stall-guard 三条全缺。

实测(同一个 middleware 实例,三次**互不相干**的运行,每次都是全新转录):

    run #1  同一签名失败 3 次 → 提示 1 次   ledger={sig: 1}
    run #2  同一签名失败 3 次 → 提示 1 次   ledger={sig: 2}
    run #3  同一签名失败 3 次 → **本回合当场被杀,一次提示都没有**

为什么必然如此:`_nudged` 是**每 agent 一个、进程生命期**的账本
(`chat/engine.py` 的 `_graph_for` 缓存图,`core/runtime.py` 的 `_orchestrator` /
`_bg_orchestrator` 都是记忆化的),而它**只增不减**:成功的运行不会让它衰减,
运行边界不会让它归零。于是两次历史提示会永久「上膛」,下一次同签名的偶发失败
在任何对话、任何实验、任何天数之后当场终结一个回合 —— 而停机文案还写着
「两次提示后仍在重试」,这句话在那个回合里是**假的**。

在此之前唯一的三条「释放」是:那次击杀自己(代价就是被杀的那个回合)、
`len(_nudged) > 64` 时的**整表 `clear()`**(一次连坐大赦,把别的签名真实的
升级计数一起抹掉),以及重启进程 —— 而「重启当释放」正是本设计明令钉死的被否方案
(`test_latch_release_never_requires_restart`)。

现在:账本带 (为什么/何时/哪个 agent/上膛没有) 可读,有授权释放口
(`POST /api/safety/clear-stall-guard`,汇总在 `GET /api/safety/latches`),
整表 clear 换成**按最久未见淘汰**且淘汰要留痕(大赦不许静默)。

升级提示通过 `override(messages=…)` 注入单次模型调用，不会写回 state，
下一次调用不能通过历史消息累计提示次数。账本因此是升级阶梯的持久依据；
若改变它的分桶或衰减方式，必须同时保证升级状态仍能跨调用准确累积。
"""
from __future__ import annotations

import logging
import re
import threading
import time
import weakref
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware

from mast.agents._shared.inject import append_new_human

#: 登记表条目 id。
PROMPT_ID_NUDGE = "mw.stall_guard.nudge"
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

# How many identical failures of ONE tool before we intervene, and how far back to
# look. 3 distinguishes a genuine spin from a normal retry-once-or-twice.
_THRESHOLD = 3
_WINDOW = 24
_MARKER = "⟦stall-guard⟧"

# After this many nudges the agent has demonstrated it will not stop on its own.
_MAX_NUDGES = 2

# Ledger bound. Above this the LEAST-RECENTLY-SEEN entries are dropped — this used
# to be a wholesale ``clear()``, i.e. a mass amnesty that took a genuinely escalated
# signature down with 64 unrelated ones, at a moment nobody could observe.
_LEDGER_MAX = 64

# A forced-stop message carries this phrase. Like _MARKER for a nudge, it is also
# the signal that the guard has ALREADY intervened on this spin — so a later turn
# can tell "the agent was warned" from "a fresh spin", and not re-stop an agent
# that has since complied. 
_STOP_SENTINEL = "空转保护"

# ── failure category ─────────────────────────────────────────────────────────
# A safety-gate rejection of a VALUE the agent supplied (setpoint_a=1.5 when it
# meant 1e-10) is the agent's own input error — the instrument is fine and the
# operator has nothing to fix. That is a different animal from a precondition or
# hardware failure (timeout, module-not-running, crash), where a human at the
# machine may be exactly what is needed. Telling the operator to "到机台检查" a
# typo is noise; is 25 minutes of an agent apologising for one
# because the guard kept escalating the SAME already-abandoned typo to the human.
_INPUT_ERROR_KEYS = (
    "safety_gate",
    "global_bounds_violation",
    "bounds_violation",
    "above global safety",
    "below global safety",
    "global safety maximum",
    "global safety minimum",
)


def _is_input_error(sig: str) -> bool:
    """True when the failure is the agent's own out-of-bounds INPUT (a safety-gate
    value rejection), not a hardware/precondition problem the operator can fix.

    Kept deliberately narrow: a bare "blocked" refusal that needs a human to do
    something manual (e.g. an open-loop coarse approach) is NOT an input error —
    only the safety gate's numeric-bound rejections are."""
    s = sig.lower()
    return any(k in s for k in _INPUT_ERROR_KEYS)

# ── signature normalisation ──────────────────────────────────────────────────
# A spin is "the same failure again", not "the same bytes again". Collapse the
# parts of an error message that vary between identical failures, or the counter
# never reaches the threshold and the guard is decorative.
# NB: NO trailing \b. "timed out after 3.02s" has no word boundary between the 2
# and the s, so a \b-anchored float pattern does not match it — and "timed out
# after 3.02s" vs "…3.14s" is precisely the pair that has to collapse. (My first
# draft had the \b, and the test caught it: the guard stayed decorative.)
_NOISE = (
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}\S*"), "<ts>"),
    (re.compile(r"[A-Za-z]:\\[^\s'\"]+|/[\w./-]{6,}"), "<path>"),
    (re.compile(r"0x[0-9a-f]+", re.I), "<hex>"),
    (re.compile(r"\d+\.\d+(?:[eE][-+]?\d+)?"), "<num>"),   # 3.02, 1.6e-9
    (re.compile(r"\d+"), "<n>"),                           # integers, after floats
)


def _normalise(text: str) -> str:
    out = text
    for pat, sub in _NOISE:
        out = pat.sub(sub, out)
    return out


# ── the ledger, made readable ────────────────────────────────────────────────
@dataclass
class _Rung:
    """One signature's position on the escalation ladder, **plus the provenance a
    human needs to judge it**: an entry that only says "2" forces the reader to
    invent the rest, which is the specific harm the emergency-latch write-up
    (``api/routes/safety.py``) is about."""

    escalation: int          # nudges delivered so far for this signature
    first_seen: float        # epoch seconds — when this signature first tripped
    last_seen: float         # epoch seconds — the most recent trip
    last_count: int          # how many identical failures the last trip saw
    category: str            # "input_error" | "hardware"

    @property
    def armed(self) -> bool:
        """True ⇒ the NEXT trip of this signature ends a turn on the spot, with no
        further warning to the agent. This is the state that was invisible."""
        return self.escalation >= _MAX_NUDGES


#: Every live guard, so the API can answer "what is armed right now" without the
#: agents having to hand their middleware up through six layers. Weak, so a
#: rebuilt graph's old middleware does not linger (a swapped model rebuilds the
#: graph — ``ChatEngine.invalidate``).
_LIVE: "weakref.WeakSet[StallGuardMiddleware]" = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


def latch_rows() -> "list[dict]":
    """Every stall signature currently on the ladder, newest trip first.

    Merged across guard instances: the SAME agent name can hold two independent
    ledgers in one process (the private-chat graph and the orchestrator graph are
    built separately), so a signature is reported once with the **highest**
    escalation any instance holds — that is the one that decides what happens next.

    Never raises. An empty list means nothing is on the ladder, which is a true
    statement, not a broken one."""
    merged: "dict[tuple[str, str], dict]" = {}
    with _LIVE_LOCK:
        guards = list(_LIVE)
    for g in guards:
        try:
            for row in g.ledger_rows():
                key = (row["agent"], row["signature"])
                prev = merged.get(key)
                if prev is None:
                    merged[key] = row
                    continue
                prev["instances"] = int(prev.get("instances", 1)) + 1
                if row["escalation"] > prev["escalation"]:
                    prev["escalation"] = row["escalation"]
                    prev["armed"] = row["armed"]
                if row["last_seen"] > prev["last_seen"]:
                    # 「最近一次是什么样」要跟着最近那一次走,不是跟着
                    # WeakSet 的迭代顺序(那是任意的)。
                    prev["last_count"] = row["last_count"]
                    prev["category"] = row["category"]
                prev["first_seen"] = min(prev["first_seen"], row["first_seen"])
                prev["last_seen"] = max(prev["last_seen"], row["last_seen"])
        except Exception:  # noqa: BLE001 — a read must never break the API
            continue
    return sorted(merged.values(), key=lambda r: (-r["last_seen"], r["agent"]))


def release(agent: str = "", signature: str = "", why: str = "") -> "list[dict]":
    """Operator release: drop matching ladder entries so the next spin starts over
    at the first nudge instead of an instant turn-kill.

    Empty *agent* / *signature* mean "every one". Matching on *signature* is a
    case-insensitive substring so an operator can type the tool name they saw in
    ``GET /api/safety/latches`` rather than the full normalised signature.

    Returns the rows that were released (empty = nothing matched, which is
    reported honestly rather than as a successful release)."""
    out: "list[dict]" = []
    with _LIVE_LOCK:
        guards = list(_LIVE)
    for g in guards:
        try:
            out.extend(g.release_ledger(agent=agent, signature=signature))
        except Exception:  # noqa: BLE001
            continue
    if out:
        logger.warning("stall-guard ladder released: %d entry/entries "
                       "(agent=%r signature=%r reason=%r)",
                       len(out), agent, signature, why)
        try:
            from mast.core.diagnostics import record as _rec

            _rec("stall", "release",
                 f"用户解除空转记账 {len(out)} 条" + (f"——{why}" if why else ""),
                 released=[r["signature"] for r in out], by="operator", why=why)
        except Exception:  # noqa: BLE001
            pass
    return out


def _failure_sig(m: ToolMessage) -> "str | None":
    """A stable ``(tool, normalised-error)`` signature for a FAILED tool message.

    A skill failure surfaces as ``status='error'`` (set by wrap_skill) and/or a
    summary like ``[GetBias] failed: …`` / ``[AutoApproach] precondition_failed: …``.
    Returns None for anything that is not a failure."""
    status = getattr(m, "status", None)
    text = str(getattr(m, "content", "") or "")
    is_fail = (
        status == "error"
        or "failed:" in text
        or "precondition_failed" in text
        or "rolled_back:" in text
        or "blocked" in text.lower()          # a safety/mode gate refusal is a spin too
    )
    if not is_fail:
        return None
    name = getattr(m, "name", None) or "?"
    return f"{name}::{_normalise(text)[:100]}"


def _nudge_count(msgs) -> int:
    return sum(
        1 for m in msgs
        if isinstance(m, HumanMessage) and _MARKER in str(getattr(m, "content", ""))
    )


def _last_intervention_idx(msgs) -> int:
    """Index of the last guard intervention in ``msgs`` — a nudge (HumanMessage
    carrying _MARKER) or a forced stop (AIMessage carrying _STOP_SENTINEL) — else
    -1.

    The spin count is scoped to failures produced AFTER this point, so an agent
    that was warned and then STOPPED repeating the failing call is treated as
    having complied — the guard goes quiet instead of re-firing on the stale
    failures that still sit in the sliding window ."""
    for i in range(len(msgs) - 1, -1, -1):
        c = str(getattr(msgs[i], "content", "") or "")
        if _MARKER in c or _STOP_SENTINEL in c:
            return i
    return -1


def _directive(worst: str, n: int, escalation: int) -> str:
    tool = worst.split("::", 1)[0]
    if _is_input_error(worst):
        # The agent's own bad value — the operator has nothing to fix. Do NOT send
        # it to request_user_action; tell it to correct the number or stop. Same
        # wording on both nudges (the stop path takes over after _MAX_NUDGES).
        return (
            f"{_MARKER} 你已连续 {n} 次以相同的**参数错误**被安全门拦截于 `{tool}`——"
            "这是你发送的**数值/输入越界**（例如把 100 pA 写成 1.5 A 而丢了指数），"
            "**不是硬件故障，用户无需到机台处理**。立即停止用相同参数重试：改正数值"
            "（核对单位与数量级）后再调用一次，或若不确定正确值就直接向用户说明并结束"
            "本回合。**不要**为此调用 request_user_action。"
        )
    if escalation == 1:
        return (
            f"{_MARKER} 你已连续 {n} 次以相同的错误失败于 `{tool}`，这是空转。"
            "立即停止重复调用该工具。改为三选一：(1) 调用 request_user_action 请用户"
            "到机台检查/处理这个前置条件或硬件问题；(2) 若卡住的是**该往哪走**而不是硬件"
            "（有两条都说得通的路，或该选哪个目标要用户定），调用 ask_user 当场把选择"
            "交给用户，拿到答复再继续；或 (3) 直接向用户说明这个阻塞点并结束本回合。"
            "不要再用相同参数重试同一工具。"
        )
    return (
        f"{_MARKER} 【最后一次警告】`{tool}` 仍在以相同的错误失败（已 {n} 次），"
        "而你已被提示过。**本回合到此为止**：不要再调用任何工具。"
        "现在只做一件事——用一段话向用户说明：哪个操作被阻塞、错误是什么、"
        "你判断需要人工做什么。然后结束回答。"
    )


def _prior_warnings_line(in_turn: int, carried: int) -> str:
    """Where the warnings that justify this stop actually happened.

    The old text said 「且两次提示后仍在重试」 unconditionally. When the ladder was
    climbed in EARLIER runs — which is the common case, because the ledger is
    process-lifetime and never decays — that sentence is simply false: this turn
    was killed without a single warning in it. Saying so, and saying where the
    release is, is the difference between a diagnosis and a dead end."""
    if in_turn >= _MAX_NUDGES:
        return f"本回合已提示 {in_turn} 次，仍在重试。"
    return (
        f"⚠️ **本回合并没有提示过你**（本回合提示 {in_turn} 次）——"
        f"用掉阶梯的那 {carried} 次提示发生在**更早的运行里**。空转记账按进程累计，"
        "不随运行重置，所以一个偶发的老错误也可能在这里直接终止回合。"
        "若判断这条失败是偶发的：在「安全 → 闩锁状态」（`GET /api/safety/latches`）"
        "里能看到这条记账，`POST /api/safety/clear-stall-guard` 解除后，"
        "下一次会重新从提示开始。**不需要重启。**"
    )


def _stop_message(worst: str, n: int, in_turn: int = _MAX_NUDGES,
                  carried: int = 0) -> str:
    tool = worst.split("::", 1)[0]
    err = worst.split("::", 1)[1] if "::" in worst else ""
    prior = _prior_warnings_line(in_turn, carried)
    if _is_input_error(worst):
        # A repeated parameter typo, not a hardware fault: end the turn, but do
        # NOT send the operator to the machine — the instrument is fine.
        return (
            f"⛔ 本回合被空转保护终止：`{tool}` 连续 {n} 次因**你发送的参数被安全门"
            f"拒绝**而失败。{prior}\n\n阻塞点：{err}\n\n"
            "这是**输入/参数错误，不是硬件问题**——仪器本身正常，**无需用户到机台"
            "处理**。请改正数值（核对单位与数量级，例如 100 pA = 1e-10 A）后重试，"
            "或在对话里告诉我你想要的目标值，我据此代为设置。\n"
            "（诊断记录：记录 → 诊断，按 `stall` 筛选可看到完整的失败序列。）"
        )
    return (
        f"⛔ 本回合被空转保护终止：`{tool}` 连续 {n} 次以相同错误失败。{prior}\n\n"
        f"阻塞点：{err}\n\n"
        "这通常是一个**前置条件**或**硬件状态**问题，智能体自己解决不了。"
        "请到机台检查后重试；或在对话里告诉我当前的仪器状态，我据此调整方案。\n"
        "（诊断记录：记录 → 诊断，按 `stall` 筛选可看到完整的失败序列。）"
    )


class StallGuardMiddleware(AgentMiddleware):
    """Escalate out of a repeated-identical-failure spin.

    BOTH hooks are implemented: ``wrap_model_call`` (sync — the GUI's
    ``graph.stream()``) and ``awrap_model_call`` (async — the CLI's
    ``graph.ainvoke()``). LangChain's base ``awrap_model_call`` raises
    NotImplementedError when only the sync hook exists, so the async twin is
    mandatory (mirrors ClaudePrefillGuardMiddleware).
    """

    def __init__(self, agent_name: str = "") -> None:
        super().__init__()
        self._agent = agent_name or "?"
        #: failure signature → its rung on the escalation ladder.
        #: The transcript cannot hold this (see the note in _decide): a nudge is
        #: injected into one model call and never persisted, so counting the
        #: transcript alone pins escalation at 1 forever.
        self._nudged: dict[str, _Rung] = {}
        #: The API reads this ledger from the HTTP thread while agent threads
        #: write it. Not held across anything slow — dict work only.
        self._ledger_lock = threading.Lock()
        with _LIVE_LOCK:
            _LIVE.add(self)

    # ── (a) readable / (b) releasable — the latch family contract ────────────
    def ledger_rows(self) -> "list[dict]":
        """This guard's ladder as plain dicts (JSON-safe). Read-only."""
        with self._ledger_lock:
            items = list(self._nudged.items())
        return [
            {
                "agent": self._agent,
                "signature": sig,
                "tool": sig.split("::", 1)[0],
                "escalation": r.escalation,
                "armed": r.armed,
                "first_seen": r.first_seen,
                "last_seen": r.last_seen,
                "last_count": r.last_count,
                "category": r.category,
                "instances": 1,
            }
            for sig, r in items
        ]

    def release_ledger(self, agent: str = "", signature: str = "") -> "list[dict]":
        """Drop matching rungs. Empty filters mean "all of mine"."""
        if agent and agent != self._agent:
            return []
        needle = signature.lower()
        rows = [r for r in self.ledger_rows()
                if not needle or needle in r["signature"].lower()]
        if not rows:
            return []
        with self._ledger_lock:
            for r in rows:
                self._nudged.pop(r["signature"], None)
        return rows

    def _evict_locked(self, keep: str) -> "list[str]":
        """Bound the ledger by dropping the LEAST-RECENTLY-SEEN entries.

        This was ``self._nudged.clear()`` — one unrelated signature crossing the
        64th slot wiped the escalation count of every other signature, including
        one already at the top of the ladder, silently. Caller must hold the lock.

        ARMED rungs go last: they are the ones that actually decide something, so
        64 signatures seen once must not push out the one signature that is one
        trip away from ending a turn. They are still evictable when everything is
        armed — the bound is not negotiable, the ORDER is."""
        excess = len(self._nudged) - _LEDGER_MAX
        if excess <= 0:
            return []
        victims = sorted(
            (k for k in self._nudged if k != keep),
            key=lambda k: (self._nudged[k].armed, self._nudged[k].last_seen))
        dropped = victims[:excess]
        for k in dropped:
            self._nudged.pop(k, None)
        return dropped

    def _diagnose(self, worst: str, n: int, escalation: int, stopped: bool,
                  category: str = "hardware", **extra: Any) -> None:
        try:
            from mast.core.diagnostics import record

            record("stall", f"{self._agent}:{worst.split('::', 1)[0]}",
                   f"同一错误连续失败 {n} 次" + ("——已强制结束本回合" if stopped
                                                else f"（第 {escalation} 次提示）"),
                   agent=self._agent, signature=worst, count=n,
                   escalation=escalation, stopped=stopped, category=category,
                   **extra)
        except Exception:  # noqa: BLE001
            pass

    def _decide(self, request: Any) -> "tuple[Any, str]":
        """Returns (possibly-overridden request, stop_text). A non-empty stop_text
        means: do not call the model at all — end the turn with this."""
        try:
            msgs = list(getattr(request, "messages", None) or [])
            recent = msgs[-_WINDOW:]
            sigs = Counter(
                s for s in (_failure_sig(m) for m in recent
                            if isinstance(m, ToolMessage)) if s
            )
            # Pick the worst signature that is still ACTIVE — produced at least once
            # SINCE the last guard intervention. A signature the agent was already
            # warned about and has stopped repeating is skipped, so the guard does
            # not keep re-nudging / re-stopping on the stale failures lingering in
            # the window. It also means a genuine SECOND spin (a different tool) is
            # still caught rather than masked by the abandoned one. 
            last_iv = _last_intervention_idx(recent)
            worst, n = "", 0
            for sig, cnt in sigs.most_common():
                if cnt < _THRESHOLD:
                    break
                if last_iv >= 0 and not any(
                    isinstance(m, ToolMessage) and _failure_sig(m) == sig
                    for m in recent[last_iv + 1:]
                ):
                    continue  # the agent complied with this one — look for another
                worst, n = sig, cnt
                break
            if not worst:
                return request, ""

            category = "input_error" if _is_input_error(worst) else "hardware"
            # Count from BOTH the transcript and our own ledger.
            #
            # The transcript alone does not work: a nudge is delivered by
            # override(messages=…) / request.messages, i.e. into THIS model call
            # only — it is never written back to the agent's state. On the next
            # turn request.messages is rebuilt without it, so _nudge_count()
            # reads 0 forever and escalation is pinned at 1. The guard could
            # never reach _MAX_NUDGES and never stop anything.
            #
            # Measured 2026-07-27 on a real end-to-end run: instrument_control
            # sent setpoint_a = 1.5 A (a dropped exponent for 1.5 nA), the
            # safety gate rejected it — including the words "切勿重试相同数值" —
            # and the agent retried the SAME value 11+ times. Every log line
            # read "nudge 1". The run burned its entire 80-step budget and
            # executed ZERO instrument actions.
            #
            # That is precisely the failure this module's docstring says it was
            # written to fix ("It only nudged … a model that ignored the nudge
            # spun on, unimpeded"). The escalation existed; the counter feeding
            # it was always zero.
            #
            # Scope note: the ledger is per-middleware-instance, i.e. per built
            # agent, shared across threads. Two conversations hitting the SAME
            # tool+error signature therefore share a count. Deliberate: over-
            # counting ends one turn early with a readable message, while
            # under-counting burns the whole recursion budget — which is the
            # bug being fixed.
            #
            # 2026-08-15: that trade-off stands, but it was being paid BLIND. The
            # ledger never decays, so "over-counting" is not one early turn — it
            # is every later turn, in every conversation, for the life of the
            # process, and the operator could neither see the armed count nor
            # reset it. The counts are kept apart from here on: their difference
            # is exactly 「本回合警告过没有」, and the release is
            # POST /api/safety/clear-stall-guard.
            in_turn = _nudge_count(recent)
            with self._ledger_lock:
                _rung = self._nudged.get(worst)
                carried = _rung.escalation if _rung is not None else 0
            nudges = max(in_turn, carried)

            if nudges >= _MAX_NUDGES:
                # It has been told twice and is STILL producing this failure. Stop
                # the turn with a readable conclusion — the alternative is burning
                # the whole recursion budget and ending with "任务步数达到上限",
                # which is what the operator actually got.
                logger.warning(
                    "StallGuard[%s]: STOPPING turn — %s ×%d after %d nudges "
                    "(this turn %d, carried over %d) (%s)",
                    self._agent, worst, n, nudges, in_turn, carried, category)
                self._diagnose(worst, n, nudges, stopped=True, category=category,
                               nudges_this_turn=in_turn, nudges_carried=carried)
                # Clear it: the turn ends here, and a later, genuinely new spin
                # on the same signature deserves the full nudge sequence again
                # rather than an instant stop.
                with self._ledger_lock:
                    self._nudged.pop(worst, None)
                return request, _stop_message(worst, n, in_turn=in_turn,
                                              carried=carried)

            escalation = nudges + 1
            now = time.time()
            with self._ledger_lock:
                rung = self._nudged.get(worst)
                if rung is None:
                    self._nudged[worst] = _Rung(
                        escalation=escalation, first_seen=now, last_seen=now,
                        last_count=n, category=category)
                else:
                    rung.escalation = escalation
                    rung.last_seen = now
                    rung.last_count = n
                    rung.category = category
                evicted = self._evict_locked(worst)
            if evicted:
                # Dropping a rung IS a release. A release nobody can see is how a
                # guard quietly stops guarding, so the amnesty gets a line in the
                # ledger the operator already reads (记录 → 诊断, 按 stall 筛选).
                logger.warning("StallGuard[%s]: ledger over %d — evicted %d "
                               "least-recently-seen signature(s): %s",
                               self._agent, _LEDGER_MAX, len(evicted), evicted[:3])
                self._diagnose(worst, n, escalation, stopped=False,
                               category=category, evicted=evicted)
            logger.warning("StallGuard[%s]: %s ×%d — nudge %d (%s)",
                           self._agent, worst, n, escalation, category)
            self._diagnose(worst, n, escalation, stopped=False, category=category)
            request = append_new_human(
                request, PROMPT_ID_NUDGE,
                HumanMessage(content=_directive(worst, n, escalation)))
            return request, ""
        except Exception as exc:  # noqa: BLE001 — a guard must never break a call
            logger.debug("StallGuard skipped: %s", exc)
        return request, ""

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        req, stop = self._decide(request)
        if stop:
            return AIMessage(content=stop)
        return handler(req)

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        req, stop = self._decide(request)
        if stop:
            return AIMessage(content=stop)
        return await handler(req)


__all__ = ["StallGuardMiddleware", "latch_rows", "release"]
