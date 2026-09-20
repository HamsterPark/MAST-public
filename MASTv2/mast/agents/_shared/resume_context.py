"""Cross-session resume context — what to hand the supervisor when the operator
says "继续", plus operator answers that came back on the 心愿单 board.

Two field failures this closes (2026-07 conversation analysis ⑥/⑦):

* **⑥ cross-session amnesia.** A new group conversation / a fresh supervisor
  thread starts with an EMPTY message stream. So when the operator opens a new
  chat and types "继续", the supervisor sees only that one word, answers
  「No prior context or ongoing task found」 and re-introduces itself — while a
  half-finished NiI2/Au(111) experiment sits `running` in the record. Worse, the
  supervisor then had no anchor and start_experiment was (before the idempotency
  fix) non-idempotent, so "继续" work spawned brand-new duplicate experiments.
  :func:`build_experiment_resume_block` reconstructs a compact description of the
  most-recent UNFINISHED experiment (id / name / goal / current sample / recent
  actions / active plan phase) so the supervisor resumes it instead of starting
  over.

* **⑦ one-way 心愿单 requests.** ``request_user_action`` posts a request to the
  board and the agent is expected to poll ``check_my_requests`` later — but a
  BLOCKED agent that stopped has no turn on which to poll, and the operator's
  answer sat on the board unread (the analysis: agent 「无记忆 / 无匹配」, forcing
  the operator to paste the path into chat). :func:`build_request_reply_block`
  surfaces answered-but-undelivered requests so the NEXT run injects them
  automatically — no polling required.

Pure + best-effort: every helper degrades to ``None`` / ``""`` on any error. A
resume-context glitch must never break a run, and none of these functions touch
hardware, the network, or an LLM.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Short continuation markers. Presence of any (case-folded substring) marks the
# operator's message as "pick up where we left off" rather than a fresh task.
# Deliberately conservative: these almost always mean continue-prior-work, so a
# normal specific instruction ("扫描 Au(111) 5nm") does not false-positive.
_RESUME_MARKERS: tuple[str, ...] = (
    "继续", "繼續", "接着", "接著", "接下来", "接下去", "接著做", "接着做",
    "上次做到", "继续之前", "继续上次", "接着上", "接著上", "往下做", "往下走",
    "resume", "continue", "go on", "keep going", "carry on",
    "pick up where", "where were we", "last time we", "as before",
)


def is_resume_intent(text: str) -> bool:
    """True if *text* reads as "continue the prior work" rather than a new task.

    Case-insensitive substring match against a small set of strong continuation
    markers (see ``_RESUME_MARKERS``). Best-effort — never raises."""
    try:
        s = (text or "").strip().casefold()
    except Exception:  # noqa: BLE001
        return False
    if not s:
        return False
    return any(m in s for m in _RESUME_MARKERS)


def _short(s, n: int = 220) -> str:
    try:
        t = " ".join(str(s or "").split())
    except Exception:  # noqa: BLE001
        return ""
    return t if len(t) <= n else t[:n] + "…"


def _resolve_unfinished_experiment(experiment_log, storage):
    """The experiment "继续" should pick up, or None.

    REVISED 2026-07-28. This used to look for ``status == 'running'`` rows.
    Experiments no longer have a lifecycle state — they are permanent and always
    continuable (「有的实验可能过了十年重启」) — so "still running" is no longer a
    meaningful question. The question that replaced it is **"which one is
    current?"**, and that is answered by the single persistent scope pointer.

    Resolution order:
      1. the live session's current experiment (in-memory pointer);
      2. the persisted ``active_scope`` pointer (a fresh log in a new session);
      3. nothing — an operator who explicitly cleared the scope gets no resume
         block, which is exactly what clearing it means.

    Deliberately NO fallback to "the newest row on disk": that would resurrect an
    experiment the operator has deliberately stepped away from.
    """
    if storage is None:
        return None
    cur = getattr(experiment_log, "current_experiment_id", None) if experiment_log else None
    if not cur:
        try:
            cur = (storage.get_active_scope() or {}).get("experiment_id")
        except Exception:  # noqa: BLE001
            cur = None
    if not cur:
        return None
    try:
        return storage.get_experiment(cur)
    except Exception:  # noqa: BLE001
        return None


def build_experiment_resume_block(
    experiment_log,
    storage,
    plan_store=None,
    *,
    max_actions: int = 6,
) -> str | None:
    """Compact markdown describing the most-recent unfinished experiment, or None.

    Injected into the supervisor's context so "继续" resumes the real experiment
    instead of dead-ending on 「No prior context」. Read-only; never mutates the
    log or creates anything."""
    try:
        exp = _resolve_unfinished_experiment(experiment_log, storage)
        if exp is None or storage is None:
            return None
        eid = str(exp.get("id", "") or "")
        lines = ["## 恢复上下文：当前实验（供你判断如何「继续」）"]
        last_active = str(exp.get("last_active_at") or exp.get("start_time") or "")[:19]
        lines.append(
            f"- 实验：{exp.get('name', '(未命名)')} "
            f"（id {eid[:8]}…，开始 {str(exp.get('start_time', '') or '')[:19]}，"
            f"上次活动 {last_active}）"
        )
        goal = (exp.get("goal_text") or "").strip()
        if goal:
            lines.append(f"- 目标：{_short(goal, 240)}")

        # 当前样品。用 last-active 而不是 status=='active'：样品没有终态，
        # 被旧代码 end 成 'completed' 的样品照样可以被切回来继续用。
        try:
            samp = (storage.get_last_active_sample(eid)
                    if hasattr(storage, "get_last_active_sample")
                    else storage.get_active_sample(eid))
        except Exception:  # noqa: BLE001
            samp = None
        if samp:
            st = (samp.get("sample_type") or "").strip()
            lines.append(
                f"- 当前样品：{samp.get('name', '')}"
                f"{(' (' + st + ')') if st else ''}"
            )
        else:
            lines.append("- 当前样品：无活动样品")

        # 当前针尖。仪器级、跨实验 —— 但「继续」的第一件事往往是判断上次停在哪、
        # 针尖还是不是那一根。lazy import:core→agents 是既有方向的反向。
        try:
            from mast.core.tip_state import current_tip_facts
            tf = current_tip_facts()
        except Exception:  # noqa: BLE001
            tf = None
        if tf:
            bits = [x for x in (tf.get("material"), tf.get("fabrication"),
                                tf.get("form")) if x]
            since = str(tf.get("installed_at") or "")[:10]
            lines.append(
                f"- 当前针尖：{tf.get('name') or '未命名'}"
                f"{(' (' + '/'.join(bits) + ')') if bits else ''}"
                f"{('，' + since + ' 装入') if since else ''}"
            )
        else:
            lines.append("- 当前针尖：未登记")

        # Recent actions (skill names only — the block must stay compact).
        try:
            actions = storage.get_actions(eid)
        except Exception:  # noqa: BLE001
            actions = []
        if actions:
            recent = actions[-max_actions:]
            names = ", ".join(getattr(a, "skill_name", "?") or "?" for a in recent)
            lines.append(
                f"- 已记录 {len(actions)} 个动作；最近：{_short(names, 240)}"
            )
        else:
            lines.append("- 尚无已记录的动作")

        # Active plan phase (断点续跑 anchor), if a plan store is available.
        if plan_store is not None:
            try:
                plan = plan_store.get_active()
            except Exception:  # noqa: BLE001
                plan = None
            if plan is not None:
                try:
                    phases = list(getattr(plan, "phases", []) or [])
                    idx = int(getattr(plan, "current_phase_idx", 0) or 0)
                    cur = phases[idx] if 0 <= idx < len(phases) else None
                    status = getattr(plan, "status", "")
                    status = getattr(status, "value", status)
                    lines.append(
                        f"- 活动计划：{getattr(plan, 'name', '') or '(未命名)'} — "
                        f"当前阶段 {idx + 1}/{len(phases)} "
                        f"{getattr(cur, 'name', '') if cur else ''}（{status}）"
                    )
                except Exception:  # noqa: BLE001
                    pass

        lines.append(
            "提示：若用户意在继续该实验，请据此恢复——路由到相应 agent 续做未完成的"
            "步骤，不要重新自我介绍、也不要从头新建实验/样品（新建会造成重复记录）。"
        )
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 — resume context never breaks a run
        logger.debug("build_experiment_resume_block failed: %s", exc)
        return None


def _within_window(resolved_at: str, within_hours: float | None) -> bool:
    """True if *resolved_at* (ISO) is within *within_hours* of now, or if the
    bound/timestamp is missing/unparseable (fail-open — better to surface a
    reply than to silently drop it)."""
    if not within_hours or not resolved_at:
        return True
    try:
        ts = datetime.fromisoformat(resolved_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        return age_h <= float(within_hours)
    except Exception:  # noqa: BLE001
        return True


def build_request_reply_block(
    board,
    *,
    agent_id: str = "",
    within_hours: float | None = 72.0,
    max_items: int = 6,
) -> tuple[str | None, list[str]]:
    """Markdown of operator answers the agent has NOT read yet + the ids to mark
    delivered, or ``(None, [])``.

    ⑦: surfaces answered-but-undelivered 心愿单 requests so a run injects them
    automatically. ``agent_id=""`` (the default) matches every agent — in
    production all agents post as the shared id "agent", and over-surfacing an
    answer is far better than a blocked agent never seeing it. The caller marks
    the returned ids delivered so each answer is handed over exactly once."""
    try:
        rows = board.resolved_requests_for(agent_id, undelivered_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("resolved_requests_for failed: %s", exc)
        return None, []
    rows = [r for r in (rows or []) if _within_window(r.get("resolved_at", ""), within_hours)]
    if not rows:
        return None, []
    lines = ["## 用户已答复你之前发起的请求（自动回程 · 据此继续，勿重复发起）"]
    ids: list[str] = []
    for r in rows[:max_items]:
        rid = str(r.get("id", "") or "")
        status = "已完成" if r.get("status") == "done" else "已忽略"
        line = f"- [{rid}] {status}：{_short(r.get('message', ''), 180)}"
        path = (r.get("path") or "").strip()
        if path:
            line += f"\n    → 用户提供的路径：{path}"
        note = (r.get("note") or "").strip()
        if note:
            line += f"\n    → 备注：{_short(note, 200)}"
        lines.append(line)
        if rid:
            ids.append(rid)
    return "\n".join(lines), ids


def build_fetch_arrival_block(
    board,
    *,
    requested_by: str = "literature",
    within_hours: float | None = 72.0,
    max_items: int = 6,
) -> tuple[str | None, list[str]]:
    """Markdown of papers that arrived without anyone telling the agent, + their ids.

    The fetch board's primary route is auto-resume: the conversation that asked
    is driven one more turn the moment the paper lands. That cannot always fire —
    a request raised inside a background run has no conversation to return to,
    the chat engine may not be up, the operator may have switched resume off — and
    in those cases the arrival used to sit on the board exactly as wishlist answers
    used to, waiting for an agent to remember to poll ``list_fetch_requests``.

    This is the sweep-up, mirroring :func:`build_request_reply_block`: whatever
    announced an arrival marks it, and whatever was never announced gets handed
    over on the agent's next turn. The caller marks the returned ids announced.
    """
    try:
        rows = board.unannounced_fulfilled(requested_by)
    except Exception as exc:  # noqa: BLE001
        logger.debug("unannounced_fulfilled failed: %s", exc)
        return None, []
    rows = [r for r in (rows or [])
            if _within_window(r.get("resolved_at", ""), within_hours)]
    if not rows:
        return None, []
    lines = ["## 你要的论文全文已经到了（自动回程 · 据此继续，勿重复发起取文请求）"]
    ids: list[str] = []
    readable = True
    for r in rows[:max_items]:
        rid = str(r.get("request_id", "") or "")
        wid = str(r.get("work_id", "") or "")
        title = str(r.get("title", "") or "").strip()
        reason = str(r.get("reason", "") or "").strip()
        line = f"- [{rid}] work_id={wid}"
        if title:
            line += f"  «{_short(title, 80)}»"
        if reason:
            line += f"\n    → 当时要它是为了：{_short(reason, 160)}"
        note = str(r.get("note", "") or "")
        if "没有可读文本层" in note:
            readable = False
            line += "\n    → 注意：这份 PDF **没有可读文本层**（扫描件未 OCR），读不出正文。"
        lines.append(line)
        if rid:
            ids.append(rid)
    lines.append(
        "可以用 read_paper_section / extract_protocol / search_fulltext 读它的全文。"
        if readable else
        "读不到正文的那几篇不要反复换工具去试 —— 如实告诉用户需要可检索的 PDF。")
    return "\n".join(lines), ids


def build_pending_reply_hint(
    board,
    *,
    within_hours: float | None = 72.0,
    max_items: int = 6,
) -> str | None:
    """A SHORT, READ-ONLY note that operator answers are waiting, for the
    SUPERVISOR's routing context — so it routes to the agent that will consume
    them. Does NOT mark anything delivered (read-only), so it never races the
    agent-side ``RequestReplyReadbackMiddleware`` that does the actual delivery.

    (⑦: the full reply content — path / note — is injected into the agent by the
    middleware; the supervisor only needs to know a reply exists so it dispatches
    the right agent rather than ending the run.)"""
    try:
        rows = board.resolved_requests_for("", undelivered_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("resolved_requests_for failed: %s", exc)
        return None
    rows = [r for r in (rows or []) if _within_window(r.get("resolved_at", ""), within_hours)]
    if not rows:
        return None
    lines = [f"## 提示：用户已答复 {len(rows)} 条智能体请求（待相应 agent 读取）"]
    for r in rows[:max_items]:
        rid = str(r.get("id", "") or "")
        status = "已完成" if r.get("status") == "done" else "已忽略"
        lines.append(f"- [{rid}] {status}：{_short(r.get('message', ''), 120)}")
    lines.append(
        "请路由到发起这些请求的 agent，让它读取并据此继续"
        "（它会在下一步自动拿到用户给的路径/备注，无需再让用户复述）。"
    )
    return "\n".join(lines)


__all__ = [
    "is_resume_intent",
    "build_experiment_resume_block",
    "build_request_reply_block",
    "build_fetch_arrival_block",
    "build_pending_reply_hint",
]
