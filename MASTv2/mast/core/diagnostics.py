"""The refusal ledger — why nothing happened.

MAST records what SUCCEEDED. It does not record what was REFUSED, and that is the
gap the 2026-07-10 field trial fell into:

  * 「进针功能调用失败」 — which layer refused it? A precondition? The safety
    gate? The mode gate? The abort gate? The operator had a failure and no cause,
    and neither did we: a block produces a ToolMessage to the model and vanishes.
  * 「在做STS第五点的时候停下来了」 — was step 5 executed and failed? Skipped
    as already-done from a stale sidecar? Aborted? All three look identical from
    the outside, and nothing wrote down which it was.
  * 「任务步数达到上限——可能某个智能体在预条件或安全门上反复失败而空转」 —
    the operator DIAGNOSED THIS THEMSELVES from the symptom, because the system
    could not.

So: one always-on ledger for every refusal and every skip. Not a trajectory (those
only exist while a run is being recorded, and a refusal outside one was dropped on
the floor) — a process-level ring buffer that also appends to a JSONL on disk, so
the record survives the run that produced it and can be read after the fact.

Contract, in the order that matters:

  * **It must never break the thing it is observing.** Every entry point swallows
    everything. A diagnostics failure is a lost line, never a failed skill.
  * **It must never block.** Bounded ring buffer; the disk append is best-effort
    and backs off (then retries) rather than blocking or dying.
  * **It must never hold anything unserialisable.** Values are coerced to JSON
    primitives and capped — no tensors, no ndarrays, no sockets, no Nanonis
    clients (the same invariant the checkpointer has).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Literal

logger = logging.getLogger(__name__)

# What kind of "nothing happened" this was. The point of an enum here is that a
# reader can ask "show me everything that REFUSED something" without knowing every
# layer's name.
Kind = Literal[
    "precondition_block",   # a skill's stated precondition was not met 
    "safety_block",         # SafetyGate refused (global bounds / Layer-0 / rules)
    "mode_block",           # the operating-mode gate refused (safe/semi)
    "abort_block",          # an abort is latched — instrument writes refused
    "comms_down",           # Nanonis TCP link judged down — circuit breaker OPEN
    "tip_crash",            # tip crash / repeated same-region crash escalation
    "module_missing",       # a required Nanonis module was not running (preflight)
    "hitl_reject",          # the operator refused it
    "step_skip",            # a composite step was SKIPPED (resume / already done)
    "step_fail",            # a composite step ran and failed
    "step_abort",           # a composite stopped because the operator aborted
    "stall",                # the same failure, over and over 
    # 本来会**打断工作**的东西,现在只通知。缓冲区关键事件不再弹
    # 确认框、DANGEROUS 技能不再等审批 —— 那条链路被割掉之后,这一行就是它留下的
    # 全部痕迹。与 ``note`` 分开是因为它必须**可查**:验收这条改动时该问的
    # 正是「本来会拦我几次」,而那是一个 kinds=("notice_only",) 的查询。
    "notice_only",
    # An exception that KILLED a run. Distinct from the *_block kinds above:
    # those are the system correctly saying no, this is the system falling over.
    # Added 2026-07-28 — a run died three times on "IndexError: list index out
    # of range" and neither the log nor 诊断 held a single frame of it .
    "run_error",
    # agent 自己铸了一个组合技能(技能工坊,2026-08-25)。与 ``note`` 分开是因为
    # 验收这条能力时该问的正是「它今天造了什么」,而那是一个
    # kinds=("skill_forged",) 的查询 —— 混在 note 里就要靠文案 grep。
    "skill_forged",
    # 目标终止判据求了一次值(2026-08-27)。与 ``note`` 分开的理由同上:用户
    # 验收这条改动时问的是「它为什么停了 / 为什么没停」,而那是一个
    # kinds=("goal_verdict",) 的查询 —— 混在 note 里就要靠文案 grep。
    # 三种决定各留一行:done 直接结束 / 判据没满足却想结束(转人问)/ 判不了
    # (照旧由模型说了算,但这件事本身要留痕)。
    "goal_verdict",
    "note",                 # anything else worth a breadcrumb
]

_MAX_ENTRIES = 2000            # ring buffer; a long run must not grow without bound
_MAX_STR = 400                 # per-field cap — an error blob must not become the log
_JSONL_MAX_BYTES = 8_000_000   # roll the file rather than fill the disk

_lock = threading.Lock()
_entries: Deque[dict] = deque(maxlen=_MAX_ENTRIES)
_run_id = ""
_seq = 0

# Disk-log health. The first draft of this flipped a ``_disk_ok`` flag to False on
# the FIRST write failure and never tried again — i.e. one transient hiccup (a
# locked file, a momentary permission blip) silently cost the process every future
# diagnostics line. That is the exact silent-failure shape this whole module exists
# to expose; writing another one into it would be a bad joke. So: back off, then
# retry.
_disk_fails = 0
_disk_retry_at = 0.0
_DISK_BACKOFF_MAX_S = 60.0


# ── coercion ─────────────────────────────────────────────────────────────────

def _safe(v: Any) -> Any:
    """JSON primitives only, capped. A diagnostics line must never carry a tensor,
    an open socket, or a 2 MB traceback."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v if len(v) <= _MAX_STR else v[:_MAX_STR] + "…"
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in list(v)[:20]]
    if isinstance(v, dict):
        return {str(k)[:64]: _safe(x) for k, x in list(v.items())[:20]}
    return _safe(repr(v))


def _dir() -> Path:
    from mast._runtime_paths import project_root

    return project_root() / "artifacts" / "diagnostics"


def _append_disk(entry: dict) -> None:
    """Best-effort durability. A read-only disk must not make every skill call pay
    for a failed open — but it must not silence the ledger for good either."""
    global _disk_fails, _disk_retry_at
    if _disk_fails and time.monotonic() < _disk_retry_at:
        return
    try:
        d = _dir()
        d.mkdir(parents=True, exist_ok=True)
        f = d / "refusals.jsonl"
        if f.exists() and f.stat().st_size > _JSONL_MAX_BYTES:
            f.replace(d / "refusals.1.jsonl")
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if _disk_fails:
            logger.info("diagnostics: disk log recovered")
            _disk_fails = 0
    except Exception as exc:  # noqa: BLE001
        _disk_fails += 1
        _disk_retry_at = time.monotonic() + min(
            _DISK_BACKOFF_MAX_S, 2.0 ** min(_disk_fails, 6))
        if _disk_fails == 1:
            logger.warning("diagnostics: disk log failing, backing off (%s)", exc)


# ── the one entry point ──────────────────────────────────────────────────────

def record(kind: Kind, subject: str, reason: str, **fields: Any) -> None:
    """Write one refusal / skip. Never raises, never blocks.

    Args:
        kind:    which layer said no (see :data:`Kind`)
        subject: what was refused — a skill name, a step id, a tool name
        reason:  the operator-facing WHY, in the words the layer already uses
        fields:  anything that makes the line actionable — args, the value that
                 failed a bound, the step ordinal, the run id
    """
    global _seq
    try:
        entry = {
            "t": time.time(),
            "kind": str(kind),
            "subject": str(subject)[:120],
            "reason": str(reason)[:_MAX_STR],
            "run_id": _run_id,
            **{k: _safe(v) for k, v in fields.items()},
        }
        with _lock:
            _seq += 1
            entry["seq"] = _seq
            _entries.append(entry)
        _append_disk(entry)
        logger.info("[diag:%s] %s — %s", kind, subject, str(reason)[:160])
    except Exception:  # noqa: BLE001 — a lost diagnostics line is never worth a crash
        pass


def set_run_id(run_id: str) -> None:
    """Stamp subsequent entries with the run they belong to, so a post-mortem can
    isolate ONE task out of a day's worth of refusals."""
    global _run_id
    _run_id = str(run_id or "")


# ── reading ──────────────────────────────────────────────────────────────────

def recent(
    n: int = 200,
    *,
    kinds: "tuple[str, ...] | None" = None,
    subject: str = "",
    run_id: str = "",
) -> list[dict]:
    """The newest entries first. Filters are AND-ed; empty means no filter."""
    with _lock:
        rows = list(_entries)
    if kinds:
        rows = [r for r in rows if r.get("kind") in kinds]
    if subject:
        s = subject.lower()
        rows = [r for r in rows if s in str(r.get("subject", "")).lower()]
    if run_id:
        rows = [r for r in rows if r.get("run_id") == run_id]
    rows.reverse()
    return rows[: max(1, int(n))]


def summary() -> dict:
    """Counts per kind + per subject — the shape of "what keeps getting refused".

    This is the view that answers #31 without anyone having to read a log: one
    subject dominating the ``precondition_block`` count IS the spin."""
    with _lock:
        rows = list(_entries)
    by_kind: dict[str, int] = {}
    by_subject: dict[str, int] = {}
    for r in rows:
        by_kind[r.get("kind", "?")] = by_kind.get(r.get("kind", "?"), 0) + 1
        key = f"{r.get('kind')}:{r.get('subject')}"
        by_subject[key] = by_subject.get(key, 0) + 1
    top = sorted(by_subject.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "total": len(rows),
        "by_kind": by_kind,
        "top_refusals": [{"what": k, "count": v} for k, v in top],
        "log_path": str(_dir() / "refusals.jsonl") if not _disk_fails else "",
    }


def clear() -> None:
    """Tests only."""
    global _seq, _disk_fails, _disk_retry_at
    with _lock:
        _entries.clear()
        _seq = 0
    _disk_fails = 0
    _disk_retry_at = 0.0


__all__ = ["record", "recent", "summary", "clear", "set_run_id", "Kind"]
