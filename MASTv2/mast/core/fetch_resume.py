"""Waking the conversation that asked for something, once the answer arrives.

The literature agent can ask the operator for a full text it cannot get itself
(``request_fulltext`` → the fetch board). The operator may satisfy that request
minutes or days later. Until now nothing connected the two ends: the board row
flipped to ``fulfilled`` and the agent found out only if someone happened to
send it another message and it happened to poll the board. "I'll continue once
you upload it" was, in practice, an empty promise.

This module is the seam. The fulfilment path (``knowledge.fulfilment``) calls
:func:`notify_fulfilled`; the runtime registers the thing that actually knows how
to drive a conversation. Deliberately kept as a registry rather than a direct
call because the layering only works in one direction: ``knowledge`` may not
import ``chat``/``runtime``, but both may import this.

The 心愿单 board has the same problem in a different costume, so it uses the same
seam via :func:`notify_request_answered`. An agent that asked "please go change
the sample" and stopped had no turn on which to notice the answer either; the
board's auto-injection only pays out if some later turn happens to occur, and for
a stopped agent none does. Two boards, one reason to wake somebody up.

Three properties everything here depends on:

* **No mast imports.** Standard library only, so any layer can import it without
  a cycle and importing it can never drag the runtime into a test process.
* **No resumer registered → no-op.** The standalone API process, a bare test
  run, and a runtime whose chat engine failed to build all take this path, and
  all of them should simply close the board row like before.
* **Never raises.** A notification failing must not turn a completed ingest into
  a reported error — the paper is already on disk either way.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "set_resumer", "clear_resumer", "has_resumer",
    "notify_fulfilled", "notify_request_answered",
    "build_resume_instruction", "build_answer_instruction",
]

#: ``(work_id, requests, *, exclude_conversation_id) -> dict``. Process-wide, so
#: tests must clear it (see the autouse fixture in test_fetch_resume.py) — a
#: leaked resumer would let one test drive another test's fake engine.
_RESUMER: Callable[..., dict] | None = None
_LOCK = threading.Lock()


def set_resumer(fn: Callable[..., dict] | None) -> None:
    """Register the callable that resumes conversations. ``None`` clears."""
    global _RESUMER
    with _LOCK:
        _RESUMER = fn


def clear_resumer() -> None:
    set_resumer(None)


def has_resumer() -> bool:
    with _LOCK:
        return _RESUMER is not None


def notify_fulfilled(work_id: str, requests: list[dict[str, Any]], *,
                     exclude_conversation_id: str = "") -> dict[str, Any]:
    """Tell the runtime that ``work_id``'s full text is now available.

    Args:
        work_id:  the paper that arrived.
        requests: the board rows as they were **before** being resolved. They
                  must be captured first — once resolved they are no longer open
                  and the list of who was waiting is gone. Each row carries
                  ``origin_conversation_id`` (who asked), ``reason`` (what they
                  wanted it for) and ``request_id`` / ``title`` for the message.
        exclude_conversation_id: a conversation that must NOT be resumed —
                  normally the one this call is running inside, which is already
                  awake and about to use the paper itself.

    Returns ``{"resumed": n, "detail": ...}``; never raises.
    """
    return _dispatch(requests, kind="fetch", work_id=work_id,
                     exclude_conversation_id=exclude_conversation_id)


def notify_request_answered(requests: list[dict[str, Any]], *,
                            exclude_conversation_id: str = "") -> dict[str, Any]:
    """Tell the runtime that the operator answered 心愿单 requests.

    ``requests`` are the answered rows, each carrying ``origin_conversation_id``
    (who asked), ``message`` (what was asked) and the operator's ``path`` /
    ``note``. Same contract as :func:`notify_fulfilled`: never raises, no-op when
    nothing is registered.
    """
    return _dispatch(requests, kind="request", work_id="",
                     exclude_conversation_id=exclude_conversation_id)


def _dispatch(requests: list[dict[str, Any]], *, kind: str, work_id: str,
              exclude_conversation_id: str) -> dict[str, Any]:
    with _LOCK:
        fn = _RESUMER
    if fn is None:
        return {"resumed": 0, "detail": "no resumer registered"}
    if not requests:
        return {"resumed": 0, "detail": "no pending requests"}
    try:
        res = fn(work_id, requests, kind=kind,
                 exclude_conversation_id=exclude_conversation_id) or {}
    except Exception as exc:  # noqa: BLE001 — a resume attempt is never fatal
        logger.warning("resumer failed (%s, %s): %s", kind, work_id or "-", exc)
        return {"resumed": 0, "detail": f"resumer failed: {exc}"}
    if not isinstance(res, dict):
        return {"resumed": 0, "detail": "resumer returned a non-dict"}
    return res


def build_resume_instruction(requests: list[dict[str, Any]]) -> str:
    """The message handed to the agent when its papers arrive.

    Aggregates every request belonging to one conversation into a single turn —
    three papers arriving together is one "carry on", not three.

    The closing line is deliberate. Being handed a paper is permission to finish
    the thing that was blocked on it, not an invitation to start a fresh survey;
    without that sentence a resumed turn tends to expand into exactly the kind of
    unrequested literature review the agent's prompt spends its length forbidding.

    Rows carrying ``fulltext_readable=False`` get the opposite message. A scanned
    PDF with no text layer ingests successfully and is readable by nobody; told
    that "the full text has arrived", the agent works through every reading tool
    hunting for text that is not there and only stops at the recursion limit
    (observed on a live run, 2026-08-01). Saying so up front costs one sentence.
    """
    rows = list(requests or [])
    readable = [r for r in rows if r.get("fulltext_readable", True)]
    unreadable = [r for r in rows if not r.get("fulltext_readable", True)]

    def _row(r: dict) -> str:
        rid = str(r.get("request_id") or "").strip()
        wid = str(r.get("work_id") or "").strip()
        title = str(r.get("title") or "").strip()
        reason = str(r.get("reason") or "").strip()
        bits = [b for b in (rid, f"work_id={wid}" if wid else "") if b]
        head = "  - " + "  ".join(bits) if bits else "  - （无标识）"
        if title:
            head += f"  «{title}»"
        if reason:
            head += f"（当时的理由：{reason}）"
        return head

    lines: list[str] = ["【取文请求已满足 · 自动继续】"]
    if readable:
        lines.append("以下论文的全文已入库：")
        lines += [_row(r) for r in readable]
        lines.append(
            "现在可以用 read_paper_section / extract_protocol / search_papers 读它的全文，"
            "需要逐篇精读时用 deep_read_papers。")
    if unreadable:
        lines.append("以下论文的 PDF 已入库，但**没有可读的文本层**（扫描件且未 OCR）：")
        lines += [_row(r) for r in unreadable]
        lines.append(
            "这几篇你**读不到正文** —— 不要反复换工具去试。如实告诉请求："
            "文件收到了但无法提取文字，需要可检索的 PDF 或配置 OCR。")
    lines += [
        "请继续当时因为缺全文而搁置的那件事，完成后正常收尾（该存报告就存报告）。",
        "如果那件事已经做完或不再需要，说明一句即可 —— 不要就此展开新的调研。",
    ]
    return "\n".join(lines)


def build_answer_instruction(requests: list[dict[str, Any]]) -> str:
    """The message handed to the agent when the operator answers its requests.

    Mirrors :func:`build_resume_instruction`. The operator's ``path`` is repeated
    verbatim on its own line because that is the whole point of storing it as a
    field: the agent should not have to parse a filename out of prose.
    """
    lines = ["【用户已答复你的请求 · 自动继续】"]
    for r in requests or []:
        rid = str(r.get("id") or "").strip()
        status = "已完成" if str(r.get("status")) == "done" else "已忽略"
        msg = str(r.get("message") or "").strip()
        head = f"  - [{rid}] {status}：{msg}" if rid else f"  - {status}：{msg}"
        lines.append(head)
        path = str(r.get("path") or "").strip()
        if path:
            lines.append(f"      → 用户给的路径：{path}")
        note = str(r.get("note") or "").strip()
        if note:
            lines.append(f"      → 备注：{note}")
    lines += [
        "请继续当时因为等这个答复而搁置的那件事，完成后正常收尾。",
        "被「已忽略」的那几条就不要再做了，也不要重复发起同一请求。",
        "如果那件事已经做完或不再需要，说明一句即可 —— 不要就此展开新的工作。",
    ]
    return "\n".join(lines)
