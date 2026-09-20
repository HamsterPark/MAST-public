"""What happens *after* a paper's full text lands on disk.

One paper can arrive four ways — an operator uploads a PDF, an operator types
metadata by hand, an operator points the server at a path, or the literature
agent fetches an open-access copy itself — and all four owe the same tail:

  1. file a **pointer** into the right library (the big index is the one real
     library; everything else points into it);
  2. record the **full-text back-reference** on that member;
  3. close any OPEN fetch-board request for the work;
  4. tell whoever asked that their paper arrived (``core.fetch_resume``).

That tail used to live in the API route layer (``routes/literature_cognition``),
which was fine while only routes needed it. The literature agent needs it too,
and ``agents → api`` is backwards — so it moved here, next to the registry,
board and ingest code it was calling into all along. The route keeps a thin
wrapper for response-shape compatibility.

Everything is best-effort and **never raises**: by the time this runs the bytes
are already safely on disk, and a curation hiccup must not be reported to the
caller as a failed ingest.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["curate_ingested_fulltext"]


def _fulltext_is_readable(slug: str) -> bool:
    """Does ``<papers>/<slug>/fulltext.txt`` actually hold text?

    This is the same file the reading tools read, so it is the only honest answer
    to "can the agent read this paper now". A scanned PDF with no text layer and
    no OCR key ingests fine — the bytes are stored, the abstract row is written —
    and yields nothing readable. Telling the agent its full text has arrived in
    that state sends it hunting through every reading tool for text that is not
    there, until the recursion limit stops it (observed live, 2026-08-01).
    """
    if not slug:
        return False
    try:
        from mast.knowledge.paths import papers_dir
        f = papers_dir() / slug / "fulltext.txt"
        return f.is_file() and bool(f.read_text(encoding="utf-8",
                                                errors="replace").strip())
    except Exception:  # noqa: BLE001 — unknown means "do not promise readable"
        return False


def _origin_turn_conversation() -> str:
    """The conversation this call is running inside, or ``""``.

    Used to EXCLUDE that conversation from the resume notification: when the
    agent fetches a paper itself, its own conversation is plainly awake and does
    not need waking. Other conversations waiting on the same paper still do.
    """
    try:
        from mast.core.turn_context import current_turn
        return str((current_turn() or {}).get("conversation_id") or "")
    except Exception:  # noqa: BLE001 — no turn context is normal (routes, tests)
        return ""


def curate_ingested_fulltext(
    work_id: str,
    library_id: str = "",
    *,
    source: str,
    reason: str,
    slug: str = "",
    added_by: str = "user",
) -> dict[str, Any]:
    """File a freshly-ingested paper and close/notify anything waiting on it.

    Args:
        work_id:    the paper's id (``W…`` or ``local:…``). Empty → no-op.
        library_id: explicit target library; empty resolves to the **effective**
                    library (the active experiment's own, else the manual
                    pointer), or to the libraries of whoever is waiting on the
                    board (see below).
        source:     provenance for the member row (``user_pdf`` / ``user_manual``
                    / ``agent``).
        reason:     why this paper was added, stored on the member row.
        slug:       ``<papers>/<slug>/`` directory name; empty means there is no
                    PDF (manual entry) so no full-text ref is recorded.
        added_by:   who filed it — ``"user"`` for operator paths, ``"agent"``
                    when the literature agent fetched it itself.

    Returns ``{pointer_added, pointer_library_id, pointer_library_source,
    fulltext_ref, fulfilled_requests, resumed}``.

    Three filing rules worth knowing (all predate this move, 2026-07-29):

    * an empty ``library_id`` resolves to the **effective** library instead of a
      single global "active library" that two parallel experiments trampled;
    * the target is pinned to any **frozen** ``experiment_id`` on the board
      request. A request raised during experiment A can be fulfilled days later
      while B is active, and filing it under B is the wrong drawer with nothing
      afterwards to reveal the mistake. Several experiments waiting on the same
      paper each get their own pointer;
    * the full text is recorded on the member (``fulltext_status`` /
      ``fulltext_ref``) — schema fields nothing had ever written.
    """
    out: dict[str, Any] = {
        "pointer_added": False, "pointer_library_id": "",
        "pointer_library_source": "", "fulltext_ref": "",
        "fulfilled_requests": 0, "resumed": 0, "fulltext_readable": False,
    }
    if not work_id:
        return out
    out["fulltext_readable"] = _fulltext_is_readable(slug)

    # Whoever is waiting on the board decides where this lands, and gets told
    # about it afterwards. Both reads must happen BEFORE resolving: resolving
    # closes the rows, and a closed row is no longer "open" to be found.
    waiting: list[str] = []
    pending: list[dict] = []
    try:
        from mast.knowledge import fetch_board as board_mod
        waiting = board_mod.open_experiment_ids(work_id)
        pending = board_mod.open_requests(work_id)
    except Exception as exc:  # pragma: no cover - non-fatal
        logger.info("fetch-board lookup for %s failed: %s", work_id, exc)

    targets: list[tuple[str, str]] = []  # (library_id, source)
    try:
        from mast.knowledge import experiment_library as expl
        from mast.knowledge import libraries as lib_mod

        if library_id:
            targets.append((library_id, "request"))
        elif waiting:
            for eid in waiting:
                lid, src = expl.resolve_effective_library(experiment_id=eid or None)
                if lid and (lid, src) not in targets:
                    targets.append((lid, src))
        if not targets:
            lid, src = expl.resolve_effective_library()
            if lid:
                targets.append((lid, src))

        for lid, src in targets:
            try:
                res = lib_mod.add_members(
                    [work_id], library_id=lid,
                    reason=reason, source=source, added_by=added_by,
                )
            except Exception as exc:  # pragma: no cover - per-target, keep going
                logger.info("pointer add into %s failed (%s): %s", lid, work_id, exc)
                continue
            out["pointer_added"] = True
            if not out["pointer_library_id"]:
                out["pointer_library_id"] = (
                    res.get("library_id", lid) if isinstance(res, dict) else lid)
                out["pointer_library_source"] = src
            if slug:
                try:
                    from mast.knowledge.ingest import record_fulltext
                    ref = record_fulltext(lid, work_id, slug)
                    out["fulltext_ref"] = out["fulltext_ref"] or ref
                except Exception as exc:  # pragma: no cover - non-fatal
                    logger.info("full-text ref for %s failed: %s", work_id, exc)
    except Exception as exc:  # pragma: no cover - non-fatal
        logger.info("pointer curation after ingest failed (%s): %s", work_id, exc)

    # The board note is what the operator reads back, so it must not claim a
    # readable full text when all that landed was an unreadable scan.
    note = ("全文/条目已入库" if out["fulltext_readable"] or not slug
            else "PDF 已入库，但没有可读文本层（扫描件未 OCR）")
    try:
        from mast.knowledge import fetch_board as board_mod
        out["fulfilled_requests"] = int(
            board_mod.resolve_work_id(work_id, "fulfilled", note=note) or 0)
    except Exception as exc:  # pragma: no cover - non-fatal
        logger.info("resolve_work_id after ingest failed (%s): %s", work_id, exc)

    # Wake whoever asked. Gated on having actually closed something: two
    # concurrent uploads of the same paper both reach here, but only the one
    # whose resolve_work_id found open rows may claim to have fulfilled them —
    # otherwise the second upload wakes the conversation a second time with
    # nothing new to say.
    if out["fulfilled_requests"] > 0 and pending:
        try:
            from mast.core import fetch_resume
            for rec in pending:
                rec["fulltext_readable"] = bool(out["fulltext_readable"] or not slug)
            res = fetch_resume.notify_fulfilled(
                work_id, pending,
                exclude_conversation_id=_origin_turn_conversation()) or {}
            out["resumed"] = int(res.get("resumed", 0) or 0)
        except Exception as exc:  # pragma: no cover - notification is never fatal
            logger.info("fetch-resume notify for %s failed: %s", work_id, exc)
    return out
