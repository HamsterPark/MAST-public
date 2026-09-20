"""GUI helpers for the 文献库 (literature library) tab.

Surfaces the committed literature backend as a Lab Console tab, following the
design principle **大库是真的库,其他库都是大库的指针,用户添加都进大库**:

  * library manager — create / switch / rename / delete user libraries, each a
    pure pointer-set of ``work_id``s (``knowledge/libraries.py``);
  * browse / semantic-search the BIG library (``literature_index.search`` +
    ``fetch_abstract``); search within a library = big-library search filtered
    to that library's member set (no second index);
  * ingest an uploaded PDF or fetch a DOI/URL → the abstract/metadata is
    **promoted into the big library** and the active library gains a *pointer*
    (we deliberately pass ``library_id=""`` to ``ingest_pdf`` so no per-library
    index is built — see the design clarification).

Functions are pure-ish (take ids / text, return HTML or a status string) so
``app.py`` just wires them to Gradio components. Modules (not names) are imported
so tests can monkeypatch ``literature_index.search`` / ``ingest_pdf`` etc.
Everything rendered into ``gr.HTML`` is escaped.
"""

from __future__ import annotations

import html
import logging

from mast.knowledge import fetch as fetch_mod
from mast.knowledge import fetch_board as board_mod
from mast.knowledge import ingest as ingest_mod
from mast.knowledge import libraries as lib_mod
from mast.knowledge import literature_index

logger = logging.getLogger(__name__)


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _parse_ids(text: str) -> list[str]:
    """Split a textarea / comma list of work_ids/DOIs into a clean list."""
    out: list[str] = []
    for tok in (text or "").replace(",", "\n").splitlines():
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _ids_missing_from_big(ids: list[str]) -> list[str]:
    """Of *ids* (work_id-shaped only), which are NOT in the big-library index.

    Returns [] when validation can't run (so we never warn spuriously). DOIs are
    excluded by the caller — the big index is keyed by work_id, so a DOI would
    legitimately 'not match' and warning on it would mislead.
    """
    wlike = [i for i in ids if i[:1] in ("W", "w")]
    if not wlike:
        return []
    try:
        from mast.knowledge import literature_index as _li
        _li._load_index()
        meta = _li._cache.get("metadata")
        if meta is None:
            return []
        known = set(meta["work_id"].astype(str))
    except Exception:  # pragma: no cover - can't validate → stay silent
        return []
    return [i for i in wlike if i not in known]


# ── library manager ───────────────────────────────────────────────────

def library_choices(*, registry=None) -> list[tuple[str, str]]:
    """[(label, library_id)] for a Gradio Dropdown."""
    try:
        libs = lib_mod.list_libraries(registry=registry)
    except Exception as exc:  # pragma: no cover
        logger.debug("library_choices failed: %s", exc)
        return [("global reading library", "reading")]
    return [(f'{l.get("name")} · {l.get("scope")} ({l.get("member_count", 0)})'
             + (" ✓" if l.get("is_active") else ""), l.get("library_id"))
            for l in libs]


def active_library_value(*, registry=None) -> str | None:
    """The active library id to pre-select a Dropdown to, or ``None``.

    Used to seed the 库管理 / 摄入目标 dropdowns so the user works in their
    active library by default (no re-picking every visit). Returns ``None`` if
    the active id is somehow not among the choices (so Gradio shows no stale
    selection). Note: the 大库检索 filter intentionally stays on 全大库 — see
    app.py — so search isn't silently narrowed to a small pointer-set.
    """
    try:
        aid = lib_mod.get_active(registry=registry).get("library_id")
    except Exception:  # pragma: no cover
        return None
    valid = {v for _, v in library_choices(registry=registry)}
    return aid if aid in valid else None


def render_libraries_html(*, registry=None) -> str:
    try:
        libs = lib_mod.list_libraries(registry=registry)
    except Exception as exc:  # pragma: no cover
        return f'<div style="color:#ef4444;padding:12px;">读取失败: {_esc(exc)}</div>'
    rows = []
    for l in libs:
        active = "✓ " if l.get("is_active") else ""
        scope = l.get("scope")
        lock = " 🔒" if scope == "global" else ""
        rows.append(
            '<div style="padding:6px 10px;border-bottom:1px solid rgba(148,163,184,0.15);">'
            f'<span style="font-weight:600;">{active}{_esc(l.get("name"))}{lock}</span> '
            f'<span style="color:#64748b;font-size:0.85em;">{_esc(scope)} · '
            f'{_esc(l.get("member_count", 0))} 篇 · <code>{_esc(l.get("library_id"))}</code></span>'
            '</div>'
        )
    return ('<div style="font-size:0.95em;">'
            '<div style="color:#94a3b8;padding:4px 10px;">'
            '库 = 大库的指针集合。所有论文都在大库；库只持有 work_id 引用。</div>'
            + "".join(rows) + '</div>')


def create_library_h(name: str, scope: str = "custom", *, registry=None) -> str:
    if not (name or "").strip():
        return "❌ 需要库名"
    try:
        rec = lib_mod.create_library(name.strip(), scope=scope or "custom",
                                     registry=registry)
        return f"✅ 已创建库「{rec.get('name')}」(id={rec.get('id')})"
    except Exception as exc:
        return f"❌ 创建失败: {exc}"


def switch_library_h(library_id: str, *, registry=None) -> str:
    if not library_id:
        return "❌ 选择一个库"
    try:
        rec = lib_mod.set_active_library(library_id, registry=registry)
        return f"✅ 已切换到「{rec.get('name')}」"
    except Exception as exc:
        return f"❌ 切换失败: {exc}"


def rename_library_h(library_id: str, new_name: str, *, registry=None) -> str:
    if not (library_id and (new_name or "").strip()):
        return "❌ 需要库与新名"
    try:
        rec = lib_mod.rename_library(library_id, new_name.strip(), registry=registry)
        return f"✅ 已重命名为「{rec.get('name')}」"
    except Exception as exc:
        return f"❌ 重命名失败: {exc}"


def delete_library_h(library_id: str, *, registry=None) -> str:
    if not library_id:
        return "❌ 选择一个库"
    try:
        ok = lib_mod.delete_library(library_id, registry=registry)
        return "🗑 已删除" if ok else "（未删除）"
    except Exception as exc:
        return f"❌ 删除失败: {exc}"  # global library is undeletable → LibraryError


def add_members_h(work_ids_text: str, library_id: str | None = None, *,
                  reason: str = "", registry=None) -> str:
    ids = _parse_ids(work_ids_text)
    if not ids:
        return "❌ 粘贴至少一个 work_id / DOI"
    try:
        res = lib_mod.add_members(ids, library_id=library_id or None,
                                  reason=reason, added_by="user", registry=registry)
        added = len(res.get("added", [])) if isinstance(res, dict) else len(ids)
        rej = len(res.get("rejected", [])) if isinstance(res, dict) else 0
        msg = f"✅ 加入 {added} 个指针"
        if rej:
            msg += f"（{rej} 个被拒：非法/超上限）"
        # honest signal: pointers whose work_id isn't in the big library won't
        # surface in library-filtered search (typo / not-yet-ingested paper).
        miss = _ids_missing_from_big(ids)
        if miss:
            msg += (f"；其中 {len(miss)} 个在大库中暂未找到（仍保留为指针，"
                    "可能是新论文/typo，库内检索将搜不到它们）")
        return msg
    except Exception as exc:
        return f"❌ 加入失败: {exc}"


def remove_members_h(work_ids_text: str, library_id: str | None = None, *,
                     registry=None) -> str:
    ids = _parse_ids(work_ids_text)
    if not ids:
        return "❌ 粘贴至少一个 work_id"
    try:
        res = lib_mod.remove_members(ids, library_id=library_id or None,
                                     registry=registry)
        n = (res.get("n_removed", len(res.get("removed", [])))
             if isinstance(res, dict) else len(ids))
        return f"🗑 移除 {n} 个指针"
    except Exception as exc:
        return f"❌ 移除失败: {exc}"


def render_library_members_html(library_id: str, *, registry=None) -> str:
    if not library_id:
        return '<div style="color:#94a3b8;padding:12px;">选择一个库查看成员。</div>'
    try:
        lib = lib_mod.get_library(library_id, registry=registry)
    except Exception as exc:
        return f'<div style="color:#ef4444;padding:12px;">{_esc(exc)}</div>'
    members = (lib.get("members") or []) if isinstance(lib, dict) else []
    if not members:
        return ('<div style="color:#94a3b8;padding:12px;">'
                f'「{_esc(lib.get("name"))}」暂无成员。</div>')
    rows = []
    for m in members:
        wid = m.get("work_id") if isinstance(m, dict) else m
        reason = m.get("reason", "") if isinstance(m, dict) else ""
        rows.append(
            '<div style="padding:4px 10px;border-bottom:1px solid rgba(148,163,184,0.12);">'
            f'<code>{_esc(wid)}</code>'
            + (f' <span style="color:#64748b;font-size:0.85em;">— {_esc(reason)}</span>'
               if reason else "")
            + '</div>'
        )
    return (f'<div style="font-size:0.95em;"><div style="color:#94a3b8;padding:4px 10px;">'
            f'{len(members)} 个指针 · 「{_esc(lib.get("name"))}」</div>'
            + "".join(rows) + '</div>')


# ── big-library search / browse ───────────────────────────────────────

def search_big_library_html(query: str, *, k: int = 8, material: str = "",
                            library_id: str | None = None, registry=None) -> str:
    if not (query or "").strip():
        return '<div style="color:#94a3b8;padding:12px;">输入查询语义搜索大库。</div>'
    # Compare on the CANONICAL work_id BOTH sides: members are stored bare 
    # while search results carry the URL form from metadata.parquet.
    member_set = None
    if library_id:
        try:
            lib = lib_mod.get_library(library_id, registry=registry)
            member_set = {literature_index.canonical_work_id(
                            m.get("work_id") if isinstance(m, dict) else m)
                          for m in (lib.get("members") or [])}
        except Exception:
            member_set = None
    try:
        # over-fetch when filtering so the library view still fills up
        fetch_k = max(k * 8, k + 50) if member_set is not None else k
        rows = literature_index.search(query, k=fetch_k,
                                       material=material or None)
    except Exception as exc:
        return f'<div style="color:#ef4444;padding:12px;">搜索失败: {_esc(exc)}</div>'
    if member_set is not None:
        rows = [r for r in rows
                if literature_index.canonical_work_id(r.get("work_id")) in member_set][:k]
    else:
        rows = rows[:k]
    if not rows:
        scope = "该库内" if library_id else "大库"
        return f'<div style="color:#94a3b8;padding:12px;">{scope}无匹配。</div>'
    items = []
    for r in rows:
        excerpt = r.get("abstract_excerpt") or ""
        src = r.get("source") or ""
        items.append(
            '<div style="padding:6px 10px;border-bottom:1px solid rgba(148,163,184,0.15);">'
            f'<div style="font-weight:600;">{_esc(r.get("title") or "(untitled)")}</div>'
            f'<div style="color:#64748b;font-size:0.82em;">{_esc(r.get("year") or "?")}'
            f' · {_esc(r.get("journal") or "")} · cited {_esc(r.get("cited") or 0)}'
            f' · <code>{_esc(r.get("work_id"))}</code>'
            f'{" · " + _esc(src) if src else ""}</div>'
            + (f'<div style="color:#cbd5e1;font-size:0.9em;margin-top:2px;">{_esc(excerpt)}</div>'
               if excerpt else "")
            + '</div>'
        )
    banner = ""
    if rows and rows[0].get("degraded"):
        banner = ('<div style="margin:2px 0;padding:6px 10px;border-radius:6px;'
                  'background:rgba(234,179,8,0.12);border:1px solid rgba(234,179,8,0.4);'
                  'color:#b45309;font-size:0.85em;">⚠ 语义检索暂不可用（缺 DashScope '
                  'key 或网络不通），已降级为关键词匹配，结果可能不如语义检索精准。</div>')
    return '<div style="font-size:0.95em;">' + banner + "".join(items) + '</div>'


def fetch_abstract_html(work_id: str) -> str:
    if not (work_id or "").strip():
        return '<div style="color:#94a3b8;padding:12px;">输入 work_id。</div>'
    try:
        rec = literature_index.fetch_abstract(work_id.strip())
    except Exception as exc:
        return f'<div style="color:#ef4444;padding:12px;">{_esc(exc)}</div>'
    if not rec or not rec.get("found"):
        note = rec.get("note") if isinstance(rec, dict) else ""
        return f'<div style="color:#94a3b8;padding:12px;">未找到。{_esc(note)}</div>'
    return (
        f'<div style="padding:10px;">'
        f'<div style="font-weight:600;">{_esc(rec.get("title") or work_id)}</div>'
        f'<div style="color:#64748b;font-size:0.82em;margin:2px 0;">'
        f'{_esc(rec.get("first_author") or "")} · {_esc(rec.get("year") or "?")} · '
        f'{_esc(rec.get("source") or "")}</div>'
        f'<pre style="white-space:pre-wrap;background:rgba(15,23,42,0.4);padding:8px;'
        f'border-radius:5px;font-size:0.9em;">{_esc(rec.get("abstract") or "(无摘要)")}</pre>'
        f'</div>'
    )


# ── ingest / fetch (promote into big library, library gets a pointer) ──

# ── fetch-request board (agent-asks-user HITL, design §8 / G5) ────────

def render_fetch_board_html(*, board=None) -> str:
    try:
        rows = board_mod.list_requests(board=board)
    except Exception as exc:  # pragma: no cover
        return f'<div style="color:#ef4444;padding:12px;">{_esc(exc)}</div>'
    if not rows:
        return ('<div style="color:#94a3b8;padding:12px;">'
                '取文请求板为空。文献 agent 在需要某篇论文全文（大库只有摘要）时'
                '会在此提交请求，你上传 PDF 或取文即可满足。</div>')
    icon = {"pending": "🟡", "fulfilled": "✅", "failed": "🔴", "dismissed": "⚪"}
    items = []
    for r in rows:
        ic = icon.get(r.get("status"), "•")
        t = f' «{_esc(r.get("title"))}»' if r.get("title") else ""
        reason = f' — {_esc(r.get("reason"))}' if r.get("reason") else ""
        note = (f'<div style="color:#64748b;font-size:0.82em;">{_esc(r.get("note"))}</div>'
                if r.get("note") else "")
        items.append(
            '<div style="padding:6px 10px;border-bottom:1px solid rgba(148,163,184,0.15);">'
            f'<span>{ic} <code>{_esc(r.get("work_id"))}</code>{t} '
            f'<span style="color:#64748b;font-size:0.82em;">[{_esc(r.get("status"))} · '
            f'{_esc(r.get("request_id"))} · {_esc(r.get("requested_by"))}]</span>{reason}</span>'
            f'{note}</div>'
        )
    pend = board_mod.pending_count(board=board)
    return (f'<div style="font-size:0.95em;"><div style="color:#94a3b8;padding:4px 10px;">'
            f'{len(rows)} 条 · {pend} 待处理</div>' + "".join(items) + '</div>')


def render_fetch_badge_html(*, board=None) -> str:
    """A small banner that surfaces when the literature agent has pending
    full-text requests, so they appear even with the board accordion collapsed.

    Returns an empty string when there are none (the banner visually vanishes),
    so it's safe to refresh on a timer. Read-only + a cheap local JSON read →
    fine to call from a gr.Timer tick (never blocks the queue worker).
    """
    try:
        n = board_mod.pending_count(board=board)
    except Exception:  # pragma: no cover
        return ""
    if not n:
        return ""
    return (
        '<div style="margin:2px 0;padding:8px 12px;border-radius:8px;'
        'background:rgba(234,179,8,0.12);border:1px solid rgba(234,179,8,0.45);'
        'color:#b45309;font-size:0.92em;">'
        f'📥 文献 agent 有 <b>{int(n)}</b> 条全文请求待你应答 —— 见下方'
        '「取文请求板」,上传 PDF / 取文即可满足。</div>'
    )


def _dashscope_key_present() -> bool:
    try:
        from mast.knowledge import literature_index as _li
        return bool(_li._load_dashscope_key())
    except Exception:
        return False


def ingest_readiness_html() -> str:
    """Pre-flight hint for the 摄取 accordion:摄取 needs a DashScope embed key +
    PyMuPDF. Surface the requirement BEFORE the user spends an upload, instead of
    only failing afterwards with '❌ 摄取失败: No DashScope key'."""
    import importlib.util
    has_fitz = importlib.util.find_spec("fitz") is not None
    has_key = _dashscope_key_present()
    if has_key and has_fitz:
        return ('<div style="padding:4px 10px;color:#16a34a;font-size:0.82em;">'
                '✅ 摄取就绪（DashScope 嵌入 key + PyMuPDF 均可用）。</div>')
    missing = []
    if not has_key:
        missing.append("DashScope 嵌入 key（设 DASHSCOPE_API_KEY 或写 api key/dashscope.env）")
    if not has_fitz:
        missing.append("PyMuPDF（pip install pymupdf）")
    return ('<div style="padding:6px 10px;border-radius:6px;'
            'background:rgba(220,38,38,0.10);border:1px solid rgba(220,38,38,0.4);'
            'color:#b91c1c;font-size:0.82em;">⚠ 摄取当前会失败，缺少：'
            + "；".join(missing) + '。上传前请先补齐。</div>')


def fetch_request_choices(*, board=None) -> list[tuple[str, str]]:
    """[(label, request_id)] of PENDING requests for a Dropdown."""
    try:
        rows = board_mod.list_requests("pending", board=board)
    except Exception:  # pragma: no cover
        return []
    return [(f'{r.get("work_id")}'
             + (f' «{r.get("title")}»' if r.get("title") else "")
             + f' ({r.get("request_id")})', r.get("request_id")) for r in rows]


def request_work_id(request_id: str, *, board=None) -> str:
    """The work_id for a request_id (so the GUI can pre-fill ingest/fetch)."""
    if not request_id:
        return ""
    try:
        r = board_mod.get_board().get_request(request_id) if board is None \
            else board.get_request(request_id)
    except Exception:  # pragma: no cover
        r = None
    return (r or {}).get("work_id", "")


def dismiss_request_h(request_id: str, *, board=None) -> str:
    if not request_id:
        return "❌ 选择一条请求"
    try:
        out = board_mod.resolve(request_id, "dismissed", note="operator dismissed",
                                board=board)
    except Exception as exc:
        return f"❌ 失败: {exc}"
    if out.get("error"):
        return f"❌ {out['error']}"
    return f"⚪ 已忽略请求 {request_id}"


def ingest_pdf_h(pdf_path: str, library_id: str | None = None, *,
                 work_id: str = "", embedder=None, extractor=None,
                 registry=None, board=None) -> str:
    """Ingest an uploaded PDF: promote into the big library, then add a POINTER
    to the chosen library. ``library_id=""`` is passed to ingest_pdf on purpose
    so NO per-library index is built (pointer model). If ``work_id`` is given
    (fulfilling a board request) it is used; any OPEN fetch-request for the
    resulting work_id is marked fulfilled (closing the agent-asks-user loop)."""
    if not (pdf_path or "").strip():
        return "❌ 先上传一个 PDF"
    try:
        res = ingest_mod.ingest_pdf(pdf_path, work_id=(work_id or "").strip(),
                                    library_id="", promote=True,
                                    source="user_pdf", embedder=embedder,
                                    extractor=extractor)
    except Exception as exc:
        return f"❌ 摄取失败: {exc}"
    status = getattr(res, "status", None) or (res.get("status") if isinstance(res, dict) else "")
    wid = getattr(res, "work_id", None) or (res.get("work_id") if isinstance(res, dict) else "")
    if status and str(status).startswith("error"):
        return f"❌ 摄取失败: {status}"
    if status and str(status).startswith("warning"):
        detail = (getattr(res, "detail", "")
                  or (res.get("detail") if isinstance(res, dict) else ""))
        return f"⚠ {detail or status}"
    if not wid:
        return f"⚠ 摄取完成但无 work_id（status={status}）"
    # add the pointer into the active/chosen library
    ptr = "（未加入库）"
    try:
        lib_mod.add_members([wid], library_id=library_id or None,
                            reason="ingested full text", added_by="user",
                            registry=registry)
        ptr = "并已作为指针加入库"
    except Exception as exc:  # pragma: no cover
        logger.debug("pointer add after ingest failed: %s", exc)
    # close any open fetch-request for this paper (notify the agent)
    closed = ""
    try:
        n = board_mod.resolve_work_id(wid, "fulfilled",
                                      note="全文已摄取进大库", board=board)
        if n:
            closed = f"，并满足了 {n} 条取文请求"
    except Exception as exc:  # pragma: no cover
        logger.debug("resolve_work_id after ingest failed: %s", exc)
    return f"✅ 已促进进大库 (work_id={wid}) {ptr}{closed}。"


def fetch_then_ingest_h(doi_or_url: str, library_id: str | None = None, *,
                        work_id: str = "", oa_url: str = "", client=None,
                        robots_fetcher=None, embedder=None, extractor=None,
                        registry=None, board=None) -> str:
    """Experimental: best-effort, ToS-respecting fetch of a DOI/URL, then ingest
    the retrieved PDF (if any) via :func:`ingest_pdf_h`."""
    if not (doi_or_url or "").strip():
        return "❌ 粘贴一个 DOI 或 URL"
    try:
        # auto_oa=True → resolve an OA PDF URL from the DOI (OpenAlex/Unpaywall)
        # up-front so a bare DOI has a real open-access target to try.
        res = fetch_mod.try_fetch_fulltext(doi_or_url.strip(), oa_url=oa_url,
                                           auto_oa=True, client=client,
                                           robots_fetcher=robots_fetcher)
    except Exception as exc:
        return f"❌ 取文失败: {exc}"
    status = res.get("status") if isinstance(res, dict) else ""
    pdf_path = res.get("pdf_path") if isinstance(res, dict) else ""
    if not pdf_path:
        # honest about why — paywall / robots / not OA. The module sets
        # 'message' (not 'reason'); read that so the real explanation shows.
        reason = (res.get("message") or res.get("reason") or status
                  or "无可合法获取的全文")
        return f"⚠ 未取到全文：{_esc(reason)}（尊重 robots/ToS，不绕付费墙）"
    msg = ingest_pdf_h(pdf_path, library_id, work_id=work_id, embedder=embedder,
                       extractor=extractor, registry=registry, board=board)
    return f"📥 已取到全文 → {msg}"


__all__ = [
    "library_choices", "active_library_value", "render_libraries_html",
    "create_library_h", "switch_library_h", "rename_library_h",
    "delete_library_h", "add_members_h",
    "remove_members_h", "render_library_members_html", "search_big_library_html",
    "fetch_abstract_html", "ingest_pdf_h", "fetch_then_ingest_h",
    "render_fetch_board_html", "render_fetch_badge_html", "fetch_request_choices",
    "request_work_id", "dismiss_request_h", "ingest_readiness_html",
]
