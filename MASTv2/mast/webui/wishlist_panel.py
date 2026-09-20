"""Render helpers for the 心愿单 (Wishlist) tab.

Pure functions over the :mod:`mast.wishlist` board — no Gradio, easy to unit-test.
Two sections: user wishes (→ update server) and agent→user requests.
"""
from __future__ import annotations

import datetime as _dt
import html as _html
from typing import Any

_WISH_STATUS = {
    "queued": ("待发送", "#d97706"),
    "sent":   ("已上报", "#16a34a"),
    "failed": ("上报失败", "#dc2626"),
}
_REQ_STATUS = {
    "pending":   ("待处理", "#d97706"),
    "done":      ("已完成", "#16a34a"),
    "dismissed": ("已忽略", "#6b7280"),
}


def _fmt_ts(iso: str) -> str:
    try:
        return _dt.datetime.fromisoformat(str(iso).replace("Z", "+00:00")).strftime("%m-%d %H:%M")
    except Exception:
        return str(iso or "")[:16]


def _badge(label: str, color: str) -> str:
    return (f'<span style="font-size:11px;padding:1px 8px;border-radius:999px;'
            f'background:{color}22;color:{color};border:1px solid {color}55;">{label}</span>')


def render_wishes_html(wishes: list[dict] | None) -> str:
    wishes = wishes or []
    if not wishes:
        return ('<div style="padding:16px;color:#94a3b8;">尚无愿望/反馈。'
                '在上方提交,会自动上报到更新服务器。</div>')
    rows = []
    for w in wishes:
        label, color = _WISH_STATUS.get(w.get("status", ""), (w.get("status", "?"), "#6b7280"))
        cat = _html.escape(str(w.get("category", "")))
        txt = _html.escape(str(w.get("text", "")))
        err = w.get("error")
        err_html = (f'<div style="color:#dc2626;font-size:11px;margin-top:2px;">'
                    f'{_html.escape(str(err))}</div>') if err else ""
        rows.append(
            f'<div style="padding:8px 12px;border:1px solid #e5e7eb;border-radius:8px;'
            f'margin-bottom:6px;background:#fff;">'
            f'<div style="display:flex;gap:8px;align-items:center;font-size:11px;color:#94a3b8;">'
            f'{_badge(label, color)}<span>{cat}</span><span style="margin-left:auto;">'
            f'{_fmt_ts(w.get("created_at",""))}</span></div>'
            f'<div style="margin-top:4px;color:#111;white-space:pre-wrap;">{txt}</div>'
            f'{err_html}</div>'
        )
    return '<div>' + "".join(rows) + '</div>'


def render_requests_html(requests: list[dict] | None) -> str:
    requests = requests or []
    if not requests:
        return ('<div style="padding:16px;color:#94a3b8;">暂无来自 agent 的请求。'
                'agent 需要你配合时(如上传全文、操作硬件),会出现在这里。</div>')
    rows = []
    for r in requests:
        label, color = _REQ_STATUS.get(r.get("status", ""), (r.get("status", "?"), "#6b7280"))
        agent = _html.escape(str(r.get("agent_id", "agent")))
        kind = _html.escape(str(r.get("kind", "")))
        msg = _html.escape(str(r.get("message", "")))
        note = r.get("note")
        note_html = (f'<div style="color:#16a34a;font-size:11px;margin-top:2px;">备注: '
                     f'{_html.escape(str(note))}</div>') if note else ""
        rows.append(
            f'<div style="padding:8px 12px;border:1px solid #e5e7eb;border-radius:8px;'
            f'margin-bottom:6px;background:#fff;">'
            f'<div style="display:flex;gap:8px;align-items:center;font-size:11px;color:#94a3b8;">'
            f'{_badge(label, color)}<b style="color:#0d9488;">{agent}</b><span>{kind}</span>'
            f'<span style="margin-left:auto;">#{_html.escape(str(r.get("id","")))} · '
            f'{_fmt_ts(r.get("created_at",""))}</span></div>'
            f'<div style="margin-top:4px;color:#111;white-space:pre-wrap;">{msg}</div>'
            f'{note_html}</div>'
        )
    return '<div>' + "".join(rows) + '</div>'


def open_request_choices(requests: list[dict] | None) -> list[tuple[str, str]]:
    """(label, id) for the resolve dropdown — open (pending) requests only."""
    out: list[tuple[str, str]] = []
    for r in (requests or []):
        if r.get("status") == "pending":
            rid = str(r.get("id", ""))
            label = f"{rid} · {r.get('agent_id','agent')}: {str(r.get('message',''))[:50]}"
            out.append((label, rid))
    return out


__all__ = [
    "render_wishes_html", "render_requests_html", "open_request_choices",
]
