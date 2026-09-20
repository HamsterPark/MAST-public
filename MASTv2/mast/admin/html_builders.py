"""Admin-specific HTML generation helpers."""

from __future__ import annotations


def status_dot(active: bool) -> str:
    """Small colored dot indicating active/inactive state."""
    color = "var(--mast-status-online)" if active else "var(--mast-text-dim)"
    return (
        f'<span style="display:inline-block;width:8px;height:8px;'
        f'border-radius:50%;background:{color};margin-right:6px"></span>'
    )


def badge(text: str, color: str = "var(--mast-accent)") -> str:
    """Inline badge element."""
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'font-size:0.75rem;font-weight:600;color:#fff;background:{color}">'
        f'{text}</span>'
    )


SAFETY_COLORS = {
    "AUTO": "var(--mast-safety-auto)",
    "CONFIRM": "var(--mast-safety-confirm)",
    "DANGEROUS": "var(--mast-safety-dangerous)",
}

SEVERITY_COLORS = {
    "critical": "var(--mast-status-offline)",
    "high": "var(--mast-safety-confirm)",
    "major": "var(--mast-safety-confirm)",
    "medium": "var(--mast-accent)",
    "minor": "var(--mast-status-online)",
}

CATEGORY_COLORS = {
    "READ": "var(--mast-status-online)",
    "WRITE": "var(--mast-accent)",
    "COMPOSITE": "#a855f7",
    "ANALYSIS": "#06b6d4",
}

SOURCE_LABELS = {
    "builtin": "内置",
    "composite": "组合",
    "paper": "论文",
    "custom": "自定义",
}


def safety_badge(level: str) -> str:
    """Safety level badge."""
    level_upper = level.upper()
    return badge(level_upper, SAFETY_COLORS.get(level_upper, "var(--mast-text-dim)"))


def category_badge(cat: str) -> str:
    """Skill category badge."""
    cat_upper = cat.upper()
    return badge(cat_upper, CATEGORY_COLORS.get(cat_upper, "var(--mast-text-dim)"))


def severity_badge(sev: str) -> str:
    """Severity badge."""
    sev_lower = sev.lower()
    return badge(sev_lower, SEVERITY_COLORS.get(sev_lower, "var(--mast-text-dim)"))


def enforcement_badge(enforcement: str) -> str:
    """Enforcement type badge (hard/soft)."""
    color = "var(--mast-status-offline)" if enforcement == "hard" else "var(--mast-safety-confirm)"
    return badge(enforcement, color)


def override_indicator(has_override: bool) -> str:
    """Small indicator showing whether a value has been overridden."""
    if has_override:
        return '<span class="admin-override-dot" title="已覆盖"></span>'
    return ""


_OVERRIDE_LABELS = {
    "safety_limits.json": "安全限制",
    "safety_checks.json": "检查规则",
    "safety_constraints.json": "安全约束",
    "skill_overrides.json": "技能",
    "knowledge_overrides.json": "知识库",
    "fault_diagnosis_overrides.json": "故障诊断",
    "guidance_overrides.json": "技能指导",
    "encyclopedia_overrides.json": "百科",
}


def build_admin_header_html(override_summary: dict[str, bool] | None = None) -> str:
    """Lab Console admin header: rounded-square check logo + title + :7861 tag."""
    return (
        '<div id="admin-header" class="mast-admin-header">'
        '<svg class="lc-logo" width="18" height="18" viewBox="0 0 18 18">'
        '<rect x="1" y="1" width="16" height="16" rx="3" '
        'fill="var(--mast-accent, #0d9488)"/>'
        '<path d="M5 9.5 L7.5 12 L13 6.5" stroke="#fff" stroke-width="1.8" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
        '<span class="mast-admin-title">MAST Admin</span>'
        '<span class="mast-admin-port">:7861</span>'
        '</div>'
    )


def build_override_banner_html(override_summary: dict[str, bool]) -> str:
    """Amber banner listing currently-active overrides. Empty string when none."""
    active = [fname for fname, is_active in override_summary.items() if is_active]
    if not active:
        return ""
    count = len(active)
    labels = [_OVERRIDE_LABELS.get(f, f.removesuffix(".json")) for f in active]
    labels_str = "、".join(labels)
    return (
        '<div class="mast-warning-banner">'
        '<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" '
        'style="flex-shrink:0">'
        '<path d="M9 2 L16.5 15.5 L1.5 15.5 Z" fill="none" stroke="currentColor" '
        'stroke-width="1.4" stroke-linejoin="round"/>'
        '<line x1="9" y1="7" x2="9" y2="11" stroke="currentColor" stroke-width="1.4"/>'
        '<circle cx="9" cy="13" r="0.8" fill="currentColor"/>'
        '</svg>'
        '<div>'
        f'<div class="mast-warning-banner-title">{count} 项配置已被覆盖</div>'
        f'<div class="mast-warning-banner-detail">{labels_str}</div>'
        '</div>'
        '<span class="mast-warning-banner-tag">override active</span>'
        '</div>'
    )


# ── Shared layout helpers ─────────────────────────────────────────────


def card_style() -> str:
    """Shared inline style for clickable cards."""
    return (
        "background:var(--mast-bg-block);"
        "border:1px solid var(--mast-border);"
        "border-radius:8px;padding:14px 16px;cursor:pointer;"
        "transition:box-shadow 0.2s, transform 0.15s;"
        "box-shadow:var(--mast-shadow-card);"
    )


_BANNER_COLORS = {
    "info": ("var(--mast-text-accent)", "rgba(59,130,246,0.12)"),
    "success": ("#4ade80", "rgba(34,197,94,0.12)"),
    "warning": ("#fbbf24", "rgba(245,158,11,0.12)"),
    "error": ("#f87171", "rgba(239,68,68,0.12)"),
}


def status_banner(message: str, level: str = "info") -> str:
    """Colored status banner for feedback messages."""
    fg, bg = _BANNER_COLORS.get(level, _BANNER_COLORS["info"])
    return (
        f'<div style="padding:8px 14px;border-radius:6px;font-size:0.85rem;'
        f'background:{bg};color:{fg};border:1px solid {fg}33;margin:6px 0">'
        f'{message}</div>'
    )


def build_table_html(
    headers: list[str],
    rows: list[list[str]],
    *,
    compact: bool = False,
) -> str:
    """Build a simple HTML table."""
    pad = "4px 8px" if compact else "8px 12px"
    header_cells = "".join(
        f'<th style="text-align:left;padding:{pad};font-weight:600;'
        f'color:var(--mast-text-primary)">{h}</th>'
        for h in headers
    )
    body_rows = ""
    for row in rows:
        cells = "".join(
            f'<td style="padding:{pad};color:var(--mast-text-secondary)">'
            f'{cell}</td>'
            for cell in row
        )
        body_rows += f"<tr>{cells}</tr>"

    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.85rem">'
        f'<thead><tr style="border-bottom:2px solid var(--mast-border)">'
        f'{header_cells}</tr></thead>'
        f'<tbody>{body_rows}</tbody></table>'
    )


def build_card_grid_html(cards_html: str, min_width: str = "260px") -> str:
    """Wrap cards HTML in a responsive CSS grid."""
    return (
        f'<div style="display:grid;grid-template-columns:'
        f'repeat(auto-fill, minmax({min_width}, 1fr));gap:12px;padding:4px 0">'
        f'{cards_html}</div>'
    )


def build_breadcrumb_html(parts: list[tuple[str, str | None]]) -> str:
    """Build a breadcrumb navigation bar.

    Args:
        parts: List of (label, onclick_js_or_none). The last item has no link.
    """
    items = []
    for i, (label, onclick) in enumerate(parts):
        if onclick and i < len(parts) - 1:
            items.append(
                f'<span style="cursor:pointer;color:var(--mast-text-accent);'
                f'text-decoration:underline" onclick="{onclick}">{label}</span>'
            )
        else:
            items.append(
                f'<span style="color:var(--mast-text-primary);'
                f'font-weight:600">{label}</span>'
            )
    sep = ' <span style="color:var(--mast-text-dim);margin:0 6px">›</span> '
    return (
        f'<div style="padding:8px 0;font-size:0.85rem;margin-bottom:8px">'
        f'{sep.join(items)}</div>'
    )


def build_diff_row_html(
    field: str, old_val: str, new_val: str,
) -> str:
    """Single diff row highlighting differences."""
    changed = old_val != new_val
    ovr_color = "var(--mast-safety-confirm)" if changed else "var(--mast-text-secondary)"
    weight = "600" if changed else "400"
    style = (
        "display:grid;grid-template-columns:140px 1fr 1fr;gap:8px;"
        "padding:4px 0;border-bottom:1px solid var(--mast-border-subtle);"
        "font-size:0.82rem;"
    )
    return (
        f'<div style="{style}">'
        f'<span style="color:var(--mast-text-dim);">{field}</span>'
        f'<span style="color:var(--mast-text-secondary);">{old_val}</span>'
        f'<span style="color:{ovr_color};font-weight:{weight};">{new_val}</span>'
        f'</div>'
    )


def build_history_html(entries: list[tuple[str, str]]) -> str:
    """Build history panel HTML.

    Args:
        entries: [(timestamp_str, filename), ...]
    """
    if not entries:
        return '<p style="color:var(--mast-text-dim)">暂无历史记录</p>'

    rows = ""
    for ts, fname in entries[:20]:  # Show last 20
        rows += f"""
        <tr>
            <td style="padding:4px 8px;font-family:monospace;font-size:0.8rem">{ts}</td>
            <td style="padding:4px 8px">{fname}</td>
        </tr>"""

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
        <thead>
            <tr style="border-bottom:1px solid var(--mast-border)">
                <th style="text-align:left;padding:4px 8px">时间</th>
                <th style="text-align:left;padding:4px 8px">文件</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """
