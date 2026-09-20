"""Experiment log viewer for MAST GUI."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mast.webui.app import MASTApp


def get_active_experiment_status(app: MASTApp) -> tuple[str, str]:
    """Return (experiment_status, sample_status) for the current experiment.

    Both are plain text strings consumed by format_experiment_status_html().
    """
    if app._experiment_log is None:
        return ("Logging unavailable", "")

    if app._experiment_log.current_experiment_id:
        eid = app._experiment_log.current_experiment_id
        exp_text = f"Active: {eid[:8]}..."
        if app._storage is not None:
            try:
                exp_info = app._storage.get_experiment(eid)
                if exp_info:
                    exp_text = f"Active: {exp_info.get('name', '')} ({eid[:8]}...)"
            except Exception:
                pass

        sample_text = ""
        if app._experiment_log.current_sample_id:
            sid = app._experiment_log.current_sample_id
            if app._storage is not None:
                try:
                    sample_info = app._storage.get_sample(sid)
                    if sample_info:
                        sample_text = f"Sample: {sample_info.get('name', '')} ({sid[:8]}...)"
                except Exception:
                    pass
            if not sample_text:
                sample_text = f"Sample: {sid[:8]}..."
        return (exp_text, sample_text)

    return ("No active experiment", "")


# (Removed 2026-05-31, finding #122: dead Markdown builders
#  get_experiment_list / get_experiment_timeline / get_action_detail —
#  never wired into build_ui; the live path uses build_experiment_*_html +
#  _build_action_timeline, all HTML-escaped. The removed Markdown builders
#  interpolated operator-supplied experiment names / params / context
#  unescaped, an injection surface that would have re-activated the moment
#  anything called them.)


# ── History viewer functions ──────────────────────────────────────────


def get_experiment_choices(app: MASTApp) -> list[tuple[str, str]]:
    """Return dropdown choices [(label, id)] for experiments."""
    if app._storage is None:
        return []
    try:
        experiments = app._storage.list_experiments(limit=50)
    except Exception:
        return []

    active_id = None
    if app._experiment_log and app._experiment_log.current_experiment_id:
        active_id = app._experiment_log.current_experiment_id

    choices = []
    for exp in experiments:
        name = exp.get("name", "Unnamed")
        ts = exp.get("start_time", "")[:16]
        status = exp.get("status", "")
        marker = " [ACTIVE]" if exp.get("id") == active_id else ""
        label = f"{name} ({ts}) [{status}]{marker}"
        choices.append((label, exp["id"]))
    return choices


def get_sample_choices(app: MASTApp, experiment_id: str) -> list[tuple[str, str]]:
    """Return dropdown choices [(label, id)] for samples in an experiment."""
    if app._storage is None or not experiment_id:
        return []
    try:
        samples = app._storage.get_samples(experiment_id)
    except Exception:
        return []

    choices = []
    for s in samples:
        name = s.get("name", "Unnamed")
        status = s.get("status", "")
        ts = s.get("start_time", "")[:16]
        label = f"{name} ({ts}) [{status}]"
        choices.append((label, s["id"]))
    return choices


def build_breadcrumb(
    exp_id: str | None = None,
    exp_name: str | None = None,
    sample_id: str | None = None,
    sample_name: str | None = None,
) -> str:
    """Build breadcrumb HTML for history navigation."""
    parts = ['<div class="history-breadcrumb">']

    if exp_id is None:
        parts.append('<span class="history-breadcrumb-current">All Experiments</span>')
    else:
        parts.append('<span class="history-breadcrumb-link">All Experiments</span>')
        parts.append('<span class="history-breadcrumb-sep"> / </span>')
        esc_name = html.escape(exp_name or exp_id[:8])
        if sample_id is None:
            parts.append(f'<span class="history-breadcrumb-current">{esc_name}</span>')
        else:
            parts.append(f'<span class="history-breadcrumb-link">{esc_name}</span>')
            parts.append('<span class="history-breadcrumb-sep"> / </span>')
            esc_sample = html.escape(sample_name or sample_id[:8])
            parts.append(f'<span class="history-breadcrumb-current">{esc_sample}</span>')

    parts.append('</div>')
    return ''.join(parts)


def build_experiment_overview_html(app: MASTApp) -> str:
    """Build compact line-by-line list of all experiments."""
    if app._storage is None:
        return _empty_state("Storage Unavailable", "Experiment storage is not initialized.")

    try:
        experiments = app._storage.list_experiments(limit=50)
    except Exception as exc:
        return _empty_state("Error", f"Could not load experiments: {exc}")

    if not experiments:
        return _empty_state(
            "No Experiments",
            "No experiments have been recorded yet. Start an experiment from the chat to begin.",
        )

    active_id = None
    if app._experiment_log and app._experiment_log.current_experiment_id:
        active_id = app._experiment_log.current_experiment_id

    rows = []
    for exp in experiments:
        eid = exp["id"]
        name = html.escape(exp.get("name", "Unnamed"))
        status = exp.get("status", "unknown")
        if eid == active_id:
            status = "running"
        ts = exp.get("start_time", "")[:16]

        sample_count = 0
        action_count = 0
        try:
            sample_count = len(app._storage.get_samples(eid))
            action_count = len(app._storage.get_actions(eid))
        except Exception:
            pass

        badge = f'<span class="log-badge log-status-{status}">{status}</span>'
        stats = f'<span class="log-dim">{sample_count} samples &middot; {action_count} actions</span>'
        rows.append(
            f'<div class="log-row">'
            f'<span class="log-ts">{ts}</span>'
            f'{badge}'
            f'<span class="log-name">{name}</span>'
            f'{stats}'
            f'</div>'
        )

    return f'<div class="log-list">{"".join(rows)}</div>'


def build_experiment_detail_html(app: MASTApp, experiment_id: str) -> str:
    """Build HTML for experiment detail: header + sample card grid."""
    if app._storage is None:
        return _empty_state("Storage Unavailable", "")

    try:
        exp = app._storage.get_experiment(experiment_id)
    except Exception:
        exp = None

    if not exp:
        return _empty_state("Not Found", f"Experiment {experiment_id[:8]}... not found.")

    name = html.escape(exp.get("name", "Unnamed"))
    status = exp.get("status", "unknown")
    goal = html.escape(exp.get("goal_text", ""))
    ts = exp.get("start_time", "")[:16]
    end_ts = exp.get("end_time", "")
    end_part = f" &middot; Ended: {end_ts[:16]}" if end_ts else ""
    notes = html.escape(exp.get("notes", ""))

    samples: list[dict] = []
    actions: list = []
    try:
        samples = app._storage.get_samples(experiment_id)
        actions = app._storage.get_actions(experiment_id)
    except Exception:
        pass

    total_time = sum(a.duration_s for a in actions)
    succeeded = sum(1 for a in actions if a.result and a.result.success)
    failed = len(actions) - succeeded

    goal_part = f' &middot; {goal}' if goal else ""

    header = (
        f'<div class="log-header">'
        f'<span class="log-name" style="font-size:1.1em;">{name}</span>'
        f'<span class="log-badge log-status-{status}">{status}</span>'
        f'<span class="log-dim">{ts}{end_part}{goal_part}</span>'
        f'</div>'
    )

    summary = (
        f'<div class="log-summary">'
        f'{len(samples)} samples &middot; {len(actions)} actions '
        f'({succeeded} ok, {failed} failed) &middot; {total_time:.1f}s'
        f'</div>'
    )

    # Sample rows
    if not samples:
        sample_list = '<div class="log-dim" style="padding:8px 0;">No samples.</div>'
    else:
        sample_rows = []
        for s in samples:
            sid = s["id"]
            sname = html.escape(s.get("name", "Unnamed"))
            sstatus = s.get("status", "unknown")
            sts = s.get("start_time", "")[:16]
            try:
                sa_count = len(app._storage.get_actions(experiment_id, sample_id=sid))
            except Exception:
                sa_count = 0
            badge = f'<span class="log-badge log-status-{sstatus}">{sstatus}</span>'
            sample_rows.append(
                f'<div class="log-row">'
                f'<span class="log-ts">{sts}</span>'
                f'{badge}'
                f'<span class="log-name">{sname}</span>'
                f'<span class="log-dim">{sa_count} actions</span>'
                f'</div>'
            )
        sample_list = f'<div class="log-list">{"".join(sample_rows)}</div>'

    # Orphan actions (no sample)
    orphan_actions = [a for a in actions if not a.sample_id]
    orphan_html = ""
    if orphan_actions:
        orphan_html = (
            f'<div class="log-dim" style="padding:6px 0 2px;">'
            f'Unassigned ({len(orphan_actions)})</div>'
            + _build_action_timeline(orphan_actions)
        )

    return header + summary + sample_list + orphan_html


def build_sample_detail_html(
    app: MASTApp, experiment_id: str, sample_id: str,
) -> str:
    """Build HTML for sample detail: header + action timeline."""
    if app._storage is None:
        return _empty_state("Storage Unavailable", "")

    try:
        sample = app._storage.get_sample(sample_id)
    except Exception:
        sample = None

    if not sample:
        return _empty_state("Not Found", f"Sample {sample_id[:8]}... not found.")

    name = html.escape(sample.get("name", "Unnamed"))
    status = sample.get("status", "unknown")
    desc = html.escape(sample.get("description", ""))
    ts = sample.get("start_time", "")[:16]
    end_ts = sample.get("end_time", "")
    end_part = f" &middot; Ended: {end_ts[:16]}" if end_ts else ""

    try:
        actions = app._storage.get_actions(experiment_id, sample_id=sample_id)
    except Exception:
        actions = []

    total_time = sum(a.duration_s for a in actions)
    succeeded = sum(1 for a in actions if a.result and a.result.success)
    failed = len(actions) - succeeded

    desc_part = f' &middot; {desc}' if desc else ""

    header = (
        f'<div class="log-header">'
        f'<span class="log-name">{name}</span>'
        f'<span class="log-badge log-status-{status}">{status}</span>'
        f'<span class="log-dim">{ts}{end_part}{desc_part}</span>'
        f'</div>'
    )

    summary = (
        f'<div class="log-summary">'
        f'{len(actions)} actions ({succeeded} ok, {failed} failed) '
        f'&middot; {total_time:.1f}s'
        f'</div>'
    )

    if not actions:
        timeline = '<div class="log-dim" style="padding:8px 0;">No actions.</div>'
    else:
        timeline = _build_action_timeline(actions)

    return header + summary + timeline


def _format_params_inline(params: dict) -> str:
    """Format parameters as inline pseudocode arguments."""
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            if abs(v) < 1e-6 and v != 0:
                parts.append(f"{k}={v:.2e}")
            elif abs(v) >= 1000:
                parts.append(f"{k}={v:.1f}")
            else:
                parts.append(f"{k}={v:g}")
        elif isinstance(v, str):
            parts.append(f'{k}="{v}"')
        elif isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _format_result_inline(result) -> str:
    """Format result data as compact inline string."""
    if result is None or not result.data:
        return ""
    parts = []
    for k, v in result.data.items():
        if k in ("raw",):
            continue
        if isinstance(v, float):
            if abs(v) < 1e-6 and v != 0:
                parts.append(f"{k}={v:.2e}")
            else:
                parts.append(f"{k}={v:g}")
        elif isinstance(v, (list, dict)):
            continue  # skip large nested data
        else:
            s = str(v)
            if len(s) > 30:
                continue
            parts.append(f"{k}={s}")
    return ", ".join(parts[:5])  # max 5 fields


def _build_action_timeline(actions: list) -> str:
    """Build compact line-by-line action log in pseudocode format.

    Each action is one line:  HH:MM:SS  [ok] SkillName(param=val, ...) -> result  0.05s
    Click to expand details.
    """
    rows = []
    for a in actions:
        success = a.result.success if a.result else False
        error_msg = a.result.error if a.result else ""
        icon = "\u2713" if success else "\u2717"
        css = "log-ok" if success else "log-err"
        ts = a.timestamp[11:19] if a.timestamp and len(a.timestamp) >= 19 else "?"

        skill = html.escape(a.skill_name)
        params_str = html.escape(_format_params_inline(a.parameters))
        result_str = html.escape(_format_result_inline(a.result))
        arrow = f' <span class="log-arrow">\u2192</span> <span class="log-result">{result_str}</span>' if result_str else ""
        err_str = f' <span class="log-error">{html.escape(error_msg[:60])}</span>' if error_msg else ""
        dur = f'<span class="log-dur">{a.duration_s:.2f}s</span>'

        # Expandable detail
        detail_parts: list[str] = []
        if a.nanonis_calls:
            for call in a.nanonis_calls:
                m = html.escape(call.method)
                ca = html.escape(str(call.args))
                ce = f' <span class="log-error">ERR: {html.escape(call.error)}</span>' if call.error else ""
                detail_parts.append(f'<div class="log-tcp"><code>{m}({ca})</code>{ce}</div>')
        if a.state_before:
            sb = a.state_before
            detail_parts.append(
                f'<div class="log-state">before: V={sb.bias_v}, I={sb.current_a}, Z={sb.z_pos_m}</div>'
            )
        if a.state_after:
            sa = a.state_after
            detail_parts.append(
                f'<div class="log-state">after:  V={sa.bias_v}, I={sa.current_a}, Z={sa.z_pos_m}</div>'
            )
        if a.context:
            detail_parts.append(f'<div class="log-ctx">{html.escape(a.context)}</div>')

        detail_html = ""
        has_detail = bool(detail_parts)
        uid = f"lt-{abs(hash(a.id)) % 99999}"
        if has_detail:
            detail_html = (
                f'<div class="log-detail">'
                + ''.join(detail_parts)
                + '</div>'
            )

        rows.append(
            f'<div class="log-action {css}">'
            f'<input type="checkbox" id="{uid}" class="log-toggle">'
            f'<label for="{uid}" class="log-action-line">'
            f'<span class="log-expand">{"+" if has_detail else " "}</span>'
            f'<span class="log-ts">{ts}</span>'
            f'<span class="log-icon">{icon}</span>'
            f'<code class="log-call">{skill}({params_str})</code>'
            f'{arrow}{err_str}'
            f'{dur}'
            f'</label>'
            f'{detail_html}'
            f'</div>'
        )

    return f'<div class="log-list">{"".join(rows)}</div>'


def _empty_state(title: str, desc: str) -> str:
    """Return empty state HTML."""
    return (
        '<div class="mast-empty-state">'
        f'<div class="mast-empty-title">{html.escape(title)}</div>'
        f'<div class="mast-empty-desc">{html.escape(desc)}</div>'
        '</div>'
    )
