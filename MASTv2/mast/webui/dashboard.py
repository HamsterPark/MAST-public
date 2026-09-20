"""System check dashboard for MAST GUI."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_system_check(app: Any) -> list[dict]:
    """Run system self-check on startup. Returns list of check results.

    Each result: {"name": str, "status": "ok"|"warning"|"error"|"unavailable", "detail": str}

    Checks:
    1. Nanonis TCP connections (4 ports)
    2. Nanonis version/license
    3. Data storage (SQLite writable)
    4. Claude API (if key configured)
    5. Environment sensors (all registered)
    """
    results: list[dict] = []

    # 1. Nanonis TCP connections
    if app._pool is not None:
        port_names = {
            "main": app.config.nanonis.port_main,
            "monitor": app.config.nanonis.port_monitor,
            "data": app.config.nanonis.port_data,
            "emergency": app.config.nanonis.port_emergency,
        }
        for role, port in port_names.items():
            try:
                app._pool.get(role)
                results.append({
                    "name": f"Nanonis {role} (port {port})",
                    "status": "ok",
                    "detail": f"Connected on port {port}",
                })
            except Exception:
                results.append({
                    "name": f"Nanonis {role} (port {port})",
                    "status": "unavailable",
                    "detail": f"Not connected — port {port}",
                })
    else:
        results.append({
            "name": "Nanonis TCP connections",
            "status": "error",
            "detail": "ConnectionPool not initialized",
        })

    # 2. Nanonis version/license
    try:
        # The real TCP call is Util_VersionGet — the old probe looked for a
        # non-existent `Version` method, so this row ALWAYS said "Cannot query
        # version" even when connected. Use the pool's
        # safe_call and parse the returned Product Line + Version strings.
        rec = app._pool.safe_call("Util_VersionGet", role="main") if app._pool else None
        detail = None
        if rec is not None and not getattr(rec, "error", ""):
            parsed = getattr(rec, "return_value", None)
            vals = parsed[2] if (isinstance(parsed, (list, tuple)) and len(parsed) > 2) else None
            strings = [v for v in (vals or []) if isinstance(v, str) and v.strip()]
            if strings:
                detail = " · ".join(strings[:2])  # Product Line · Version
        if detail:
            results.append({"name": "Nanonis version", "status": "ok", "detail": detail})
        else:
            results.append({
                "name": "Nanonis version",
                "status": "unavailable",
                "detail": "Cannot query version (not connected)",
            })
    except Exception as exc:
        results.append({
            "name": "Nanonis version",
            "status": "unavailable",
            "detail": f"Cannot query version: {exc}",
        })

    # 3. Data storage
    if app._storage is not None:
        try:
            # Try a lightweight read to confirm DB is writable
            app._storage.list_experiments(limit=1)
            results.append({
                "name": "Data storage (SQLite)",
                "status": "ok",
                "detail": f"Database at {app.config.db_path}",
            })
        except Exception as exc:
            results.append({
                "name": "Data storage (SQLite)",
                "status": "error",
                "detail": f"Database error: {exc}",
            })
    else:
        results.append({
            "name": "Data storage (SQLite)",
            "status": "error",
            "detail": "ExperimentStorage not initialized",
        })

    # 4. Claude API
    api_key = app.config.llm.api_key
    if api_key:
        if app._planner is not None:
            results.append({
                "name": "Claude API",
                "status": "ok",
                "detail": f"Model: {app.config.llm.model} ({app.config.llm.provider})",
            })
        else:
            results.append({
                "name": "LLM API",
                "status": "warning",
                "detail": "API key set but MissionPlanner not initialized",
            })
    else:
        results.append({
            "name": "LLM API",
            "status": "unavailable",
            "detail": (
                f"No API key for provider {app.config.llm.provider!r} "
                f"(checked env vars + api key/*.env)"
            ),
        })

    # 5. Environment sensors
    if app._monitor is not None:
        health = app._monitor.check_health()
        for sensor_name, is_healthy in health.items():
            if is_healthy:
                results.append({
                    "name": f"Sensor: {sensor_name}",
                    "status": "ok",
                    "detail": "Healthy",
                })
            else:
                # Check if it's a placeholder
                readings = app._monitor.get_latest()
                reading = readings.get(sensor_name)
                if reading and reading.status == "unavailable":
                    results.append({
                        "name": f"Sensor: {sensor_name}",
                        "status": "unavailable",
                        "detail": "Placeholder sensor (no hardware connected)",
                    })
                else:
                    results.append({
                        "name": f"Sensor: {sensor_name}",
                        "status": "warning",
                        "detail": "Sensor unhealthy",
                    })
    else:
        results.append({
            "name": "Environment sensors",
            "status": "error",
            "detail": "EnvironmentMonitor not initialized",
        })

    # 6. Skill registry
    if app._registry is not None:
        skills = app._registry.list_skills()
        results.append({
            "name": "Skill registry",
            "status": "ok" if skills else "warning",
            "detail": f"{len(skills)} skill(s) registered",
        })
    else:
        results.append({
            "name": "Skill registry",
            "status": "error",
            "detail": "SkillRegistry not initialized",
        })

    return results


def format_check_results(results: list[dict]) -> str:
    """Format check results as Markdown for Gradio display."""
    if not results:
        return "_No system check results available._"

    status_icons = {
        "ok": "\u2705",         # green check
        "warning": "\u26a0\ufe0f",  # warning
        "error": "\u274c",      # red x
        "unavailable": "\U0001f527",  # wrench
    }

    lines = ["| Status | Component | Detail |", "|:---:|:---|:---|"]
    for r in results:
        icon = status_icons.get(r["status"], "?")
        lines.append(f"| {icon} | **{r['name']}** | {r['detail']} |")

    # Summary
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    summary_parts = []
    for status in ["ok", "warning", "error", "unavailable"]:
        if status in counts:
            summary_parts.append(f"{status_icons[status]} {counts[status]} {status}")

    lines.append("")
    lines.append("**Summary:** " + " &nbsp; ".join(summary_parts))

    return "\n".join(lines)
