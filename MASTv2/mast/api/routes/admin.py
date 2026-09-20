"""Domain E — Settings write + admin overrides + safety PIN + environment
sensors + Nanonis hardware (TS-rewrite Phase 3).

Mirrors the live Gradio handlers (gui/admin_panel.py, gui/settings_store.py,
gui/route_auth.py, admin/override_store.py, environment/config.py,
gui/dashboard.py). The API layer is a THIN passthrough: it only calls into the
core (ConfigOverrideRegistry, SettingsStore, environment.config, the live app's
connection pool / monitor). NO safety check, NO override merge, NO PIN
comparison lives here — that authority stays in core (R6).

GRACEFUL DEGRADATION is mandatory: this app must boot standalone with no live
core wired. Every endpoint checks ctx for the subsystem it needs; if it's absent
or any call raises, it returns a valid empty/degraded body (``degraded: true``)
— never a 500, never a crash on import. Heavy core modules are LAZY-imported
INSIDE the handler in try/except (mirrors routes/skills.py).

WRITE endpoints are defined for contract completeness but degrade to a typed
no-op (``ok: false, degraded: true``) when the live core is absent; real wiring
to the live singletons + the safety/override passthrough is the integrator's
integration step.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Request, Response, status

from mast.api.schemas import (
    AdminPinSetRequest,
    AdminPinSetResponse,
    AdminPinStatusResponse,
)
from mast.api.schemas_admin import (
    CapturedMessageModel,
    DiscoveredDevice,
    DiscoverResponse,
    InstrumentState,
    InstrumentStateResponse,
    NanonisConnectAccepted,
    NanonisConnectionResponse,
    NanonisConnectRequest,
    NanonisPortStatus,
    OverrideHistoryEntry,
    OverrideHistoryResponse,
    OverrideResponse,
    OverrideWriteRequest,
    OverrideWriteResponse,
    PinUnlockRequest,
    PinUnlockResponse,
    PromptCaptureClearResponse,
    PromptCaptureDetail,
    PromptCaptureListResponse,
    PromptCaptureSummary,
    PromptDetail,
    PromptListResponse,
    PromptOverrideRequest,
    PromptOverrideResponse,
    PromptSummary,
    SensorAdoptRequest,
    SensorAdoptResponse,
    SensorDeleteResponse,
    SensorEntry,
    SensorRescanResponse,
    SensorsResponse,
    SensorWriteRequest,
    SensorWriteResponse,
    SettingsUpdateRequest,
    SettingsUpdateResponse,
    SystemCheckItem,
    SystemCheckResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])

# Category → ConfigOverrideRegistry filename. Single source for the mapping so
# GET / POST / history / restore all agree. Mirrors admin/override_store.py's
# known-files constants WITHOUT importing them at module load (degrade-safe).
_CATEGORY_FILES: dict[str, str] = {
    "safety_limits": "safety_limits.json",
    "checks": "safety_checks.json",
    "constraints": "safety_constraints.json",
    "skill": "skill_overrides.json",
    "knowledge": "knowledge_overrides.json",
    "guidance": "guidance_overrides.json",
    "encyclopedia": "encyclopedia_overrides.json",
    "agent": "agent_overrides.json",
}

# Reserved sensor fields modelled explicitly; everything else is type-specific
# and rides in ``extra`` (mirrors environment/config.py's open JSON schema).
_SENSOR_RESERVED = ("id", "name", "type", "port", "unit")

# Nanonis TCP role → NanonisConfig attribute. Single source so the connection
# snapshot pre-fills the same four ports the old settings hardware pane showed
# (gui/app.py: _nano_p1..p4 read config.nanonis.port_main/monitor/data/emergency).
_NANONIS_PORT_ATTRS: dict[str, str] = {
    "main": "port_main",
    "monitor": "port_monitor",
    "data": "port_data",
    "emergency": "port_emergency",
}


def _nanonis_config(ctx: Any):
    """Best-effort NanonisConfig: the live ctx's config.nanonis when wired,
    else a fresh ``mast.config`` default carrying the canonical 6501-6504 ports.
    Returns None only if even the module defaults can't be loaded (degrade)."""
    live = getattr(getattr(ctx, "config", None), "nanonis", None)
    if live is not None:
        return live
    try:  # lazy: standalone boot may have no live config wired onto ctx
        from mast.config import NanonisConfig

        return NanonisConfig()
    except Exception as exc:
        logger.warning("nanonis default config load failed: %s", exc)
        return None


def _port_for_role(nano_cfg: Any, role: str) -> Optional[int]:
    """Configured TCP port for one role, or None if unresolvable."""
    attr = _NANONIS_PORT_ATTRS.get(role)
    if nano_cfg is None or attr is None:
        return None
    val = getattr(nano_cfg, attr, None)
    return int(val) if isinstance(val, (int, float)) else None


def _override_registry(ctx: Any):
    """Best-effort handle to a live ConfigOverrideRegistry.

    Prefers one already wired onto the context (set at integration
    time); otherwise None. We do NOT construct one here in standalone mode — that
    would touch the shared ``config/overrides`` dir and start mutating real files
    from a dev process. Absent ⇒ degrade."""
    return getattr(ctx, "override_registry", None)


def _sensor_to_entry(raw: dict) -> SensorEntry:
    extra = {k: v for k, v in raw.items() if k not in _SENSOR_RESERVED}
    return SensorEntry(
        id=str(raw.get("id", "")),
        name=raw.get("name"),
        type=raw.get("type"),
        port=raw.get("port"),
        unit=raw.get("unit"),
        extra=extra,
    )


# NOTE: POST /settings moved to routes/settings_admin_write.py (the real unified
# write that ALSO live-applies config.llm.use/set_thinking). This placeholder was
# removed to avoid a duplicate POST /api/settings route.


# ── Admin overrides ─────────────────────────────────────────────────────────
@router.get("/admin/overrides/{category}", response_model=OverrideResponse)
def get_override(request: Request, category: str) -> OverrideResponse:
    """Read the raw override payload for one category (empty if none / unwired)."""
    ctx = request.app.state.ctx
    filename = _CATEGORY_FILES.get(category)
    if filename is None:
        return OverrideResponse(category=category, degraded=True)
    reg = _override_registry(ctx)
    if reg is None:
        return OverrideResponse(category=category, degraded=True)
    try:
        data = reg.get_raw(filename) or {}
        return OverrideResponse(
            category=category,
            data=data,
            has_override=bool(data),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("override read failed (%s): %s", category, exc)
        return OverrideResponse(category=category, degraded=True)


def _restart_required(ctx, category: str, hooks_fired: bool) -> bool | None:
    """Did the write actually reach the running process? None ⇒ can't tell.

    For ``safety_limits`` we can answer from evidence rather than inference,
    and there are TWO independent things to check — this is the trap
    KNOWN_ISSUES §1.1 spelled out ("一个只对了一半的『已生效』，比现在这个诚实的
    『要重启』更危险"):

    * the MANUAL path — compare what the live ``SafetyGuard`` holds against the
      freshly-merged on-disk values (``safety_view.pending_restart``); and
    * the AGENT path — the envelope is baked into each tool's pydantic schema at
      graph-build time, so it needs a graph REBUILD, not a re-merge.
      ``reload_wiring.agent_refresh_pending`` reports the observed scheduling
      result of that rebuild.

    Answering from the first alone would report False the moment the guard
    re-merged, while every agent still ran the old schema. So a confirmed
    "still stale" on EITHER side wins.

    When the agent side reports None it has simply never run — this process has
    no reload wiring — and then the manual comparison is the whole answer again
    (unwired, both paths were built from the same on-disk state at boot, so they
    go stale together). That keeps the pre-§1.1 behaviour exactly where §1.1's
    mechanism is absent.

    Everything else has no live object to interrogate, so the only honest signal
    is whether any reload hook ran — and a False there means "nothing
    re-derived", i.e. restart required.

    Never raises: a failure to answer must degrade to None ("unknown"), never to
    False. Reporting "already in effect" when we do not know is precisely the
    bug this function exists to retire.
    """
    try:
        if category == "safety_limits":
            from mast.admin.reload_wiring import agent_refresh_pending
            from mast.api.safety_view import pending_restart

            manual = pending_restart(ctx)
            agent = agent_refresh_pending()
            if manual is True or agent is True:
                return True
            if agent is False and manual is None:
                # Agent graphs refreshed but we cannot see the manual guard —
                # "half of it is confirmed" is not "it is in effect".
                return None
            if manual is not None:
                return manual
        return not hooks_fired
    except Exception as exc:  # noqa: BLE001
        logger.warning("restart_required undetermined (%s): %s", category, exc)
        return None


@router.post("/admin/overrides/{category}", response_model=OverrideWriteResponse)
def write_override(
    request: Request, category: str, body: OverrideWriteRequest
) -> OverrideWriteResponse:
    """Persist a whole-file override + trigger the core hot-reload (degrade-safe).

    Empty payload ⇒ the core resets to code defaults (deletes the file). The
    merge/validation + signal_reload stays entirely in the registry — the API
    only forwards the bytes.

    ``reloaded`` / ``restart_required`` are OBSERVED, not assumed. Until
    2026-08-03 this returned a hardcoded ``reloaded=True`` while
    ``register_reload_hook`` had zero production subscribers — so a widened
    safety envelope reported success, stayed invisible at ``GET
    /api/safety/limits``, and only took effect at the next restart, with no
    surface anywhere saying so."""
    ctx = request.app.state.ctx
    filename = _CATEGORY_FILES.get(category)
    if filename is None:
        return OverrideWriteResponse(ok=False, category=category, degraded=True)
    reg = _override_registry(ctx)
    if reg is None:
        return OverrideWriteResponse(ok=False, category=category, degraded=True)
    try:
        # save_or_delete_and_reload persists (or deletes on empty) THEN fires the
        # registered reload hooks (SkillRegistry / SafetyGate derived caches),
        # returning how many actually ran.
        fired = reg.save_or_delete_and_reload(filename, dict(body.data))
        data = reg.get_raw(filename) or {}
        return OverrideWriteResponse(
            ok=True, category=category,
            reloaded=bool(fired),
            restart_required=_restart_required(ctx, category, bool(fired)),
            data=data, degraded=False,
        )
    except Exception as exc:
        logger.warning("override write failed (%s): %s", category, exc)
        return OverrideWriteResponse(ok=False, category=category, degraded=True)


@router.get(
    "/admin/overrides/{category}/history", response_model=OverrideHistoryResponse
)
def get_override_history(request: Request, category: str) -> OverrideHistoryResponse:
    """List timestamped backups of one category's override file."""
    ctx = request.app.state.ctx
    filename = _CATEGORY_FILES.get(category)
    if filename is None:
        return OverrideHistoryResponse(category=category, degraded=True)
    reg = _override_registry(ctx)
    if reg is None:
        return OverrideHistoryResponse(category=category, degraded=True)
    try:
        raw = reg.get_history(filename) or []
        entries = [
            OverrideHistoryEntry(timestamp=str(ts), filename=getattr(path, "name", str(path)))
            for ts, path in raw
        ]
        return OverrideHistoryResponse(
            category=category, entries=entries, count=len(entries), degraded=False
        )
    except Exception as exc:
        logger.warning("override history failed (%s): %s", category, exc)
        return OverrideHistoryResponse(category=category, degraded=True)


@router.post(
    "/admin/overrides/{category}/restore/{ts}", response_model=OverrideWriteResponse
)
def restore_override(request: Request, category: str, ts: str) -> OverrideWriteResponse:
    """Restore a category to a historical timestamp (re-saves + hot-reloads)."""
    ctx = request.app.state.ctx
    filename = _CATEGORY_FILES.get(category)
    if filename is None:
        return OverrideWriteResponse(ok=False, category=category, degraded=True)
    reg = _override_registry(ctx)
    if reg is None:
        return OverrideWriteResponse(ok=False, category=category, degraded=True)
    try:
        # restore() re-saves the historical payload (which itself backs up the
        # current one); then signal_reload to refresh derived caches in-process.
        data = reg.restore(filename, ts) or {}
        fired = 0
        try:
            fired = reg.signal_reload() or 0
        except Exception:  # restore succeeded; a reload hiccup is non-fatal
            logger.debug("signal_reload after restore failed (%s)", category)
        # `reloaded` used to be set True merely because signal_reload() did not
        # raise — which it never does when there is nothing subscribed to it.
        return OverrideWriteResponse(
            ok=True, category=category,
            reloaded=bool(fired),
            restart_required=_restart_required(ctx, category, bool(fired)),
            data=data, degraded=False,
        )
    except Exception as exc:
        logger.warning("override restore failed (%s @ %s): %s", category, ts, exc)
        return OverrideWriteResponse(ok=False, category=category, degraded=True)


# ── Admin PIN unlock ────────────────────────────────────────────────────────
@router.post("/admin/unlock-pin", response_model=PinUnlockResponse)
def unlock_pin(request: Request, body: PinUnlockRequest) -> PinUnlockResponse:
    """Compare the entered PIN (SHA256-hex) against the launcher-written
    admin_pin.txt and mint a session token on match (degrade-safe).

    The raw PIN is never stored; the comparison is the same one the live web
    side does (gui/app.py:_read_admin_pin_hash). Reasons: 'no_pin_set' (file
    absent/empty), 'empty' (no PIN entered), 'wrong' (hash mismatch),
    'degraded' (PIN file unreadable). The session-token issuance + auth wiring
    is finalised at integration; here it is a deterministic placeholder token."""
    import hashlib

    entered = (body.pin or "").strip()

    # Resolve the stored hash exactly like the live web side: read the
    # launcher-written file under <project_root>/config/admin_pin.txt.
    stored = ""
    try:
        from mast._runtime_paths import project_root

        p = project_root() / "config" / "admin_pin.txt"
        if p.exists():
            stored = p.read_text(encoding="utf-8").strip().lower()
    except Exception as exc:  # any read error ⇒ treat as degraded (no gate)
        logger.warning("admin pin read failed: %s", exc)
        return PinUnlockResponse(ok=False, reason="degraded", degraded=True)

    if not stored:
        return PinUnlockResponse(ok=False, reason="no_pin_set", degraded=False)
    if not entered:
        return PinUnlockResponse(ok=False, reason="empty", degraded=False)

    digest = hashlib.sha256(entered.encode("utf-8")).hexdigest()
    if digest != stored:
        return PinUnlockResponse(ok=False, reason="wrong", degraded=False)

    # Match. Mint a placeholder session token (integration wires the real token
    # store + route-auth check; the API never holds the raw PIN).
    token = hashlib.sha256(("session:" + digest).encode("utf-8")).hexdigest()
    return PinUnlockResponse(ok=True, token=token, degraded=False)


# ── Admin PIN: status + set/change ──────────────────────────────────────────
# The PIN now GATES something (it used to mint a token nobody checked): the two
# settings keys that grant the agent capability — `hardware_modules` (switching one
# on hands it DANGEROUS skills: a laser, an RF amplifier, colliding probes) and
# `advanced_capabilities` (powers that step around a protection). See
# mast.api.admin_pin for the threat model — it is a guard against a human hand, not
# against the model, which has no path to this API at all.
@router.get("/admin/pin-status", response_model=AdminPinStatusResponse)
def get_pin_status(request: Request) -> AdminPinStatusResponse:
    from mast.api.admin_pin import pin_is_set
    return AdminPinStatusResponse(pin_is_set=pin_is_set())


@router.post("/admin/set-pin", response_model=AdminPinSetResponse)
def set_admin_pin(request: Request, body: AdminPinSetRequest) -> AdminPinSetResponse:
    """Set the admin PIN, or change it (changing requires the current one).

    Only the SHA-256 hash reaches the disk. Lost it? Delete config/admin_pin.txt —
    deliberately recoverable: this is a bench instrument, and locking the operator out
    of their own microscope would be a worse failure than the one the PIN guards.
    """
    from mast.api.admin_pin import reason_text, set_pin
    ok, reason = set_pin(body.new_pin, body.current_pin)
    return AdminPinSetResponse(ok=ok, reason=reason,
                               message="" if ok else reason_text(reason))


# ── Environment sensors ─────────────────────────────────────────────────────
@router.get("/environment/sensors", response_model=SensorsResponse)
def get_sensors(request: Request) -> SensorsResponse:
    """Read the persisted environment-sensor config (empty if unreadable)."""
    try:
        from mast.environment.config import load_config

        cfg = load_config()
        sensors = [_sensor_to_entry(s) for s in (cfg.get("sensors") or []) if isinstance(s, dict)]
        return SensorsResponse(
            autodetect=bool(cfg.get("autodetect", True)),
            sensors=sensors,
            count=len(sensors),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("sensor config read failed: %s", exc)
        return SensorsResponse(degraded=True)


@router.post("/environment/sensors", response_model=SensorWriteResponse)
def add_or_update_sensor(request: Request, body: SensorWriteRequest) -> SensorWriteResponse:
    """Add or update one sensor entry (degrade-safe).

    Persists via environment.config.update_sensor — the core owns the merge +
    atomic write + id/name derivation. The live serial re-open / monitor rebuild
    is the integration step (it touches the exclusive serial bus)."""
    try:
        from mast.environment.config import update_sensor

        fields: dict[str, Any] = {
            "name": body.name,
            "type": body.type,
            "port": body.port,
            "unit": body.unit,
        }
        # Pass type-specific fields (alarm/address/channel/baudrate/...) verbatim.
        for k, v in (body.extra or {}).items():
            if k not in _SENSOR_RESERVED:
                fields[k] = v
        saved = update_sensor(body.id, **fields)
        return SensorWriteResponse(ok=True, sensor=_sensor_to_entry(saved), degraded=False)
    except Exception as exc:
        logger.warning("sensor write failed (%s): %s", body.id, exc)
        return SensorWriteResponse(ok=False, degraded=True)


@router.delete("/environment/sensors/{sensor_id}", response_model=SensorDeleteResponse)
def delete_sensor(request: Request, sensor_id: str) -> SensorDeleteResponse:
    """Remove one sensor entry from the config (degrade-safe)."""
    try:
        from mast.environment.config import remove_sensor

        removed = bool(remove_sensor(sensor_id))
        return SensorDeleteResponse(ok=True, removed=removed, degraded=False)
    except Exception as exc:
        logger.warning("sensor delete failed (%s): %s", sensor_id, exc)
        return SensorDeleteResponse(ok=False, degraded=True)


def _rebuild_sensor_set(request, build_environment_sensors) -> list:
    """Full sensor set for a live rebuild: serial/autodetected + Nanonis-backed.

    The Nanonis-backed sensors (tunnelling-current mirror, analogue-input
    gauges) are constructed by ``core.runtime`` because they need a live
    ConnectionPool. Rebuilding with ``build_environment_sensors()`` alone —
    which is what both rescan paths used to do — swaps the monitor's sensor set
    for one that does not contain them, so a single "Rescan" silently retired
    every Nanonis reading until the next restart. Their names are also reserved
    so the placeholder pass does not shadow them.
    """
    extra_fn = getattr(getattr(request.app.state, "ctx", None), "env_extra_sensors", None)
    extra: list = []
    if callable(extra_fn):
        try:
            extra = list(extra_fn() or [])
        except Exception as exc:  # noqa: BLE001 — a serial rescan must not fail
            logger.warning("rescan: nanonis env sensors unavailable: %s", exc)
            extra = []
    names = []
    for s in extra:
        try:
            names.append(s.name())
        except Exception:  # noqa: BLE001
            continue
    sensors = list(build_environment_sensors(reserved_names=names))
    sensors.extend(extra)
    return sensors


def _release_bus(monitor) -> bool:
    """Stop the monitor loop and close every serial handle; report prior state.

    Windows opens COM ports EXCLUSIVELY, so any operation that probes or
    re-opens a port must first take the bus away from the live monitor —
    otherwise the probe fails on a port MAST itself is holding and the caller
    wrongly concludes "no device there". Returns whether the archive loop had
    been running, for :func:`_restore_bus`.
    """
    was_running = bool(getattr(monitor, "is_running", False))
    stop = getattr(monitor, "stop", None)
    if callable(stop):
        stop()
    return was_running


def _restore_bus(monitor, *, running: bool) -> None:
    """Restart the archive/alarm loop iff it should be running now."""
    start = getattr(monitor, "start", None)
    if running and callable(start):
        start()


def _resync_env_history_gates(sensors) -> None:
    """Re-tell the env-history sink which series are quiet-gated. Never raises.

    ``EnvHistorySink.set_gated_sensors``' own docstring says "运行时接线（传感器
    集合变了）会调它" — and until now nothing did. The list was computed ONCE at
    boot from the boot-time sensor set (``CoreRuntime._build_env_history_
    recorder``), so every live swap through here left the archive gating a
    sensor set that no longer exists: a hot-plugged quiet-gated series would be
    bucketed while the instrument is noisy, which is precisely the pollution the
    gate exists to prevent. The producer was wired and the consumer was absent.

    ``None`` vs ``[]`` is load-bearing on the far side (None = keep the default
    name table, [] = nothing is gated), so we always pass a concrete list — a
    rig with no Nanonis-backed sensors really does have nothing to gate.
    """
    try:
        from mast.envhistory.recorder import get_recorder

        rec = get_recorder()
        sink = getattr(rec, "sink", None)
        setter = getattr(sink, "set_gated_sensors", None)
        if not callable(setter):
            return
        names = []
        for s in sensors or []:
            inner = getattr(s, "_inner", s)  # unwrap autodetect._RenamedSensor
            if getattr(inner, "quiet_gated", False):
                names.append(s.name())
        setter(names)
    except Exception as exc:  # noqa: BLE001 — bookkeeping must not fail a rescan
        logger.debug("env history gate resync skipped: %s", exc)


@router.get("/environment/discover", response_model=DiscoverResponse)
def discover_environment_devices(request: Request) -> DiscoverResponse:
    """Scan the serial ports and DESCRIBE what is on them. Persists nothing.

    First half of「扫描设备接口」: the user clicks scan, this identifies each
    port and reads enough live state (model, per-input label + temperature,
    heater range/output) for the confirmation dialog to show what was found.
    Nothing is monitored or written until the user adopts it via
    POST /environment/sensors/adopt — so a wrong identification costs a
    dismissed dialog, not a bad config.
    """
    ctx = request.app.state.ctx
    monitor = getattr(ctx, "environment_monitor", None)
    if monitor is None:
        # Standalone API process: never open serial ports the live app owns.
        return DiscoverResponse(degraded=True)
    was_running = _release_bus(monitor)
    found: list[dict] = []
    try:
        from mast.environment.autodetect import discover_devices

        found = discover_devices()
    except Exception as exc:  # noqa: BLE001 — a failed scan must not 500
        logger.warning("environment discovery failed: %s", exc)
    finally:
        _restore_bus(monitor, running=was_running)
    devices: list[DiscoveredDevice] = []
    for d in found:
        try:
            devices.append(DiscoveredDevice(**d))
        except Exception as exc:  # noqa: BLE001 — skip one malformed entry
            logger.debug("discovery entry dropped: %s", exc)
    return DiscoverResponse(
        devices=devices,
        identified_count=sum(1 for d in devices if d.identified),
        degraded=False,
    )


@router.post("/environment/sensors/adopt", response_model=SensorAdoptResponse)
def adopt_sensors(request: Request, body: SensorAdoptRequest) -> SensorAdoptResponse:
    """Persist the entries the user confirmed, then bring them live.

    Second half of「扫描设备接口」. Persisting is what makes the port stick: on
    every later boot ``build_sensors_from_config`` rebuilds these entries before
    autodetect runs, so the instrument is found without another scan — and
    because the port is then CLAIMED, autodetect skips probing it at all.
    """
    from mast.environment.config import SENSOR_TYPES, update_sensor

    accepted: list[dict] = []
    rejected: list[str] = []
    for raw in body.sensors or []:
        if not isinstance(raw, dict):
            rejected.append(str(raw)[:40])
            continue
        sid = str(raw.get("id") or "").strip()
        stype = str(raw.get("type") or "").strip()
        # Validate here rather than trusting the round-trip: these entries reach
        # a serial driver, and an unknown type would persist a row that can
        # never build into a sensor.
        if not sid:
            rejected.append(f"<missing id> ({stype or 'unknown type'})")
            continue
        if stype not in SENSOR_TYPES:
            rejected.append(f"{sid} (unsupported type {stype!r})")
            continue
        accepted.append(raw)

    persisted: list[SensorEntry] = []
    for entry in accepted:
        fields = {k: v for k, v in entry.items() if k != "id"}
        try:
            persisted.append(_sensor_to_entry(update_sensor(entry["id"], **fields)))
        except Exception as exc:  # noqa: BLE001 — report, don't 500
            logger.warning("sensor adopt failed (%s): %s", entry.get("id"), exc)
            rejected.append(f"{entry.get('id')} (write failed)")

    live_count = 0
    monitor = getattr(request.app.state.ctx, "environment_monitor", None)
    if monitor is not None and persisted:
        was_running = _release_bus(monitor)
        try:
            from mast.environment.autodetect import (
                build_environment_sensors,
                has_real_sensors,
            )

            new_sensors = _rebuild_sensor_set(request, build_environment_sensors)
            if hasattr(monitor, "replace_sensors"):
                monitor.replace_sensors(new_sensors)
            _resync_env_history_gates(new_sensors)
            _restore_bus(monitor, running=has_real_sensors(new_sensors))
            live_count = len(list(getattr(monitor, "sensor_names", lambda: [])()))
        except Exception as exc:  # noqa: BLE001 — config is saved either way
            logger.warning("adopt: live rebuild failed (config persisted): %s", exc)
            _restore_bus(monitor, running=was_running)

    return SensorAdoptResponse(
        ok=bool(persisted),
        adopted=len(persisted),
        rejected=rejected,
        sensors=persisted,
        live_count=live_count,
        degraded=monitor is None,
    )


@router.get("/environment/instrument-state", response_model=InstrumentStateResponse)
def instrument_state(request: Request) -> InstrumentStateResponse:
    """Read-only settings/heater readout of the instruments MAST is connected to.

    Reads over each live sensor's ALREADY-OPEN port instead of taking the bus,
    so the settings panel can refresh while the monitor keeps polling (the
    transport serialises the two readers). One instrument is reported once even
    though its inputs are separate sensors sharing that port.

    Everything here is a query — the panel can show that a heater is on, and has
    no path to turn one on.
    """
    ctx = request.app.state.ctx
    monitor = getattr(ctx, "environment_monitor", None)
    if monitor is None:
        return InstrumentStateResponse(degraded=True)
    try:
        from mast.environment.lakeshore_temp import LakeshoreTemperatureSensor
    except Exception:  # pragma: no cover - import guard
        return InstrumentStateResponse(degraded=True)

    sensors = getattr(monitor, "_sensors", {}) or {}
    seen_ports: set[str] = set()
    out: list[InstrumentState] = []
    for sensor in list(sensors.values()):
        inner = getattr(sensor, "_inner", sensor)   # unwrap _RenamedSensor
        if not isinstance(inner, LakeshoreTemperatureSensor):
            continue
        settings = getattr(inner, "_settings", None)
        port = getattr(settings, "port", "") or ""
        key = port.upper()
        if key and key in seen_ports:
            continue      # sibling input of an instrument already reported
        try:
            snap = inner.instrument_state()
        except Exception as exc:  # noqa: BLE001 — panel must never 500
            logger.debug("instrument_state failed on %s: %s", port, exc)
            continue
        if snap is None:
            continue
        if key:
            seen_ports.add(key)
        data = snap.as_dict()
        data.pop("settings", None)
        out.append(InstrumentState(kind="lakeshore_temp", **data))
    return InstrumentStateResponse(instruments=out, degraded=False)


@router.get("/environment/sensors/rescan", response_model=SensorRescanResponse)
def rescan_sensors(request: Request) -> SensorRescanResponse:
    """Re-enumerate serial ports + rebuild the live monitor (degrade-safe).

    The rescan touches the exclusive serial bus, which only the LIVE app owns
    (its EnvironmentMonitor). With no monitor wired we degrade to the persisted
    config — never opening serial ports from this standalone process."""
    ctx = request.app.state.ctx
    monitor = getattr(ctx, "environment_monitor", None)
    if monitor is None:
        # No live monitor → report the persisted config without touching serial.
        try:
            from mast.environment.config import load_config

            cfg = load_config()
            sensors = [
                _sensor_to_entry(s) for s in (cfg.get("sensors") or []) if isinstance(s, dict)
            ]
        except Exception:
            sensors = []
        return SensorRescanResponse(ok=False, count=len(sensors), sensors=sensors, degraded=True)
    try:
        # ACTUALLY rescan: rebuild the sensor set (enumerate COM ports + re-open
        # devices from the persisted config/autodetect) and swap it into the live
        # monitor. The old code only echoed the current names and returned ok=True
        # without touching serial, so the "Rescan" button was a no-op (review
        # 2026-07-03). Falls back to reporting the current set if the rebuild or
        # replace_sensors API isn't available.
        rebuilt = False
        # Free the bus BEFORE re-probing: a sensor that is already connected
        # still holds its port, autodetect's probe would then fail, and
        # replace_sensors would swap a WORKING gauge out for a placeholder.
        # monitor.py's replace_sensors docstring has always required this
        # ordering — the method it pointed at (MASTApp._rebuild_environment_monitor)
        # no longer exists, so the requirement was silently unmet here.
        was_running = _release_bus(monitor)
        try:
            from mast.environment.autodetect import (
                build_environment_sensors,
                has_real_sensors,
            )

            try:
                new_sensors = _rebuild_sensor_set(request, build_environment_sensors)
            except Exception:
                # Rebuild failed and the old set's handles are now closed. Put
                # the loop back the way we found it so a failed rescan can't
                # leave monitoring silently dead.
                _restore_bus(monitor, running=was_running)
                raise
            if hasattr(monitor, "replace_sensors"):
                monitor.replace_sensors(new_sensors)
                rebuilt = True
            _resync_env_history_gates(new_sensors)
            # Start the archive/alarm loop when the rescan actually found
            # hardware. It is otherwise only ever started at boot (core.runtime),
            # so a run that booted with no instrument would show live values in
            # the panel but never archive them and never escalate an over-limit
            # into an E_STOP.
            _restore_bus(monitor, running=has_real_sensors(new_sensors))
        except Exception as exc:  # noqa: BLE001 — degrade to reporting current set
            logger.warning("sensor rescan rebuild failed (reporting current): %s", exc)
        names = list(getattr(monitor, "sensor_names", lambda: [])())
        sensors = [SensorEntry(id=n, name=n) for n in names]
        return SensorRescanResponse(
            ok=rebuilt, count=len(sensors), sensors=sensors, degraded=not rebuilt
        )
    except Exception as exc:
        logger.warning("sensor rescan failed: %s", exc)
        return SensorRescanResponse(ok=False, degraded=True)


@router.post("/admin/shutdown")
def admin_shutdown(request: Request) -> dict:
    """Graceful service shutdown: close the Nanonis pool + stop daemons, then
    ask uvicorn to exit. Loopback-only + a one-time token (MAST2_SHUTDOWN_TOKEN)
    so it can't be triggered remotely. The launcher calls this BEFORE its
    taskkill fallback so a windowless frozen service stops gracefully instead of
    being TerminateProcess'd mid-TCP (which corrupts the port; review
    2026-07-03)."""
    import os as _os
    # Loopback only.
    client_host = getattr(getattr(request, "client", None), "host", "") or ""
    if client_host not in ("127.0.0.1", "::1", "localhost", ""):
        return {"ok": False, "error": "shutdown is loopback-only"}
    # Token gate (if configured on the service env).
    expected = _os.environ.get("MAST2_SHUTDOWN_TOKEN", "")
    if expected:
        import secrets
        token = request.headers.get("x-mast-shutdown-token", "")
        if not secrets.compare_digest(token, expected):
            return {"ok": False, "error": "bad shutdown token"}
    ctx = request.app.state.ctx
    app_handle = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
    closed = False
    try:
        if app_handle is not None and hasattr(app_handle, "shutdown"):
            app_handle.shutdown()
            closed = True
    except Exception as exc:  # pragma: no cover
        logger.warning("admin_shutdown: app.shutdown failed: %s", exc)
    # Ask uvicorn to exit if the launcher wired the server onto app.state.
    server = getattr(request.app.state, "uvicorn_server", None)
    if server is not None:
        try:
            server.should_exit = True
        except Exception:  # pragma: no cover
            pass
    _arm_exit_backstop()
    return {"ok": True, "pool_closed": closed, "exiting": server is not None}


#: How long the process may take to exit on its own after a graceful shutdown
#: before the backstop forces it. Generous: the point is to catch a HANG, not to
#: race a slow-but-working teardown (uvicorn drains connections, the pool closes
#: four sockets, the vision worker may be mid-inference).
_EXIT_BACKSTOP_S = 20.0


def _arm_exit_backstop() -> None:
    """Make sure the process actually dies after a graceful shutdown — and say
    WHAT was still holding it.

    A graceful ``/api/admin/shutdown`` has been observed to return
    ``ok/pool_closed/exiting`` while the process shell stayed alive and needed
    a manual kill (netstat showed no TCP residue at that point).
    **Not reproduced on the dev machine** — four clean exits in a row — which is
    exactly why the diagnostic half matters more than the exit half: the
    difference is the real instrument's hardware (four live Nanonis sockets, CUDA
    loaded, the Osci1T pump mid-segment), and nobody can guess which from here.

    So: wait, and if the process is still up, DUMP THE SURVIVING THREADS before
    forcing exit. Every Python thread in this tree is already ``daemon=True``
    (checked, including ``SafetyWatchdog``, which sets it as a class attribute),
    so whatever is holding it is either not a Python thread or is stuck inside a
    C call — and the thread dump is the only thing that will say which.

    ``os._exit`` skips atexit handlers and buffer flushing, which is acceptable
    ONLY because it runs after ``app.shutdown()`` has already closed the pool:
    the thing that must not be skipped (a graceful TCP close, or the Nanonis
    port stays corrupted until Nanonis restarts) has already happened."""
    import os as _os
    import threading as _th

    def _backstop() -> None:
        import faulthandler
        import sys

        alive = [t for t in _th.enumerate()
                 if t is not _th.current_thread() and t.is_alive()]
        logger.error(
            "admin_shutdown: 进程在 graceful shutdown 后 %.0f s 仍未退出,"
            "强制结束。仍存活的线程:%s",
            _EXIT_BACKSTOP_S,
            [f"{t.name}(daemon={t.daemon})" for t in alive])
        try:
            # The stack of every surviving thread — the one artefact that turns
            # "it hung again" into a diagnosable report.
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        except Exception:  # noqa: BLE001
            pass
        try:
            from mast.core.diagnostics import record as _diag

            _diag("shutdown_backstop", "admin",
                  "graceful shutdown 后进程未退出,已强制结束",
                  threads=[t.name for t in alive])
        except Exception:  # noqa: BLE001
            pass
        _os._exit(0)

    t = _th.Timer(_EXIT_BACKSTOP_S, _backstop)
    t.name = "shutdown-exit-backstop"
    t.daemon = True          # must never itself keep the process alive
    t.start()


# ── Nanonis hardware ────────────────────────────────────────────────────────
@router.post(
    "/nanonis/connect",
    response_model=NanonisConnectAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def nanonis_connect(
    request: Request, body: NanonisConnectRequest, response: Response
) -> NanonisConnectAccepted:
    """Kick off an async (graceful, multi-port) Nanonis connect → 202 Accepted.

    Force-kill of a live TCP corrupts the port permanently, so the actual
    connect is always graceful and runs in a background worker owned by the live
    app; the frontend polls GET /api/nanonis/connection for the outcome. With no
    connection pool wired we still 202 (accepted) but flag degraded — nothing
    actually connects from this standalone process."""
    ctx = request.app.state.ctx
    pool = getattr(ctx, "connection_pool", None)
    live_app = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
    if pool is None or live_app is None or not hasattr(live_app, "reconnect"):
        # Accepted-but-degraded: keep the async contract (202) so the frontend
        # flow is identical; the body says nothing will actually connect here.
        return NanonisConnectAccepted(accepted=True, status="unavailable", degraded=True)

    # Apply optional host/port overrides to the live config BEFORE reconnecting
    # (reconnect() reads config.nanonis). Then kick off a GRACEFUL background
    # reconnect — this endpoint used to just echo "connecting" and never call
    # reconnect(), so the Connect button did nothing.
    try:
        nano = getattr(getattr(live_app, "config", None), "nanonis", None)
        if nano is not None:
            if body.host:
                nano.host = body.host
            for attr in ("port_main", "port_monitor", "port_data", "port_emergency"):
                val = getattr(body, attr, None)
                if val and hasattr(nano, attr):
                    setattr(nano, attr, int(val))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("nanonis_connect override apply failed: %s", exc)

    # ⚠️ **worker 的结局必须落在一个能被读到的地方。**
    # 2026-08-25 之前它只进 logger.warning —— 而 console=False 的构建吞 stdout、
    # 磁盘上也没有运行时日志，于是「失败了」与「还在连」在轮询端点上永远同形。
    import time as _time

    def _record(state: str) -> None:
        try:
            ctx.last_nanonis_connect = (state, _time.time())
        except Exception:  # noqa: BLE001 — 记录失败绝不能弄坏重连本身
            pass

    _record("connecting")

    def _do_reconnect() -> None:
        try:
            n = live_app.reconnect()
            logger.info("nanonis_connect: background reconnect connected %s port(s)", n)
            _record("ok:%s" % n)
        except Exception as exc:  # noqa: BLE001 — never crash the worker thread
            logger.warning("nanonis_connect: background reconnect failed: %s", exc)
            _record("failed: %s" % str(exc)[:200])

    import threading
    threading.Thread(target=_do_reconnect, name="nanonis-connect", daemon=True).start()
    return NanonisConnectAccepted(accepted=True, status="connecting", degraded=False)


@router.get("/nanonis/connection", response_model=NanonisConnectionResponse)
def nanonis_connection(request: Request) -> NanonisConnectionResponse:
    """Snapshot the Nanonis connection across the four TCP roles (degrade-safe)."""
    ctx = request.app.state.ctx
    pool = getattr(ctx, "connection_pool", None)
    roles = ("main", "monitor", "data", "emergency")
    # Always resolve the configured ports (live config.nanonis when wired, else
    # the canonical mast.config defaults) so the UI pre-fills 6501-6504 even when
    # nothing is connected. connected/degraded stay driven by the pool below.
    nano_cfg = _nanonis_config(ctx)
    host_val = getattr(nano_cfg, "host", None) if nano_cfg is not None else None
    if pool is None:
        ports = [
            NanonisPortStatus(
                role=r, port=_port_for_role(nano_cfg, r), connected=False, detail="not wired"
            )
            for r in roles
        ]
        _lr = getattr(ctx, "last_nanonis_connect", None)
        return NanonisConnectionResponse(
            connected=False, host=host_val, ports=ports, degraded=True,
            last_connect_result=(_lr[0] if _lr else None),
            last_connect_at=(_lr[1] if _lr else None),
        )
    try:
        ports: list[NanonisPortStatus] = []
        any_connected = False
        for role in roles:
            connected = False
            detail = None
            try:
                # pool.get(role) raises if the role isn't connected (mirrors
                # dashboard.run_system_check). No reconnection attempt here.
                pool.get(role)
                connected = True
                any_connected = True
            except Exception as exc:
                detail = f"not connected: {exc}"
            ports.append(
                NanonisPortStatus(
                    role=role,
                    port=_port_for_role(nano_cfg, role),
                    connected=connected,
                    detail=detail,
                )
            )
        _lr = getattr(ctx, "last_nanonis_connect", None)
        return NanonisConnectionResponse(
            connected=any_connected, host=host_val, ports=ports, degraded=False,
            last_connect_result=(_lr[0] if _lr else None),
            last_connect_at=(_lr[1] if _lr else None),
        )
    except Exception as exc:
        logger.warning("nanonis connection snapshot failed: %s", exc)
        ports = [
            NanonisPortStatus(role=r, port=_port_for_role(nano_cfg, r), connected=False)
            for r in roles
        ]
        _lr = getattr(ctx, "last_nanonis_connect", None)
        return NanonisConnectionResponse(
            connected=False, ports=ports, degraded=True,
            last_connect_result=(_lr[0] if _lr else None),
            last_connect_at=(_lr[1] if _lr else None),
        )


# ── System self-check ───────────────────────────────────────────────────────
@router.get("/system/check", response_model=SystemCheckResponse)
def system_check(request: Request) -> SystemCheckResponse:
    """Run the system self-check (Nanonis / storage / LLM / sensors).

    Needs the live app singletons (connection pool / storage / monitor). With no
    live app wired we degrade to an empty list — never probing hardware from the
    standalone process."""
    ctx = request.app.state.ctx
    app_handle = getattr(ctx, "app", None) or getattr(ctx, "live_app", None)
    if app_handle is None:
        return SystemCheckResponse(degraded=True)
    try:
        from mast.webui.dashboard import run_system_check

        raw = run_system_check(app_handle) or []
        items = [
            SystemCheckItem(
                name=str(r.get("name", "")),
                status=r.get("status", "unavailable"),
                detail=str(r.get("detail", "")),
            )
            for r in raw
            if isinstance(r, dict)
        ]
        return SystemCheckResponse(items=items, count=len(items), degraded=False)
    except Exception as exc:
        logger.warning("system check failed: %s", exc)
        return SystemCheckResponse(degraded=True)


# ── 上下文注入 (context injection) ───────────────────────────────────────────
# The running-application half of MASTv2/scripts/dump_prompts_html.py: read the
# assembled prompt surface, edit the parts that are editable, and — separately —
# look at what a real request actually carried.
#
# THE HONESTY RULE this surface is built around: a block that needs live hardware
# or per-request state is reported as unavailable, with the reason. It is never
# backfilled with an illustrative sample. The 2026-07-27 coordinate incident
# (model wrote 1.2531 for 1.2531e-6 m) was traced to OUR injection printing
# "(= 1253.1 nm)" — the value of this page is that it shows the real text, and a
# convincing fake would make it actively harmful.
#
# Heavy core is lazy-imported inside the handler (same rule as every other route
# here), so a standalone API process degrades instead of failing to import.

def _prompt_preview(text: str, limit: int = 220) -> str:
    one = " ".join(text.split())
    return one if len(one) <= limit else one[:limit] + "…"


def _prompt_bodies(entry: Any) -> tuple[str, Optional[str], str, str]:
    """``(default_text, override_text, effective_text, unavailable_reason)``."""
    from mast.prompts import overrides as ovr
    from mast.prompts import registry as reg

    default_text, err = reg.render_default(entry)
    override_text = ovr.get(entry.id) if entry.overridable else None
    effective = override_text if (override_text or "").strip() else default_text
    reason = err or entry.unavailable_reason
    # A successful render that produced nothing is a DIFFERENT fact from "we
    # could not read it", and the page must not let the two look alike.
    if not reason and not effective.strip() and entry.empty_note:
        reason = entry.empty_note
    return default_text, override_text, effective, reason


def _prompt_summary_kwargs(entry: Any) -> tuple[dict, tuple[str, Optional[str], str]]:
    """``(summary_kwargs, (default, override, effective))`` for one entry."""
    default_text, override_text, effective, reason = _prompt_bodies(entry)
    kwargs = {
        "id": entry.id,
        "label": entry.label,
        "category": entry.category,
        "agent": entry.agent,
        "availability": entry.availability,
        "source": entry.source,
        "note": entry.note,
        "overridable": entry.overridable,
        "overridden": bool((override_text or "").strip()),
        "default_chars": len(default_text),
        "effective_chars": len(effective),
        "preview": _prompt_preview(effective),
        "unavailable_reason": reason,
    }
    return kwargs, (default_text, override_text, effective)


@router.get("/admin/prompts", response_model=PromptListResponse)
def list_prompts(request: Request) -> PromptListResponse:
    """Inventory of every context-injection text (metadata + preview, no bodies)."""
    try:
        from mast.prompts import registry as reg
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt registry unavailable: %s", exc)
        return PromptListResponse(degraded=True)
    items: list[PromptSummary] = []
    for entry in reg.entries():
        try:
            kw, _bodies = _prompt_summary_kwargs(entry)
            items.append(PromptSummary(**kw))
        except Exception as exc:  # noqa: BLE001 — one bad entry must not hide the rest
            logger.warning("prompt entry %s failed: %s", getattr(entry, "id", "?"), exc)
    return PromptListResponse(
        items=items,
        count=len(items),
        overridden_count=sum(1 for i in items if i.overridden),
        degraded=False,
    )


@router.get("/admin/prompts/{prompt_id}", response_model=PromptDetail)
def get_prompt(request: Request, prompt_id: str, response: Response) -> PromptDetail:
    """Full bodies for one entry: code default, override, and what is in force."""
    try:
        from mast.prompts import registry as reg
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt registry unavailable: %s", exc)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return PromptDetail(id=prompt_id, label=prompt_id, degraded=True)
    entry = reg.get_entry(prompt_id)
    if entry is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return PromptDetail(id=prompt_id, label=prompt_id,
                            unavailable_reason="未知的注入话术 ID")
    kw, (default_text, override_text, effective) = _prompt_summary_kwargs(entry)
    return PromptDetail(**kw, default_text=default_text,
                        override_text=override_text, effective_text=effective)


@router.post("/admin/prompts/{prompt_id}", response_model=PromptOverrideResponse)
def write_prompt_override(
    request: Request, prompt_id: str, body: PromptOverrideRequest,
    response: Response,
) -> PromptOverrideResponse:
    """Persist an override. Empty text clears it (back to the code default).

    Persistence goes to ``config/overrides/prompt_overrides.json`` via
    ConfigOverrideRegistry — the same layer as safety limits, so the edit
    survives a restart and lands in 覆盖历史.
    """
    try:
        from mast.prompts import overrides as ovr
        from mast.prompts import registry as reg
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt registry unavailable: %s", exc)
        return PromptOverrideResponse(prompt_id=prompt_id, degraded=True,
                                      message="内核未接入，写入未生效。")
    entry = reg.get_entry(prompt_id)
    if entry is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return PromptOverrideResponse(prompt_id=prompt_id, message="未知的注入话术 ID")
    if not entry.overridable:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return PromptOverrideResponse(
            prompt_id=prompt_id,
            message="该注入块的内容由运行时状态计算，没有可替换的固定文本。",
        )
    text = body.text or ""
    try:
        ok = ovr.set_override(prompt_id, text) if text.strip() else ovr.clear(prompt_id)
    except ValueError as exc:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return PromptOverrideResponse(prompt_id=prompt_id, message=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt override write failed for %s: %s", prompt_id, exc)
        return PromptOverrideResponse(prompt_id=prompt_id, degraded=True,
                                      message="写入失败。")
    _, override_text, effective, _ = _prompt_bodies(entry)
    return PromptOverrideResponse(
        ok=ok, prompt_id=prompt_id,
        overridden=bool((override_text or "").strip()),
        effective_chars=len(effective),
        message=(("已保存；" if text.strip() else "已恢复默认；")
                 + ("下一次编排器重建（或重启）后生效。"
                    if entry.category == "agent_system"
                    else "下一次模型调用即生效。")),
        degraded=not ok,
    )


@router.delete("/admin/prompts/{prompt_id}", response_model=PromptOverrideResponse)
def clear_prompt_override(
    request: Request, prompt_id: str, response: Response,
) -> PromptOverrideResponse:
    """Drop the override — restore the shipped default."""
    return write_prompt_override(request, prompt_id, PromptOverrideRequest(text=""),
                                 response)


@router.get("/admin/prompt-capture", response_model=PromptCaptureListResponse)
def list_prompt_capture(request: Request) -> PromptCaptureListResponse:
    """The last few REAL model requests (newest first) — metadata only.

    Real, possibly stale, and empty until a model call has actually run in this
    process. That is deliberately not papered over with a dry-run render: a
    reconstruction cannot reproduce live hardware reads, conversation history, or
    the graph's middleware ordering, so it would show text that merely resembles
    what was sent.
    """
    try:
        from mast.prompts import capture as cap
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt capture unavailable: %s", exc)
        return PromptCaptureListResponse(degraded=True)
    import time as _time

    ring = cap.get_ring()
    now = _time.time()
    items = [
        PromptCaptureSummary(
            index=i, seq=s.seq, ts=s.ts, age_s=max(0.0, now - s.ts),
            source=s.source,
            model_id=s.model_id, provider=s.provider,
            message_count=len(s.messages), total_chars=s.total_chars,
            system_chars=sum(m.chars for m in s.messages if m.role == "system"),
        )
        for i, s in enumerate(ring.list())
    ]
    # "off" and "idle" both show an empty list; saying which is which is the
    # whole job of this note.
    if items:
        note = ""
    elif not ring.enabled:
        note = "快照记录已关闭，因此这里是空的 —— 不是没有发生过模型调用。"
    else:
        note = ("本进程还没有发生过模型调用，所以没有可展示的真实注入。"
                "这里只记录真实请求，不做离线模拟渲染。")
    return PromptCaptureListResponse(
        items=items, count=len(items), enabled=ring.enabled,
        total_seen=ring.total_seen, capacity=cap.MAX_SNAPSHOTS, note=note,
    )


def _capture_detail(snap, *, index: int) -> PromptCaptureDetail:
    import time as _time

    return PromptCaptureDetail(
        index=index, seq=snap.seq, ts=snap.ts,
        age_s=max(0.0, _time.time() - snap.ts),
        source=snap.source, model_id=snap.model_id, provider=snap.provider,
        messages=[
            CapturedMessageModel(role=m.role, content=m.content, chars=m.chars,
                                 truncated=m.truncated)
            for m in snap.messages
        ],
        total_chars=snap.total_chars, dropped_messages=snap.dropped_messages,
        found=True,
    )


@router.get("/admin/prompt-capture/by-seq/{seq}",
            response_model=PromptCaptureDetail)
def get_prompt_capture_by_seq(request: Request, seq: int,
                              response: Response) -> PromptCaptureDetail:
    """Full message list of the request with this ``seq`` — **the stable handle**.

    The index route below renumbers on every model call: list, pick index 3,
    fetch index 3, and if one call landed in between you get a different request
    with nothing in the response saying so. Unobservable when a human clicks
    through the admin inspector; constant in the chat view, where the agent is
    calling the model while the operator expands a turn.

    ``404`` here means *that* snapshot is gone (evicted past ``MAX_SNAPSHOTS``),
    which is a real answer — not "here is a different one".
    """
    try:
        from mast.prompts import capture as cap
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt capture unavailable: %s", exc)
        return PromptCaptureDetail(seq=seq, degraded=True)
    ring = cap.get_ring()
    snap = ring.get_by_seq(seq)
    if snap is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return PromptCaptureDetail(seq=seq, found=False)
    # ``index`` is only a display position; it is correct at this instant and
    # may be stale by the time the caller reads it. ``seq`` is the identity.
    try:
        index = next(i for i, s in enumerate(ring.list()) if s.seq == seq)
    except StopIteration:  # evicted between the two reads — the seq still holds
        index = -1
    return _capture_detail(snap, index=index)


@router.get("/admin/prompt-capture/{index}", response_model=PromptCaptureDetail)
def get_prompt_capture(request: Request, index: int,
                       response: Response) -> PromptCaptureDetail:
    """Full message list of one captured request (index 0 = newest).

    ⚠️ ``index`` shifts whenever a model call lands. Anything that can race with
    a running agent should use ``/admin/prompt-capture/by-seq/{seq}``.
    """
    try:
        from mast.prompts import capture as cap
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt capture unavailable: %s", exc)
        return PromptCaptureDetail(index=index, degraded=True)

    snap = cap.get_ring().get(index)
    if snap is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return PromptCaptureDetail(index=index, found=False)
    return _capture_detail(snap, index=index)


@router.delete("/admin/prompt-capture", response_model=PromptCaptureClearResponse)
def clear_prompt_capture(request: Request) -> PromptCaptureClearResponse:
    """Empty the capture ring."""
    try:
        from mast.prompts import capture as cap
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt capture unavailable: %s", exc)
        return PromptCaptureClearResponse(ok=False)
    ring = cap.get_ring()
    n = len(ring.list())
    ring.clear()
    return PromptCaptureClearResponse(ok=True, cleared=n)
