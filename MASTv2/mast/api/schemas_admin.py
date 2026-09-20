"""Pydantic request/response models for Domain E — Settings write + admin
overrides + safety pin + environment sensors + Nanonis hardware (TS-rewrite
Phase 3).

These are the typed contract for the write/admin surface that mirrors the live
Gradio handlers (``gui/admin_panel.py``, ``gui/settings_store.py``,
``gui/route_auth.py``, ``admin/override_store.py``). The shapes are exported via
``/openapi.json`` and consumed by the frontend type generators; the SINGLE
SOURCE OF TYPES rule (F4) means we re-use the core models verbatim where they
exist instead of forking a parallel copy that could drift.

EVERY response carries a ``degraded`` boolean so a standalone API process (no
live core wired) returns an empty-but-valid body instead of 500-ing (R: UI must
never freeze; the seam must boot standalone).

Safety / override business logic NEVER lives here — these models only carry data
to and from the core (R6).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Re-use the core hardware-safety model verbatim (no parallel copy that drifts).
from mast.config import SafetyLimits as SafetyLimits  # noqa: F401  (re-export)

# The eight override categories the admin surface can edit. Mirrors the
# ConfigOverrideRegistry known-files set (admin/override_store.py): each maps to
# one JSON override file the core hot-reloads on save.
OverrideCategory = Literal[
    "safety_limits",
    "checks",
    "constraints",
    "skill",
    "knowledge",
    "guidance",
    "encyclopedia",
    "agent",
]


# ── Settings write ──────────────────────────────────────────────────────────
class SettingsUpdateRequest(BaseModel):
    """A partial settings write. Only whitelisted keys (SettingsStore.KNOWN_KEYS)
    are persisted; anything else is silently dropped by the core, and a None
    value means 'leave unchanged'. Mirrors SettingsResponse field-for-field."""

    model_alias: Optional[str] = None
    thinking: Optional[str] = None
    voice: Optional[str] = None
    voice_autoplay: Optional[bool] = None
    font_scale: Optional[str] = None
    theme: Optional[str] = None
    nanonis_host: Optional[str] = None
    nanonis_port_main: Optional[int] = None
    nanonis_port_monitor: Optional[int] = None
    nanonis_port_data: Optional[int] = None
    nanonis_port_emergency: Optional[int] = None
    qa_model: Optional[str] = None
    codex_live_search: Optional[bool] = None


class SettingsUpdateResponse(BaseModel):
    """Result of a settings write. ``persisted`` echoes the whitelisted subset
    actually stored. ``degraded`` is True when no SettingsStore is wired (the
    write was a no-op)."""

    ok: bool = False
    persisted: dict[str, Any] = Field(default_factory=dict)
    degraded: bool = False


# ── Admin overrides ─────────────────────────────────────────────────────────
class OverrideResponse(BaseModel):
    """The raw override payload for one category (the JSON the core merges over
    code defaults). ``data`` is empty when no override file exists OR when the
    live ConfigOverrideRegistry is not wired (``degraded`` then True)."""

    category: str
    data: dict[str, Any] = Field(default_factory=dict)
    has_override: bool = False
    degraded: bool = False


class OverrideWriteRequest(BaseModel):
    """A whole-file override payload to persist for one category. An empty
    ``data`` means 'reset to code defaults' (the core deletes the override file
    rather than persisting an empty stub)."""

    data: dict[str, Any] = Field(default_factory=dict)


class OverrideWriteResponse(BaseModel):
    """Result of an override write/restore.

    ``reloaded`` reports whether any in-process consumer actually re-derived from
    the new values — i.e. whether ``signal_reload`` had a subscriber to fire. It
    was a hardcoded ``True`` until 2026-08-03, while
    ``register_reload_hook`` had (and still has) zero subscribers in production:
    every override write claimed a hot-reload that never happened. Persisting and
    taking effect are different events and this model now reports them separately.

    ``restart_required`` True ⇒ persisted, but the running process is still using
    the old values. None ⇒ we could not determine it (no live consumer to compare
    against) — which must NOT be read as False.
    """

    ok: bool = False
    category: str
    reloaded: bool = False
    restart_required: bool | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    degraded: bool = False


class OverrideHistoryEntry(BaseModel):
    """One timestamped backup of an override file (config/overrides/_history)."""

    timestamp: str
    filename: str


class OverrideHistoryResponse(BaseModel):
    category: str
    entries: list[OverrideHistoryEntry] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


# ── Admin PIN unlock ────────────────────────────────────────────────────────
class PinUnlockRequest(BaseModel):
    """The raw PIN entered by the operator. The API only forwards it to the core
    comparison (SHA256-hex vs the launcher-written admin_pin.txt) — the raw PIN
    is never persisted, and the comparison/authoritative gate stays in core."""

    pin: str = Field(description="raw admin PIN (compared SHA256-hex against admin_pin.txt)")


class PinUnlockResponse(BaseModel):
    """Result of a PIN unlock attempt. On success ``token`` is a short-lived
    session token the frontend presents for subsequent admin writes. ``reason``
    explains a failure ('no_pin_set' | 'empty' | 'wrong' | 'degraded').
    ``degraded`` True when the PIN file / core gate is unavailable."""

    ok: bool = False
    token: Optional[str] = None
    reason: Optional[str] = None
    degraded: bool = False


# ── Environment sensors ─────────────────────────────────────────────────────
class SensorEntry(BaseModel):
    """One configured environment sensor. Mirrors environment/config.py's free
    JSON schema — type-specific fields (port/address/channel/baudrate/alarm) are
    carried through the open ``extra`` map so the contract never blocks a new
    sensor type."""

    id: str
    name: Optional[str] = None
    type: Optional[str] = None
    port: Optional[str] = None
    unit: Optional[str] = None
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="type-specific fields (address/channel/baudrate/alarm/...) passed through verbatim",
    )


class SensorsResponse(BaseModel):
    """The persisted sensor config. ``autodetect`` mirrors the config flag.
    ``degraded`` True when the environment config module can't be read."""

    autodetect: bool = True
    sensors: list[SensorEntry] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


class SensorWriteRequest(BaseModel):
    """Add or update one sensor. ``id`` identifies the target; unknown id with a
    ``type`` creates a new entry (mirrors environment.config.update_sensor).
    ``alarm`` and other type-specific fields ride in ``extra``."""

    id: str
    name: Optional[str] = None
    type: Optional[str] = None
    port: Optional[str] = None
    unit: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SensorWriteResponse(BaseModel):
    ok: bool = False
    sensor: Optional[SensorEntry] = None
    degraded: bool = False


class SensorDeleteResponse(BaseModel):
    ok: bool = False
    removed: bool = False
    degraded: bool = False


class SensorRescanResponse(BaseModel):
    """Result of a serial-port rescan. ``count`` is the number of sensors now
    monitored. ``degraded`` True when the environment monitor isn't wired (the
    rescan touches the exclusive serial bus, which only the live app owns)."""

    ok: bool = False
    count: int = 0
    sensors: list[SensorEntry] = Field(default_factory=list)
    degraded: bool = False


# ── Device discovery + read-only instrument state ───────────────────────────
# Backing the「扫描设备接口」flow: scan → show what was identified → the user
# confirms → adopt into the persisted config → found automatically on every
# later boot. Discovery itself writes NOTHING; adoption is the separate step.


class DiscoveredChannel(BaseModel):
    """One sensor input. ``label`` is the instrument's own front-panel name."""

    channel: str
    label: str = ""
    kelvin: Optional[float] = None
    celsius: Optional[float] = None
    sensor_value: Optional[float] = None
    sensor_type: str = ""
    faults: Optional[list[str]] = Field(
        default=None,
        description=(
            "decoded RDGST? flags — THREE-state. [] = the instrument answered "
            "and every fault bit is clear. Non-empty = it answered and the "
            "reading is NOT trustworthy. null = the status query never "
            "answered, so NOTHING is known about this reading's validity; it "
            "is not 'clean'. Read `health` rather than testing this field, "
            "since `faults ?? []` silently reads null as clean."
        ),
    )
    health: str = Field(
        default="unknown",
        description=(
            "clean | faulty | unknown — the verdict on `faults`, decided once "
            "in the backend so no client re-derives it. Defaults to 'unknown' "
            "so an older/partial payload never claims a channel is healthy."
        ),
    )


class DiscoveredOutput(BaseModel):
    """One control output — the heater readout.

    ``heater_on`` derives from the range code alone: range 0 is off regardless
    of setpoint or control mode, so a parked setpoint never reads as heating.
    """

    output: int
    range_code: int = -1
    range_label: str = ""
    heater_on: bool = False
    heater_pct: Optional[float] = None
    setpoint: Optional[float] = None
    mode: str = ""
    control_input: str = ""
    powerup_enabled: bool = False
    ramping: bool = False
    ramp_rate: Optional[float] = None


class InstrumentState(BaseModel):
    """Everything MAST can read off an instrument without writing to it."""

    port: str = ""
    kind: str = ""
    model: str = ""
    idn: str = ""
    firmware: str = ""
    serial_number: str = ""
    any_heater_on: bool = False
    channels: list[DiscoveredChannel] = Field(default_factory=list)
    outputs: list[DiscoveredOutput] = Field(default_factory=list)


class DiscoveredDevice(InstrumentState):
    """One scanned port. Unidentified ports are reported too (``identified``
    False) so the dialog can say what it looked at and found nothing on."""

    description: str = ""
    hwid: str = ""
    identified: bool = False
    already_configured: bool = False
    suggested_sensors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="ready-to-adopt config entries, one per input, named from INNAME?",
    )


class DiscoverResponse(BaseModel):
    devices: list[DiscoveredDevice] = Field(default_factory=list)
    identified_count: int = 0
    degraded: bool = False


class InstrumentStateResponse(BaseModel):
    """Live state of the instruments MAST is currently connected to."""

    instruments: list[InstrumentState] = Field(default_factory=list)
    degraded: bool = False


class SensorAdoptRequest(BaseModel):
    """Adopt the entries the user ticked in the confirmation dialog.

    ``sensors`` are ``suggested_sensors`` entries from GET /environment/discover
    (possibly renamed by the user). Persisting them is what makes the port stick
    across restarts."""

    sensors: list[dict[str, Any]] = Field(default_factory=list)


class SensorAdoptResponse(BaseModel):
    ok: bool = False
    adopted: int = 0
    rejected: list[str] = Field(default_factory=list)
    sensors: list[SensorEntry] = Field(default_factory=list)
    live_count: int = 0
    degraded: bool = False


# ── Nanonis hardware ────────────────────────────────────────────────────────
class NanonisConnectRequest(BaseModel):
    """Optional connection overrides. When omitted the core uses the persisted
    Nanonis host/ports. Force-kill of a live TCP corrupts the port permanently,
    so the core always reconnects gracefully — never the API."""

    host: Optional[str] = None
    port_main: Optional[int] = None
    port_monitor: Optional[int] = None
    port_data: Optional[int] = None
    port_emergency: Optional[int] = None


class NanonisConnectAccepted(BaseModel):
    """202 ack for an async connect. The connect runs in a background worker
    (graceful, multi-port); the frontend polls GET /api/nanonis/connection for
    the outcome. ``degraded`` True when no connection pool is wired (the request
    was accepted but cannot actually run here)."""

    accepted: bool = True
    status: Literal["connecting", "unavailable"] = "connecting"
    degraded: bool = False


class NanonisPortStatus(BaseModel):
    role: str = Field(description="main | monitor | data | emergency")
    port: Optional[int] = None
    connected: bool = False
    detail: Optional[str] = None


class NanonisConnectionResponse(BaseModel):
    """Current Nanonis connection snapshot across the four TCP roles.
    ``degraded`` True when no connection pool is wired (all ports report
    disconnected)."""

    connected: bool = False
    host: Optional[str] = None
    ports: list[NanonisPortStatus] = Field(default_factory=list)
    degraded: bool = False
    # 最近一次连接后台 worker 的结果。
    # None 表示尚未发起；connecting 表示等待结果；ok:N 表示连上 N 个端口；
    # failed: <原因> 表示失败。必须显式上报失败，不能让它与仍在连接不可区分。
    last_connect_result: Optional[str] = None
    #: 那次结局发生的 Unix 时间戳（秒）。配合上面那条判断「是不是刚试过」。
    last_connect_at: Optional[float] = None


# ── System self-check ───────────────────────────────────────────────────────
class SystemCheckItem(BaseModel):
    """One self-check row (mirrors dashboard.run_system_check result dicts)."""

    name: str
    status: Literal["ok", "warning", "error", "unavailable"] = "unavailable"
    detail: str = ""


class SystemCheckResponse(BaseModel):
    """Aggregate self-check. ``degraded`` True when no live app is wired (the
    check needs the connection pool / storage / monitor singletons)."""

    items: list[SystemCheckItem] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


# ── Context injection (上下文注入) ───────────────────────────────────────────
# The inventory of everything MAST injects into an agent's context, plus the
# operator override layer and the real-request capture ring. See mast/prompts/.
class PromptSummary(BaseModel):
    """One inventory row — metadata only, no full body (the list stays small).

    ``availability`` is the honest bit:
      * ``static``          — a constant; ``preview``/detail show it verbatim
      * ``live``            — rendered now from CURRENT persisted state (real)
      * ``needs_hardware``  — needs a Nanonis read; NOT rendered, reason given
      * ``needs_request``   — per-request only; NOT rendered, reason given
    A non-renderable row carries an EMPTY body and a reason. It is never filled
    with a sample — an invented block would be debugged as if it were real.
    """

    id: str
    label: str
    category: Literal["agent_system", "routing", "middleware", "sub_llm"] = "middleware"
    agent: str = ""
    availability: Literal["static", "live", "needs_hardware", "needs_request"] = "static"
    source: str = ""
    note: str = ""
    overridable: bool = False
    overridden: bool = False
    default_chars: int = 0
    effective_chars: int = 0
    preview: str = ""
    unavailable_reason: str = ""


class PromptListResponse(BaseModel):
    items: list[PromptSummary] = Field(default_factory=list)
    count: int = 0
    overridden_count: int = 0
    degraded: bool = False


class PromptDetail(PromptSummary):
    """Full bodies for one entry. ``effective_text`` is what the agent gets."""

    default_text: str = ""
    override_text: Optional[str] = None
    effective_text: str = ""
    degraded: bool = False


class PromptOverrideRequest(BaseModel):
    """Replacement text. Empty/whitespace clears the override (back to default)."""

    text: str = ""


class PromptOverrideResponse(BaseModel):
    ok: bool = False
    prompt_id: str = ""
    overridden: bool = False
    effective_chars: int = 0
    message: str = ""
    degraded: bool = False


class CapturedMessageModel(BaseModel):
    """One message as the provider received it. ``chars`` is the length BEFORE
    truncation, so a clipped body can never be misread as a short prompt."""

    role: str = ""
    content: str = ""
    chars: int = 0
    truncated: bool = False
    #: 这条消息里各段的来源（``mast.prompts.ledger`` 的块清单）。
    #: ``None`` = 查不到归属 —— 与「没有注入」是两件事。
    blocks: Optional[list[dict]] = None


class PromptCaptureSummary(BaseModel):
    """Metadata for one captured real request (newest first, index 0)."""

    index: int = 0
    #: Stable id. ``index`` shifts every time a model call lands — fetch detail
    #: by ``seq`` from anywhere that can race with a running agent.
    seq: int = 0
    ts: float = 0.0
    age_s: float = 0.0
    source: str = ""
    model_id: str = ""
    provider: str = ""
    message_count: int = 0
    total_chars: int = 0
    system_chars: int = 0


class PromptCaptureListResponse(BaseModel):
    """The capture ring. Empty ``items`` means no model call has run yet in this
    process — NOT that nothing is injected."""

    items: list[PromptCaptureSummary] = Field(default_factory=list)
    count: int = 0
    enabled: bool = True
    total_seen: int = 0
    capacity: int = 0
    note: str = ""
    degraded: bool = False


class PromptCaptureDetail(BaseModel):
    """The full message list of one captured request."""

    index: int = 0
    seq: int = 0
    ts: float = 0.0
    age_s: float = 0.0
    source: str = ""
    model_id: str = ""
    provider: str = ""
    messages: list[CapturedMessageModel] = Field(default_factory=list)
    total_chars: int = 0
    dropped_messages: int = 0
    found: bool = False
    degraded: bool = False


class PromptCaptureClearResponse(BaseModel):
    ok: bool = False
    cleared: int = 0


__all__ = [
    "SafetyLimits",
    "OverrideCategory",
    "PromptSummary",
    "PromptListResponse",
    "PromptDetail",
    "PromptOverrideRequest",
    "PromptOverrideResponse",
    "CapturedMessageModel",
    "PromptCaptureSummary",
    "PromptCaptureListResponse",
    "PromptCaptureDetail",
    "PromptCaptureClearResponse",
    "SettingsUpdateRequest",
    "SettingsUpdateResponse",
    "OverrideResponse",
    "OverrideWriteRequest",
    "OverrideWriteResponse",
    "OverrideHistoryEntry",
    "OverrideHistoryResponse",
    "PinUnlockRequest",
    "PinUnlockResponse",
    "SensorEntry",
    "SensorsResponse",
    "SensorWriteRequest",
    "SensorWriteResponse",
    "SensorDeleteResponse",
    "SensorRescanResponse",
    "DiscoveredChannel",
    "DiscoveredOutput",
    "InstrumentState",
    "DiscoveredDevice",
    "DiscoverResponse",
    "InstrumentStateResponse",
    "SensorAdoptRequest",
    "SensorAdoptResponse",
    "NanonisConnectRequest",
    "NanonisConnectAccepted",
    "NanonisPortStatus",
    "NanonisConnectionResponse",
    "SystemCheckItem",
    "SystemCheckResponse",
]
