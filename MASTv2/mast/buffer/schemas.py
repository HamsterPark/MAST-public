"""Buffer layer Pydantic schemas — s-scale handshake between Vision and Agents.

From compass §3.1. All models are frozen (immutable), enabling safe sharing
across threads and cheap equality checks.

Producer (Vision thread) creates these; Consumer (agent coroutines) reads them
via the buffer_tools (read_latest_tip_status, get_scan_progress, etc.).

Seqno: monotonic per stream (TipStatus / RegionMap / ScanProgress independent).
t_mono_ns: time.monotonic_ns() at producer-side creation time. Preferred over
wall-clock for cross-thread ordering.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now_ns() -> int:
    return time.monotonic_ns()


def _uid() -> str:
    return uuid.uuid4().hex


# ─────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────

class TipQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    BAD = "bad"
    UNKNOWN = "unknown"


class VisionEventType(str, Enum):
    TIP_QUALITY_DROP = "tip_quality_drop"
    FEATURE_OF_INTEREST = "feature_of_interest"
    SCAN_COMPLETE = "scan_complete"
    SENSOR_FAULT = "sensor_fault"
    EMERGENCY_RETRACT_NEEDED = "emergency_retract_needed"
    # Compass §3.1 additions — full 6-event coverage required by guide
    # (kept generic; payload shape enforced by factory helpers below).
    VISION_ERROR = "vision_error"
    SETPOINT_CHANGE = "setpoint_change"
    E_STOP = "e_stop"
    # instrument-control tip-shaping outcome (current/z readback three-step verdict)
    TIP_SHAPE_VERDICT = "tip_shape_verdict"


# Allowed enumerations for factory payloads (mirrors compass §3.1 Literal sets)
VISION_ERROR_KINDS = ("cuda_oom", "model_error", "preprocess_fail", "context_loss")
SETPOINT_SOURCES = ("user", "planner", "monitor")
#: 每加一个发起方,都要在这里加它的名字 —— 否则 :func:`make_e_stop` 抛
#: ValueError,而**每个调用点都用宽 except 包着**,于是急停会被静默丢掉。
#: 这个形状已经炸过两次:
#:   ``"operator"`` 被拒 ⇒ 用户按下的急停没有事件;
#:   ``"environment"`` 被拒 ⇒ 环境告警**退了针、挂了急停闩**,
#:               而唯一会告诉人的那条通道把自己拒了 —— 用户看到的是
#:               「什么都没发生」,仪器却已经锁死。
#: 白名单与调用点的一致性由 test_e_stop_reasons_are_all_allowed 结构闸门看着。
E_STOP_REASONS = ("user", "watchdog", "temperature", "vacuum", "force",
                  "environment")


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


# ─────────────────────────────────────────────────────────────────────
# Buffer payloads
# ─────────────────────────────────────────────────────────────────────

class TipStatus(BaseModel):
    """Point-in-time tip assessment pushed from vision thread."""
    model_config = ConfigDict(frozen=True)

    seqno: int
    t_mono_ns: int = Field(default_factory=_now_ns)
    quality: TipQuality
    confidence: float = Field(ge=0, le=1)
    embedding_sha: str | None = None            # identifies which backbone output
    scan_id: str
    frame_idx: int
    #: True when SAFE mode produced this verdict rather than the vision model —
    #: `quality` reads "good" and `confidence` is a rewritten number, so a reader
    #: (WAL forensics, the GUI, a later analysis) can tell a mode from a
    #: measurement. Deliberately a BOOLEAN and not the raw verdict: this object
    #: is `model_dump()`ed straight into six agents' context by
    #: `buffer_tools.read_latest_tip_status`, and carrying the real "bad" here
    #: would hand back the very argument SAFE exists to remove.
    safe_mode: bool = False


class RegionMap(BaseModel):
    """Segmentation mask for a scan, serialized as run-length-encoded bytes.

    Mask IS bytes, not an array — safe to serialize through LangGraph state.
    Decode with mast.vision.seg_utils.decode_rle(mask_rle, shape).
    """
    model_config = ConfigDict(frozen=True)

    seqno: int
    scan_id: str
    mask_rle: bytes
    shape: tuple[int, int]
    t_mono_ns: int = Field(default_factory=_now_ns)


class ScanProgress(BaseModel):
    """Line-by-line scan acquisition progress."""
    model_config = ConfigDict(frozen=True)

    seqno: int
    scan_id: str
    line_idx: int
    lines_total: int
    eta_s: float
    t_mono_ns: int = Field(default_factory=_now_ns)


class VisionEvent(BaseModel):
    """Edge-triggered event published to subscriber queues.

    Examples:
      TIP_QUALITY_DROP    — emitted on rising edge good → degraded/bad
      FEATURE_OF_INTEREST — defect cluster / molecule detected in segmentation
      SCAN_COMPLETE       — last line acquired, file closed
      SENSOR_FAULT        — unrecoverable error from a vision head
      EMERGENCY_RETRACT_NEEDED — race-pattern wake-up for orchestrator
    """
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=_uid)
    seqno: int
    kind: VisionEventType
    severity: Severity
    payload: dict[str, Any] = Field(default_factory=dict)
    cause_ref: str | None = None                # e.g., f"tip_status#{seqno}"
    t_mono_ns: int = Field(default_factory=_now_ns)


# ─────────────────────────────────────────────────────────────────────
# Factory helpers — convenience constructors that validate payload shape
# for the typed event kinds (compass §3.1). The generic VisionEvent stays
# the only Pydantic model; helpers just enforce well-known keys / enums
# at producer side so consumers can trust `ev.payload["error_kind"]` etc.
# ─────────────────────────────────────────────────────────────────────

def _coerce_severity(sev: Severity | str) -> Severity:
    if isinstance(sev, Severity):
        return sev
    return Severity(sev)


def make_vision_error(
    error_kind: str,
    detail: str,
    *,
    seqno: int,
    severity: Severity | str = Severity.CRITICAL,
    cause_ref: str | None = None,
    **extra: Any,
) -> VisionEvent:
    """Build a VISION_ERROR VisionEvent. error_kind must be one of
    {cuda_oom, model_error, preprocess_fail, context_loss}. `detail` is
    truncated to 1024 chars (compass §3.1 VisionError schema cap).
    """
    if error_kind not in VISION_ERROR_KINDS:
        raise ValueError(
            f"error_kind={error_kind!r} not in {VISION_ERROR_KINDS}"
        )
    payload: dict[str, Any] = {
        "error_kind": error_kind,
        "detail": str(detail)[:1024],
    }
    payload.update(extra)
    return VisionEvent(
        seqno=seqno,
        kind=VisionEventType.VISION_ERROR,
        severity=_coerce_severity(severity),
        payload=payload,
        cause_ref=cause_ref,
    )


def make_setpoint_change(
    bias_v: float | None,
    current_a: float | None,
    source: str,
    *,
    seqno: int,
    severity: Severity | str = Severity.INFO,
    cause_ref: str | None = None,
    **extra: Any,
) -> VisionEvent:
    """Build a SETPOINT_CHANGE VisionEvent. source must be one of
    {user, planner, monitor}. bias_v/current_a may be None (partial change).
    """
    if source not in SETPOINT_SOURCES:
        raise ValueError(f"source={source!r} not in {SETPOINT_SOURCES}")
    payload: dict[str, Any] = {
        "bias_v": bias_v,
        "current_a": current_a,
        "source": source,
    }
    payload.update(extra)
    return VisionEvent(
        seqno=seqno,
        kind=VisionEventType.SETPOINT_CHANGE,
        severity=_coerce_severity(severity),
        payload=payload,
        cause_ref=cause_ref,
    )


def make_tip_shape_verdict(
    verdict: str,
    delta_nm: float,
    *,
    seqno: int,
    file_path: str | None = None,
    severity: Severity | str = Severity.INFO,
    cause_ref: str | None = None,
    **extra: Any,
) -> VisionEvent:
    """Build a TIP_SHAPE_VERDICT VisionEvent from a TipShapeWithReadback result.

    ``verdict`` ∈ {no_change, cluster, tip_changed_or_pit, insufficient_data}.
    ``file_path`` (the rendered z/current PNG) makes the Vision Buffer tab show a
    thumbnail. Keep the payload small — the full traces live in the PNG / records,
    NOT here (no tensors/arrays in the buffer).
    """
    payload: dict[str, Any] = {
        "verdict": str(verdict),
        "delta_nm": float(delta_nm),
    }
    if file_path:
        payload["file_path"] = str(file_path)
    payload.update(extra)
    return VisionEvent(
        seqno=seqno,
        kind=VisionEventType.TIP_SHAPE_VERDICT,
        severity=_coerce_severity(severity),
        payload=payload,
        cause_ref=cause_ref,
    )


def make_e_stop(
    reason: str,
    detail: str,
    *,
    seqno: int,
    severity: Severity | str = Severity.CRITICAL,
    cause_ref: str | None = None,
    **extra: Any,
) -> VisionEvent:
    """Build an E_STOP VisionEvent. reason must be one of
    {user, watchdog, temperature, vacuum, force}. E-stop is terminal —
    severity defaults to CRITICAL.
    """
    if reason not in E_STOP_REASONS:
        raise ValueError(f"reason={reason!r} not in {E_STOP_REASONS}")
    payload: dict[str, Any] = {
        "reason": reason,
        "detail": str(detail)[:1024],
    }
    payload.update(extra)
    return VisionEvent(
        seqno=seqno,
        kind=VisionEventType.E_STOP,
        severity=_coerce_severity(severity),
        payload=payload,
        cause_ref=cause_ref,
    )


__all__ = [
    "TipQuality", "VisionEventType", "Severity",
    "TipStatus", "RegionMap", "ScanProgress", "VisionEvent",
    "VISION_ERROR_KINDS", "SETPOINT_SOURCES", "E_STOP_REASONS",
    "make_vision_error", "make_setpoint_change", "make_e_stop",
    "make_tip_shape_verdict",
]
