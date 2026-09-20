"""Signal-channels endpoint for the 信号捕获 + FFT redesign (实验性功能 tab).

``GET /api/experimental/signals`` lists the acquirable Nanonis signal channels
(name + index + unit) the live instrument exposes, the Osci1T oscilloscope
timebase options, and the static FFT enums (window functions + output modes)
the UI renders. This is the read-only metadata seam the new TypeScript capture
tab calls before a capture; the actual trace acquisition + FFT stays on the IC
skills / capture path (relay only — NO new acquisition logic here).

House style (mirrors routes/admin.py, routes/records.py):
  - the handler takes ``(request: Request)`` and reads ``ctx =
    request.app.state.ctx``;
  - JSON handler has a ``response_model``;
  - GRACEFUL DEGRADATION is mandatory — this app must boot standalone with no
    live core wired. When the connection pool / registry / state are absent (or
    any call raises) the endpoint returns the SENSIBLE DEFAULT channel list with
    ``degraded=True`` / ``source="default"`` — never a 500;
  - heavy backends (ExecutionContext + the IC skills) are LAZY-imported inside
    the handler in try/except;
  - the live signal names are RELAYED from the existing ``ListSignalChannels`` /
    ``GetOsciTimebases`` skills — the API adds no hardware logic of its own.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from mast.api.schemas_signals import (
    OutputModeOption,
    SignalChannel,
    SignalChannelsResponse,
    TimebaseOption,
    WindowOption,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["signals"])


# ── Static UI enums (the FFT runs over an already-captured trace, so these are
# pure presentation; mirrors the old gui/exp_capture.py dropdowns) ────────────
_WINDOWS: list[WindowOption] = [
    WindowOption(value="hann", label="Hann"),
    WindowOption(value="hamming", label="Hamming"),
    WindowOption(value="rect", label="矩形/无窗 (Rect)"),
]
_OUTPUT_MODES: list[OutputModeOption] = [
    OutputModeOption(value="magnitude", label="幅度 |FFT|"),
    OutputModeOption(value="power", label="功率谱 PSD"),
]


def _is_current_name(name: str) -> bool:
    """Heuristic current-channel detector (same rule as skills/builtins/signals.py
    ``_is_current_name``): Nanonis exposes tunnelling current under names that
    start with 'current'."""
    return (name or "").strip().lower().startswith("current")


# Sensible degraded default list — the canonical Nanonis V5e signal names for
# the low channels the UI most needs, so the capture tab still renders a usable
# dropdown with no live instrument. Indices 0-127 exist on hardware; here we
# expose the common named inputs (current = amperes) and leave the rest to a
# live refresh.
_DEFAULT_CHANNEL_NAMES: list[str] = [
    "Current (A)",
    "Bias (V)",
    "Z (m)",
    "Current 2 (A)",
    "LI Demod 1 X (A)",
    "LI Demod 1 Y (A)",
    "LI Demod 2 X (A)",
    "LI Demod 2 Y (A)",
    "Input 1 (V)",
    "Input 2 (V)",
    "Input 3 (V)",
    "Input 4 (V)",
    "Output 1 (V)",
    "Output 2 (V)",
    "Frequency Shift (Hz)",
    "Excitation (V)",
]


def _unit_for(name: str) -> str | None:
    """Best-effort physical unit off the Nanonis display name's trailing (unit).

    Current channels are amperes; otherwise parse a trailing ``(X)`` suffix
    (e.g. 'Bias (V)' → 'V'). Unknown ⇒ None (the UI shows the raw value)."""
    if _is_current_name(name):
        return "A"
    n = (name or "").strip()
    if n.endswith(")") and "(" in n:
        unit = n[n.rindex("(") + 1 : -1].strip()
        return unit or None
    return None


def _default_channels() -> list[SignalChannel]:
    out: list[SignalChannel] = []
    for i, name in enumerate(_DEFAULT_CHANNEL_NAMES):
        out.append(
            SignalChannel(
                index=i,
                name=name,
                unit=_unit_for(name),
                is_current=_is_current_name(name),
            )
        )
    return out


def _degraded_response(detail: str | None = None) -> SignalChannelsResponse:
    """The standalone / no-hardware payload: the static default channel list +
    the static FFT enums, flagged ``degraded`` so the UI knows it isn't live."""
    channels = _default_channels()
    return SignalChannelsResponse(
        channels=channels,
        n_channels=len(channels),
        current_indices=[c.index for c in channels if c.is_current],
        timebases=[],
        current_timebase_index=None,
        osci_available=False,
        windows=_WINDOWS,
        output_modes=_OUTPUT_MODES,
        source="default",
        degraded=True,
        detail=detail,
    )


def _execution_context(ctx: Any):
    """Best-effort one-shot ExecutionContext from the live singletons wired onto
    the context at integration (pool / state / registry). Returns None when any
    is absent — we then degrade rather than touch hardware from a half-wired
    process. Heavy import is lazy + guarded."""
    pool = getattr(ctx, "connection_pool", None)
    state = getattr(ctx, "state", None) or getattr(ctx, "instrument_state", None)
    registry = getattr(ctx, "skill_registry", None) or getattr(ctx, "registry", None)
    if pool is None or state is None or registry is None:
        # 说出**缺的是哪一个**。2026-08-02 实机上 ``ctx.state`` 因为 bootstrap 漏
        # 了一行而恒为 None，调用方却把这一律报成「nanonis not wired」——而当时
        # Nanonis 连得好好的（电流监控正在同一个池上轮询）。一句方向指错的诊断
        # 比没有诊断更费时间。
        missing = [n for n, v in (("connection_pool", pool), ("state", state),
                                  ("skill_registry", registry)) if v is None]
        logger.warning("signals: ExecutionContext 不可用，ctx 上缺少 %s",
                       "/".join(missing))
        return None
    try:
        from mast.core.execution_context import ExecutionContext

        # Thread the process-wide stop event in. Without it this context got a
        # PRIVATE, dead Event: a long signal capture started from here could not
        # be aborted at all, and E_STOP could not touch it either (2026-07-11).
        app = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
        abort = getattr(app, "_orch_abort", None)
        # owner: this is the THIRD independent driver of the one instrument
        # (group chat, private chat, here). Naming it means a refused caller is
        # told which entry point is holding the instrument token, instead of a
        # bare "busy" (审计 致命一).
        return ExecutionContext(pool=pool, state=state, registry=registry,
                                abort_event=abort, owner="信号采集 API")
    except Exception as exc:  # noqa: BLE001 — degrade, never crash
        logger.warning("signals: ExecutionContext build failed: %s", exc)
        return None


# ── GET /api/experimental/signals ─────────────────────────────────────────────
@router.get("/experimental/signals", response_model=SignalChannelsResponse)
def list_signal_channels(request: Request) -> SignalChannelsResponse:
    """List the acquirable signal channels + Osci timebases + FFT enums.

    Relays the live ``ListSignalChannels`` / ``GetOsciTimebases`` IC skills when
    a Nanonis connection is wired; otherwise returns the sensible degraded
    default list (``degraded=True``, ``source="default"``). The Osci timebases
    degrade independently — the channel list can be live while the Osci1T module
    is unloaded (then ``osci_available=False``, ``timebases=[]``)."""
    ctx = request.app.state.ctx
    ec = _execution_context(ctx)
    if ec is None:
        # No live core wired → standalone default list, never a 500.
        # 「nanonis not wired」在 2026-08-02 的实机上是**假话**：Nanonis 连着，
        # 缺的是 ctx 上的某个单例（当时是 state）。别让一句诊断把人往连接问题上引。
        return _degraded_response("内核单例未接线（见服务日志 signals: 那一行），"
                                  "与 Nanonis 连接状态无关")

    # ── Channels (Signals_NamesGet via ListSignalChannels) ──
    channels: list[SignalChannel] = []
    current_indices: list[int] = []
    declared_n: int | None = None
    truncated = False
    source = "default"
    degraded = True
    detail: str | None = None
    try:
        res = ec.run("ListSignalChannels", {})
        if getattr(res, "success", False):
            data = res.data or {}
            raw = data.get("channels") or []
            for c in raw:
                name = str(c.get("name", ""))
                idx = int(c.get("index"))
                channels.append(
                    SignalChannel(
                        index=idx,
                        name=name,
                        unit=_unit_for(name),
                        is_current=_is_current_name(name),
                    )
                )
            current_indices = [int(i) for i in (data.get("current_indices") or [])]
            declared_n = data.get("declared_n")
            truncated = bool(data.get("truncated"))
            source = "live"
            degraded = False
        else:
            detail = getattr(res, "error", None) or "ListSignalChannels failed"
    except Exception as exc:  # noqa: BLE001 — relay failure ⇒ degrade
        logger.warning("signals: ListSignalChannels relay failed: %s", exc)
        detail = f"{type(exc).__name__}: {exc}"

    if not channels:
        # Live core present but the read failed/empty → still hand back the
        # default list so the UI renders, flagged degraded.
        return _degraded_response(detail or "no channels returned")

    # ── Osci1T timebases (optional; module may be unloaded) ──
    timebases: list[TimebaseOption] = []
    current_tb: int | None = None
    osci_available = False
    try:
        res_tb = ec.run("GetOsciTimebases", {})
        if getattr(res_tb, "success", False):
            tdata = res_tb.data or {}
            for t in tdata.get("timebases") or []:
                timebases.append(
                    TimebaseOption(
                        index=int(t.get("index")),
                        dt_s=float(t.get("dt_s", 0.0) or 0.0),
                        fs_hz=float(t.get("fs_hz", 0.0) or 0.0),
                    )
                )
            cti = tdata.get("current_index")
            current_tb = int(cti) if cti is not None else None
            osci_available = bool(timebases)
    except Exception as exc:  # noqa: BLE001 — Osci optional; channels still live
        logger.info("signals: GetOsciTimebases unavailable: %s", exc)

    return SignalChannelsResponse(
        channels=channels,
        n_channels=len(channels),
        declared_n=declared_n,
        truncated=truncated,
        current_indices=current_indices,
        timebases=timebases,
        current_timebase_index=current_tb,
        osci_available=osci_available,
        windows=_WINDOWS,
        output_modes=_OUTPUT_MODES,
        source=source,
        degraded=degraded,
        detail=detail,
    )
