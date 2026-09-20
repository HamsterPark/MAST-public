"""电流监控 — tunnelling-current monitor endpoints.

Thin relay over ``mast.monitoring``: the store and the daemon do the work, these
handlers only shape it. House style follows routes/usage.py + routes/signals.py:

  - every handler is wrapped so it returns ``degraded=True`` instead of a 500 —
    the app must boot standalone with no core wired, and the monitoring package
    may not even be importable;
  - the monitoring backend is LAZY-imported inside each handler;
  - ZERO TCP. Everything here reads the SQLite store or the daemon's in-memory
    status. The whole point of the monitor is that the UI can watch the current
    without adding traffic to the fragile Nanonis ports;
  - waveforms are decimated by the STORE before they reach here — raw ``.npy``
    never goes over the wire;
  - literal paths are registered before ``{seg_id}`` paths, or the path
    parameter shadows them.

``/status`` doubles as the REST fallback for the WebSocket feed: the event bus
has no HTTP replay route of its own, so a client that loses the socket polls
here (the same shape ``/api/hardware/live-readings`` has for the top bar).
"""
from __future__ import annotations

import base64
import logging
from typing import Callable, Optional, TypeVar

from fastapi import APIRouter

from mast.api.schemas_monitoring import (
    AlertAckResponse,
    AlertEvidenceResponse,
    AlertRow,
    AlertsResponse,
    AuxChannelState,
    AuxSeriesResponse,
    AuxSnapshot,
    BaselineActivateRequest,
    BaselineActivateResult,
    BaselineCheckResponse,
    BaselineCurveResponse,
    BaselineDetailResponse,
    BaselineListResponse,
    BaselinePoint,
    BaselineRow,
    FeatureRow,
    FeatureSeriesResponse,
    LabelRequest,
    LabelResult,
    LatestFeature,
    LiveTraceResponse,
    MonitoringConfigResponse,
    MonitoringControlResult,
    MonitoringStatus,
    PinRequest,
    PinResult,
    SegmentDataResponse,
    SegmentListResponse,
    SegmentRow,
    ThresholdKnob,
)
from mast.api.schemas_vision import FFTResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])

T = TypeVar("T")

#: Feature columns promoted into the flat ``metrics`` map. Everything the
#: extractor produces except bookkeeping and the context labels.
_SKIP_METRIC_KEYS = frozenset({
    "segment_id", "t_start", "alert_level", "extra_json", "ctx_skill",
    "ctx_scanning", "ctx_bias_v", "ctx_setpoint_a", "ctx_z_m", "ctx_zctrl_on",
    "ctx_stale",
})


def _guarded(fn: Callable[[], T], degraded_factory: Callable[[str], T]) -> T:
    """Run a handler body; on any failure return its degraded shape."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — an endpoint must never 500 here
        logger.debug("monitoring endpoint degraded", exc_info=True)
        return degraded_factory(str(exc))


def _store():
    from mast.monitoring.store import get_store
    return get_store()


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _opt_num(v) -> Optional[float]:
    """Float, or None. Never 0.0 for "missing" — see AuxSeriesResponse."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _opt_bool(v) -> Optional[bool]:
    """Tri-state: None stays None. "Never observed" is not "False"."""
    return None if v is None else bool(v)


def _metrics(row: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in row.items():
        if k in _SKIP_METRIC_KEYS or v is None:
            continue
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


#: Longest live-current window the chart may ask for.
#:
#: Was 600 s, but a longer real-time current window was needed, and the data was never the
#: constraint: ``segments.envelope`` rows are permanent (only the raw ``.npy``
#: is swept at ``cm_keep_hours``), so an hours-long window has always been
#: readable. The 600 was a guard against the cost of expanding every envelope
#: point into a Python float, which ``store.live_tail`` now avoids by reducing
#: inside each segment first.
#:
#: Six hours, not 24: at this ceiling one output point already covers ~18 s of
#: wall clock, and a "live" chart that spans a whole day is a history browser
#: wearing the wrong label — that is what the 段浏览 and 环境历史 pages are for.
_MAX_TRACE_WINDOW_S: float = 21600.0

#: Default chart series: the raw value of each channel — what the operator
#: watches. Derived statistics are opt-in via ``columns`` (the commissioning
#: tool asks for those once over many rows; the chart polls these three every
#: few seconds, and shipping all eighteen columns on that cadence would be
#: ~150 KB a poll for numbers nobody is drawing).
#: What ``/aux/series`` returns when the caller names no columns — one raw value
#: per channel, plus the lock-in modulation flag.
#:
#: ``lockin_mod_on`` rides along with ``lockin_a`` rather than being opt-in
#: because it is not extra detail about that curve, it is **what that curve is**:
#: with modulation off the demodulator puts out crosstalk and noise, same unit
#: and same order of magnitude as a real dI/dV. A caller that gets the values
#: without the flag cannot tell the two apart, and the cheap reading is always
#: "it is dI/dV" — dI/dV must indicate, in some form, whether lock-in is on.
_AUX_SERIES_COLUMNS: tuple[str, ...] = (
    "z_m", "amp_m", "df_hz", "lockin_a", "lockin_mod_on",
)

def _aux_selectable() -> frozenset[str]:
    """Everything a caller may request from ``aux_samples``.

    A whitelist, not a filter over the row: an unchecked column name would let a
    query name ``rules`` or ``extra_json`` and get a string where the schema
    promises floats.

    DERIVED from the writer's own column list rather than hand-copied. The copy
    was already drifting — adding the lock-in channel meant editing four lists
    in three files, and the one that silently does nothing when you forget it is
    this one: the column is recorded, the chart asks for it, and the endpoint
    quietly drops the name and returns the defaults instead. Same shape as
    ``SettingsStore.KNOWN_KEYS`` and ``override_store._ALL_FILES``, which this
    repo has paid for three times, and the same fix ``store._aux_column_spec()``
    already uses one layer down.
    """
    try:
        from mast.monitoring.aux_channels import AUX_METRIC_COLUMNS
        return frozenset(AUX_METRIC_COLUMNS)
    except Exception:  # noqa: BLE001 — monitoring is an optional install
        logger.debug("aux column list unavailable (swallowed)", exc_info=True)
        return frozenset(_AUX_SERIES_COLUMNS)


_AUX_SELECTABLE: frozenset[str] = _aux_selectable()


def _aux_snapshot(raw) -> Optional[AuxSnapshot]:
    """Shape the daemon's aux dict into the response model.

    ``None`` in, ``None`` out: no daemon means no auxiliary sampler, which is a
    different statement from "this rig has no Z channel" (that one lives in
    ``channels[].available``). Conflating them would make a stopped monitor look
    like missing hardware.
    """
    if not isinstance(raw, dict):
        return None
    channels = [
        AuxChannelState(
            kind=str(c.get("kind") or ""),
            label_zh=str(c.get("label_zh") or ""),
            unit=str(c.get("unit") or ""),
            available=bool(c.get("available")),
            signal_index=int(c.get("signal_index", -1)),
            signal_name=str(c.get("signal_name") or ""),
            judged=bool(c.get("judged")),
            value=c.get("value"),
            ts=c.get("ts"),
            verdict=str(c.get("verdict") or "unknown"),
            note=str(c.get("note") or ""),
            metrics={k: float(v) for k, v in (c.get("metrics") or {}).items()},
        )
        for c in (raw.get("channels") or [])
        if isinstance(c, dict)
    ]
    return AuxSnapshot(
        enabled=bool(raw.get("enabled")),
        alerts_enabled=bool(raw.get("alerts_enabled")),
        interval_s=float(raw.get("interval_s") or 0.0),
        # 实测节奏必须跟着设置值一起出去。它在守护进程里算好了
        # （``AuxSampler.observed_interval_s()``，最近 21 个间隔的中位数），
        # 而这里漏掉它,响应模型的默认值 None 就把「还没测够」和「这一层忘了传」
        # 变成了同一个样子 —— 而 ``observed_interval_s`` 存在的**全部理由**正是
        # 「别拿设置值当真实节奏」——配置值与实测值并不总是相等；这个字段若
        # 恒为 None，验证真实节奏就只能靠外部去量数据点的间距。
        observed_interval_s=raw.get("observed_interval_s"),
        window_s=float(raw.get("window_s") or 0.0),
        sampled=int(raw.get("sampled") or 0),
        skipped_busy=int(raw.get("skipped_busy") or 0),
        last_ts=raw.get("last_ts"),
        baseline_amp_m=raw.get("baseline_amp_m"),
        amp_tau_s=raw.get("amp_tau_s"),
        amp_oversampled=raw.get("amp_oversampled"),
        z_limits_m=[float(v) for v in (raw.get("z_limits_m") or [])],
        z_limits_enabled=raw.get("z_limits_enabled"),
        z_travel_m=[float(v) for v in (raw.get("z_travel_m") or [])],
        z_travel_source=str(raw.get("z_travel_source") or ""),
        detail=str(raw.get("detail") or ""),
        channels=channels,
    )


def _segment_row(d: dict) -> SegmentRow:
    return SegmentRow(
        seg_id=int(d.get("id") or d.get("seg_id") or 0),
        t_start=float(d.get("t_start") or 0.0),
        t_end=float(d.get("t_end") or 0.0),
        fs_hz=float(d.get("fs_hz") or 0.0),
        n_samples=int(d.get("n_samples") or 0),
        gap_s=float(d.get("gap_s") or 0.0),
        discontinuity=bool(d.get("discontinuity")),
        channel_name=str(d.get("channel_name") or ""),
        pinned=bool(d.get("pinned")),
        pin_reason=str(d.get("pin_reason") or ""),
        label=d.get("label"),
        label_note=str(d.get("label_note") or ""),
        label_ts=d.get("label_ts"),
        has_file=bool(d.get("has_file", bool(d.get("npy_path")))),
        file_bytes=int(d.get("npy_bytes") or 0),
        verdict=str(d.get("alert_level") or "ok"),
    )


# ── status / control ────────────────────────────────────────────────────────

@router.get("/monitoring/status", response_model=MonitoringStatus)
def monitoring_status() -> MonitoringStatus:
    """Daemon state + storage totals + the most recent segment's features."""

    def body() -> MonitoringStatus:
        from mast.monitoring.service import get_service
        from mast.monitoring.thresholds import get_monitor_thresholds

        svc = get_service()
        th = get_monitor_thresholds()
        if svc is not None:
            st = svc.status()
        else:
            # No daemon (standalone API, or the core never came up). History is
            # still readable — a stopped monitor must not blank the page.
            st = {"running": False, "state": "no_pool",
                  "detail": "监控服务未运行(独立 API 模式或内核未启动)",
                  "enabled_in_settings": bool(th.enabled),
                  "alerts_enabled": bool(th.alerts_enabled),
                  "segment_seconds": float(th.cm_segment_s),
                  "retention_hours": float(th.cm_keep_hours),
                  "retention_gb": float(th.cm_keep_gb)}

        stats = _store().storage_stats()
        latest_row = _store().latest_feature()
        latest = None
        if latest_row:
            latest = LatestFeature(
                ts=float(latest_row.get("t_start") or 0.0),
                seg_id=int(latest_row.get("segment_id") or 0),
                mean_a=latest_row.get("mean_a"),
                rms_detrended_a=latest_row.get("rms_detrended_a"),
                min_a=latest_row.get("min_a"),
                max_a=latest_row.get("max_a"),
                verdict=str(latest_row.get("alert_level") or "unknown"),
                metrics=_metrics(latest_row),
            )
        return MonitoringStatus(
            running=bool(st.get("running")),
            enabled_in_settings=bool(st.get("enabled_in_settings", th.enabled)),
            alerts_enabled=bool(st.get("alerts_enabled", th.alerts_enabled)),
            state=str(st.get("state") or "unknown"),
            detail=str(st.get("detail") or ""),
            retry_in_s=float(st.get("retry_in_s") or 0.0),
            strategy=st.get("strategy"),
            # 三态,不要 `bool(...)`:True/False 是探测结论,None 是「还没探过」。
            # 这一行是 ⑲ 的另一半 —— 光给 schema 加字段不够,逐字段构造的映射层
            # 不写它就照样丢。
            hr_available=_opt_bool(st.get("hr_available")),
            fs_hz=float(st.get("fs_hz") or 0.0),
            channel_name=str(st.get("channel_name") or ""),
            n_buffer=int(st.get("n_buffer") or 0),
            rt_freq_hz=float(st.get("rt_freq_hz") or 0.0),
            timebases_s=[float(v) for v in (st.get("timebases_s") or [])],
            timebase_index=int(st.get("timebase_index", -1)),
            timebase_check=str(st.get("timebase_check") or ""),
            pump_stats={k: int(v) for k, v in (st.get("pump_stats") or {}).items()},
            segment_seconds=float(st.get("segment_seconds") or th.cm_segment_s),
            connected=bool(st.get("connected")),
            segments_done=int(st.get("segments_done") or 0),
            gaps_total_s=float(st.get("gaps_total_s") or 0.0),
            last_segment_ts=st.get("last_segment_ts"),
            segments_total=int(stats.get("segments") or 0),
            segments_on_disk=int(stats.get("segments_on_disk") or 0),
            store_bytes=int(stats.get("store_bytes") or 0),
            pinned_count=int(stats.get("pinned") or 0),
            retention_hours=float(st.get("retention_hours") or th.cm_keep_hours),
            retention_gb=float(st.get("retention_gb") or th.cm_keep_gb),
            latest=latest,
            aux=_aux_snapshot(st.get("aux")),
            # 服务没跑时留空字典 ⇒ 前端读到 available=False，与「没有基线」同义:
            # 两种情况下判据用的都是固定阈值,而这个块回答的就是「判据在用什么」。
            baseline=(svc.baseline_status()
                      if (svc is not None and hasattr(svc, "baseline_status"))
                      else {}),
        )

    return _guarded(body, lambda e: MonitoringStatus(
        degraded=True, state="unknown",
        detail="监控模块未装载", detail_error=e))


@router.post("/monitoring/start", response_model=MonitoringControlResult)
def monitoring_start() -> MonitoringControlResult:
    def body() -> MonitoringControlResult:
        from mast.monitoring.service import get_service
        svc = get_service()
        if svc is None:
            return MonitoringControlResult(
                ok=False, running=False, degraded=True,
                note="监控服务未装载(独立 API 模式下无法启动)")
        svc.start()
        return MonitoringControlResult(ok=True, running=svc.status()["running"],
                                       note="已启动")

    return _guarded(body, lambda e: MonitoringControlResult(
        ok=False, degraded=True, note=f"启动失败:{e}"))


@router.post("/monitoring/stop", response_model=MonitoringControlResult)
def monitoring_stop() -> MonitoringControlResult:
    def body() -> MonitoringControlResult:
        from mast.monitoring.service import get_service
        svc = get_service()
        if svc is None:
            return MonitoringControlResult(ok=True, running=False,
                                           note="监控服务本就未运行")
        svc.stop()
        return MonitoringControlResult(ok=True, running=False, note="已停止")

    return _guarded(body, lambda e: MonitoringControlResult(
        ok=False, degraded=True, note=f"停止失败:{e}"))


@router.get("/monitoring/config", response_model=MonitoringConfigResponse)
def monitoring_config() -> MonitoringConfigResponse:
    """Effective settings plus the knob catalogue the settings UI renders.

    The catalogue is data, not a hard-coded list in the frontend, so adding a
    knob is a one-line backend change.
    """

    def body() -> MonitoringConfigResponse:
        from mast.monitoring.thresholds import get_monitor_thresholds, knob_catalog
        th = get_monitor_thresholds()
        return MonitoringConfigResponse(
            enabled=bool(th.enabled),
            alerts_enabled=bool(th.alerts_enabled),
            retention_hours=float(th.cm_keep_hours),
            retention_gb=float(th.cm_keep_gb),
            segment_seconds=float(th.cm_segment_s),
            knobs=[ThresholdKnob(**k) for k in knob_catalog()],
        )

    return _guarded(body, lambda e: MonitoringConfigResponse(degraded=True, detail=e))


# ── series ──────────────────────────────────────────────────────────────────

@router.get("/monitoring/live-trace", response_model=LiveTraceResponse)
def monitoring_live_trace(window_s: float = 60.0,
                          max_points: int = 1200) -> LiveTraceResponse:
    """Recent envelopes stitched into one min/max band — the live chart's source.

    The chart cannot ride the WebSocket: that bus replays only the last 100
    events and carries scalars by design, so waveform data is pulled here and
    the socket is used to know WHEN to pull.

    ``null`` in the band arrays = the daemon was not acquiring at that instant,
    which is different from a reading of zero. See ``LiveTraceResponse``.
    """
    w = max(1.0, min(_MAX_TRACE_WINDOW_S, float(window_s)))
    n = _clamp(max_points, 100, 5000)

    def body() -> LiveTraceResponse:
        out = _store().live_tail(window_s=w, max_points=n)
        return LiveTraceResponse(
            t_s=out.get("t_s") or [], i_min_a=out.get("i_min_a") or [],
            i_max_a=out.get("i_max_a") or [], window_s=w,
            last_ts=out.get("last_ts"), n_segments=int(out.get("n_segments") or 0),
            n_gaps=int(out.get("n_gaps") or 0),
        )

    return _guarded(body, lambda e: LiveTraceResponse(window_s=w, degraded=True,
                                                      detail=e))


@router.get("/monitoring/features", response_model=FeatureSeriesResponse)
def monitoring_features(since: Optional[float] = None,
                        until: Optional[float] = None,
                        limit: int = 2000, offset: int = 0,
                        max_points: Optional[int] = None) -> FeatureSeriesResponse:
    lim = _clamp(limit, 1, 20000)
    off = max(0, int(offset))
    mp = _clamp(max_points, 10, 20000) if max_points else None

    def body() -> FeatureSeriesResponse:
        rows, total, thinned = _store().features_query(
            since=since, until=until, limit=lim, offset=off, max_points=mp)
        return FeatureSeriesResponse(
            rows=[FeatureRow(
                ts=float(r.get("t_start") or 0.0),
                seg_id=int(r.get("segment_id") or 0),
                verdict=str(r.get("alert_level") or "ok"),
                scanning=None if r.get("ctx_scanning") is None else bool(r["ctx_scanning"]),
                bias_v=r.get("ctx_bias_v"),
                setpoint_a=r.get("ctx_setpoint_a"),
                z_ctrl_on=None if r.get("ctx_zctrl_on") is None else bool(r["ctx_zctrl_on"]),
                skill=str(r.get("ctx_skill") or ""),
                metrics=_metrics(r),
            ) for r in rows],
            total=total, since=since, until=until, thinned=thinned,
        )

    return _guarded(body, lambda e: FeatureSeriesResponse(degraded=True, detail=e))


@router.get("/monitoring/aux/series", response_model=AuxSeriesResponse)
def monitoring_aux_series(window_s: float = 600.0,
                          since: Optional[float] = None,
                          until: Optional[float] = None,
                          max_points: int = 600,
                          columns: Optional[str] = None) -> AuxSeriesResponse:
    """Z / qPlus-amplitude / Δf history for the small-multiples chart.

    ``window_s`` is a convenience: it means "the last N seconds" and is ignored
    when ``since`` is given explicitly. ``columns`` is a comma-separated subset
    of the aux columns (whitelisted); omitted, it returns the three raw values.

    Values stay in SI (metres, hertz) — the same rule the rest of this router
    follows, so a display-unit number can never end up in an ``_m`` field.

    Missing readings come back as ``null``, not ``0``: a collapsed qPlus
    amplitude genuinely IS zero, so the two must stay distinguishable.
    """
    import time as _time

    w = max(10.0, min(86400.0, float(window_s)))
    lo = since if since is not None else (_time.time() - w)
    mp = _clamp(max_points, 50, 5000)
    asked = [c.strip() for c in (columns or "").split(",")]
    wanted = tuple(c for c in asked if c in _AUX_SELECTABLE) or _AUX_SERIES_COLUMNS

    def body() -> AuxSeriesResponse:
        # ``window_s`` 语义 = 「最近 N 秒」⇒ **从最新一端截**。
        # 显式给了 since/until 的调用是分页语义,照旧从早的那一端。
        rows, total, thinned = _store().aux_query(
            since=lo, until=until, limit=20000, max_points=mp,
            newest=(since is None and until is None))
        t_s = [float(r.get("ts") or 0.0) for r in rows]
        series: dict[str, list[Optional[float]]] = {
            col: [_opt_num(r.get(col)) for r in rows] for col in wanted
        }
        return AuxSeriesResponse(
            t_s=t_s, series=series,
            verdicts=[str(r.get("verdict") or "ok") for r in rows],
            # Context rides along so a remote caller can split populations the
            # same way the local one does. Calibrating Z drift across "parked"
            # and "scanning" segments produces a threshold that fits neither —
            # the same reason commission._split() exists for the current path.
            scanning=[_opt_bool(r.get("ctx_scanning")) for r in rows],
            z_ctrl_on=[_opt_bool(r.get("ctx_zctrl_on")) for r in rows],
            skills=[str(r.get("ctx_skill") or "") for r in rows],
            window_s=w, total=total, thinned=thinned,
        )

    return _guarded(body, lambda e: AuxSeriesResponse(window_s=w, degraded=True,
                                                      detail=e))


# ── alerts ──────────────────────────────────────────────────────────────────

@router.get("/monitoring/alerts", response_model=AlertsResponse)
def monitoring_alerts(since: Optional[float] = None, limit: int = 50,
                      level: Optional[str] = None) -> AlertsResponse:
    lim = _clamp(limit, 1, 500)
    lvl = level if level in ("warn", "critical") else None

    def body() -> AlertsResponse:
        rows, total = _store().alerts_query(since=since, limit=lim, level=lvl)
        return AlertsResponse(
            alerts=[AlertRow(
                id=int(r.get("id") or 0), ts=float(r.get("ts") or 0.0),
                level=str(r.get("level") or "warn"), rule=str(r.get("rule") or ""),
                summary_zh=str(r.get("summary_zh") or ""),
                seg_id=r.get("segment_id"),
                evidence_available=bool(r.get("evidence_available")),
                emitted_buffer=bool(r.get("emitted_buffer")),
                acked=bool(r.get("acked")),
                delivered_agent=bool(r.get("delivered_agent")),
            ) for r in rows],
            total=total,
        )

    return _guarded(body, lambda e: AlertsResponse(degraded=True, detail=e))


@router.get("/monitoring/alerts/{alert_id}/evidence",
            response_model=AlertEvidenceResponse)
def monitoring_alert_evidence(alert_id: int) -> AlertEvidenceResponse:
    """The trace + spectrum PNG attached to an alert, base64 encoded."""

    def body() -> AlertEvidenceResponse:
        raw = _store().alert_evidence_png(alert_id)
        if not raw:
            return AlertEvidenceResponse(ok=False, alert_id=alert_id,
                                         detail="该告警没有可用的证据图")
        return AlertEvidenceResponse(ok=True, alert_id=alert_id,
                                     png_b64=base64.b64encode(raw).decode("ascii"))

    return _guarded(body, lambda e: AlertEvidenceResponse(
        alert_id=alert_id, degraded=True, detail=e))


@router.post("/monitoring/alerts/{alert_id}/ack", response_model=AlertAckResponse)
def monitoring_alert_ack(alert_id: int) -> AlertAckResponse:
    """人把这条告警点掉:「我知道了,别再提醒我」。

    ``store.ack_alert()`` 曾经存在却**全树零调用方** —— 没有路由、没有前端,
    写进去的 ``acked`` 永远是 ``false``,那不是「没人点」,是**点不了**:
    一个零可达调用方的原语,从测试外面看和一个能用的逃生门一模一样。

    ## 它**不**做什么

    ``acked`` 与 ``delivered_agent`` 是**两个主体**(见 ``monitoring/store`` 的建表
    注释)。点掉这条:

    * **不会**让它不再送给 agent —— 人点掉的理由多半是「我不需要弹窗」,
      不是「agent 不必知道」;
    * **不会**替 agent 确认它看过了。

    也不解除任何拦截、不动任何硬件。纯记账。
    """

    def body() -> AlertAckResponse:
        _store().ack_alert(alert_id)
        return AlertAckResponse(ok=True, alert_id=alert_id, acked=True)

    return _guarded(body, lambda e: AlertAckResponse(
        alert_id=alert_id, degraded=True, detail=e))


# ── segments ────────────────────────────────────────────────────────────────

@router.get("/monitoring/segments", response_model=SegmentListResponse)
def monitoring_segments(since: Optional[float] = None,
                        until: Optional[float] = None,
                        pinned: Optional[bool] = None,
                        label: Optional[str] = None,
                        limit: int = 100, offset: int = 0) -> SegmentListResponse:
    """Segment index. ``label='unlabeled'`` selects the ones awaiting a verdict."""
    lim = _clamp(limit, 1, 1000)
    off = max(0, int(offset))

    def body() -> SegmentListResponse:
        out = _store().segments_query(since=since, until=until, pinned=pinned,
                                      label=label, limit=lim, offset=off)
        return SegmentListResponse(
            segments=[_segment_row(d) for d in out.get("segments") or []],
            total=int(out.get("total") or 0),
            pinned_count=int(out.get("pinned_count") or 0),
            labeled_count=int(out.get("labeled_count") or 0),
        )

    return _guarded(body, lambda e: SegmentListResponse(degraded=True, detail=e))


@router.get("/monitoring/segments/{seg_id}/data", response_model=SegmentDataResponse)
def monitoring_segment_data(seg_id: int, max_points: int = 4000,
                            include_psd: bool = False) -> SegmentDataResponse:
    """One segment's waveform, min/max decimated so spikes survive.

    ``include_psd`` computes the spectrum from the FULL-rate samples, never from
    the decimated view — decimation would alias everything above the new Nyquist
    onto the bands the operator is reading.
    """
    mp = _clamp(max_points, 100, 20000)

    def body() -> SegmentDataResponse:
        store = _store()
        out = store.read_segment_decimated(seg_id, max_points=mp)
        if not out:
            return SegmentDataResponse(ok=False, seg_id=seg_id,
                                       detail="该段不存在,或原始数据与包络都已不可用")
        psd = None
        if include_psd:
            raw = store.segment_psd(seg_id)
            if raw:
                psd = FFTResponse(ok=True, **{k: v for k, v in raw.items()
                                              if k in FFTResponse.model_fields})
        meta = out.get("meta") or {}
        return SegmentDataResponse(
            ok=True, seg_id=seg_id, t0=float(out.get("t0") or 0.0),
            fs_hz=float(out.get("fs_hz") or 0.0),
            n_samples_raw=int(out.get("n_samples_raw") or 0),
            t_s=out.get("t_s") or [], i_a=out.get("i_a") or [],
            source=str(out.get("source") or "raw"),
            decimated=bool(out.get("decimated")),
            psd=psd, meta=_segment_row(meta) if meta else None,
        )

    return _guarded(body, lambda e: SegmentDataResponse(seg_id=seg_id,
                                                        degraded=True, detail=e))


@router.post("/monitoring/segments/{seg_id}/pin", response_model=PinResult)
def monitoring_pin_segment(seg_id: int, body_in: PinRequest) -> PinResult:
    """Pin (or unpin) a segment. Pinned segments survive the retention sweep."""

    def body() -> PinResult:
        ok = _store().set_pin(seg_id, bool(body_in.pinned),
                              body_in.reason or "manual")
        return PinResult(ok=ok, seg_id=seg_id, pinned=bool(body_in.pinned))

    return _guarded(body, lambda e: PinResult(seg_id=seg_id, degraded=True))


@router.post("/monitoring/segments/{seg_id}/label", response_model=LabelResult)
def monitoring_label_segment(seg_id: int, body_in: LabelRequest) -> LabelResult:
    """Record a human verdict on a segment; ``label=null`` clears it.

    Labelling pins the segment. A judged segment is a corpus item, and letting
    the retention sweep delete the waveform behind a label would leave a verdict
    with nothing to train on.
    """

    def body() -> LabelResult:
        store = _store()
        meta = store.segment_meta(seg_id)
        if not meta:
            return LabelResult(ok=False, seg_id=seg_id)
        store.clear_labels(seg_id, source="human")
        label = (body_in.label or "").strip() or None
        pinned = bool(meta.get("pinned"))
        if label:
            store.add_label(t_start=float(meta.get("t_start") or 0.0),
                            t_end=float(meta.get("t_end") or 0.0),
                            label=label, segment_id=seg_id, source="human",
                            note=body_in.note)
            store.set_pin(seg_id, True, f"labeled:{label}")
            pinned = True
        return LabelResult(ok=True, seg_id=seg_id, label=label, pinned=pinned)

    return _guarded(body, lambda e: LabelResult(seg_id=seg_id, degraded=True))


# ── noise baseline ──────────────────────────────────────────────────────────
#
# 基线是「实时参考」的那一半：判据拿它当**分母**（当前噪声是同一电流下预期值的
# 多少倍），而不是拿它当阈值。阈值仍然只在 thresholds.py 一处。
# 设计与论据见 docs/v2/design/current_noise_baseline.md。

def _baseline_row(d: dict) -> BaselineRow:
    return BaselineRow(**{k: v for k, v in (d or {}).items()
                          if k in BaselineRow.model_fields})


@router.get("/monitoring/baselines", response_model=BaselineListResponse)
def monitoring_baselines(limit: int = 50) -> BaselineListResponse:
    """已记录的噪声基线，新的在前。``active_id`` 是判据正在用的那一份。"""
    lim = _clamp(limit, 1, 200)

    def body() -> BaselineListResponse:
        rows = _store().baselines(limit=lim)
        act = next((r["id"] for r in rows if r.get("active")), None)
        return BaselineListResponse(
            baselines=[_baseline_row(r) for r in rows], active_id=act)

    return _guarded(body, lambda e: BaselineListResponse(degraded=True, detail=e))


@router.get("/monitoring/baselines/{baseline_id}",
            response_model=BaselineDetailResponse)
def monitoring_baseline(baseline_id: int) -> BaselineDetailResponse:
    """一份基线的全部内容：条件快照、两个模型、谱线归因、每个工况点。"""

    def body() -> BaselineDetailResponse:
        d = _store().baseline(baseline_id, with_points=True)
        if not d:
            return BaselineDetailResponse(ok=False, detail="没有这份基线")
        pts = [BaselinePoint(**{k: v for k, v in p.items()
                                if k in BaselinePoint.model_fields})
               for p in (d.get("points") or [])]
        return BaselineDetailResponse(ok=True, baseline=_baseline_row(d), points=pts)

    return _guarded(body, lambda e: BaselineDetailResponse(degraded=True, detail=e))


@router.get("/monitoring/baselines/points/{point_id}/curve",
            response_model=BaselineCurveResponse)
def monitoring_baseline_curve(point_id: int, kind: str = "psd") -> BaselineCurveResponse:
    """一个工况点的谱（``kind=psd``）或宽度直方图（``kind=hist``）。

    文件没了就返回 ``ok=false`` —— 不拿别的点的曲线顶上，也不返回空数组冒充
    「这里就是平的」。
    """
    k = "hist" if str(kind).lower().startswith("h") else "psd"

    def body() -> BaselineCurveResponse:
        c = _store().baseline_point_curve(point_id, kind=k)
        if not c:
            return BaselineCurveResponse(ok=False, kind=k,
                                         detail="这个点的曲线文件不在了")
        return BaselineCurveResponse(ok=True, kind=k, x=c["x"], y=c["y"])

    return _guarded(body, lambda e: BaselineCurveResponse(kind=k, degraded=True,
                                                          detail=e))


@router.post("/monitoring/baselines/activate", response_model=BaselineActivateResult)
def monitoring_activate_baseline(body_in: BaselineActivateRequest) -> BaselineActivateResult:
    """选定判据要用的基线。``baseline_id=null`` = 停用，回到固定阈值。

    只有 ``status='complete'`` 的能被激活；拒绝时如实说，不静默失败。
    """

    def body() -> BaselineActivateResult:
        ok = _store().activate_baseline(body_in.baseline_id)
        if not ok:
            return BaselineActivateResult(
                ok=False, detail=("激活失败 —— 只有跑完整（status=complete）的"
                                  "基线才能启用；没跑完的那份没有 sigma 曲线。"))
        return BaselineActivateResult(ok=True, active_id=body_in.baseline_id)

    return _guarded(body, lambda e: BaselineActivateResult(detail=e))


@router.delete("/monitoring/baselines/{baseline_id}",
               response_model=BaselineActivateResult)
def monitoring_delete_baseline(baseline_id: int) -> BaselineActivateResult:
    """删一份基线。**正在被判据使用的那份拒删** —— 删掉分母会让判据在下一段
    静默换回固定阈值，而没有任何地方说过这件事。先停用，再删。"""

    def body() -> BaselineActivateResult:
        ok = _store().delete_baseline(baseline_id)
        return BaselineActivateResult(
            ok=ok, detail=None if ok else "删不掉：它不存在，或正在被判据使用（先停用）")

    return _guarded(body, lambda e: BaselineActivateResult(detail=e))


@router.get("/monitoring/baseline-check", response_model=BaselineCheckResponse)
def monitoring_baseline_check() -> BaselineCheckResponse:
    """当前噪声相对基线的位置 —— 「现在比参考差多少」的那个数。

    判不了就说原因（没有基线 / 电流读不到 / 工作点在标定区间之外），
    ``ratio`` 保持 ``None``。
    """

    def body() -> BaselineCheckResponse:
        from mast.monitoring.service import get_service
        svc = get_service()
        if svc is None or not hasattr(svc, "baseline_check"):
            return BaselineCheckResponse(judged=False,
                                         reason="电流监控服务没在跑")
        d = svc.baseline_check() or {}
        return BaselineCheckResponse(**{k: v for k, v in d.items()
                                        if k in BaselineCheckResponse.model_fields})

    return _guarded(body, lambda e: BaselineCheckResponse(degraded=True, detail=e))


