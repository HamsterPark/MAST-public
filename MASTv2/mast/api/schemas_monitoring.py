"""Response models for the tunnelling-current monitor endpoints.

Everything defaults to an empty/zero value and carries ``degraded`` so a handler
can always answer — the monitoring package may not be importable at all in
standalone API mode, and the UI has to render something rather than a 500.

Feature values ride in an open ``metrics`` map instead of one field per column.
The extractor gains features over time; pinning each one into this schema would
mean a coordinated backend + openapi + frontend change to add a number nobody
has to type-check.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from mast.api.schemas_vision import FFTResponse


class LatestFeature(BaseModel):
    ts: float = 0.0
    seg_id: int = 0
    mean_a: Optional[float] = None
    rms_detrended_a: Optional[float] = None
    min_a: Optional[float] = None
    max_a: Optional[float] = None
    verdict: str = "unknown"          # ok | warn | critical | suppressed | unknown
    metrics: dict[str, float] = Field(default_factory=dict)


class AuxChannelState(BaseModel):
    """One auxiliary channel (Z position / qPlus amplitude / frequency shift).

    ``verdict`` has two values the current path does not have, and the
    difference matters:

    * ``unavailable`` — this rig does not expose the signal at all. An STM with
      no qPlus sensor is a normal configuration, not a fault.
    * ``unjudged`` — recorded but deliberately not judged: either the channel
      carries no rules by design (Δf), or the thresholds have not been
      calibrated on this instrument yet, or the amplitude baseline is missing.

    Neither one means "fine". ``note`` says which it is and what to do.
    """

    kind: str = ""                    # z | amplitude | df
    label_zh: str = ""
    unit: str = ""                    # m | Hz
    available: bool = False
    signal_index: int = -1
    signal_name: str = ""
    judged: bool = False
    value: Optional[float] = None
    ts: Optional[float] = None
    verdict: str = "unknown"          # ok | warn | suppressed | unjudged | unavailable
    note: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)


class AuxSnapshot(BaseModel):
    """The auxiliary sampler's own state. Zero TCP — the daemon's last read.

    ``amp_tau_s`` is the qPlus amplitude's 1/e relaxation time ``Q/(π f₀)``,
    computed from the MEASURED resonance the PLL sweep wrote into the instrument
    profile. It is the checkable answer to "are we sampling fast enough for
    amplitude?": the amplitude of a high-Q resonator physically cannot move
    faster than that. ``None`` when the resonance has never been swept — no
    guessing from a nameplate value.

    ``interval_s`` 是设置值，``observed_interval_s`` 是成功采样间隔的统计值。
    机会式采样可能错过节流后的机会，因此采样是否足够快应优先使用观测间隔。
    观测点数不足时才回退设置值，并保留 ``observed_interval_s=None`` 声明缺失。
    """

    enabled: bool = False
    alerts_enabled: bool = False
    interval_s: float = 0.0
    #: 实测节奏（相邻成功采样间隔的中位数）。``None`` = 点数还不够，测不出来
    #: —— 与「测出来很慢」是两回事，所以不拿设置值充数。
    observed_interval_s: Optional[float] = None
    window_s: float = 0.0
    sampled: int = 0
    skipped_busy: int = 0
    last_ts: Optional[float] = None
    baseline_amp_m: Optional[float] = None
    amp_tau_s: Optional[float] = None
    amp_oversampled: Optional[bool] = None
    #: ``ZCtrl_LimitsGet`` 的**原始**软限值，以及它启没启用（三态：None = 没读到）。
    z_limits_m: list[float] = Field(default_factory=list)
    z_limits_enabled: Optional[bool] = None
    #: 余量**真正用的**那对数字，以及是哪一条读答出来的
    #: （``Piezo_RangeGet/2`` 或 ``ZCtrl_LimitsGet (enabled)``）。
    #:
    #: 分开报不是啰嗦：软限值在未启用时不起任何作用，而 2026-08-10 之前余量正是
    #: 对着它算的 —— **一个算错分母的百分比和一个算对的长得一模一样**，来源是
    #: 唯一能把它们分开的东西。空 = 三条读都没成功 ⇒ 余量为 None、判据静默。
    z_travel_m: list[float] = Field(default_factory=list)
    z_travel_source: str = ""
    detail: str = ""
    channels: list[AuxChannelState] = Field(default_factory=list)


class AuxSeriesResponse(BaseModel):
    """Aux time series for the small-multiples chart.

    Columnar, and one array per channel aligned to a single ``t_s`` — the whole
    point of the wide ``aux_samples`` table. ``null`` inside a series means that
    channel had no reading at that instant, which is different from zero: a
    collapsed qPlus amplitude IS zero.
    """

    t_s: list[float] = Field(default_factory=list)
    series: dict[str, list[Optional[float]]] = Field(default_factory=dict)
    verdicts: list[str] = Field(default_factory=list)
    #: Per-row context, parallel to ``t_s``. A remote caller needs it to split
    #: populations before calibrating: Z drift while scanning and Z drift on a
    #: parked tip are two different distributions, and one threshold fitted
    #: across both fits neither.
    scanning: list[Optional[bool]] = Field(default_factory=list)
    z_ctrl_on: list[Optional[bool]] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    window_s: float = 0.0
    total: int = 0
    thinned: bool = False
    degraded: bool = False
    detail: Optional[str] = None


class MonitoringStatus(BaseModel):
    running: bool = False
    enabled_in_settings: bool = False
    alerts_enabled: bool = False
    state: str = "unknown"            # disabled|no_pool|comms_down|probing|running|unavailable|paused
    detail: str = ""
    retry_in_s: float = 0.0
    strategy: Optional[str] = None    # osci1t | osci2t | oscihr | None
    # Oscilloscope High Resolution 能力的探测状态。None 表示尚未探测。
    # 该字段必须在响应模型中声明，避免 FastAPI 过滤掉 service.status 的值，
    # 把未探测和结果未传出混为一谈。
    hr_available: Optional[bool] = None
    fs_hz: float = 0.0
    channel_name: str = ""
    n_buffer: int = 0
    #: Commissioning readouts — the timebase table and the achieved RT rate can
    #: only be known on the instrument, and pump_stats separates "keeping up"
    #: from "the scope stopped refilling" without issuing any extra query.
    rt_freq_hz: float = 0.0
    timebases_s: list[float] = Field(default_factory=list)
    timebase_index: int = -1
    #: 时基表自校验的结论:``ok`` / ``unverified: …`` / ``mismatch: …``。
    #: ``""`` = 泵还没读过表。三态,不是两态 —— 「没对上」和「没法对」是两句话。
    timebase_check: str = ""
    pump_stats: dict[str, int] = Field(default_factory=dict)
    segment_seconds: float = 0.0
    connected: bool = False
    segments_done: int = 0
    gaps_total_s: float = 0.0
    last_segment_ts: Optional[float] = None
    segments_total: int = 0
    segments_on_disk: int = 0
    store_bytes: int = 0
    pinned_count: int = 0
    retention_hours: float = 0.0
    retention_gb: float = 0.0
    #: 活跃噪声基线的摘要。``available=False`` 意味着 rms_high 用的是固定
    #: 阈值 ``cm_rms_warn_a`` —— 与本功能上线前逐字节相同的行为。
    baseline: dict = Field(default_factory=dict)
    latest: Optional[LatestFeature] = None
    #: 辅助通道（Z / qPlus 振幅 / Δf）。``None`` = 采集守护未运行 —— 与
    #: 「这台机器没有这些通道」是两件事，后者在 channels[].available 里说。
    aux: Optional[AuxSnapshot] = None
    degraded: bool = False
    detail_error: Optional[str] = None


class MonitoringControlResult(BaseModel):
    ok: bool = False
    running: bool = False
    note: str = ""
    degraded: bool = False


class LiveTraceResponse(BaseModel):
    """Min/max envelope band for the live chart. Absolute unix seconds on t.

    ``null`` inside the band arrays marks an ACQUISITION GAP — the daemon was not
    measuring at that instant. It is not a reading of zero, and it is not a
    missing field: it is the one thing the arrays cannot say by being shorter,
    because the two sides of a gap would then become adjacent and the chart would
    draw a straight line across the outage.
    Where the gaps are is decided in ``store.trace_gap_marks``.
    """

    t_s: list[float] = Field(default_factory=list)
    i_min_a: list[Optional[float]] = Field(default_factory=list)
    i_max_a: list[Optional[float]] = Field(default_factory=list)
    window_s: float = 60.0
    #: Newest REAL reading. Never a gap marker — the page measures staleness
    #: against it, and a null would make "stopped updating" itself disappear.
    last_ts: Optional[float] = None
    n_segments: int = 0
    #: Acquisition gaps inside the window. The chart says so in words; a break
    #: nobody labels reads as a rendering glitch.
    n_gaps: int = 0
    degraded: bool = False
    detail: Optional[str] = None


class FeatureRow(BaseModel):
    ts: float = 0.0
    seg_id: int = 0
    verdict: str = "ok"
    scanning: Optional[bool] = None
    bias_v: Optional[float] = None
    setpoint_a: Optional[float] = None
    z_ctrl_on: Optional[bool] = None
    skill: str = ""
    gap_s: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)


class FeatureSeriesResponse(BaseModel):
    rows: list[FeatureRow] = Field(default_factory=list)
    total: int = 0
    since: Optional[float] = None
    until: Optional[float] = None
    thinned: bool = False
    degraded: bool = False
    detail: Optional[str] = None


class AlertRow(BaseModel):
    id: int = 0
    ts: float = 0.0
    level: str = "warn"               # warn | critical
    rule: str = ""
    summary_zh: str = ""
    seg_id: Optional[int] = None
    evidence_available: bool = False
    emitted_buffer: bool = False
    #: 人在 UI 上点掉了这条。
    acked: bool = False

    # delivered_agent 表示进入模型上下文，与人工 acked 不同。
    # API 同时暴露发射与送达状态，用于分辨已发出但尚未送达的告警。
    delivered_agent: bool = False


class AlertAckResponse(BaseModel):
    """``POST /monitoring/alerts/{id}/ack`` 的回执。

    ``acked`` 只说「人点掉了」。它**不**回答「agent 看没看见」——
    那是 ``AlertRow.delivered_agent``,另一个主体。
    """

    ok: bool = False
    alert_id: int = 0
    acked: bool = False
    degraded: bool = False
    detail: Optional[str] = None


class AlertsResponse(BaseModel):
    alerts: list[AlertRow] = Field(default_factory=list)
    total: int = 0
    degraded: bool = False
    detail: Optional[str] = None


class AlertEvidenceResponse(BaseModel):
    ok: bool = False
    alert_id: int = 0
    png_b64: Optional[str] = None
    degraded: bool = False
    detail: Optional[str] = None


class SegmentRow(BaseModel):
    seg_id: int = 0
    t_start: float = 0.0
    t_end: float = 0.0
    fs_hz: float = 0.0
    n_samples: int = 0
    gap_s: float = 0.0
    discontinuity: bool = False
    channel_name: str = ""
    pinned: bool = False
    pin_reason: str = ""
    label: Optional[str] = None
    label_note: str = ""
    label_ts: Optional[float] = None
    has_file: bool = False
    file_bytes: int = 0
    verdict: str = "ok"


class SegmentListResponse(BaseModel):
    segments: list[SegmentRow] = Field(default_factory=list)
    total: int = 0
    pinned_count: int = 0
    labeled_count: int = 0
    degraded: bool = False
    detail: Optional[str] = None


class SegmentDataResponse(BaseModel):
    """Decimated waveform for one segment.

    ``source='envelope'`` means the raw ``.npy`` was swept by the retention
    policy and this is the stored min/max band instead — coarser, but honest,
    and the UI says so.
    """

    ok: bool = False
    seg_id: int = 0
    t0: float = 0.0
    fs_hz: float = 0.0
    n_samples_raw: int = 0
    t_s: list[float] = Field(default_factory=list)
    i_a: list[float] = Field(default_factory=list)
    source: str = "raw"               # raw | envelope
    decimated: bool = False
    psd: Optional[FFTResponse] = None
    meta: Optional[SegmentRow] = None
    degraded: bool = False
    detail: Optional[str] = None


class PinRequest(BaseModel):
    pinned: bool = True
    reason: Optional[str] = None


class PinResult(BaseModel):
    ok: bool = False
    seg_id: int = 0
    pinned: bool = False
    degraded: bool = False


class LabelRequest(BaseModel):
    label: Optional[str] = None       # good | bad | … ; None clears
    note: Optional[str] = None


class LabelResult(BaseModel):
    ok: bool = False
    seg_id: int = 0
    label: Optional[str] = None
    pinned: bool = False
    degraded: bool = False


class ThresholdKnob(BaseModel):
    key: str
    label_zh: str = ""
    hint_zh: str = ""
    min: float = 0.0
    max: float = 0.0
    step: float = 0.0
    default: float = 0.0
    value: float = 0.0
    is_bool: bool = False
    #: ``current`` | ``aux`` —— 只给 UI 分栏用。两组阈值的量纲不同(安培 vs 米/赫兹),
    #: 混在一张长表里读的人很容易以为它们是一套。
    group: str = "current"


class MonitoringConfigResponse(BaseModel):
    enabled: bool = True
    alerts_enabled: bool = True
    retention_hours: float = 24.0
    retention_gb: float = 4.0
    segment_seconds: float = 1.0
    knobs: list[ThresholdKnob] = Field(default_factory=list)
    degraded: bool = False
    detail: Optional[str] = None


# ── noise baseline ──────────────────────────────────────────────────────────

class BaselineRow(BaseModel):
    """一份噪声基线的元信息。曲线本身走 ``/points/{pid}/curve``。"""

    id: int = 0
    ts: float = 0.0
    label: str = ""
    note: str = ""
    #: ``complete`` | ``aborted`` | ``running``。**只有 complete 能被激活** ——
    #: 一份没跑完的表征没有跨点模型，激活它等于把判据的分母换成 None，
    #: 而那与「没有基线」是同一件事，却会在界面上显示成「有基线」。
    status: str = "running"
    active: bool = False
    n_points: int = 0
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    fs_hz: Optional[float] = None
    conditions: Optional[dict] = None
    sigma_model: Optional[dict] = None
    white_model: Optional[dict] = None
    # 正负偏压配对检查。**加一个字段是双边动作** —— schema 漏了它，

    # FastAPI 会按 model_fields 静默过滤掉，读出来是 null，

    # 与「根本没采到」长得一模一样（2026-08-18 已经栽过一次）。

    polarity: Optional[dict] = None


    # 偏压幅度依赖。还要有 bias_magnitude —— 加一个字段永远是


    # 双边动作，schema 漏了它 FastAPI 会静默过滤成 null。


    bias_magnitude: Optional[dict] = None
    lines: Optional[dict] = None
    repeatability: Optional[dict] = None


class BaselinePoint(BaseModel):
    """一个工况点。i_measured_a 是测得的电流，而非设定点；两者不能互相替代。"""

    id: int = 0
    ordinal: int = 0
    tag: str = ""
    ts: float = 0.0
    bias_v: Optional[float] = None
    setpoint_a: Optional[float] = None
    i_measured_a: Optional[float] = None
    z_m: Optional[float] = None
    n_segments: int = 0
    seg_id_lo: Optional[int] = None
    seg_id_hi: Optional[int] = None
    sigma_a: Optional[float] = None
    sigma_iqr_a: Optional[float] = None
    sigma_mad_a: Optional[float] = None
    fwhm_a: Optional[float] = None
    fwhm_over_sigma: Optional[float] = None
    ptp_a: Optional[float] = None
    kurtosis: Optional[float] = None
    skewness: Optional[float] = None
    white_a2hz: Optional[float] = None
    iqr_a: Optional[float] = None
    p99_p1_a: Optional[float] = None
    mean_a: Optional[float] = None
    line_ratio: Optional[float] = None
    band_1_10_a2: Optional[float] = None
    band_10_45_a2: Optional[float] = None
    band_45_65_a2: Optional[float] = None
    band_65_200_a2: Optional[float] = None
    band_200_1k_a2: Optional[float] = None

    # Z 与 Z-电流联合量必须同时出现在存储和响应模型中，避免已保存字段被 API 过滤成缺失。
    z_sigma_m: Optional[float] = None
    z_step_rms_m: Optional[float] = None
    z_white_m2hz: Optional[float] = None
    z_ptp_m: Optional[float] = None
    z_fwhm_m: Optional[float] = None
    z_mean_m: Optional[float] = None
    z_n_runs: Optional[int] = None
    coh_fraction: Optional[float] = None
    coh_n_pairs: Optional[int] = None
    kappa_per_m: Optional[float] = None
    apparent_barrier_ev: Optional[float] = None
    #: 逐点的细节（谱线、Z 摘要、耦合诊断、采集时的状态快照）。判不了的时候，
    #: 原因就在这里面 —— 不给出去的话「判不了」和「没问题」在 API 这一层又合流了。
    extra_json: Optional[str] = None


class BaselineListResponse(BaseModel):
    baselines: list[BaselineRow] = Field(default_factory=list)
    active_id: Optional[int] = None
    degraded: bool = False
    detail: Optional[str] = None


class BaselineDetailResponse(BaseModel):
    ok: bool = False
    baseline: Optional[BaselineRow] = None
    points: list[BaselinePoint] = Field(default_factory=list)
    degraded: bool = False
    detail: Optional[str] = None


class BaselineCurveResponse(BaseModel):
    """一个工况点的谱或宽度直方图。``kind='psd'`` 时 x=Hz、y=A²/Hz。"""

    ok: bool = False
    kind: str = "psd"
    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    degraded: bool = False
    detail: Optional[str] = None


class BaselineActivateRequest(BaseModel):
    #: ``None`` = 全部停用,判据回到固定阈值 ``cm_rms_warn_a``。
    baseline_id: Optional[int] = None


class BaselineActivateResult(BaseModel):
    ok: bool = False
    active_id: Optional[int] = None
    detail: Optional[str] = None


class BaselineCheckResponse(BaseModel):
    """当前噪声相对基线的位置。

    ``judged=False`` 时 ``ratio`` 是 ``None`` 而不是 1.0 —— 「判不了」和
    「正常」在这个子系统里永远是两个词,一个看起来正常的 1.0 会让两者分不开。
    """

    judged: bool = False
    reason: str = ""
    sigma_a: Optional[float] = None
    expected_a: Optional[float] = None
    ratio: Optional[float] = None
    in_range: bool = False
    extrapolation: Optional[float] = None
    baseline_id: Optional[int] = None
    conditions_match: Optional[bool] = None
    condition_diff: list[str] = Field(default_factory=list)
    seg_id: Optional[int] = None
    ts: Optional[float] = None
    scanning: Optional[bool] = None
    degraded: bool = False
    detail: Optional[str] = None


__all__ = [
    "LatestFeature", "MonitoringStatus", "MonitoringControlResult",
    "LiveTraceResponse", "FeatureRow", "FeatureSeriesResponse",
    "AlertRow", "AlertsResponse", "AlertEvidenceResponse",
    "SegmentRow", "SegmentListResponse", "SegmentDataResponse",
    "PinRequest", "PinResult", "LabelRequest", "LabelResult",
    "ThresholdKnob", "MonitoringConfigResponse",
    "AuxChannelState", "AuxSnapshot", "AuxSeriesResponse",
    "BaselineRow", "BaselinePoint", "BaselineListResponse",
    "BaselineDetailResponse", "BaselineCurveResponse",
    "BaselineActivateRequest", "BaselineActivateResult",
    "BaselineCheckResponse",
]
