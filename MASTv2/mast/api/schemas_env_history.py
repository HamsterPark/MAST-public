"""环境参数历史记录器的响应模型。

与 :mod:`mast.api.schemas_monitoring` 同一套约定：每个字段都有空/零默认值，
每个响应都带 ``degraded`` —— envhistory 包在纯 API 模式下可能根本 import 不了,
而界面需要渲染出点什么，而不是一个 500。

一条设计上的取舍写在这里：序列点用**扁平的固定字段**（mean/min/max/std/n）,
不用 monitoring 那种开放 ``metrics`` map。理由是这五个量是统计桶的定义本身,
不会随时间增加；而电流特征会（那正是它用开放 map 的原因）。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class EnvHistoryKnob(BaseModel):
    """一个可调旋钮的完整描述，供设置页动态渲染。"""

    key: str
    label_zh: str = ""
    hint_zh: str = ""
    min: float = 0.0
    max: float = 0.0
    step: float = 0.0
    default: float = 0.0
    value: float = 0.0
    is_bool: bool = False


class EnvHistoryConfigResponse(BaseModel):
    enabled: bool = False
    spectra_enabled: bool = False
    z_enabled: bool = False
    bucket_s: float = 0.0
    raw_keep_days: float = 0.0
    spectrum_interval_s: float = 0.0
    knobs: list[EnvHistoryKnob] = Field(default_factory=list)
    degraded: bool = False
    detail: str = ""


class EnvSensorSummary(BaseModel):
    sensor: str
    unit: str = ""
    n_buckets: int = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None


class EnvSensorsResponse(BaseModel):
    sensors: list[EnvSensorSummary] = Field(default_factory=list)
    degraded: bool = False
    detail: str = ""


class EnvSeriesPoint(BaseModel):
    ts: float
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    std: Optional[float] = None
    n: int = 0
    n_excluded: int = 0
    #: 桶内最差状态。**抽稀时按严重度合并，不是取样** —— 一张缩小的趋势图
    #: 绝不能把真的发生过的越限藏起来。
    worst_status: str = "ok"


class EnvSeriesResponse(BaseModel):
    sensor: str = ""
    unit: str = ""
    points: list[EnvSeriesPoint] = Field(default_factory=list)
    #: 实际返回的桶宽（秒）。请求点数不够时服务端会在 SQL 里加权再分桶，
    #: 这个数会大于原生桶宽。
    bucket_s_effective: float = 0.0
    #: 是否发生了再分桶。前端据此决定要不要提示"当前为概览视图"。
    thinned: bool = False
    total: int = 0
    degraded: bool = False
    detail: str = ""


class SpectrumMeta(BaseModel):
    """一条谱的元数据。

    ``ctx_*`` 在**列表**里也带着，不只在详情里：挑一条谱是按「哪一条是在
    -1.2 V 隧穿下测的」来挑的，不是按行号。要它必须先看得见。

    每个 ``ctx_*`` 都可以是 ``None``，而 ``None`` 的意思是「当时读不到」，不是
    「零」/「关」。谱是永久记录，事后没有第二次机会补测。
    """

    id: int
    ts: float
    channel: str = "current"
    span_s: float = 0.0
    n_segments: int = 0
    fs_hz: float = 0.0
    f_lo_hz: float = 0.0
    f_hi_hz: float = 0.0
    n_points: int = 0
    unit: str = ""
    quietness: str = "quiet"
    ctx_bias_v: Optional[float] = None
    ctx_setpoint_a: Optional[float] = None
    ctx_zctrl_on: Optional[bool] = None
    #: 攒谱期间 bias / setpoint / Z 反馈有没有变过。``False`` 时上面那几个数
    #: 描述的是窗口的**起点**，这条谱本身是混合的。``None`` = 老行，没记过。
    ctx_stable: Optional[bool] = None
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None


class SpectraListResponse(BaseModel):
    """谱**元数据**列表 —— 数组不在这里。一次 500 条的 BLOB 是 1 MB 白流量。"""

    spectra: list[SpectrumMeta] = Field(default_factory=list)
    degraded: bool = False
    detail: str = ""


class SpectrumResponse(BaseModel):
    id: int = 0
    ts: float = 0.0
    channel: str = ""
    span_s: float = 0.0
    n_segments: int = 0
    fs_hz: float = 0.0
    unit: str = ""
    quietness: str = "quiet"
    freqs_hz: list[float] = Field(default_factory=list)
    psd: list[float] = Field(default_factory=list)
    ctx_bias_v: Optional[float] = None
    ctx_setpoint_a: Optional[float] = None
    ctx_zctrl_on: Optional[bool] = None
    ctx_stable: Optional[bool] = None
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None
    found: bool = False
    degraded: bool = False
    detail: str = ""


class RecorderSpectrumState(BaseModel):
    channel: str = ""
    n_accum: int = 0
    fs_hz: Optional[float] = None
    window_s: Optional[float] = None
    n_points: int = 0
    last_ts: Optional[float] = None
    #: I 谱专用：窗口早就到了却一直攒不够安静段（仪器一直很忙）。
    insufficient_quiet: bool = False
    #: Z 谱专用：上一轮 burst 的结果或跳过原因。
    last_result: dict = Field(default_factory=dict)


class EnvHistoryStatus(BaseModel):
    enabled: bool = False
    spectra_enabled: bool = False
    z_enabled: bool = False
    recording: bool = False
    bucket_s: float = 0.0
    sink: dict = Field(default_factory=dict)
    spectra: dict[str, RecorderSpectrumState] = Field(default_factory=dict)
    spectra_written: int = 0
    sweep: dict = Field(default_factory=dict)
    store: dict = Field(default_factory=dict)
    degraded: bool = False
    detail: str = ""


class ExperimentEnvRow(BaseModel):
    ts: float = 0.0
    timestamp: str = ""
    sensor: str = ""
    value: Optional[float] = None
    unit: str = ""
    status: str = "ok"
    sample_id: Optional[str] = None


class ExperimentEnvironmentResponse(BaseModel):
    """一个实验的环境读数。

    ``source`` 说明这批数字是从哪来的：``raw`` 是逐条原始读数，``buckets`` 是
    统计桶（原始行已经超过保留期被清掉了）。诚实标注是必须的 —— 否则用户
    看到一条比预期粗的曲线会以为是采样出了问题。
    """

    experiment_id: str = ""
    source: str = "raw"                    # raw | buckets | none
    rows: list[ExperimentEnvRow] = Field(default_factory=list)
    points: list[EnvSeriesPoint] = Field(default_factory=list)
    sensors: list[str] = Field(default_factory=list)
    #: 早于这个时刻的原始逐条读数已被清扫（None = 未清扫过 / 未知）。
    raw_pruned_before: Optional[str] = None
    bucket_s_effective: float = 0.0
    degraded: bool = False
    detail: str = ""


__all__ = [
    "EnvHistoryKnob", "EnvHistoryConfigResponse",
    "EnvSensorSummary", "EnvSensorsResponse",
    "EnvSeriesPoint", "EnvSeriesResponse",
    "SpectrumMeta", "SpectraListResponse", "SpectrumResponse",
    "RecorderSpectrumState", "EnvHistoryStatus",
    "ExperimentEnvRow", "ExperimentEnvironmentResponse",
]
