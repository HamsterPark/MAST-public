"""环境历史 —— environment-history query endpoints.

Thin relay over :mod:`mast.envhistory`. House style identical to
``routes/monitoring.py``:

  - every handler returns ``degraded=True`` instead of a 500 — the app must boot
    standalone with no core wired, and the envhistory package may not even be
    importable;
  - the backend is LAZY-imported inside each handler;
  - **ZERO TCP, and zero serial.** Everything here reads SQLite. Looking at a
    year of temperature history must not cost the instrument a single packet;
  - thinning happens in the STORE (in SQL, by weighted re-aggregation), never
    here and never in the browser — see ``EnvHistoryStore.series``;
  - literal paths are registered before ``{spectrum_id}`` paths, or the path
    parameter shadows them.

``/env-history/status`` is the one handler that also reaches into the live
recorder (an in-memory object). It still issues no I/O of any kind.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, TypeVar

from fastapi import APIRouter, Request

from mast.api.schemas_env_history import (
    EnvHistoryConfigResponse,
    EnvHistoryKnob,
    EnvHistoryStatus,
    EnvSensorSummary,
    EnvSensorsResponse,
    EnvSeriesPoint,
    EnvSeriesResponse,
    ExperimentEnvironmentResponse,
    ExperimentEnvRow,
    RecorderSpectrumState,
    SpectraListResponse,
    SpectrumMeta,
    SpectrumResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["env-history"])

T = TypeVar("T")

#: 服务端硬上限。前端可以要更少，不能要更多 —— 一个手打的 max_points=10^7
#: 会让 SQLite 把一年的桶全序列化出来。
_MAX_SERIES_POINTS = 4000
_MAX_SPECTRA_ROWS = 500
_MAX_RAW_ROWS = 5000


def _guarded(fn: Callable[[], T], degraded_factory: Callable[[str], T]) -> T:
    """Run a handler body; on any failure return its degraded shape."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — an endpoint must never 500 here
        logger.debug("env-history endpoint degraded", exc_info=True)
        return degraded_factory(str(exc))


def _store():
    from mast.envhistory.store import get_store
    return get_store()


def _tri_bool(v) -> Optional[bool]:
    """SQLite 的 0 / 1 / NULL → ``False`` / ``True`` / ``None``。

    ``bool(None)`` 是 ``False``，而这两件事在一条永久记录上完全不同：
    「Z 反馈是关的」是一个断言，「当时读不到 Z 反馈状态」不是。
    同一个坑在 ``monitoring.commission._tri_bool`` 里已经咬过一次
    （``0 is False`` 为假，让整个标定分组恒空）。
    """
    return None if v is None else bool(v)


def _clamp(v, lo: int, hi: int, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _point(d: dict) -> EnvSeriesPoint:
    return EnvSeriesPoint(
        ts=float(d.get("ts") or 0.0),
        mean=d.get("mean"), min=d.get("min"), max=d.get("max"), std=d.get("std"),
        n=int(d.get("n") or 0), n_excluded=int(d.get("n_excluded") or 0),
        worst_status=str(d.get("worst_status") or "ok"),
    )


# ── config / status ─────────────────────────────────────────────────────────

@router.get("/env-history/config", response_model=EnvHistoryConfigResponse)
def env_history_config() -> EnvHistoryConfigResponse:
    """生效设置 + 设置页要渲染的旋钮目录。

    目录是数据而不是前端里的一张硬编码键表，所以加一个旋钮是纯后端改动。
    **注意写回时必须整 dict 送全部键** —— 设置存储是整体替换，只送改动的那个
    键等于把其余键全部重置为默认。
    """

    def body() -> EnvHistoryConfigResponse:
        from mast.envhistory.thresholds import (
            get_env_history_thresholds,
            knob_catalog,
        )
        th = get_env_history_thresholds()
        return EnvHistoryConfigResponse(
            enabled=bool(th.enabled),
            spectra_enabled=bool(th.spectra_enabled),
            z_enabled=bool(th.z_enabled),
            bucket_s=float(th.bucket_s),
            raw_keep_days=float(th.eh_raw_keep_days),
            spectrum_interval_s=float(th.eh_spectrum_interval_s),
            knobs=[EnvHistoryKnob(**k) for k in knob_catalog()],
        )

    return _guarded(body, lambda e: EnvHistoryConfigResponse(degraded=True, detail=e))


@router.get("/env-history/status", response_model=EnvHistoryStatus)
def env_history_status(request: Request) -> EnvHistoryStatus:
    """记录器在不在跑、攒到哪了、库多大、上次清扫删了多少。"""

    def body() -> EnvHistoryStatus:
        from mast.envhistory.recorder import get_recorder
        rec = get_recorder() or getattr(
            getattr(request.app.state, "ctx", None), "env_history", None)
        if rec is None:
            # 记录器没接上（纯 API 模式 / core 未装配）。库仍然可读 —— 历史
            # 数据的价值不依赖于此刻有没有人在记。
            from mast.envhistory.thresholds import get_env_history_thresholds
            th = get_env_history_thresholds()
            return EnvHistoryStatus(
                enabled=bool(th.enabled), spectra_enabled=bool(th.spectra_enabled),
                z_enabled=bool(th.z_enabled), recording=False,
                bucket_s=float(th.bucket_s), store=_store().storage_stats(),
            )
        st = rec.status()
        spectra = {k: RecorderSpectrumState(**v) for k, v in (st.get("spectra") or {}).items()}
        sink = st.get("sink") or {}
        return EnvHistoryStatus(
            enabled=bool(st.get("enabled")),
            spectra_enabled=bool(st.get("spectra_enabled")),
            z_enabled=bool(st.get("z_enabled")),
            # "在记录" = 总开关开着 且 sink 没有因为写失败而自禁。
            recording=bool(st.get("enabled")) and not bool(sink.get("disabled")),
            bucket_s=float(st.get("bucket_s") or 0.0),
            sink=sink, spectra=spectra,
            spectra_written=int(st.get("spectra_written") or 0),
            sweep=st.get("sweep") or {}, store=st.get("store") or {},
        )

    return _guarded(body, lambda e: EnvHistoryStatus(degraded=True, detail=e))


# ── series ──────────────────────────────────────────────────────────────────

@router.get("/env-history/sensors", response_model=EnvSensorsResponse)
def env_history_sensors() -> EnvSensorsResponse:
    """历史里出现过的序列 —— 给选择器用。"""

    def body() -> EnvSensorsResponse:
        rows = _store().list_sensors()
        return EnvSensorsResponse(sensors=[
            EnvSensorSummary(
                sensor=str(r.get("sensor") or ""), unit=str(r.get("unit") or ""),
                n_buckets=int(r.get("n_buckets") or 0),
                first_ts=r.get("first_ts"), last_ts=r.get("last_ts"),
            ) for r in rows
        ])

    return _guarded(body, lambda e: EnvSensorsResponse(degraded=True, detail=e))


@router.get("/env-history/series", response_model=EnvSeriesResponse)
def env_history_series(sensor: str, since: Optional[float] = None,
                       until: Optional[float] = None,
                       max_points: Optional[int] = None) -> EnvSeriesResponse:
    """一条序列的统计桶。点数超限时服务端加权再分桶（不是取样）。"""
    mp = _clamp(max_points, 10, _MAX_SERIES_POINTS, _MAX_SERIES_POINTS) \
        if max_points else _MAX_SERIES_POINTS

    def body() -> EnvSeriesResponse:
        d = _store().series(str(sensor), since=since, until=until, max_points=mp)
        return EnvSeriesResponse(
            sensor=str(d.get("sensor") or sensor), unit=str(d.get("unit") or ""),
            points=[_point(p) for p in (d.get("points") or [])],
            bucket_s_effective=float(d.get("bucket_s_effective") or 0.0),
            thinned=bool(d.get("thinned")), total=int(d.get("total") or 0),
        )

    return _guarded(body, lambda e: EnvSeriesResponse(sensor=sensor, degraded=True,
                                                      detail=e))


# ── spectra (literal path first, then the {spectrum_id} one) ────────────────

@router.get("/env-history/spectra", response_model=SpectraListResponse)
def env_history_spectra(channel: Optional[str] = None,
                        since: Optional[float] = None,
                        until: Optional[float] = None,
                        limit: int = 200) -> SpectraListResponse:
    """噪声谱快照的元数据列表（不含数组）。"""
    lim = _clamp(limit, 1, _MAX_SPECTRA_ROWS, 200)

    def body() -> SpectraListResponse:
        rows = _store().spectra_query(channel=channel, since=since,
                                      until=until, limit=lim)
        return SpectraListResponse(spectra=[SpectrumMeta(**r) for r in rows])

    return _guarded(body, lambda e: SpectraListResponse(degraded=True, detail=e))


@router.get("/env-history/spectra/{spectrum_id}", response_model=SpectrumResponse)
def env_history_spectrum(spectrum_id: int) -> SpectrumResponse:
    """一条完整的谱（频率 + 功率数组）。"""

    def body() -> SpectrumResponse:
        d = _store().spectrum(int(spectrum_id))
        if not d:
            return SpectrumResponse(id=int(spectrum_id), found=False)
        return SpectrumResponse(
            id=int(d.get("id") or 0), ts=float(d.get("ts") or 0.0),
            channel=str(d.get("channel") or ""), span_s=float(d.get("span_s") or 0.0),
            n_segments=int(d.get("n_segments") or 0),
            fs_hz=float(d.get("fs_hz") or 0.0), unit=str(d.get("unit") or ""),
            quietness=str(d.get("quietness") or "quiet"),
            freqs_hz=list(d.get("freqs_hz") or []), psd=list(d.get("psd") or []),
            ctx_bias_v=d.get("ctx_bias_v"), ctx_setpoint_a=d.get("ctx_setpoint_a"),
            # 三态，不是布尔：SQLite 存的是 0/1/NULL，而 NULL 的意思是「当时读不到
            # Z 反馈状态」。压成 False 就等于替一条永久记录断言「反馈是关的」。
            ctx_zctrl_on=_tri_bool(d.get("ctx_zctrl_on")),
            ctx_stable=_tri_bool(d.get("ctx_stable")),
            experiment_id=d.get("experiment_id"), sample_id=d.get("sample_id"),
            found=True,
        )

    return _guarded(body, lambda e: SpectrumResponse(id=int(spectrum_id),
                                                     degraded=True, detail=e))


# ── per-experiment environment ──────────────────────────────────────────────

@router.get("/experiments/{experiment_id}/environment",
            response_model=ExperimentEnvironmentResponse)
def experiment_environment(request: Request, experiment_id: str,
                           sensor: Optional[str] = None,
                           limit: int = 2000) -> ExperimentEnvironmentResponse:
    """一个实验期间的环境读数。

    这是 ``ExperimentStorage.get_environment_history`` 的**第一个调用方** ——
    在此之前那张表以 2 秒一条的速度只写不读（设计文档
    ``experiment_folder_persistence.md`` §13 记了这笔欠账）。

    两级回退，并且**如实标注用的是哪一级**：

    1. ``raw`` —— 逐条原始读数，带样品归属。保留期内的实验走这里。
    2. ``buckets`` —— 原始行已被清扫，回退到统计桶。曲线更粗但覆盖全时段。

    不标注的话，用户看到一条比预期粗的曲线只会以为是采样出了问题。
    """
    lim = _clamp(limit, 1, _MAX_RAW_ROWS, 2000)

    def body() -> ExperimentEnvironmentResponse:
        ctx = getattr(request.app.state, "ctx", None)
        storage = getattr(ctx, "experiment_storage", None)
        out = ExperimentEnvironmentResponse(experiment_id=str(experiment_id))
        if storage is None:
            out.source = "none"
            out.degraded = True
            out.detail = "no experiment storage wired"
            return out

        names = [sensor] if sensor else [
            str(r.get("sensor_name") or "")
            for r in storage.list_environment_sensors(experiment_id=experiment_id)
        ]
        names = [n for n in names if n]
        out.sensors = sorted(set(names))

        rows: list[ExperimentEnvRow] = []
        for nm in names:
            for r in storage.get_environment_history(
                    nm, experiment_id=experiment_id, limit=lim):
                rows.append(ExperimentEnvRow(
                    ts=_iso_to_epoch(r.get("timestamp")),
                    timestamp=str(r.get("timestamp") or ""),
                    sensor=str(r.get("sensor_name") or nm),
                    value=r.get("value"), unit=str(r.get("unit") or ""),
                    status=str(r.get("status") or "ok"),
                    sample_id=r.get("sample_id"),
                ))
        rows.sort(key=lambda x: x.timestamp)
        out.rows = rows[:lim]
        out.raw_pruned_before = _pruned_before()

        if rows:
            out.source = "raw"
            return out

        # 原始行没有了（或从来没写过）。回退到统计桶，按实验的时间窗取。
        since, until = _experiment_window(storage, experiment_id)
        pick = sensor or (out.sensors[0] if out.sensors else None)
        if pick is None:
            out.source = "none"
            return out
        d = _store().series(pick, since=since, until=until,
                            max_points=_MAX_SERIES_POINTS)
        out.source = "buckets"
        out.points = [_point(p) for p in (d.get("points") or [])]
        out.bucket_s_effective = float(d.get("bucket_s_effective") or 0.0)
        if not out.sensors:
            out.sensors = [pick]
        return out

    return _guarded(body, lambda e: ExperimentEnvironmentResponse(
        experiment_id=str(experiment_id), source="none", degraded=True, detail=e))


def _iso_to_epoch(ts) -> float:
    """``environment_log`` 的本地无时区 ISO 串 → epoch 秒。解析不了返回 0。"""
    if not ts:
        return 0.0
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(ts)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _pruned_before() -> Optional[str]:
    """上一次清扫的截止时刻（有的话）。早于它的逐条读数已经不在了。"""
    try:
        from mast.envhistory.recorder import get_recorder
        rec = get_recorder()
        if rec is None:
            return None
        return (rec.status().get("sweep") or {}).get("cutoff")
    except Exception:  # noqa: BLE001
        return None


def _experiment_window(storage, experiment_id: str) -> tuple[Optional[float], Optional[float]]:
    """实验的 [开始, 最后活动] 时间窗，epoch 秒。取不到就返回 (None, None)。"""
    try:
        exp = storage.get_experiment(experiment_id) or {}
    except Exception:  # noqa: BLE001
        return None, None
    lo = _iso_to_epoch(exp.get("started_at") or exp.get("created_at")) or None
    hi = _iso_to_epoch(exp.get("last_active_at") or exp.get("updated_at")) or None
    return lo, hi


__all__ = ["router"]


# ── 标定参考曲线(只读)──────────────────────────────────────────────────
#
# 落在这条路由下,因为它和环境历史是同一类东西:**长期留存、给人看、零 TCP**。
# 但归属不同 —— 曲线是**仪器域**(<project_root>/calibration/),不是实验域,
# 也不是 envhistory 的 SQLite。这里只是查询入口。
#
# ⚠️ 每条曲线都带 ``describe``:「实测于某机某针某日 —— 参考曲线不是出厂常数」。
# 展示层必须把它显示出来;不显示,曲线看起来就像一条物理常数,而它不是。

@router.get("/calibration/curves")
def calibration_curves(kind: Optional[str] = None) -> dict:
    """标定参考曲线列表(含数据点)。零 TCP、零串口 —— 读的是 JSON 文件。"""
    def body() -> dict:
        from mast.core.calibration_curves import describe, load_curves

        out = []
        for c in load_curves(kind=kind):
            out.append({
                "kind": c.get("kind"),
                "conditions": c.get("conditions") or {},
                "meta": c.get("meta") or {},
                "points": c.get("points") or [],
                "n_points": len(c.get("points") or []),
                # 出处那句话由后端给,不让每个前端各写一遍(写漏了就等于没写)。
                "describe": describe(c),
                "path": c.get("_path"),
            })
        return {"curves": out, "count": len(out), "degraded": False}

    return _guarded(body, lambda e: {"curves": [], "count": 0,
                                     "degraded": True, "detail": e})


@router.get("/calibration/kinds")
def calibration_kinds() -> dict:
    """已知曲线种类,以及每种的**可比性条件** —— 前端筛选与「能不能用」都靠它。"""
    def body() -> dict:
        from mast.core.calibration_curves import MATCH_KEYS

        return {"kinds": {k: list(v) for k, v in MATCH_KEYS.items()},
                "degraded": False}

    return _guarded(body, lambda e: {"kinds": {}, "degraded": True, "detail": e})
