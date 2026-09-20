"""环境历史记录器的 operator 旋钮 —— live-read holder。

形制与 :mod:`mast.monitoring.thresholds` 完全一致：一个进程级不可变快照，
写在启动 hydration 与每次 ``POST /api/settings``，读在每次翻桶 / 每段 / 每次
清扫。改一个数下一拍生效，不重启、不重载。envhistory 层从不 import settings,
接线是单向的。

这些**不在** ``MASTConfig`` 里，理由与电流监控相同：它们都是仪器运行期间要重
调的量，放进 config 既要求重启，又给同一批数字造出第二个真源。

默认值的取舍
------------

* ``eh_enabled`` **默认开**。P0 的标量链路全程零 TCP、写的是本地 SQLite 的
  微量行、失败自禁；而 ``environment_log`` 无限增长是一个**现存**的生产问题
  （2 s × N 传感器 = 43,200 行/天/传感器，且至今零读取方），保留策略默认关就
  永远修不到它。
* ``eh_z_enabled`` **默认关**。它是本方案唯一新增 TCP 行为，且依赖尚未上过真机
  的 Osci2T 补丁。真机验收通过后再翻默认。
* ``eh_raw_keep_days`` 是**唯一一个会删数据的旋钮**。它删的是 DB 里的原始行；
  实验文件夹里的 CSV 是权威记录且**永不**被本子系统触碰，所以这里删过头的代价
  是"查询变粗"，不是"数据没了"。
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, fields, replace

#: 在设置 UI 里露出的键。前端从 :func:`knob_catalog` 动态渲染，所以加旋钮
#: 只需要动这个元组。
EDITABLE_KEYS: tuple[str, ...] = (
    "eh_enabled",
    "eh_spectra_enabled",
    "eh_z_enabled",
    "eh_bucket_s",
    "eh_raw_keep_days",
    "eh_sweep_interval_s",
    "eh_spectrum_interval_s",
    "eh_spectrum_bins",
    "eh_spectrum_min_segments",
    "eh_spectrum_accum_every_s",
    "eh_z_interval_s",
    "eh_z_burst_s",
)

FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "eh_enabled": (0.0, 1.0),
    "eh_spectra_enabled": (0.0, 1.0),
    "eh_z_enabled": (0.0, 1.0),
    # 下界 10 s：比环境监控自己的 2 s 采样周期粗，否则一个"桶"里只有几个点,
    # std 没有意义而行数逼近原始表 —— 那就白聚合了。
    "eh_bucket_s": (10.0, 3600.0),
    # 下界 1 天：这是删除阈值,允许 0 就等于"写完立刻删",一次误操作能把当天
    # 的原始行清空（桶还在,但分钟内的细节回不来了）。
    "eh_raw_keep_days": (1.0, 3650.0),
    "eh_sweep_interval_s": (600.0, 86400.0),
    "eh_spectrum_interval_s": (300.0, 86400.0),
    "eh_spectrum_bins": (60.0, 480.0),
    "eh_spectrum_min_segments": (3.0, 64.0),
    "eh_spectrum_accum_every_s": (1.0, 120.0),
    "eh_z_interval_s": (300.0, 86400.0),
    # 上界 120 s：burst 期间 Osci2T 与电流监控抢 data 角色锁,电流监控会把等待
    # 诚实记成 gap。一次 burst 拖到几分钟就等于让针尖监控在那段时间近乎失明。
    "eh_z_burst_s": (5.0, 120.0),
}

#: 中文标签 + 一句提示，给设置 UI。
KNOB_LABELS: dict[str, tuple[str, str]] = {
    "eh_enabled": ("记录环境历史", "1=后台聚合并保留环境参数历史,0=完全关闭"),
    "eh_spectra_enabled": ("记录电流噪声谱", "定期从电流监控的段流里存一条全谱快照"),
    "eh_z_enabled": ("记录 Z 噪声谱", "需要 Osci2T 双通道;会短暂占用 data 通道"),
    "eh_bucket_s": ("聚合粒度(秒)", "多长时间聚成一个统计桶;桶永久保留"),
    "eh_raw_keep_days": ("原始读数保留(天)", "超期的逐条读数滚删,只留统计桶(告警行永久保留)"),
    "eh_sweep_interval_s": ("清扫间隔(秒)", "多久检查一次过期的原始读数"),
    "eh_spectrum_interval_s": ("噪声谱间隔(秒)", "多久存一条噪声谱快照"),
    "eh_spectrum_bins": ("噪声谱点数", "对数分箱后的频点数;越多越细也越占地方"),
    "eh_spectrum_min_segments": ("噪声谱最少段数", "攒够这么多安静段才敢发一条快照"),
    "eh_spectrum_accum_every_s": ("噪声谱取样间隔(秒)", "每隔多久从段流里收一段进来攒"),
    "eh_z_interval_s": ("Z 噪声谱间隔(秒)", "多久采一次 Z 噪声谱"),
    "eh_z_burst_s": ("Z 采集时长(秒)", "每次 Z 采集持续多久;越长分辨率越好也越占通道"),
}

#: 实为布尔的旋钮（0/1）—— UI 渲染成开关。
BOOL_KEYS: frozenset[str] = frozenset({
    "eh_enabled", "eh_spectra_enabled", "eh_z_enabled",
})


@dataclass(frozen=True)
class EnvHistoryThresholds:
    """记录器旋钮的不可变快照。"""

    eh_enabled: float = 1.0
    eh_spectra_enabled: float = 1.0
    eh_z_enabled: float = 0.0
    eh_bucket_s: float = 60.0
    eh_raw_keep_days: float = 14.0
    eh_sweep_interval_s: float = 3600.0
    eh_spectrum_interval_s: float = 1800.0
    eh_spectrum_bins: float = 240.0
    eh_spectrum_min_segments: float = 8.0
    eh_spectrum_accum_every_s: float = 10.0
    eh_z_interval_s: float = 1800.0
    eh_z_burst_s: float = 30.0

    # ── 便利访问器（调用方读这些，不读裸 float） ──────────────────────
    @property
    def enabled(self) -> bool:
        return self.eh_enabled >= 0.5

    @property
    def spectra_enabled(self) -> bool:
        """I 谱开关。总开关关掉时它自动无效 —— 子开关从不越过总开关。"""
        return self.enabled and self.eh_spectra_enabled >= 0.5

    @property
    def z_enabled(self) -> bool:
        return self.enabled and self.eh_z_enabled >= 0.5

    @property
    def bucket_s(self) -> float:
        return max(1.0, float(self.eh_bucket_s))

    @property
    def spectrum_bins(self) -> int:
        return max(2, int(round(self.eh_spectrum_bins)))

    @property
    def spectrum_min_segments(self) -> int:
        return max(1, int(round(self.eh_spectrum_min_segments)))

    def to_mapping(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_mapping(cls, m: dict | None) -> "EnvHistoryThresholds":
        """从（可能残缺的）映射构造 —— 对持久化设置容错。

        未知键忽略、缺键取默认、数值按 :data:`FIELD_BOUNDS` 夹紧。布尔值被接受
        为 0/1，这样 UI 送来的 JSON ``true`` 做的是对的事。
        """
        if not m:
            return cls()
        known = {f.name for f in fields(cls)}
        clean: dict[str, float] = {}
        for k, v in m.items():
            if k not in known:
                continue
            if isinstance(v, bool):
                val = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                val = float(v)
            else:
                continue
            lo, hi = FIELD_BOUNDS.get(k, (float("-inf"), float("inf")))
            clean[k] = float(min(hi, max(lo, val)))
        return replace(cls(), **clean)


_LOCK = threading.Lock()
_ACTIVE = EnvHistoryThresholds()


def get_env_history_thresholds() -> EnvHistoryThresholds:
    """返回当前生效的不可变快照（无锁的原子引用读）。"""
    return _ACTIVE


def set_env_history_thresholds(
    m: "dict | EnvHistoryThresholds | None",
) -> EnvHistoryThresholds:
    """换掉生效快照。``None`` / 空 → 恢复默认。"""
    global _ACTIVE
    new = m if isinstance(m, EnvHistoryThresholds) else EnvHistoryThresholds.from_mapping(m)
    with _LOCK:
        _ACTIVE = new
    return new


def knob_catalog() -> list[dict]:
    """描述每个旋钮给设置 UI：范围、默认、当前值。

    以数据形式下发，前端因此不硬编码键名清单 —— 往 :data:`EDITABLE_KEYS` 加一
    项就够让它出现在界面上。
    """
    active = get_env_history_thresholds().to_mapping()
    defaults = EnvHistoryThresholds().to_mapping()
    out: list[dict] = []
    for key in EDITABLE_KEYS:
        lo, hi = FIELD_BOUNDS.get(key, (0.0, 0.0))
        label, hint = KNOB_LABELS.get(key, (key, ""))
        out.append({
            "key": key,
            "label_zh": label,
            "hint_zh": hint,
            "min": float(lo),
            "max": float(hi),
            "step": 0.0,
            "default": float(defaults.get(key, 0.0)),
            "value": float(active.get(key, 0.0)),
            "is_bool": key in BOOL_KEYS,
        })
    return out


__all__ = [
    "EnvHistoryThresholds",
    "get_env_history_thresholds",
    "set_env_history_thresholds",
    "knob_catalog",
    "EDITABLE_KEYS",
    "FIELD_BOUNDS",
    "BOOL_KEYS",
]
