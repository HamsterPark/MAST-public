"""环境参数历史记录 —— 后台、静默、有间隔地保留环境时间序列。

设计文档：``docs/v2/design/environment_history.md``

这个包只做四件事：**聚合、落盘、保留、查询**。它不采样（采样由既有的
:class:`mast.environment.EnvironmentMonitor` 与 :mod:`mast.monitoring` 做），
不判级，不告警，不碰仪器。

布局::

    thresholds.py  operator 旋钮 live-read holder（settings → 下一拍生效）
    quiet.py       纯函数：仪器此刻算不算"安静"（零 TCP，只读两个快照）
    buckets.py     纯对象：把 2 s 读数流增量聚合成 1 min 桶（Welford）
    spectra.py     纯对象：把段流的 PSD 压成对数分箱 median 快照
    store.py       独立 SQLite（buckets + spectra），范式同 monitoring/store.py
    sink.py        EnvironmentMonitor 的 sink：翻桶写库 + 节流触发清扫
    recorder.py    协调器单例：谱钩子入口、清扫单飞线程、status()
    zburst.py      Osci2T 双通道 burst，给 Z 噪音谱（默认关）

跨文件不变式（改任何一个文件之前先读这三条）：

* **零新增常驻采样线程。** 标量寄生在 EnvironmentMonitor 的 2 s 线程上，
  I 谱寄生在 CurrentMonitorService 的段流上，清扫与 Z burst 是有界单飞线程。
  项目已经为"孤儿后台线程撞正在拆的 Nanonis 端口"付过学费
  （``ExperimentalMonitor`` 至今没挂 reconnect）—— 我们干脆不养常驻线程。
* **纯记录，零判据。** 不复制 :mod:`mast.monitoring` 的任何告警规则；尤其
  Z 谱只记录不判级 —— 那四个电流判据全是先对纯高斯白噪声验过误报率才敢上的，
  Z 判据没验证过就不上。
* **绝不反噬实验。** 每个写入点自己吞异常；sink 写失败自禁 60 s；谱钩子在
  monitoring 侧包在 try/except 里。记录器坏掉的正确表现是"没有历史"，
  而不是"采集停摆"。
"""
from __future__ import annotations

__all__ = [
    "get_store",
    "set_store_for_test",
    "get_env_history_thresholds",
    "set_env_history_thresholds",
    "get_recorder",
    "set_recorder",
]


def __getattr__(name: str):
    """惰性 re-export：``import mast.envhistory`` 不拉 numpy/sqlite。

    API 层会探测这个包（monitoring 同款做法）；numpy 缺失时探测应当得到一个
    干净的 ImportError，而不是在 import 包的那一刻就炸。
    """
    if name in ("get_store", "set_store_for_test"):
        from mast.envhistory import store
        return getattr(store, name)
    if name in ("get_env_history_thresholds", "set_env_history_thresholds"):
        from mast.envhistory import thresholds
        return getattr(thresholds, name)
    if name in ("get_recorder", "set_recorder"):
        from mast.envhistory import recorder
        return getattr(recorder, name)
    raise AttributeError(name)
