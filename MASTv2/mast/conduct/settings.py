"""ConductDirector 的 operator 旋钮 —— live-read holder。

形制与 :mod:`mast.monitoring.thresholds` / :mod:`mast.envhistory.thresholds`
完全一致:一个进程级不可变快照,写在启动 hydration 与每次设置写入,读在每次
启动决策与每次面板渲染。conduct 层从不 import settings,接线是单向的。

## 为什么只有这几个旋钮

设计 ``campaign_director_design.md`` 里绝大多数「可调的数」都在 **ConductSpec**
里(tick 间隔、等待周期、预算、熔断次数)——那是**每份 conduct 自己的**参数,
approve 时随 params 一起冻结。放进全局设置会造出第二个真源,而且改一个全局数
会悄悄改掉一份**正在跑**的 conduct 的行为。

留在这里的只有两个真正属于「这台机器」而不属于「这次实验」的量:

* ``cd_enabled`` —— Director 这条常驻线程**跑不跑**。**默认 0(关)**:
  M1-c 之前系统里根本没有这条线程,默认关 = 逐字节等于今天。真机验收
  (设计 §9 的 M1)通过之后再由用户翻开。
* ``cd_stall_grace_s`` —— 停滞告警阈值里的**余量**。设计 §6-1 定的判据是
  ``heartbeat 年龄 > max(3×tick_interval, 当前步 timeout_s + 余量)``;
  前两个量来自 spec,只有「余量」是「这台机器上的 TCP 有多慢」这个本地事实。

## 「关」是什么意思

``cd_enabled=0`` 时:线程不起、状态机不推进。**但 API 路由照常在**——
读得到已有 conduct 的状态、abort/pause 意图照常入队(且 abort 照常立刻置
per-run abort Event)。理由是「能停不能解=死锁」那条:一个功能被关掉,不该
连带把**停下它**的路一起关掉。
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, fields, replace

#: 在设置 UI 里露出的键。前端从 :func:`knob_catalog` 动态渲染。
EDITABLE_KEYS: tuple[str, ...] = (
    "cd_enabled", "cd_stall_grace_s", "cd_autonomy", "cd_ignition_delay_s")

FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "cd_enabled": (0.0, 1.0),
    # 下界 30 s:比任何一次 TCP 往返都宽。设成 0 等于「步一超时就报停滞」,
    # 而设计 §3-7 已经写明**不做超时杀步**——一个必然误报的告警会被无视,
    # 于是真的停滞发生时也没人看。上界 2 h:再长就等于没有停滞告警。
    "cd_stall_grace_s": (30.0, 7200.0),
    # 自主度档位 0=attended / 1=supervised / 2=autonomous。
    # 上界跟着 AUTONOMY_LEVELS 走,别在这里写死一个 2 —— 加一档而边界没跟上,
    # 新档位会被静默夹回去,而设置页会显示成「设上了」。
    "cd_autonomy": (0.0, 2.0),
    # supervised 档的撤销窗。下界 0 = 不等(等于把 supervised 用成 autonomous,
    # 允许但要显式);上界 1 h —— 再长就不是「来得及后悔」而是「批了等于没批」。
    "cd_ignition_delay_s": (0.0, 3600.0),
}

#: 中文标签 + 一句提示,给设置 UI。
KNOB_LABELS: dict[str, tuple[str, str]] = {
    "cd_enabled": ("启用 conduct 指挥线程",
                   "1=常驻线程按 conduct 状态机推进,0=完全不起线程(路由仍可读可中止)"),
    "cd_stall_grace_s": ("停滞告警余量(秒)",
                         "心跳年龄超过 max(3×tick, 当前步超时+这个数) 才报停滞"),
    "cd_autonomy": ("自主度(谁能点头)",
                    "0=有人值守(人批人 ack) 1=半自主(agent 可批,点火前留撤销窗) "
                    "2=自主(包络内即批即跑)。**三档下安全包络完全一样**——"
                    "参数超界照样拒绝、DANGEROUS 照样过闸,变的只是谁点这个头。"
                    "模板可以把自己钉得更严,取两者中更严的那个。"),
    "cd_ignition_delay_s": ("半自主的撤销窗(秒)",
                            "agent 批准之后等这么久才点火,期间 abort 能撤回。"
                            "只在自主度=1 时起作用"),
}

#: 真的是布尔的旋钮(0/1)——UI 渲染成开关。
BOOL_KEYS: frozenset[str] = frozenset({"cd_enabled"})


@dataclass(frozen=True)
class ConductKnobs:
    """不可变快照。"""

    #: **默认关。** 见模块 docstring。
    cd_enabled: float = 0.0
    cd_stall_grace_s: float = 300.0
    #: 自主度档位(0/1/2)。**默认 0 = attended**,与这一档落地之前逐字节等价。
    cd_autonomy: float = 0.0
    cd_ignition_delay_s: float = 600.0

    @property
    def enabled(self) -> bool:
        return self.cd_enabled >= 0.5

    @property
    def autonomy(self) -> str:
        """当前自主度的**名字**。数值只活在设置层,代码层一律用名字。"""
        from mast.conduct.autonomy import from_code

        return from_code(self.cd_autonomy)

    def to_mapping(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_mapping(cls, m: "dict | None") -> "ConductKnobs":
        """从(可能只有一部分的)映射建。未知键忽略,缺的用默认,数值夹到边界内。

        布尔也收:UI 传 JSON ``true`` 时做对的事。**非数值非布尔的值丢弃而不是
        强转** —— 一个 ``"1"`` 字符串强转成 1.0 会让「设错了类型」看起来像
        「设对了」。
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
_ACTIVE = ConductKnobs()


def get_conduct_knobs() -> ConductKnobs:
    """当前快照(无锁的原子引用读)。"""
    return _ACTIVE


def set_conduct_knobs(m: "dict | ConductKnobs | None") -> ConductKnobs:
    """换快照。``None`` / 空 = 回默认(即**关**)。"""
    global _ACTIVE
    new = m if isinstance(m, ConductKnobs) else ConductKnobs.from_mapping(m)
    with _LOCK:
        _ACTIVE = new
    return new


def knob_catalog() -> list[dict]:
    """给设置 UI 的旋钮清单:边界、默认值、当前值、是不是开关。

    字段名与 ``monitoring`` / ``envhistory`` 的目录**逐字一致**(``label_zh`` /
    ``hint_zh`` / ``is_bool`` / ``step``)—— 前端有一份共用的旋钮渲染件
    (``frontend/src/lib/knobs.ts``),第三份目录换一套字段名就等于逼它长出
    第二条分支。消费者是 ``GET /api/conducts/config``。
    """
    cur = get_conduct_knobs().to_mapping()
    default = ConductKnobs().to_mapping()
    out: list[dict] = []
    for key in EDITABLE_KEYS:
        lo, hi = FIELD_BOUNDS.get(key, (0.0, 0.0))
        label, hint = KNOB_LABELS.get(key, (key, ""))
        out.append({"key": key, "label_zh": label, "hint_zh": hint,
                    "min": float(lo), "max": float(hi), "step": 0.0,
                    "default": float(default.get(key, 0.0)),
                    "value": float(cur.get(key, 0.0)),
                    "is_bool": key in BOOL_KEYS,
                    "choices": _choices_for(key)})
    return out


def _choices_for(key: str) -> list[dict]:
    """有限档位的旋钮把档位名一起发出去（2026-08-27）。

    ``cd_autonomy`` 是 0/1/2 三档，不是一个连续量 —— 让用户在一个数字框里
    敲「2」，等于要求他记住那三个数分别是什么。档位名的**唯一真源**是
    :data:`mast.conduct.autonomy.AUTONOMY_LEVELS` 与 :func:`describe`；在前端
    再写一份中文档位名，就是那个名字的第二处定义，而两处定义迟早会岔开
    （本仓为镜像清单付过好几次税）。

    其余旋钮回空列表 —— 目录里多一个恒空的字段，比让消费者去分辨「这个目录
    有没有 choices」要省事，而且 monitoring / envhistory 的渲染件对空列表天然
    是 no-op。
    """
    if key != "cd_autonomy":
        return []
    try:
        from mast.conduct.autonomy import AUTONOMY_LEVELS, describe

        return [{"value": float(i), "label_zh": describe(level)}
                for i, level in enumerate(AUTONOMY_LEVELS)]
    except Exception:  # noqa: BLE001 — 目录坏了不许拖垮整个设置页
        return []


__all__ = ["ConductKnobs", "get_conduct_knobs", "set_conduct_knobs",
           "knob_catalog", "EDITABLE_KEYS", "FIELD_BOUNDS", "KNOB_LABELS",
           "BOOL_KEYS"]
