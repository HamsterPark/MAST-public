"""告警投递策略与判定分离。critical 始终逐条送达；普通规则按最新一条与计数折叠；静音规则仍落库并在界面可见。未分类规则默认送达，避免新增规则静默丢失。默认静音的工频提示针对环境排查，仪器或针尖异常仍进入上下文。"""
from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum


class DeliveryClass(str, Enum):
    """一条告警对 agent 上下文的命运。"""

    #: 每条单独列出,永不折叠、永不静音。
    ALWAYS = "always"
    #: 同 rule 在窗口内折叠成「最新一条 + 计数」。
    FOLD = "fold"
    #: 不进 agent 上下文(仍入表、仍在 UI 可见)。
    MUTE = "mute"


#: 出厂静音的规则。**这是默认值,不是注册表** —— 见模块 docstring。
#: 只有 ``line_hum``:它的处方指向房间(接地/屏蔽),不指向针尖。
DEFAULT_MUTED_RULES: frozenset[str] = frozenset({"line_hum"})

#: 一次注入最多列几条。CRITICAL 先占位,占不下的 WARN 被丢掉并如实说明丢了几条
#: —— 悄悄截断会让「没有更多」和「还有很多没给你看」长得一模一样。
DEFAULT_MAX_ITEMS: int = 8

#: 往回看多久。比这更老的告警不再注入 —— 一个 agent 刚接手时不该被半小时前
#: 已经处理完的历史刷屏。**同一个窗口也是折叠窗口**:窗口内同 rule 的多条合成
#: 一行 + 一个计数。
#:
#: 刻意**只有一个窗口**。曾经有两个(回看 900 s + 折叠 600 s),而那会开出一个
#: 没人负责的缝:一条 700 s 前的 warn,若同 rule 有更新的一条,它既不被显示、
#: 也不被计数、也不被标记 —— 每一轮都被重新扫出来,再被重新丢掉。
#: 「代码看了它一眼,什么都没决定,然后悄悄扔掉」正是本仓反复出事的形状,
#: 而两个窗口的差值越大,落进那条缝的行越多。
DEFAULT_LOOKBACK_S: float = 900.0


@dataclass(frozen=True)
class AlertRouting:
    """投递策略的不可变快照。与 :class:`~mast.monitoring.thresholds.MonitorThresholds`
    同一形状(live-read holder):改一次,下一次注入即生效,不重启、不重建图。

    刻意**不**并进 ``MonitorThresholds``:那个 dataclass 全是 float,
    ``to_mapping()`` 对每个字段做 ``float(v)``,一个规则名集合塞进去会被静默毁掉。
    """

    #: 不打扰 agent 的规则名。只对**非 critical** 的行有效(见 :func:`classify`)。
    muted_rules: frozenset[str] = DEFAULT_MUTED_RULES
    max_items: int = DEFAULT_MAX_ITEMS
    #: 回看窗口 = 折叠窗口。只有一个,理由见 :data:`DEFAULT_LOOKBACK_S`。
    lookback_s: float = DEFAULT_LOOKBACK_S

    @classmethod
    def from_mapping(cls, m: dict | None) -> "AlertRouting":
        """从(可能残缺的)映射构造 —— 对持久化设置容错。

        未知键忽略、缺键取默认、类型不对的值**整个忽略而不是强转**:一个
        ``muted_rules="line_hum"``(字符串而非列表)若被当成可迭代对象,会静音
        掉 ``l``/``i``/``n``/``e``… 这类「能跑的错版本」正是本仓的常见死法。
        """
        if not m:
            return cls()
        clean: dict = {}

        raw = m.get("muted_rules")
        if isinstance(raw, (list, tuple, set, frozenset)):
            names = {str(r).strip() for r in raw}
            clean["muted_rules"] = frozenset(n for n in names if n)

        v = m.get("lookback_s")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            clean["lookback_s"] = float(min(86400.0, max(1.0, float(v))))

        v = m.get("max_items")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            clean["max_items"] = int(min(50, max(1, int(v))))

        return replace(cls(), **clean)

    def to_mapping(self) -> dict:
        return {
            "muted_rules": sorted(self.muted_rules),
            "max_items": int(self.max_items),
            "lookback_s": float(self.lookback_s),
        }


def classify(rule: str, level: str,
             routing: "AlertRouting | None" = None) -> DeliveryClass:
    """一条告警该怎么投递。纯函数,永不抛。

    ``level`` 来自**告警行自己**(store 的 ``alerts.level`` 列),不是从 ``rule``
    名字反查出来的。这是 ALWAYS 分支不可被配置绕过的全部原因:静音表在
    ``critical`` 这条路上根本没有被读到,所以「不小心把 saturation 静音了」不是
    一件需要校验来防住的事,而是一件走不到的事。
    """
    routing = routing or get_alert_routing()
    if str(level or "").strip().lower() == "critical":
        return DeliveryClass.ALWAYS
    try:
        if str(rule or "") in routing.muted_rules:
            return DeliveryClass.MUTE
    except Exception:  # noqa: BLE001 — 策略读坏了也不能让告警消失
        return DeliveryClass.FOLD
    return DeliveryClass.FOLD


_LOCK = threading.Lock()
_ACTIVE = AlertRouting()


def get_alert_routing() -> AlertRouting:
    """当前生效的快照(无锁的原子引用读,与 thresholds 同一形状)。"""
    return _ACTIVE


def set_alert_routing(m: "dict | AlertRouting | None") -> AlertRouting:
    """换掉生效快照。``None`` / 空 → 恢复默认。"""
    global _ACTIVE
    new = m if isinstance(m, AlertRouting) else AlertRouting.from_mapping(m)
    with _LOCK:
        _ACTIVE = new
    return new


__all__ = [
    "AlertRouting",
    "DeliveryClass",
    "classify",
    "get_alert_routing",
    "set_alert_routing",
    "DEFAULT_MUTED_RULES",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_LOOKBACK_S",
]
