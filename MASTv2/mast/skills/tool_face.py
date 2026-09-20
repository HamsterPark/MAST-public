"""工具面的三道门 —— **唯一定义处**。

哪些技能不该出现在 agent 的工具表里，这个问题有三个互相独立的答案：

============== ================================== =========================
门              问的是什么                          谁拥有它
============== ================================== =========================
硬件模块        这台机器有没有这个硬件               ``hardware_modules``
高级能力        这个动作能不能绕过一层保护           ``advanced_capabilities``
**订阅**        **这位用户这段时间在做什么**       ``subscription``
============== ================================== =========================

三道门的语义完全不同，但**并集只该算一次**。在这个模块存在之前它被算了三遍：

* ``agents/instrument_control/tools.py::build_instrument_skill_tools`` —— 装配真身；
* 同文件 ``expected_wrap_fingerprint`` —— 生效探针；
* ``webui/agents_api.py::_compute_agent_tools`` —— UI 镜像。

三份里漏改任何一份的后果都不一样，而且都不报错：漏改探针 ⇒ ``fingerprint_matches``
永远说「没跟上」；漏改镜像 ⇒ 界面上列的工具表**不是模型手上那份**（那个文件自己的
注释把这件事叫做「同一件事的另一种说谎方式」）。所以并集搬到这里，三处都调它。

**这个模块永不抛。** 一道读不出来的门不是一道门 —— 它 fail-open（不过滤）并把门名
记进 ``unreadable``，让调用方能如实说出「本次没过滤，因为某道门读不出来」。这条与
``agents_api`` 原来那句 "a gate we cannot read is not a gate" 是同一条，只是现在
三处共享同一份实现，不会有一处忘了 try。

注意三道门的**失败方向不同**，这是有意的：硬件/高级能力门在**自己模块内部**
fail-closed（读不懂的持久化状态回默认全关），订阅门 fail-open（读不懂就全订阅）。
理由见 ``subscription`` 的模块 docstring：前两者关错了只是少几个必然失败的工具，
订阅关错了是 agent 一夜之间失能，而订阅本就不是保护。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceSkip:
    """一次并集计算的结果 —— 以及它是**怎么**算出来的。"""

    hardware: frozenset[str] = frozenset()
    advanced: frozenset[str] = frozenset()
    unsubscribed: frozenset[str] = frozenset()
    #: 读不出来的门名（``()`` = 三道门都读到了）。非空即表示本次**少过滤**了。
    unreadable: tuple[str, ...] = ()

    @property
    def names(self) -> frozenset[str]:
        return self.hardware | self.advanced | self.unsubscribed

    def __contains__(self, name: str) -> bool:
        return (name in self.hardware or name in self.advanced
                or name in self.unsubscribed)

    def describe(self) -> str:
        bits = [f"{len(self.hardware)} 因硬件模块关闭",
                f"{len(self.advanced)} 因高级能力关闭",
                f"{len(self.unsubscribed)} 因未订阅"]
        if self.unreadable:
            bits.append(f"⚠ 读不出来的门：{'/'.join(self.unreadable)}（本次未过滤）")
        return "、".join(bits)


def compute(all_names: Iterable[str] | None = None) -> FaceSkip:
    """三道门的并集。``all_names`` = 市场全集（订阅门要靠它算补集）。

    不传 ``all_names`` 时订阅门不过滤（它没法在不知道全集的情况下算补集），
    硬件/高级能力两道门照常 —— 这样一个只关心那两道门的老调用点仍然拿到正确答案。
    """
    hw: frozenset[str] = frozenset()
    adv: frozenset[str] = frozenset()
    unsub: frozenset[str] = frozenset()
    bad: list[str] = []

    try:
        from mast.skills.hardware_modules import disabled_skill_names as _hw
        hw = _hw()
    except Exception as exc:  # noqa: BLE001
        logger.warning("工具面：硬件模块门读不出来（%s）—— 本次不按它过滤", exc)
        bad.append("硬件模块")

    try:
        from mast.skills.advanced_capabilities import disabled_skill_names as _adv
        adv = _adv()
    except Exception as exc:  # noqa: BLE001
        logger.warning("工具面：高级能力门读不出来（%s）—— 本次不按它过滤", exc)
        bad.append("高级能力")

    try:
        from mast.skills.subscription import unloaded_skill_names as _sub
        unsub = _sub(all_names)
    except Exception as exc:  # noqa: BLE001
        logger.warning("工具面：订阅门读不出来（%s）—— 本次不按它过滤", exc)
        bad.append("订阅")

    return FaceSkip(hardware=hw, advanced=adv, unsubscribed=unsub,
                    unreadable=tuple(bad))


def skip_names(all_names: Iterable[str] | None = None) -> frozenset[str]:
    """只要并集本身的调用方用这个。"""
    return compute(all_names).names


def names_from_registry(registry) -> frozenset[str]:
    """注册表里现在有哪些技能名（= 订阅门要用的「市场全集」）。"""
    try:
        return frozenset(m.name for m in registry.list_skills())
    except Exception as exc:  # noqa: BLE001
        logger.warning("工具面：注册表列不出技能名（%s）", exc)
        return frozenset()


__all__ = ["FaceSkip", "compute", "skip_names", "names_from_registry"]
