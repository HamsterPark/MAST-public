"""Lock-in 常用参数组 —— 解析层,不是第二存储。

与 :mod:`mast.core.zctrl_presets` 同一套架构,同一条纪律:**数字不经过模型**。
组里的每个值都从仪器档案(``instrument_profile``)取,逐键标注来源;档案没填的键
**不下发**,不是补一个默认值下去。

调制侧 phase 不由本组下发：目标固件或配置未必提供该写入能力，
不能靠重试同值来证明可用。参数组只包含已声明的频率与幅度。
相位操作走独立的解调侧 ``LockIn_DemodPhasSet`` 与 ``AutoPhase`` 路径，
同样须检查目标仪器的响应。
"""
from __future__ import annotations

import logging
from typing import Any

from mast.core import instrument_profile as _iprof

logger = logging.getLogger(__name__)

#: 保留组名。与 zctrl 的 ``approach`` / ``scan`` 同性质:名字固定,值来自档案。
PRESET_DIDV = "didv"
RESERVED_NAMES = (PRESET_DIDV,)

#: 组会下发的档案键 → (Nanonis 参数, 人读的来源说明)。
#:
#: ⚠️ **用的是仓里已有的两个键,没有新建。** 设计交底里建议的名字是
#: ``lockin_mod_frequency_hz`` / ``lockin_mod_amplitude_v``,而
#: ``lockin_mod_freq_hz`` / ``lockin_mod_amp_v`` **早就在 instrument_profile 里,
#: 而且已经在设置页渲染出来了**(``approach.py`` 记 dI/dV 标定时就在读后者)。
#: 新建一对同义键 = 同一个物理量两个来源,迟早各自漂移 —— 而且会白白触发那条
#: 「加用户可编辑键是双边动作」的陷阱。**先查仓里有没有人已经在答这个问题。**
#:
#: phase 不在参数组中；增加此能力前须核验目标仪器的操作面及写入语义。
_PROFILE_KEYS: "dict[str, tuple[str, str]]" = {
    "lockin_mod_freq_hz": ("frequency_hz", "仪器档案 lockin_mod_freq_hz"),
    "lockin_mod_amp_v": ("amplitude_v", "仪器档案 lockin_mod_amp_v"),
}


class PresetRejected(RuntimeError):
    """组不可用,并且说得出为什么 —— 静默不下发是假成功。"""


class ResolvedLockInPreset:
    """一个解析好的组:下发什么、每个数从哪来、以及**哪些键故意没有**。"""

    __slots__ = ("name", "values", "sources", "unset", "notes")

    def __init__(self, name: str, values: "dict[str, float]",
                 sources: "dict[str, str]", unset: "list[str]",
                 notes: "list[str] | None" = None):
        self.name = name
        self.values = values
        self.sources = sources
        self.unset = unset
        self.notes = notes or []

    def skill_params(self, *, mod_on: bool = True) -> "dict[str, Any]":
        """``ConfigureLockIn`` 的参数 —— **只包含档案里真的填了的键**。

        没填的键连键名都不出现,而不是传一个 None 或 0:那两种写法在本仓都已经
        咬过人(声明默认值把「没说」变成「说了 0」)。``phase_deg`` 永远不在里面。
        """
        out: "dict[str, Any]" = {"mod_on": bool(mod_on)}
        out.update(self.values)
        return out

    def as_dict(self) -> "dict[str, Any]":
        return {
            "name": self.name,
            "values": dict(self.values),
            "sources": dict(self.sources),
            "unset": list(self.unset),
            "usable": self.usable,
            "why": self.why(),
            "notes": list(self.notes),
            # 明说出来,免得有人以为是漏了。
            "phase_deg": None,
            "phase_note": ("调制侧 phase 永不下发:本机 Modulate 区没有该字段,"
                           "TCP 写恒拒(写同值也拒)。相位在解调侧 Ref. Phase。"),
        }

    @property
    def usable(self) -> bool:
        """至少有一个值可下发,这个组才有意义。"""
        return bool(self.values)

    def why(self) -> str:
        if self.usable:
            got = "、".join(f"{k}={v}" for k, v in self.values.items())
            tail = (f";档案未填、因而不下发:{'、'.join(self.unset)}"
                    if self.unset else "")
            return f"将下发 {got}{tail}"
        return ("仪器档案里这一组一个值都没有配置(" + "、".join(self.unset) + ")。"
                "请在「设置 → 仪器档案 → lock-in」里填写 —— 这组数只由用户输入,"
                "不经模型。")


def resolve(name: str = PRESET_DIDV) -> ResolvedLockInPreset:
    """把组名解析成「下发什么」。未知组名直接拒绝,不猜。"""
    lowered = str(name or "").strip().lower()
    if lowered != PRESET_DIDV:
        raise PresetRejected(
            f"没有名为 {name!r} 的 lock-in 参数组;目前只有 "
            f"{'、'.join(RESERVED_NAMES)}。")

    values: "dict[str, float]" = {}
    sources: "dict[str, str]" = {}
    unset: "list[str]" = []
    for key, (param, origin) in _PROFILE_KEYS.items():
        raw = _iprof.get_config(key, None)
        if raw in (None, ""):
            unset.append(key)
            continue
        try:
            values[param] = float(raw)
        except (TypeError, ValueError):
            logger.warning("lock-in 参数组:%s=%r 不是数字,按未配置处理", key, raw)
            unset.append(key)
            continue
        sources[param] = origin
    return ResolvedLockInPreset(PRESET_DIDV, values, sources, unset)


def list_presets() -> "list[dict[str, Any]]":
    """所有保留组的解析结果(含 usable / why)—— 给只读盘点用。"""
    out = []
    for name in RESERVED_NAMES:
        try:
            out.append(resolve(name).as_dict())
        except PresetRejected as exc:
            out.append({"name": name, "usable": False, "why": str(exc)})
    return out


__all__ = ["PRESET_DIDV", "RESERVED_NAMES", "PresetRejected",
           "ResolvedLockInPreset", "resolve", "list_presets"]
