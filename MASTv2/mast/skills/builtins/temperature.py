"""读温度 —— 以及「读不到」时到底是哪一种读不到。

在此之前温度只有一个私有取数口（``CoreRuntime._latest_temperature_k``），
``mast/agents/**`` 与 ``mast/skills/**`` 对它**零引用**：驱动齐全、数据落库，
但没有任何工作流能问一句「现在几度」。等降温、判「到温了没有」这类条件因此
只能靠人看面板。

这个模块是**只读**的，不加任何能力：不取仪器令牌、不碰 Nanonis、不写任何东西。
判读逻辑在 :mod:`mast.core.temperature`（一次计算、几个受众 —— 与真空互锁、
扫描地图分析同一形状）。
"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

#: 每种「没有值」该做什么。放在**数据里**而不是只写在描述里 —— 描述会被裁剪、
#: 会被总结，而这一句是跟着答案一起到调用方手上的。
_WHAT_TO_DO = {
    "no_sensor": (
        "这台机器上没有真的温度传感器（只有占位实现）。**等下去永远等不到** —— "
        "不要轮询、不要重试；需要温度判据的步骤应当直接拒绝并告诉用户。"
    ),
    "unavailable": (
        "已安装温度传感器，但此刻拿不到读数。可能是串口被其他程序占用，"
        "或归档循环尚未完成一次读取。"
        "**可能会好**：可以等，但要请用户去看端口占用，不要无限轮询。"
    ),
    "unknown_channel": (
        "你指名的那个通道这台机器上没有。看 available_channels 里的名字，"
        "**不是**「没有温度计」。"
    ),
    "ambiguous_channel": (
        "你给的名字同时对上了不止一个通道。从 available_channels 里挑一个完整名字。"
    ),
    "no_source": (
        "MAST 自己这一侧没接上温度源（独立 API 进程，或环境监控没建起来）。"
        "与仪器无关；重启/接线才会变。"
    ),
}


class GetTemperature(BaseSkill):
    """读当前温度，并说明读不到时是哪一种读不到。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetTemperature",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读当前温度(K)，带**读数年龄**和**出处**。只读，不碰仪器。\n"
                "`value_k` 为 null 时看 `reason`,三种「读不到」含义完全不同、"
                "要做的事相反:\n"
                "  • `no_sensor` —— 这台机器没有温度计,**等下去永远等不到**,别轮询;\n"
                "  • `unavailable` —— 有温度计但此刻不给数(常见:串口被别的程序占着),"
                "可以等,并请用户看端口;\n"
                "  • `no_source` —— MAST 这一侧没接上,与仪器无关。\n"
                "`age_s` 是这个读数多旧(秒);**`age_s` 为 null 表示不知道多旧,不是刚读的**。"
                "要判「够不够新」就传 `max_age_s`,结果里的 `freshness` 会给 "
                "fresh/stale/unknown 三态。\n"
                "多个温度通道时默认取样品台(SPM/stage/sample/tip/cryo);"
                "磁体杜瓦的温度**不等于**样品台的温度,要哪个就用 `channel` 指名"
                "(名字见 `available_channels`)。"
            ),
            parameters=[
                ParameterSpec(
                    name="channel",
                    type="str",
                    description=(
                        "指名读哪个温度通道（传感器名，如 'SPM' / 'Magnet'）。"
                        "不传 = 自动选样品台优先的那个。名字见结果里的 "
                        "available_channels。"
                    ),
                    required=False,
                    default=None,
                ),
                ParameterSpec(
                    name="max_age_s",
                    type="float",
                    description=(
                        "多旧算旧（秒）。传了才会给 freshness 判定；"
                        "**不传就不判** —— 这个阈值只有调用方知道，"
                        "这里不替你拍一个默认值。"
                    ),
                    unit="s",
                    required=False,
                    default=None,
                    min_value=0.0,
                ),
            ],
            estimated_duration_s=0.1,
            composition_level=0,
            tags=["temperature", "environment", "read", "cryo"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from mast.core import temperature as temp

        channel = params.get("channel")
        channel = str(channel).strip() if channel is not None else None
        reading = temp.latest_temperature(channel or None)

        data = reading.as_dict()
        if reading.reason:
            data["what_to_do"] = _WHAT_TO_DO.get(reading.reason, "")

        # 有哪些通道可选。``None`` = 问不到（源没接上），与「一个都没有」不同 ——
        # 前者不该让 agent 得出「这台机器没温度计」的结论。
        chans = temp.channels()
        if chans is None:
            data["available_channels"] = None
        else:
            data["available_channels"] = [
                {"name": c.name, "unit": c.unit, "status": c.status,
                 "driver": c.driver, "real": c.real, "value_k": c.kelvin()}
                for c in chans
            ]

        # 陈旧判定是 explicit-only：不传 max_age_s 就不给 freshness 字段，
        # 而不是给一个「用默认阈值算出来的」结论（那种结论看起来和真的一样）。
        if params.get("max_age_s") is not None:
            try:
                data["freshness"] = reading.freshness(float(params["max_age_s"]))
                data["max_age_s"] = float(params["max_age_s"])
            except (TypeError, ValueError):
                data["freshness"] = "unknown"

        # 查询跑通了就是 success —— 「读不到」是一个**答案**，不是工具坏了。
        # 报成 error 会让一次寻常的「还没装温度计」看起来像故障，
        # 并且把 reason 里那句可执行的信息一起丢掉（同 GetChamberPressure）。
        return SkillResult(skill_name="GetTemperature", success=True, data=data)


__all__ = ["GetTemperature"]
