"""EstimateScanDuration —— 估算一帧扫描要多久（纯计算，不与仪器通信）。

Nanonis 的一帧 = 行数 ×（正扫 + 回扫）。每一行针尖要走两倍扫描宽度，所以
单行耗时 = 2 × 宽度 / 针尖线速度，再加上每行的固定开销（换行时的稳定等待之类）。

排一晚的扫描计划、或者判断「这一帧值不值得扫」之前，先知道它要花多久。
像素数不改变耗时（线速度一定时，像素多只是采得更密），但决定每个像素的积分时间，
一起报出来：积分时间太短，图像噪声会变大。

这是 contrib 的 Python 技能范例：category=ANALYSIS、safety_level=AUTO、带量纲参数
都写了 unit 与 min/max、描述里只教 SI 前缀写法、不碰执行上下文。
"""

from __future__ import annotations

from mast.core.si_quantity import format_si_readable, parse_quantity
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

_NAME = "EstimateScanDuration"


class EstimateScanDuration(BaseSkill):
    """估算单帧扫描耗时（纯计算）。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "估算一帧扫描要多久：行数 ×（正扫 + 回扫）÷ 针尖线速度，另报每像素积分时间。"
                "纯计算，不与仪器通信；排扫描计划、估算一晚能扫几帧时用。"
            ),
            parameters=[
                ParameterSpec(
                    name="width_m",
                    type="float",
                    description="扫描宽度（快扫方向），写成带 SI 前缀的字符串，如 '50n'、'1u'。",
                    unit="m",
                    min_value=1e-9,
                    max_value=1e-5,
                ),
                ParameterSpec(
                    name="lines",
                    type="int",
                    description="一帧的行数。",
                    min_value=1,
                    max_value=8192,
                ),
                ParameterSpec(
                    name="speed_m_per_s",
                    type="float",
                    description="针尖线速度，写成带 SI 前缀的字符串，如 '200n'（即 200 nm/s）。",
                    unit="m/s",
                    min_value=1e-10,
                    max_value=5e-6,
                ),
                ParameterSpec(
                    name="pixels",
                    type="int",
                    description="每行像素数；不改变耗时，只决定每像素的积分时间。",
                    required=False,
                    default=256,
                    min_value=1,
                    max_value=8192,
                ),
                ParameterSpec(
                    name="overhead_s_per_line",
                    type="float",
                    description="每行的固定开销（秒），例如换行时的稳定等待。",
                    unit="s",
                    required=False,
                    default=0.0,
                    min_value=0.0,
                    max_value=10.0,
                ),
            ],
            preconditions=[],
            capabilities=frozenset(),
            estimated_duration_s=0.01,
            composition_level=2,
            tags=["scan", "planning", "time", "analysis"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        try:
            width = parse_quantity(params.get("width_m"), strict=False, what="width_m")
            speed = parse_quantity(params.get("speed_m_per_s"), strict=False, what="speed_m_per_s")
            overhead = parse_quantity(params.get("overhead_s_per_line", 0.0), strict=False,
                                      what="overhead_s_per_line")
            lines = int(params.get("lines"))
            pixels = int(params.get("pixels", 256))
        except (TypeError, ValueError) as exc:
            return SkillResult(skill_name=_NAME, success=False, error=f"参数读不懂：{exc}")
        if width <= 0 or speed <= 0 or lines < 1 or pixels < 1 or overhead < 0:
            return SkillResult(
                skill_name=_NAME,
                success=False,
                error="扫描宽度、线速度、行数、像素数都必须为正，每行开销不能为负。",
            )
        sweep_s = width / speed                  # 单程
        line_s = 2.0 * sweep_s + overhead        # 正扫 + 回扫 + 每行开销
        frame_s = line_s * lines
        dwell_s = sweep_s / pixels
        return SkillResult(
            skill_name=_NAME,
            success=True,
            data={
                "frame_time_s": frame_s,
                "line_time_s": line_s,
                "pixel_dwell_s": dwell_s,
                "lines": lines,
                "pixels": pixels,
            },
            summary=(
                f"一帧约 {_human_duration(frame_s)}：每行 {format_si_readable(line_s, 's')}"
                f"（正扫 + 回扫），每像素积分 {format_si_readable(dwell_s, 's')}。"
            ),
        )


def _human_duration(seconds: float) -> str:
    """``3725.0`` → ``"1 h 02 min"``；``256.0`` → ``"4 min 16 s"``；``12.3`` → ``"12.3 s"``。"""
    if seconds < 60:
        return f"{seconds:.1f} s"
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {secs:02d} s"
