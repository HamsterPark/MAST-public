"""扫描分辨率(像素 / 线数)写入 —— 在这个文件之前,全系统没有任何路径能设它。

`ConfigureScan` 里唯一一处 ``Scan_BufferSet`` 调用是
``Scan_BufferSet(channel_indexes, 0, 0)``(imaging.py),后两个参数就是
pixels / lines,恒传 0 = 保持 Nanonis 现值。``GetScanBuffer`` 能读、
``experiment_prefs`` 有 ``scan_lines`` 字段、设置界面有输入框、提示词会把它渲染
给模型看 —— 但**没有任何工具能执行它**。这是「措辞承诺了做不到的事」的教科书
案例:界面和提示词一起许诺了一个不存在的能力。

本 skill 补上写入侧。

**为什么要先读再写**:``Scan_BufferSet`` 的第一个参数是通道索引列表,不是可选
的。想只改分辨率、不动通道,就必须先 ``Scan_BufferGet`` 把当前通道读回来再原样
写回去。传空列表会把采集通道清空 —— 扫出来的图一个通道都没有。
"""

from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import decode_reply, parse_buffer_get
from mast.skills.base import BaseSkill

_log = logging.getLogger(__name__)

#: 分辨率的合法范围。下界 16 是「再小就不是图」;上界 4096 是常见控制器上限。
#: 真机上是否强制 2 的幂 / 16 的倍数**尚未证实**(真机验证项),所以这里只做
#: 范围修剪,不做整除约束 —— 凭猜想加约束会拒掉本来合法的值。
PIXELS_MIN = 16
PIXELS_MAX = 4096

# 扫描缓冲区解析统一使用 mast.io.nanonis_files.parse_buffer_get。
# 数值数组元素可能为单元素元组，调用方不能自行对原始项强转 int。

def _dim(px, ln) -> str:
    """``512x512``,读不出来时写「未知」而不是印 ``Nonex None``。"""
    return "未知" if px is None or ln is None else f"{px}x{ln}"


class SetScanBuffer(BaseSkill):
    """Set the scan resolution (pixels per line and number of lines)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetScanBuffer",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置扫描分辨率：每行像素数与行数。"
                "会保留当前选中的采集通道（先回读、再原样写回）。"
                "优先用 ScanAt —— 它会按用户的分尺度策略表，为所请求的扫描"
                "尺寸挑好分辨率；只有当要求了某个具体分辨率、或你正在"
                "调试仪器时，才用这个技能。"
            ),
            parameters=[
                ParameterSpec(
                    name="pixels",
                    type="int",
                    description=(
                        "每条扫描线的像素数（如 256、512、1024）。"
                        f"取值范围 {PIXELS_MIN}..{PIXELS_MAX}。"
                    ),
                    unit="px",
                    required=True,
                    min_value=PIXELS_MIN,
                    max_value=PIXELS_MAX,
                ),
                ParameterSpec(
                    name="lines",
                    type="int",
                    description=(
                        "扫描行数。想要方形扫描框就省略它"
                        "（lines = pixels），这也是常规情形。"
                    ),
                    unit="px",
                    required=False,
                    min_value=PIXELS_MIN,
                    max_value=PIXELS_MAX,
                ),
            ],
            preconditions=["scan_not_running"],
            estimated_duration_s=0.5,
            composition_level=1,
            tags=["scan", "buffer", "resolution", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        pixels = int(params["pixels"])
        lines = params.get("lines")
        lines = int(lines) if lines is not None else pixels

        calls = []

        # 先读回当前通道 —— 不能凭空造一个通道列表,更不能传空列表(那会把采集
        # 通道清空,扫出来的图一个通道都没有)。
        rec_get = context.safe_call("Scan_BufferGet")
        calls.append(rec_get)
        if rec_get.error:
            return SkillResult(
                skill_name="SetScanBuffer",
                success=False,
                error=(
                    f"读取当前扫描缓冲失败,无法在保留通道的前提下改分辨率: "
                    f"{rec_get.error}"
                ),
                nanonis_calls=calls,
            )

        current = parse_buffer_get(rec_get.return_value)
        if current is None:
            return SkillResult(
                skill_name="SetScanBuffer",
                success=False,
                error=(
                    "Scan_BufferGet 回包格式无法解析,拒绝写入 —— 猜一个通道列表"
                    "写回去会破坏当前的采集配置。"
                ),
                nanonis_calls=calls,
                data={"raw": decode_reply(rec_get.return_value)},
            )

        channels = current["channel_indexes"]
        if not channels:
            return SkillResult(
                skill_name="SetScanBuffer",
                success=False,
                error=(
                    "当前没有选中任何采集通道。先用 ConfigureScan 选通道,"
                    "再设分辨率(本技能刻意不替你挑通道)。"
                ),
                nanonis_calls=calls,
            )

        rec_set = context.safe_call("Scan_BufferSet", channels, pixels, lines)
        calls.append(rec_set)
        if rec_set.error:
            return SkillResult(
                skill_name="SetScanBuffer",
                success=False,
                error=rec_set.error,
                nanonis_calls=calls,
            )

        # 回读确认。「硬件没报错」不等于「值真的变了」—— 这是本项目反复吃过亏
        # 的地方(硬件"停止"≠"达标")。回读失败不算整体失败,但要如实说没验上。
        applied_pixels = applied_lines = None
        verified = False
        rec_verify = context.safe_call("Scan_BufferGet")
        calls.append(rec_verify)
        if not rec_verify.error:
            after = parse_buffer_get(rec_verify.return_value)
            if after is not None:
                applied_pixels = after["pixels"]
                applied_lines = after["lines"]
                verified = (applied_pixels == pixels and applied_lines == lines)
                if not verified:
                    _log.warning(
                        "SetScanBuffer 回读不符: 请求 %sx%s,实际 %sx%s",
                        pixels, lines, applied_pixels, applied_lines,
                    )

        data = {
            "pixels": pixels,
            "lines": lines,
            "channel_indexes": channels,
            "previous_pixels": current["pixels"],
            "previous_lines": current["lines"],
            "applied_pixels": applied_pixels,
            "applied_lines": applied_lines,
            "verified": verified,
        }
        if not verified:
            data["warning"] = (
                "分辨率写入后回读未能确认"
                f"(请求 {pixels}x{lines},回读 {applied_pixels}x{applied_lines})"
                " —— 硬件可能对该值做了调整"
            )

        return SkillResult(
            skill_name="SetScanBuffer",
            success=True,
            data=data,
            summary=(
                # previous_* 可能是 None(回包里 pixels/lines 读不出来)—— 那不该
                # 拦下写入(通道保住了、新分辨率照写),但也不能印成 "Nonex None"。
                f"扫描分辨率 {_dim(current['pixels'], current['lines'])} → "
                f"{pixels}x{lines}"
                + ("" if verified else "(回读未确认)")
            ),
            nanonis_calls=calls,
        )
