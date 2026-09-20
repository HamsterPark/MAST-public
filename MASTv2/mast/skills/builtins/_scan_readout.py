"""扫描中读出的共用处理：通道、尺度与有效行。

Scan_FrameDataGrab 接受 Scan_BufferGet 返回的信号索引，不是缓冲位。
错误索引可能导致回包布局不匹配，应先检查通道选择。
像素尺度由当前视野与像素数计算，不能使用外部假定或旧扫描设置。
未完成行按有效数据规则排除，不进入图像分析。
这些转换由共用入口维护，避免不同技能各自推断而产生分歧。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: ``Signals_NamesGet`` 里 Z 与电流的写法（小写比对）。
Z_NAME_HINTS = ("z (m)", "z(m)", "z")
CURRENT_NAME_HINTS = ("current (a)", "current(a)", "current")


def first_values(record) -> Any:
    """Nanonis ``(header, body, values)`` 里的 values。读不到回 ``None``。"""
    if record is None or getattr(record, "error", ""):
        return None
    parsed = getattr(record, "return_value", None)
    if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
        return None
    return parsed[2]


class ScanReadout:
    """一次「看现在扫到哪儿了」需要的全部上下文。

    ``why`` 非空即表示这次解不出来 —— 那时候 :attr:`ok` 是 False，
    **不要**拿默认值接着算。
    """

    __slots__ = ("z_index", "nm_per_px", "pixels", "lines", "channels",
                 "width_m", "z_source", "why")

    def __init__(self) -> None:
        self.z_index: Optional[int] = None
        self.nm_per_px: Optional[float] = None
        self.pixels: Optional[int] = None
        self.lines: Optional[int] = None
        self.channels: list[int] = []
        self.width_m: Optional[float] = None
        self.z_source: str = ""
        self.why: str = ""

    @property
    def ok(self) -> bool:
        return not self.why and self.z_index is not None and self.nm_per_px is not None

    def as_dict(self) -> dict:
        return {"channel_index": self.z_index, "channel_source": self.z_source,
                "nm_per_px": self.nm_per_px, "pixels": self.pixels,
                "lines": self.lines, "scan_channels": list(self.channels),
                "frame_width_m": self.width_m}


def resolve_readout(context, calls: list, *,
                    forced_channel: int = -1) -> ScanReadout:
    """去问仪器：Z 的信号索引、像素数、视野 → nm/px。

    读不到就把 ``why`` 填上并返回 —— **不猜一个 30，也不猜一个 5 nm**。
    """
    from mast.io.nanonis_files import parse_buffer_get

    out = ScanReadout()

    rec_buf = context.safe_call("Scan_BufferGet")
    calls.append(rec_buf)
    if getattr(rec_buf, "error", ""):
        out.why = "读不到扫描缓冲：%s" % rec_buf.error
        return out
    buf = parse_buffer_get(getattr(rec_buf, "return_value", None))
    if not buf or not buf.get("channel_indexes"):
        out.why = ("扫描缓冲的回包读不懂 —— 不去猜通道号。原始回包：%s"
                   % str(getattr(rec_buf, "return_value", None))[:200])
        return out
    out.channels = [int(c) for c in buf["channel_indexes"]]
    out.pixels = buf.get("pixels")
    out.lines = buf.get("lines")

    if forced_channel >= 0:
        out.z_index, out.z_source = int(forced_channel), "参数指定"
    else:
        rec_names = context.safe_call("Signals_NamesGet")
        calls.append(rec_names)
        names = first_values(rec_names)
        if isinstance(names, (list, tuple)):
            flat = [str(n[0] if isinstance(n, (list, tuple)) and n else n).strip().lower()
                    for n in names]
            for c in out.channels:
                if 0 <= c < len(flat) and flat[c] in Z_NAME_HINTS:
                    out.z_index, out.z_source = c, "按信号名 Z 解出"
                    break
        if out.z_index is None:
            # 兜底：缓冲里**不是**电流的那一路。会在结果里说明是兜底来的 ——
            # 一个不说自己是兜底的兜底值，正是最难发现的那种错。
            non_cur = [c for c in out.channels if c != 0]
            if non_cur:
                out.z_index, out.z_source = non_cur[0], "兜底：缓冲里非电流的那一路"
    if out.z_index is None:
        out.why = ("扫描缓冲里找不到 Z 通道，缓冲里只有 %s。"
                   "把 Z 加进扫描通道，或者显式指定通道号。" % out.channels)
        return out

    rec_fr = context.safe_call("Scan_FrameGet")
    calls.append(rec_fr)
    vals = first_values(rec_fr)
    if isinstance(vals, (list, tuple)) and len(vals) >= 3:
        try:
            out.width_m = float(vals[2])
        except (TypeError, ValueError):
            out.width_m = None
    if not out.width_m or not out.pixels:
        out.why = ("算不出 nm/px：视野 %s、像素 %s —— 宁可拒绝，"
                   "也不按一个猜的尺度做判读。" % (out.width_m, out.pixels))
        return out
    out.nm_per_px = float(out.width_m) * 1e9 / float(out.pixels)
    return out


def grab_frame(context, channel_index: int, direction: int, calls: list):
    """抓一次帧缓冲（不停扫）。回 ``(ndarray_or_None, why)``。"""
    from mast.skills.builtins.scan_frame import _grab_channel_array

    arr, rec = _grab_channel_array(context, int(channel_index), int(direction),
                                   shape_2d=True)
    calls.append(rec)
    if arr is None:
        return None, (getattr(rec, "error", "") or "回包读不懂")
    return arr, ""
