"""提供用于远程诊断的只读桌面截图。

截图须在能够访问交互桌面的进程中进行；SSH、服务进程或无人登录时可能
没有有效桌面。错误应区分无交互桌面与截图失败，避免误导排查。

整张图片只有一种颜色时标记 blank=True 并说明原因，不能仅因成功写出 PNG
就宣称截图有效。此模块只读取当前屏幕，不点击、不移动、不修改窗口。
"""
from __future__ import annotations

import io
import logging
import sys
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: 回传前缩到这个宽度。桌面截图原图可能好几 MB，而诊断要看的是「波形动没动」
#: 「哪个模块开着」这类信息，1600 px 足够，再大只是让每次调用更贵。
DEFAULT_MAX_WIDTH = 1600


@dataclass
class Capture:
    """一次截图尝试的结果。``png`` 为空就看 ``reason``。"""

    png: bytes = b""
    width: int = 0
    height: int = 0
    #: 原始（缩放前）尺寸，用来判断报回来的分辨率是不是那个 1024×768 的假值
    raw_width: int = 0
    raw_height: int = 0
    #: 整张图只有一种颜色 —— 抓到了，但抓到的是空的
    blank: bool = False
    ok: bool = False
    #: 机器可判的失败类别：no_desktop / unsupported_platform / no_pillow / error
    reason: str = ""
    detail: str = ""
    notes: list[str] = field(default_factory=list)


def _is_uniform(img) -> bool:
    """整张图是不是只有一种颜色。

    用 ``getextrema()``：每个通道的 (min, max) 相等就说明该通道恒定，全部通道都
    恒定 = 纯色图。比逐像素比较便宜得多，而且不需要把整张图拉成 list。
    """
    ex = img.convert("RGB").getextrema()
    return all(lo == hi for lo, hi in ex)


def capture(max_width: int = DEFAULT_MAX_WIDTH) -> Capture:
    """抓当前桌面。**绝不抛异常** —— 诊断工具坏掉不该变成 500。"""
    if not sys.platform.startswith("win"):
        return Capture(reason="unsupported_platform",
                       detail=f"桌面截图目前只在 Windows 上实现（当前 {sys.platform}）")
    try:
        from PIL import ImageGrab
    except Exception as exc:  # noqa: BLE001
        return Capture(reason="no_pillow", detail=f"Pillow 不可用：{exc}")

    try:
        img = ImageGrab.grab(all_screens=True)
    except OSError as exc:
        # 这是从无桌面会话（SSH / 服务）里调它的典型下场。单独归类，别让人以为
        # 是功能坏了。
        return Capture(
            reason="no_desktop",
            detail=f"抓不到交互桌面：{exc}",
            notes=["MAST 必须跑在用户登录的交互会话里（双击启动）才能截图；"
                   "从 SSH 或以服务方式运行时抓不到桌面（session 0 隔离）。"])
    except Exception as exc:  # noqa: BLE001
        return Capture(reason="error", detail=f"{type(exc).__name__}: {exc}")

    out = Capture(raw_width=img.width, raw_height=img.height)
    try:
        out.blank = _is_uniform(img)
        if out.blank:
            out.notes.append(
                "整张截图只有一种颜色 —— 抓到了但内容是空的。常见于没有交互桌面"
                "（此时分辨率往往报成 1024×768 这类默认值），而不是屏幕真的是空白。")
        if max_width and img.width > max_width:
            h = max(1, round(img.height * max_width / img.width))
            img = img.resize((max_width, h))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        out.png = buf.getvalue()
        out.width, out.height = img.width, img.height
        # blank 时仍然 ok=True：图是真抓回来了，"抓到的是空的"由 blank 表达。
        # 把它算成失败会丢掉「分辨率是多少」这个诊断线索。
        out.ok = True
    except Exception as exc:  # noqa: BLE001
        return Capture(reason="error", detail=f"编码失败：{type(exc).__name__}: {exc}",
                       raw_width=img.width, raw_height=img.height)
    finally:
        try:
            img.close()
        except Exception:  # noqa: BLE001
            pass
    return out
