"""Typed models for the refusal ledger (GET /api/diagnostics).

One row = one thing the system REFUSED or SKIPPED, with enough context to answer
"why did nothing happen?" without reading a log file. See core/diagnostics.py for
why this exists at all.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class DiagnosticEntry(BaseModel):
    """One refusal / skip."""

    seq: int = 0
    t: float = 0.0
    kind: str = Field("", description="precondition_block | safety_block | "
                                      "mode_block | abort_block | hitl_reject | "
                                      "step_skip | step_fail | step_abort | stall")
    subject: str = Field("", description="被拒绝的对象：技能名 / 步骤 id / 工具名")
    reason: str = Field("", description="用户可读的原因——该层自己的措辞")
    run_id: str = ""
    fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="让这条记录可行动的一切：参数、失败的边界值、当时的仪器状态",
    )


class DiagnosticsSummary(BaseModel):
    """The SHAPE of the problem — one subject dominating a count IS the spin."""

    total: int = 0
    by_kind: Dict[str, int] = Field(default_factory=dict)
    top_refusals: List[Dict[str, Any]] = Field(default_factory=list)
    log_path: str = Field("", description="JSONL 落盘位置（空 = 磁盘日志不可用）")


class DiagnosticsResponse(BaseModel):
    entries: List[DiagnosticEntry] = Field(default_factory=list)
    count: int = 0
    summary: DiagnosticsSummary = Field(default_factory=DiagnosticsSummary)
    degraded: bool = True


class ScreenshotResponse(BaseModel):
    """一张用户桌面的截图，给远程诊断用。

    ``ok`` 与 ``blank`` 是**两件事**：``ok=True, blank=True`` 表示图抓回来了但内容
    是纯色的 —— 通常意味着 MAST 没跑在交互桌面会话里，而不是屏幕真的空白。把它
    读成「截图成功」是本仓库反复吃亏的那类误判（看着像真数据的降级）。
    """

    ok: bool = False
    png_b64: str = Field("", description="PNG 的 base64；失败时为空")
    width: int = 0
    height: int = 0
    raw_width: int = Field(0, description="缩放前的真实分辨率宽")
    raw_height: int = 0
    blank: bool = Field(False, description="整张图只有一种颜色 —— 抓到了但是空的")
    reason: str = Field("", description="失败类别：no_desktop / unsupported_platform "
                                        "/ no_pillow / error；成功时为空")
    detail: str = ""
    notes: List[str] = Field(default_factory=list)
    degraded: bool = True


__all__ = ["DiagnosticEntry", "DiagnosticsSummary", "DiagnosticsResponse",
           "ScreenshotResponse"]
