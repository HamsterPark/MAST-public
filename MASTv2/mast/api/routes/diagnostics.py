"""GET /api/diagnostics — the refusal ledger, read by a human.

Some failures were previously impossible to diagnose, because the system
recorded what SUCCEEDS and nothing about what is REFUSED — a call refused
outright, a run stopping partway through, and an agent spinning on a
precondition or safety gate looked identical from outside, with no way to
tell which layer or state was responsible.

``core.diagnostics`` now writes every refusal and every skip. This is the window
onto it. Degrade-safe: an empty ledger means nothing has been refused — which is a
true statement, not a broken one.
"""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, Query

from mast.api.schemas_diagnostics import (
    DiagnosticsResponse,
    DiagnosticEntry,
    DiagnosticsSummary,
    ScreenshotResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["diagnostics"])

# Grouped so the UI can offer "只看拒绝" without knowing every layer's name.
_REFUSALS = ("precondition_block", "safety_block", "mode_block",
             "abort_block", "hitl_reject")
_STEPS = ("step_skip", "step_fail", "step_abort")
# 「本来会打断工作、现在只通知」的那一类。刻意**不并进 _REFUSALS**
# —— 拒绝和放行是两件相反的事,混在一个筛选里会让「今天被拦了几次」这个问题答不出来。
_NOTICES = ("notice_only",)


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def get_diagnostics(
    limit: int = Query(200, ge=1, le=1000),
    group: str = Query("", description="refusals | steps | stall | notices | '' = 全部"),
    subject: str = Query("", description="按技能/步骤名过滤（子串）"),
    run_id: str = Query("", description="只看某一次运行"),
) -> DiagnosticsResponse:
    """Recent refusals + skips, newest first, with a shape-of-the-problem summary."""
    try:
        from mast.core.diagnostics import recent, summary

        kinds: "tuple[str, ...] | None" = None
        if group == "refusals":
            kinds = _REFUSALS
        elif group == "steps":
            kinds = _STEPS
        elif group == "stall":
            kinds = ("stall",)
        elif group == "notices":
            kinds = _NOTICES

        rows = recent(limit, kinds=kinds, subject=subject, run_id=run_id)
        s = summary()
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.warning("diagnostics read failed: %s", exc)
        return DiagnosticsResponse(degraded=True)

    return DiagnosticsResponse(
        entries=[
            DiagnosticEntry(
                seq=int(r.get("seq", 0)),
                t=float(r.get("t", 0.0)),
                kind=str(r.get("kind", "")),
                subject=str(r.get("subject", "")),
                reason=str(r.get("reason", "")),
                run_id=str(r.get("run_id", "") or ""),
                fields={k: v for k, v in r.items()
                        if k not in ("seq", "t", "kind", "subject", "reason", "run_id")},
            )
            for r in rows
        ],
        count=len(rows),
        summary=DiagnosticsSummary(
            total=int(s.get("total", 0)),
            by_kind=dict(s.get("by_kind") or {}),
            top_refusals=list(s.get("top_refusals") or []),
            log_path=str(s.get("log_path", "")),
        ),
        degraded=False,
    )


@router.get("/diagnostics/screenshot", response_model=ScreenshotResponse)
def get_screenshot(
    max_width: int = Query(1600, ge=160, le=7680,
                           description="回传前缩到这个宽度"),
) -> ScreenshotResponse:
    """读取用户当前桌面的截图，用于远程诊断。

    不点击、不移动、不修改窗口。截图需要交互桌面；服务或 SSH 会话
    可能没有有效桌面，错误原因会在结果中说明。

    截图包含 MAST 以外的窗口，使用与其他接口一致的认证和网络边界，
    每次调用记录日志。诊断失败也返回 200，由结果字段表达具体原因。
    """
    from mast.core.desktop_capture import capture

    cap = capture(max_width=max_width)
    logger.info("screenshot: ok=%s blank=%s %dx%d (raw %dx%d) reason=%s",
                cap.ok, cap.blank, cap.width, cap.height,
                cap.raw_width, cap.raw_height, cap.reason or "-")
    return ScreenshotResponse(
        ok=cap.ok,
        png_b64=base64.b64encode(cap.png).decode("ascii") if cap.png else "",
        width=cap.width, height=cap.height,
        raw_width=cap.raw_width, raw_height=cap.raw_height,
        blank=cap.blank, reason=cap.reason, detail=cap.detail,
        notes=list(cap.notes),
        degraded=not cap.ok,
    )
