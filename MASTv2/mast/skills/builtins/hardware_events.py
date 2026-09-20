"""让 agent 能看见「拦住它的那件事」—— 硬件事件的只读窗口。

## 为什么存在(2026-08-05 实机)

`buffer_hitl` 会因为缓冲区里一条 CRITICAL 事件拦下 agent 的写入类工具,拦截文案
点名了事件种类(如 `tip_quality_drop`)。而**被拦的那一方看不到任何证据**:

* agent 手上唯一相关的只读工具是 `read_latest_tip_status`,它读的是 **TipStatus**;
* 曾经那条 CRITICAL 来自 **`current_monitor`**(电流监控),它发的是一条
  `VisionEvent`,**并没有写 TipStatus** —— 于是那个工具老老实实返回
  `seqno=-1, tip=None`。

⇒ agent 被一件它**完全看不见**的事情拦住,只能猜。这正是这个仓一直在拆的形状:
「拦截理由存在,而被拦的人拿不到它」。本 skill 补的是**暴露侧**,
一行都不碰拦截逻辑。

## 与 v6.2「打断屏蔽」的关系(重要,别当成给拦截流程打的补丁)

v6.2 之后 `tip_quality_drop` 不再拦路(降级为面板通知)。**那不会让这个窗口变得
多余,反而让它变成主要入口**:监控仍在判、仍在发事件,只是不再打断谁 ——
于是「监控现在在报什么」从「解释我为什么被拦」升级成
**agent 自主决定要不要主动修针的常设读口**。

所以本 skill 的两块内容是**独立的**:`blocking` 那块在屏蔽之后大多数时候会是空的,
而 `recent` 那块的价值不减反增。**不要因为屏蔽上线就把它当成死代码删掉。**

## 三条设计约束

1. **绝不阻塞、绝不抛。** 它是给「已经被拦住的人」用的;一个在故障态下自己也失败的
   诊断工具没有意义。每一块独立 try,一块读不到不影响其余。
2. **「没有事件」与「读不到」必须分开。** 这条正是本 skill 存在的原因 ——
   `seqno=-1` 当初就被读成了「一切正常」。每一块都带自己的 `available` 与 `why`。
3. **指标优先取自事件自身。** `alerts.emit_critical` 已经把 `features`
   (rms / spike σ / sat_frac / max_step …)放进 payload 了,所以**不依赖监控数据库
   也能给出可核实的数字**。`cause_ref` 的库内查询只是**补充**,查不到不影响主答案。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: `cause_ref` 形如 ``current_monitor#132024`` —— 前缀是来源,井号后是段号。
_CAUSE_SEP = "#"


def _gate_block() -> dict:
    """谁在拦、拦的是什么。读的是 `buffer_hitl` 的**只读快照函数**。"""
    try:
        from mast.agents._shared.buffer_hitl import gate_states
        states = gate_states()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "why": f"读不到审批闸门状态: {exc}", "gates": []}
    blocking = [g for g in states if g.get("closed") or g.get("unresolved")]
    return {
        "available": True,
        "why": "",
        "any_blocking": bool(blocking),
        "gates": states,
    }


def _event_dict(ev: Any) -> dict:
    """一条事件 → 纯标量 dict。payload 原样带出 —— 里面就是可核实的数字。"""
    kind = getattr(ev, "kind", None)
    sev = getattr(ev, "severity", None)
    payload = getattr(ev, "payload", None) or {}
    ts_ns = getattr(ev, "t_mono_ns", None)
    out: dict[str, Any] = {
        "seqno": getattr(ev, "seqno", None),
        "kind": getattr(kind, "value", kind),
        "severity": getattr(sev, "value", sev),
        "cause_ref": getattr(ev, "cause_ref", None),
        "source": payload.get("source"),
        "signal": payload.get("signal"),
        # 这条是给人读的一句话,`alerts.summarize_zh` 生成,带具体数字。
        "summary_zh": payload.get("summary_zh"),
        # 建议动作 —— **只是建议,本 skill 不执行任何东西**。
        "recommend": list(payload.get("recommend") or []),
        # ⚠️ 可核实的指标本体。没有它,agent 只拿到一个段号,仍然是失明。
        "features": {k: v for k, v in (payload.get("features") or {}).items()},
        "frame_path": payload.get("frame_path") or "",
    }
    if ts_ns is not None:
        out["t_mono_ns"] = ts_ns
    return out


def _enrich_from_monitor(cause_ref: str | None) -> dict | None:
    """``current_monitor#<seg>`` → 那一段在监控库里的判级与特征。

    **补充,不是主答案**:事件 payload 里已经有 features。这里多给的是段落级
    上下文(判级、时间、是否被钉住)。查不到就返回 None —— 监控库可能根本没起。
    """
    if not cause_ref or _CAUSE_SEP not in cause_ref:
        return None
    src, _, seg_txt = cause_ref.partition(_CAUSE_SEP)
    if src != "current_monitor" or not seg_txt.isdigit():
        return None
    try:
        from mast.monitoring.store import get_store
        store = get_store()
        if store is None:
            return None
        seg_id = int(seg_txt)
        meta = store.segment_meta(seg_id) or {}
        feats = store.feature_row(seg_id) or {}
    except Exception:  # noqa: BLE001 — 补充信息永远不该反噬主答案
        logger.debug("cause_ref enrichment failed (swallowed)", exc_info=True)
        return None
    if not meta and not feats:
        return None
    keep = ("rms_detrended_a", "spike_max_sigma", "sat_frac", "max_step_a",
            "rtn_score", "jump_rate_hz", "line_ratio", "frozen")
    return {
        "segment_id": seg_id,
        "alert_level": meta.get("alert_level"),
        "t_start": meta.get("t_start"),
        "fs_hz": meta.get("fs_hz"),
        "pinned": bool(meta.get("pinned")),
        "ctx_scanning": feats.get("ctx_scanning"),
        "ctx_skill": feats.get("ctx_skill"),
        "metrics": {k: feats[k] for k in keep if feats.get(k) is not None},
    }


class ReadHardwareEvents(BaseSkill):
    """只读:当前拦住我的事件 + 最近的硬件事件(含可核实的指标)。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ReadHardwareEvents",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读取硬件事件缓冲区:①当前有没有审批闸门拦着你、拦的是哪条事件;"
                "②最近 N 条事件(种类/严重度/来源/一句话摘要/**触发时的实测指标**/"
                "建议动作)。**纯读,不动任何硬件,也不解除任何拦截。**\n"
                "\n"
                "⚠️ **被 buffer_hitl 拦住写入类工具时,先调用本工具看证据再决定**——"
                "拦截文案只给事件种类,指标在这里。不要反复重试被拒的工具。\n"
                "\n"
                "也用于日常自主判断:监控在报什么、要不要主动修针。"
                "注意 `read_latest_tip_status` 读的是视觉 TipStatus,"
                "**电流监控发的事件不写 TipStatus** —— 那条路上它会返回 seqno=-1，"
                "看起来像「一切正常」。本工具才是事件本身。"
            ),
            parameters=[
                ParameterSpec(
                    name="limit",
                    type="int",
                    description="最多返回多少条最近事件(默认 10)",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=100,
                ),
                ParameterSpec(
                    name="min_severity",
                    type="str",
                    description=(
                        "只返回不低于此严重度的事件:info / warn / critical。"
                        "留空返回全部。"
                    ),
                    required=False,
                    default="",
                ),
            ],
            estimated_duration_s=0.05,
            composition_level=0,
            tags=["read", "events", "monitoring", "hitl", "diagnostic"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        limit = int(params.get("limit") or 10)
        want = str(params.get("min_severity") or "").strip().lower()

        data: dict[str, Any] = {"blocking": _gate_block()}

        # ── 最近事件 ────────────────────────────────────────────────────
        events: list[dict] = []
        ev_block: dict[str, Any] = {"available": False, "why": "", "events": []}
        try:
            from mast.buffer.active import get_active_buffer
            buf = get_active_buffer()
            if buf is None:
                ev_block["why"] = (
                    "事件缓冲区未启动(standalone / 未接视觉服务)。"
                    "**这不代表没有事件,是这条路读不到。**")
            else:
                raw = buf.get_event_history(since_seqno=-1, limit=max(1, limit))
                events = [_event_dict(e) for e in raw]
                if want:
                    # ⚠️ 严重度字面量取自 buffer.schemas.Severity —— 是 "warn" 不是
                    # "warning"。写错的那个不会报错,只会**静默地一条都过滤不掉**。
                    order = {"info": 0, "warn": 1, "critical": 2}
                    floor = order.get(want)
                    if floor is not None:
                        events = [e for e in events
                                  if order.get(str(e.get("severity") or "").lower(),
                                               -1) >= floor]
                for e in events:
                    detail = _enrich_from_monitor(e.get("cause_ref"))
                    if detail is not None:
                        e["cause_detail"] = detail
                ev_block.update({"available": True, "events": events})
        except Exception as exc:  # noqa: BLE001 — 诊断工具不许自己炸
            ev_block["why"] = f"读事件历史失败: {exc}"
        data["recent"] = ev_block

        # ⚠️ 「一条都没有」与「读不到」是两句话。前者是结论,后者不是。
        n = len(events)
        if not ev_block["available"]:
            summary = f"事件缓冲区读不到({ev_block['why']});" \
                      f"审批闸门:{'有拦截' if data['blocking'].get('any_blocking') else '无'}"
        elif n == 0:
            summary = "缓冲区里没有事件(已确认读到,不是读不到)。"
        else:
            worst = max(events, key=lambda e: {"info": 0, "warn": 1,
                                               "critical": 2}.get(
                str(e.get("severity") or "").lower(), -1))
            summary = (f"{n} 条事件,最高 {worst.get('severity')} "
                       f"{worst.get('kind')}(来源 {worst.get('source') or '?'})"
                       f":{worst.get('summary_zh') or ''}")
        data["n_events"] = n
        data["read_at"] = time.time()

        return SkillResult(skill_name="ReadHardwareEvents", success=True,
                           data=data, summary=summary)
