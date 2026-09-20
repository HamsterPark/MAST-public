"""Buffer summarizer runnable + helper functions.

The two public entry points are pure functions for use by VisionModule
listeners; ``BufferSummarizerNode`` wraps them as a LangChain Runnable
in case we want to integrate it into a graph.

`summarize_tip_status(raw)` and `summarize_segmentation(raw)` each:
  1. Render the raw pydantic dict + agent prompt
  2. Call the lightweight LLM (default Kimi K2.6)
  3. Return the short Chinese summary string

Both are cached on (kind, content_hash) — the same vision output
won't trigger a second LLM call. Cache is a process-local dict;
restart wipes it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mast.agents._shared.models import make_chat_model, KIMI_K3
from mast.agents.buffer_summarizer.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from mast.prompts.registry import resolve as resolve_prompt

logger = logging.getLogger(__name__)


# ── In-process memoization (vision events repeat) ──────────────────────
_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 2048

# Singleton LLM client — built once, shared by all summarize_* calls.
_LLM_CACHE: dict[str, Any] = {}
_LLM_LOCK = threading.Lock()


def _get_llm(model_id: str = KIMI_K3):
    with _LLM_LOCK:
        if model_id not in _LLM_CACHE:
            _LLM_CACHE[model_id] = make_chat_model(
                agent=None, model_id=model_id,
                max_tokens=128, temperature=0.0,
            )
        return _LLM_CACHE[model_id]


def _hash_payload(kind: str, payload: dict) -> str:
    h = hashlib.sha256()
    h.update(kind.encode("utf-8"))
    h.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def _summarize_generic(kind: str, payload: dict, model_id: str = KIMI_K3) -> str:
    """Run the summarizer LLM on a single vision payload, with memoization.

    Falls back to a deterministic non-LLM template if the LLM call fails
    so the buffer pipeline never blocks on network errors.
    """
    key = _hash_payload(kind, payload)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    try:
        llm = _get_llm(model_id)
        msgs = [
            SystemMessage(content=resolve_prompt(
                "agent.buffer_summarizer.system", SYSTEM_PROMPT)),
            HumanMessage(content=resolve_prompt(
                "agent.buffer_summarizer.user_template", USER_TEMPLATE).format(
                kind=kind,
                payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
            )),
        ]
        resp = llm.invoke(msgs)
        text = str(resp.content).strip()
        # Guard against degenerate responses
        if not text or len(text) > 200:
            text = _fallback_summary(kind, payload)
    except Exception as exc:
        logger.warning("summarizer LLM call failed (%s); using fallback", exc)
        text = _fallback_summary(kind, payload)

    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = text
    return text


def describe(kind: str, payload: dict) -> str:
    """Deterministic, instant, non-LLM Chinese narration of a vision result.

    This is the FAST translation path used by the scan-progress vision monitor
    (a per-1/8-milestone LLM call would add latency/network/cost to a live
    scan). It is also the fallback for the LLM summarizers below. Templates are
    enriched with the M12-specific fields (R_tip / sharpness / fine morphology /
    multi-tip / 4-class segmentation fractions) so the narration is informative.
    """
    if kind == "tip_coarse":
        label = payload.get("label", "未知")
        conf = payload.get("confidence", 0.0)
        label_zh = {"good": "良好", "bad": "较差", "unknown": "未知"}.get(label, label)
        extra = ""
        # 措辞纪律（物理真值验收 2026-07-27）：锐度类信号只支持「状态档位」
        # 判断，不支持连续半径测量——模型的 R 输出按粗细档位描述，不报数值。
        r_tip = payload.get("tip_radius_nm")
        if r_tip is not None:
            tier = "偏钝" if float(r_tip) > 5.0 else ("中等" if float(r_tip) > 1.0 else "较锐")
            extra += f"，锐度档位{tier}"
        else:
            sharp = payload.get("sharpness_log10")
            if sharp is not None:
                extra += f"，锐度 log10(R/scan)={float(sharp):.2f}"
        return f"针尖状态{label_zh}，置信度 {conf:.2f}{extra}。"
    if kind == "tip_fine":
        morph = payload.get("morph") or payload.get("label", "未知")
        flags = []
        if payload.get("multi_tip"):
            # 双/多针尖没有经过验收的可靠检测器（vigil_truth_validation）——
            # 模型旗标按低置信疑似措辞，绝不写成确认。
            flags.append("疑似多尖（低置信）")
        if payload.get("switching"):
            flags.append("跳变")
        if payload.get("drift"):
            flags.append("漂移")
        if payload.get("perturbation"):
            flags.append("扰动")
        flag_zh = ("，存在" + "、".join(flags)) if flags else ""
        usable = payload.get("is_usable")
        usable_zh = "" if usable is None else ("，可用" if usable else "，不可用")
        return f"针尖形貌 {morph}{flag_zh}{usable_zh}。"
    if kind == "segmentation":
        counts = payload.get("class_counts", {})
        if not counts:
            return "分割结果为空。"
        total = sum(counts.values()) or 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        parts = [f"{_seg_zh(name)} {count / total * 100:.0f}%" for name, count in top]
        lvl = payload.get("level")
        lvl_zh = f"L{lvl} " if lvl is not None else ""
        return f"{lvl_zh}分割主要类别：{', '.join(parts)}。"
    if kind == "partial":
        coarse = payload.get("coarse_label", "未知")
        coarse_zh = {"good": "良好", "bad": "较差", "unknown": "未知"}.get(coarse, coarse)
        frac = payload.get("frac_acquired", 0.0)
        q = payload.get("quality_pred")
        q_zh = f"，质量分 {float(q):.2f}" if q is not None else ""
        return f"部分扫描 ({frac * 100:.0f}%)，初步针尖评估：{coarse_zh}{q_zh}。"
    return f"视觉输出 ({kind})：{json.dumps(payload, ensure_ascii=False)[:60]}"


_SEG_ZH = {
    "TERRACE": "台面", "terrace": "台面",
    "STEP": "台阶", "step": "台阶",
    "DEFECT": "缺陷", "defect": "缺陷",
    "CONTAMINATION": "污染", "contamination": "污染",
    "NOT_TERRACE": "非台面",
}


def _seg_zh(name: str) -> str:
    return _SEG_ZH.get(name, name)


# Backwards-compatible alias: the LLM summarizers use this as their fallback.
def _fallback_summary(kind: str, payload: dict) -> str:
    return describe(kind, payload)


# ── Public API ────────────────────────────────────────────────────────

def summarize_tip_status(tip_coarse: dict, *, model_id: str = KIMI_K3) -> str:
    """Return a 1-2 sentence Chinese summary of a TipCoarseResult dict."""
    return _summarize_generic("tip_coarse", tip_coarse, model_id=model_id)


def summarize_segmentation(seg: dict, *, model_id: str = KIMI_K3) -> str:
    """Return a 1-2 sentence Chinese summary of a SegmentationResult dict.

    `mask_rle` is stripped before sending to the LLM (too big and the LLM
    can't reason about RLE bytes anyway).
    """
    payload = {k: v for k, v in seg.items() if k != "mask_rle"}
    return _summarize_generic("segmentation", payload, model_id=model_id)


def summarize_partial(partial: dict, *, model_id: str = KIMI_K3) -> str:
    return _summarize_generic("partial", partial, model_id=model_id)


def summarize_tip_fine(tip_fine: dict, *, model_id: str = KIMI_K3) -> str:
    return _summarize_generic("tip_fine", tip_fine, model_id=model_id)


# ── Optional LangChain Runnable wrapper ────────────────────────────────

class BufferSummarizerNode:
    """Stateful summarizer node — pluggable into a LangGraph if needed.

    The buffer service itself can subscribe to VisionModule via its own
    asyncio loop; this class is mostly for tests / future graph integration.
    """

    def __init__(self, model_id: str = KIMI_K3):
        self.model_id = model_id

    def invoke(self, payload: dict) -> str:
        kind = payload.pop("__kind__", "tip_coarse")
        return _summarize_generic(kind, payload, model_id=self.model_id)


__all__ = [
    "describe",
    "summarize_tip_status",
    "summarize_segmentation",
    "summarize_partial",
    "summarize_tip_fine",
    "BufferSummarizerNode",
]
