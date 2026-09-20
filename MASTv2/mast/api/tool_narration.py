"""Turn a tool call / tool return into ONE honest line of Chinese.

群聊显示**太多太杂**，应转写自然语言 + 参数折叠.
What the group-chat panel actually showed:

    show_plan_on_map(steps=[{'kind': 'scan', 'label': '+1V 50nm', 'w_m': 5.0, , )
    {"seqno": 1036, "progress": {"seqno": 1036, "scan_id": "…", "line_idx": 2, …}}

Two different defects wearing the same clothes:

  * the CALL was rendered by ``f"{name}({', '.join(f'{k}={v}')[:120]})"`` — the
    join was cut at 120 chars and the ``)`` pasted on afterwards, which is where
    the ``, )`` comes from. It also silently dropped every argument past the
    fourth. A reader cannot tell that from a call that really had two arguments;
  * the RETURN was the raw JSON the tool handed the model. That text is written
    for an LLM, not for a person watching a run.

THE RULE THIS MODULE IS BUILT ON — a summary may only say things that are
mechanically derivable from the call. It may name the tool, count and echo
arguments, and use a curated Chinese phrase for a tool we have actually read.
It may **never** guess what an unknown tool does. An unknown tool gets its own
name and an argument count, which is a true sentence about any tool that ever
exists. The failure mode this module exists to avoid is looking complete when it wasn't ("看起来完成了其实没有"); a prettier line that quietly asserts something false is worse than the raw
JSON it replaced.

The curated map covers the tools that actually flood a run (planning, lifecycle,
buffer, memory, figures, the paper loop, literature). It is NOT a second registry
of what tools exist — ``tests/v2/unit/api/test_tool_narration.py`` asserts every
key here is a REAL tool name taken from the agents' live tool lists, so a
renamed tool breaks the test instead of silently falling back forever.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "NARRATORS",
    "narrate_tool_call",
    "narrate_tool_result",
    "summarize_args",
]

#: Serialized arguments longer than this are clipped — WITH a marker. Silent
#: truncation is the defect this module exists to remove, so it is never done
#: quietly anywhere in here.
_ARGS_JSON_MAX = 20_000
_VALUE_CHARS = 60


def _s(v: Any, limit: int = _VALUE_CHARS) -> str:
    """One short, faithful rendering of a value — never a truncated fragment
    that could pass for the whole thing."""
    if isinstance(v, str):
        text = v
    elif isinstance(v, bool):
        # JSON spelling, not Python's. This digest claims every token in it
        # appears verbatim in the payload; "False" would not.
        return "true" if v else "false"
    elif isinstance(v, (list, tuple, set)):
        return f"{len(v)} 项"
    elif isinstance(v, dict):
        return f"{len(v)} 个字段"
    else:
        text = str(v)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _n(args: dict, key: str) -> int | None:
    """Length of a list-valued argument, or None when it isn't one."""
    v = args.get(key)
    return len(v) if isinstance(v, (list, tuple)) else None


def _steps(args: dict) -> str:
    n = _n(args, "steps")
    return f"（{n} 步）" if n else ""


# ── the curated phrases ──────────────────────────────────────────────────────
#
# Each entry maps a REAL tool name to a function of its arguments. Anything that
# reads an argument must tolerate the argument being absent: an LLM omits
# optional keys constantly, and a summary that says "写入记忆 · None" is the
# kind of small lie this module refuses to tell.
NARRATORS: dict[str, Callable[[dict], str]] = {
    # ── planning ──
    "create_plan": lambda a: f"创建实验计划{_steps(a)}"
                             + (f" · {_s(a['title'])}" if a.get("title") else ""),
    "approve_plan": lambda a: "批准实验计划",
    "advance_plan": lambda a: "推进实验计划到下一步",
    "pause_plan": lambda a: "暂停实验计划",
    "resume_plan": lambda a: "恢复实验计划",
    "list_plans": lambda a: "查询实验计划列表",
    "get_plan_progress": lambda a: "查询实验计划进度",
    "show_plan_on_map": lambda a: f"在扫描地图上预览计划路线{_steps(a)}",
    "clear_plan_on_map": lambda a: "清除扫描地图上的计划路线",
    # ── experiment / sample lifecycle ──
    "start_experiment": lambda a: "开始实验"
                                  + (f" · {_s(a['name'])}" if a.get("name") else ""),
    "end_experiment": lambda a: "结束实验",
    "rename_experiment": lambda a: "重命名实验"
                                   + (f" → {_s(a['name'])}" if a.get("name") else ""),
    "start_sample": lambda a: "开始新样品"
                              + (f" · {_s(a['name'])}" if a.get("name") else ""),
    "end_sample": lambda a: "结束当前样品",
    "rename_sample": lambda a: "重命名样品"
                               + (f" → {_s(a['name'])}" if a.get("name") else ""),
    # ── vision buffer (read-only) ──
    "read_latest_tip_status": lambda a: "读取最新针尖状态",
    "get_scan_progress": lambda a: "读取扫描进度",
    "get_tip_history_since": lambda a: "读取针尖状态历史",
    # ── long-term memory ──
    "memory_write": lambda a: "写入长期记忆"
                              + (f" · {_s(a['path'])}" if a.get("path") else ""),
    "memory_read": lambda a: "读取长期记忆"
                             + (f" · {_s(a['path'])}" if a.get("path") else ""),
    "memory_list": lambda a: "列出长期记忆",
    "memory_search": lambda a: "检索长期记忆"
                               + (f" · {_s(a['query'])}" if a.get("query") else ""),
    # ── scans ──
    "get_latest_scan_info": lambda a: "查询最近一次扫描的信息",
    "get_latest_scan_file": lambda a: "查询最近一次扫描的文件",
    "load_scan_file": lambda a: "载入扫描文件"
                                + (f" · {_s(a['path'])}" if a.get("path") else ""),
    "load_scan": lambda a: "载入扫描数据"
                           + (f" · {_s(a['path'])}" if a.get("path") else ""),
    "get_next_scan_position": lambda a: "计算下一个扫描位置",
    "mark_area_used": lambda a: "标记该区域已扫过",
    # ── knowledge / reference lookups ──
    "query_knowledge": lambda a: "查询知识库"
                                 + (f" · {_s(a['query'])}" if a.get("query") else ""),
    "get_workflow_advice": lambda a: "查询工作流建议",
    "get_skill_guidance": lambda a: "查询技能使用指导"
                                    + (f" · {_s(a['skill_name'])}"
                                       if a.get("skill_name") else ""),
    "get_literature_parameters": lambda a: "查询文献参数",
    "get_fault_diagnosis": lambda a: "查询故障诊断参考",
    "get_noise_reference": lambda a: "查询噪声参考",
    "get_measurement_template": lambda a: "查询测量模板",
    "search_deep_reference": lambda a: "深度检索参考资料"
                                       + (f" · {_s(a['query'])}"
                                          if a.get("query") else ""),
    "read_reference_section": lambda a: "阅读参考资料章节",
    # ── background ──
    "spawn_background_task": lambda a: "转入后台运行一个任务",
    # ── figures ──
    "plot_scan": lambda a: "绘制扫描图",
    "plot_spectrum": lambda a: "绘制谱图",
    "mosaic_scans": lambda a: "拼接扫描大图",
    "embed_figure": lambda a: "把图插入稿件",
    "list_figures": lambda a: "列出已渲染的图",
    # ── the paper loop ──
    "save_draft": lambda a: "保存论文草稿"
                            + (f" · {_s(a['title'])}" if a.get("title") else ""),
    "load_draft": lambda a: "读取论文草稿",
    "draft_section": lambda a: "撰写稿件章节"
                               + (f" · {_s(a['section'])}" if a.get("section") else ""),
    "save_review": lambda a: "保存评审报告",
    "load_review": lambda a: "读取评审报告",
    "produce_review": lambda a: "生成评审报告",
    "check_citations": lambda a: "核对引文",
    "check_methodology": lambda a: "核查方法学",
    "check_data_reasoning": lambda a: "核查数据推理",
    # ── literature ──
    "search_papers": lambda a: "检索文献"
                               + (f" · {_s(a['query'])}" if a.get("query") else ""),
    "search_local_corpus": lambda a: "检索本地文献语料"
                                     + (f" · {_s(a['query'])}"
                                        if a.get("query") else ""),
    "propose_citations": lambda a: "推荐可引用文献",
    "lookup_citation": lambda a: "查证一条引文",
    "lib_create": lambda a: "新建文献库",
    "lib_add": lambda a: "把文献加入文献库",
    "lib_remove": lambda a: "从文献库移除文献",
    "lib_switch": lambda a: "切换当前文献库",
    "lib_list": lambda a: "列出文献库",
    "lib_search": lambda a: "在文献库内检索",
    # ── experiment records ──
    "query_experiment_records": lambda a: "查询实验记录",
    "query_past_experiments": lambda a: "查询历史实验",
    "lookup_sample": lambda a: "查询样品信息",
}


def narrate_tool_call(name: str, args: Any) -> str:
    """One line of Chinese describing a tool CALL. Never invents behaviour.

    A tool in :data:`NARRATORS` gets its curated phrase. Everything else — and
    this system has ~490 tools, so "everything else" is the common case — gets
    ``<name> · N 个参数``: the tool's own name plus a count, which is true of any
    tool whether or not anyone has ever read it.
    """
    name = str(name or "tool")
    args = args if isinstance(args, dict) else {}
    fn = NARRATORS.get(name)
    if fn is not None:
        try:
            text = fn(args)
            if text:
                return text
        except Exception as exc:  # noqa: BLE001 — fall through to the honest form
            logger.debug("tool narration failed for %s: %s", name, exc)
    return f"{name} · {len(args)} 个参数" if args else name


def summarize_args(args: Any) -> tuple[str, bool]:
    """``(pretty JSON, clipped)`` for the collapsed 参数 panel.

    Returns the FULL arguments — the panel is where the detail the summary
    dropped has to remain available, otherwise this whole change is just hiding
    information. Only a genuinely huge payload is clipped, and the flag says so
    out loud so the UI can label it rather than letting it read as complete.
    """
    if not isinstance(args, dict) or not args:
        return "", False
    try:
        text = json.dumps(args, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tool args serialisation failed: %s", exc)
        return repr(args)[:_ARGS_JSON_MAX], len(repr(args)) > _ARGS_JSON_MAX
    if len(text) <= _ARGS_JSON_MAX:
        return text, False
    return text[:_ARGS_JSON_MAX], True


def narrate_tool_result(text: str) -> str:
    """A one-line digest of a tool RETURN, or "" when the raw text is fine.

    JSON returns are what make the feed unreadable — ``{"seqno": 1036,
    "progress": {"seqno": 1036, "scan_id": …}}`` says nothing to a person at a
    glance. The digest is a MECHANICAL transcription of the top-level scalar
    fields (``seqno=1036 · line_idx=2 · advancing=false``), not an
    interpretation: every token in it appears verbatim in the payload.

    Returns "" for plain prose, for a handoff note, and for JSON with no scalar
    fields worth a line — in those cases the caller shows the text itself, which
    was already readable.
    """
    raw = (text or "").strip()
    if not raw or not raw.startswith(("{", "[")):
        return ""
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return ""
    if isinstance(obj, list):
        return f"返回 {len(obj)} 项"
    if not isinstance(obj, dict):
        return ""
    parts: list[str] = []
    nested = 0
    for k, v in obj.items():
        if isinstance(v, (dict, list, tuple)):
            nested += 1
            continue
        if v is None:
            continue
        parts.append(f"{k}={_s(v, 40)}")
    if not parts:
        # All-nested payload (the get_scan_progress shape: everything hides one
        # level down). Say what is there rather than pretending to summarise it.
        return f"返回 {len(obj)} 个字段" if obj else ""
    head = " · ".join(parts[:6])
    # Fields this line does NOT show — the ones past the sixth PLUS every nested
    # object skipped above. Counting only the overflow (and letting the nested
    # ones vanish) is how "{"seqno": 1036, "progress": {…}}" would summarise to a
    # bare "seqno=1036", quietly losing the entire payload.
    rest = max(0, len(parts) - 6) + nested
    return head + (f" · 另有 {rest} 项" if rest > 0 else "")
