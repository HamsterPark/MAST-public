"""QuickAskAgent — single-turn read-only LLM helper.

Sits alongside MissionPlanner. Use it for the GUI's "查询助手" tab where the
operator wants to ask **questions** while the main chat is busy executing
something. The contract:

  * **Single turn**: the agent receives one query, runs an internal tool-use
    loop (≤ ``max_steps``), returns one final text answer. No conversation
    history is preserved between calls.

  * **Read-only**: the tool list is filtered to ``SkillCategory.READ`` and
    ``SkillCategory.ANALYSIS`` skills (no instrument writes), plus a
    knowledge-only subset of meta-tools (look up workflow / fault diagnosis /
    glossary / literature index). All experiment-state-mutation meta-tools
    (start/end_experiment, create/execute_plan, mark_area_used, …) are
    deliberately excluded.

  * **No side effects on persistent stores**: this agent does NOT write to
    ``ExperimentLog``, the chat-history file, or the plan store.

  * **Concurrency**: each call is independent — the GUI can drive this in
    parallel with MissionPlanner without sharing state.

The agent uses the existing ``ClaudeClient`` (any provider) and reuses
``SkillExecutor`` for the read-only skills it does invoke. WRITE skills are
double-gated: they aren't in the tool list AND ``one_shot()`` rejects any
tool whose registry entry is not WHITE-listed at run time.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from mast.core.types import SkillCategory, SkillResult

logger = logging.getLogger(__name__)


# ── Read-only meta-tool whitelist ─────────────────────────────────────
# Names mirror MissionPlanner._build_meta_tools but DROP every state-mutator.
_READ_META_TOOLS: frozenset[str] = frozenset({
    "get_latest_scan_info",
    "get_workflow_advice",
    "get_skill_guidance",
    "get_literature_parameters",
    "get_fault_diagnosis",
    "get_noise_reference",
    "get_measurement_template",
    "search_deep_reference",
    "read_reference_section",
    "query_knowledge",
    "load_scan_file",
    "search_local_corpus",   # OpenAlex semantic search
    "lookup_glossary",       # STM term resolver
    "get_material_coverage", # OpenAlex paper counts per material
})


_QUICKASK_SYSTEM_PROMPT = """You are MAST 查询助手 (Quick Ask Helper) — a read-only \
information assistant.

You answer the operator's questions in a single turn. You can call read-only
tools to inspect the current instrument state, look up literature, and query
MAST's knowledge base, but you MUST NOT modify any state — no scan starts,
no bias changes, no experiment starts/stops, no plan edits.

Workflow:
  1. Decide if you need any tool to answer.
  2. If yes, call one or more read-only tools (≤ {max_steps} round-trips).
  3. Produce a concise final answer in 中文 (or the user's language).

Style:
  * Lead with the answer, then 1-3 sentences of justification.
  * Quote specific numbers from the tool results — don't paraphrase.
  * If a tool fails, say so and answer with what you have.
  * Keep total response under 300 中文字符 unless the user asks for detail.

You do NOT have control of the instrument. Refer the operator to the main
对话 tab for any action that would change instrument state.
"""


class QuickAskAgent:
    """Single-turn, read-only LLM helper.

    Args:
        client: a configured `ClaudeClient` (provider-agnostic). The agent
            reuses ``client.chat(messages, system, tools)`` directly; it does
            NOT mutate the client's model or thinking settings.
        registry: SkillRegistry — used to filter the read-only skill subset.
        executor: SkillExecutor — invoked for read-only skill tool_use.
        state: InstrumentState — exposed implicitly through skill execution
            results (no special handling here).
        experiment_log: optional, used **read-only** for context (sample
            type / id) when building the system prompt prefix.
        plan_store: optional, used **read-only** if the prompt needs to
            reference an active plan.
    """

    def __init__(
        self,
        client,
        executor,
        registry,
        state,
        *,
        experiment_log=None,
        plan_store=None,
    ) -> None:
        self._client = client
        self._executor = executor
        self._registry = registry
        self._state = state
        self._experiment_log = experiment_log
        self._plan_store = plan_store
        self._lock = threading.Lock()  # one-call-at-a-time per agent instance

    # ── Tool construction ────────────────────────────────────────────

    def _build_read_only_skill_tools(self) -> tuple[list[dict], set[str]]:
        """Return (tool_defs, allowed_names) restricted to READ + ANALYSIS skills."""
        tools: list[dict] = []
        allowed: set[str] = set()
        for meta in self._registry.list_skills():
            cat = getattr(meta, "category", None)
            if cat not in (SkillCategory.READ, SkillCategory.ANALYSIS):
                continue
            allowed.add(meta.name)
        # Build full tool dicts via existing converter then filter (cheaper
        # than rebuilding by hand and stays in sync with registry schema).
        for tool in self._registry.to_tool_definitions():
            if tool.get("name") in allowed:
                tools.append(tool)
        return tools, allowed

    def _build_meta_tools(self) -> list[dict]:
        """Read-only meta-tools (knowledge / advisory). State-mutating ops dropped."""
        return [
            {
                "name": "get_latest_scan_info",
                "description": "Get the file path of the most recent scan file (.sxm).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_workflow_advice",
                "description": (
                    "Get recommended experiment workflow for a sample type or material. "
                    "Returns phases, parameters, success criteria, common issues."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Sample type or material, e.g. 'Au(111)'"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_skill_guidance",
                "description": "Get expert guidance for a skill (when to use, related skills).",
                "input_schema": {
                    "type": "object",
                    "properties": {"skill_name": {"type": "string"}},
                    "required": ["skill_name"],
                },
            },
            {
                "name": "get_literature_parameters",
                "description": "Look up literature-recommended parameters for a material/phase.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "material": {"type": "string"},
                        "phase": {"type": "string"},
                    },
                    "required": ["material"],
                },
            },
            {
                "name": "get_fault_diagnosis",
                "description": "Diagnose a fault from a symptom description.",
                "input_schema": {
                    "type": "object",
                    "properties": {"symptom": {"type": "string"}},
                    "required": ["symptom"],
                },
            },
            {
                "name": "get_noise_reference",
                "description": "Look up STM noise frequency catalog by frequency or noise type.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "frequency_hz": {"type": "number"},
                        "noise_type": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "name": "get_measurement_template",
                "description": "Get measurement template (Kondo STS, QPI, SC gap, …).",
                "input_schema": {
                    "type": "object",
                    "properties": {"measurement_type": {"type": "string"}},
                    "required": ["measurement_type"],
                },
            },
            {
                "name": "search_deep_reference",
                "description": "Search MAST-reference doc index by keyword. Returns sections.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "read_reference_section",
                "description": "Read a specific section from MAST-reference docs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "report_id": {"type": "string"},
                        "heading": {"type": "string"},
                    },
                    "required": ["report_id", "heading"],
                },
            },
            {
                "name": "query_knowledge",
                "description": "Natural-language query over MAST knowledge base.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "detail_level": {
                            "type": "string",
                            "enum": ["conceptual", "parameters", "full"],
                            "default": "conceptual",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "load_scan_file",
                "description": "Load a Nanonis scan file (.sxm/.3ds/.dat) for OFFLINE inspection.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "search_local_corpus",
                "description": (
                    "Semantic search over the local OpenAlex STM corpus (~50k papers). "
                    "Cross-lingual; returns DOI / title / year / cited count."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "lookup_glossary",
                "description": (
                    "Resolve an English/Chinese STM term: returns canonical English form + "
                    "Chinese aliases + abbreviation + domain."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"term": {"type": "string"}},
                    "required": ["term"],
                },
            },
            {
                "name": "get_material_coverage",
                "description": (
                    "Look up OpenAlex paper count + top-cited DOIs for a material name "
                    "(e.g. 'Au(111)', 'graphene', 'FeSe/SrTiO3')."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"material": {"type": "string"}},
                    "required": ["material"],
                },
            },
        ]

    # ── Meta-tool dispatch (read-only) ──────────────────────────────

    def _handle_meta_tool(self, tool_name: str, params: dict) -> dict:
        if tool_name == "get_latest_scan_info":
            try:
                from mast.webui.scan_preview import get_latest_scan
                experiments_dir = ""
                if self._experiment_log and hasattr(self._experiment_log, "_storage"):
                    pass  # storage path inferred elsewhere
                path = get_latest_scan(experiments_dir, None, None)
                return {"success": True, "path": path or None}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "get_workflow_advice":
            try:
                from mast.knowledge import match_material, format_conceptual_for_llm
                m = match_material(params.get("query", ""))
                if not m:
                    return {"success": False, "error": "no matching sample type"}
                t, name = m
                return {"success": True, "advice": format_conceptual_for_llm(t, name)}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "get_skill_guidance":
            try:
                from mast.knowledge import format_skill_guidance_for_llm
                txt = format_skill_guidance_for_llm(params.get("skill_name", ""))
                return ({"success": True, "guidance": txt} if txt
                        else {"success": False, "error": "no guidance"})
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "get_literature_parameters":
            try:
                from mast.knowledge import match_material, format_literature_params
                m = match_material(params.get("material", ""))
                if not m:
                    return {"success": False, "error": "no matching material"}
                t, name = m
                return {
                    "success": True,
                    "parameters": format_literature_params(t, name, params.get("phase", "")),
                }
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "get_fault_diagnosis":
            try:
                from mast.knowledge.fault_diagnosis import match_faults, format_fault_for_llm
                ms = match_faults(params.get("symptom", ""), top_n=5)
                return ({"success": True,
                         "diagnosis": "\n\n".join(format_fault_for_llm(f) for f in ms)}
                        if ms else {"success": False, "error": "no match"})
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "get_noise_reference":
            try:
                from mast.knowledge.stm_noise import (
                    lookup_by_frequency, lookup_by_type, format_noise_entry_for_llm,
                )
                freq = params.get("frequency_hz")
                ntype = params.get("noise_type")
                entries = (lookup_by_frequency(float(freq)) if freq is not None
                           else lookup_by_type(ntype) if ntype else [])
                return ({"success": True,
                         "reference": "\n\n".join(format_noise_entry_for_llm(e) for e in entries)}
                        if entries else {"success": False, "error": "no match"})
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "get_measurement_template":
            try:
                from mast.knowledge.skill_guidance import (
                    get_measurement_template as _get,
                    format_measurement_template_for_llm as _fmt,
                    MEASUREMENT_TEMPLATES,
                )
                key = params.get("measurement_type", "")
                tmpl = _get(key)
                if tmpl:
                    if key not in MEASUREMENT_TEMPLATES:
                        for k, v in MEASUREMENT_TEMPLATES.items():
                            if v is tmpl:
                                key = k
                                break
                    return {"success": True, "template": _fmt(key)}
                return {"success": False, "error": "no template"}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "search_deep_reference":
            try:
                from mast.knowledge.reference_index import search_sections
                rs = search_sections(params.get("query", ""), top_n=5)
                if not rs:
                    return {"success": False, "error": "no sections"}
                lines = [
                    f"[{r['report_id']}] {r['heading']} — {r['summary']} "
                    f"(lines {r['lines'][0]}-{r['lines'][1]})" for r in rs
                ]
                return {"success": True, "sections": "\n".join(lines)}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "read_reference_section":
            try:
                from mast.knowledge.reference_index import read_section
                txt = read_section(params.get("report_id", ""), params.get("heading", ""))
                return ({"success": True, "content": txt} if txt
                        else {"success": False, "error": "section not found"})
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "query_knowledge":
            try:
                from mast.knowledge import get_retriever
                retriever = get_retriever()
                results = retriever.query(params.get("query", ""), top_k=10)
                if not results:
                    return {"success": False, "error": "no match"}
                parts = []
                for chunk, _ in results:
                    try:
                        parts.append(chunk.content())
                    except Exception:
                        pass
                return {"success": True, "knowledge": "\n\n".join(parts)}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "load_scan_file":
            try:
                from mast.io.nanonis_files import load_scan_file
                path = params.get("path", "")
                data = load_scan_file(path)
                summary: dict = {"success": True, "file": path}
                if "channels" in data:
                    summary["channels"] = list(data["channels"].keys())
                if "header" in data:
                    summary["header_keys"] = list(data["header"].keys())
                return summary
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "search_local_corpus":
            try:
                from mast.knowledge.literature_index import search
                hits = search(params.get("query", ""), k=int(params.get("k") or 10))
                return {"success": True, "hits": hits}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "lookup_glossary":
            try:
                from mast.knowledge.glossary import lookup, format_for_prompt
                ms = lookup(params.get("term", ""), k=8)
                return ({"success": True, "matches": format_for_prompt(ms)}
                        if ms else {"success": False, "error": "no match"})
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if tool_name == "get_material_coverage":
            try:
                import json as _json
                from pathlib import Path
                p = Path(__file__).resolve().parents[1] / "knowledge" / "material_coverage.json"
                if not p.exists():
                    return {"success": False, "error": "coverage file missing"}
                cov = _json.loads(p.read_text(encoding="utf-8"))
                mat = (params.get("material") or "").strip()
                rec = (cov.get("by_material") or {}).get(mat)
                if rec is None:
                    needle = mat.lower()
                    for name, r in (cov.get("by_material") or {}).items():
                        if needle and (needle in name.lower() or name.lower() in needle):
                            rec = r
                            break
                return ({"success": True, "coverage": rec, "material_resolved": mat}
                        if rec else {"success": False, "error": "no coverage entry"})
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        return {"success": False, "error": f"unknown tool: {tool_name}"}

    # ── Public single-turn entry point ───────────────────────────────

    # Per-agent scope hints appended to the system prompt. The QA still
    # has access to every read-only tool — the hint nudges the model
    # towards the relevant domain instead of fanning out across every
    # skill registered. ``"all"`` is the legacy default.
    _SCOPE_HINTS: "dict[str, str]" = {
        "all": "",
        "literature": (
            "本次查询的视角限定为「文献 LIT」agent —— 优先回答与文献搜索、"
            "样品制备协议抽取、参考参数(温度/偏压/I_set 等)、相关综述的问题。"
            "如果用户的问题不属于文献领域，请说明并建议切换查询范围。"
        ),
        "experiment_design": (
            "本次查询的视角限定为「实验设计 XD」agent —— 优先回答与研究问题"
            "拆解、ExperimentPlan 编排、阶段/参数/成功判据相关的问题。"
            "如果用户的问题不属于实验设计领域，请说明并建议切换查询范围。"
        ),
        "instrument_control": (
            "本次查询的视角限定为「仪器控制 IC」agent —— 优先回答与 Nanonis "
            "状态、TCP skill、扫描/谱图/针尖操作相关的问题。可使用 READ 类 "
            "skill 获取实时数据，但不要执行 ACTUATE / DANGEROUS skill。"
        ),
        "data_processing": (
            "本次查询的视角限定为「数据处理 DP」agent —— 优先回答 .sxm/.dat/.3ds "
            "解析、统计指标、缺陷计数、谱图拟合相关的问题。可使用 ANALYSIS "
            "skill。"
        ),
        "paper_writing": (
            "本次查询的视角限定为「论文写作 PW」agent —— 优先回答与稿件 intro/"
            "methods/results/discussion 撰写、句式、术语、参考文献格式相关的问题。"
            "通常不需要调用 skill，只用 LLM 知识。"
        ),
        "paper_review": (
            "本次查询的视角限定为「论文审稿 PR」agent —— 优先按照「方法可重现性 / "
            "结论支持度 / 局限说明」三轴 rubric 回答。"
            "通常不需要调用 skill，只用 LLM 知识。"
        ),
        "buffer_summarizer": (
            "本次查询的视角限定为「视觉摘要 BUF」agent —— 优先回答与 DINOv3 "
            "视觉输出、扫描事件流摘要、tip 状态评估相关的问题。"
        ),
    }

    def one_shot(self, query: str, *, max_steps: int = 8, scope: str = "all") -> str:
        """Run a single read-only tool-use loop and return the final text answer.

        Args:
            query: user question.
            max_steps: tool-use loop budget.
            scope: agent-domain hint, one of the keys in ``_SCOPE_HINTS``.
                ``"all"`` (default) keeps the legacy behaviour.
        """
        if not query or not query.strip():
            return "(empty query)"

        # Serialise concurrent calls per agent instance — keeps the executor
        # / state happy. Multiple QuickAskAgent instances would still run in
        # parallel for true concurrency.
        with self._lock:
            return self._run(query, max_steps, scope)

    def _run(self, query: str, max_steps: int, scope: str = "all") -> str:
        skill_tools, skill_allowed = self._build_read_only_skill_tools()
        meta_tools = self._build_meta_tools()
        meta_names = {t["name"] for t in meta_tools}
        tools = skill_tools + meta_tools

        system = _QUICKASK_SYSTEM_PROMPT.format(max_steps=max_steps)
        # Lightweight instrument-state preface so the agent can answer
        # "what is the current bias?" without extra tool calls.
        try:
            state_text = self._format_state()
            if state_text:
                system += f"\n\nCurrent instrument state:\n{state_text}\n"
        except Exception:
            pass
        # Append a scope hint when the operator has narrowed the query.
        hint = self._SCOPE_HINTS.get(scope or "all", "")
        if hint:
            system += "\n\n### 查询范围\n" + hint + "\n"

        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        text_response = ""

        for step in range(max_steps):
            try:
                response = self._client.chat(
                    messages=messages,
                    system=system,
                    tools=tools if tools else None,
                )
            except Exception as exc:
                logger.warning("QuickAsk LLM call failed: %s", exc)
                return f"查询失败：{type(exc).__name__}: {exc}"

            content_blocks = response.get("content", []) or []
            tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

            if not tool_use_blocks:
                text_parts = [b.get("text", "") for b in content_blocks
                              if b.get("type") == "text" and b.get("text")]
                text_response = "\n\n".join(text_parts).strip()
                break

            messages.append({"role": "assistant", "content": content_blocks})

            tool_results: list[dict[str, Any]] = []
            for tb in tool_use_blocks:
                tname = tb.get("name", "")
                tparams = tb.get("input", {}) or {}
                tid = tb.get("id", "")

                if tname in meta_names:
                    out = self._handle_meta_tool(tname, tparams)
                elif tname in skill_allowed:
                    try:
                        skill_result: SkillResult = self._executor.run(
                            skill_name=tname,
                            params=tparams,
                            approval_source="quickask-readonly",
                        )
                        if skill_result.success:
                            out = {"success": True,
                                   "data": skill_result.data,
                                   "elapsed_s": round(skill_result.elapsed_s, 3)}
                        else:
                            out = {"success": False,
                                   "error": skill_result.error,
                                   "elapsed_s": round(skill_result.elapsed_s, 3)}
                    except Exception as exc:
                        out = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
                else:
                    # Defensive: LLM hallucinated a tool not in our list, or
                    # somehow asked for a WRITE skill — refuse.
                    out = {
                        "success": False,
                        "error": (
                            f"tool {tname!r} is not available in 查询助手 "
                            "(read-only mode). Use the main 对话 tab for write actions."
                        ),
                    }

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": json.dumps(out, default=str, ensure_ascii=False),
                })

            messages.append({"role": "user", "content": tool_results})
        else:
            text_response = (text_response
                             or "(达到查询助手最大步数，未给出最终答案)")

        return text_response or "(没有文本回复)"

    # ── Helpers ──────────────────────────────────────────────────────

    def _format_state(self) -> str:
        try:
            hw = self._state.snapshot()
        except Exception:
            return ""
        if hw is None:
            return ""
        rows: list[str] = []
        if hw.bias_v is not None:
            rows.append(f"- bias = {hw.bias_v:.4e} V")
        if hw.current_a is not None:
            rows.append(f"- current = {hw.current_a:.4e} A")
        if hw.z_pos_m is not None:
            rows.append(f"- z = {hw.z_pos_m:.4e} m")
        if hw.x_pos_m is not None and hw.y_pos_m is not None:
            rows.append(f"- xy = ({hw.x_pos_m:.4e}, {hw.y_pos_m:.4e}) m")
        if hw.z_controller_status is not None:
            rows.append(f"- z controller = {hw.z_controller_status}")
        elif hw.z_controller_on is not None:
            rows.append(f"- z controller = {'ON' if hw.z_controller_on else 'OFF'}")
        if hw.scan_running is not None:
            rows.append(f"- scan = {'RUNNING' if hw.scan_running else 'STOPPED'}")
        if hw.setpoint_a is not None:
            rows.append(f"- setpoint = {hw.setpoint_a:.4e} A")
        return "\n".join(rows)


__all__ = ["QuickAskAgent"]
