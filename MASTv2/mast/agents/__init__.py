"""mast.agents — 7 LangGraph agents + Orchestrator + shared utilities.

Layout (each agent is a self-contained subgraph):
  orchestrator/         custom StateGraph supervisor (Sonnet 4.6 default, Opus 4.7 escalate)
  research_director/    Campaign 层「为什么做」— campaign CRUD + 既往记录只读，
                        产出 hypothesis + plan_request；不碰仪器（2026-08-21）
  literature/           Opus 4.7, PyMuPDF + TF-IDF retriever (cosine + tag overlap) + optional web_search
  experiment_design/    Sonnet 4.6, sample/skill/past-experiment tools → ExperimentPlan
  instrument_control/   Sonnet 4.6, ~156 skills wrapped, buffer tools, SafetyGateMiddleware
  data_processing/      Sonnet 4.6, 21 analysis skills + sandboxed run_numpy_snippet
  paper_writing/        Sonnet 4.6, experiment SQL + CitationManager + section templates
  paper_review/         Opus 4.7, rubric checklists + line-level comments
  _shared/              handoff, skill_adapter (wrap_skill), buffer_tools, safety_mw, models

Inter-agent: via MASTState TypedDict in state.py (NOT chat messages).
Handoff: Command(goto=target, graph=Command.PARENT, update={...}).
Invariant enforced by agent_boundary 钩子（不随仓）: no agents.A → agents.B cross-imports
except via agents.state and agents._shared.*.
"""
