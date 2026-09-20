"""Shared agent utilities — cross-cutting helpers reused by all 6 agents.

Phase 3 fills:
  handoff.py       make_handoff(target, description) factory
  skill_adapter.py wrap_skill(skill, context_provider) — only BaseSkill importer
  buffer_tools.py  make_buffer_tools(buf) — read-only buffer queries for agents
  safety_mw.py     SafetyGateMiddleware ports v1 SafetyGuard (admin override)
  models.py        centralized Claude model IDs (R16 mitigation)
"""
