"""Tests for gui/agents_api: per-agent model + thinking payload (, #116).

These pin two regressions:
  * #66 — ``_resolve_agent_models`` read a non-existent ``DEFAULT_AGENT_MODELS``
    attribute, so ``models`` was always empty and ``build_agents_payload``
    returned ``{}`` (full mock) even though a real registry exists.
  * #116 — ``thinking`` was hardcoded empty despite the docstring claiming it
    is real. It must now carry the *effective* thinking level per agent.

No LLM / network: the shared models module is pure-Python registry data.
"""
from __future__ import annotations

# ── path bootstrap: MASTv2/ must win over any v1 mast on sys.path ──
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found above " + str(Path(__file__).resolve()))


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import mast.webui.agents_api as agents_api  # noqa: E402
from mast.agents._shared import models as agent_models  # noqa: E402

assert "MASTv2" in agents_api.__file__.replace("\\", "/"), (
    f"agents_api resolved to v1 path: {agents_api.__file__}"
)


class TestResolveAgentModels:
    def test_reads_AGENT_MODEL_registry(self):
        """#66: models must be populated from the real AGENT_MODEL dict."""
        models, thinking = agents_api._resolve_agent_models()
        assert models, "models should not be empty when AGENT_MODEL exists"
        # Every agent id present in the shared registry should round-trip.
        for aid in agents_api._AGENT_IDS:
            if aid in agent_models.AGENT_MODEL:
                assert models[aid] == agent_models.AGENT_MODEL[aid]

    def test_thinking_is_effective_not_blank(self):
        """#116: thinking must carry the real effective level per agent."""
        models, thinking = agents_api._resolve_agent_models()
        assert thinking, "thinking should not be empty"
        assert set(thinking) == set(models)
        for aid, model_id in models.items():
            assert thinking[aid] == agent_models.effective_thinking(model_id, None)

    def test_default_kimi_is_pinned_high(self):
        """Default model (Kimi K2.6) is an always-high reasoning model."""
        models, thinking = agents_api._resolve_agent_models()
        # All agents default to Kimi K2.6 → effective thinking is "high (固定)".
        assert all(v == "high (固定)" for v in thinking.values()), thinking

    def test_registry_missing_returns_empty(self, monkeypatch):
        monkeypatch.delattr(agent_models, "AGENT_MODEL", raising=False)
        models, thinking = agents_api._resolve_agent_models()
        assert models == {}
        assert thinking == {}


class TestBuildAgentsPayload:
    def test_payload_has_models_and_thinking(self):
        """#66 + #116: payload is no longer always-empty mock."""
        payload = agents_api.build_agents_payload()
        assert payload, "payload must not be {} when registry resolves"
        assert "models" in payload and payload["models"]
        assert "thinking" in payload and payload["thinking"]

    def test_payload_empty_when_no_registry(self, monkeypatch):
        monkeypatch.delattr(agent_models, "AGENT_MODEL", raising=False)
        assert agents_api.build_agents_payload() == {}

    def test_write_agents_data_roundtrip(self, tmp_path):
        import json
        out = tmp_path / "agents_data.json"
        payload = agents_api.write_agents_data(out)
        on_disk = json.loads(out.read_text(encoding="utf-8"))
        assert on_disk == payload
        assert on_disk["models"]
        assert on_disk["thinking"]
