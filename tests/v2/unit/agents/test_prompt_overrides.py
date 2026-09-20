"""The override layer must actually reach the injection — not just persist.

An override that saves cleanly and then changes nothing is the exact failure
shape this feature exists to expose (``SettingsStore.KNOWN_KEYS`` has produced
it twice). So these tests assert on the CONSUMING side: the middleware output,
the assembled system message, the storage wiring — and one guard that every
entry advertised as overridable is genuinely wired somewhere in the source.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mast.admin.override_store import ConfigOverrideRegistry
from mast.prompts import overrides as ovr
from mast.prompts import registry as reg


@pytest.fixture(autouse=True)
def isolated_overrides(tmp_path: Path):
    ConfigOverrideRegistry.reset()
    ConfigOverrideRegistry.get(tmp_path / "overrides")
    yield tmp_path / "overrides"
    ConfigOverrideRegistry.reset()


# ── storage wiring ───────────────────────────────────────────────────────────

def test_override_file_is_registered_for_load():
    """Missing from _ALL_FILES → written on save, ignored on the next boot.
    Same silent no-op as an unlisted SettingsStore.KNOWN_KEYS entry."""
    from mast.admin import override_store

    assert override_store.PROMPT_OVERRIDES in override_store._ALL_FILES


def test_resolve_falls_back_to_the_default():
    assert ovr.resolve("nothing.stored.here", "DEFAULT") == "DEFAULT"


def test_blank_override_never_becomes_a_blank_prompt():
    ovr.set_override("sub.tool_refine", "   ")
    assert ovr.resolve("sub.tool_refine", "DEFAULT") == "DEFAULT"


def test_resolve_degrades_to_default_when_storage_breaks(monkeypatch):
    monkeypatch.setattr(ovr, "_registry", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ovr.resolve("sub.tool_refine", "DEFAULT") == "DEFAULT"


def test_override_survives_registry_rebuild(isolated_overrides: Path):
    ovr.set_override("mw.mode_belief.safe", "PERSISTED")
    ConfigOverrideRegistry.reset()
    ConfigOverrideRegistry.get(isolated_overrides)
    assert ovr.get("mw.mode_belief.safe") == "PERSISTED"


# ── the override reaches the real injection ──────────────────────────────────

def test_mode_belief_block_uses_the_override():
    from mast.agents._shared import mode_mw
    from mast.core.types import OperatingMode

    _, default = mode_mw._belief_block(lambda: OperatingMode.SAFE)
    assert "安全模式" in default

    ovr.set_override("mw.mode_belief.safe", "只做实验，不修针。")
    assert mode_mw._belief_block(lambda: OperatingMode.SAFE) == (
        "mw.mode_belief.safe", "只做实验，不修针。")
    # SEMI is a separate entry and must be untouched by the SAFE override
    assert mode_mw._belief_block(lambda: OperatingMode.SEMI) == (
        "mw.mode_belief.semi", mode_mw._SEMI_BELIEF)


def test_tool_pair_guard_synth_text_uses_the_override():
    from langchain_core.messages import AIMessage, HumanMessage

    from mast.agents._shared import tool_pair_guard_mw as tpg

    ovr.set_override("mw.tool_pair_guard.synth", "[占位-已覆写]")
    orphan = AIMessage(content="", tool_calls=[
        {"name": "handoff_to_x", "args": {}, "id": "call_1"}])
    out = tpg.repair_tool_pairs([HumanMessage(content="hi"), orphan])
    assert out is not None
    assert any("[占位-已覆写]" in str(getattr(m, "content", "")) for m in out)


def test_agent_system_prompt_override_reaches_the_built_graph():
    """The end-to-end claim for an agent prompt: edit it, rebuild, and the model
    genuinely receives the new text."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    from mast.agents.paper_review import graph as pr_graph
    from mast.prompts import capture as cap

    class _FakeChatModel(GenericFakeChatModel):
        """create_agent calls bind_tools(); the upstream fake doesn't have it."""

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

    ovr.set_override("agent.paper_review.system", "你是一个被覆写过的审稿人。")

    model = _FakeChatModel(messages=iter([AIMessage(content="ok")]))
    agent = pr_graph.build(None, model=model)

    cap.get_ring().clear()
    cb = cap.make_callback(source="paper_review", model_id="fake", provider="test")
    agent.invoke({"messages": [HumanMessage(content="审一下")]},
                 config={"callbacks": [cb]})

    snap = cap.get_ring().get(0)
    system = next((m.content for m in snap.messages if m.role == "system"), "")
    assert "你是一个被覆写过的审稿人。" in system
    cap.get_ring().clear()


# ── inventory invariants ─────────────────────────────────────────────────────

def test_every_id_is_unique():
    ids = [e.id for e in reg.entries()]
    assert len(ids) == len(set(ids))


def test_overridable_entries_are_actually_wired_into_the_code():
    """An entry listed as overridable but never resolved anywhere = a save
    button that persists a value nothing reads. Grep the source for each id."""
    root = Path(__file__).resolve().parents[4] / "MASTv2" / "mast"
    # mast/prompts/ is where the ids are DECLARED; a hit there proves nothing.
    consumers = [p.read_text(encoding="utf-8", errors="ignore")
                 for p in root.rglob("*.py") if p.parent.name != "prompts"]
    unwired = [
        e.id for e in reg.entries()
        if e.overridable
        and not any(f'"{e.id}"' in text or f"'{e.id}'" in text for text in consumers)
    ]
    assert not unwired, f"overridable but never resolved in code: {unwired}"


def test_unrenderable_entries_have_a_reason_and_no_loader():
    for entry in reg.entries():
        if entry.availability in ("needs_hardware", "needs_request"):
            assert entry.loader is None, entry.id
            assert entry.unavailable_reason.strip(), entry.id
            text, err = reg.render_default(entry)
            assert text == "", f"{entry.id} invented text it cannot know"
            assert err.strip(), entry.id


def test_static_entries_render_non_empty_defaults():
    for entry in reg.entries():
        if entry.availability != "static":
            continue
        text, err = reg.render_default(entry)
        assert not err, f"{entry.id}: {err}"
        assert text.strip(), f"{entry.id} rendered empty"


def test_source_paths_point_at_files_that_exist():
    root = Path(__file__).resolve().parents[4]
    for entry in reg.entries():
        rel = re.sub(r":[^:]+$", "", entry.source)   # strip ":SYMBOL"
        assert (root / rel).exists(), f"{entry.id} → missing {rel}"
