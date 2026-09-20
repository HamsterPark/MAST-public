"""GUI cognition-tab helpers (memory / phases / dreaming / brainstorm).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/gui/test_cognition_panel.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.webui import cognition_panel as cp  # noqa: E402
from mast.memory.sharding import PhaseManager  # noqa: E402
from mast.memory.store import MemoryStore  # noqa: E402

XSS = "<script>alert(1)</script>"


def _store(tmp_path):
    s = MemoryStore(tmp_path / "exp.db")
    s.write("global", "shared.md", "global note", kind="note")
    s.write("experiment:E1", "insights/a.md", "dI/dV at -0.5V", kind="insight",
            title="STS finding", author="literature")
    s.write("experiment:E1", "dreams/d.md", "🌙 pattern", kind="dream", pinned=True)
    return s


def test_namespaces_global_first(tmp_path):
    s = _store(tmp_path)
    ns = cp.memory_namespaces(s)
    assert ns[0] == "global"
    assert "experiment:E1" in ns


def test_list_html_lists_pinned_first_and_counts(tmp_path):
    s = _store(tmp_path)
    out = cp.render_memory_list_html(s, "experiment:E1")
    assert "insights/a.md" in out and "dreams/d.md" in out
    assert "📌" in out                  # pinned dream marked
    assert "2 条" in out


def test_list_html_kind_filter(tmp_path):
    s = _store(tmp_path)
    out = cp.render_memory_list_html(s, "experiment:E1", kind="insight")
    assert "insights/a.md" in out and "dreams/d.md" not in out


def test_detail_escapes_content(tmp_path):
    s = _store(tmp_path)
    s.write("global", "evil.md", XSS, kind="note")
    out = cp.render_memory_detail_html(s, "global", "evil.md")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_save_and_delete_and_pin(tmp_path):
    s = _store(tmp_path)
    assert "✅" in cp.save_memory(s, "global", "new.md", "body", kind="protocol")
    assert s.read("global", "new.md")["kind"] == "protocol"
    assert "📌" in cp.toggle_pin_memory(s, "global", "new.md", True)
    assert s.read("global", "new.md")["pinned"] is True
    assert "🗑" in cp.delete_memory(s, "global", "new.md")
    assert s.read("global", "new.md") is None


def test_save_requires_path(tmp_path):
    s = _store(tmp_path)
    assert "❌" in cp.save_memory(s, "global", "", "body")


def test_phase_summaries_html(tmp_path):
    s = _store(tmp_path)
    pm = PhaseManager(tmp_path / "exp.db", memory_store=s)
    pm.start_phase("针尖准备", experiment_id="E1")
    # no conversation_log rows → summary is the empty-phase note, still renders
    pm.end_phase("E1")
    out = cp.render_phase_summaries_html(pm, "E1")
    assert "针尖准备" in out


def test_dream_now_writes(tmp_path):
    s = _store(tmp_path)
    # custom consolidator so we don't depend on real experiment rows
    def _c(ctx):
        return [{"path": "dreams/panel.md", "title": "t", "kind": "dream",
                 "content": "consolidated"}]
    msg = cp.run_dream_now(str(tmp_path / "exp.db"), s, consolidator=_c)
    assert "🌙" in msg and "dreams/panel.md" in msg
    assert s.read("global", "dreams/panel.md") is not None


def test_brainstorm_graceful_when_backend_absent(tmp_path):
    # backend may not be importable yet — must degrade, not crash
    s = _store(tmp_path)
    html_out, summary = cp.run_brainstorm_panel(
        str(tmp_path / "exp.db"), "E1", topic="next step", store=s)
    assert isinstance(html_out, str) and isinstance(summary, str)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
