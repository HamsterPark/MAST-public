"""Agent memory tools (write/read/list/search) over a real MemoryStore.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/memory/test_memory_tools.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.agents._shared.memory_tools import make_memory_tools
from mast.memory.store import MemoryStore


def _tools(tmp_path, namespace="experiment:E1"):
    store = MemoryStore(tmp_path / "exp.db")
    ctx = {"store": store, "namespace": namespace, "experiment_id": "E1",
           "author": "literature"}
    tools = make_memory_tools(lambda: ctx)
    return {t.name: t for t in tools}, store


def test_write_then_read_crosses_sessions(tmp_path):
    tools, store = _tools(tmp_path)
    out = tools["memory_write"].invoke({"path": "insights/a.md",
                                        "content": "dI/dV at -0.5V", "kind": "insight"})
    assert "已保存" in out
    # simulate a NEW session: fresh store on the same db
    store2 = MemoryStore(tmp_path / "exp.db")
    assert store2.read("experiment:E1", "insights/a.md")["content"] == "dI/dV at -0.5V"
    # read tool returns it
    r = tools["memory_read"].invoke({"path": "insights/a.md"})
    assert "dI/dV at -0.5V" in r and "insight" in r


def test_list_and_search(tmp_path):
    tools, _ = _tools(tmp_path)
    tools["memory_write"].invoke({"path": "p1.md", "content": "Si(111) 7x7", "tags": ["surf"]})
    tools["memory_write"].invoke({"path": "p2.md", "content": "HOPG"})
    lst = tools["memory_list"].invoke({})
    assert "p1.md" in lst and "p2.md" in lst
    s = tools["memory_search"].invoke({"query": "7x7"})
    assert "p1.md" in s and "p2.md" not in s


def test_read_missing(tmp_path):
    tools, _ = _tools(tmp_path)
    assert "无此记忆" in tools["memory_read"].invoke({"path": "nope.md"})


def test_global_fallback_read(tmp_path):
    tools, store = _tools(tmp_path, namespace="experiment:E2")
    store.write("global", "shared.md", "global note")
    # read from an experiment namespace falls back to global
    assert "global note" in tools["memory_read"].invoke({"path": "shared.md"})


def test_no_store_graceful(tmp_path):
    tools = {t.name: t for t in make_memory_tools(lambda: {})}
    assert "unavailable" in tools["memory_write"].invoke({"path": "x", "content": "y"})
    assert "unavailable" in tools["memory_read"].invoke({"path": "x"})


def test_search_does_not_leak_across_experiments(tmp_path):
    """memory_search must stay within the caller's namespace (+global), not
    return another experiment's memory (security 审查, Low #8)."""
    store = MemoryStore(tmp_path / "exp.db")
    # experiment B writes a private note; a shared global note also exists
    store.write("experiment:B", "secret.md", "B private dI/dV at -0.8V")
    store.write("global", "shared.md", "shared 7x7 note")
    # the tool is bound to experiment A
    ctx = {"store": store, "namespace": "experiment:A", "author": "x"}
    tools = {t.name: t for t in make_memory_tools(lambda: ctx)}
    store.write("experiment:A", "mine.md", "A note 7x7")
    res = tools["memory_search"].invoke({"query": "7x7"})
    assert "mine.md" in res          # own namespace
    assert "shared.md" in res        # global is shared
    # B's private memory must NOT appear even though it would match "dI/dV"
    leak = tools["memory_search"].invoke({"query": "dI/dV"})
    assert "secret.md" not in leak


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
