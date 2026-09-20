"""MemoryStore — persistent filesystem-style agent memory.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/memory/test_store.py -x -v
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

from mast.memory.store import MemoryStore, sanitize_path


def _store(tmp_path):
    return MemoryStore(tmp_path / "exp.db")


def test_write_read_upsert(tmp_path):
    s = _store(tmp_path)
    s.write("global", "insights/tip.md", "pulse 3V works", title="Tip", kind="insight",
            tags=["tip"], author="agent")
    r = s.read("global", "insights/tip.md")
    assert r["content"] == "pulse 3V works" and r["kind"] == "insight"
    assert r["tags"] == ["tip"] and r["title"] == "Tip"
    # re-write same path UPDATES (no duplicate)
    s.write("global", "insights/tip.md", "pulse 4V better", title="Tip v2")
    assert s.read("global", "insights/tip.md")["content"] == "pulse 4V better"
    assert len(s.list("global")) == 1


def test_list_pinned_first(tmp_path):
    s = _store(tmp_path)
    s.write("global", "a.md", "a")
    s.write("global", "b.md", "b", pinned=True)
    paths = [r["path"] for r in s.list("global")]
    assert paths[0] == "b.md"   # pinned first


def test_search(tmp_path):
    s = _store(tmp_path)
    s.write("global", "x.md", "Si(111) 7x7 reconstruction notes", tags=["surface"])
    s.write("global", "y.md", "HOPG defects")
    assert [r["path"] for r in s.search("7x7")] == ["x.md"]
    assert [r["path"] for r in s.search("surface")] == ["x.md"]   # tag hit
    assert s.search("nonexistent") == []


def test_namespaces_and_scope(tmp_path):
    s = _store(tmp_path)
    s.write("global", "g.md", "g")
    s.write("experiment:E1", "e.md", "e", experiment_id="E1")
    assert set(s.namespaces()) == {"global", "experiment:E1"}
    assert [r["path"] for r in s.list("experiment:E1")] == ["e.md"]


def test_delete_pin(tmp_path):
    s = _store(tmp_path)
    w = s.write("global", "z.md", "z")
    s.pin(w["id"], True)
    assert s.get(w["id"])["pinned"] is True
    assert s.delete("global", "z.md") is True
    assert s.read("global", "z.md") is None


def test_index_markdown(tmp_path):
    s = _store(tmp_path)
    assert "空" in s.index_markdown()           # empty index
    s.write("global", "insights/tip.md", "body", title="Tip", pinned=True)
    idx = s.index_markdown("global")
    assert "insights/tip.md" in idx and "📌" in idx and "MEMORY" in idx


def test_path_sanitization():
    # traversal / separators can't escape the namespace
    assert ".." not in sanitize_path("../../etc/passwd")
    assert sanitize_path("/abs/path").startswith("abs")
    assert sanitize_path("a\\b\\c") == "a/b/c"
    assert sanitize_path("") == "note"
    # CJK kept
    assert sanitize_path("洞见/针尖.md") == "洞见/针尖.md"


def test_write_traversal_path_is_contained(tmp_path):
    s = _store(tmp_path)
    w = s.write("global", "../../evil.md", "x")
    assert ".." not in w["path"]
    assert s.read("global", w["path"]) is not None


def test_from_storage(tmp_path):
    # shares the experiment DB file
    from mast.logging.storage import ExperimentStorage
    st = ExperimentStorage(str(tmp_path / "shared.db"))
    ms = MemoryStore.from_storage(st)
    ms.write("global", "m.md", "lives in the same db")
    assert ms.read("global", "m.md")["content"] == "lives in the same db"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
