"""v2 unit tests for the MASTState `libraries` channel + `merge_libraries` reducer (P1).

Design (owner G): the big OpenAlex index is the ONE true library; every other
library is a SET OF work_id string pointers into it. The state `libraries`
channel is a LIGHTWEIGHT JSON snapshot (active_id + per-library member work_id
pointers) so a checkpoint / handoff can carry it. The authoritative store stays
in artifacts/literature_libs/registry.json.

These tests pin:
  - reducer identity / idempotency
  - incremental member union (never drops history)
  - active_id last-non-empty-writer-wins
  - name/scope right-side-wins
  - new-library merge keeps both sides
  - the merged value is pure-JSON-serializable (no tensors/handles)
  - `libraries` is a declared NotRequired field with merge_libraries attached

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/test_state_libraries.py -q -p no:randomly
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (canonical block for tests/v2/) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import json

from mast.agents.state import MASTState, merge_libraries


# ──────────────────────────────────────────────────────────────────────
# Identity / idempotency
# ──────────────────────────────────────────────────────────────────────

def test_merge_identity_none_right():
    x = {"active_id": "reading", "items": {"reading": {"name": "reading library",
                                                       "scope": "global", "members": ["W1"]}}}
    assert merge_libraries(x, None) == x


def test_merge_identity_none_left():
    x = {"active_id": "reading", "items": {"reading": {"name": "reading library",
                                                       "scope": "global", "members": ["W1"]}}}
    assert merge_libraries(None, x) == x


def test_merge_both_none_is_empty():
    assert merge_libraries(None, None) == {}


def test_merge_idempotent():
    x = {"active_id": "lib_a", "items": {"lib_a": {"name": "A", "scope": "custom",
                                                   "members": ["W1", "W2"]}}}
    assert merge_libraries(x, x) == x


# ──────────────────────────────────────────────────────────────────────
# Incremental member union (never drops history)
# ──────────────────────────────────────────────────────────────────────

def test_member_union_is_additive():
    left = {"active_id": "lib_a",
            "items": {"lib_a": {"name": "A", "scope": "custom", "members": ["W1", "W2"]}}}
    right = {"active_id": "lib_a",
             "items": {"lib_a": {"name": "A", "scope": "custom", "members": ["W2", "W3"]}}}
    merged = merge_libraries(left, right)
    # order-preserving union: left first, then new from right
    assert merged["items"]["lib_a"]["members"] == ["W1", "W2", "W3"]


def test_member_union_preserves_left_only_member():
    # A stale right snapshot that "lost" W1 must NOT erase it from the merged view.
    left = {"items": {"lib_a": {"name": "A", "scope": "custom", "members": ["W1", "W2"]}}}
    right = {"items": {"lib_a": {"name": "A", "scope": "custom", "members": ["W2"]}}}
    merged = merge_libraries(left, right)
    assert merged["items"]["lib_a"]["members"] == ["W1", "W2"]


def test_new_library_on_right_is_kept():
    left = {"active_id": "reading",
            "items": {"reading": {"name": "reading library", "scope": "global", "members": []}}}
    right = {"active_id": "kondo",
             "items": {"kondo": {"name": "Kondo", "scope": "custom", "members": ["W100"]}}}
    merged = merge_libraries(left, right)
    assert set(merged["items"].keys()) == {"reading", "kondo"}
    assert merged["items"]["kondo"]["members"] == ["W100"]


def test_new_library_on_left_is_kept():
    left = {"items": {"kondo": {"name": "Kondo", "scope": "custom", "members": ["W100"]}}}
    right = {"items": {"reading": {"name": "reading library", "scope": "global", "members": []}}}
    merged = merge_libraries(left, right)
    assert set(merged["items"].keys()) == {"reading", "kondo"}


# ──────────────────────────────────────────────────────────────────────
# active_id + name/scope resolution
# ──────────────────────────────────────────────────────────────────────

def test_active_id_right_wins_when_present():
    left = {"active_id": "reading", "items": {}}
    right = {"active_id": "kondo", "items": {}}
    assert merge_libraries(left, right)["active_id"] == "kondo"


def test_active_id_keeps_left_when_right_empty():
    left = {"active_id": "reading", "items": {}}
    right = {"active_id": "", "items": {"kondo": {"name": "K", "scope": "custom", "members": []}}}
    assert merge_libraries(left, right)["active_id"] == "reading"


def test_name_and_scope_right_wins():
    left = {"items": {"lib_a": {"name": "old name", "scope": "custom", "members": ["W1"]}}}
    right = {"items": {"lib_a": {"name": "new name", "scope": "experiment", "members": []}}}
    merged = merge_libraries(left, right)
    rec = merged["items"]["lib_a"]
    assert rec["name"] == "new name"
    assert rec["scope"] == "experiment"
    # members still union (left's W1 preserved)
    assert rec["members"] == ["W1"]


# ──────────────────────────────────────────────────────────────────────
# JSON serializability (checkpoint-safety: no tensors/handles)
# ──────────────────────────────────────────────────────────────────────

def test_merged_value_is_json_serializable():
    left = {"active_id": "reading",
            "items": {"reading": {"name": "reading library", "scope": "global",
                                  "members": ["W1", "W2"]}}}
    right = {"active_id": "kondo",
             "items": {"kondo": {"name": "Kondo", "scope": "custom",
                                 "members": ["W100", "local:abc"]}}}
    merged = merge_libraries(left, right)
    s = json.dumps(merged)  # must not raise
    back = json.loads(s)
    assert back == merged
    # every member is a string pointer (work_id), never an object
    for rec in back["items"].values():
        assert all(isinstance(m, str) for m in rec["members"])


# ──────────────────────────────────────────────────────────────────────
# Field declaration / reducer wiring
# ──────────────────────────────────────────────────────────────────────

def test_libraries_is_declared_state_field_with_reducer():
    # `libraries` is a NotRequired Annotated field carrying merge_libraries.
    # Resolve string annotations (PEP 563 / `from __future__ import annotations`)
    # WITH extras so Annotated metadata survives.
    import typing
    hints = typing.get_type_hints(MASTState, include_extras=True)
    assert "libraries" in hints, "MASTState must declare a `libraries` channel"
    # NotRequired[Annotated[dict, merge_libraries]] — confirm the reducer is wired.
    found_reducer = merge_libraries in _collect_metadata(hints["libraries"])
    assert found_reducer, "merge_libraries must be the reducer for `libraries`"


def _collect_metadata(tp) -> list:
    """Recursively pull __metadata__ entries out of nested typing constructs."""
    out: list = []
    meta = getattr(tp, "__metadata__", None)
    if meta:
        out.extend(meta)
    for arg in getattr(tp, "__args__", ()) or ():
        out.extend(_collect_metadata(arg))
    return out


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-q"])
