"""Regression pin for the 2026-07-03 review — compaction failure must not wipe
the conversation.

The base SummarizationMiddleware returns "Error generating summary: ..." when the
summarizer LLM fails, and then REPLACES the whole deleted history with that
string — irreversibly losing the experiment context. Our subclass must drop such
a compaction (keep the full history this turn) instead.
"""
from __future__ import annotations

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

from mast.agents._shared.compaction_mw import _MemorySinkSummarization


def test_guard_drops_failed_compaction():
    mw = _MemorySinkSummarization.__new__(_MemorySinkSummarization)
    # A base result that replaced history with the failure sentinel.
    class _Msg:
        def __init__(self, content):
            self.content = content
    bad = {"messages": [_Msg("Here is a summary of the conversation to date:\n\n"
                             "Error generating summary: boom")]}
    assert mw._guard_compaction(bad) is None, "failed compaction must be dropped"


def test_guard_keeps_good_compaction():
    mw = _MemorySinkSummarization.__new__(_MemorySinkSummarization)
    class _Msg:
        def __init__(self, content):
            self.content = content
    good = {"messages": [_Msg("Here is a summary of the conversation to date:\n\n"
                              "Operator scanned Au(111), tip conditioned, region A used.")]}
    assert mw._guard_compaction(good) is good, "a real summary must be applied"


def test_guard_passthrough_none():
    mw = _MemorySinkSummarization.__new__(_MemorySinkSummarization)
    assert mw._guard_compaction(None) is None
