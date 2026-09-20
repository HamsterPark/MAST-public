"""Waking the right conversation, and refusing to wake the wrong one.

The runtime half of auto-resume. Driven against a stub runtime rather than a real
CoreRuntime: what is under test is the dispatch decision (which conversation, by
which mechanism, or why not at all), not the machinery it dispatches into.

The two concurrency cases are the ones worth staring at — a resume that lands on
top of a turn already in flight would interleave writes into the same checkpoint.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_fetch_resume_runtime.py -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (canonical block for tests/v2/) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import threading

import pytest

from mast.core.runtime import CoreRuntime


class _Store:
    def __init__(self, convs):
        self._convs = convs

    def get(self, cid):
        return self._convs.get(cid)


class _Engine:
    def __init__(self, *, active=False, first_yield=None, on_stream=None):
        self._active = active
        self._first_yield = first_yield
        self._on_stream = on_stream
        self.calls: list[tuple[str, str]] = []
        self.done = threading.Event()

    def is_active(self, thread_id):
        return self._active

    def stream_turn(self, conversation_id, text, *, abort=None, **kw):
        self.calls.append((conversation_id, text))
        if self._on_stream is not None:
            self._on_stream()
        if self._first_yield is not None:
            yield self._first_yield
        self.done.set()


class _Manager:
    def __init__(self):
        self.spawned: list[dict] = []

    def spawn(self, **kw):
        self.spawned.append(kw)
        return {"run_id": "bg1"}


def _runtime(convs, *, engine=None, manager=None, setting=None):
    """A CoreRuntime shell carrying only what the resumer reads."""
    rt = CoreRuntime.__new__(CoreRuntime)
    rt._conv_store = _Store(convs)
    rt._conv_engine = engine
    rt._settings = None if setting is None else _Settings(setting)
    rt._ensure_background_manager = lambda: manager
    return rt


class _Settings:
    def __init__(self, value):
        self._value = value

    def get(self, key, default=None):
        return self._value if key == "literature_fetch_auto_resume" else default


def _req(cid="c1", rid="fr-1", work_id="W1", reason="需要 methods 的偏压"):
    return {"request_id": rid, "work_id": work_id, "reason": reason,
            "title": "A paper", "origin_conversation_id": cid}


_PRIVATE = {"c1": {"conversation_id": "c1", "agent_id": "literature",
                   "kind": "private", "thread_id": "t1"}}
_GROUP = {"g1": {"conversation_id": "g1", "agent_id": "literature",
                 "kind": "group", "thread_id": "gt1"}}


# ── private chats ────────────────────────────────────────────────────────

def test_private_literature_chat_is_continued():
    eng = _Engine()
    rt = _runtime(_PRIVATE, engine=eng)
    out = rt._resume_after_fetch_fulfilled("W1", [_req()])
    assert eng.done.wait(5), "the resume turn never ran"
    assert out["resumed"] == 1
    cid, text = eng.calls[0]
    assert cid == "c1"
    assert "取文请求已满足" in text and "fr-1" in text
    assert "需要 methods 的偏压" in text


def test_several_requests_for_one_chat_are_one_turn():
    eng = _Engine()
    rt = _runtime(_PRIVATE, engine=eng)
    out = rt._resume_after_fetch_fulfilled(
        "W1", [_req(rid="fr-1"), _req(rid="fr-2")])
    assert eng.done.wait(5)
    assert out["resumed"] == 1 and len(eng.calls) == 1
    assert "fr-1" in eng.calls[0][1] and "fr-2" in eng.calls[0][1]


def test_a_turn_in_flight_is_never_interrupted():
    """The whole point of the pre-check: the operator's turn wins."""
    eng = _Engine(active=True)
    rt = _runtime(_PRIVATE, engine=eng)
    out = rt._resume_after_fetch_fulfilled("W1", [_req()])
    assert out["resumed"] == 0 and eng.calls == []
    assert "mid-turn" in out["skipped"][0]["reason"]


def test_toctou_rejection_is_absorbed_quietly():
    """is_active can go stale between the check and the call.

    The engine's own guard then refuses the stream. That refusal arrives as a
    rendered message, which the drain thread consumes; nothing retries, nothing
    raises, and the turn already running is untouched.
    """
    eng = _Engine(first_yield={"text": "⚠️ 已有进行中的回合"})
    rt = _runtime(_PRIVATE, engine=eng)
    out = rt._resume_after_fetch_fulfilled("W1", [_req()])
    assert eng.done.wait(5)
    assert out["resumed"] == 1 and len(eng.calls) == 1


def test_a_failing_resume_thread_does_not_escape():
    def _boom():
        raise RuntimeError("engine exploded")

    eng = _Engine(on_stream=_boom)
    rt = _runtime(_PRIVATE, engine=eng)
    out = rt._resume_after_fetch_fulfilled("W1", [_req()])   # must not raise
    assert out["resumed"] == 1


def test_non_literature_private_chat_is_left_alone():
    """A paper arriving must not interrupt somebody's instrument conversation."""
    convs = {"c1": {"conversation_id": "c1", "agent_id": "instrument_control",
                    "kind": "private", "thread_id": "t1"}}
    eng = _Engine()
    out = _runtime(convs, engine=eng)._resume_after_fetch_fulfilled("W1", [_req()])
    assert out["resumed"] == 0 and eng.calls == []
    assert "not resumable" in out["skipped"][0]["reason"]


def test_no_engine_means_no_resume():
    out = _runtime(_PRIVATE, engine=None)._resume_after_fetch_fulfilled("W1", [_req()])
    assert out["resumed"] == 0


# ── group runs ───────────────────────────────────────────────────────────

def test_group_run_goes_through_the_background_manager():
    """The group's own thread belongs to the run-task machine; do not touch it."""
    mgr = _Manager()
    rt = _runtime(_GROUP, engine=_Engine(), manager=mgr)
    out = rt._resume_after_fetch_fulfilled("W1", [_req(cid="g1")])
    assert out["resumed"] == 1
    spawn = mgr.spawned[0]
    assert spawn["conversation_id"] == "g1"
    assert spawn["agents"] == ("literature",)
    assert "取文请求已满足" in spawn["instruction"]


def test_group_without_a_manager_is_skipped():
    rt = _runtime(_GROUP, engine=_Engine(), manager=None)
    out = rt._resume_after_fetch_fulfilled("W1", [_req(cid="g1")])
    assert out["resumed"] == 0 and "background manager" in out["skipped"][0]["reason"]


# ── who gets skipped ─────────────────────────────────────────────────────

def test_the_conversation_that_fetched_it_is_excluded():
    """It is already awake and about to use the paper itself."""
    eng = _Engine()
    rt = _runtime(_PRIVATE, engine=eng)
    out = rt._resume_after_fetch_fulfilled(
        "W1", [_req(cid="c1")], exclude_conversation_id="c1")
    assert out["resumed"] == 0 and eng.calls == []


def test_requests_without_an_origin_are_skipped():
    """Background runs post with no conversation — there is no session to resume."""
    eng = _Engine()
    rt = _runtime(_PRIVATE, engine=eng)
    out = rt._resume_after_fetch_fulfilled("W1", [_req(cid="")])
    assert out["resumed"] == 0 and eng.calls == []


def test_a_deleted_conversation_is_skipped():
    eng = _Engine()
    out = _runtime({}, engine=eng)._resume_after_fetch_fulfilled("W1", [_req()])
    assert out["resumed"] == 0 and "gone" in out["skipped"][0]["reason"]


def test_the_setting_can_turn_it_off():
    eng = _Engine()
    rt = _runtime(_PRIVATE, engine=eng, setting=False)
    out = rt._resume_after_fetch_fulfilled("W1", [_req()])
    assert out["resumed"] == 0 and eng.calls == []


def test_absent_setting_defaults_to_on():
    """Uploading a paper IS the instruction to use it."""
    eng = _Engine()
    rt = _runtime(_PRIVATE, engine=eng, setting=None)
    assert rt._resume_after_fetch_fulfilled("W1", [_req()])["resumed"] == 1
    assert eng.done.wait(5)


def test_setting_key_is_whitelisted():
    """An unknown key is silently dropped by SettingsStore.update()."""
    from mast.webui.settings_store import KNOWN_KEYS
    assert "literature_fetch_auto_resume" in KNOWN_KEYS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
