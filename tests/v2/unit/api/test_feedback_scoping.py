"""Feedback must retain its sample, experiment and current conversation context; missing client fields use server-side state, while inactive tasks must not supply stale identifiers."""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402


class _Log:
    """Stands in for the active experiment log."""
    current_experiment_id = "exp-abc"
    current_sample_id = "sample-xyz"


class _Storage:
    def __init__(self):
        self.rows: list[dict] = []

    def log_feedback(self, **kw):
        self.rows.append(kw)
        return len(self.rows)


@pytest.fixture()
def route(monkeypatch):
    """post_feedback with a fake storage, active log and live task slot."""
    from mast.api.routes import records as R

    storage = _Storage()
    monkeypatch.setattr("mast.logging.experiment_log.get_active_log",
                        lambda: _Log(), raising=False)

    class _Ctx:
        experiment_storage = storage

    class _App:
        _agents_api_state = {"task": {"active": True,
                                      "conversation_id": "conv-live-123"}}

        class state:                       # noqa: N801 - mimics FastAPI
            ctx = _Ctx()

    class _Req:
        app = _App()

    return R, _Req(), storage


def _body(**kw):
    from mast.api.schemas_records import FeedbackRequest
    base = {"rating": "", "comment": "扫描图有问题", "experiment_id": None,
            "sample_id": None, "conversation_id": None, "agent": "",
            "meta": {}}
    base.update(kw)
    return FeedbackRequest(**base)


# ════════════════════════════════════════════════════════════════════════════

def test_sample_id_falls_back_to_the_active_sample(route):
    """Missing sample id uses the active sample."""
    R, req, storage = route
    R.post_feedback(_body(), req)
    assert storage.rows[-1]["sample_id"] == "sample-xyz", (
        "feedback still lands with no sample attached")


def test_conversation_id_falls_back_to_the_live_task(route):
    R, req, storage = route
    R.post_feedback(_body(), req)
    assert storage.rows[-1]["conversation_id"] == "conv-live-123"


def test_experiment_id_fallback_still_works(route):
    """Pin the 2026-07-06 fix so this change cannot regress it."""
    R, req, storage = route
    R.post_feedback(_body(), req)
    assert storage.rows[-1]["experiment_id"] == "exp-abc"


def test_explicit_values_are_never_overridden(route):
    """A caller that knows better wins — the fallback fills gaps, it does not
    correct the caller."""
    R, req, storage = route
    R.post_feedback(_body(experiment_id="exp-given",
                          sample_id="sample-given",
                          conversation_id="conv-given"), req)
    row = storage.rows[-1]
    assert row["experiment_id"] == "exp-given"
    assert row["sample_id"] == "sample-given"
    assert row["conversation_id"] == "conv-given"


def test_no_live_task_means_no_invented_conversation(route, monkeypatch):
    """An inactive task must not provide a conversation identifier."""
    R, req, storage = route
    req.app._agents_api_state = {"task": {"active": False,
                                          "conversation_id": "conv-stale"}}
    R.post_feedback(_body(), req)
    assert storage.rows[-1]["conversation_id"] is None


def test_missing_active_log_does_not_break_the_write(route, monkeypatch):
    """Feedback is the operator telling us something is wrong — losing it
    because the scoping lookup failed would be the worst possible trade."""
    R, req, storage = route

    def _boom():
        raise RuntimeError("no active log")

    monkeypatch.setattr("mast.logging.experiment_log.get_active_log",
                        _boom, raising=False)
    res = R.post_feedback(_body(), req)
    assert res.ok is True
    assert storage.rows[-1]["comment"] == "扫描图有问题"


@pytest.mark.parametrize("state", [None, {}, {"task": None}, "not-a-dict"])
def test_malformed_agents_state_does_not_break_the_write(route, state):
    """Same trade as above: never lose the operator's words to a lookup."""
    R, req, storage = route
    req.app._agents_api_state = state
    res = R.post_feedback(_body(), req)
    assert res.ok is True
    assert storage.rows[-1]["conversation_id"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
