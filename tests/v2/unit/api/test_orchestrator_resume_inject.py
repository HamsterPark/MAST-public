"""run-task prepends resume / operator-reply context (2026-07 analysis ⑥/⑦).

The supervisor's initial message stream on a fresh thread was JUST the operator's
"继续", so it answered 「No prior context or ongoing task found」. ``run_task`` now
prepends, best-effort, (⑥) the unfinished-experiment block on a continuation
intent and (⑦) any answered-but-undelivered 心愿单 request — marking the latter
delivered so it lands exactly once. These pin ``_resume_lead_messages`` directly.
"""

from __future__ import annotations

import pytest

from mast.api.routes.orchestrator import _resume_lead_messages
from mast.core.types import ActionRecord
from mast.logging.experiment_log import ExperimentLog
from mast.logging.storage import ExperimentStorage
from mast.wishlist import get_board, post_agent_request, reset_default_board, resolve_agent_request


class _App:
    """Minimal stand-in for the CoreRuntime attributes the helper reads."""

    def __init__(self, storage=None, log=None, plan_store=None):
        self._storage = storage
        self._experiment_log = log
        self._plan_store = plan_store


@pytest.fixture()
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    reset_default_board()
    yield get_board()
    reset_default_board()


def _texts(msgs) -> str:
    return "\n".join(getattr(m, "content", "") for m in msgs)


def _running_exp(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    log = ExperimentLog(st)
    log.start_experiment("NiI2 study", "characterize NiI2")
    log.start_sample("film A")
    log.log_skill_execution(ActionRecord(skill_name="StartScan"))
    return _App(st, log)


# ── ⑥ resume experiment context on 继续 ───────────────────────────────────────
def test_continue_injects_experiment_context(board, tmp_path):
    app = _running_exp(tmp_path)
    lead = _resume_lead_messages(app, "继续")
    assert lead, "继续 with a running experiment must inject context"
    assert "NiI2 study" in _texts(lead)
    assert "恢复上下文" in _texts(lead)


def test_fresh_specific_task_does_not_inject_experiment_context(board, tmp_path):
    app = _running_exp(tmp_path)
    lead = _resume_lead_messages(app, "扫描 Au(111) 5nm 区域")
    # not a continuation → no experiment resume block (no undelivered replies either)
    assert "NiI2 study" not in _texts(lead)


def test_continue_with_no_running_experiment_injects_nothing(board, tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    app = _App(st, ExperimentLog(st))
    assert _resume_lead_messages(app, "继续") == []


# ── ⑦ operator reply — supervisor gets a READ-ONLY routing hint ───────────────
# The full path/note is delivered to the AGENT by RequestReplyReadbackMiddleware
# (before_model); the route only nudges the supervisor to dispatch that agent, and
# must NOT mark anything delivered (else it would race the middleware).
def test_answered_request_injects_routing_hint_read_only(board, tmp_path):
    app = _App()  # no experiment needed — this is the ⑦ path
    rid = post_agent_request("data_processing", "请提供 Au111 图路径", kind="info")["id"]
    resolve_agent_request(rid, "done", path=r"D:\Data\Au111.sxm")

    lead = _resume_lead_messages(app, "扫描下一个区域")
    text = _texts(lead)
    assert rid in text                       # the hint names the answered request
    assert "用户已答复" in text              # hint header
    assert r"D:\Data\Au111.sxm" not in text  # the PATH is the middleware's job, not the route
    # read-only: the route did NOT consume it — the middleware still has it to deliver
    assert get_board().resolved_requests_for("", undelivered_only=True), (
        "the route hint must be read-only so the agent middleware still delivers")


def test_both_blocks_can_be_injected_together(board, tmp_path):
    app = _running_exp(tmp_path)
    rid = post_agent_request("data_processing", "路径?", kind="info")["id"]
    resolve_agent_request(rid, "done", path="D:\\x.sxm")
    lead = _resume_lead_messages(app, "继续")
    text = _texts(lead)
    assert "NiI2 study" in text          # ⑥ experiment resume block
    assert rid in text                   # ⑦ reply routing hint (names the request)


def test_never_raises_without_any_core(board):
    # a bare app with no storage/log must degrade to [] (or reply-only), never raise
    assert isinstance(_resume_lead_messages(_App(), "继续"), list)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
