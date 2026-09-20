"""GET /api/diagnostics — the refusal ledger, served to the operator.

The unit tests prove the refusal SITES write. This proves the operator can READ
them: a real refusal, produced by the real gate, comes back through the real
endpoint with the fields that make it actionable.

Without this seam the ledger is a file nobody opens — which is how the 2026-07-10
trial went: the information existed in principle (a log line here, a ToolMessage
there) and no one could get at it. 「进针功能调用失败」 with no cause, twice.
"""
from __future__ import annotations

import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.diagnostics import router
from mast.core import diagnostics as diag


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    diag.clear()
    diag.set_run_id("")
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    yield TestClient(app)
    diag.clear()


def test_empty_is_not_degraded(client):
    """Nothing refused yet is a TRUE statement, not a broken layer."""
    b = client.get("/api/diagnostics").json()
    assert b["degraded"] is False
    assert b["count"] == 0 and b["entries"] == []
    assert b["summary"]["total"] == 0


def test_a_real_precondition_refusal_reaches_the_operator(client):
    """End to end through the REAL gate: AutoApproach declares `bias_nonzero`, the
    bias is 0, the skill is refused — and the operator can now see WHY, and against
    WHAT STATE. That second half is what nobody had."""
    from mast.core.types import HardwareState
    from mast.skills.builtins.approach import AutoApproach

    AutoApproach().check_preconditions(HardwareState(bias_v=0.0))

    b = client.get("/api/diagnostics", params={"group": "refusals"}).json()
    assert b["degraded"] is False
    assert b["count"] == 1
    e = b["entries"][0]
    assert e["kind"] == "precondition_block"
    assert e["subject"] == "AutoApproach"
    assert e["reason"]
    assert e["fields"]["declared"] == ["bias_nonzero"]
    assert e["fields"]["state"]["bias_v"] == 0.0


def test_a_post_abort_write_refusal_reaches_the_operator(client):
    from mast.core.execution_context import ExecutionContext
    from mast.core.types import NanonisCallRecord

    class _Pool:
        def safe_call(self, m, *a, role="main"):
            return NanonisCallRecord(method=m, args=a)

    ev = threading.Event()
    ev.set()
    ExecutionContext(pool=_Pool(), state=None, registry=None,
                     abort_event=ev, run_id="task-9").safe_call("Bias_Set", 5.0)

    b = client.get("/api/diagnostics", params={"run_id": "task-9"}).json()
    assert b["count"] == 1
    assert b["entries"][0]["kind"] == "abort_block"
    assert b["entries"][0]["subject"] == "Bias_Set"


def test_the_summary_names_the_spin_without_reading_a_log(client):
    """#31, answered at a glance: one subject dominating the refusal count IS the
    spin. The operator had to infer this from 「任务步数达到上限」."""
    for _ in range(6):
        diag.record("precondition_block", "AutoApproach", "bias_nonzero 未满足")
    diag.record("safety_block", "SetBias", "超出全局上限")

    s = client.get("/api/diagnostics").json()["summary"]
    assert s["total"] == 7
    assert s["by_kind"]["precondition_block"] == 6
    assert s["top_refusals"][0]["what"] == "precondition_block:AutoApproach"
    assert s["top_refusals"][0]["count"] == 6
    assert s["log_path"].endswith("refusals.jsonl")


class TestFilters:
    def _seed(self):
        diag.record("precondition_block", "AutoApproach", "a")
        diag.record("step_skip", "GridSTS.sts_0_0", "b")
        diag.record("stall", "instrument_control:GetBias", "c")

    def test_group_refusals(self, client):
        self._seed()
        b = client.get("/api/diagnostics", params={"group": "refusals"}).json()
        assert [e["kind"] for e in b["entries"]] == ["precondition_block"]

    def test_group_steps(self, client):
        self._seed()
        b = client.get("/api/diagnostics", params={"group": "steps"}).json()
        assert [e["kind"] for e in b["entries"]] == ["step_skip"]

    def test_group_stall(self, client):
        self._seed()
        b = client.get("/api/diagnostics", params={"group": "stall"}).json()
        assert [e["subject"] for e in b["entries"]] == ["instrument_control:GetBias"]

    def test_subject_substring(self, client):
        self._seed()
        b = client.get("/api/diagnostics", params={"subject": "gridsts"}).json()
        assert b["count"] == 1 and b["entries"][0]["subject"] == "GridSTS.sts_0_0"


def test_newest_first(client):
    diag.record("note", "first", "1")
    diag.record("note", "second", "2")
    b = client.get("/api/diagnostics").json()
    assert [e["subject"] for e in b["entries"]] == ["second", "first"]
