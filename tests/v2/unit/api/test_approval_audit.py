"""Who authorised the dangerous operation must be recoverable afterwards.

2026-07-27 forensics: 11 CONFIRM-level dangerous operations ran that session and
the v2 ``approvals`` table had **0 rows**. The table, its schema and
``ApprovalService.issue()`` had all existed since the v2 logging layer landed —
nothing ever called them. From the audit trail there is no way to establish who
authorised any of those 11, or whether a human was involved at all.

The write goes on the resolve path because that is where a human actually
decides: the request carries the verdict, the edited arguments and the comment,
and it is only reached while the interrupt is live.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/api/test_approval_audit.py -q
"""
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

import json  # noqa: E402

import pytest  # noqa: E402

from mast.api.routes.agents_control import _record_approval  # noqa: E402


class _Body:
    def __init__(self, decision="approve", comment="", edited_args=None):
        self.decision = decision
        self.comment = comment
        self.edited_args = edited_args or {}


class _Svc:
    def __init__(self):
        self.rows: list[dict] = []

    def issue(self, **kw):
        self.rows.append(kw)
        return "appr-1"


class _App:
    def __init__(self, svc):
        class _R:
            approvals = svc
        self._v2_repos = _R() if svc is not None else None


_PENDING = {
    "skill": "BiasPulse",
    "params": {"bias_v": 3.0, "width_s": 0.05},
    "kind": "dangerous_skill",
    "action_id": "act-77",
}


# ════════════════════════════════════════════════════════════════════════════

def test_an_approval_is_recorded():
    svc = _Svc()
    _record_approval(_App(svc), "int-1", _PENDING,
                     _Body("approve", "看过参数，可以"), {"type": "accept"})
    assert len(svc.rows) == 1, "the approval left no audit row — this is the bug"
    row = svc.rows[0]
    assert row["action_id"] == "act-77"
    assert row["approver_kind"] == "human_operator"
    assert row["approval_method"] == "gui_click"


def test_the_evidence_carries_what_was_approved():
    """An audit row saying only "someone approved something" is not an audit."""
    svc = _Svc()
    _record_approval(_App(svc), "int-1", _PENDING,
                     _Body("approve", "看过参数，可以"), {"type": "accept"})
    ev = json.loads(svc.rows[0]["approval_evidence"])
    assert ev["skill"] == "BiasPulse"
    assert ev["params"]["bias_v"] == 3.0
    assert ev["verdict"] == "approve"
    assert ev["comment"] == "看过参数，可以"


def test_a_rejection_is_recorded_too():
    """Refusals matter as much as approvals — "nobody approved it" and "someone
    refused it" are different facts."""
    svc = _Svc()
    _record_approval(_App(svc), "int-2", _PENDING,
                     _Body("reject", "偏压太高"), {"type": "reject"})
    ev = json.loads(svc.rows[0]["approval_evidence"])
    assert ev["verdict"] == "reject"
    assert ev["comment"] == "偏压太高"


def test_edited_arguments_are_recorded():
    """An operator who approves 1.0 V instead of the requested 3.0 V has
    approved a DIFFERENT action; the row must show both."""
    svc = _Svc()
    _record_approval(_App(svc), "int-3", _PENDING,
                     _Body("approve", "", {"bias_v": 1.0}), {"type": "accept"})
    ev = json.loads(svc.rows[0]["approval_evidence"])
    assert ev["params"]["bias_v"] == 3.0        # what was asked for
    assert ev["edited_args"]["bias_v"] == 1.0   # what was allowed


def test_approver_id_is_honest():
    """This build has no per-user auth. "operator" is true; a made-up user id
    would be a fabricated audit trail, which is worse than a coarse one."""
    svc = _Svc()
    _record_approval(_App(svc), "int-4", _PENDING, _Body(), {"type": "accept"})
    assert svc.rows[0]["approver_id"] == "operator"


def test_evidence_is_bounded():
    svc = _Svc()
    big = dict(_PENDING, params={f"k{i}": "x" * 200 for i in range(100)})
    _record_approval(_App(svc), "int-5", big, _Body(), {"type": "accept"})
    assert len(svc.rows[0]["approval_evidence"]) <= 4000


# ── the audit must never be able to break the approval itself ───────────────

def test_no_repos_is_silent():
    _record_approval(_App(None), "int-6", _PENDING, _Body(), {"type": "accept"})


def test_a_failing_service_does_not_raise():
    """The operator has already given the verdict and the worker has already
    been woken. An audit write must not be able to fail that."""
    class _Boom:
        def issue(self, **kw):
            raise RuntimeError("db locked")

    _record_approval(_App(_Boom()), "int-7", _PENDING, _Body(), {"type": "accept"})


@pytest.mark.parametrize("pending", [{}, {"skill": None}, {"params": None}])
def test_sparse_pending_does_not_raise(pending):
    svc = _Svc()
    _record_approval(_App(svc), "int-8", pending, _Body(), {"type": "accept"})
    assert svc.rows and svc.rows[0]["action_id"] == "int-8"  # falls back to the id


def test_unserializable_params_do_not_raise():
    svc = _Svc()
    _record_approval(_App(svc), "int-9", dict(_PENDING, params={"h": object()}),
                     _Body(), {"type": "accept"})
    json.loads(svc.rows[0]["approval_evidence"])   # must still be valid JSON


# ════════════════════════════════════════════════════════════════════════════
# The one that matters: does the ROUTE call it?
# ════════════════════════════════════════════════════════════════════════════

def test_resolving_an_interrupt_writes_the_audit_row():
    """Testing the helper alone is not enough. The first version of this file
    passed unchanged when the call was deleted from resolve_interrupt — an audit
    writer nobody calls is precisely the defect being fixed (the table and
    ApprovalService.issue existed all along; nothing invoked them)."""
    import threading

    from mast.api.routes.agents_control import resolve_interrupt
    from mast.api.schemas_agents_control import ResolveInterruptRequest

    svc = _Svc()
    ev = threading.Event()

    class _LiveApp:
        _orch_interrupts = {
            "lock": threading.RLock(),
            "pending": {"int-x": {
                "skill": "BiasPulse",
                "params": {"bias_v": 3.0},
                "allowed_decisions": ["approve", "reject"],
                "kind": "dangerous_skill",
                "action_id": "act-99",
            }},
            "events": {"int-x": ev},
            "resolved": {},
        }
        _agents_api_state = {"interrupts": {}}

        class _R:
            approvals = svc
        _v2_repos = _R()

    class _Ctx:
        live_app = _LiveApp()

    class _App:
        class state:            # noqa: N801 — mimics FastAPI
            ctx = _Ctx()

    class _Req:
        app = _App()

    res = resolve_interrupt(
        "instrument_control", "int-x",
        ResolveInterruptRequest(decision="approve", comment="确认过参数"),
        _Req(),
    )
    assert res.ok is True and res.applied is True, f"resolve failed: {res}"
    assert ev.is_set(), "the blocked worker was never woken"
    assert svc.rows, (
        "the interrupt resolved but left no audit row — 11 dangerous "
        "operations went unrecorded exactly this way"
    )
    ev_json = json.loads(svc.rows[0]["approval_evidence"])
    assert ev_json["skill"] == "BiasPulse"
    assert ev_json["verdict"] == "approve"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
