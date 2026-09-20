"""CoreRuntime._record_v1_action persists EVERY skill (success AND failure) into
the V1 experiment record — the store the Records UI reads (get_experiment_detail
→ storage.get_actions).

审查: the agent path wrote nothing to the V1 actions table, so a
whole crashed-tip session (Nanonis timeouts, CRASH_DETECTED, module-not-running)
showed a BLANK action timeline — the failures were invisible in the records.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/core/test_record_v1_action.py -x -v
"""
from __future__ import annotations

from types import SimpleNamespace

from mast.core.runtime import CoreRuntime
from mast.logging.storage import ExperimentStorage


def _rt(storage, eid):
    # duck-typed self: _record_v1_action only touches _storage + _experiment_log.
    el = SimpleNamespace(current_experiment_id=eid, current_sample_id=None)
    return SimpleNamespace(_storage=storage, _experiment_log=el)


def test_records_success_and_failure(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("NiI2")
    rt = _rt(st, eid)

    CoreRuntime._record_v1_action(rt, {
        "skill": "GetCurrent", "skill_version": "1.0.0",
        "params": {"channel": 0}, "success": True, "error": "", "duration_ms": 12,
    })
    CoreRuntime._record_v1_action(rt, {
        "skill": "GetBias", "params": {}, "success": False,
        "error": "TimeoutError: timed out", "duration_ms": 3000,
    })

    acts = st.get_actions(eid)
    assert len(acts) == 2
    by = {a.skill_name: a for a in acts}
    assert by["GetCurrent"].result.success is True
    assert by["GetBias"].result.success is False           # the failure IS recorded
    assert "timed out" in (by["GetBias"].result.error or "")
    assert by["GetBias"].duration_s == 3.0


def test_fail_safe_without_storage():
    # no storage wired → must be a silent no-op, never raise.
    rt = SimpleNamespace(_storage=None, _experiment_log=None)
    CoreRuntime._record_v1_action(rt, {"skill": "X", "success": True})


def test_non_serialisable_params_do_not_raise(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    rt = _rt(st, eid)
    # a stray object in params must not crash action_to_dict's json.dumps.
    CoreRuntime._record_v1_action(rt, {
        "skill": "SetBias", "params": {"obj": object(), "v": 1.0},
        "success": True, "error": "", "duration_ms": 5,
    })
    acts = st.get_actions(eid)
    assert len(acts) == 1
    assert acts[0].skill_name == "SetBias"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
