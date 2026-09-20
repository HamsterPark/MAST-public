"""What a skill returned must reach the operator, not just the database.

 the v1 ``actions`` table had data / state / nanonis_calls /
artifact_path empty for the entire session — the result dict was discarded, so
every action read ``"data": {}``, which is indistinguishable from a skill that
returned nothing. That storage side is fixed.

But ``ActionSummary`` still flattened those columns away, so the fix existed in
the database and nowhere the operator could see it — and the operator is the
person the record exists for. This pins the surfacing, and the two ways it can
go wrong: an unbounded payload, and an unserializable one.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/api/test_action_payload_surface.py -q
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

from mast.api.routes.records import (  # noqa: E402
    _ACTION_DATA_MAX_CHARS,
    _action_payload,
)
from mast.api.schemas_records import ActionSummary  # noqa: E402


class _Action:
    def __init__(self, **kw):
        self.nanonis_calls = kw.get("nanonis_calls", [])
        self.artifact_path = kw.get("artifact_path")
        self.data = kw.get("data")


# ════════════════════════════════════════════════════════════════════════════

def test_a_real_result_reaches_the_operator():
    p = _action_payload(_Action(
        data={"setpoint_a": 1e-10, "bias_v": 0.5},
        artifact_path=r"D:\scan\Au111_001.sxm",
        nanonis_calls=[1, 2, 3],
    ))
    assert p["data"] == {"setpoint_a": 1e-10, "bias_v": 0.5}
    assert p["artifact_path"] == r"D:\scan\Au111_001.sxm"
    assert p["nanonis_calls"] == 3
    assert p["data_truncated"] is False


def test_empty_stays_empty():
    """An action that genuinely returned nothing must not gain a fake payload."""
    p = _action_payload(_Action())
    assert p == {"data": {}, "artifact_path": None,
                 "nanonis_calls": 0, "data_truncated": False}


def test_a_huge_payload_is_summarised_not_inlined():
    """A spectrum is a legitimate action result; a timeline of them is megabytes."""
    p = _action_payload(_Action(data={"spectrum": list(range(5000))}))
    assert p["data_truncated"] is True
    assert "spectrum" in p["data"]["_keys"]
    assert len(json.dumps(p["data"])) < _ACTION_DATA_MAX_CHARS
    assert "_summary" in p["data"], "truncation must say what was dropped"


def test_unserializable_values_do_not_break_the_timeline():
    """A skill result can hold a Path, a numpy scalar, a client handle. Handing
    that straight to Pydantic would 500 the WHOLE timeline over one bad action —
    and the record exists precisely for the runs that went wrong."""
    p = _action_payload(_Action(data={
        "path": Path(r"D:\x.sxm"), "obj": object(), "n": 42,
    }))
    # Whatever survives must be pure JSON.
    json.dumps(p["data"])            # must not raise
    assert p["data"]["n"] == 42
    assert isinstance(p["data"]["path"], str)


def test_the_whole_row_survives_pydantic_and_json():
    """End-to-end: the shape the route actually returns."""
    p = _action_payload(_Action(data={"path": Path(r"D:\x.sxm")},
                                nanonis_calls=[1]))
    row = ActionSummary(id="a1", skill_name="FullScan", **p)
    blob = json.dumps(row.model_dump(), ensure_ascii=False)
    assert "FullScan" in blob


@pytest.mark.parametrize("bad", [None, "", 0, [], "a string", 123])
def test_non_dict_data_degrades_quietly(bad):
    p = _action_payload(_Action(data=bad))
    assert p["data"] == {}
    assert p["data_truncated"] is False


def test_missing_attributes_degrade_quietly():
    """An older row, or a different record type, must not raise."""
    class _Bare:
        pass

    assert _action_payload(_Bare())["data"] == {}


def test_the_route_actually_surfaces_the_payload():
    """The one that matters: does GET /experiments/{id} carry it?

    Testing the helper alone is not enough — the first version of this file
    passed unchanged when ``**_action_payload(a)`` was deleted from the route,
    because nothing here exercised the route. A helper nobody calls is exactly
    the shape of the bug being fixed.
    """
    from mast.api.routes import records as R

    class _Rec:
        id = "act-1"
        experiment_id = "exp-1"
        sample_id = None
        timestamp = "2026-07-27T15:03:29"
        skill_name = "AcquireSTS"
        skill_version = "1.0.0"
        parameters = {"bias_v": 0.5}
        success = True
        error = ""
        duration_s = 3.2
        context = None
        approval_source = None
        data = {"spectrum_path": r"D:\d\sts_001.dat", "n_points": 401}
        artifact_path = r"D:\d\sts_001.dat"
        nanonis_calls = [1, 2]

    class _Storage:
        def get_experiment(self, *a, **k):
            # Must be truthy — the route short-circuits to found=False otherwise.
            return {"id": "exp-1", "name": "synthetic_sample", "status": "running"}

        def get_actions(self, *a, **k):
            return [_Rec()]

        def get_samples(self, *a, **k):
            return []

        def get_map_markers(self, *a, **k):
            return []

        def get_feedback(self, *a, **k):
            return []

    class _Ctx:
        experiment_storage = _Storage()

    class _App:
        class state:            # noqa: N801 — mimics FastAPI
            ctx = _Ctx()

    class _Req:
        app = _App()

    detail = R.get_experiment_detail("exp-1", _Req())
    assert detail.actions, "route returned no actions"
    a = detail.actions[0]
    assert a.artifact_path == r"D:\d\sts_001.dat", (
        "artifact_path is in the database but not in the API response")
    assert a.data.get("n_points") == 401, (
        "the skill's result never reached the operator")
    assert a.nanonis_calls == 2


def test_schema_actually_carries_the_fields():
    """Guard against the field being dropped from the model again — that is
    exactly how the storage fix stayed invisible."""
    f = ActionSummary.model_fields
    for name in ("data", "artifact_path", "nanonis_calls", "data_truncated"):
        assert name in f, f"ActionSummary lost {name!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
