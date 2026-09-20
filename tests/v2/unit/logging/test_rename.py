"""Rename experiment / sample — MAST records names (NOT Nanonis fields).

The group (群聊) instrument_control agent used to lack the experiment/sample
session tools AND rename was not implemented anywhere, so it misread '改实验/样品
名称、新建测试实验' as a Nanonis field it couldn't set. These pin the rename path
end to end (storage → ExperimentLog → meta-tool exposure).
"""

from __future__ import annotations

from mast.logging.storage import ExperimentStorage
from mast.logging.experiment_log import ExperimentLog


def test_storage_rename_experiment_and_sample(tmp_path) -> None:
    s = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = s.create_experiment("old exp", "goal")
    assert s.rename_experiment(eid, "new exp") is True
    assert s.get_experiment(eid)["name"] == "new exp"
    assert s.rename_experiment("nonexistent", "x") is False  # honest miss

    sid = s.create_sample(eid, "old sample")
    assert s.rename_sample(sid, "new sample") is True
    assert s.get_sample(sid)["name"] == "new sample"
    assert s.rename_sample("nope", "x") is False


def test_experiment_log_rename_current(tmp_path) -> None:
    s = ExperimentStorage(str(tmp_path / "exp.db"))
    log = ExperimentLog(s)
    log.start_experiment("exp1")
    log.start_sample("sample1")
    # rename defaults to the CURRENT experiment / sample
    assert log.rename_experiment("exp1-renamed") is True
    assert s.get_experiment(log.current_experiment_id)["name"] == "exp1-renamed"
    assert log.rename_sample("sample1-renamed") is True
    assert s.get_sample(log.current_sample_id)["name"] == "sample1-renamed"


def test_meta_tools_expose_experiment_session_tools() -> None:
    """make_meta_tools must include the experiment/sample SESSION tools (create +
    rename) — these are the MAST records tools the group IC was missing."""
    from mast.agents._shared.meta_tools import make_meta_tools

    tools = make_meta_tools(lambda: {})
    names = {t.name for t in tools}
    assert {"start_experiment", "end_experiment", "start_sample", "end_sample",
            "rename_experiment", "rename_sample"} <= names


def test_meta_tool_rename_routes_to_experiment_log(tmp_path) -> None:
    """The rename_experiment tool drives ExperimentLog (MAST records), not Nanonis."""
    from mast.agents._shared.meta_tools import make_meta_tools

    s = ExperimentStorage(str(tmp_path / "exp.db"))
    log = ExperimentLog(s)
    log.start_experiment("orig")
    tools = {t.name: t for t in make_meta_tools(lambda: {"experiment_log": log})}
    out = tools["rename_experiment"].invoke({"name": "renamed-by-tool"})
    assert '"success": true' in out.lower()
    assert s.get_experiment(log.current_experiment_id)["name"] == "renamed-by-tool"
