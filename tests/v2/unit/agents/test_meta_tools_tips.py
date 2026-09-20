"""针尖登记的 agent 工具。

形状照样品工具:**接受名字不是 UUID**(模型看不见 DB)、词表 miss 回候选、
provider 拿不到东西时如实降级而不是抛。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/agents/test_meta_tools_tips.py -q
"""
from __future__ import annotations

import json

import pytest

from mast.agents._shared.meta_tools import (
    TIP_READ_TOOL_NAMES,
    TIP_TOOL_NAMES,
    make_meta_tools,
)
from mast.core import instrument_profile as iprof
from mast.core import tip_state
from mast.logging.storage import ExperimentStorage


@pytest.fixture(autouse=True)
def _clean():
    iprof.set_persist_sink(None)
    iprof.set_profile({})
    tip_state.set_current_tip(None)
    yield
    iprof.set_persist_sink(None)
    iprof.set_profile({})
    tip_state.set_current_tip(None)


def _tools(storage):
    by_name = {t.name: t for t in make_meta_tools(lambda: {"storage": storage})}
    return by_name


def _call(tool, **kw) -> dict:
    return json.loads(tool.invoke(kw))


@pytest.fixture
def tools(tmp_path):
    return _tools(ExperimentStorage(str(tmp_path / "exp.db")))


# ── 暴露 ────────────────────────────────────────────────────────────────────

def test_all_five_tools_are_exposed(tools) -> None:
    for name in TIP_TOOL_NAMES:
        assert name in tools, f"{name} 没有出现在 make_meta_tools() 里"


def test_read_tools_are_a_subset_of_all_tip_tools() -> None:
    assert set(TIP_READ_TOOL_NAMES) < set(TIP_TOOL_NAMES)
    assert "register_tip" not in TIP_READ_TOOL_NAMES, (
        "登记换针会清掉学习标定 —— 不该出现在只读子集里")


def test_tip_tools_are_not_in_the_lifecycle_subset() -> None:
    """LIFECYCLE 子集整体发给 experiment_design;写工具进去 = XD 能清标定。"""
    from mast.agents._shared.meta_tools import LIFECYCLE_TOOL_NAMES

    assert not (set(TIP_TOOL_NAMES) & set(LIFECYCLE_TOOL_NAMES))


# ── register ────────────────────────────────────────────────────────────────

def test_register_then_get_current_round_trips(tools) -> None:
    out = _call(tools["register_tip"], material="钨", fabrication="电化学腐蚀")
    assert out["success"] is True
    assert out["tip"]["material"] == "W"
    assert out["tip"]["fabrication"] == "etched"

    cur = _call(tools["get_current_tip"])
    assert cur["registered"] is True
    assert cur["tip"]["material"] == "W"


def test_get_current_says_not_registered_rather_than_failing(tools) -> None:
    out = _call(tools["get_current_tip"])
    assert out["success"] is True and out["registered"] is False
    assert "登记" in out["message"]


def test_register_reports_which_calibration_it_cleared(tools) -> None:
    """模型要知道换针清掉了什么 —— 下一步该重新标定,而不是照用旧值。"""
    _call(tools["register_tip"], material="W")
    iprof.set_profile({"didv_at_contact_v": 2e-3, "qplus_amplitude_baseline": 9.0})

    out = _call(tools["register_tip"], material="PtIr", fabrication="cut")
    assert "didv_at_contact_v" in out["cleared_calibration"]
    assert "qplus_amplitude_baseline" in out["cleared_calibration"]
    assert "清除" in out["message"]


def test_unknown_material_returns_candidates(tools) -> None:
    """miss 要给候选,否则模型只会换个说法重试同一个错。"""
    out = _call(tools["register_tip"], material="镝钪合金")
    assert out["success"] is True                      # 自由文本仍然被接受
    assert "material_candidates" in out
    assert any("PtIr" in c for c in out["material_candidates"])
    assert out["warnings"]


def test_unknown_fabrication_returns_candidates(tools) -> None:
    out = _call(tools["register_tip"], material="W", fabrication="魔法")
    assert "fabrication_candidates" in out
    assert any("etched" in c for c in out["fabrication_candidates"])


def test_register_accepts_a_backdated_install_date(tools) -> None:
    out = _call(tools["register_tip"], material="W", installed_at="2026-07-20")
    assert out["tip"]["installed_at"].startswith("2026-07-20")


def test_register_records_qplus_parameters(tools) -> None:
    out = _call(tools["register_tip"], material="PtIr", form="qplus",
                qplus_sensor_model="TF-32k", qplus_f0_hz=32768.0, qplus_q=30000.0)
    assert out["tip"]["form"] == "qplus"
    assert out["tip"]["qplus_f0_hz"] == 32768.0
    assert tip_state.is_qplus() is True


# ── list / update / remove ──────────────────────────────────────────────────

def test_list_tips_is_the_change_history(tools) -> None:
    _call(tools["register_tip"], material="W", name="first")
    _call(tools["register_tip"], material="PtIr", name="second")
    out = _call(tools["list_tips"])
    assert out["count"] == 2
    assert out["tips"][0]["name"] == "second"
    assert out["tips"][1]["removed_at"], "上一根针应已退役"


def test_update_defaults_to_the_current_tip(tools) -> None:
    _call(tools["register_tip"], material="W", fabrication="etched")
    out = _call(tools["update_tip"], wire_diameter_mm=0.25, note="第一根")
    assert out["success"] is True
    assert out["tip"]["wire_diameter_mm"] == 0.25
    assert out["tip"]["note"] == "第一根"


def test_update_takes_a_name_not_a_uuid(tools) -> None:
    """模型永远拿不到 UUID —— 它看不见 DB。"""
    _call(tools["register_tip"], material="W", name="老针")
    _call(tools["register_tip"], material="PtIr", name="新针")

    out = _call(tools["update_tip"], tip_name="老针", note="退役时已钝")
    assert out["success"] is True
    assert out["tip"]["note"] == "退役时已钝"
    assert out["tip"]["name"] == "老针"


def test_update_with_an_unknown_name_lists_what_exists(tools) -> None:
    _call(tools["register_tip"], material="W", name="老针")
    out = _call(tools["update_tip"], tip_name="不存在的针", note="x")
    assert out["success"] is False
    assert "老针" in out["known_tips"]


def test_update_does_not_clear_calibration(tools) -> None:
    """补记属性不是换针 —— 清标定只该发生在装入新针时。"""
    _call(tools["register_tip"], material="W")
    iprof.set_profile({"didv_at_contact_v": 2e-3})
    _call(tools["update_tip"], note="补记一句")
    assert iprof.get_profile().get("didv_at_contact_v") == 2e-3


def test_update_rejects_unknown_vocabulary_without_touching_the_row(tools) -> None:
    _call(tools["register_tip"], material="W")
    out = _call(tools["update_tip"], material="镝钪合金")
    assert out["success"] is False
    assert out["warnings"]


def test_remove_current_tip(tools) -> None:
    _call(tools["register_tip"], material="W")
    out = _call(tools["remove_current_tip"], note="换样品时取下")
    assert out["success"] is True and out["changed"] is True
    assert _call(tools["get_current_tip"])["registered"] is False


def test_remove_when_nothing_registered_is_honest(tools) -> None:
    out = _call(tools["remove_current_tip"])
    assert out["success"] is True and out["changed"] is False


# ── 降级 ────────────────────────────────────────────────────────────────────

def test_tools_degrade_when_storage_is_absent() -> None:
    """provider 里没有 storage 时如实说不可用,而不是抛给模型一个 traceback。"""
    tools = _tools(None)
    for name in TIP_TOOL_NAMES:
        out = _call(tools[name]) if name != "update_tip" else _call(
            tools[name], note="x")
        assert out["success"] is False
        assert "不可用" in out["error"]


def test_tools_survive_a_broken_storage() -> None:
    class Broken:
        def get_current_tip(self): raise RuntimeError("db gone")
        def list_tips(self, **k): raise RuntimeError("db gone")
        def create_tip(self, *a, **k): raise RuntimeError("db gone")
        def retire_current_tip(self, **k): raise RuntimeError("db gone")
        def get_active_scope(self): raise RuntimeError("db gone")

    tools = _tools(Broken())
    assert _call(tools["get_current_tip"])["success"] is False
    assert _call(tools["list_tips"])["success"] is False
    assert _call(tools["register_tip"], material="W")["success"] is False
