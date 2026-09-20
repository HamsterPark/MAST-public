"""可执行冒烟模板用于核验整条工作流的结构与线程生命周期。
线程测试覆盖启动、停止和不自动启动；调度语义由手动 step_tick 的测试验证。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from _harness import FakeReading, FakeTemperature, build

from mast.conduct import analyses as A
from mast.conduct import validator as V
from mast.conduct.templates import TEMPLATE_MODULES, get_template
from mast.core.registry import SkillRegistry

SPEC_ID = "_smoke_v1"

PARAMS = {
    "target_temperature_k": 300.0,
    "temperature_stale_after_s": 120.0,
    "probe_positions_m": "1e-8,2e-8; -3e-8,0",
    "probe_points_max": 4,
}


@pytest.fixture(scope="module")
def registry():
    r = SkillRegistry()
    r.discover("mast.skills.builtins", "mast.skills.composite")
    return r


@pytest.fixture
def spec_():
    return get_template(SPEC_ID)


# ── 它今天就能 approve ──────────────────────────────────────────────────

def test_the_smoke_template_is_approvable_today(spec_, registry):
    """只用注册表里已有的技能与已注册的分析函数 —— 与 ``synthetic_sample_v1`` 恰好相反。"""
    rep = V.validate_spec(spec_, skills=V.skill_index(registry),
                          analyses=A.known_names())
    assert rep.approvable is True, rep.describe()


def test_it_carries_no_judgement_logic():
    assert V.lint_template_module(TEMPLATE_MODULES[SPEC_ID]) == []


def test_every_skill_it_names_exists(spec_, registry):
    have = set(V.skill_index(registry))
    used = {s.skill for st in spec_.stages for s in st.all_steps
            if s.touches_hardware}
    assert used <= have


def test_it_retracts_before_it_waits(spec_):
    """等待可能是几小时。把针留在隧道结上等,比做一次确认式退针危险得多 ——
    校验器规则①不区分等的是人还是温度。"""
    steps = [s.step_id for s in spec_.stages[0].all_steps]
    retract = next(i for i, s in enumerate(spec_.stages[0].all_steps)
                   if s.skill == "SafeRetract")
    wait = next(i for i, s in enumerate(spec_.stages[0].all_steps)
                if s.kind == "wait")
    assert retract < wait, steps


def test_its_parameters_are_all_operator_supplied(spec_):
    assert spec_.params_schema
    assert all(p.default is None for p in spec_.params_schema)
    assert V.check_params(spec_, PARAMS) == []


# ── 端到端 ───────────────────────────────────────────────────────────────

def _rig(tmp_path, *, value_k=4.0, age_s=1.0):
    return build(tmp_path, get_template(SPEC_ID), params=PARAMS,
                 temperature=FakeTemperature(FakeReading(value_k=value_k,
                                                         age_s=age_s)))


def test_the_whole_chain_runs_from_approve_to_completed(tmp_path):
    rig = _rig(tmp_path)
    for _ in range(12):
        rig.tick()
        if rig.row()["status"] == "waiting_condition":
            break
    assert rig.row()["status"] == "waiting_condition", rig.row()["status"]
    # 退针发生了,点位算出来了,现在停在等温度上
    assert "SafeRetract" in rig.executor.skills_called()
    flat = rig.director._produced_flat(rig.conduct_id)
    assert flat["steps.SMOKE.02_points.n_points"] == 2
    # 温度已达标,hold_s=0 ⇒ 下一 tick 放行,然后跑完
    for _ in range(8):
        rig.tick()
        if rig.row()["status"] == "completed":
            break
    row = rig.row()
    assert row["status"] == "completed"
    assert row["active_slot"] is None
    assert rig.executor.skills_called().count("GetTemperature") == 2


def test_a_dead_temperature_link_downgrades_the_wait_instead_of_hanging(tmp_path):
    """温度输入长期不可用时应在 stale 超时后要求人工检查，而不是持续静默等待。"""
    rig = _rig(tmp_path, value_k=None, age_s=None)
    for _ in range(12):
        rig.tick()
        if rig.row()["status"] == "waiting_operator":
            break
    assert rig.row()["status"] == "waiting_operator"
    assert "读不到不等于没到" in rig.row()["status_reason"]


def test_bad_positions_stop_the_chain_at_the_analysis_step(tmp_path):
    """参数错就停在分析步 —— 而且说清是哪一步、为什么。"""
    bad = dict(PARAMS, probe_positions_m="5,0")       # 五米外
    rig = build(tmp_path, get_template(SPEC_ID), params=bad)
    for _ in range(8):
        rig.tick()
        if rig.row()["status"] == "waiting_operator":
            break
    assert rig.row()["status"] == "waiting_operator"
    assert "不夹紧" in rig.row()["status_reason"]


# ── 线程 ─────────────────────────────────────────────────────────────────

def test_the_director_does_not_start_itself(tmp_path):
    """**不自动启动**:构造出来只是一个对象。runtime 挂点与设置键是 M1-c 的事,
    在那之前谁也不该因为 import 了一个模块就多出一条驱动仪器的线程。"""
    rig = _rig(tmp_path)
    assert rig.director.is_running is False


def test_start_and_stop_are_symmetric(tmp_path):
    rig = _rig(tmp_path)
    rig.director.start()
    assert rig.director.is_running is True
    rig.director.start()                     # 幂等
    rig.director.stop(timeout_s=5.0)
    assert rig.director.is_running is False


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
