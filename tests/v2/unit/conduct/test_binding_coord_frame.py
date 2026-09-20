"""绑定这一层的**坐标代次**核对。

背景与诊断更正:这条洞原来被描述成「`_resolve_params` 不问代次,重启前算的坐标
能喂进硬件技能」。查下来**instrument 选错了**:

* ``coord_epoch``  = 本作用域粗动过几次(COUNT ``coarse_move`` 标记)。横向粗动
  之后同一个 ``(x, y)`` 指的是**另一片表面** —— 那是一个坐标会变错的唯一原因。
  它由记录派生,**跨进程重启不变**。
* ``evidence_epoch`` = conduct 自己的账,只在**重启清算**与**进绕道**时 +1
  (全仓只有两处写它),**粗动一次都不 bump 它**。

所以拿 ``evidence_epoch`` 守绑定会两头错:真危险(粗动)漏过,假危险(重启/绕道)
误拦。这份测试同时钉住**两个方向**——只有 ``coord_epoch`` 变了才拦,
``evidence_epoch`` 变了不拦。

四态各走各的,照 ``mast.core.coord_epoch`` 的闭集,而且尊重那个模块 docstring 里的
告诫:**不要把 UNVERIFIABLE / UNSTAMPED 当成陈旧拒掉**,那会造出一道解不开的闸。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from _harness import FakeExecutor, build, outcome, spec, stage, step

from mast.core import coord_epoch as ce


@pytest.fixture
def epoch(monkeypatch):
    """当前坐标代次的可控替身。``set(None)`` = 查不到(UNVERIFIABLE)。"""

    box = {"value": 7}

    def fake_read():
        return box["value"]

    monkeypatch.setattr(ce, "read_current_epoch", fake_read)
    return box


def _spec_with_coord_binding():
    """上游产坐标,下游把它喂给一个硬件技能。"""
    return spec([stage("S", steps=(
        step("S.01_plan", produces=("positions_json", "coord_epoch")),
        step("S.02_run", bindings={"positions": "steps.S.01_plan.positions_json"}),
    ))])


def _rig(tmp_path, plan_data, **kw):
    return build(tmp_path, _spec_with_coord_binding(),
                 executor=FakeExecutor({"ScanAt": [outcome(True, data=plan_data),
                                                   outcome(True, data={})]}),
                 **kw)


def _run_two_steps(rig):
    rig.tick()          # adopt
    rig.tick()          # S.01_plan
    rig.tick()          # S.02_run(或在这里被绑定拦下)
    return rig


def _params_of(rig, step_id: str) -> dict:
    for ev in rig.events(kind="step_started"):
        if ev["step_id"] == step_id:
            return dict((ev["payload"] or {}).get("params") or {})
    return {}


def _frames_of(rig, step_id: str) -> list:
    for ev in rig.events(kind="step_started"):
        if ev["step_id"] == step_id:
            return list((ev["payload"] or {}).get("coord_frames") or [])
    return []


def _last_fail(rig) -> str:
    for ev in reversed(rig.events(kind="step_finished")):
        p = ev["payload"] or {}
        if p.get("ok") is False:
            return str(p.get("why") or "")
    return ""


# ── MATCH:章对得上 ─────────────────────────────────────────────────────

def test_a_stamp_that_matches_passes_the_value_through(tmp_path, epoch):
    rig = _rig(tmp_path, {"positions_json": "[[1,2]]", "coord_epoch": 7})
    _run_two_steps(rig)
    assert _params_of(rig, "S.02_run")["positions"] == "[[1,2]]"
    frames = _frames_of(rig, "S.02_run")
    assert frames and frames[0]["state"] == ce.MATCH
    assert not any(f.get("unprotected") for f in frames)


# ── STALE:粗动过了 ⇒ 拒绝 ─────────────────────────────────────────────

def test_a_coarse_move_since_the_plan_refuses_the_binding(tmp_path, epoch):
    """**这是这条洞真正的形状。** 计划在第 7 代排的,现在是第 8 代 ——
    同一个 (x, y) 已经指向另一片表面。"""
    rig = _rig(tmp_path, {"positions_json": "[[1,2]]", "coord_epoch": 7})
    rig.tick()
    rig.tick()                       # S.01_plan 产出并盖第 7 代章
    epoch["value"] = 8               # 有人粗动了一次
    rig.tick()                       # S.02_run 该被拦下
    assert _params_of(rig, "S.02_run") == {}, "拦下来的步不该有 step_started"
    why = _last_fail(rig)
    assert "拒绝" in why and "第 7 代" in why and "第 8 代" in why


def test_the_refusal_offers_replanning_and_never_a_conversion(tmp_path, epoch):
    """陈旧坐标的处置只有一个:**拒绝,让人重新规划**。既不夹紧也不换算 ——
    换算出来的坐标看上去和真坐标一模一样,而它是编的。"""
    rig = _rig(tmp_path, {"positions_json": "[[1,2]]", "coord_epoch": 7})
    rig.tick()
    rig.tick()
    epoch["value"] = 8
    rig.tick()
    why = _last_fail(rig)
    assert "重新规划" in why
    assert "换算" in why and "不做跨代次换算" in why


# ── 两个代次不是一回事(两个方向都钉)────────────────────────────────

def test_a_restart_alone_does_not_refuse_a_binding(tmp_path, epoch):
    """**假危险不许拦。** 重启 bump 的是 ``evidence_epoch``;没人粗动 ⇒ 坐标好好的。

    拿 ``evidence_epoch`` 当守卫的那一版会在这里把一份能续的 conduct 拦下来。
    """
    rig = _rig(tmp_path, {"positions_json": "[[1,2]]", "coord_epoch": 7})
    rig.tick()
    rig.tick()
    before = int(rig.row()["evidence_epoch"])
    rig.store.record(rig.conduct_id, "status_change",
                     changes={"evidence_epoch": before + 1})
    rig.tick()
    assert _params_of(rig, "S.02_run")["positions"] == "[[1,2]]", (
        "证据代次变了就拦 —— 那是拿另一件事的尺子在量坐标")
    assert _last_fail(rig) == ""


def test_a_coarse_move_is_caught_even_though_evidence_epoch_never_moved(
        tmp_path, epoch):
    """**真危险不许漏。** 粗动一次都不 bump ``evidence_epoch``(全仓只有重启清算
    与进绕道两处写它)—— 所以只看 ``evidence_epoch`` 的守卫在这里什么都不会做。"""
    rig = _rig(tmp_path, {"positions_json": "[[1,2]]", "coord_epoch": 7})
    rig.tick()
    rig.tick()
    before = int(rig.row()["evidence_epoch"])
    epoch["value"] = 8
    rig.tick()
    assert int(rig.row()["evidence_epoch"]) == before, "前提没成立"
    assert "拒绝" in _last_fail(rig)


# ── UNSTAMPED / UNVERIFIABLE:放行,但**留痕** ─────────────────────────

def test_an_unstamped_producer_passes_but_says_it_was_not_checked(tmp_path, epoch):
    """产出方从没声称过一个坐标系 ⇒ 没有东西可核对。

    **这是「没检查」,不是「检查通过」** —— 所以它必须在 ``not_checked`` 和
    ``step_started.coord_frames`` 两处都看得见。
    """
    rig = _rig(tmp_path, {"positions_json": "[[1,2]]"})   # 不盖章
    rig.tick()
    rig.tick()
    rep = rig.tick()
    assert _params_of(rig, "S.02_run")["positions"] == "[[1,2]]"
    frames = _frames_of(rig, "S.02_run")
    assert frames and frames[0]["state"] == ce.UNSTAMPED
    assert frames[0]["unprotected"] is True
    assert any("没有代次保护" in n for n in rep.not_checked), rep.not_checked


def test_an_unreadable_current_epoch_does_not_become_an_unopenable_gate(
        tmp_path, epoch):
    """记录存储读不到 ⇒ **放行 + 留痕**,不是拒绝。

    ``mast.core.coord_epoch`` 的 docstring 逐字告诫过:不要把 UNVERIFIABLE 当成
    陈旧拒掉,那会让「读不到记录」变成一道解不开的闸。而真正把针开过去的那一层
    (``SpectroscopyAtPositions`` 只放行确凿的 MATCH)比这里严 —— 严格该待在硬件
    边界上,不该在这里再摞一层用户解不开的。
    """
    rig = _rig(tmp_path, {"positions_json": "[[1,2]]", "coord_epoch": 7})
    rig.tick()
    rig.tick()
    epoch["value"] = None            # 存储读不到
    rep = rig.tick()
    assert _params_of(rig, "S.02_run")["positions"] == "[[1,2]]"
    frames = _frames_of(rig, "S.02_run")
    assert frames and frames[0]["state"] == ce.UNVERIFIABLE
    assert frames[0]["unprotected"] is True
    assert any("没有代次保护" in n for n in rep.not_checked)


# ── 只有 steps.* 走这条检查 ────────────────────────────────────────────

def test_a_params_binding_is_not_touched_by_this_check(tmp_path, epoch):
    """``params.*`` 是 approve 时冻结的,没有产出方、没有章可查。

    它们**另有一个问题**(用户手填的坐标属于哪一代,今天由手填的
    ``params.coord_epoch`` 声明),而那不是这条检查能答的 —— 不假装答它。
    """
    from mast.conduct.spec import ParamSpec

    s = spec([stage("S", steps=(
        step("S.01", bindings={"center_x_m": "params.x_m"}),))],
        params=(ParamSpec(name="x_m", type="float"),))
    rig = build(tmp_path, s, params={"x_m": 1e-9},
                executor=FakeExecutor())
    rig.tick()
    rig.tick()
    assert _params_of(rig, "S.01")["center_x_m"] == pytest.approx(1e-9)
    assert _frames_of(rig, "S.01") == []


# ── 生产侧:两个并排的规划器,同一条要求 ───────────────────────────────

def test_both_planners_stamp_the_coordinate_generation(monkeypatch):
    """``plan_sts_points`` 原来不盖章,而它的兄弟 ``plan_bias_series`` 盖 ——
    同一条 修复项 生产侧要求,只有一个做了。

    章必须**实时查得**,不能靠用户在 ``params.coord_epoch`` 手填一个数:
    手填的数与这批点位的真实出身之间没有任何东西对得上账。
    """
    from mast.conduct import analyses

    monkeypatch.setattr(ce, "read_current_epoch", lambda: 5)
    out = analyses.get("plan_sts_points")(
        {"positions_m": "1e-8,2e-8; -3e-8,0", "n_points": 5})
    assert out["coord_epoch"] == 5


def test_a_planner_that_cannot_read_the_generation_stamps_nothing(monkeypatch):
    """查不到代次就**不盖**,不盖 0 —— 0 是「这个作用域还没粗动过」这个真实答案。"""
    from mast.conduct import analyses

    monkeypatch.setattr(ce, "read_current_epoch", lambda: None)
    out = analyses.get("plan_sts_points")(
        {"positions_m": "1e-8,2e-8", "n_points": 5})
    assert "coord_epoch" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
