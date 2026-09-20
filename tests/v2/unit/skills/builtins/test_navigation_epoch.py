"""MoveToXY × 坐标代次(coord_epoch)—— explicit-only 的那道闸。

设计:``docs/v2/design/p0_fixes_design.md`` 修复项 第 3 条。

一次横向粗动之后,同样的 (x, y) 指的是另一片表面。调用方**说得出**这对坐标属于
哪一代时(取自地图标记 / 计划 / 存下来的站点),就该被拦;说不出的时候(修针
relocate 用的是动作瞬间从仪器读回的坐标,天然新鲜)一个字节都不该变。

所以这个参数是 explicit-only 的三态,三态各钉一条:
  * 不传 ⇒ 行为逐字节不变(连 Nanonis 调用序列都一样);
  * 传了且是当前代次 ⇒ 放行;
  * 传了且已过期 ⇒ 拒绝,**不换算**。

``tests/v2/unit/skills/builtins/test_navigation.py`` 测的是移动本身(发出即返回 +
自轮询),那份不动;代次是纯增量,放这里。
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core import coord_epoch
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.navigation import MoveToXY

TARGET_X = 1.3e-8
TARGET_Y = -2.9e-8


# ── 假记录层(与 composite 侧同形状)─────────────────────────────────────

class FakeStorage:
    def __init__(self, epoch: int):
        self.epoch = epoch
        self.epoch_queries: list[tuple] = []

    def current_epoch(self, experiment_id=None, sample_id=None) -> int:
        self.epoch_queries.append((experiment_id, sample_id))
        return self.epoch


class FakeLog:
    current_experiment_id = "exp-1"
    current_sample_id = "smp-1"

    def __init__(self, storage):
        self._storage = storage


@pytest.fixture
def epoch_now(monkeypatch):
    storage = FakeStorage(epoch=5)
    import mast.logging.experiment_log as el
    monkeypatch.setattr(el, "get_active_log", lambda: FakeLog(storage))
    return storage


@pytest.fixture
def epoch_unreadable(monkeypatch):
    import mast.logging.experiment_log as el
    monkeypatch.setattr(el, "get_active_log", lambda: None)


# ── 假 context ───────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        entry = self.canned.get(method)
        if entry is not None:
            return NanonisCallRecord(method=method, args=args,
                                     return_value=entry.get("return_value"),
                                     error=entry.get("error", ""))
        return NanonisCallRecord(method=method, args=args,
                                 error=f"unmocked: {method}")


def _ctx() -> FakeCtx:
    """针尖一问就已经在目标点 —— 轮询第一拍就返回,测试不等待。"""
    return FakeCtx(canned={
        "FolMe_XYPosSet": {"return_value": ("", b"", [])},
        "FolMe_XYPosGet": {"return_value": ("", b"", [TARGET_X, TARGET_Y])},
        "FolMe_SpeedGet": {"return_value": ("", b"", [5.0e-9, 0.0])},
    })


def _move(ctx, **extra):
    params = {"x_m": TARGET_X, "y_m": TARGET_Y, "wait": True, **extra}
    return MoveToXY().execute(ctx, params)


# ── 不传 ⇒ 逐字节不变 ───────────────────────────────────────────────────

def test_omitting_the_epoch_leaves_the_move_untouched(epoch_now):
    """省略是常态,不是疏忽。

    修针 relocate 用的是动作瞬间读回的坐标,天然属于当前代次 —— 一道「查不到
    就报警」的闸卡住它,就是拿一条新加的保护去砸一条本来好的路径。
    """
    ctx = _ctx()
    res = _move(ctx)
    assert res.success, res.error
    assert ctx.calls[0][0] == "FolMe_XYPosSet"
    assert ctx.calls[0][1] == (TARGET_X, TARGET_Y, 0)
    assert "refusal_code" not in (res.data or {})


def test_omitting_the_epoch_never_queries_the_generation(epoch_now):
    """不传就**根本不查** —— 查了就意味着某天会因为查不到而改行为。"""
    _move(_ctx())
    assert epoch_now.epoch_queries == []


def test_the_call_sequence_is_identical_with_and_without_a_matching_epoch(epoch_now):
    """传一个当前代次,Nanonis 调用序列必须与不传时一模一样。"""
    a, b = _ctx(), _ctx()
    _move(a)
    _move(b, coord_epoch=5)
    assert a.calls == b.calls


def test_a_none_epoch_is_the_same_as_omitting_it(epoch_now):
    """``ParameterSpec.default=None`` 会被 wrap_skill 物化进参数,所以「没传」
    到达 execute 时长得就是 ``coord_epoch=None`` —— 它必须等价于省略。"""
    a, b = _ctx(), _ctx()
    _move(a)
    _move(b, coord_epoch=None)
    assert a.calls == b.calls
    assert epoch_now.epoch_queries == []


# ── 传了 ⇒ 核对 ─────────────────────────────────────────────────────────

def test_a_current_epoch_is_allowed_through(epoch_now):
    ctx = _ctx()
    res = _move(ctx, coord_epoch=5)
    assert res.success, res.error
    assert epoch_now.epoch_queries, "传了代次却没查权威值"


def test_a_stale_epoch_refuses_the_move(epoch_now):
    """粗动之后,这对坐标指向的已经是另一片表面 —— 不许走过去。"""
    epoch_now.epoch = 6
    ctx = _ctx()
    res = _move(ctx, coord_epoch=5)
    assert not res.success
    assert res.data["refusal_code"] == "coord_epoch_stale"
    assert res.data["requested_coord_epoch"] == 5
    assert res.data["current_coord_epoch"] == 6
    assert res.data["moved"] is False
    assert ctx.calls == [], "拒绝了却还是把移动指令发了出去"
    assert "第 5 代" in res.error and "第 6 代" in res.error


def test_an_unreadable_epoch_does_not_block_the_move(epoch_unreadable):
    """「查不到」不是「陈旧」。

    折叠成陈旧 ⇒ 记录库一不可用,所有带代次的移动全被锁死,而且没有解法。
    折叠成当前 ⇒ 保护静默消失。这里选择放行 —— 与 ExecuteScanPlan 同一口径。
    """
    ctx = _ctx()
    res = _move(ctx, coord_epoch=5)
    assert res.success, res.error
    assert ctx.calls[0][0] == "FolMe_XYPosSet"


# ── 钉住被否掉的方案 ─────────────────────────────────────────────────────

def test_no_cross_epoch_coordinate_translation(epoch_now):
    """**拒绝里不许出现任何坐标** —— 换算过的固然不行,原样抄回来的也不行。

    出处是 ``io/coarse_map.py:20-27``(「WHY STEPS, NOT METRES」):粗动步进
    开环,步长随驱动幅度、负载、温度漂移(同样 100 步在 300 K 能比 4 K 远五倍),
    ``xy_motor_step_m`` 只是标注 ——「没有任何东西拿它计算,也没有任何标记会被
    它重投影」。想「聪明地换算一下」的人先撞见这条。
    """
    epoch_now.epoch = 6
    res = _move(_ctx(), coord_epoch=5)
    blob = json.dumps(res.data, ensure_ascii=False, default=str)
    for key in ("x_m", "y_m", "requested_x_m", "requested_y_m",
                "translated", "shifted", "offset"):
        assert key not in blob, f"拒绝里出现了坐标字段 {key}"
    text = f"{res.error} {blob}"
    assert not re.search(r"\d(?:\.\d+)?e-\d+", text), f"拒绝里出现了米量级数字: {text}"


def test_mutation_removing_the_epoch_gate_lets_a_stale_move_through(epoch_now,
                                                                    monkeypatch):
    """变异验证:先证明变异生效,再看被守卫的行为红。"""
    epoch_now.epoch = 6
    import mast.core.coord_epoch as ce
    monkeypatch.setattr(
        ce, "verify",
        lambda stamped, **k: ce.EpochVerdict(state=ce.MATCH, stamped=stamped,
                                             current=stamped))
    # ① 变异已应用
    assert ce.verify(5, current=6).stale is False
    # ② 陈旧坐标被放了过去
    ctx = _ctx()
    res = _move(ctx, coord_epoch=5)
    assert res.success and ctx.calls[0][0] == "FolMe_XYPosSet"


# ── 参数形状 ─────────────────────────────────────────────────────────────

def test_the_epoch_parameter_is_optional_and_defaults_to_none():
    """默认必须是 None。

    写一个具体数字(哪怕 0)就再也分不出「没传」和「显式传了它」—— wrap_skill
    会把 ``ParameterSpec.default`` 物化进 pydantic 字段。而 0 恰恰是一个合法
    代次(还没粗动过),那样每一次省略都会变成「声称自己是第 0 代」。
    """
    spec = {p.name: p for p in MoveToXY().metadata().parameters}["coord_epoch"]
    assert spec.required is False
    assert spec.default is None
    assert spec.type == "int"

    tool = wrap_skill(MoveToXY, lambda: FakeCtx())
    fields = tool.args_schema.model_fields
    assert "coord_epoch" in fields
    assert not fields["coord_epoch"].is_required()
    assert fields["coord_epoch"].default is None


def test_the_epoch_parameter_tells_the_model_when_to_omit_it():
    """explicit-only 的契约得写在模型看得见的地方,否则它会逢参数必填。"""
    spec = {p.name: p for p in MoveToXY().metadata().parameters}["coord_epoch"]
    desc = spec.description.lower()
    assert "omit" in desc
    assert "coarse move" in desc


def test_the_refusal_code_is_the_shared_one():
    """两个强制点报同一个码 —— 下游按码分支时不该有两套写法。"""
    assert coord_epoch.REFUSAL_CODE == "coord_epoch_stale"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
