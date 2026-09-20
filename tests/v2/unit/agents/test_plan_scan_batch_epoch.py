"""``plan_scan_batch`` 盖坐标代次的章 —— 生产侧,以及它与消费侧的对账。

设计:``docs/v2/design/p0_fixes_design.md`` 修复项 第 1 条。

计划里的每一帧都是一对米坐标,而米坐标只在一个代次内有意义。盖章这件事本身很小,
容易错的是另外三处:

1. **代次要查,不要数行。** ``storage.current_epoch()`` 是对整个作用域 COUNT;
   从 ``get_markers()`` 的截断窗口里数 coarse_move 会漏报(窗口只有最新 limit 行),
   于是陈旧坐标被判成当前 —— coarse_motion 已经踩过一次。
2. **查不到就不盖章,不盖 0。** 0 是「这个作用域还没粗动过」这个真实答案。
3. **章要能被消费侧认出来。** 生产侧写 ``coord_epoch``、消费侧读别的名字,
   两边各自都「实现了」,合起来是零 —— 所以这里有一条端到端的往返测试。
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
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

import pytest

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.composite.execute_scan_plan import ExecuteScanPlan

_PLAN_KW = {"kind": "repeat", "n_images": 2, "size_nm": 20,
            "publish_to_map": False}


class FakeStorage:
    def __init__(self, epoch: int):
        self.epoch = epoch
        self.epoch_queries: list[tuple] = []

    def current_epoch(self, experiment_id=None, sample_id=None) -> int:
        self.epoch_queries.append((experiment_id, sample_id))
        return self.epoch

    def get_markers(self, *a, **k):  # pragma: no cover - 被调用即失败
        raise AssertionError(
            "代次不许从 get_markers() 的截断窗口里数 —— 用 current_epoch()")


class FakeLog:
    current_experiment_id = "exp-1"
    current_sample_id = "smp-1"

    def __init__(self, storage):
        self._storage = storage


@pytest.fixture
def epoch_now(monkeypatch):
    storage = FakeStorage(epoch=4)
    import mast.logging.experiment_log as el
    monkeypatch.setattr(el, "get_active_log", lambda: FakeLog(storage))
    return storage


def _tool():
    from mast.agents._shared.meta_tools import make_meta_tools
    return {t.name: t for t in make_meta_tools(lambda: {})}["plan_scan_batch"]


def _plan(**over) -> dict:
    return json.loads(_tool().invoke({**_PLAN_KW, **over}))


# ── 盖章 ─────────────────────────────────────────────────────────────────

def test_a_plan_is_stamped_with_the_current_generation(epoch_now):
    out = _plan()
    assert out["success"] is True
    assert out["plan"]["coord_epoch"] == 4
    assert epoch_now.epoch_queries, "盖章却没查权威代次"
    assert epoch_now.epoch_queries[0] == ("exp-1", "smp-1")


def test_the_stamp_follows_the_authoritative_query(epoch_now):
    """章的值就是 ``current_epoch()`` 回的那个数,不是别处推出来的。"""
    assert _plan()["plan"]["coord_epoch"] == 4
    epoch_now.epoch = 11
    assert _plan()["plan"]["coord_epoch"] == 11


def test_the_stamp_is_at_the_top_level_of_plan_json(epoch_now):
    """消费侧读的是 ``plan_json``,不是 ``plan`` —— 章必须在它里面。"""
    out = _plan()
    assert json.loads(out["plan_json"])["coord_epoch"] == 4


def test_generation_zero_is_stamped_like_any_other(epoch_now):
    """0 = 「还没粗动过」,是一个真实代次,不是「没有代次」。"""
    epoch_now.epoch = 0
    out = _plan()
    assert out["plan"]["coord_epoch"] == 0
    assert "coord_epoch_unavailable" not in out


# ── 查不到 ⇒ 不盖章(而不是盖 0)────────────────────────────────────────

def test_an_unreadable_generation_leaves_the_plan_unstamped(monkeypatch):
    import mast.logging.experiment_log as el
    monkeypatch.setattr(el, "get_active_log", lambda: None)
    out = _plan()
    assert out["success"] is True, "查不到代次不该让规划失败"
    assert "coord_epoch" not in out["plan"], "把「查不到」盖成了一个具体代次"
    assert "coord_epoch" not in json.loads(out["plan_json"])
    assert out["coord_epoch_unavailable"] is True
    assert "没盖坐标代次的章" in out["message"]


# ── 生产侧 ↔ 消费侧对账(键名漂了就是零)────────────────────────────────

class _Ctx:
    def __init__(self):
        self.runs: list[tuple[str, dict]] = []

    def run(self, skill_name, params, version=None):
        self.runs.append((skill_name, dict(params)))
        return SkillResult(skill_name=skill_name, success=True,
                           data={"saved_path": "/scans/x.sxm"})

    def safe_call(self, method, *args, **kwargs):
        return NanonisCallRecord(method=method, args=args)

    def check_abort(self):
        return False


def test_a_stamped_plan_is_refused_after_a_coarse_move(epoch_now):
    """端到端:这里排的计划,粗动之后被 ExecuteScanPlan 整批拒。

    两侧各自的单测都绿、合起来是零 —— 这条测的正是那个缝:键名、位置、取值来源
    三者必须真的对得上。
    """
    plan_json = _plan()["plan_json"]
    epoch_now.epoch += 1          # 一次横向粗动
    ctx = _Ctx()
    res = ExecuteScanPlan().execute(ctx, {"plan_json": plan_json})
    assert not res.success
    assert res.data["refusal_code"] == "coord_epoch_stale"
    assert res.data["plan_coord_epoch"] == 4
    assert res.data["current_coord_epoch"] == 5
    assert [n for n, _ in ctx.runs] == [], "拒绝了却还是扫了图"


def test_the_same_plan_runs_when_nothing_moved(epoch_now):
    plan_json = _plan()["plan_json"]
    ctx = _Ctx()
    res = ExecuteScanPlan().execute(ctx, {"plan_json": plan_json})
    assert res.success, res.error
    assert [n for n, _ in ctx.runs].count("ScanAt") == 2


def test_mutation_dropping_the_stamp_disarms_the_consumer(epoch_now, monkeypatch):
    """变异验证:生产侧不盖章 ⇒ 消费侧那道闸整条失效(计划照跑)。

    先证明变异生效(计划里真的没有章),再看被守卫的行为红。
    """
    import mast.agents._shared.meta_tools as mt
    monkeypatch.setattr(mt, "read_current_epoch", lambda: None)
    # ① 变异已应用
    out = _plan()
    assert "coord_epoch" not in out["plan"]
    # ② 粗动之后依然照跑,一句「无章」的告警是仅剩的痕迹
    epoch_now.epoch += 1
    ctx = _Ctx()
    res = ExecuteScanPlan().execute(ctx, {"plan_json": out["plan_json"]})
    assert res.success
    assert [n for n, _ in ctx.runs].count("ScanAt") == 2
    assert any("coord_epoch" in w for w in res.data.get("warnings", []))


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
