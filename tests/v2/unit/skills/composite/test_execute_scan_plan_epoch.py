"""ExecuteScanPlan × 坐标代次(coord_epoch)—— 强制点,以及重扫的偏压。

设计:``docs/v2/design/p0_fixes_design.md`` 修复项 + 补录第一条。

一份计划里的每一帧都是一对**米坐标**,而米坐标只在一个代次内有意义:一次横向
粗动之后,同样的 (x, y) 指的是另一片表面。机制本来就在(``map_markers`` 写时盖
章、``storage.current_epoch()`` 权威查询),缺的是**有人拦** —— 在此之前,一份
粗动之前排的计划在粗动之后照样能原样执行,每一帧都扫在错的地方,而且扫得很成功。

这里钉四件事:

1. 陈旧代次的计划**整批拒绝**,一帧都不扫;中途粗动 ⇒ 剩余帧全拒同因。
2. 拒绝**不换算**坐标(``test_no_cross_epoch_coordinate_translation``)。
3. 「查不到代次」和「计划没盖章」都**不是**陈旧 —— 照常执行 + 说出来。
   把「读不到」折叠成「陈旧」,就是把一道保护变成一道解不开的闸。
4. 重扫**重设偏压**(补录第一条):原来偏压只在第一次尝试时设。

每条守卫都配一条变异验证:**先证明变异真的生效,再看被守卫的行为红**。
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

import pytest

from mast.core import coord_epoch
from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.composite import execute_scan_plan as esp
from mast.skills.composite.execute_scan_plan import ExecuteScanPlan

#: 计划里用的坐标步距。取一个不会和代次号、帧数、尺寸撞上的值,好让
#: ``test_no_cross_epoch_coordinate_translation`` 能按数字认坐标。
STEP_M = 1.3e-7
SIZE_M = 4.7e-8


# ── 假的记录层:代次可读、可变、可失联 ───────────────────────────────────

class FakeStorage:
    """代次来自 current_epoch 权威查询，不从历史 markers 推断；替身仅提供所需查询以检测错误路径。"""

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
    """把权威代次接到一个可读可改的假记录层上。返回 FakeStorage。

    走的是真正的那条缝(``get_active_log()._storage.current_epoch``),不是把
    ``read_current_epoch`` 整个替换掉 —— 后者会把「代次到底从哪儿查的」这个
    问题一起测没了。
    """
    storage = FakeStorage(epoch=7)
    import mast.logging.experiment_log as el
    monkeypatch.setattr(el, "get_active_log", lambda: FakeLog(storage))
    return storage


@pytest.fixture
def epoch_unreadable(monkeypatch):
    """没有活动实验 / 记录存储不可用 —— 代次**查不到**(不是 0)。"""
    import mast.logging.experiment_log as el
    monkeypatch.setattr(el, "get_active_log", lambda: None)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("mast.skills.composite.execute_scan_plan.time.sleep",
                        lambda *_a, **_k: None)


# ── 假 context(与 test_execute_scan_plan.py 同形状,另加中途粗动)────────

class PlanCtx:
    def __init__(self, scan_outcomes=None, *, storage=None, coarse_move_after=None):
        self.runs: list[tuple[str, dict]] = []
        self._outcomes = list(scan_outcomes or [])
        self._n = 0
        self._storage = storage
        #: 扫完第 n 次 ScanAt 之后,悄悄发生一次横向粗动(代次 +1)。
        self._coarse_after = coarse_move_after

    def run(self, skill_name, params, version=None):
        self.runs.append((skill_name, dict(params)))
        if skill_name != "ScanAt":
            return SkillResult(skill_name=skill_name, success=True, data={})
        outcome = (self._outcomes[self._n] if self._n < len(self._outcomes)
                   else "ok")
        self._n += 1
        if self._coarse_after is not None and self._n == self._coarse_after:
            self._storage.epoch += 1
        if outcome == "bad":
            return SkillResult(skill_name=skill_name, success=False,
                               error="质量不合格", data={})
        return SkillResult(skill_name=skill_name, success=True,
                           data={"saved_path": f"/scans/{self._n}.sxm"})

    def safe_call(self, method, *args, **kwargs):
        return NanonisCallRecord(method=method, args=args)

    def check_abort(self):
        return False


def _plan(n=3, *, stamp: "int | None" = 7, **frame_overrides) -> str:
    plan: dict = {
        "kind": "survey", "n_frames": n,
        "frames": [
            {"index": i, "center_x_m": i * STEP_M, "center_y_m": -i * STEP_M,
             "size_m": SIZE_M, "label": f"f{i}", **frame_overrides}
            for i in range(n)
        ],
    }
    if stamp is not None:
        plan["coord_epoch"] = stamp
    return json.dumps(plan)


def _run(ctx, plan_json, **params):
    params.setdefault("plan_json", plan_json)
    return ExecuteScanPlan().execute(ctx, params)


def _scans(ctx):
    return [p for name, p in ctx.runs if name == "ScanAt"]


def _biases(ctx):
    return [p for name, p in ctx.runs if name == "BiasSettleChange"]


# ── 陈旧代次:整批拒绝 ───────────────────────────────────────────────────

def test_a_plan_from_an_older_generation_is_refused_outright(epoch_now):
    """粗动之后,旧计划的每一帧都指向另一片表面 —— 一帧都不许扫。"""
    epoch_now.epoch = 9
    ctx = PlanCtx(storage=epoch_now)
    res = _run(ctx, _plan(3, stamp=7))
    assert not res.success
    assert res.data["refusal_code"] == coord_epoch.REFUSAL_CODE == "coord_epoch_stale"
    assert res.data["plan_coord_epoch"] == 7
    assert res.data["current_coord_epoch"] == 9
    assert res.data["frames_refused"] == 3
    assert _scans(ctx) == [], "陈旧计划却扫了图"
    assert res.data["done"] == 0 and res.data["scanned_paths"] == []


def test_the_refusal_names_both_generations_and_asks_for_a_replan(epoch_now):
    """拒绝要能被行动:说清楚哪一代 vs 哪一代,以及下一步是重排而不是重试。"""
    epoch_now.epoch = 9
    res = _run(PlanCtx(storage=epoch_now), _plan(2, stamp=7))
    assert "第 7 代" in res.error and "第 9 代" in res.error
    assert "重新规划" in res.error
    assert "粗动" in res.error


def test_a_plan_from_the_current_generation_runs(epoch_now):
    ctx = PlanCtx(storage=epoch_now)
    res = _run(ctx, _plan(3, stamp=7))
    assert res.success, res.error
    assert res.data["done"] == 3
    assert len(_scans(ctx)) == 3
    assert "refusal_code" not in res.data
    assert "warnings" not in res.data, f"同代次不该有告警: {res.data.get('warnings')}"
    # 这批图属于哪一代要留在结果里:.sxm 自己不带代次
    assert res.data["coord_epoch"] == 7


def test_a_coarse_move_mid_batch_refuses_the_remaining_frames(epoch_now):
    """批次中途本不该粗动,但绕行分支可能做 —— 做了之后剩下每一帧都是错地方。"""
    ctx = PlanCtx(storage=epoch_now, coarse_move_after=1)
    res = _run(ctx, _plan(4, stamp=7))
    assert not res.success
    assert res.data["aborted_at"] == 1, "中途粗动没在下一帧之前拦住"
    assert len(_scans(ctx)) == 1
    assert res.data["refusal_code"] == "coord_epoch_stale"
    assert res.data["current_coord_epoch"] == 8
    assert res.data["frames_refused"] == 3
    assert res.data["done"] == 1, "已经扫成的那一帧仍然算数"


def test_the_generation_is_checked_before_every_frame(epoch_now):
    """开跑一次 + 每帧一次。查询是 COUNT,廉价 —— 省下来的那几次正是漏网的入口。"""
    ctx = PlanCtx(storage=epoch_now)
    _run(ctx, _plan(3, stamp=7))
    assert len(epoch_now.epoch_queries) >= 4, epoch_now.epoch_queries
    # 而且是按作用域查的,不是全库
    assert epoch_now.epoch_queries[0] == ("exp-1", "smp-1")


# ── 三态:没盖章 / 查不到,都不是陈旧 ─────────────────────────────────────

def test_a_plan_without_a_stamp_still_runs_but_says_so(epoch_now):
    """旧格式计划照常执行 —— 但不许假装它受过保护。"""
    ctx = PlanCtx(storage=epoch_now)
    res = _run(ctx, _plan(2, stamp=None))
    assert res.success, res.error
    assert len(_scans(ctx)) == 2
    warn = " ".join(res.data.get("warnings") or [])
    assert "coord_epoch" in warn and "没有" in warn
    assert "refusal_code" not in res.data


def test_an_unreadable_generation_is_not_stale(epoch_unreadable):
    """「查不到」既不是陈旧也不是当前。

    折叠成陈旧 ⇒ 记录库一不可用,任何盖过章的计划都跑不了,而且**没有解法**
    (重排也要同一个记录库)—— 那是「能停不能解」。折叠成当前 ⇒ 保护静默消失。
    所以:跑,并且把「这一次没有代次保护」说出来。
    """
    ctx = PlanCtx()
    res = _run(ctx, _plan(2, stamp=7))
    assert res.success, res.error
    assert len(_scans(ctx)) == 2
    assert "refusal_code" not in res.data
    warn = " ".join(res.data.get("warnings") or [])
    assert "查不到" in warn
    assert coord_epoch.verify(7).state == coord_epoch.UNVERIFIABLE


def test_the_four_states_never_collapse():
    """闭集的四个值互不相等,而且只有一个是「拒绝」。"""
    assert len(set(coord_epoch.STATES)) == 4
    assert coord_epoch.REFUSAL_CODE == coord_epoch.STALE
    assert coord_epoch.verify(3, current=3).stale is False
    assert coord_epoch.verify(3, current=4).stale is True
    assert coord_epoch.verify(None, current=4).stale is False
    assert coord_epoch.verify(3, current=None).stale is False
    # 0 是一个真实答案(还没粗动过),不是「读不到」
    assert coord_epoch.verify(0, current=0).state == coord_epoch.MATCH
    assert coord_epoch.verify(0, current=1).stale is True


def test_unreadable_reads_as_none_not_zero(monkeypatch):
    """``read_current_epoch`` 读不到时返回 None —— 返回 0 会把「不知道」说成
    「还没粗动过」,而后者是一个具体答案。"""
    import mast.logging.experiment_log as el
    monkeypatch.setattr(el, "get_active_log", lambda: None)
    assert coord_epoch.read_current_epoch() is None

    class Broken:
        def current_epoch(self, *a, **k):
            raise RuntimeError("db locked")

    monkeypatch.setattr(el, "get_active_log", lambda: FakeLog(Broken()))
    assert coord_epoch.read_current_epoch() is None


# ── 钉住被否掉的方案 ─────────────────────────────────────────────────────

def test_no_cross_epoch_coordinate_translation(epoch_now):
    """**跨代次坐标换算在本仓有意不存在,拒绝里也不许偷偷出现一个。**

    ``io/coarse_map.py:20-27``(「WHY STEPS, NOT METRES」)是这条的出处:粗动
    步进是开环的,步长随驱动幅度、负载、温度漂移 —— 同样 100 步在 300 K 能比
    4 K 远五倍。``xy_motor_step_m`` 只是一个可选标定,「没有任何东西拿它计算,
    也没有任何标记会被它重投影」。所以下一个想「聪明地把旧坐标换算到新代次」的人
    会先撞见这条测试。

    判据是结构性的:陈旧拒绝的整个回包里**一个坐标都不许有** —— 既没有换算过的,
    也没有原样抄回来的(抄回来的那个下一步就会被当成可用坐标)。
    """
    epoch_now.epoch = 9
    ctx = PlanCtx(storage=epoch_now)
    res = _run(ctx, _plan(3, stamp=7))
    assert not res.success

    blob = json.dumps(res.data, ensure_ascii=False, default=str)
    for key in ("center_x_m", "center_y_m", "x_m", "y_m", "size_m",
                "translated", "shifted", "offset"):
        assert key not in blob, f"拒绝里出现了坐标字段 {key}"

    # 连数字也不许:米量级的数一旦出现在拒绝里,就已经是「这里有个可用坐标」
    # 的暗示 —— 不管它是换算的还是抄的。
    text = f"{res.error} {res.summary} {blob}"
    assert not re.search(r"\d(?:\.\d+)?e-\d+", text), f"拒绝里出现了米量级数字: {text}"
    assert _scans(ctx) == []


def test_mutation_removing_the_epoch_gate_lets_a_stale_plan_run(epoch_now,
                                                               monkeypatch):
    """变异验证:把代次核对拆掉,上面那几条守卫必须变红。

    先证明变异**真的应用上了**(核对不再报陈旧),再看被守卫的行为:陈旧计划
    整整三帧全扫了。没有第一步,这条测试证明不了它自己在测什么。
    """
    epoch_now.epoch = 9
    monkeypatch.setattr(
        coord_epoch, "verify",
        lambda stamped, **k: coord_epoch.EpochVerdict(
            state=coord_epoch.MATCH, stamped=stamped, current=stamped))
    # ① 变异已应用
    assert coord_epoch.verify(7, current=9).stale is False
    # ② 被守卫的行为红了
    ctx = PlanCtx(storage=epoch_now)
    res = _run(ctx, _plan(3, stamp=7))
    assert res.success and len(_scans(ctx)) == 3
    assert "refusal_code" not in res.data


# ── 补录:重扫要重设偏压 ─────────────────────────────────────────────────

def test_a_rescan_sets_the_bias_again(epoch_now):
    """两次尝试 = 两次设偏压。

    原来偏压只在 ``attempts == 0`` 时设,而 ``bias_v`` 设完就从下发参数里 pop
    掉了 ⇒ 重扫既不过安全通道也不带 bias ⇒ **用上一次留下的偏压重扫**。两次
    尝试之间会插事情(针尖复核、脉冲),偏压系列里这等于把某一帧悄悄换成另一个
    偏压的图 —— 数据看着完好,标签是错的。
    """
    ctx = PlanCtx(scan_outcomes=["bad", "ok"], storage=epoch_now)
    res = _run(ctx, _plan(1, stamp=7, bias_v=-1.5))
    assert res.success, res.error
    assert len(_scans(ctx)) == 2, "没重扫,这条测的就不是重扫"
    assert len(_biases(ctx)) == 2, "重扫那一次没重设偏压"
    assert all(c["bias_v"] == -1.5 for c in _biases(ctx))


def test_the_rescan_bias_goes_through_the_safe_channel_not_scan_at(epoch_now):
    """重扫也走 BiasSettleChange —— 穿零直接设会把针尖推向表面。"""
    ctx = PlanCtx(scan_outcomes=["bad", "ok"], storage=epoch_now)
    _run(ctx, _plan(1, stamp=7, bias_v=-1.5))
    assert [n for n, _ in ctx.runs][:2] == ["BiasSettleChange", "ScanAt"]
    assert all("bias_v" not in c for c in _scans(ctx)), "偏压绕过了安全通道"


def test_a_frame_without_a_bias_never_sets_one(epoch_now):
    """None 是「保持现值」。重扫也不许把它变成一次写。"""
    ctx = PlanCtx(scan_outcomes=["bad", "bad"], storage=epoch_now)
    _run(ctx, _plan(1, stamp=7))
    assert _biases(ctx) == []


def test_the_bias_rule_does_not_look_at_the_attempt_number():
    assert esp.needs_bias_change({"bias_v": -1.5}, 0) is True
    assert esp.needs_bias_change({"bias_v": -1.5}, 1) is True
    assert esp.needs_bias_change({"bias_v": None}, 0) is False
    assert esp.needs_bias_change({}, 1) is False


def test_mutation_bias_only_on_the_first_attempt(epoch_now, monkeypatch):
    """变异验证:把规则退回 ``attempts == 0``,重扫那次就不再设偏压。"""
    monkeypatch.setattr(
        esp, "needs_bias_change",
        lambda frame, attempts: attempts == 0 and frame.get("bias_v") is not None)
    # ① 变异已应用
    assert esp.needs_bias_change({"bias_v": -1.5}, 1) is False
    # ② 被守卫的行为红了
    ctx = PlanCtx(scan_outcomes=["bad", "ok"], storage=epoch_now)
    _run(ctx, _plan(1, stamp=7, bias_v=-1.5))
    assert len(_scans(ctx)) == 2
    assert len(_biases(ctx)) == 1, "变异没生效,这条变异验证是假的"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
