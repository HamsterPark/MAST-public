"""修针流程的状态机 —— 每一条分支都对着用户口述的那句话。

用一个假 ExecutionContext 驱动真正的 GraphExecutor：``run`` 按技能名返回可编排的
结果，所以测的是**流程的决策**（打几发、什么时候翻极性、什么时候加深、什么时候
转入临界浅扎），不是 Nanonis 协议。
"""
from __future__ import annotations

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

import inspect

import pytest  # noqa: E402

from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite.prepare_noble_tip import (  # noqa: E402
    PokeConditionTip,
    PrepareNobleTip,
    PulseConditionTip,
)


#: 让某个技能失败的哨兵。不能用 None —— 那和「script 里没这个技能」撞了。
FAIL = object()


class FakeCtx:
    """按技能名派活的假上下文。

    ``script`` 把技能名映射到 dict、或 callable(params, nth) -> dict。返回 ``FAIL``
    表示这一步失败。没写进 script 的技能一律成功且 data 为空 —— 所以「没配置」和
    「配置成失败」必须用不同的值表示，用 None 兼任两者会让一个本该失败的步骤
    悄悄变成成功（本文件第一版就是这么错的）。
    """

    def __init__(self, script=None, *, abort_after=None, halt_after=None):
        self.script = dict(script or {})
        self.calls: list[tuple[str, dict]] = []
        self._n: dict[str, int] = {}
        self._abort_after = abort_after
        self._halt_after = halt_after
        self.run_id = "test-run"

    # -- ExecutionContext surface used by GraphExecutor ---------------
    def run(self, skill_name, params, version=None):
        self.calls.append((skill_name, dict(params)))
        n = self._n.get(skill_name, 0)
        self._n[skill_name] = n + 1
        fn = self.script.get(skill_name, {})
        data = fn(params, n) if callable(fn) else fn
        if data is FAIL:
            return SkillResult(skill_name=skill_name, success=False,
                               error="scripted failure")
        return SkillResult(skill_name=skill_name, success=True, data=dict(data))

    def check_abort(self):
        return (self._abort_after is not None
                and len(self.calls) >= self._abort_after)

    def check_halt(self):
        if self._halt_after is not None and len(self.calls) >= self._halt_after:
            return "针尖质量掉了"
        return ""

    def safe_call(self, method, *args, role="main"):
        class _R:
            error = ""
            return_value = ("", b"", [0.0])
            method = ""
            args = ()
        return _R()

    def count(self, name):
        return sum(1 for c, _ in self.calls if c == name)

    def params_for(self, name):
        return [p for c, p in self.calls if c == name]


# ── 脚本片段 ────────────────────────────────────────────────────────────────

def spot(dx_nm=50.0):
    """FindCleanSpot：每次给一个新位置，永远有地方可去。"""
    def _f(params, n):
        return {"x_m": (n + 1) * dx_nm * 1e-9, "y_m": 0.0,
                "distance_m": dx_nm * 1e-9, "map_known": True}
    return _f


NO_SPOT = lambda params, n: FAIL           # noqa: E731 — 表面用完了


def pulse(dz_by_shot):
    """BiasPulseWithReadback：按发数给出 Z 跳变（nm）。"""
    def _f(params, n):
        dz = dz_by_shot[min(n, len(dz_by_shot) - 1)]
        direction = "none" if abs(dz) < 0.5 else ("up" if dz > 0 else "down")
        return {"step": {"direction": direction, "delta_m": dz * 1e-9},
                "bias_v": params.get("bias_v")}
    return _f


def poke(verdicts):
    """TipShapeWithReadback：按次序给出扎针判定。"""
    def _f(params, n):
        v = verdicts[min(n, len(verdicts) - 1)]
        return {"indent": {"verdict": v,
                           "delta_m": (0.3e-9 if v == "cluster"
                                       else -0.2e-9 if v == "tip_changed_or_pit"
                                       else 0.0)}}
    return _f


def cluster(*, round_=True, peaks=1, multi=None):
    """AssessClusterRoundness 替身保留 multi_tip 三态，连通域数量不直接代表针尖数量。"""
    def _f(params, n):
        return {"equivalent_axis_ratio": 0.9 if round_ else 0.2,
                "is_round": round_, "n_components": peaks,
                "multi_tip": multi,
                "multi_tip_undecidable": None if multi is not None else "判不了"}
    return _f


def no_cluster_on_the_image():
    """簇图上**没有大到能判的东西** —— ``is_round=None`` + 面积很小。

    2026-08-17 起这才是「没扎上」的读数。在那之前是 Z 跳变说了算,而拿那 100 针
    的标定数据一对:图上确有簇的 89 针里,**55 针(62%)的 |dz| 小于 40 pm 阈值**
    (真簇 |dz| 中位数只有 5.4 pm,比基线噪声 σ≈10 pm 还小)。
    ⇒ Z 这个量没有判别力,不该由它决定「要不要加深」。
    """
    def _f(params, n):
        return {"equivalent_axis_ratio": None, "is_round": None,
                "roundness_undecidable": "只有 3 像素,低于 20 像素下限",
                "area_px": 3, "n_components": 1,
                "multi_tip": None, "multi_tip_undecidable": "判不了"}
    return _f


def flat_sites(n=6, dx_nm=15.0):
    """``FindFlatRegion`` 的替身:一批**已知平坦**的落点(单台面内)。

    字段名照真技能成功时返回的那些取(``sites`` / ``center_*_m`` / ``rms_m`` /
    ``window_side_m``)—— 名字写错的话流程会当成「这一片没有台面」换地方,
    而测试会以为是逻辑错了。
    """
    def _f(params, n_call):
        base = 1e-7 + n_call * 3e-7
        return {
            "sites": [{"center_x_m": base + i * dx_nm * 1e-9,
                       "center_y_m": 0.0,
                       "rms_m": 8e-12} for i in range(n)],
            "center_x_m": base, "center_y_m": 0.0,
            "rms_m": 8e-12, "window_side_m": 35e-9,
        }
    return _f


SCAN_FILE = {"path": "/tmp/scan.sxm"}


def run(skill, ctx, **params):
    return skill.execute(ctx, params)


# ── A 阶段：电脉冲大修 ─────────────────────────────────────────────────────

def test_stops_as_soon_as_z_jumps_up_enough():
    """「跳变为向上几十 nm 我就满意」——够了就停，不多打。"""
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([0.0, 0.0, 25.0, 25.0])})
    res = run(PulseConditionTip(), ctx)
    assert res.success, res.error
    assert ctx.count("BiasPulseWithReadback") == 3
    out = res.data["phases"][0]
    assert out["satisfied"] and out["fired"] == 3


def test_small_upward_step_is_not_enough():
    """向上 2 nm 不算 —— 判据是几十 nm。"""
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([2.0])})
    res = run(PulseConditionTip(), ctx, pulse_budget=3)
    assert res.data["phases"][0]["satisfied"] is False
    assert ctx.count("BiasPulseWithReadback") == 3


def test_moves_to_a_new_spot_before_every_shot():
    """每轮按声明的重复次数预算选择落点，成功后继续换位。"""
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([0.0])})
    run(PulseConditionTip(), ctx, pulse_budget=4, pulse_same_spot_budget=1)
    assert ctx.count("FindCleanSpot") == 4
    xs = [p["x_m"] for p in ctx.params_for("MoveToXY")]
    assert len(xs) == 4 and len(set(xs)) == 4, "每一发都要换到不同的地方"


def test_used_spots_are_fed_back_so_two_shots_never_share_one():
    """刚打过的点由调用方传回去 —— marker 可能还没落库。

    豁免关掉时的语义，见上一条的说明。
    """
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([0.0])})
    run(PulseConditionTip(), ctx, pulse_budget=3, pulse_same_spot_budget=1)
    excludes = [p["exclude_spots"] for p in ctx.params_for("FindCleanSpot")]
    assert excludes[0] == ""
    assert excludes[1].count(";") == 0 and excludes[1]
    assert excludes[2].count(";") == 1


def test_duds_may_stay_on_one_spot_up_to_the_exemption_budget():
    """连续脉冲的豁免：**没打动针尖**时允许原地再来若干发。

    为什么需要它（2026-08-13）：有 XY 粗动的仪器上脉冲避让半径是 500 nm、
    中心区也是 500 nm —— 一发就把中心区盖满。没有豁免的话，第二发无效脉冲
    会直接撞上「这片表面没落点了」而中止大修，**而它其实只是还没打动**。
    """
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([0.0])})
    run(PulseConditionTip(), ctx, pulse_budget=4, pulse_same_spot_budget=4)
    assert ctx.count("BiasPulseWithReadback") == 4, "四发都该打出去"
    assert ctx.count("FindCleanSpot") == 1, (
        f"无效脉冲在豁免额度内不该换地方（找了 {ctx.count('FindCleanSpot')} 次落点）")


def test_the_exemption_runs_out_and_then_it_must_move():
    """豁免是**额度**不是许可证：攒满就必须挪窝。"""
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([0.0])})
    run(PulseConditionTip(), ctx, pulse_budget=5, pulse_same_spot_budget=2)
    assert ctx.count("BiasPulseWithReadback") == 5
    # 2 发一组 ⇒ 第 1、3、5 发之前各找一次落点
    assert ctx.count("FindCleanSpot") == 3, (
        f"额度用完没有换地方（找了 {ctx.count('FindCleanSpot')} 次）")


def test_a_successful_shot_is_never_exempt():
    """打**成功**的那一发不吃豁免 —— 成功意味着确实有材料动了，那里从此是坑。

    （成功即结束大修阶段，所以这里验的是「它停了」，而落点已经进脏点表，
    后面的 verify 会躲开它。）
    """
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([0.0, 25.0, 25.0, 25.0])})
    run(PulseConditionTip(), ctx, pulse_budget=4, pulse_same_spot_budget=4)
    assert ctx.count("BiasPulseWithReadback") == 2, "第二发就满意了，不该继续打"


def test_flips_polarity_after_a_run_of_duds():
    """「先一种，无效再换」—— 连打 N 发没效果就翻极性。"""
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([0.0])})
    run(PulseConditionTip(), ctx, pulses_per_polarity=2, pulse_budget=4)
    biases = [p["bias_v"] for p in ctx.params_for("BiasPulseWithReadback")]
    assert biases[0] > 0 and biases[1] > 0, biases
    assert biases[2] < 0 and biases[3] < 0, "两发无效之后应当翻极性"


def test_budget_is_an_action_cap_not_a_retry_count():
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([0.0])})
    res = run(PulseConditionTip(), ctx, pulse_budget=5)
    assert ctx.count("BiasPulseWithReadback") == 5
    assert "预算" in res.data["phases"][0]["reason"]


def test_spent_surface_stops_the_run_and_says_so():
    """没干净地方了就停下报告，不在坑里接着打。"""
    ctx = FakeCtx({"FindCleanSpot": NO_SPOT})
    res = run(PulseConditionTip(), ctx)
    out = res.data["phases"][0]
    assert out.get("surface_spent")
    assert "粗动" in out["reason"] or "换区" in out["reason"]
    assert ctx.count("BiasPulseWithReadback") == 0


def test_junction_conditions_are_set_before_pulsing():
    """「进针到表面之后，我会调低偏压，调大电流」。"""
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([25.0])})
    run(PulseConditionTip(), ctx)
    names = [c for c, _ in ctx.calls]
    assert names.index("SetBias") < names.index("BiasPulseWithReadback")
    assert ctx.params_for("SetBias")[0]["bias_v"] == pytest.approx(0.05)
    assert ctx.params_for("SetSetpoint")[0]["setpoint_a"] == pytest.approx(1e-9)


def test_a_failed_pulse_stops_the_run():
    ctx = FakeCtx({"FindCleanSpot": spot(), "BiasPulseWithReadback": FAIL})
    res = run(PulseConditionTip(), ctx)
    assert res.success is False


def test_abort_stops_between_shots():
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([0.0])},
                  abort_after=6)
    res = run(PulseConditionTip(), ctx, pulse_budget=20)
    assert res.success is False
    assert ctx.count("BiasPulseWithReadback") < 20


def test_tip_halt_stops_the_run():
    """针尖质量掉了要停这一轮 —— composite 跑起来后 middleware 够不着。"""
    ctx = FakeCtx({"FindCleanSpot": spot(),
                   "BiasPulseWithReadback": pulse([0.0])},
                  halt_after=5)
    res = run(PulseConditionTip(), ctx, pulse_budget=20)
    assert res.success is False
    assert ctx.count("BiasPulseWithReadback") < 20


# ── D 阶段：扎针尖 ─────────────────────────────────────────────────────────

def _poke_ctx(verdicts, **extra):
    # 找不到平坦区域时不得继续扎针。
    script = {"FindCleanSpot": spot(20.0),
              "TipShapeWithReadback": poke(verdicts),
              "SaveScan": SCAN_FILE, "GetLatestScanFile": SCAN_FILE,
              "FindFlatRegion": flat_sites(),
              "AssessClusterRoundness": cluster()}
    script.update(extra)
    return FakeCtx(script)


def test_nothing_on_the_image_means_go_deeper():
    """加深与否由簇图判据决定；Z 读数只留档，不能单独触发加深。"""
    ctx = _poke_ctx(["no_change"],
                    AssessClusterRoundness=no_cluster_on_the_image())
    run(PokeConditionTip(), ctx, poke_budget=3)
    depths = [abs(p["tip_lift_m"]) for p in ctx.params_for("TipShapeWithReadback")]
    assert depths[1] > depths[0] and depths[2] > depths[1], depths


def test_a_cluster_on_the_image_stops_the_deepening_even_when_z_says_no_change():
    """检测到团簇时停止加深，并记录所用深度。"""
    ctx = _poke_ctx(["no_change"])          # 默认 cluster() = 图上有个圆簇
    run(PokeConditionTip(), ctx, poke_budget=3)
    depths = [abs(p["tip_lift_m"]) for p in ctx.params_for("TipShapeWithReadback")]
    assert depths[1] <= depths[0], (
        f"图上有簇却还在加深:{depths} —— Z 又当家了?")


def test_deep_stage_stops_when_the_depth_cap_is_reached():
    ctx = _poke_ctx(["no_change"],
                    AssessClusterRoundness=no_cluster_on_the_image())
    res = run(PokeConditionTip(), ctx, poke_budget=50, poke_depth_nm=7.0)
    assert res.data["phases"][0]["reason"]
    assert ctx.count("TipShapeWithReadback") < 50


def test_seeing_something_on_the_image_moves_on_to_the_threshold_stage():
    """图像发现可判读目标后转入阈值阶段，不根据孤立 Z 变化切换策略。"""
    ctx = _poke_ctx(["tip_changed_or_pit", "no_change"])
    res = run(PokeConditionTip(), ctx, poke_budget=3)
    assert res.data["phases"][0]["stage"] == "critical"


def test_a_z_pit_with_nothing_on_the_image_does_not_switch_strategy():
    """Z 说「扎出坑」而图上什么都没有 ⇒ **不切策略**,按没扎上加深。"""
    ctx = _poke_ctx(["tip_changed_or_pit", "tip_changed_or_pit"],
                    AssessClusterRoundness=no_cluster_on_the_image())
    res = run(PokeConditionTip(), ctx, poke_budget=3)
    assert res.data["phases"][0]["stage"] == "deep", (
        "凭一个 Z 读数就切到临界阶段了 —— 那正是「锁死在 200」的来源")


def test_a_lumpy_cluster_moves_before_it_digs_deeper():
    """扎上了但不够圆 ⇒ **换个地方,同一深度再来**;试满几次才加深。

    ## 这条测试掉过头,记下为什么

    原来它叫 ``test_a_lumpy_cluster_keeps_going_deeper``,断言的是「不圆就加深」,
    出处是「尤其是有双针尖或特别不圆我们就会来深的」。**2026-08-10 的用户范本
    推翻了它**(`docs/v2/design/operator_tip_forging_reference_20260810.md` §一.6):
    实际序列是 500 → 500 → 500 → 200 pm —— 不圆先**换地方**,深度不动;
    而收尾是**减浅**,不是加深。

    范本那份文档写明「凡与代码冲突之处以用户为准」,所以这里改的是判据本身,
    不是把测试放宽。加深仍然存在,只是退到了「同一深度换了几个地方都不圆」之后 ——
    那才是"深度不够"的证据。
    """
    ctx = _poke_ctx(["cluster"], AssessClusterRoundness=cluster(round_=False))
    run(PokeConditionTip(), ctx, poke_budget=3)
    depths = [abs(p["tip_lift_m"]) for p in ctx.params_for("TipShapeWithReadback")]
    spots = ctx.params_for("MoveToXY")

    assert depths[1] == pytest.approx(depths[0]), "不圆是位置的问题,深度不该动"
    assert len(spots) >= 2 and spots[0] != spots[1], "应当换个地方再扎"


def test_it_goes_shallower_once_several_spots_all_came_out_lumpy():
    """同一深度换了 ``poke_same_depth_retries`` 个地方仍不圆 ⇒ 才动深度,**往浅里动**。

    没有这一条,「不圆就永远不动深度」也能让上面那条变绿 —— 而那会让流程在一个
    不对的深度上把预算耗光。

    ## 这条测试第二次掉头,记下为什么

    上一版叫 ``test_it_does_deepen_once_several_spots_all_came_out_lumpy``,断言
    第 n+1 次**加深**。它是在应用 2026-08-10 范本时留下的半截:范本序列是
    500 → 500 → 500 → **200** pm,上一版把前三步的「不加深」实现了,却把最后
    那一步的**减浅**读成了加深 —— 上面那条测试的 docstring 里甚至已经写着
    「收尾是减浅,不是加深」,而它下一句就说「加深仍然存在」。

    2026-08-15 一百针标定给出了独立的第二个理由(`artifacts/poke_calibration.html`):
    能走到这一行说明**已经接触上了**,不圆是搬多了不是碰得浅。
    深度 → 轴比 ρ=−0.46 (p=0.005, n=35);同一检验在没接触那组 p=0.19。

    现在断言的就是范本那一串本身。
    """
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE as W

    n = int(W.poke_same_depth_retries)
    ctx = _poke_ctx(["cluster"], AssessClusterRoundness=cluster(round_=False))
    run(PokeConditionTip(), ctx, poke_budget=n + 2)
    depths = [abs(p["tip_lift_m"]) for p in ctx.params_for("TipShapeWithReadback")]

    assert len(depths) > n, f"预算不够跑到第 {n + 1} 次"
    assert depths[n] < depths[0], (
        f"同一深度试了 {n} 个地方都不圆,第 {n + 1} 次应当**变浅**:{depths}")
    # 逐字对上范本:500 → 500 → 500 → 200 pm。
    pm = [round(d * 1e12) for d in depths[:n + 1]]
    assert pm == [500, 500, 500, 200], f"与用户范本序列不一致:{pm}"


def test_a_detected_multi_tip_moves_first_just_like_a_lumpy_one():
    """判**出**多针尖时走同一条路 —— 它和「不圆」是同一个判据的两半。

    2026-08-15 起 ``multi_tip`` 是**三态**,所以替身要显式说 True。
    从前这里传的是 ``peaks=2``,因为当时的判据是 ``n_components > 1``。
    """
    ctx = _poke_ctx(["cluster"],
                    AssessClusterRoundness=cluster(round_=True, peaks=2,
                                                   multi=True))
    run(PokeConditionTip(), ctx, poke_budget=3)
    depths = [abs(p["tip_lift_m"]) for p in ctx.params_for("TipShapeWithReadback")]
    assert depths[1] == pytest.approx(depths[0])


def test_more_than_one_blob_is_no_longer_called_a_double_tip():
    """多个连通域可能来自表面结构，不能直接判为多针尖。"""
    ctx = _poke_ctx(["cluster"],
                    AssessClusterRoundness=cluster(round_=True, peaks=5))
    res = run(PokeConditionTip(), ctx, poke_budget=3)
    out = res.data["phases"][0]
    assert out["stage"] == "critical", (
        "5 个连通域 + 圆 ⇒ 应当照常转临界浅扎,而不是当成双针尖换地方")


def test_multi_tip_is_decided_by_several_comparable_blobs_not_by_counting_them():
    """多针尖判据 = 「**几坨体量相当的东西,离得不远**」,不是「连通域 > 1」。

    ## 这一条当天写了两版,记下为什么

    第一版钉的是「**没有任何东西会产生 multi_tip=True**」—— 当时数连通域、
    自相关、跨帧一致三条路都判不了,所以如实把它留成三态的 None。

    那个结论被推翻了:把所有的连通域都算上、不关心它们是否连通、全都取了,
    那么多针尖算出来一定不圆 —— 对的。
    此前一直只量 ``select`` 选中的**那一块**,而多针尖的每个顶点各自扎出一坨
    **正常的**簇:单看一坨,形状尺寸和好针尖逐维完全重叠。
    **证据不在任何一坨里面,在它们之间的关系里。**

    97 张盲标上的成绩:边界轴比 <0.75 单独抓 63/68,漏的 5 张**全是多针尖**;
    加上这一条 → **67/68**,只漏 #88。误杀 4/12 → 6/12。
    """
    from mast.skills.builtins.cluster_roundness import (
        MULTI_TIP_SIZE_FRAC, MULTI_TIP_SPAN_NM, AssessClusterRoundness)

    assert MULTI_TIP_SIZE_FRAC == pytest.approx(0.25)
    assert MULTI_TIP_SPAN_NM == pytest.approx(6.0)

    src = inspect.getsource(AssessClusterRoundness)
    # 三态仍在:算不出间距(读不到像素尺寸)时必须是**判不了**,不是「没有」。
    assert "multi_tip = None" in src
    assert "multi_tip_undecidable" in src
    assert "读不到像素物理尺寸" in src, "判不了那一支要说得出为什么"
    # 而且它现在**真的会说 True** —— 上一版这里断言的是相反的事。
    assert "multi_tip = bool(len(near) >= 2)" in src


def test_an_undecidable_multi_tip_still_does_not_block_finishing():
    """``multi_tip`` 判不了(None)时**不拦收工** —— 与本流程对锐度的做法一致。

    拦不拦是一回事,**说不说**是另一回事:判不了时报文要写出来,
    否则「簇单峰且圆」会把一个没验证过的前提说成已验证。
    """
    ctx = _poke_ctx(["cluster"], AssessClusterRoundness=cluster(round_=True))
    res = run(PokeConditionTip(), ctx, poke_budget=8, critical_repeat_n=1)
    out = res.data["phases"][0]
    assert out.get("refined") is True, "判不了不该拦住收工"
    assert "判不了" in str(out.get("reason", "")), (
        f"收工理由没说出多针尖判不了:{out.get('reason')}")


def test_single_round_cluster_switches_to_threshold_poking():
    """单峰圆簇触发从浅到深的阈值搜索。"""
    ctx = _poke_ctx(["cluster", "no_change"])
    res = run(PokeConditionTip(), ctx, poke_budget=4)
    out = res.data["phases"][0]
    assert out["stage"] == "critical"
    depths_pm = [abs(p["tip_lift_m"]) * 1e12
                 for p in ctx.params_for("TipShapeWithReadback")]
    from mast.core.noble_tip_workflow import NobleTipWorkflow
    # 起始深度来自配置，不写死在流程中。
    assert depths_pm[1] == pytest.approx(NobleTipWorkflow().critical_start_pm)


def test_threshold_stage_steps_up_until_the_image_shows_something():
    """「这时候肯定碰不到；然后 +50；再 +50」—— **由扫图说没碰到,不由 Z**。

    2026-08-17 起 D2 也改成看图判。那 100 针的标定说 Z 没有判别力:
    图上确有簇的 89 针里 **55 针(62%)** 的 |dz| 低于 40 pm 阈值。
    所以这条脚本改成让**簇图**说「什么都没有」,阶梯才走得起来。
    """
    from mast.core.noble_tip_workflow import NobleTipWorkflow
    wf = NobleTipWorkflow()
    start, step = wf.critical_start_pm, wf.critical_step_pm

    # 第 1 针扎出圆簇 → 转 D2;之后两针图上没东西 → 加两级;第 4 针又有簇。
    def look(params, n):
        return (cluster()(params, n) if n in (0, 3)
                else no_cluster_on_the_image()(params, n))

    ctx = _poke_ctx(["cluster"] * 5, AssessClusterRoundness=look)
    run(PokeConditionTip(), ctx, poke_budget=5, critical_repeat_n=1)
    depths_pm = [round(abs(p["tip_lift_m"]) * 1e12)
                 for p in ctx.params_for("TipShapeWithReadback")]
    assert depths_pm[1:4] == [round(start), round(start + step),
                              round(start + 2 * step)], depths_pm


def test_it_accepts_after_several_consecutive_round_clusters():
    """每次动作重新核对深度阈值，连续团簇单独计数。"""
    from mast.core.noble_tip_workflow import NobleTipWorkflow
    start = NobleTipWorkflow().critical_start_pm

    ctx = _poke_ctx(["cluster"] * 6)
    res = run(PokeConditionTip(), ctx, poke_budget=10, critical_repeat_n=3)
    out = res.data["phases"][0]
    assert out["refined"] is True
    assert out["repeats"] == 3

    # 每一针都回到起步深度 —— 不许出现「锁在某个深度反复扎」的序列。
    depths_pm = [round(abs(p["tip_lift_m"]) * 1e12)
                 for p in ctx.params_for("TipShapeWithReadback")]
    assert depths_pm[1:] == [round(start)] * (len(depths_pm) - 1), (
        f"D2 没有每次从头找:{depths_pm}")


def test_a_lumpy_cluster_resets_the_consecutive_count():
    """连续性断了就归零 —— 收工要的是**连续 N 次确实圆**。

    「连续 N 次没被否掉」和「连续 N 次确实圆」是两回事:前者把「判不了」
    也算成合格,而判不了恰恰是最该继续扎的那一档。
    """
    def look(params, n):
        # 圆、圆、不圆、圆、圆、圆 ⇒ 连续 3 次只能在最后三针达成。
        return (cluster(round_=False)(params, n) if n == 2
                else cluster()(params, n))

    ctx = _poke_ctx(["cluster"] * 8, AssessClusterRoundness=look)
    res = run(PokeConditionTip(), ctx, poke_budget=10, critical_repeat_n=3)
    out = res.data["phases"][0]
    assert out["refined"] is True
    assert ctx.count("TipShapeWithReadback") >= 6, (
        "不圆那一针没有把连续计数打断")


def test_feedback_is_always_restored():
    """收尾必须发生 —— 而且参数名是 enable（曾写成 on，被 optional 静默吞掉）。"""
    ctx = _poke_ctx(["no_change"])
    run(PokeConditionTip(), ctx, poke_budget=2)
    restores = ctx.params_for("ZControllerOnOff")
    assert restores, "没有恢复反馈"
    assert restores[-1] == {"enable": True}


def test_feedback_restored_even_when_the_surface_is_spent():
    ctx = FakeCtx({"FindCleanSpot": NO_SPOT})
    run(PokeConditionTip(), ctx)
    assert ctx.params_for("ZControllerOnOff")[-1] == {"enable": True}


def test_poke_does_not_change_bias():
    """「这个过程中可以同时加电脉冲，我的习惯是不加」。"""
    ctx = _poke_ctx(["cluster"])
    run(PokeConditionTip(), ctx, poke_budget=1)
    p = ctx.params_for("TipShapeWithReadback")[0]
    assert p["change_bias"] is False
    assert p["restore_feedback"] is True
    assert p["bias_settling_s"] == pytest.approx(0.5)   # 「500ms，时间可调」


def test_plunge_goes_down_and_comes_back_up():
    ctx = _poke_ctx(["cluster"])
    run(PokeConditionTip(), ctx, poke_budget=1)
    p = ctx.params_for("TipShapeWithReadback")[0]
    assert p["tip_lift_m"] < 0 < p["lift_height_m"]
    assert abs(p["tip_lift_m"]) == pytest.approx(abs(p["lift_height_m"]))


# ── 全流程 ─────────────────────────────────────────────────────────────────

def _full_ctx(**over):
    script = {
        "FindCleanSpot": spot(),
        "BiasPulseWithReadback": pulse([25.0]),
        "PreScanCheck": {"tip_ready": True, "similarity": 0.93},
        "SaveScan": SCAN_FILE, "GetLatestScanFile": SCAN_FILE,
        "AnalyzeFrameTilt": {"surface_rms_m": 1e-10, "step_dominated": True},
        # 5e-11 = 50 pm 是一个**不可能的成功**:2026-08-11 起 `FindFlatRegion`
        # 成功即意味着局部残差 ≤ 25 pm(超过就失败并说「这里没有」)。
        "FindFlatRegion": {"center_x_m": 1e-8, "center_y_m": 2e-8, "rms_m": 8e-12},
        "AutoTilt": {"action": "leveled"},
        "TipShapeWithReadback": poke(["cluster"]),
        "AssessClusterRoundness": cluster(),
        "AssessTipSharpness": {"edge_resolution_nm": 0.4, "verdict": "measured",
                               "has_step": True},
    }
    script.update(over)
    return FakeCtx(script)


def test_full_run_reaches_ready():
    ctx = _full_ctx()
    res = run(PrepareNobleTip(), ctx, critical_repeat_n=1)
    assert res.success, res.error
    assert res.data["outcome"] == "ready"
    assert "基础态" in res.data["summary_cn"]


def test_verification_failure_goes_back_to_pulsing():
    """「如果不重合那针尖还是很差，重新打电脉冲」。"""
    ctx = _full_ctx(PreScanCheck={"tip_ready": False, "similarity": 0.3})
    res = run(PrepareNobleTip(), ctx, max_verify_rounds=3)
    assert res.data["outcome"] == "verify_exhausted"
    assert ctx.count("PreScanCheck") == 3, "验证↔大修来回三轮后认输"


def test_second_round_uses_the_descending_voltages(monkeypatch):
    """「有时候在大幅的电脉冲和扎针尖之间我们会搞几个 7V，5V，3V」。

    **必须登记一根金属丝针**（2026-08-10）。这条测的是递减序列，不是包络；
    而不登记针尖时用的是**通用保守档**（`max_abs_pulse_v = 6 V`），流程表默认的
    10 V 在它上面违法 ⇒ 出厂默认与包络的对账会把首轮降到 3 V，这条测试就变成在
    测对账而不是测递减序列。钨丝档的上限正好是 10 V，首轮 10 V 在它上面合法。
    """
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "current_tip_facts", lambda: {
        "name": "w-wire", "material": "W", "fabrication": "etched",
        "form": "stm_wire"}, raising=False)

    ctx = _full_ctx(PreScanCheck={"tip_ready": False, "similarity": 0.3},
                    BiasPulseWithReadback=pulse([0.0]))
    run(PrepareNobleTip(), ctx, max_verify_rounds=2, pulse_budget=3)
    volts = [abs(p["bias_v"]) for p in ctx.params_for("BiasPulseWithReadback")]
    assert volts[:3] == [10.0, 10.0, 10.0], volts
    assert volts[3:6] == [7.0, 5.0, 3.0], volts


def test_an_unregistered_tip_pulses_at_the_generic_tier_not_at_10_v(monkeypatch):
    """未登记针尖：**不是不对账，是按通用保守档对账**。

    通用档 `max_abs_pulse_v = 6 V` 而流程表默认 10 V ⇒ 在此之前
    **未登记时一发脉冲都打不出去**（`BiasPulseWithReadback` 被包络门拒绝），
    而「未登记」是本系统最常见的状态。现在降到通用档自己的 3 V 并出声。

    ⚠️ 这是一个**取舍**，已列进设计文档的「待用户定夺」：
    降到 3 V 让流程跑得起来但弱于用户的做法；另一条路是拒绝并要求先登记针尖。
    """
    import mast.core.tip_state as tip_state
    from mast.core.tip_conditioning_policy import resolve_policy

    monkeypatch.setattr(tip_state, "current_tip_facts", lambda: None,
                        raising=False)
    ctx = _full_ctx(PreScanCheck={"tip_ready": False, "similarity": 0.3},
                    BiasPulseWithReadback=pulse([0.0]))
    run(PrepareNobleTip(), ctx, max_verify_rounds=1, pulse_budget=2)
    volts = [abs(p["bias_v"]) for p in ctx.params_for("BiasPulseWithReadback")]
    assert volts, "一发都没打"
    limit = float(resolve_policy(None)["max_abs_pulse_v"])
    assert max(volts) <= limit, (
        f"未登记针尖上打了 {max(volts)} V，超出通用档包络 {limit} V —— 会被拒绝")


def test_qplus_skips_the_descending_pulses(monkeypatch):
    """音叉上那几伏可能把叉臂也一起修了。"""
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: {
        "id": 1, "name": "qp", "material": "PtIr", "form": "qplus"})
    ctx = _full_ctx(PreScanCheck={"tip_ready": False, "similarity": 0.3},
                    BiasPulseWithReadback=pulse([0.0]))
    run(PrepareNobleTip(), ctx, max_verify_rounds=2, pulse_budget=2, pulse_v=2.0)
    volts = [abs(p["bias_v"]) for p in ctx.params_for("BiasPulseWithReadback")]
    assert set(volts) == {2.0}, volts


def test_inconclusive_verification_is_not_treated_as_failure():
    """读不到线数据 ≠ 针尖不好 —— 当成不好会让流程一直打脉冲。"""
    ctx = _full_ctx(PreScanCheck={"tip_ready": False, "similarity": None})
    res = run(PrepareNobleTip(), ctx, max_verify_rounds=5)
    assert res.data["outcome"] == "verify_inconclusive"
    assert ctx.count("PreScanCheck") == 1, "判不了就停下让人看，不要接着打"
    assert "不等于" in res.data["summary_cn"]


def test_skip_refine_stops_after_levelling():
    ctx = _full_ctx()
    res = run(PrepareNobleTip(), ctx, skip_refine=True)
    assert res.data["outcome"] == "leveled_no_refine"
    assert ctx.count("TipShapeWithReadback") == 0


def test_levelling_uses_a_flat_patch_not_the_step_frame():
    """「选一个无台阶的 20~100nm 的区域执行调平」。"""
    ctx = _full_ctx()
    run(PrepareNobleTip(), ctx, critical_repeat_n=1, flat_region_nm=50.0)
    # 2026-08-14:``FindFlatRegion`` 现在有**两个**调用方 —— level 相找调平用的
    # 平区(这条测试的对象),以及 D 相找**扎针落点**(要求:「在扎针之前不是
    # 要找台面和调平吗」)。所以次数不再是 1;这条测试要钉的是**调平那一次**
    # 用的是平区窗口而不是整张台阶图,那由下面 next_frame_m 那行断言。
    calls = ctx.params_for("FindFlatRegion")
    assert calls, "调平那一步没有去找平区"
    assert calls[0]["min_window_m"] == pytest.approx(50e-9), (
        f"第一次 FindFlatRegion 不是调平那一次(窗口 {calls[0].get('min_window_m')})")
    assert ctx.params_for("AutoTilt")[0]["next_frame_m"] == pytest.approx(50e-9)


def test_acceptance_returns_to_the_stepped_area():
    """验收要看台阶陡不陡 —— 得回到有台阶的地方，不是簇上面。"""
    ctx = _full_ctx()
    run(PrepareNobleTip(), ctx, critical_repeat_n=1)
    assert ctx.count("AssessTipSharpness") == 1


def test_unknown_map_is_reported_not_swallowed():
    """读不到记录 ≠ 表面干净 —— 这句必须出现在报告里。"""
    def blind(params, n):
        return {"x_m": n * 5e-8, "y_m": 0.0, "distance_m": 5e-8,
                "map_known": False}
    ctx = _full_ctx(FindCleanSpot=blind)
    res = run(PrepareNobleTip(), ctx, critical_repeat_n=1)
    assert "读不到实验记录" in res.data["summary_cn"]


# ── 门控 ───────────────────────────────────────────────────────────────────

def test_all_three_are_confirm_level():
    """CONFIRM = agent 路上一次批准跑完整条流程（子步骤不重新过门）。"""
    from mast.core.types import SafetyLevel
    for cls in (PulseConditionTip, PokeConditionTip, PrepareNobleTip):
        assert cls().metadata().safety_level == SafetyLevel.CONFIRM


def test_capabilities_are_declared_for_mode_gating():
    """三档操作模式认的是 capabilities，不是 safety_level。"""
    assert "bias_pulse" in PulseConditionTip().metadata().capabilities
    assert "tip_shaping" in PokeConditionTip().metadata().capabilities
    assert {"bias_pulse", "tip_shaping"} <= PrepareNobleTip().metadata().capabilities


def test_poke_runs_on_qplus_without_being_asked_twice(monkeypatch):
    """qPlus 闸默认关闭，深度包络与偏压守卫仍独立生效。"""
    monkeypatch.delenv("MAST_QPLUS_POKE_GUARD", raising=False)
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: {
        "id": 1, "name": "qp", "material": "PtIr", "form": "qplus"})
    ctx = _poke_ctx(["cluster"])
    run(PokeConditionTip(), ctx, poke_budget=1, poke_depth_nm=0.3)
    assert ctx.count("TipShapeWithReadback") == 1


def test_poke_is_refused_on_qplus_when_the_guard_is_turned_back_on(monkeypatch):
    """把护栏开回来之后**还拦得住**。

    默认值改了 ≠ 门可以烂掉。一道关着的门还得能用,否则哪天有人开回来,
    等着他的是一道看着在防护、其实早就坏了的门。
    """
    monkeypatch.setenv("MAST_QPLUS_POKE_GUARD", "1")
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: {
        "id": 1, "name": "qp", "material": "PtIr", "form": "qplus"})
    ctx = _poke_ctx(["cluster"])
    res = run(PokeConditionTip(), ctx)
    assert res.success is False
    assert "qPlus" in (res.error or "")
    assert ctx.count("TipShapeWithReadback") == 0


def test_poke_allowed_on_qplus_when_signed_off(monkeypatch):
    """护栏开着时,显式签字仍然放行 —— 这条出路没被新默认值挤掉。"""
    monkeypatch.setenv("MAST_QPLUS_POKE_GUARD", "1")
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: {
        "id": 1, "name": "qp", "material": "PtIr", "form": "qplus"})
    ctx = _poke_ctx(["cluster"])
    run(PokeConditionTip(), ctx, allow_on_qplus=True, poke_budget=1,
        poke_depth_nm=0.3)
    assert ctx.count("TipShapeWithReadback") == 1


def test_deep_plunge_is_refused_on_a_qplus_envelope(monkeypatch):
    """超出 qPlus 下压包络的深扎必须被拒，且不许夹紧。

    探针值**跟着包络走**：这一档曾从 0.5 nm 抬到 5 nm，
    而原来写死的 1.5 nm 探针就此落进包络内 —— 断言会安静地变成一条永远绿的空话。
    """
    import mast.core.tip_state as tip_state
    from mast.core.tip_conditioning_policy import resolve_policy

    facts = {"id": 1, "name": "qp", "material": "PtIr", "form": "qplus"}
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: facts)
    limit_m = float(resolve_policy(facts)["max_poke_depth_m"])
    errs = PokeConditionTip().validate_params(
        {"poke_depth_nm": limit_m * 1e9 * 1.2})
    assert errs and any(f"{limit_m:.3e}" in e for e in errs), errs


def test_monitoring_suppresses_current_alarms_during_the_run():
    """修针期间电流本来就该剧烈 —— 名字不在抑制表里会被判成 CRITICAL。"""
    from mast.monitoring.service import SUPPRESS_SKILL_PATTERNS
    for name in ("PulseConditionTip", "PokeConditionTip", "PrepareNobleTip",
                 "BiasPulseWithReadback"):
        assert any(pat in name.lower() for pat in SUPPRESS_SKILL_PATTERNS), name


def test_registered_for_the_frozen_build():
    """判据是「被 __init__ import」——不是写在 __all__ 里就够。"""
    import mast.skills.composite as c
    for n in ("PulseConditionTip", "PokeConditionTip", "PrepareNobleTip"):
        assert hasattr(c, n) and n in c.__all__


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ══════════════════════════════════════════════════════════════════════
# ⑯(2026-08-06 首演,同点复现两次):流程把自己锁死在「关环 → 位移」上
# ══════════════════════════════════════════════════════════════════════
#
# `MoveToXY` 的前置是 `z_controller_on`,而修针流程**上一步刚把控制器挂起**:
# 脉冲带 `z_hold=1`(固件在脉冲期间挂起 Z 环,否则电流暴涨时 Z 会一头扎进表面),
# 扎针也开环。于是「关环 → 位移」在同一个循环里首尾相接:两次运行都停在同一个
# MoveToXY 上,报 `z_controller_on is False` —— 而**关它的正是这个流程自己**。
#
# 这几条用**贯穿状态**的假机器(不是单步 mock):控制器开关是一个真的状态位,
# 脉冲会把它关掉,MoveToXY 会因为它关着而失败。单步 mock 看不见这类缺陷 ——
# 每一步单独看都是对的,错的是它们之间的顺序。


class StatefulCtx(FakeCtx):
    """有状态替身维护控制器状态。"""

    def __init__(self, script=None, **kw):
        super().__init__(script, **kw)
        self.z_on = True
        self.move_refusals = 0

    def run(self, skill_name, params, version=None):
        if skill_name == "MoveToXY" and not self.z_on:
            self.calls.append((skill_name, dict(params)))
            self.move_refusals += 1
            return SkillResult(
                skill_name=skill_name, success=False,
                error="precondition failed: z_controller_on is False")
        res = super().run(skill_name, params, version)
        if skill_name in ("BiasPulseWithReadback", "TipShapeWithReadback"):
            self.z_on = False          # z_hold:固件在动作期间挂起 Z 环
        elif skill_name == "ZControllerOnOff":
            self.z_on = bool(params.get("enable", True))
        return res


def test_the_pulse_loop_does_not_lock_itself_out_of_moving():
    """**⑯ 的主判据。** 第一发脉冲关掉控制器之后,第二发的换位仍然走得通。

    修在 `_relocate` 里:**要移动的人负责建立自己的前置**。补在那一个位移入口上,
    就没有第六个调用点会忘。
    """
    ctx = StatefulCtx({"FindCleanSpot": spot(),
                       "BiasPulseWithReadback": pulse([0.0, 0.0, 0.0, 0.0])})
    run(PulseConditionTip(), ctx)

    assert ctx.count("BiasPulseWithReadback") >= 2, (
        f"第二发没打出去:{[c for c, _ in ctx.calls]}")
    assert ctx.move_refusals == 0, "还是被自己关掉的控制器挡住了"


def test_the_feedback_is_restored_before_every_move():
    """顺序钉死:每一次 MoveToXY 之前紧挨着一次「把反馈开回来」。"""
    ctx = StatefulCtx({"FindCleanSpot": spot(),
                       "BiasPulseWithReadback": pulse([0.0, 0.0, 0.0, 0.0])})
    run(PulseConditionTip(), ctx)

    names = [c for c, _ in ctx.calls]
    assert "MoveToXY" in names
    for i, name in enumerate(names):
        if name == "MoveToXY":
            assert i > 0 and names[i - 1] == "ZControllerOnOff", (
                f"第 {i} 步 MoveToXY 前面不是恢复反馈:{names[max(0, i - 3):i + 1]}")


def test_a_failed_move_aborts_the_plan_instead_of_pulsing_somewhere_else():
    """换位失败时必须中止后续动作。"""
    class _MoveAlwaysFails(StatefulCtx):
        def run(self, skill_name, params, version=None):
            if skill_name == "MoveToXY":
                self.calls.append((skill_name, dict(params)))
                return SkillResult(skill_name=skill_name, success=False,
                                   error="motor stuck")
            return super().run(skill_name, params, version)

    ctx = _MoveAlwaysFails({"FindCleanSpot": spot()})
    res = run(PulseConditionTip(), ctx)
    assert not res.success
    assert "MoveToXY" in (res.error or ""), res.error
    # 没有在别的地方打过脉冲。
    assert ctx.count("BiasPulseWithReadback") == 0


def test_a_genuinely_spent_surface_still_says_so():
    """探针有效性:真的没落点时,那句话和 surface_spent 都要还在。"""
    ctx = StatefulCtx({"FindCleanSpot": NO_SPOT})
    res = run(PulseConditionTip(), ctx)
    data = (res.data or {}).get("pulse_result") or {}
    assert data.get("surface_spent") is True, data
    assert "落点" in str(data.get("reason", ""))
