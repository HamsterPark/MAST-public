"""自动调平闭环(TiltCalibrate + AutoTilt)。

重点在**失败路径**:调平会写一个影响之后每一张图的全局硬件状态,所以
「什么时候拒绝动手」比「怎么算」更要紧。

  * 没有响应标定 → 一律跳过(猜错符号 = 把倾斜往反方向加倍);
  * 不收敛 → 回滚到**原始**倾斜(不是 0);
  * 迭代用尽 → 如实说没达标,不把改善了一半当成功;
  * 验收阈不能低于测量分辨率,否则只是在追噪声。
"""

from __future__ import annotations

import math

import pytest

from mast.core import instrument_profile
from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.composite.auto_tilt import AutoTilt, TiltCalibrate


@pytest.fixture(autouse=True)
def _clean_profile(monkeypatch):
    """隔离进程级 profile,并把每步之间的稳定等待缩到 0(测试不等真实秒)。"""
    instrument_profile.set_profile({})
    monkeypatch.setattr("mast.skills.composite.auto_tilt.TILT_STEP_SETTLE_S", 0.0)
    monkeypatch.setattr("mast.skills.composite.auto_tilt.time.sleep",
                        lambda *_a, **_k: None)
    yield
    instrument_profile.set_profile({})


class TiltRig:
    """一台假仪器:表面有固定倾斜,压电倾斜按响应矩阵抵消它。

    ``response`` 是「施加一个单位 tilt_x 会让测到的斜率变化多少」。单位阵 =
    轴不交换、符号为正、增益 1。
    """

    def __init__(self, surface=(0.5, -0.3), tilt=(0.0, 0.0),
                 response=((1.0, 0.0), (0.0, 1.0)),
                 *, frame_m=1e-7, noise_deg=0.0, measure_error=None,
                 write_error=None, gain_drift=1.0):
        self.surface = list(surface)
        self.tilt = list(tilt)
        self.response = response
        self.frame_m = frame_m
        self.noise_deg = noise_deg
        self.measure_error = measure_error
        self.write_error = write_error
        self.gain_drift = gain_drift
        self.calls: list[tuple[str, tuple]] = []
        self.runs: list[tuple[str, dict]] = []
        self.tilt_writes: list[tuple[float, float]] = []
        self._n = 0

    # -- 测到的斜率 = 表面倾斜 + 压电倾斜经响应矩阵的贡献 --
    def measured(self):
        r = self.response
        dx = r[0][0] * self.tilt[0] + r[0][1] * self.tilt[1]
        dy = r[1][0] * self.tilt[0] + r[1][1] * self.tilt[1]
        # gain_drift 模拟「标定已经不准了」:实际响应比标定时弱/强
        return (self.surface[0] + dx * self.gain_drift,
                self.surface[1] + dy * self.gain_drift)

    def safe_call(self, method, *args, **kwargs):
        self.calls.append((method, args))
        if method == "Piezo_TiltGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=["", b"", [self.tilt[0], self.tilt[1]]])
        if method == "Piezo_TiltSet":
            if self.write_error:
                return NanonisCallRecord(method=method, args=args,
                                         error=self.write_error)
            self.tilt = [float(args[0]), float(args[1])]
            self.tilt_writes.append((self.tilt[0], self.tilt[1]))
            return NanonisCallRecord(method=method, args=args)
        if method == "Scan_FrameGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=["", b"", [0.0, 0.0, self.frame_m, self.frame_m, 0.0]])
        return NanonisCallRecord(method=method, args=args)

    def run(self, skill_name, params, version=None):
        self.runs.append((skill_name, dict(params)))
        if skill_name != "TiltProbeCircle":
            return SkillResult(skill_name=skill_name, success=True, data={})
        if self.measure_error:
            return SkillResult(skill_name=skill_name, success=False,
                               error=self.measure_error)
        self._n += 1
        mx, my = self.measured()
        if self.noise_deg:
            import numpy as np
            rng = np.random.default_rng(self._n)
            mx += float(rng.normal(0, self.noise_deg))
            my += float(rng.normal(0, self.noise_deg))
        return SkillResult(
            skill_name=skill_name, success=True,
            data={
                "valid": True,
                "tilt_x_deg": mx, "tilt_y_deg": my,
                "slope_mag_deg": math.degrees(math.atan(
                    math.hypot(math.tan(math.radians(mx)),
                               math.tan(math.radians(my))))),
                "residual_rms_m": 15.4e-12,
                "noise_floor_m": 15.4e-12,
                "radius_m": 20e-9,
                "n_points": 24,
                "center_x_m": 0.0, "center_y_m": 0.0,
            })

    def check_abort(self):
        return False


def _calibrated(g=((-1.0, 0.0), (0.0, -1.0))):
    """写入一个响应标定。G = -M⁻¹;M=单位阵时 G = -单位阵。"""
    instrument_profile.set_tilt_calibration(g, cond=1.0)


# ══════════════════════════════════════════════════════════════════════════
#  一、没有标定就绝不动硬件
# ══════════════════════════════════════════════════════════════════════════

def test_auto_tilt_skips_without_a_calibration():
    """猜错符号 = 把倾斜往反方向加倍。这是最重要的一条拒绝。"""
    rig = TiltRig(surface=(2.0, 0.0))
    res = AutoTilt().execute(rig, {})
    assert not res.success
    assert res.data["outcome"] == "skipped"
    assert res.data["reason"] == "calibration_missing"
    assert res.data["next_action_hint"] == "run_tilt_calibrate"
    assert rig.tilt_writes == [], "没有标定却动了压电倾斜"


def test_auto_tilt_does_not_even_measure_without_calibration():
    rig = TiltRig()
    AutoTilt().execute(rig, {})
    assert not any(name == "TiltProbeCircle" for name, _ in rig.runs)


# ══════════════════════════════════════════════════════════════════════════
#  二、闸门:够平就不动
# ══════════════════════════════════════════════════════════════════════════

def test_no_action_when_the_tilt_is_within_budget_for_the_frame():
    _calibrated()
    # 0.001° 在 100 nm 帧上只吃 2.5 pm 的 Z
    rig = TiltRig(surface=(0.001, 0.0), frame_m=1e-7)
    res = AutoTilt().execute(rig, {})
    assert res.success
    assert res.data["outcome"] == "no_action_needed"
    assert rig.tilt_writes == []


def test_the_same_tilt_triggers_on_a_large_frame_but_not_a_small_one():
    """统一判据是「这一帧吃掉多少 Z」—— 粗扫敏感、精扫宽容自动成立,
    不需要为粗扫/精扫各设一个角度阈值。"""
    _calibrated()
    tilt = 4.0          # 1 µm 帧上吃 99 nm > 5% 的 1.5 µm Z 量程(75 nm)

    small = TiltRig(surface=(tilt, 0.0))
    res_small = AutoTilt().execute(small, {"next_frame_m": 1e-8})
    assert res_small.data["outcome"] == "no_action_needed"

    big = TiltRig(surface=(tilt, 0.0))
    res_big = AutoTilt().execute(big, {"next_frame_m": 1e-6})
    assert res_big.data["outcome"] == "applied"


def test_a_slope_that_swamps_the_topography_triggers_even_within_z_budget():
    """这是要求的那句「扫出的图明显是倾斜的」。

    0.5° 在 1 µm 帧上只吃掉 0.8% 的 Z 量程(安全上毫无问题),但它在 15 pm 起伏
    的原子级平台上产生 12 nm 的斜坡 —— 形貌被彻底淹没。两个理由独立,取「或」。
    """
    _calibrated()
    quiet = TiltRig(surface=(0.5, 0.0), frame_m=1e-6)
    assert AutoTilt().execute(quiet, {}).data["outcome"] == "no_action_needed"

    rig = TiltRig(surface=(0.5, 0.0), frame_m=1e-6)
    res = AutoTilt().execute(rig, {"surface_rms_m": 15.4e-12})
    assert res.data["outcome"] == "applied"


def test_circle_residual_must_not_be_used_as_the_surface_roughness():
    """圆是特意跑在平地上的,它的残差按构造就是噪声。拿它当「表面起伏」会把
    形貌判据变成一个噪声判据 —— 于是几乎每一次都触发调平。

    这条用例钉住「surface_rms 必须由调用方从帧上量,不能从圆的残差里偷」。
    """
    _calibrated()
    rig = TiltRig(surface=(0.02, 0.0), frame_m=1e-7)
    res = AutoTilt().execute(rig, {})
    assert res.data["outcome"] == "no_action_needed", (
        "没给 surface_rms 却触发了 —— 检查是不是又把圆残差当形貌用了")


def test_force_compensates_even_within_budget():
    _calibrated()
    rig = TiltRig(surface=(0.001, 0.0))
    res = AutoTilt().execute(rig, {"force": True})
    assert res.data["outcome"] == "applied"


# ══════════════════════════════════════════════════════════════════════════
#  三、闭环收敛
# ══════════════════════════════════════════════════════════════════════════

def test_compensates_a_tilted_surface_and_verifies():
    _calibrated()
    rig = TiltRig(surface=(0.5, -0.3), frame_m=1e-6)
    res = AutoTilt().execute(rig, {"surface_rms_m": 15.4e-12})
    assert res.success, res.error
    assert res.data["outcome"] == "applied"
    # 施加的倾斜应该抵消掉表面倾斜
    assert rig.measured()[0] == pytest.approx(0.0, abs=0.02)
    assert rig.measured()[1] == pytest.approx(0.0, abs=0.02)


def test_axis_swapped_rig_is_handled_by_the_matrix():
    """一般情况 G ≈ ±单位阵,但轴交换的机器上按标量算就会把倾斜加到错误的轴。"""
    swap = ((0.0, 1.0), (1.0, 0.0))          # M:x 的一步出现在测到的 y 上
    _calibrated(g=((0.0, -1.0), (-1.0, 0.0)))   # G = -M⁻¹
    rig = TiltRig(surface=(0.6, 0.0), response=swap, frame_m=1e-6)
    res = AutoTilt().execute(rig, {"surface_rms_m": 15.4e-12})
    assert res.data["outcome"] == "applied"
    assert rig.measured()[0] == pytest.approx(0.0, abs=0.02)


def test_large_correction_is_applied_in_bounded_steps():
    """tilt 阶跃会让扫描平面突转 → Z 瞬态。小步是防撞针的硬要求。"""
    _calibrated()
    rig = TiltRig(surface=(3.0, 0.0), frame_m=1e-6)
    AutoTilt().execute(rig, {"surface_rms_m": 15.4e-12})
    steps = [abs(b[0] - a[0]) for a, b in
             zip([(0.0, 0.0)] + rig.tilt_writes, rig.tilt_writes)]
    assert steps, "一次都没写"
    assert max(steps) <= 1.0 + 1e-9, f"单步最大 {max(steps):.3f}°,超过 1° 上限"


def test_tilt_is_truncated_at_the_instrument_limit():
    # set_profile 整体替换会抹掉标定 —— 配置先写、标定后写。
    instrument_profile.set_profile({"tilt_limit_deg": 1.0})
    _calibrated()
    assert instrument_profile.get_tilt_calibration() is not None
    rig = TiltRig(surface=(4.0, 0.0), frame_m=1e-6)
    AutoTilt().execute(rig, {"surface_rms_m": 15.4e-12})
    assert all(abs(x) <= 1.0 + 1e-9 and abs(y) <= 1.0 + 1e-9
               for x, y in rig.tilt_writes)


# ══════════════════════════════════════════════════════════════════════════
#  四、发散与回滚
# ══════════════════════════════════════════════════════════════════════════

def test_diverging_loop_rolls_back_to_the_original_tilt():
    """回滚目标是**原始**倾斜,不是 0 —— 用户之前设的值是他的,不是我们的。

    标定符号反了(实际响应与标定相反)时会越调越歪,必须停下来还原。
    """
    _calibrated()                                  # 标定说 G = -I
    rig = TiltRig(surface=(1.0, 0.0), tilt=(0.4, 0.2),
                  gain_drift=-1.0, frame_m=1e-6)   # 实际响应反了
    res = AutoTilt().execute(rig, {"surface_rms_m": 15.4e-12})
    assert not res.success
    assert res.data["outcome"] == "rolled_back"
    assert res.data["reason"] == "diverged"
    assert rig.tilt == pytest.approx([0.4, 0.2])


def test_verify_measurement_failure_rolls_back():
    class FlakyRig(TiltRig):
        def run(self, skill_name, params, version=None):
            res = super().run(skill_name, params, version)
            if skill_name == "TiltProbeCircle" and self._n >= 2:
                return SkillResult(skill_name=skill_name, success=False,
                                   error="圆上跨过了台阶")
            return res

    _calibrated()
    rig = FlakyRig(surface=(1.0, 0.0), tilt=(0.3, 0.0), frame_m=1e-6)
    res = AutoTilt().execute(rig, {"surface_rms_m": 15.4e-12})
    assert res.data["outcome"] == "rolled_back"
    assert res.data["reason"] == "verify_failed"
    assert rig.tilt == pytest.approx([0.3, 0.0])


def test_hardware_reject_rolls_back_the_partial_correction():
    """写失败时前面的小步已经生效了 —— 把针尖留在走了一半的补偿上比不补偿更糟。"""
    class RejectAfterOne(TiltRig):
        """第一次写成功,第二次被硬件拒绝,之后(回滚那次)放行。"""

        def __init__(self, **kw):
            super().__init__(**kw)
            self._rejected = False

        def safe_call(self, method, *args, **kwargs):
            if (method == "Piezo_TiltSet" and len(self.tilt_writes) == 1
                    and not self._rejected):
                self._rejected = True
                self.calls.append((method, args))
                return NanonisCallRecord(method=method, args=args,
                                         error="tilt out of range")
            return super().safe_call(method, *args, **kwargs)

    _calibrated()
    rig = RejectAfterOne(surface=(3.0, 0.0), tilt=(0.1, 0.0), frame_m=1e-6)
    res = AutoTilt().execute(rig, {"surface_rms_m": 15.4e-12})
    assert res.data["outcome"] == "failed"
    assert res.data["reason"] == "hw_reject"
    assert rig.tilt == pytest.approx([0.1, 0.0]), "没有回到原始倾斜"


def test_first_measurement_failure_is_a_skip_not_a_write():
    _calibrated()
    rig = TiltRig(measure_error="圆上跨过了台阶,换一块更平的地方")
    res = AutoTilt().execute(rig, {})
    assert res.data["outcome"] == "skipped"
    assert res.data["reason"] == "measure_failed"
    assert res.data["next_action_hint"] == "survey_first"
    assert rig.tilt_writes == []


# ══════════════════════════════════════════════════════════════════════════
#  五、如实报告
# ══════════════════════════════════════════════════════════════════════════

def test_outcome_is_always_from_the_known_enum():
    known = {"applied", "no_action_needed", "skipped", "failed", "rolled_back"}
    _calibrated()
    for rig, params in [
        (TiltRig(surface=(0.001, 0.0)), {}),
        (TiltRig(surface=(0.5, 0.0), frame_m=1e-6), {"surface_rms_m": 15.4e-12}),
        (TiltRig(measure_error="x"), {}),
    ]:
        res = AutoTilt().execute(rig, params)
        assert res.data["outcome"] in known


def test_report_carries_the_thresholds_it_judged_by():
    """用户要能看出「为什么它说不用调」,而不是只看到一个结论。"""
    _calibrated()
    res = AutoTilt().execute(TiltRig(surface=(0.001, 0.0)), {})
    for key in ("trigger_z_span_m", "accept_z_span_m", "hard_limit_z_span_m",
                "frame_diagonal_m", "before", "measurement_resolution_deg"):
        assert key in res.data, f"报告缺 {key}"


def test_acceptance_threshold_is_never_below_the_measurement_resolution():
    """验收阈低于分辨率时,「残余倾斜没达标」只是在追噪声 —— 会白转三轮。"""
    # 注意顺序:set_profile 是**整体替换**,会连标定一起抹掉,所以配置先写、
    # 标定后写。(生产路径上写设置时会带着已持久化的标定一起回传,不会丢。)
    instrument_profile.set_profile({"z_range_m": 1e-9})
    _calibrated()
    rig = TiltRig(surface=(0.5, 0.0), frame_m=1e-6)
    res = AutoTilt().execute(rig, {"surface_rms_m": 15.4e-12})
    assert res.data.get("accept_raised_to_resolution") is True


def test_history_records_each_iteration():
    _calibrated()
    rig = TiltRig(surface=(3.0, 0.0), frame_m=1e-6)
    res = AutoTilt().execute(rig, {"surface_rms_m": 15.4e-12})
    hist = res.data.get("history") or []
    assert hist
    assert {"iteration", "applied_tilt", "residual_slope_deg"} <= set(hist[0])


# ══════════════════════════════════════════════════════════════════════════
#  六、TiltCalibrate
# ══════════════════════════════════════════════════════════════════════════

def test_calibrate_solves_an_identity_response():
    rig = TiltRig(surface=(0.0, 0.0), response=((1.0, 0.0), (0.0, 1.0)))
    res = TiltCalibrate().execute(rig, {})
    assert res.success, res.error
    g = res.data["matrix_g"]
    assert g[0][0] == pytest.approx(-1.0, abs=0.05)
    assert g[1][1] == pytest.approx(-1.0, abs=0.05)
    assert abs(g[0][1]) < 0.05 and abs(g[1][0]) < 0.05


def test_calibrate_solves_an_axis_swapped_response():
    rig = TiltRig(surface=(0.0, 0.0), response=((0.0, 1.0), (1.0, 0.0)))
    res = TiltCalibrate().execute(rig, {})
    assert res.success
    g = res.data["matrix_g"]
    assert abs(g[0][0]) < 0.05 and abs(g[1][1]) < 0.05
    assert g[0][1] == pytest.approx(-1.0, abs=0.05)


def test_calibrate_stores_the_matrix_in_the_profile():
    rig = TiltRig(surface=(0.0, 0.0))
    TiltCalibrate().execute(rig, {})
    stored = instrument_profile.get_tilt_calibration()
    assert stored is not None
    assert stored["g"][0][0] == pytest.approx(-1.0, abs=0.05)


def test_calibrate_restores_the_original_tilt():
    """标定是测量,不是设置 —— 结束时仪器必须回到原来的倾斜。"""
    rig = TiltRig(surface=(0.0, 0.0), tilt=(0.25, -0.15))
    TiltCalibrate().execute(rig, {})
    assert rig.tilt == pytest.approx([0.25, -0.15])


def test_calibrate_refuses_when_an_axis_does_not_respond():
    """没有响应的轴意味着接线不对或位置不够平 —— 存一个坏矩阵比不存更危险。"""
    rig = TiltRig(surface=(0.0, 0.0), response=((1.0, 0.0), (0.0, 0.0)))
    res = TiltCalibrate().execute(rig, {})
    assert not res.success
    assert "响应幅度" in res.error
    assert instrument_profile.get_tilt_calibration() is None


def test_calibrate_refuses_a_collinear_response():
    rig = TiltRig(surface=(0.0, 0.0), response=((1.0, 1.0), (1.0, 1.0)))
    res = TiltCalibrate().execute(rig, {})
    assert not res.success
    assert instrument_profile.get_tilt_calibration() is None


def test_profile_rejects_an_ill_conditioned_matrix():
    assert instrument_profile.set_tilt_calibration(
        [[1.0, 0.0], [0.0, 1.0]], cond=999.0) is None
    assert instrument_profile.get_tilt_calibration() is None


def test_profile_rejects_a_non_finite_matrix():
    # cond 显式传:这条测试的名字说的是**非有限矩阵**那道守卫。原来省略 cond,
    # 于是它的红/绿取决于两道守卫的先后顺序 —— 非有限那道若被删掉,它可能仍然
    # 因为「cond 缺失」而通过,测的就不是它名字里那件事了。
    # (顺带:它省略 cond 而长期通过,等于在测试套件里给「cond 可以不传」立了个
    # 证人 —— 一条通过的测试也会教人这样调用没问题。)
    assert instrument_profile.set_tilt_calibration(
        [[float("nan"), 0.0], [0.0, 1.0]], cond=1.0) is None
    assert instrument_profile.get_tilt_calibration() is None



# ── 条件数缺失必须被拒绝，不是被放行（v6.1.3, KNOWN_ISSUES）──────────────
#
# 闸门原来写作 `if cond is not None and cond > MAX` —— **未知即放行**，
# 也就是宣布「没有证据 = 没有问题」。而产生侧算不出条件数时兜底成 `1.0`
# （条件数的**最优值**），把闸门彻底解除。两侧一起修。


def test_missing_cond_is_a_TypeError_not_a_silent_write():
    """缺参数是**程序员错误**，要响。它和「条件数超限」是两类失败：
    后者返回 None（数据不合格，本函数的既有语义），前者应当当场炸。
    合并它们会在高一层重建「两个状态共用一个信号」。"""
    with pytest.raises(TypeError):
        instrument_profile.set_tilt_calibration([[1.0, 0.0], [0.0, 1.0]])
    assert instrument_profile.get_tilt_calibration() is None


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_unknown_cond_is_refused_not_waved_through(bad):
    """显式传 None/NaN/inf 仍然可能发生（必需参数拦不住 `cond=None`），
    所以闸门自己也要挡住「条件数未知」。**算不出条件数不是条件数良好的证据。**"""
    assert instrument_profile.set_tilt_calibration(
        [[1.0, 0.0], [0.0, 1.0]], cond=bad) is None
    assert instrument_profile.get_tilt_calibration() is None


def test_cond_field_is_never_stale():
    """`tilt_cal_cond` 以前只在 `cond is not None` 时更新 —— 于是 profile 里会留着
    **上一次**标定的条件数，配着**这一次**的矩阵。一个看起来有依据的数，描述的是
    另一个已经不在那里的矩阵，比没有更坏。"""
    assert instrument_profile.set_tilt_calibration(
        [[1.0, 0.0], [0.0, 1.0]], cond=2.0) is not None
    assert instrument_profile.get_tilt_calibration()["cond"] == 2.0
    # 第二次写入换一个条件数：字段必须跟着走。
    assert instrument_profile.set_tilt_calibration(
        [[2.0, 0.0], [0.0, 2.0]], cond=5.0) is not None
    assert instrument_profile.get_tilt_calibration()["cond"] == 5.0


def test_calibrate_fails_when_the_condition_number_cannot_be_computed(monkeypatch):
    """产生侧：算不出就让技能失败，**不写入、也不编数**。
    断言落在「profile 里什么都没有」，不是「返回了错误」——
    报错但仍然写入，等于没修。"""
    import numpy as np

    def boom(*_a, **_k):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "cond", boom)
    rig = TiltRig(surface=(0.0, 0.0), response=((1.0, 0.0), (0.0, 1.0)))
    res = TiltCalibrate().execute(rig, {})
    assert not res.success
    assert "条件数" in (res.error or "")
    assert instrument_profile.get_tilt_calibration() is None, "编不出数却写进去了"
    assert "1.00" not in (res.summary or ""), "把编造的条件数印给用户了"
