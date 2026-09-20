"""恒流内接圆测倾斜(fit_circle_tilt + TiltProbeCircle)。

这是 Nanonis SmarTilt 做法的自研版。与帧法的关键差别是**两个方向同样可信** ——
整圈几秒跑完,不像一帧图的慢扫轴那样跨越几十分钟、混进热漂移。

同样从纯噪声的误报开始:一个在平表面上"测出"倾斜的原语,会让系统去调一个不存在
的倾斜,而 tilt 是影响之后每一张图的全局硬件状态。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.builtins.tilt_probe import TiltProbeCircle
from mast.vision.tilt import CIRCLE_MIN_POINTS, fit_circle_tilt


NOISE_M = 20e-12  # 独立合成高斯噪声
RADIUS_M = 25e-9


def _circle(tilt_x_deg, tilt_y_deg, *, n=24, radius=RADIUS_M, sigma=NOISE_M,
            seed=0, drift_rate=0.0, dwell_s=0.2):
    """物理合成:恒流下沿圆走一圈读到的 Z(θ)。

    Z = slope_x·x + slope_y·y + C,(x, y) 在半径 radius 的圆上;可叠加线性热漂移。
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    sx = math.tan(math.radians(tilt_x_deg))
    sy = math.tan(math.radians(tilt_y_deg))
    times = np.arange(n) * dwell_s
    z = (sx * radius * np.cos(theta) + sy * radius * np.sin(theta)
         + drift_rate * times
         + rng.normal(0.0, sigma, size=n))
    return theta, z, times


# ══════════════════════════════════════════════════════════════════════════
#  一、纯噪声的误报(铁律)
# ══════════════════════════════════════════════════════════════════════════

def test_flat_surface_yields_a_tilt_within_the_measurement_resolution():
    """平表面上估计出的倾斜应位于解析测量分辨率允许的波动范围。

    合成例证采用 25 nm 半径、24 点、20 pm 高斯噪声；理论角度标准差由
    σ_z·√(2/n)/r 给出。检验重复抽样的最大误差，不要求带噪输入输出精确零值。
    """
    from mast.vision.tilt import circle_tilt_resolution_deg

    resolution = circle_tilt_resolution_deg(NOISE_M, RADIUS_M, 24)
    assert resolution == pytest.approx(0.013232, abs=0.0001)

    worst = 0.0
    for seed in range(40):
        theta, z, t = _circle(0.0, 0.0, seed=seed)
        fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t,
                              noise_floor_m=NOISE_M)
        assert fit.valid
        worst = max(worst, fit.slope_mag_deg)
    assert worst < 6 * resolution, (
        f"平表面上测出 {worst:.4f}°,超过 6× 分辨率 {resolution:.4f}°")


def test_measurement_resolution_improves_with_radius_and_points():
    """要测更小的倾斜,只能加大半径、加密取点或降噪 —— 这是算法之外的物理事实。

    调平的验收阈必须留在分辨率之上,否则「残余倾斜没达标」只是在追噪声。
    """
    from mast.vision.tilt import circle_tilt_resolution_deg

    base = circle_tilt_resolution_deg(NOISE_M, RADIUS_M, 24)
    assert circle_tilt_resolution_deg(NOISE_M, RADIUS_M * 4, 24) < base / 3
    assert circle_tilt_resolution_deg(NOISE_M, RADIUS_M, 96) < base / 1.9
    assert circle_tilt_resolution_deg(NOISE_M / 4, RADIUS_M, 24) < base / 3


def test_flat_surface_residual_stays_within_the_step_veto():
    """平表面的拟合残差必须低于台阶否决阈,否则真正平的地方也会被拒。

    实测平面上限:max|残差| 3.56σ、最大跳变 5.18σ。阈值分别是 4.5 与 7.0。
    """
    worst_peak = worst_jump = 0.0
    for seed in range(40):
        theta, z, t = _circle(0.0, 0.0, seed=seed)
        fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t,
                              noise_floor_m=NOISE_M)
        assert fit.valid, f"seed={seed} 被误判为有台阶"
        worst_peak = max(worst_peak, fit.max_residual_ratio)
        worst_jump = max(worst_jump, fit.max_jump_ratio)
    from mast.vision import tilt as _T
    assert worst_peak < _T.CIRCLE_MAX_RESIDUAL_SIGMA
    assert worst_jump < _T.CIRCLE_MAX_JUMP_SIGMA


# ══════════════════════════════════════════════════════════════════════════
#  二、倾斜回收
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tx,ty", [
    (0.1, 0.0), (0.0, 0.1), (0.3, 0.4), (-0.5, 0.2), (1.0, -1.0), (2.0, 0.0),
])
def test_recovers_both_axes(tx, ty):
    """两个方向都要准 —— 这正是圆相对帧法的全部意义。"""
    theta, z, t = _circle(tx, ty, seed=1)
    fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t, noise_floor_m=NOISE_M)
    assert fit.valid
    assert fit.tilt_x_deg == pytest.approx(tx, abs=0.02)
    assert fit.tilt_y_deg == pytest.approx(ty, abs=0.02)


def test_slope_magnitude_and_downhill_direction():
    theta, z, t = _circle(0.3, 0.0, seed=2)      # 沿 +x 上坡
    fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t, noise_floor_m=NOISE_M)
    assert fit.slope_mag_deg == pytest.approx(0.3, abs=0.02)
    # 上坡在 +x → 下坡在 −x → 180°
    assert fit.downhill_deg == pytest.approx(180.0, abs=5.0)


def test_larger_radius_gives_a_better_measurement():
    """半径越大,同样倾斜产生的 Z 起伏越高过噪声。"""
    small = fit_circle_tilt(*_circle(0.05, 0.0, radius=2e-9, seed=3)[:2],
                            2e-9, noise_floor_m=NOISE_M)
    big = fit_circle_tilt(*_circle(0.05, 0.0, radius=100e-9, seed=3)[:2],
                          100e-9, noise_floor_m=NOISE_M)
    assert abs(big.tilt_x_deg - 0.05) < abs(small.tilt_x_deg - 0.05)


def test_radius_conversion_is_applied():
    """A = slope·r —— 不除以半径就得不到角度。"""
    theta, z, _t = _circle(1.0, 0.0, radius=RADIUS_M, sigma=0.0, seed=4)
    right = fit_circle_tilt(theta, z, RADIUS_M)
    wrong = fit_circle_tilt(theta, z, RADIUS_M * 2)
    assert right.tilt_x_deg == pytest.approx(1.0, abs=0.01)
    assert math.tan(math.radians(wrong.tilt_x_deg)) == pytest.approx(
        math.tan(math.radians(right.tilt_x_deg)) / 2, rel=0.01)


# ══════════════════════════════════════════════════════════════════════════
#  三、漂移分离(圆闭合差的严格版)
# ══════════════════════════════════════════════════════════════════════════

def test_linear_drift_is_separated_from_tilt():
    """圆跑得快,但几秒里仍可能漂几十皮米。漂移在圆上是与 θ 无关、与时间线性
    相关的分量,正好能从倾斜里分离出来。

    漂移率本身的估计精度受噪声限制(σ/(std(t)·√n),这里约 23%),所以只要求
    它的**量级与符号**对,重点是倾斜没有被漂移带偏。
    """
    drift = 200e-12 / 5.0        # 5 秒漂 200 pm
    theta, z, t = _circle(0.3, 0.0, seed=5, drift_rate=drift, dwell_s=0.2)
    fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t, noise_floor_m=NOISE_M)
    assert fit.valid
    assert fit.tilt_x_deg == pytest.approx(0.3, abs=0.04)
    assert fit.drift_rate_m_s == pytest.approx(drift, rel=0.4)


def test_drift_biases_the_result_when_times_are_not_supplied():
    """不给时间戳就分不出漂移 —— 这条用例钉住「时间戳不是可选的装饰」。"""
    drift = 200e-12 / 5.0
    theta, z, t = _circle(0.0, 0.0, seed=6, drift_rate=drift, dwell_s=0.2)
    without = fit_circle_tilt(theta, z, RADIUS_M)
    with_t = fit_circle_tilt(theta, z, RADIUS_M, times_s=t)
    assert with_t.slope_mag_deg < without.slope_mag_deg


def test_slow_axis_is_trusted_unlike_the_frame_method():
    theta, z, t = _circle(0.2, 0.2, seed=7)
    fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t)
    assert fit.slow_axis_trusted is True


# ══════════════════════════════════════════════════════════════════════════
#  四、台阶否决(测量层面,比统计判据更直接)
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", range(6))
def test_a_step_crossing_the_circle_is_rejected(seed):
    """圆本来就该落在一块平地上。"""
    theta, z, t = _circle(0.2, 0.0, seed=seed)
    z = np.asarray(z, dtype=float)
    z[len(z) // 2:] += 240e-12          # 半圈落在高一个原子台阶的台面上
    fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t, noise_floor_m=NOISE_M)
    assert not fit.valid
    assert fit.invalid_reason == "residual_too_large"


def test_residual_rms_alone_would_have_missed_the_step():
    """钉住主判据的选择理由:台阶的**基频**会被正弦拟合吸收成一个假倾斜,
    RMS 残差里只剩高次谐波 —— 240 pm 台阶的 RMS 比只有 ~2.5,而平面能到 1.2,
    余量薄到不能用。真正有分辨力的是残差里的**不连续**。

    这条用例存在的意义是:将来若有人"简化"成只看 RMS,立刻会看到代价。
    """
    theta, z, t = _circle(0.2, 0.0, seed=8)
    z = np.asarray(z, dtype=float)
    z[len(z) // 2:] += 240e-12
    fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t, noise_floor_m=NOISE_M)

    from mast.vision import tilt as _T
    assert fit.residual_ratio < _T.CIRCLE_RESIDUAL_MAX_RATIO, (
        "前提变了:RMS 判据现在抓得到台阶了")
    assert fit.max_jump_ratio > _T.CIRCLE_MAX_JUMP_SIGMA


def test_half_atomic_step_is_honestly_below_the_detection_limit():
    """~120 pm 的跨圆台阶测不出来 —— 两个判据都落在平面涨落带里。

    把已知极限钉成用例,好过让它在真机上变成一次莫名其妙的错误调平。
    """
    hits = 0
    for seed in range(6):
        theta, z, t = _circle(0.2, 0.0, seed=seed)
        z = np.asarray(z, dtype=float)
        z[len(z) // 2:] += 120e-12
        if not fit_circle_tilt(theta, z, RADIUS_M, times_s=t,
                               noise_floor_m=NOISE_M).valid:
            hits += 1
    assert hits <= 1, "半个原子台阶居然稳定可测了?那要重新标定阈值"


def test_a_contamination_spike_is_rejected():
    theta, z, t = _circle(0.1, 0.0, seed=9)
    z = np.asarray(z, dtype=float)
    z[7] += 500e-12                     # 一颗吸附物
    fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t, noise_floor_m=NOISE_M)
    assert not fit.valid
    assert fit.invalid_reason == "residual_too_large"


def test_without_a_noise_floor_the_residual_veto_is_disabled():
    """噪声底未知时不能凭空否决 —— 但也要如实标注比值为 0(未评估)。"""
    theta, z, t = _circle(0.2, 0.0, seed=10)
    z = np.asarray(z, dtype=float)
    z[len(z) // 2:] += 240e-12
    fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t, noise_floor_m=0.0)
    assert fit.valid                    # 没有否决
    assert fit.residual_ratio == 0.0    # 但明说没评估


# ══════════════════════════════════════════════════════════════════════════
#  五、无效输入
# ══════════════════════════════════════════════════════════════════════════

def test_too_few_points_is_reported():
    theta, z, _t = _circle(0.1, 0.0, n=CIRCLE_MIN_POINTS - 1)
    fit = fit_circle_tilt(theta, z, RADIUS_M)
    assert not fit.valid and fit.invalid_reason == "too_few_points"


def test_bad_radius_is_reported():
    theta, z, _t = _circle(0.1, 0.0)
    assert fit_circle_tilt(theta, z, 0.0).invalid_reason == "bad_radius"


def test_shape_mismatch_is_reported():
    fit = fit_circle_tilt([0.0, 1.0, 2.0], [1.0, 2.0], RADIUS_M)
    assert not fit.valid and fit.invalid_reason == "shape_mismatch"


def test_nan_points_are_dropped_not_fatal():
    theta, z, t = _circle(0.3, 0.0, n=32, seed=11)
    z = np.asarray(z, dtype=float)
    z[3] = np.nan
    z[17] = np.nan
    fit = fit_circle_tilt(theta, z, RADIUS_M, times_s=t, noise_floor_m=NOISE_M)
    assert fit.valid
    assert fit.n_points == 30
    assert fit.tilt_x_deg == pytest.approx(0.3, abs=0.03)


# ══════════════════════════════════════════════════════════════════════════
#  六、硬件循环(TiltProbeCircle skill)
# ══════════════════════════════════════════════════════════════════════════

class FakeCtx:
    """按倾斜面回答 ZCtrl_ZPosGet 的假仪器。"""

    def __init__(self, tilt_x_deg=0.3, tilt_y_deg=0.0, *, frame_m=100e-9,
                 fail_moves=0, z_error=False):
        self.calls: list[tuple[str, tuple]] = []
        self.sx = math.tan(math.radians(tilt_x_deg))
        self.sy = math.tan(math.radians(tilt_y_deg))
        self.frame_m = frame_m
        self.pos = (0.0, 0.0)
        self._fail_moves = fail_moves
        self._z_error = z_error
        self._rng = np.random.default_rng(0)

    def safe_call(self, method, *args, **kwargs):
        self.calls.append((method, args))
        if method == "FolMe_XYPosSet":
            if self._fail_moves > 0:
                self._fail_moves -= 1
                return NanonisCallRecord(method=method, args=args,
                                         error="move failed")
            self.pos = (float(args[0]), float(args[1]))
            return NanonisCallRecord(method=method, args=args)
        if method == "FolMe_XYPosGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=["", b"", [self.pos[0], self.pos[1]]])
        if method == "Scan_FrameGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=["", b"", [0.0, 0.0, self.frame_m, self.frame_m, 0.0]])
        if method == "ZCtrl_ZPosGet":
            if self._z_error:
                return NanonisCallRecord(method=method, args=args,
                                         error="z read failed")
            z = (self.sx * self.pos[0] + self.sy * self.pos[1]
                 + float(self._rng.normal(0.0, NOISE_M)))
            return NanonisCallRecord(method=method, args=args,
                                     return_value=["", b"", [z]])
        return NanonisCallRecord(method=method, args=args)

    def check_abort(self) -> bool:
        return False


def _run(ctx, **params):
    params.setdefault("settle_s", 0.0)
    params.setdefault("noise_floor_m", NOISE_M)
    return TiltProbeCircle().execute(ctx, params)


def test_skill_measures_the_tilt_of_a_synthetic_surface():
    ctx = FakeCtx(tilt_x_deg=0.3, tilt_y_deg=-0.2)
    res = _run(ctx, n_points=24)
    assert res.success, res.error
    assert res.data["tilt_x_deg"] == pytest.approx(0.3, abs=0.03)
    assert res.data["tilt_y_deg"] == pytest.approx(-0.2, abs=0.03)


def test_skill_derives_the_radius_from_the_scan_frame():
    ctx = FakeCtx(frame_m=100e-9)
    res = _run(ctx, n_points=16)
    assert res.success
    assert res.data["radius_m"] == pytest.approx(100e-9 * 0.4)
    assert "geometry_note" in res.data


def test_skill_returns_the_tip_to_its_starting_position():
    """把针尖丢在圆周上某个随机角度,会让调用方之后的一切位置推理都错位。"""
    ctx = FakeCtx()
    _run(ctx, n_points=12, center_x_m=0.0, center_y_m=0.0)
    moves = [a for m, a in ctx.calls if m == "FolMe_XYPosSet"]
    assert moves, "一次移动都没有?"
    assert moves[-1][0] == pytest.approx(0.0, abs=1e-15)
    assert moves[-1][1] == pytest.approx(0.0, abs=1e-15)


def test_skill_does_not_write_any_tilt_setting():
    """测量就是测量 —— 闭环归 AutoTilt,这一步绝不动硬件的 tilt。"""
    ctx = FakeCtx()
    _run(ctx, n_points=12)
    assert not any(m.startswith("Piezo_Tilt") for m, _ in ctx.calls)


def test_skill_fails_loudly_when_too_many_points_are_lost():
    ctx = FakeCtx(z_error=True)
    res = _run(ctx, n_points=16)
    assert not res.success
    assert "有效点" in res.error


def test_skill_reports_a_step_crossing_as_an_explicit_failure():
    """跨台阶时给一个数字,下游会拿它去调硬件 —— 必须显式失败。"""
    class SteppedCtx(FakeCtx):
        def safe_call(self, method, *args, **kwargs):
            rec = super().safe_call(method, *args, **kwargs)
            if method == "ZCtrl_ZPosGet" and not rec.error:
                if self.pos[1] > 0:          # 上半圈落在高一个台阶的台面
                    rec.return_value[2][0] += 240e-12
            return rec

    res = _run(SteppedCtx(), n_points=24)
    assert not res.success
    assert "台阶" in res.error
    assert res.data["invalid_reason"] == "residual_too_large"


def test_skill_requires_feedback_on():
    """反馈关着走这一圈 = 针尖以固定高度扫过倾斜表面,轻则测不到,重则撞上去。"""
    meta = TiltProbeCircle().metadata()
    assert "z_controller_on" in meta.preconditions
    assert "scan_not_running" in meta.preconditions


def test_skill_is_confirm_not_dangerous():
    """半径受扫描框约束、反馈开着、每步都是小位移 —— 与任意大位移的
    MoveProbeXY(DANGEROUS)本质不同,与 MoveToXY 同级。"""
    from mast.core.types import SafetyLevel
    assert TiltProbeCircle().metadata().safety_level == SafetyLevel.CONFIRM
