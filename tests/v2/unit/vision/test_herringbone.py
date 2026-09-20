"""herringbone 判据使用解析构造的条纹、折臂、噪声和缺陷作为正反对照。

Au(111) 的通用重构周期取约6.3 nm；折臂模型保持垂直周期不变，
避免相位位移模型引入几何偏差。合成幅值、噪声、像素与视野独立指定。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mast.vision import herringbone as H

# ── 物理合成 ─────────────────────────────────────────────────────────────────

PERIOD_NM = 6.3                  # soliton 线对重复周期
CHEVRON_NM = 30.0                # zigzag 全周期
ARM_TILT_DEG = 16.0              # 折臂相对平均取向的倾角
CORRUGATION_PM = 24.0            # 基频峰峰起伏
FRAME_NM = 48.0                  # 独立选定的合成视野
PIXELS = 256
NMPP = FRAME_NM / PIXELS         # 0.1953 nm/px


def _herringbone(*, period_nm=PERIOD_NM, chevron_nm=CHEVRON_NM,
                 arm_tilt_deg=ARM_TILT_DEG, corrugation_pm=CORRUGATION_PM,
                 hcp_frac=0.30, angle_deg=30.0, n=PIXELS, nmpp=NMPP,
                 noise_pm=2.0, seed=0):
    """一帧 Au(111) herringbone 的高度图（米）。

    ``angle_deg`` 是**调制波矢 k** 相对快扫轴(+列)的方向:
    0° = 条纹垂直于快扫方向(安全几何),90° = 条纹平行于快扫方向(退化几何)。

    折臂:``φ = 2π(u·cosα + sinα·tri(v))/T``,``tri`` 是斜率 ±1、周期 L 的三角波。
    每条臂里 ``|∇φ| = 2π/T`` 精确成立 —— 这正是「垂直间距恒为 T」。
    """
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    xs, ys = x * nmpp, y * nmpp
    th, al = math.radians(angle_deg), math.radians(arm_tilt_deg)
    u = xs * math.cos(th) + ys * math.sin(th)
    v = -xs * math.sin(th) + ys * math.cos(th)
    L = float(chevron_nm)
    tri = (L / (2.0 * math.pi)) * np.arcsin(np.sin(2.0 * math.pi * v / L))
    phi = 2.0 * math.pi * (u * math.cos(al) + math.sin(al) * tri) / float(period_nm)
    a1 = 0.5 * float(corrugation_pm) * 1e-12          # 基频峰峰值 = corrugation_pm
    h = a1 * (np.cos(phi) + float(hcp_frac) * np.cos(2.0 * phi + math.pi / 3.0))
    if noise_pm:
        h = h + np.random.default_rng(seed).normal(0.0, noise_pm * 1e-12, h.shape)
    return h


def _noise_frame(*, n=PIXELS, sigma_pm=2.0, seed=0):
    return np.random.default_rng(seed).normal(0.0, sigma_pm * 1e-12, (n, n))


def _flat_terrace(*, n=PIXELS, tilt_pm=2.0, noise_pm=3.0, seed=0):
    """独立构造的倾斜平面与白噪声，作为没有重构条纹的对照。"""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    h = tilt_pm * 1e-12 * (x / n) + 0.5 * tilt_pm * 1e-12 * (y / n)
    return h + np.random.default_rng(seed).normal(0.0, noise_pm * 1e-12, h.shape)


def _row_streaks(*, n=PIXELS, offset_pm=20.0, noise_pm=2.0, seed=0):
    """逐行随机偏移 —— 1/f、蠕变、热漂移在慢轴上留下的条纹。

    **这是最重要的一个对照组**:它与「条纹平行于快扫方向的真 herringbone」
    在一帧图里没法分开,所以正确答案是 ``undetermined``,不是 ``absent``。
    """
    rng = np.random.default_rng(seed)
    h = np.repeat(rng.normal(0.0, offset_pm * 1e-12, (n, 1)), n, axis=1)
    return h + rng.normal(0.0, noise_pm * 1e-12, (n, n))


def _bandpass_noise(*, seed=0, period_nm=PERIOD_NM, n=PIXELS, nmpp=NMPP,
                    frac=0.15, amp_pm=20.0):
    """带通白噪声 —— 针尖抖动/反馈振铃造出的**准周期**条纹。

    与原子相判据那边同一个对照组的思路:它在窗里产生一个合格的谱峰,峰强度类判据
    一个都拦不住,只有「离散条纹峰 vs 弥散环」分得开。
    """
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.normal(0.0, 1.0, (n, n)))
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    fr = np.hypot(fy, fx)
    f0 = nmpp / float(period_nm)
    band = np.exp(-((fr - f0) ** 2) / (2.0 * (frac * f0) ** 2))
    out = np.real(np.fft.ifft2(F * band))
    return out / out.std() * (float(amp_pm) * 1e-12)


def _with_defects(h, *, n_def=25, depth_pm=60.0, r_px=3.0, seed=0):
    """撒一把点缺陷 —— **双针尖检测器只能靠非周期内容说话**。"""
    rng = np.random.default_rng(seed)
    out = h.copy()
    yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]]
    for _ in range(n_def):
        cy, cx = rng.integers(int(r_px * 3), h.shape[0] - int(r_px * 3), 2)
        out = out - (depth_pm * 1e-12) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * r_px ** 2))
    return out


def _ghost(h, dy, dx, weight=0.6):
    """双针尖 = 与自己的一个平移副本相加。"""
    return h + weight * np.roll(np.roll(h, dy, axis=0), dx, axis=1)


def _verdicts(frames):
    return [H.assess_herringbone(f, nm_per_px=NMPP).verdict for f in frames]


# ── 1. 误报:一帧都不许说「有 herringbone」 ─────────────────────────────────

def test_pure_noise_never_says_herringbone():
    """40 个种子的纯高斯白噪声。"""
    v = _verdicts([_noise_frame(seed=s) for s in range(40)])
    assert v.count("herringbone") == 0, f"纯噪声上出现误报：{v}"


def test_pure_noise_is_absent_not_undetermined():
    """**被否掉的那一版判据钉在这里。**

    早先的版本只看标准平场后的 k 指不指着慢轴,不看另一条路对不对得上。
    ``align_rows_mediandiff`` 在没有真实行偏移的帧上减掉的是一串中值差的**累积和**
    —— 那是一条随机游走,全部落在 kx=0 上,而且相干:实测带内 SNR 9.5..27.8
    (门是 4)、方向 89..91°。于是 30 个纯噪声帧里有 21 个被判成 ``undetermined``,
    给用户的话是「把扫描框转 ~30° 重扫」——**在一张什么都没有的图上**。

    正确答案是 ``absent``:这儿什么都没有。这条测试红了就说明那道
    「另一条路对不对得上」的核验又被绕过去了。
    """
    v = _verdicts([_noise_frame(seed=s) for s in range(30)])
    assert v.count("undetermined") == 0, (
        f"纯噪声被判成「判不了」（30 帧里 {v.count('undetermined')} 帧）——"
        f"那会让用户去转扫描框重扫一张空图")
    assert v.count("absent") == 30


def test_empty_terrace_never_says_herringbone():
    """只有平面与噪声的合成台面不应被判作存在 herringbone。"""
    v = _verdicts([_flat_terrace(seed=s) for s in range(20)])
    assert v.count("herringbone") == 0, f"空台面上出现误报：{v}"


def test_quasi_periodic_tip_ringing_never_says_herringbone():
    """**最难的对照组**:带通噪声在窗里有合格谱峰,但不是条纹。"""
    v = _verdicts([_bandpass_noise(seed=s) for s in range(30)])
    assert v.count("herringbone") == 0, f"准周期抖动被判成 herringbone：{v}"


def test_angular_concentration_separates_stripes_from_ringing():
    """判据的分离度必须留有余量 —— 这是沿用阈值 20 的实测依据。

    余量塌了要么是判据被改坏了,要么是合成变得不物理。
    """
    from mast.vision.atomic_phase import angular_concentration
    from mast.vision.seg_scale_adaptive import (
        DEFAULTS,
        detect_texture,
        flatten_robust,
    )

    band = (H.PERIOD_BAND_LO * PERIOD_NM, H.PERIOD_BAND_HI * PERIOD_NM)
    band_px = (band[0] / NMPP, band[1] / NMPP)

    def _conc(frame):
        flat = flatten_robust(frame)
        p = dict(DEFAULTS)
        p["atomic_band_nm"] = band
        p["lat_snr"] = 4.0
        if not detect_texture(flat, NMPP, p).get("atomic"):
            return None
        pk = H.stripe_peak(flat, band_px)
        return None if pk is None else angular_concentration(flat, pk[0])

    # 合成的低信噪比对照。
    worst = min(c for s in range(5) for a in (10.0, 30.0, 50.0)
                for c in (_conc(_herringbone(noise_pm=20.0, seed=s,
                                             angle_deg=a)),)
                if c is not None)
    best_ring = max(c for s in range(20)
                    for c in (_conc(_bandpass_noise(seed=s)),) if c is not None)
    assert worst > 20.0 > best_ring, (
        f"阈值 20 没有夹在两组之间：最差条纹 {worst:.1f}，最强抖动 {best_ring:.1f}")
    assert worst > best_ring * 5.0, (
        f"分离度只有 {worst / best_ring:.1f} 倍，余量不足")


# ── 2. 真条纹:必须认出来,而且报的数要对 ───────────────────────────────────

@pytest.mark.parametrize("angle_deg", [0.0, 20.0, 30.0, 45.0, 60.0])
def test_clean_herringbone_is_detected(angle_deg):
    res = H.assess_herringbone(_herringbone(angle_deg=angle_deg, seed=1),
                               nm_per_px=NMPP)
    assert res.verdict == "herringbone", f"漏检：{res.reasons}"
    assert res.passed is True
    assert res.scale == "full"
    assert res.period_nm == pytest.approx(PERIOD_NM, rel=0.10)
    assert res.angular_concentration >= 20.0
    assert res.stripe_corrugation_pm is not None


def test_period_is_not_pulled_onto_the_chevron_satellite():
    """``_band_peak`` 会锁到 zigzag 卫星上（实测 −1.1%..−7.2%），所以周期这个数
    由 :func:`stripe_peak` 给。直条纹(无 zigzag)上必须准到 1% 以内。"""
    from mast.vision.seg_scale_adaptive import flatten_robust

    band_px = (H.PERIOD_BAND_LO * PERIOD_NM / NMPP,
               H.PERIOD_BAND_HI * PERIOD_NM / NMPP)
    for angle in (0.0, 20.0, 40.0, 60.0):
        frame = _herringbone(angle_deg=angle, arm_tilt_deg=0.0, seed=1)
        t_px, _ang = H.stripe_peak(flatten_robust(frame), band_px)
        assert t_px * NMPP == pytest.approx(PERIOD_NM, rel=0.01), (
            f"θ={angle}° 上周期偏了：{t_px * NMPP:.3f} nm")


@pytest.mark.parametrize("angle_deg", [0.0, 20.0, 40.0, 60.0])
def test_direction_is_recovered(angle_deg):
    """直条纹上方向必须准。有 zigzag 时 argmax 落在**某一条臂**上（±16°），
    所以方向这个数的精度天生是 ``atan(1/N_periods)`` 加臂倾角 —— 见模块注释。"""
    res = H.assess_herringbone(
        _herringbone(angle_deg=angle_deg, arm_tilt_deg=0.0, seed=1),
        nm_per_px=NMPP)
    assert res.verdict == "herringbone"
    assert res.k_angle_deg == pytest.approx(angle_deg, abs=3.0)
    # 条纹走向 = 调制方向 + 90°
    assert res.stripe_angle_deg == pytest.approx((angle_deg + 90.0) % 180.0,
                                                 abs=3.0)


@pytest.mark.parametrize("noise_pm", [1.0, 2.0, 5.0, 10.0, 20.0])
def test_survives_realistic_noise(noise_pm):
    """起伏 20 pm、噪声 1..20 pm —— 20 pm 就是噪声与信号 1:1。"""
    res = H.assess_herringbone(_herringbone(noise_pm=noise_pm, seed=5),
                               nm_per_px=NMPP)
    assert res.verdict == "herringbone", f"噪声 {noise_pm} pm 下漏检：{res.reasons}"


@pytest.mark.parametrize("corrugation_pm", [5.0, 10.0, 20.0, 50.0])
def test_corrugation_is_reported_in_picometres(corrugation_pm):
    """``stripe_corrugation_pm`` 报的是**带通基频分量的峰峰值**。

    合成侧把基频振幅直接设成 ``corrugation_pm/2``,所以真值就是 ``corrugation_pm``。
    实测残差 ~3%:zigzag 把一部分功率挪到了 ±35% 窗外的卫星上(直条纹上只有
    −0.5%),这是**物理**不是判据偏差。
    """
    res = H.assess_herringbone(
        _herringbone(corrugation_pm=corrugation_pm, noise_pm=1.0, seed=1),
        nm_per_px=NMPP)
    assert res.stripe_corrugation_pm == pytest.approx(corrugation_pm, rel=0.08)


def test_a_step_in_the_frame_inflates_corrugation_and_is_flagged():
    """局部台阶会抬高全帧起伏估计，不能直接当成遍布全帧的条纹起伏。
    测试要求起伏读数同时携带空间均匀性证据。"""
    clean = _herringbone(angle_deg=30.0, seed=1)
    res_clean = H.assess_herringbone(clean, nm_per_px=NMPP)
    assert res_clean.corrugation_quadrant_spread is not None
    assert res_clean.corrugation_quadrant_spread < H.CORRUGATION_SPREAD_MAX
    assert "corrugation_inhomogeneous" not in res_clean.warnings

    # 在解析条纹上叠加一道斜切整帧的单原子台阶边。
    # 检验条纹判决仍可成立，而局部台阶污染幅值时必须同时给出非均匀警告。
    y, x = np.mgrid[0:PIXELS, 0:PIXELS].astype(np.float64)
    stepped = clean.copy()
    stepped[y > 0.62 * PIXELS + 0.25 * (x - PIXELS / 2)] += 236e-12
    res_step = H.assess_herringbone(stepped, nm_per_px=NMPP)
    assert res_step.verdict == "herringbone", (
        f"这条测试要测的是「认得出条纹但起伏被污染」，而它 {res_step.reasons}")
    assert res_step.corrugation_quadrant_spread > H.CORRUGATION_SPREAD_MAX, (
        f"台阶没有把散布抬起来：{res_step.corrugation_quadrant_spread}")
    assert "corrugation_inhomogeneous" in res_step.warnings
    # 而且台阶确实把那个数抬高了 —— 这正是要警告的理由
    assert res_step.stripe_corrugation_pm > res_clean.stripe_corrugation_pm * 1.5


def test_a_stronger_out_of_band_period_is_flagged_not_hidden():
    """``h2 > 1`` = 2k 处功率比 k 处还强 ⇒ **这帧最强的周期性判据没在报**。

    窗的构造保证 ``band_hi ≤ 2·band_lo``,所以 2k 的周期必然在窗外 —— 也就是
    ``period_nm`` 可能只报次要成分；测试注入窗外谐波来核验这个诊断。
    不标出来,读的人会以为报的那个就是这帧的主旋律。
    """
    # 二次谐波压过基频:hcp_frac > 1 就是「线对比周期本身更显眼」
    strong_h2 = _herringbone(hcp_frac=2.5, angle_deg=30.0, noise_pm=1.0, seed=1)
    res = H.assess_herringbone(strong_h2, nm_per_px=NMPP)
    assert res.second_harmonic_ratio > 1.0, res.second_harmonic_ratio
    assert "dominant_period_outside_band" in res.warnings

    normal = _herringbone(hcp_frac=0.30, angle_deg=30.0, noise_pm=1.0, seed=1)
    res_n = H.assess_herringbone(normal, nm_per_px=NMPP)
    assert res_n.second_harmonic_ratio < 1.0
    assert "dominant_period_outside_band" not in res_n.warnings


def test_corrugation_scales_with_the_real_corrugation():
    """针尖变钝 = 起伏变小。这个数必须**单调**跟着走,否则它当不了验收量。"""
    vals = [H.assess_herringbone(
        _herringbone(corrugation_pm=c, noise_pm=1.0, seed=1),
        nm_per_px=NMPP).stripe_corrugation_pm for c in (4.0, 8.0, 16.0, 32.0)]
    assert all(b > a * 1.5 for a, b in zip(vals, vals[1:])), vals


# ── 3. 尺度门:判不了要说判不了 ─────────────────────────────────────────────

def test_atomic_scale_frame_refuses_rather_than_reporting_absence():
    """5 nm / 256 px —— 原子相判据的档位。帧里只装得下 0.79 个周期。

    这正是本判据要填的那个空档的**另一侧**:``AssessAtomicPhase`` 在 0.1953 nm/px
    上拒判,本判据在 0.0195 nm/px 上拒判。两个判据各自说「判不了」,而不是各自
    说「没有」。
    """
    res = H.assess_herringbone(_herringbone(seed=1), nm_per_px=5.0 / 256)
    assert res.verdict == "undetermined"
    assert res.scale == "off"
    assert res.reasons == ("scale_gate",)
    assert res.periods_in_frame == pytest.approx(0.79, abs=0.02)


def test_coarse_pixels_refuse_rather_than_reporting_absence():
    """1000 nm / 256 px = 3.9 nm/px —— 一个周期只有 1.6 px,复用的机器根本不动。"""
    res = H.assess_herringbone(_herringbone(seed=1), nm_per_px=1000.0 / 256)
    assert res.verdict == "undetermined"
    assert res.scale == "off"
    assert res.reasons == ("scale_gate",)


def test_unknown_pixel_size_refuses_to_judge():
    res = H.assess_herringbone(_herringbone(seed=1), nm_per_px=None)
    assert res.verdict == "undetermined"
    assert res.reasons == ("unknown_pixel_size",)
    assert res.scale is None


def test_scale_gate_boundaries():
    """两个条件独立,而且它们的下限正好让搜索窗不被 detect_texture 内部夹住。"""
    shape = (256, 256)
    # 帧尺寸(周期数)那一侧 —— 硬底 5.35 个周期
    assert H.scale_gate(30.0 / 256, shape, PERIOD_NM)[0] == "off"       # 4.76 个
    assert H.scale_gate(33.0 / 256, shape, PERIOD_NM)[0] == "off"       # 5.24 个
    assert H.scale_gate(34.0 / 256, shape, PERIOD_NM)[0] == "reduced"   # 5.40 个
    assert H.scale_gate(36.0 / 256, shape, PERIOD_NM)[0] == "reduced"   # 5.71 个
    assert H.scale_gate(40.0 / 256, shape, PERIOD_NM)[0] == "full"      # 6.35 个
    assert H.scale_gate(50.0 / 256, shape, PERIOD_NM)[0] == "full"      # 7.94 个
    # 像素粗细那一侧
    assert H.scale_gate(2.5, shape, PERIOD_NM)[0] == "off"              # 2.5 px/T
    assert H.scale_gate(1.5, shape, PERIOD_NM)[0] == "reduced"          # 4.2 px/T
    assert H.scale_gate(0.9, shape, PERIOD_NM)[0] == "full"             # 7.0 px/T
    assert H.scale_gate(None, shape, PERIOD_NM)[0] is None
    assert H.scale_gate(0.0, shape, PERIOD_NM)[0] is None

    # 满权重档的两个下限 = 「窗完整存在」（detect_texture 内部的两道夹子）:
    #   下沿不被 max(2.5, ·) 夹  ⟺ PERIOD_BAND_LO × px_per_period ≥ 2.5
    #   上沿不被 min(N/4, ·) 夹  ⟺ PERIOD_BAND_HI × T/nmpp ≤ N/4
    #                            ⟺ periods_in_frame ≥ 4 × PERIOD_BAND_HI
    assert H.PX_PER_PERIOD_FULL * H.PERIOD_BAND_LO >= 2.5
    assert H.PERIODS_IN_FRAME_FULL >= 4.0 * H.PERIOD_BAND_HI


def test_periods_floor_is_where_angular_concentration_stops_working():
    """``PERIODS_IN_FRAME_OFF`` 是从 ``angular_concentration`` 自己的守卫解出来的
    (``ring_px < n_bins → return 0.0``),不是拍出来的。

    环的半径(像素)恰好 = 帧里的周期数。这条测试直接量那个环:低于下限时判据
    **静默返回 0**,于是不论图上有什么都会落进 ``not_a_stripe_pattern``。
    这正是「守卫把一切都拒了,而判据只朝一侧失败,所以看不出来」那一类。
    """
    from mast.vision.atomic_phase import _ring_mask, angular_concentration

    assert H.PERIODS_IN_FRAME_OFF == pytest.approx(5.353, abs=0.01)

    n = 256
    for periods, expect_alive in ((H.PERIODS_IN_FRAME_OFF - 0.4, False),
                                  (H.PERIODS_IN_FRAME_OFF + 0.4, True)):
        t_px = n / periods                      # 环半径 f0 = N/T_px = periods
        sel, _ = _ring_mask((n, n), n / t_px)
        assert (int(sel.sum()) >= 72) is expect_alive, (
            f"{periods:.2f} 个周期时环上有 {int(sel.sum())} 个像素")
        # 一帧真的有这个周期的条纹,集中度是不是真的塌成 0
        y, x = np.mgrid[0:n, 0:n].astype(np.float64)
        stripes = 1e-11 * np.cos(2.0 * math.pi * x / t_px)
        conc = angular_concentration(stripes, t_px)
        assert (conc > 0.0) is expect_alive, (
            f"{periods:.2f} 个周期上集中度 = {conc}")


def test_below_the_periods_floor_allow_reduced_scale_cannot_fake_a_verdict():
    """**被否掉的那一版下限(4.0)钉在这里。**

    5.08 个周期的帧里集中度静默返回 0 ⇒ 每一帧都 ``not_a_stripe_pattern``。
    下限写成 4.0 时,配上 ``allow_reduced_scale=True`` 就会**稳定地**报
    ``absent`` —— 不论图上画的是什么。现在它落在 ``off`` 档,拒判。
    """
    nmpp = 32.0 / 256                           # 5.08 个周期
    frame = _herringbone(nmpp=nmpp, n=256, seed=1)
    for allow in (False, True):
        res = H.assess_herringbone(frame, nm_per_px=nmpp,
                                   allow_reduced_scale=allow)
        assert res.scale == "off"
        assert res.verdict == "undetermined"
        assert res.reasons == ("scale_gate",)


def test_reduced_scale_is_undetermined_not_absent():
    """过渡带的**测量本身**是降级的,所以它既撑不住「有」也撑不住「没有」。

    判成 ``absent`` 会让 forge 据此接着扰动针尖 —— 而该做的是换个视野重扫。
    """
    nmpp = 36.0 / 256                     # 5.71 个周期 → reduced
    res = H.assess_herringbone(
        _herringbone(nmpp=nmpp, n=256, seed=1), nm_per_px=nmpp)
    assert res.scale == "reduced"
    assert res.verdict == "undetermined"
    assert "scale_reduced" in res.reasons
    # 降级档里量到的数照报 —— 「判不了」不等于一片空白。
    assert res.period_nm is not None
    res2 = H.assess_herringbone(
        _herringbone(nmpp=nmpp, n=256, seed=1), nm_per_px=nmpp,
        allow_reduced_scale=True)
    assert res2.verdict == "herringbone"
    assert "scale_reduced" in res2.warnings


def test_chevron_is_declared_unchecked_not_absent():
    """zigzag 在 50 nm 帧上物理不可分(Δk = 50/30 = 1.7 px,落在峰宽里面)。

    判据**没查**它 —— 这件事要作为警告说出来,而不是让读的人以为查过了。
    """
    res = H.assess_herringbone(_herringbone(seed=1), nm_per_px=NMPP)
    assert "chevron_not_resolvable" in res.warnings
    assert res.chevron_min_frame_nm == pytest.approx(180.0)


# ── 4. 慢轴退化:判据自己会把这个几何的信号吃掉 ─────────────────────────────

@pytest.mark.parametrize("angle_deg", [85.0, 88.0, 90.0, 92.0])
def test_stripes_along_the_fast_axis_refuse_rather_than_report_absence(angle_deg):
    """条纹平行于快扫方向 ⇒ 两重退化叠在一起:

    1. ``flatten_robust`` 里的 ``align_rows_mediandiff`` 会把它当行偏移减掉;
    2. 扫描线噪声产生的就是同一个几何,一帧图里分不开。

    所以答案必须是「判不了 + 转 30° 重扫」,不能是「这儿没有 herringbone」——
    后者会让 forge 接着去扎一根其实没问题的针。
    """
    res = H.assess_herringbone(
        _herringbone(angle_deg=angle_deg, arm_tilt_deg=0.0, seed=1),
        nm_per_px=NMPP)
    assert res.verdict == "undetermined", f"θ={angle_deg}° 被判成 {res.verdict}"
    assert "stripes_along_fast_axis" in res.reasons
    assert "转" in res.notes.get("slow_axis", "")


def test_row_streaks_are_undetermined_not_absent():
    """真实的扫描线伪影 —— 与上一条是同一个几何,所以答案也必须一样。"""
    v = _verdicts([_row_streaks(seed=s) for s in range(30)])
    assert v.count("herringbone") == 0
    assert v.count("undetermined") == 30, f"行偏移条纹的判定分布：{set(v)}"


def test_row_free_band_snr_is_not_a_scratch_index():
    """row_free_band_snr 不包含方向信息；强条纹本身也能产生高分。
    合成无慢轴分量的 herringbone 应同时给出高窗内信噪比与低慢轴功率比例。"""
    clean = _herringbone(angle_deg=30.0, arm_tilt_deg=0.0, noise_pm=1.0, seed=1)
    res = H.assess_herringbone(clean, nm_per_px=NMPP)
    assert res.verdict == "herringbone"
    assert res.row_free_band_snr > 20.0, "干净鱼骨的 raw SNR 本来就高"
    assert res.slow_axis_power_ratio is not None
    assert res.slow_axis_power_ratio < 0.3, (
        f"一帧没有慢轴分量的图，方向性占比却有 {res.slow_axis_power_ratio:.3f}")


def test_slow_axis_power_ratio_is_directional():
    """带方向的那个数:条纹转到慢轴上时它必须涨到接近 1。"""
    ratios = {}
    for angle in (0.0, 30.0, 60.0, 80.0, 90.0):
        r = H.assess_herringbone(
            _herringbone(angle_deg=angle, arm_tilt_deg=0.0, seed=1),
            nm_per_px=NMPP)
        ratios[angle] = r.slow_axis_power_ratio
    assert all(v is not None for v in ratios.values()), ratios
    assert ratios[90.0] > 0.9, f"条纹就在慢轴上，占比却只有 {ratios[90.0]}"
    assert ratios[0.0] < 0.3 and ratios[30.0] < 0.3, ratios
    # 单调:越靠近慢轴越大
    assert ratios[0.0] <= ratios[60.0] <= ratios[80.0] <= ratios[90.0], ratios


def test_slow_axis_power_ratio_must_be_measured_before_row_alignment():
    """**这个数必须在不做行对齐的图上算,否则它对划痕完全失明。**

    ``align_rows_mediandiff`` 的整个用途就是杀掉慢轴条纹 —— 在它之后再问「慢轴上
    有多少功率」,答案恒等于「没有」。合成实测(安全几何 θ=30 加一族真实行偏移
    划痕,划痕幅度 10/20/40 pm):

        不做行对齐   占比 0.031 → 0.126 → 0.504   随划痕单调上升
        标准平场后   占比 0.000 → 0.000 → 0.000   **一个都看不见**

    上面那条「θ=90 转到慢轴」的测试拦不住这个错:那个几何下标准平场既杀掉了真条纹、
    又注入了自己的 kx=0 随机游走,两条路的占比都是 1.0,分不出来。要分出来必须让
    主条纹待在安全方向上,只把划痕放到慢轴。
    """
    base = _herringbone(angle_deg=30.0, arm_tilt_deg=0.0, noise_pm=1.0, seed=1)
    rng = np.random.default_rng(7)
    ratios, verdicts = [], []
    for amp_pm in (0.0, 10.0, 20.0, 40.0):
        scratch = np.repeat(
            rng.normal(0.0, amp_pm * 1e-12, (PIXELS, 1)), PIXELS, axis=1)
        r = H.assess_herringbone(base + scratch, nm_per_px=NMPP)
        ratios.append(r.slow_axis_power_ratio)
        verdicts.append(r.verdict)
    assert all(v is not None for v in ratios), ratios
    # 划痕轻的时候照常判;重到能跟主条纹相比时,判据自己会拒判（那是设计,不是漏检）
    # —— 而**拒判时这个数照样要给**，否则读的人不知道是为什么被拒的。
    assert verdicts[:3] == ["herringbone"] * 3, verdicts
    assert verdicts[3] == "undetermined", verdicts
    # 单调上升，而且最强的那一档必须明显离开 0 —— 在标准平场上算的话全是 0.0。
    assert ratios == sorted(ratios), f"划痕加重时占比没有跟着涨：{ratios}"
    assert ratios[-1] > 0.25, (
        f"40 pm 划痕只让占比到 {ratios[-1]:.3f} —— 这个数大概率是在"
        f"行对齐之后算的，那里划痕已经被减掉了")
    assert ratios[0] < 0.05, f"没有划痕时就有 {ratios[0]:.3f}"


def test_safe_geometry_is_not_refused():
    """离慢轴够远的几何不该被这道缺口误伤。"""
    for angle in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0):
        res = H.assess_herringbone(
            _herringbone(angle_deg=angle, arm_tilt_deg=0.0, seed=1),
            nm_per_px=NMPP)
        assert res.verdict == "herringbone", (
            f"θ={angle}° 被 {res.reasons} 误伤")


# ── 5. 搜索窗:上沿 ≤ 2 × 下沿 ─────────────────────────────────────────────

def test_band_that_admits_the_second_harmonic_refuses():
    """``_band_peak`` 在强峰里取**周期最小**的那个。窗里同时装得下 T 与 T/2 时,
    它会稳定地报出半个周期 —— 那比拒判坏得多。"""
    res = H.assess_herringbone(_herringbone(seed=1), nm_per_px=NMPP,
                               period_band_nm=(3.0, 12.0))    # 12 > 2×3
    assert res.verdict == "undetermined"
    assert res.reasons == ("band_admits_harmonic",)
    # 刚好 2× 是允许的
    ok = H.assess_herringbone(_herringbone(seed=1), nm_per_px=NMPP,
                              period_band_nm=(4.41, 8.82))
    assert ok.verdict == "herringbone"


def test_default_band_excludes_the_second_harmonic():
    assert H.PERIOD_BAND_HI <= 2.0 * H.PERIOD_BAND_LO


def test_peak_at_the_band_edge_is_flagged():
    """搜索被窗截断这件事要看得见,不能悄悄报一个贴着窗沿的数。"""
    res = H.assess_herringbone(
        _herringbone(period_nm=PERIOD_NM * 1.38, seed=1), nm_per_px=NMPP)
    assert "period_at_band_edge" in res.warnings


# ── 6. 双针尖 ───────────────────────────────────────────────────────────────

def test_double_tip_on_a_pure_sinusoid_is_unidentifiable():
    """**定理,不是实现缺陷。**

    双针尖是卷积 ``I = I_true ∗ (δ + a·δ_d)``,频域 ``Î(k)·(1 + a·e^{-ik·d})``。
    对纯单 k 的周期信号,这只是把那**一个** Bragg 系数乘上一个复数 —— 结果仍是
    同周期的纯正弦,谱上不产生任何新峰。

    所以:任何只吃周期分量的统计量都分不开单针尖与双针尖。这条测试直接证明
    「双针尖后的纯正弦」逐点等于「另一个振幅相位的纯正弦」。
    """
    n = PIXELS
    t_px = PERIOD_NM / NMPP
    _yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    a, shift = 1e-11, t_px / 6.0
    single = a * np.cos(2.0 * math.pi * xx / t_px)
    ghosted = single + 0.6 * a * np.cos(2.0 * math.pi * (xx - shift) / t_px)

    # 两个同频余弦之和 = 一个同频余弦。解析地把它写出来:
    #   cos θ + b·cos(θ − φ) = (1 + b cos φ)·cos θ + (b sin φ)·sin θ
    #                        = A·cos(θ − ψ),  ψ = atan2(b sin φ, 1 + b cos φ)
    ph = 2.0 * math.pi * shift / t_px
    amp = a * math.hypot(1.0 + 0.6 * math.cos(ph), 0.6 * math.sin(ph))
    psi = math.atan2(0.6 * math.sin(ph), 1.0 + 0.6 * math.cos(ph))
    equivalent = amp * np.cos(2.0 * math.pi * xx / t_px - psi)
    assert np.abs(ghosted - equivalent).max() < 1e-24, (
        "双针尖后的纯正弦不是纯正弦 —— 那这条定理的前提就写错了")

    # ⇒ 判据里所有周期类的量必须一致(振幅除外 —— 它按 |1+a·e^{-ik·d}| 变)。
    r1 = H.assess_herringbone(single, nm_per_px=NMPP)
    r2 = H.assess_herringbone(ghosted, nm_per_px=NMPP)
    assert r1.period_nm == pytest.approx(r2.period_nm, rel=1e-3)
    assert r1.second_harmonic_ratio == pytest.approx(r2.second_harmonic_ratio,
                                                     abs=1e-6)
    # 起伏**会**变 —— 双针尖可以让它变大。所以起伏本身也不是干净的针尖判据。
    assert r2.stripe_corrugation_pm > r1.stripe_corrugation_pm * 1.2


def test_double_tip_score_separates_once_there_is_aperiodic_content():
    """有缺陷时 ``detect_double_tip`` 分得开 —— 合成实测的分隔度钉在这里。"""
    base = _with_defects(_herringbone(angle_deg=30.0, seed=1), seed=2)
    single = H.assess_herringbone(base, nm_per_px=NMPP).double_tip_score
    doubles = [H.assess_herringbone(_ghost(base, dy, dx, w),
                                    nm_per_px=NMPP).double_tip_score
               for dy, dx, w in ((8, 10, 0.6), (12, 16, 0.6), (5, 7, 0.9))]
    assert single is not None and all(d is not None for d in doubles)
    assert min(doubles) > single * 2.0, (
        f"单针尖 {single:.3f} vs 双针尖 {doubles} —— 分不开")


def test_double_tip_answer_does_not_depend_on_the_unit_the_height_is_in():
    """同一合成表面用不同高度单位表达时，归一化判据应给出一致结果。
    防除零守卫不得依赖绝对米数值，否则会错误排除小幅值有效信号。"""
    from mast.vision.double_tip import detect_double_tip

    base = _with_defects(_herringbone(angle_deg=30.0, seed=1), seed=2)
    ghosted = _ghost(base, 8, 10)
    assert float(ghosted.std()) < 1e-9           # 前提：合成信号低于旧绝对幅值守卫

    in_m = detect_double_tip(ghosted, nm_per_px=NMPP)                  # 米
    in_nm = detect_double_tip(ghosted * 1e9, nm_per_px=NMPP)           # 纳米
    unitless = detect_double_tip(ghosted / ghosted.std(), nm_per_px=NMPP)

    assert in_m.verdict == in_nm.verdict == unitless.verdict
    assert in_m.score == pytest.approx(unitless.score, rel=1e-3)
    assert in_m.score == pytest.approx(in_nm.score, rel=1e-3)
    assert in_m.score > 0.0                      # 不再是那个假的 0
    # 判据自己那条路照样报得出数
    assert H.assess_herringbone(ghosted, nm_per_px=NMPP).double_tip_score > 0.0


def test_aperiodic_fraction_says_when_the_double_tip_number_can_speak():
    """无特征帧上 ``double_tip_detected=False`` 的意思是「没有证据」。

    ``aperiodic_fraction`` 就是那个前提:它必须能把「有缺陷」与「一片干净条纹」
    分开,否则读结果的人无从知道那个 False 值不值钱。
    """
    clean = _herringbone(angle_deg=30.0, noise_pm=0.5, seed=1)
    defected = _with_defects(clean, seed=2)
    a_clean = H.assess_herringbone(clean, nm_per_px=NMPP).aperiodic_fraction
    a_def = H.assess_herringbone(defected, nm_per_px=NMPP).aperiodic_fraction
    assert a_clean is not None and a_def is not None
    assert a_def > a_clean * 1.5, (
        f"非周期内容占比分不开：干净 {a_clean:.3f} vs 有缺陷 {a_def:.3f}")


# ── 7. 复用的那几件东西必须真的是同一件 ─────────────────────────────────────

def test_bandpass_matches_seg_scale_adaptive():
    """``bandpass`` 的掩膜与 ``seg_scale_adaptive.bandpass_envelope`` 逐字相同。

    那个文件冻结成「与验证过的原型逐行一致」,所以不在那里加辅助函数;代价是这里
    有一份拷贝 —— 这条测试让那份拷贝**自己检查自己**。
    """
    from scipy import ndimage as ndi

    from mast.vision.seg_scale_adaptive import bandpass_envelope, flatten_robust

    flat = flatten_robust(_herringbone(seed=3))
    for t_px in (20.0, 32.0, 45.0):
        mine = ndi.gaussian_filter(np.abs(H.bandpass(flat, t_px)),
                                   min(1.5 * t_px, 30.0))
        theirs = bandpass_envelope(flat, t_px, 1.5)
        assert np.allclose(mine, theirs, rtol=1e-9, atol=0.0), (
            f"T={t_px} px 上两份带通不一致 —— 拷贝漂移了")


def test_concentration_comes_from_atomic_phase_not_a_second_copy():
    """集中度那个**数**只有一份实现(``atomic_phase.angular_concentration``)。

    本模块只多要一个方向(``stripe_peak``),不重写判据本身。
    """
    from mast.vision.atomic_phase import angular_concentration
    from mast.vision.seg_scale_adaptive import flatten_robust

    frame = _herringbone(angle_deg=30.0, seed=1)
    res = H.assess_herringbone(frame, nm_per_px=NMPP)
    flat = flatten_robust(frame)
    band_px = (H.PERIOD_BAND_LO * PERIOD_NM / NMPP,
               H.PERIOD_BAND_HI * PERIOD_NM / NMPP)
    t_px, _ = H.stripe_peak(flat, band_px)
    assert res.angular_concentration == angular_concentration(flat, t_px)


def test_frame_validity_is_the_shared_one():
    """死平帧走的是 ``vision.frame_validity.judge_frame`` —— 与
    ``AssessTipSharpness`` / ``AssessClusterRoundness`` / ``FindFlatRegion`` /
    ``CheckLineQuality`` **同一份**判据,不是第二套 ``std <= 0``。"""
    dead = np.zeros((PIXELS, PIXELS))
    res = H.assess_herringbone(dead, nm_per_px=NMPP)
    assert res.verdict == "undetermined"
    assert res.reasons == ("frame_unusable",)
    assert "不是针尖的问题" in res.notes.get("frame_unusable", "")
    # 倾斜的死平帧(浮点舍入那一档)也要被同一份判据拦下
    y, x = np.mgrid[0:PIXELS, 0:PIXELS].astype(np.float64)
    tilted = 1e-9 * (x + 0.3 * y)
    assert H.assess_herringbone(tilted, nm_per_px=NMPP).reasons == (
        "frame_unusable",)


# ── 8. 纯函数纪律 ───────────────────────────────────────────────────────────

def test_never_raises_on_garbage():
    for bad in (np.zeros((4, 4)), np.full((64, 64), np.nan),
                np.zeros((PIXELS, PIXELS)), np.ones((PIXELS, PIXELS)),
                np.full((PIXELS, PIXELS), np.inf)):
        res = H.assess_herringbone(bad, nm_per_px=NMPP)
        assert res.verdict in ("absent", "undetermined")
        assert res.passed is False


def test_result_is_frozen():
    res = H.assess_herringbone(_herringbone(seed=1), nm_per_px=NMPP)
    with pytest.raises(Exception):
        res.verdict = "absent"           # type: ignore[misc]


def test_thresholds_are_parameters_not_globals():
    frame = _herringbone(seed=1)
    assert H.assess_herringbone(frame, nm_per_px=NMPP).verdict == "herringbone"
    assert H.assess_herringbone(frame, nm_per_px=NMPP,
                                concentration_min=1e9).verdict == "absent"
    assert H.assess_herringbone(frame, nm_per_px=NMPP,
                                snr_min=1e9).verdict == "absent"


def test_period_prior_is_required_and_not_defaulted_silently():
    """没有先验就没有搜索窗 —— 拒判,不假装按 Au(111) 处理。"""
    res = H.assess_herringbone(_herringbone(seed=1), nm_per_px=NMPP,
                               period_prior_nm=0.0)
    assert res.verdict == "undetermined"
    assert res.reasons == ("no_period_prior",)


def test_unusable_frame_still_reports_what_it_can_answer():
    """拒判 ≠ 一片空白。与帧内容无关的量照报 —— 「测出来是零」和「没测」是两件事。"""
    res = H.assess_herringbone(np.zeros((PIXELS, PIXELS)), nm_per_px=NMPP)
    assert res.nm_per_px == pytest.approx(NMPP)
    assert res.px_per_period == pytest.approx(PERIOD_NM / NMPP)
    assert res.periods_in_frame == pytest.approx(PIXELS * NMPP / PERIOD_NM)
    assert res.frame_nm == pytest.approx(FRAME_NM)
    assert res.corrugation_rms_m == 0.0
    assert res.period_prior_nm == pytest.approx(PERIOD_NM)


def test_corrugation_is_withheld_when_z_is_not_metres():
    """报一个未知单位的皮米数比不报更坏。"""
    res = H.assess_herringbone(_herringbone(seed=1), nm_per_px=NMPP,
                               z_unit_is_m=False)
    assert res.verdict == "herringbone"
    assert res.stripe_corrugation_pm is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
