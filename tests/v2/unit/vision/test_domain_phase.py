"""畴指纹判据(``mast.vision.domain_phase``)。

**第一节是反例**:纯噪声、准周期抖动、尺度门 off、死平帧 —— 一个都不许落成
「新畴」。这与判据本身同等重要:一个判据在正例上有信号不算数,**要证明它在反例上
没有**(本仓两次同款教训:``atomic_phase`` 的带通抖动对照、``herringbone`` 的纯白
噪声对照)。

**第三节是「判据有效性四检验」**,跑在分离度**之前**:分离度问「分得开吗」,四检验
问「分开的是不是你要的那个东西」。可重复 ≠ 有效。

§5.6 的变异验证在最后一节:每条都先证明**变异已应用**(替身确实换了行为),再证明
**测试红了**。其中「bracket 的 epoch 校验去掉」那一条属于状态机(composite 层),
不在本文件射程内。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import _domain_synth as S          # 同目录的合成语料(pytest 会把它加进 sys.path)
from mast.vision import domain_phase as D
from mast.vision import domain_reference as R

SYM = S.HEX_SYMMETRY_DEG


# ── 工具 ────────────────────────────────────────────────────────────────────

def fp(image, *, nmpp: float = S.NMPP, angle: float | None = 0.0, **kw):
    return D.extract_fingerprint(image, nm_per_px=nmpp, scan_angle_deg=angle, **kw)


def dist(a, b, sym: float = SYM) -> float | None:
    return D.fingerprint_distance(a, b, symmetry_deg=sym)


def make_reference(fps: dict, *, match_tol: float = 0.06,
                   ambiguity_margin: float = 0.03,
                   mixed_coverage_min: float = 0.8,
                   symmetry_deg: float = SYM, **over):
    """从几个指纹造一个已标定的参照系(测试用,不落盘)。"""
    body = {
        "schema": 1, "version": "v001", "sample": "synthetic",
        "created": "2000-08-14", "confirmed_by": "test",
        "provenance": "合成语料;本文件自造,不代表任何真实样品。",
        "symmetry_deg": symmetry_deg, "labels": list(fps),
        "prototypes": {k: {"peaks": [list(t) for t in v.triples()]}
                       for k, v in fps.items()},
        "match": {"w_angle": 1.0, "w_period": 1.0, "match_tol": match_tol,
                  "ambiguity_margin": ambiguity_margin,
                  "mixed_coverage_min": mixed_coverage_min},
    }
    body.update(over)
    ref = R.reference_from_mapping(body)
    assert ref is not None and ref.calibrated
    return ref


@pytest.fixture(scope="module")
def same_domain():
    """组 ①:同一个畴,噪声 1-10 pm + 剪切 0-0.3 px/行 + 行偏移 + 随机相位。"""
    return [fp(f) for f in S.same_domain_batch(n_frames=20)]


@pytest.fixture(scope="module")
def low_shear_domain():
    """组 ①',同上但剪切 ≤0.1 px/行 —— 漂移受控时的分母。"""
    n = 20
    return [fp(S.lattice_frame(
        theta_deg=17.0, seed=200 + i, noise=(1.0 + 9.0 * i / (n - 1)) * 1e-12,
        shear_px_per_row=0.1 * i / (n - 1), row_offset_m=3e-12 * (i % 3)))
        for i in range(n)]


@pytest.fixture(scope="module")
def domain_a():
    return [fp(S.lattice_frame(theta_deg=17.0, seed=300 + i, noise=2e-12))
            for i in range(5)]


def _pairwise(fps):
    return [dist(a, b) for i, a in enumerate(fps) for b in fps[i + 1:]]


def _cross(xs, ys):
    return [dist(a, b) for a in xs for b in ys]


# ══ 1. 反例:一个都不许落成「新畴」 ════════════════════════════════════════

def test_pure_noise_is_undetermined_not_a_new_domain():
    """40 个种子的纯高斯白噪声。**照抄 herringbone 的先例**。

    而且给的话必须是「这儿什么都没有」,**不是**「转个角度再扫一张」——
    herringbone 早先的版本把 30 个纯噪声帧里的 21 个判成「转 30° 重扫」,那条被否
    方案是被 ``test_pure_noise_is_absent_not_undetermined`` 钉住的。
    """
    bad = []
    for s in range(40):
        f = fp(S.noise_frame(seed=s))
        assert not f.comparable, f"种子 {s} 的纯噪声落成了可比指纹"
        assert f.blocking_reason == "no_atomic_phase", (s, f.reasons)
        if "slow_axis_degenerate" in f.reasons:
            bad.append(s)
    assert not bad, f"纯噪声帧被建议「转 30° 重扫」: {bad}"


def test_bandpass_jitter_is_undetermined_not_a_new_domain():
    """带通白噪声(针尖抖动/反馈振铃)—— 峰强度类判据一个都拦不住的那一组。"""
    for s in range(20):
        f = fp(S.bandpass_noise_frame(seed=s))
        assert not f.comparable, f"种子 {s} 的准周期抖动落成了可比指纹"
        assert f.blocking_reason == "no_atomic_phase", (s, f.reasons)


def test_scale_gate_is_undetermined_not_absent():
    """50 nm / 256 px = 0.195 nm/px:**拒判**,而且不能被读成「这里没有畴」。"""
    coarse = 50.0 / 256
    f = fp(S.lattice_frame(nmpp=coarse, seed=7), nmpp=coarse)
    assert f.reasons == ("scale_gate",)
    assert not f.comparable and f.n_peaks == 0
    assert f.scale == "off"
    # 拒判时与帧内容无关的读数照报 —— 「测出来是零」和「没测」是两句话。
    assert f.nm_per_px == pytest.approx(coarse)


def test_dead_flat_is_undetermined():
    f = fp(S.dead_flat_frame())
    assert not f.comparable
    assert f.blocking_reason == "no_atomic_phase"
    assert "dead_flat" in (f.notes.get("atomic_reasons") or [])


def test_unknown_pixel_size_is_its_own_reason():
    f = fp(S.lattice_frame(seed=1), nmpp=None)
    assert f.reasons == ("unknown_pixel_size",)
    assert not f.comparable


def test_resampled_input_is_refused_by_declaration():
    """调用方声明重采样过 ⇒ 直接拒判(尺度门在重采样帧上等于没有,见模块注释)。"""
    f = fp(S.upsample2(S.lattice_frame(seed=1)), nmpp=S.NMPP / 2,
           native_sampling=False)
    assert f.reasons == ("resampled_input",)
    assert not f.comparable


def test_every_reason_is_in_the_closed_set():
    """任何一条 reason 都必须在 :data:`UNDETERMINED_REASONS` 里,否则上层没法穷举。"""
    frames = [
        S.noise_frame(seed=0), S.bandpass_noise_frame(seed=0),
        S.dead_flat_frame(), S.lattice_frame(seed=0),
        S.lattice_frame(seed=0, n=64), S.lattice_frame(nmpp=0.195, seed=0),
    ]
    seen = set()
    for img in frames:
        for nmpp in (S.NMPP, 0.195, None):
            seen |= set(fp(img, nmpp=nmpp).reasons)
    assert seen, "一条 reason 都没触发,这个测试就没在测东西"
    assert seen <= set(D.UNDETERMINED_REASONS), seen - set(D.UNDETERMINED_REASONS)


# ══ 2. 分离度:四组对照(D6)。**分母比分子重要** ═══════════════════════════

def test_same_domain_frames_stay_close(same_domain, low_shear_domain):
    """组 ①:这一组的**最大值就是分离度的分母**。数字写在模块 docstring 里。"""
    assert all(f.comparable for f in same_domain)
    hi = max(_pairwise(same_domain))
    lo = max(_pairwise(low_shear_domain))
    # 区间钉住(合成实测 0.1167 / 0.0401)。上界松一点留给 numpy 版本差异,
    # 下界不能松:分母塌下去会让下面每一条「分得开」都变成假的。
    assert 0.08 <= hi <= 0.15, f"剪切≤0.3 的同畴分母 {hi:.4f} 跑出区间"
    assert 0.02 <= lo <= 0.06, f"剪切≤0.1 的同畴分母 {lo:.4f} 跑出区间"
    assert lo < hi, "剪切放宽反而让同畴距离变小 —— 分母的成因搞错了"


def test_shear_is_what_sets_the_denominator():
    """分母几乎**全部**来自慢轴剪切,不是噪声。

    这条决定了用户该拧哪个旋钮:同畴散布是**漂移**定的,不是判据定的 ⇒
    要提高取向分辨率就降扫速/等漂移稳,而不是去调判据里的常数。
    合成实测:只噪声 0.007、只行偏移 0.0005、只随机相位 0.0005、只剪切 0.117。
    """
    def batch(**kw):
        return [fp(S.lattice_frame(theta_deg=17.0, seed=200 + i, **{
            k: (v(i) if callable(v) else v) for k, v in kw.items()}))
            for i in range(10)]

    noise_only = max(_pairwise(batch(noise=lambda i: (1.0 + 9.0 * i / 9) * 1e-12)))
    shear_only = max(_pairwise(batch(noise=2e-12,
                                     shear_px_per_row=lambda i: 0.3 * i / 9)))
    assert noise_only < 0.02, f"只变噪声就散了 {noise_only:.4f}"
    assert shear_only > 5 * noise_only, (
        f"剪切 {shear_only:.4f} 没有主导噪声 {noise_only:.4f} —— "
        "分母的成因变了,模块 docstring 的结论要重写")


def test_orientation_difference_is_monotone(domain_a, same_domain,
                                            low_shear_domain):
    """组 ②:纯取向差 ⇒ 距离**单调**,并给出「多小的 Δθ 分不开」这个数。"""
    denom_hi = max(_pairwise(same_domain))
    denom_lo = max(_pairwise(low_shear_domain))
    got = []
    for dtheta in (2.0, 5.0, 10.0, 20.0, 30.0):
        other = [fp(S.lattice_frame(theta_deg=17.0 + dtheta, seed=400 + i,
                                    noise=2e-12)) for i in range(5)]
        got.append((dtheta, min(_cross(domain_a, other))))
    assert [d for _, d in got] == sorted(d for _, d in got), got
    # 每一度的代价 ≈ 1/(symmetry/2) —— 角度项是线性的,这里顺手钉住量纲。
    assert dict(got)[30.0] == pytest.approx(0.5, abs=0.02)
    # 分得开的最小 Δθ:剪切 ≤0.3 时是 10°,剪切 ≤0.1 时是 5°。
    assert dict(got)[5.0] < denom_hi < dict(got)[10.0], (got, denom_hi)
    assert dict(got)[2.0] < denom_lo < dict(got)[5.0], (got, denom_lo)


def test_different_layout_separates(domain_a, same_domain, low_shear_domain):
    """组 ③:周期集合不同(周期比 1.15;六角 vs 矩形)。"""
    denom_hi = max(_pairwise(same_domain))
    denom_lo = max(_pairwise(low_shear_domain))
    scaled = [fp(S.lattice_frame(theta_deg=17.0, seed=500 + i, noise=2e-12,
                                 period_nm=S.ROW_SPACING_NM * 1.15))
              for i in range(5)]
    d_scaled = min(_cross(domain_a, scaled))
    rect = [fp(S.lattice_frame(theta_deg=17.0, seed=600 + i, noise=2e-12,
                               symmetry_deg=90.0,
                               second_period_nm=S.ROW_SPACING_NM * 1.3))
            for i in range(5)]
    d_rect = min(_cross(domain_a, [f for f in rect if f.comparable]))
    # 周期比 1.15 只有在漂移受控时才分得开 —— 这正是「分母比分子重要」。
    assert d_scaled > denom_lo, (d_scaled, denom_lo)
    assert d_scaled < denom_hi, (
        f"周期比 1.15 的距离 {d_scaled:.4f} 已经越过高剪切分母 {denom_hi:.4f} —— "
        "结论变好了是好事,但模块 docstring 里那句「剪切 ≤0.3 时分不开」要改")
    assert d_rect > denom_hi, (d_rect, denom_hi)


def test_negative_group_never_produces_a_domain():
    """组 ④ 汇总:22 帧反例,**零**个可比指纹。"""
    neg = ([S.noise_frame(seed=s) for s in range(10)]
           + [S.bandpass_noise_frame(seed=s) for s in range(10)]
           + [S.dead_flat_frame()])
    leaked = [i for i, img in enumerate(neg) if fp(img).comparable]
    assert not leaked, f"反例落成了可比指纹: {leaked}"
    assert not fp(S.lattice_frame(nmpp=0.195, seed=1), nmpp=0.195).comparable


# ══ 3. 判据有效性四检验 —— 可重复 ≠ 有效 ═══════════════════════════════════

def test_check1_zero_information_injection_leaves_the_fingerprint_unchanged():
    """检验 ①(最有力也最便宜):双线性升采样 ×2,**零新信息**。

    ``nm_per_px`` 减半、像素数翻倍。角度与周期**必须逐项不变** —— 变化的那部分就
    是判据在量它自己。合成实测:角度 Δ ≤ 0.002°,周期相对 Δ ≤ 3e-5。
    """
    da = dp = 0.0
    for i in range(10):
        img = S.lattice_frame(theta_deg=17.0, seed=700 + i, noise=2e-12)
        a, b = fp(img), fp(S.upsample2(img), nmpp=S.NMPP / 2)
        assert a.comparable and b.comparable
        assert a.n_peaks == b.n_peaks
        for pa in a.peaks:
            pb = min(b.peaks, key=lambda q: abs(
                (q.k_angle_sample_deg - pa.k_angle_sample_deg + 90) % 180 - 90))
            da = max(da, abs((pb.k_angle_sample_deg - pa.k_angle_sample_deg + 90)
                             % 180 - 90))
            dp = max(dp, abs(pb.period_nm - pa.period_nm) / pa.period_nm)
    assert da < 0.05, f"升采样把角度改了 {da:.4f}° —— 判据在量自己"
    assert dp < 1e-3, f"升采样把周期改了 {dp:.5f} —— 判据在量自己"


def test_check1_the_scale_gate_itself_is_fooled_by_upsampling():
    """检验 ① 的副产品(陷阱 19):**被骗的是门,不是指纹**。

    这条是拒收重采样输入的**理由**,写成测试免得下一个人把 ``native_sampling``
    当多余参数删掉。
    """
    from mast.vision.atomic_phase import scale_gate
    assert scale_gate(0.06) == "off"
    assert scale_gate(0.03) == "reduced"      # 升采样 ×2 之后 —— 零新信息,却过了门
    img = S.lattice_frame(nmpp=0.06, seed=1)
    assert fp(img, nmpp=0.06).reasons == ("scale_gate",)
    # 同一帧升采样后,门放行了 —— 所以只能靠调用方声明。
    up = fp(S.upsample2(img), nmpp=0.03, allow_reduced_scale=True)
    assert up.scale == "reduced"
    assert fp(S.upsample2(img), nmpp=0.03, allow_reduced_scale=True,
              native_sampling=False).reasons == ("resampled_input",)


def test_check2_information_deprivation_finds_the_real_resolution_need():
    """检验 ②:固定物理视野,像素数 512→256→128→64→32。

    合成实测:64 px(6.4 px/周期)之前指纹**几乎不变**(角度 Δ ≤ 0.031°),
    32 px(3.2 px/周期)撞尺度门拒判。⇒ **指纹要的不是像素数,是 nm/px**;
    满权重档以上再加像素,对指纹没有增量(对别的判据可能仍有)。
    """
    frame_nm, n = 2.5, 512
    nmpp0 = frame_nm / n
    big = S.lattice_frame(theta_deg=17.0, n=n, nmpp=nmpp0, seed=800, noise=2e-12)
    ref = fp(big, nmpp=nmpp0, allow_reduced_scale=True)
    assert ref.comparable and ref.n_peaks == 3
    worst = 0.0
    for f_ in (2, 4, 8):
        g = fp(S.downsample(big, f_), nmpp=nmpp0 * f_, allow_reduced_scale=True)
        assert g.comparable, (f_, g.reasons)
        assert g.n_peaks == ref.n_peaks
        worst = max(worst, dist(ref, g))
    assert worst < 0.005, f"降到 64 px 指纹就变了 {worst:.4f}"
    g32 = fp(S.downsample(big, 16), nmpp=nmpp0 * 16, allow_reduced_scale=True)
    assert g32.reasons == ("scale_gate",), g32.reasons


def test_check3_fingerprint_is_not_proportional_to_any_internal_constant():
    """检验 ③:逐个扫判据链路上写死的常数。指纹**不得正比于**任何一个。

    合成实测:质心窗 3×3→5×5→7×7 角度 Δ ≤ 0.065°;环排他半径 3→8 bin 完全不变
    (2 bin 时会把同一个峰数两次 —— 那正是默认取 3 的理由);功率下限 0.02→0.2
    完全不变。
    """
    img = S.lattice_frame(theta_deg=17.0, seed=900, noise=2e-12)
    ref = fp(img)

    def drift(**kw) -> float:
        g = fp(img, **kw)
        assert g.comparable, (kw, g.reasons)
        return max(min(abs((q.k_angle_sample_deg - p.k_angle_sample_deg + 90)
                           % 180 - 90) for q in g.peaks) for p in ref.peaks)

    for win in (2, 3):
        assert drift(centroid_half_window=win) < 0.2
    for excl in (3.0, 5.0, 8.0):
        assert drift(exclusion_bins=excl) < 0.05
        assert fp(img, exclusion_bins=excl).n_peaks == ref.n_peaks
    for floor in (0.02, 0.2):
        assert drift(min_rel_power=floor) < 0.05
    # 排他半径 2 bin < Hann 主瓣半宽 ⇒ 同一个峰被数两次。默认 3 的推导由此成立。
    assert fp(img, exclusion_bins=2.0).n_peaks > ref.n_peaks


def test_check3_direction_does_not_come_from_angular_bins():
    """检验 ③ 的重点:方向来自二维谱峰的亚像素质心,**不是**角向 bin 的中心。

    D2 选 ``stripe_peak`` 那套配方的理由是「角向 bin 的量化 = 360/n_bins」——
    这条理由**必须被扫描证明**,不能只是转述。所以这里同时跑一个用
    ``_angular_bins`` bin 中心报方向的**替身**:它随 ``n_bins`` 变(量化步长),
    而真实现一动不动。
    """
    from mast.vision.atomic_phase import _angular_bins, _ring_mask

    img = S.lattice_frame(theta_deg=17.0, seed=901, noise=2e-12)
    ref = fp(img)
    t_px = ref.peaks[0].period_nm / S.NMPP

    def bin_centre_direction(n_bins: int) -> float:
        """替身:角向 bin 里最大的那个 bin 的**中心** —— 被否掉的那个做法。"""
        from mast.vision.seg_scale_adaptive import flatten_robust
        flat = flatten_robust(img)
        x = flat - flat.mean()
        win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
        P = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2
        sel, ang = _ring_mask(P.shape, float(P.shape[0]) / t_px)
        bins = _angular_bins(P[sel], ang[sel], n_bins)
        centre = -180.0 + (int(np.argmax(bins)) + 0.5) * 360.0 / n_bins
        return centre % 180.0

    mutant = {nb: bin_centre_direction(nb) for nb in (36, 72, 144)}
    spread = max(abs((a - b + 90) % 180 - 90) for a in mutant.values()
                 for b in mutant.values())
    assert spread >= 360.0 / 144, (
        f"替身没有表现出 bin 量化({mutant})—— 这条测试没在证明什么,先修替身")
    # 真实现:峰表逐项不变(角向 bin 数根本不进这条路径)。
    same = fp(img)
    assert [round(p.k_angle_sample_deg, 9) for p in same.peaks] == \
           [round(p.k_angle_sample_deg, 9) for p in ref.peaks]


def test_check4_external_ruler_lattice_constant_and_known_misorientation():
    """合成语料提供独立周期与取向真值；算法内部残差与自洽性不能替代外部标尺。"""
    img = S.lattice_frame(theta_deg=17.0, seed=1000, noise=2e-12)
    g = fp(img)
    for p in g.peaks:
        assert abs(p.period_nm - S.ROW_SPACING_NM) / S.ROW_SPACING_NM < 0.01
        assert min(abs((p.k_angle_sample_deg - t + 90) % 180 - 90)
                   for t in (17.0, 77.0, 137.0)) < 0.5
    # 已知取向差 25°:测出来必须是 25°,不是别的数(实测 25.010°)。
    # 取 25 而不是 12,是因为 12 会把一个峰推到慢轴缺口边上 —— 那个峰会被平场
    # 拉偏 1.8°(实测 89.0° → 87.2°),那是**盲区的代价**,由
    # ``test_slow_axis_blind_spot_is_reported_not_silent`` 单独量。
    tw = fp(S.lattice_frame(theta_deg=42.0, seed=1001, noise=2e-12))
    assert not tw.warnings, tw.warnings
    diffs = [abs((p.k_angle_sample_deg - q.k_angle_sample_deg + 90) % 180 - 90)
             for p in tw.peaks for q in g.peaks]
    got = min(diffs, key=lambda x: abs(x - 25.0))
    assert got == pytest.approx(25.0, abs=0.5), (got, diffs)


# ══ 4. 被否掉的方案钉成测试(§5.4) ════════════════════════════════════════

def test_single_argmax_direction_flips_by_60deg_on_hex():
    """「只取 argmax 当主方向」这条被否方案**会假报畴界**。

    同一个畴的连续两帧,argmax 落在 6 个布拉格点的哪一个由针尖衬度各向异性
    (±15%,随针尖状态帧帧在变)决定 ⇒ 方向可以差 60°。合成实测:argmax 散布
    60.1°,而峰集合指纹的同畴距离只有 0.0013 —— 差了 40 倍以上。
    """
    fps = [fp(S.lattice_frame(theta_deg=17.0, seed=300 + i, noise=3e-12,
                              amp_jitter_frac=0.15)) for i in range(20)]
    angs = [f.peaks[0].k_angle_sample_deg for f in fps]
    argmax_spread = max(abs((a - b + 90) % 180 - 90) for a in angs for b in angs)
    fingerprint_spread = max(_pairwise(fps))
    assert argmax_spread > 55.0, (
        f"argmax 只散了 {argmax_spread:.1f}° —— 各向异性没造出翻转,这条测试就没在"
        "证明被否方案的致命性(先修语料,别放宽断言)")
    assert fingerprint_spread < 0.02, fingerprint_spread
    # 换算成同一把尺子:60° 的方向翻转 = 距离 1.0(mod 60 之后其实是 0)——
    # 峰集合指纹之所以没事,正是因为比对在商空间里做。
    assert argmax_spread / 30.0 > 50 * fingerprint_spread


def test_frame_rotation_does_not_change_the_fingerprint():
    """旋转扫描框后，样品坐标下的指纹应保持不变。
    合成数据与算法共享坐标约定，因此这只验证归一化被执行，符号仍需独立校准。"""
    base = fp(S.lattice_frame(theta_deg=17.0, scan_angle_deg=0.0, seed=900,
                              noise=2e-12), angle=0.0)
    for a in (30.0, 60.0, 90.0, 137.0):
        g = fp(S.lattice_frame(theta_deg=17.0, scan_angle_deg=a, seed=901,
                               noise=2e-12), angle=a)
        assert g.comparable, (a, g.reasons)
        assert dist(base, g) < 0.01, (a, dist(base, g))
    # 帧系角**必须**真的跟着转过 —— 否则「样品系不变」是因为根本没转过帧。
    turned = fp(S.lattice_frame(theta_deg=17.0, scan_angle_deg=25.0, seed=902,
                                noise=2e-12), angle=25.0)
    frame_shift = min(abs((p.k_angle_frame_deg - 17.0 + 90) % 180 - 90)
                      for p in turned.peaks)
    assert frame_shift > 20.0, (
        f"帧系角只动了 {frame_shift:.1f}° —— 语料根本没转帧,「不变」是假的")


@pytest.mark.skip(reason="需要独立仪器校准来确认扫描角符号约定；"
                         "指纹变不变。合成语料按 `+` 的约定生成,证明不了约定本身。")
def test_angle_sign_convention_matches_the_instrument():
    """扫描角符号需要独立仪器校准。整体翻符号与连续的压电各向异性应分别识别，
    需多个非共线角度支持区分。"""


def test_unknown_frame_angle_is_undetermined_not_zero():
    """帧角读不到 ⇒ ``unknown_frame_angle``,**不按 0° 处理**。

    折叠成 0.0 会让同一个畴在两种帧角下报成两个畴,而且零报错。
    """
    img = S.lattice_frame(theta_deg=17.0, scan_angle_deg=40.0, seed=903,
                          noise=2e-12)
    blind = fp(img, angle=None)
    assert blind.reasons == ("unknown_frame_angle",)
    assert not blind.comparable
    assert blind.scan_angle_deg is None
    assert blind.n_peaks == 3                      # 诊断读数照报
    assert all(p.k_angle_sample_deg is None for p in blind.peaks)
    assert blind.triples() == ()                   # 半个指纹不许流出去
    with pytest.raises(ValueError):
        blind.peaks[0].as_triple()
    # 「按 0° 处理」会得到什么:与真值差 40° 的指纹 —— 那正是要防的假畴。
    as_zero = fp(img, angle=0.0)
    truth = fp(S.lattice_frame(theta_deg=17.0, scan_angle_deg=0.0, seed=904,
                               noise=2e-12), angle=0.0)
    # 40° mod 60 = 20° 的假取向差 ⇒ d = 20/30 × 0.5 = 0.334,是同畴分母(0.04)的
    # **八倍** —— 也就是同一个畴会被稳稳地报成两个畴,而且零报错。
    assert dist(as_zero, truth) > 0.2, "按 0° 处理居然没造成偏差?检查语料"


def test_no_reference_never_yields_a_label(domain_a):
    """D3:没有参照系(或参照系未标定)⇒ **永远**不出 label。"""
    f = domain_a[0]
    assert D.classify(f, None).verdict == "undetermined"
    assert D.classify(f, None).reason == "no_reference"
    assert D.classify(f, None).label is None
    # ``match.*`` 全 0 = 未标定,**不是零容差**。
    body = {
        "schema": 1, "version": "v001", "sample": "s", "created": "2000-08-14",
        "provenance": "p", "symmetry_deg": 60.0, "labels": ["A"],
        "prototypes": {"A": {"peaks": [list(t) for t in f.triples()]}},
        "match": {"match_tol": 0.0, "ambiguity_margin": 0.0,
                  "mixed_coverage_min": 0.0},
    }
    ref0 = R.reference_from_mapping(body)
    assert ref0 is not None and not ref0.calibrated
    v = D.classify(f, ref0)
    assert (v.verdict, v.reason, v.label) == ("undetermined", "no_reference", None)


def test_frame_level_reason_outranks_no_reference():
    """普查阶段每一帧都没有参照系 ⇒ ``no_reference`` 不可行动;帧级的码才是。

    两条都不出 label,所以先后不动「没标定就不许出 A/B」那条硬规矩;但把
    ``scale_gate`` 藏在 ``no_reference`` 后面,会让整轮普查看不见「这些帧太粗」。
    """
    coarse = fp(S.lattice_frame(nmpp=0.195, seed=1), nmpp=0.195)
    assert D.classify(coarse, None).reason == "scale_gate"
    good = fp(S.lattice_frame(theta_deg=17.0, seed=1, noise=2e-12))
    assert D.classify(good, None).reason == "no_reference"
    # 有参照系时,帧级的码同样要盖过距离判定(不可比的指纹不许算距离)。
    ref = make_reference({"A": good})
    v = D.classify(coarse, ref)
    assert (v.reason, v.label, v.distances) == ("scale_gate", None, {})


def test_mixed_requires_both_prototypes_peaks():
    """陷阱 11:``mixed`` 与 ``ambiguous`` 是**两个判据**,不是一个阈值。

    * ``mixed`` = A 的峰**和** B 的峰**都在**(畴界就在这一帧里,最有价值的信号);
    * ``ambiguous_match`` = 哪个的峰都不全在,而距离又拉不开。
    """
    fa = fp(S.lattice_frame(theta_deg=0.0, seed=1, noise=2e-12))
    fb = fp(S.lattice_frame(theta_deg=25.0, seed=2, noise=2e-12))
    ref = make_reference({"A": fa, "B": fb})

    a_frame = fp(S.lattice_frame(theta_deg=0.0, seed=3, noise=2e-12))
    b_frame = fp(S.lattice_frame(theta_deg=25.0, seed=4, noise=2e-12))
    boundary = fp(S.lattice_frame(theta_deg=0.0, seed=11, noise=2e-12)
                  + S.lattice_frame(theta_deg=25.0, seed=12, noise=2e-12))

    assert D.classify(a_frame, ref).verdict == "A"
    assert D.classify(b_frame, ref).verdict == "B"
    v = D.classify(boundary, ref)
    assert v.verdict == "mixed", (v.verdict, v.reason, v.distances, v.coverage)
    assert min(v.coverage.values()) >= ref.mixed_coverage_min
    # 到两家的距离都不小 —— 只看距离的话它和「判不了」长得一模一样。
    assert min(v.distances.values()) > ref.match_tol
    assert v.label is None


def test_ambiguous_is_not_mixed_when_the_prototypes_overlap():
    """两个原型自己就分不开时,单畴帧必须是 ``ambiguous_match`` 而**不是** ``mixed``。

    报成 ``mixed`` 会让二分以为「畴界就在脚下」并当场收敛到一个不存在的边界。
    """
    fa = fp(S.lattice_frame(theta_deg=0.0, seed=1, noise=2e-12))
    fb = fp(S.lattice_frame(theta_deg=2.0, seed=2, noise=2e-12))
    ref = make_reference({"A": fa, "B": fb}, match_tol=0.10,
                         ambiguity_margin=0.03, mixed_coverage_min=0.8)
    proto_d = D.peak_distance(ref.peaks_of("A"), ref.peaks_of("B"),
                              symmetry_deg=SYM)
    assert proto_d < ref.match_tol, "原型没有重叠,这条测试没在测它要测的东西"
    v = D.classify(fp(S.lattice_frame(theta_deg=1.0, seed=5, noise=2e-12)), ref)
    assert v.verdict == "undetermined" and v.reason == "ambiguous_match", v


def test_a_third_cluster_is_no_match_not_ambiguous():
    """像谁都不像 ⇒ ``no_match``(证据矛盾,升级问人),不许被读成「判不了」。"""
    fa = fp(S.lattice_frame(theta_deg=0.0, seed=1, noise=2e-12))
    fb = fp(S.lattice_frame(theta_deg=25.0, seed=2, noise=2e-12))
    ref = make_reference({"A": fa, "B": fb})
    v = D.classify(fp(S.lattice_frame(theta_deg=40.0, seed=5, noise=2e-12)), ref)
    assert v.verdict == "undetermined" and v.reason == "no_match", v
    assert min(v.distances.values()) > ref.match_tol
    assert max(v.coverage.values()) < ref.mixed_coverage_min


def _level_only(h):
    """替身用:只做二阶平场,**不做行对齐** —— 把慢轴分量留下来。"""
    from mast.vision.seg_scale_adaptive import _vander
    h = np.asarray(h, np.float64)
    A = _vander(h.shape, 2)
    coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
    return (h.ravel() - A @ coef).reshape(h.shape)


def test_slow_axis_peak_needs_the_no_row_align_control(monkeypatch):
    """陷阱 9:慢轴上的峰,只看标准平场分不出真假 —— 要「另一条路对不对得上」。

    自然帧造不出这个状态(标准平场把**真的**慢轴峰减没了,而**假的**只出现在原子
    相判据先一步拒掉的帧上),所以这里把两条路各换一次替身,证明**结论确实由对照
    决定**:同一帧、同一主峰,对照说「有」就放行,对照说「没有」就拒判。
    """
    import mast.vision.seg_scale_adaptive as SS

    img = S.lattice_frame(theta_deg=90.0, symmetry_deg=90.0, seed=11, noise=2e-12)
    # 替身 1:平场不做行对齐 ⇒ 慢轴上那个真峰活下来,成为主峰。
    monkeypatch.setattr(SS, "flatten_robust", SS.level_iterative)
    kept = fp(img)
    assert kept.peaks and abs(kept.peaks[0].k_angle_frame_deg - 90.0) <= 10.0, (
        "替身没生效:主峰没有落在慢轴缺口里,后面两条断言就不是在测对照")
    assert "slow_axis_degenerate" not in kept.reasons, kept.reasons

    # 替身 2:对照那条路返回纯噪声 ⇒ 对照看不到这个峰 ⇒ 必须拒判。
    monkeypatch.setattr(SS, "flatten_robust", _level_only)
    monkeypatch.setattr(SS, "level_iterative",
                        lambda h, *a, **k: S.noise_frame(seed=999, n=h.shape[0]))
    blinded = fp(img)
    assert blinded.peaks and abs(blinded.peaks[0].k_angle_frame_deg - 90.0) <= 10.0
    assert "slow_axis_degenerate" in blinded.reasons, blinded.reasons
    assert not blinded.comparable


def test_slow_axis_blind_spot_is_reported_not_silent():
    """标准平场会**吃掉** k 沿慢轴的那个方向 —— 零报错。必须说出来。

    六角(mod 60 等价)丢一个峰不影响比对;**非六角**丢的那个峰带着独立周期,
    比对会偏(实测:矩形晶格两个帧角下 d=0.029,而同畴分母只有 0.04)。
    """
    eaten = fp(S.lattice_frame(theta_deg=90.0, seed=5, noise=2e-12))
    assert eaten.n_peaks == 2                       # 三个方向只剩两个
    assert "slow_axis_blind_spot" in eaten.warnings
    assert "转 ~30° 重扫" in eaten.notes["slow_axis_blind_spot"]
    # 六角:少一个峰不影响比对(商空间里三个方向本来就等价)。
    full = fp(S.lattice_frame(theta_deg=90.0, scan_angle_deg=30.0, seed=7,
                              noise=2e-12), angle=30.0)
    assert full.n_peaks == 3
    assert dist(eaten, fp(S.lattice_frame(theta_deg=90.0, seed=6, noise=2e-12))) < 0.01
    # 矩形:同一个畴、两个帧角,指纹**真的**差了 —— 这是盲区的代价,量出来。
    r0 = fp(S.lattice_frame(theta_deg=0.0, symmetry_deg=90.0, scan_angle_deg=0.0,
                            second_period_nm=S.ROW_SPACING_NM * 1.3, seed=9,
                            noise=2e-12), angle=0.0)
    r30 = fp(S.lattice_frame(theta_deg=0.0, symmetry_deg=90.0, scan_angle_deg=30.0,
                             second_period_nm=S.ROW_SPACING_NM * 1.3, seed=10,
                             noise=2e-12), angle=30.0)
    assert "slow_axis_blind_spot" in r0.warnings and r0.n_peaks == 1
    assert r30.n_peaks == 2
    assert 0.01 < dist(r0, r30, 90.0) < 0.10
    # 缺口**边上**的峰活得下来,但被拉偏:真值 89.0° 报成 87.2°(实测 1.8°)。
    near = fp(S.lattice_frame(theta_deg=29.0, seed=1001, noise=2e-12))
    assert "slow_axis_peak_present" in near.warnings
    got = min(p.k_angle_sample_deg for p in near.peaks
              if abs(p.k_angle_sample_deg - 89.0) < 5.0)
    assert 1.0 < abs(got - 89.0) < 3.0, got


def test_too_few_periods_asks_for_a_bigger_frame_not_a_new_tip():
    """帧里周期太少 ⇒ 核心判据静默返回 0。这时说「没有原子相」是撒谎。

    两条码的下一步**相反**(修针/换点 vs 换更大的帧),所以 ``too_few_periods``
    必须排在 ``no_atomic_phase`` 前面。
    """
    small = S.lattice_frame(theta_deg=17.0, n=64, nmpp=0.0049, seed=1, noise=2e-12)
    f = fp(small, nmpp=0.0049)
    assert f.blocking_reason == "too_few_periods", f.reasons
    assert f.periods_in_frame is not None and f.periods_in_frame < 5.36


# ══ 5. 距离本身的性质 ════════════════════════════════════════════════════════

def test_distance_is_symmetric_and_bounded(domain_a):
    a, b = domain_a[0], fp(S.lattice_frame(theta_deg=47.0, seed=1, noise=2e-12))
    assert dist(a, b) == pytest.approx(dist(b, a))
    assert 0.0 <= dist(a, b) <= 1.0
    assert dist(a, a) == pytest.approx(0.0, abs=1e-12)


def test_distance_is_none_not_zero_when_a_side_is_unusable(domain_a):
    """不可比返回 ``None`` —— **不是 0**(那会读成「一模一样」),也不是 inf。"""
    assert dist(domain_a[0], fp(S.noise_frame(seed=0))) is None
    assert dist(fp(S.noise_frame(seed=0)), domain_a[0]) is None
    assert D.peak_distance((), domain_a[0].triples(), symmetry_deg=SYM) is None
    assert D.peak_distance(domain_a[0].triples(), domain_a[0].triples(),
                           symmetry_deg=0.0) is None


def test_isotropic_calibration_bias_does_not_change_the_distance():
    """合成数据验证：固定的共同倍率不改变距离，单侧倍率变化则改变距离。

    重新标定后应重建参照系；本测试以任意合成倍率检验这一性质。
    """
    a = [S.lattice_frame(theta_deg=17.0, seed=1100 + i, noise=2e-12) for i in range(3)]
    b = [S.lattice_frame(theta_deg=27.0, seed=1200 + i, noise=2e-12) for i in range(3)]
    true = [dist(fp(x), fp(y)) for x in a for y in b]
    bias = [dist(fp(x, nmpp=S.NMPP * 0.8), fp(y, nmpp=S.NMPP * 0.8))
            for x in a for y in b]
    assert max(abs(t - c) for t, c in zip(true, bias)) < 1e-9
    one_sided = [dist(fp(x), fp(y, nmpp=S.NMPP * 0.8)) for x in a for y in b]
    assert max(abs(t - c) for t, c in zip(true, one_sided)) > 0.05


def test_strongest_ring_peak_matches_stripe_peak():
    """结构闸门:本模块的最强峰必须与 ``herringbone.stripe_peak`` **逐位一致**。

    ``ring_peaks`` 是那个函数的多峰版拷贝(它只给一个峰、拿不到谱,没法直接复用)。
    这条测试就是那份拷贝的同步性保证 —— 照
    ``test_bandpass_matches_seg_scale_adaptive`` 的既有做法。
    """
    from mast.vision.herringbone import stripe_peak
    from mast.vision.seg_scale_adaptive import flatten_robust

    for seed in range(5):
        flat = flatten_robust(S.lattice_frame(theta_deg=17.0 + 7 * seed, seed=seed))
        band_px = (max(3.0, 0.18 / S.NMPP), 0.80 / S.NMPP)
        mine = D.ring_peaks(flat, band_px)[0]
        theirs = stripe_peak(flat, band_px)
        assert mine[0] == pytest.approx(theirs[0], rel=1e-12)
        assert mine[1] == pytest.approx(theirs[1], rel=1e-12)


# ══ 6. 变异验证(§5.6):先证明变异已应用,再证明测试红了 ════════════════════

def test_mutation_symmetry_zero_kills_the_separation(domain_a):
    """``symmetry_deg: 60 → 0`` ⇒ 距离算不出来 ⇒ 分离度测试必红。"""
    a, b = domain_a[0], domain_a[1]
    assert isinstance(D.fingerprint_distance(a, b, symmetry_deg=60.0), float)  # 变异前
    assert D.fingerprint_distance(a, b, symmetry_deg=0.0) is None              # 变异后
    with pytest.raises(TypeError):
        assert D.fingerprint_distance(a, b, symmetry_deg=0.0) < 0.2


def test_mutation_dropping_frame_angle_normalisation_breaks_rotation():
    """``k_sample = k_frame``(去掉帧角归一)⇒ 转帧角的测试必红。"""
    import dataclasses

    def unnormalised(f):
        return dataclasses.replace(f, peaks=tuple(
            dataclasses.replace(p, k_angle_sample_deg=p.k_angle_frame_deg)
            for p in f.peaks))

    base = fp(S.lattice_frame(theta_deg=17.0, scan_angle_deg=0.0, seed=900,
                              noise=2e-12), angle=0.0)
    turned = fp(S.lattice_frame(theta_deg=17.0, scan_angle_deg=25.0, seed=901,
                                noise=2e-12), angle=25.0)
    assert dist(base, turned) < 0.01                          # 真实现:不变
    mutant = dist(unnormalised(base), unnormalised(turned))   # 变异后
    assert mutant > 0.3, f"变异没生效(d={mutant:.4f}),这条钉子是空的"


def test_mutation_nearest_label_fallback_would_leak_a_label(domain_a):
    """「兜底出最近的 label」这条变异 ⇒ ``no_reference`` 那条测试必红。

    替身在同一份输入上**确实**给得出 label,所以真实现给 ``undetermined`` 是
    守卫的功劳,不是数据碰巧。
    """
    f = domain_a[0]
    body = {
        "schema": 1, "version": "v001", "sample": "s", "created": "2000-08-14",
        "provenance": "p", "symmetry_deg": 60.0, "labels": ["A"],
        "prototypes": {"A": {"peaks": [list(t) for t in f.triples()]}},
        "match": {"match_tol": 0.0, "ambiguity_margin": 0.0,
                  "mixed_coverage_min": 0.0},
    }
    ref0 = R.reference_from_mapping(body)

    def naive_nearest_label(fingerprint, ref):
        """被否掉的那个做法:不管标没标定,报最近的那个。"""
        ds = {lab: D.peak_distance(fingerprint.triples(), ref.peaks_of(lab),
                                   symmetry_deg=ref.symmetry_deg)
              for lab in ref.labels}
        return min(ds, key=lambda k: ds[k])

    assert naive_nearest_label(f, ref0) == "A"          # 变异确实给得出 label
    assert D.classify(f, ref0).label is None            # 真实现不给


def test_mutation_absolute_period_difference_loses_calibration_immunity():
    """``d_T`` 从「相对差」改成「绝对差 nm」⇒ 标定偏差的免疫性消失。

    D5 的那句「相对量对各向同性标定误差免疫」由此变成**可证伪**的,而不是断言。
    """
    a = fp(S.lattice_frame(theta_deg=17.0, seed=1100, noise=2e-12))
    b = fp(S.lattice_frame(theta_deg=27.0, seed=1200, noise=2e-12))
    a2 = fp(S.lattice_frame(theta_deg=17.0, seed=1100, noise=2e-12),
            nmpp=S.NMPP * 0.875)
    b2 = fp(S.lattice_frame(theta_deg=27.0, seed=1200, noise=2e-12),
            nmpp=S.NMPP * 0.875)

    def absolute_variant(x, y) -> float:
        """变异:周期项用绝对差(纳米),不再是比。"""
        num = den = 0.0
        for pa in x.triples():
            best = min(abs((pa[0] - pb[0] + 30) % 60 - 30) / 30.0 + abs(pa[1] - pb[1])
                       for pb in y.triples())
            num += pa[2] * best
            den += pa[2]
        return num / den

    assert absolute_variant(a, b) != pytest.approx(absolute_variant(a2, b2),
                                                   abs=1e-9)   # 变异已生效
    assert dist(a, b) == pytest.approx(dist(a2, b2), abs=1e-9)  # 真实现免疫


def test_mutation_folding_unknown_angle_to_zero_creates_a_fake_domain():
    """「帧角读不到就按 0°」这条变异 ⇒ 同一个畴报成两个畴。"""
    img40 = S.lattice_frame(theta_deg=17.0, scan_angle_deg=40.0, seed=903,
                            noise=2e-12)
    truth = fp(S.lattice_frame(theta_deg=17.0, scan_angle_deg=0.0, seed=904,
                               noise=2e-12), angle=0.0)
    assert fp(img40, angle=None).blocking_reason == "unknown_frame_angle"
    folded = fp(img40, angle=0.0)                    # 变异:None → 0.0
    assert dist(folded, truth) > 0.2                 # 假畴,而且零报错
    assert dist(fp(img40, angle=40.0), truth) < 0.01  # 真值就在那儿


# ══ 7. 纯函数纪律 ═══════════════════════════════════════════════════════════

def test_extract_never_raises_on_junk():
    """不抛异常:输入再脏也只是「判不了」。"""
    junk = [np.zeros((4, 4)), np.zeros((256, 256)) + np.nan,
            np.arange(256 * 256).reshape(256, 256).astype(float),
            np.zeros((1, 64, 64)), np.zeros((256,))]
    for img in junk:
        f = D.extract_fingerprint(img, nm_per_px=S.NMPP, scan_angle_deg=0.0)
        assert isinstance(f, D.DomainFingerprint)
        assert not f.comparable
        assert f.blocking_reason in D.UNDETERMINED_REASONS


def test_extract_does_not_mutate_its_input():
    img = S.lattice_frame(seed=3)
    before = img.copy()
    fp(img)
    assert np.array_equal(img, before)


def test_classify_never_raises_on_a_broken_reference(domain_a):
    for ref in (None, R.reference_from_mapping({"schema": 99})):
        v = D.classify(domain_a[0], ref)
        assert v.verdict == "undetermined" and v.label is None
        assert v.reason in D.UNDETERMINED_REASONS


def test_math_module_is_used_for_angles_not_degrees_confusion():
    """圆周距离在 ``symmetry_deg`` 的商空间里:59° 与 1° 差 2°,不是 58°。"""
    assert D._circular_deg(58.0, 60.0) == pytest.approx(2.0)
    assert D._circular_deg(-58.0, 60.0) == pytest.approx(2.0)
    assert D._circular_deg(30.0, 60.0) == pytest.approx(30.0)
    assert D._circular_deg(90.0, 90.0) == pytest.approx(0.0)
    assert math.isclose(D._circular_deg(179.0, 180.0), 1.0)
