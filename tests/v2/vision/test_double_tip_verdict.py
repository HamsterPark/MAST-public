"""多针尖的三态判定与弃权。

独立合成孤立特征、台阶和前向模型，验证单位不变性、可判读前提、形貌闸门和跨帧共识。
所有输入均在测试中生成，不依赖仪器参考帧或实验数据。"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from mast.vision.double_tip import agree_across_frames, detect_double_tip  # noqa: E402

TWO_KAPPA_PER_NM = 20.0      # 2κ ≈ 20 nm⁻¹(功函数 ~4.5 eV)
AU_STEP_NM = 0.236           # Au(111) 单原子台阶

# ── 造帧 ────────────────────────────────────────────────────────────────────
def _features(n=256, k=30, sigma=3.0, seed=0):
    """平背景 + 孤立特征 —— 这个方法**真正被验证过**的形貌。"""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n), np.float32)
    for y, x in zip(rng.randint(28, n - 28, k), rng.randint(28, n - 28, k)):
        img += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
    return img + 0.03 * rng.randn(n, n).astype(np.float32)


def _linear_ghost(img, dy, dx, a=0.5):
    return ((1 - a) * img + a * ndi.shift(img, (dy, dx), order=1,
                                          mode="nearest")).astype(np.float32)


def _staircase(n=256, terrace_px=14.0, angle_deg=20.0, seed=0, noise_nm=0.008):
    """解析倾斜阶梯采用 Au(111) 通用单原子台阶高度；台面重复是自相关中的强干扰，不能等同于微尖间距。"""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    th = np.radians(angle_deg)
    u = xx * np.cos(th) + yy * np.sin(th)
    z = -AU_STEP_NM * np.floor(u / terrace_px)
    return (z + noise_nm * rng.randn(n, n)).astype(np.float64)


def _multitip(z_nm, apexes):
    """真前向模型:恒流下 N 个微尖 ⇒ z = (1/2κ)·ln Σ a_i e^{2κ z(r−d_i)}。

    **不是** ``(1-a)·img + a·shift(img)`` —— 那是**电流**里的线性回声,
    高度被 log 压成 soft-max。用错模型造出来的鬼影,检出了也不算数。
    """
    acc = None
    for dy, dx, h_nm in apexes:
        s = np.roll(np.roll(z_nm, dy, axis=0), dx, axis=1) + h_nm
        e = TWO_KAPPA_PER_NM * s
        acc = e if acc is None else np.logaddexp(acc, e)
    return acc / TWO_KAPPA_PER_NM


# ── 1. 单位 ─────────────────────────────────────────────────────────────────
def test_answer_does_not_depend_on_the_unit():
    """同一合成形貌换用米、纳米或无量纲表示时，判定与归一化得分应一致。"""
    img = _linear_ghost(_features(seed=1), 16, 12)
    in_m = detect_double_tip(img * 1e-9, nm_per_px=0.5)
    in_nm = detect_double_tip(img, nm_per_px=0.5)
    in_pm = detect_double_tip(img * 1e3, nm_per_px=0.5)
    assert in_m.verdict == in_nm.verdict == in_pm.verdict
    assert in_m.score == pytest.approx(in_nm.score, rel=1e-3)
    assert in_m.score == pytest.approx(in_pm.score, rel=1e-3)


def test_a_real_amplitude_frame_is_not_silently_discarded():
    """缩小独立合成图的幅值后仍应得到有效得分，不能因绝对单位尺度而静默返回零。"""
    img = _linear_ghost(_features(seed=2), 16, 12)
    r = detect_double_tip(img * 2e-11, nm_per_px=0.5)
    assert r.score > 0.0
    assert r.verdict in ("multi_tip", "single_tip", "undetermined")


# ── 2. 三态:「没检出」≠「判不了」 ──────────────────────────────────────────
def test_verdict_is_tri_state_not_a_bool():
    assert detect_double_tip(_linear_ghost(_features(seed=3), 16, 12)
                             ).verdict == "multi_tip"
    assert detect_double_tip(_features(seed=3)).verdict == "single_tip"
    # 死平帧:判不了,而**不是**「干净」
    flat = detect_double_tip(np.zeros((64, 64), np.float32))
    assert flat.verdict == "undetermined"
    assert flat.is_double is False          # 但也绝不能说成 multi_tip


def test_undetermined_never_masquerades_as_clean():
    """弃权必须有明确 verdict 和 reason，不能与已确认无双针尖共用一个布尔值。"""
    flat = detect_double_tip(np.zeros((64, 64), np.float32))
    assert flat.verdict == "undetermined" and flat.reason
    stair = detect_double_tip(_staircase(seed=4), nm_per_px=0.6)
    assert stair.verdict == "undetermined" and stair.reason


# ── 3. 说「干净」要有可证伪的前提 ───────────────────────────────────────────
def test_featureless_frame_cannot_claim_single_tip():
    """缺少非周期特征时，未检测到鬼影不构成单针尖证据；应明确弃权。"""
    rng = np.random.RandomState(5)
    n = 192
    yy, xx = np.mgrid[0:n, 0:n]
    pure_lattice = (np.sin(xx * 2 * np.pi / 8) * np.sin(yy * 2 * np.pi / 8)
                    + 0.01 * rng.randn(n, n)).astype(np.float32)
    r = detect_double_tip(pure_lattice)
    assert r.verdict != "single_tip"
    assert r.verdict == "undetermined" and "aperiodic" in (r.reason or "")


def test_row_streaked_frame_abstains():
    """给孤立特征叠加逐行随机偏置，独立验证 row_offset 拒绝路径。
    偏置必须足以主导行差，同时保留可检测的非周期内容，避免另一道闸提前遮蔽本测试。"""
    rng = np.random.RandomState(6)
    n = 192
    img = _features(n=n, seed=6)
    img = img + (0.9 * rng.randn(n, 1)).astype(np.float32)   # 每行一个偏置
    r = detect_double_tip(img)
    assert r.verdict == "undetermined" and "row_offset" in (r.reason or "")


# ── 4. ⭐ 多针尖 vs「表面本来就密集台阶」 ───────────────────────────────────
def test_dense_staircase_is_never_called_multi_tip():
    """密集倾斜阶梯的重复峰不应被解释为多针尖。
    纯解析阶梯可能先因非周期信息不足而弃权；另用不规则合成地形核验形貌闸门。"""
    for seed in range(4):
        for terrace in (10.0, 14.0, 20.0):
            r = detect_double_tip(_staircase(terrace_px=terrace, seed=seed),
                                  nm_per_px=0.6)
            assert r.verdict != "multi_tip", (
                f"terrace={terrace}px seed={seed} 被误判成多针尖:{r.reason}")


def test_morphology_ratio_separates_the_two_regimes():
    """独立构造的孤立特征和阶梯应位于形貌比例门槛两侧，验证判据对两类输入的区分。"""
    from mast.vision.double_tip import _morphology_ratio, _to_2d

    def _ratio(a):
        f = _to_2d(a)
        f = f - float(f.mean())
        return _morphology_ratio((f / f.std()).astype(np.float32))

    for seed in range(3):
        assert _ratio(_features(seed=seed)) < 1.5          # 被验证的形貌
        assert _ratio(_staircase(seed=seed)) > 1.5         # 台阶主导


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_synthetic_morphology_dominated_frames_abstain(seed):
    """不规则特征保证非周期信息充足，台阶占优则必须由形貌闸门明确弃权。"""
    z = _staircase(terrace_px=17, angle_deg=27, seed=seed, noise_nm=0.004)
    z = z + _features(seed=seed)
    r = detect_double_tip(z, nm_per_px=0.6)
    assert r.score > 0 and r.candidates, r
    assert r.verdict == "undetermined", r
    assert "morphology" in r.reason, r


def test_staircase_with_a_real_ghost_still_does_not_claim_clean():
    """阶梯上注入真鬼影(正确的 soft-max 模型):判不出来可以,**不许说干净**。

    单帧在这个形貌里分不开,所以正确答案是弃权 —— 一个指向错误方向的「干净」
    比没有判据坏得多。
    """
    base = _staircase(seed=7)
    ghosted = _multitip(base, [(0, 0, 0.0), (5, 9, -0.08), (-7, 4, -0.14)])
    assert np.abs(ghosted - base).max() > 0.02      # 前提:注入确实落了地
    r = detect_double_tip(ghosted, nm_per_px=0.6)
    assert r.verdict != "single_tip", f"在真鬼影上说了「干净」:{r.reason}"


def test_roughness_is_not_multi_tip():
    """粗糙 = 不重复。相关粗糙度必须判不出多针尖(它没有固定位移)。"""
    n = 256
    fr = np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :])
    fr[0, 0] = 1.0
    amp = fr ** -0.75
    amp[0, 0] = 0.0
    for seed in range(4):
        ph = np.random.RandomState(seed).uniform(0, 2 * np.pi, (n, n))
        rough = np.real(np.fft.ifft2(amp * np.exp(1j * ph)))
        rough = (rough / rough.std()).astype(np.float32)   # 真有结构,不是噪声底
        r = detect_double_tip(rough, nm_per_px=0.5)
        assert r.verdict != "multi_tip", f"粗糙被判成多针尖:{r.reason}"


def test_a_frame_with_no_ghost_is_not_promoted_by_a_weak_peak():
    """显著性门限必须真的挡住东西。

    没有鬼影的特征帧,自相关里照样有 sig≈3.3 的峰(纯属特征分布的巧合)。
    把 ``significance_min`` 放掉,它们就会被判成多针尖 —— 变异验证证实过。
    """
    for seed in range(3):
        img = _features(seed=seed)
        assert detect_double_tip(img, nm_per_px=0.5).verdict == "single_tip"
        # 门限一放掉,同一张图就被「检出」多针尖 —— 门限是唯一挡住它的东西
        loose = detect_double_tip(img, nm_per_px=0.5, significance_min=0.0)
        assert loose.verdict == "multi_tip", (
            "这条测试假定弱峰会在无门限时冒头;若不再如此,门限的理由要重新写")


# ── 5. 位移向量:用户要的是「几个峰、各偏多少」 ──────────────────────────
def test_reports_the_displacement_vector_not_just_a_bool():
    img = _linear_ghost(_features(seed=9), 16, 12)
    r = detect_double_tip(img, nm_per_px=0.5)
    assert r.verdict == "multi_tip"
    assert r.candidates, "判成多针尖却没给位移向量"
    c = r.candidates[0]
    assert abs(abs(c.dy_px) - 16) <= 2 and abs(abs(c.dx_px) - 12) <= 2
    assert c.separation_nm == pytest.approx(np.hypot(16, 12) * 0.5, abs=1.5)


# ── 6. ⭐ 跨帧一致:针尖的性质 vs 表面的性质 ────────────────────────────────
def test_same_tip_across_frames_agrees_surface_does_not():
    """同一个针尖的微尖间距在每张帧里都是同一个向量;不同区域的台面重复不是。

    这条是**判据的核心**:单帧分不开的东西,跨帧能分开。
    """
    d = (16, 12)
    same_tip = []
    for seed in range(4):
        img = _linear_ghost(_features(seed=seed), *d)
        same_tip.append((detect_double_tip(img, nm_per_px=0.5), 0.5, 0.0))
    ok = agree_across_frames(same_tip)
    assert ok["agrees"] is True
    assert ok["separation_nm"] == pytest.approx(np.hypot(*d) * 0.5, abs=1.5)

    # 不同区域各自的表面结构 —— 不该凑出一个公共位移
    different = []
    for seed in range(4):
        img = _linear_ghost(_features(seed=seed), 10 + 3 * seed, 6 + 4 * seed)
        different.append((detect_double_tip(img, nm_per_px=0.5), 0.5, 0.0))
    assert agree_across_frames(different)["agrees"] is False


def test_rotated_frame_is_derotated_into_sample_coordinates():
    """90° 帧可以直接混进来:候选向量按扫描角反旋回**样品坐标**再比对。

    多针尖鬼影钉在样品坐标里(它随表面一起转);扫描伪影钉在扫描坐标里(不转)。
    """
    d = (16, 12)
    frames = []
    for seed in range(3):
        img = _linear_ghost(_features(seed=seed), *d)
        frames.append((detect_double_tip(img, nm_per_px=0.5), 0.5, 0.0))
    # 同一个针尖,扫描框转了 90° ⇒ 图像坐标里的向量跟着转
    rot = _linear_ghost(_features(seed=7), d[1], -d[0])
    frames.append((detect_double_tip(rot, nm_per_px=0.5), 0.5, 90.0))
    out = agree_across_frames(frames)
    assert out["agrees"] is True
    assert out["n_frames"] == 4, f"转过的那张没被归到一起:{out['reason']}"


def test_agreement_needs_a_majority_not_a_lucky_pair():
    """少数帧偶然共享位移不能证明跨帧一致。
    合成六帧中仅一对共享位移，其余独立，用于验证多数支持门槛。"""
    frames = []
    disp = [(16, 12), (16, 12),                       # ← 偶然撞上的那一对
            (10, 6), (22, 18), (9, 21), (25, 7)]      # 其余各不相同
    for seed, (dy, dx) in enumerate(disp):
        img = _linear_ghost(_features(seed=seed), dy, dx)
        frames.append((detect_double_tip(img, nm_per_px=0.5), 0.5, 0.0))
    out = agree_across_frames(frames)
    assert out["n_frames"] >= 2, "前提:那一对确实撞上了,否则这条测试没在测东西"
    assert out["agrees"] is False, f"两张撞上就下结论了:{out['reason']}"


# ── 7. ⭐ 用户的判据:次生台阶共享同一个偏置 ──────────────────────────────
def _split_staircase(n=256, terrace_px=26.0, d_px=6.0, seed=0,
                     a2_over_a1=0.02, noise_nm=0.004):
    """按**正确的** soft-max 前向模型劈开的阶梯 —— 每条台阶都被劈成同一组。

    这才是「多针尖」该有的样子:中间平台的宽度恰好是 |d|、高度只由 a2/a1 决定,
    因此整帧共享同一个偏置。用线性混合造出来的鬼影**不算数**(高度里没有线性回声)。
    """
    rng = np.random.RandomState(seed)
    x = np.arange(n)
    z1 = -AU_STEP_NM * np.floor(x / terrace_px)
    z2 = -AU_STEP_NM * np.floor((x - d_px) / terrace_px)
    e = np.logaddexp(TWO_KAPPA_PER_NM * z1,
                     TWO_KAPPA_PER_NM * z2 + np.log(a2_over_a1))
    row = e / TWO_KAPPA_PER_NM
    return np.tile(row, (n, 1)) + noise_nm * rng.randn(n, n)


def _plain_staircase(n=256, terrace_px=26.0, seed=0, noise_nm=0.004):
    rng = np.random.RandomState(seed)
    x = np.arange(n)
    row = -AU_STEP_NM * np.floor(x / terrace_px)
    return np.tile(row, (n, 1)) + noise_nm * rng.randn(n, n)


def test_split_steps_are_detected_and_the_offset_is_recovered():
    """⭐ 用户的判据:每条台阶被劈成同一组 ⇒ 偏置向量整帧一致。

    这条不依赖「线性回声」假设(恒流成像下那个假设本来就错),
    只要求「同一个针尖在整帧里是同一个针尖」。
    """
    from mast.vision.double_tip import detect_step_splitting

    nmpp = 0.4
    for d_px in (5.0, 8.0):
        r = detect_step_splitting(_split_staircase(d_px=d_px), nm_per_px=nmpp)
        assert r["verdict"] == "multi_tip", f"d={d_px}px 没检出:{r['reason']}"
        assert abs(r["offset_nm"] - d_px * nmpp) < 1.0, (
            f"偏置报错了:{r['offset_nm']:.2f} nm vs 真值 {d_px*nmpp:.2f} nm")


def test_plain_staircase_shows_no_splitting():
    """密集阶梯没有被劈 ⇒ 不许报多针尖。"""
    from mast.vision.double_tip import detect_step_splitting

    for seed in range(3):
        r = detect_step_splitting(_plain_staircase(seed=seed), nm_per_px=0.4)
        assert r["verdict"] != "multi_tip", f"干净阶梯被判成多针尖:{r['reason']}"


def test_an_offset_equal_to_the_terrace_repeat_is_not_a_split():
    """与台面重复尺度相同的候选间距不能被解释为更细小的共享台阶劈分。"""
    from mast.vision.double_tip import detect_step_splitting

    r = detect_step_splitting(_plain_staircase(terrace_px=26.0), nm_per_px=0.4)
    if "distinctness" in r:
        assert r["distinctness"] > 0.6, (
            "干净阶梯上不该出现明显小于台面重复的共享偏置")
    assert r["verdict"] != "multi_tip"


def test_step_splitting_abstains_when_pixels_cannot_resolve_a_split():
    """像素太粗 ⇒ 劈分本来就看不见 ⇒ 必须弃权,不能报「没有」。"""
    from mast.vision.double_tip import detect_step_splitting

    r = detect_step_splitting(np.zeros((64, 64)), nm_per_px=2.0)
    assert r["verdict"] == "undetermined"
    r2 = detect_step_splitting(_plain_staircase(n=64), nm_per_px=None)
    assert r2["verdict"] == "undetermined"


def test_random_substeps_near_edges_are_not_a_split():
    """局部子边可接近主台阶，但若偏置随机分散，不应报告整帧共享的多针尖偏置。"""
    from mast.vision.double_tip import detect_step_splitting

    n, terrace_px = 256, 26.0
    rng = np.random.RandomState(11)
    rows = []
    for _ in range(n):
        x = np.arange(n)
        z = -AU_STEP_NM * np.floor(x / terrace_px)
        for e in range(1, int(n / terrace_px)):
            # 每条台阶各自随机一个偏置(粗糙),而不是整帧共享一个
            off = int(rng.uniform(2, 10))
            p = int(e * terrace_px) + off
            if p < n:
                z[p:] -= 0.05
        rows.append(z)
    img = np.asarray(rows) + 0.004 * rng.randn(n, n)
    r = detect_step_splitting(img, nm_per_px=0.4)
    assert r["verdict"] != "multi_tip", (
        f"随机位置的子结构被判成多针尖:{r.get('reason')}")
    if "tightness" in r:
        assert r["tightness"] > 0.35, "前提:这些偏置确实是散的,否则没在测东西"
