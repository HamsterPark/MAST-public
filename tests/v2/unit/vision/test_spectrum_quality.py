"""谱数据质量判据（``mast.vision.spectroscopy`` 的谱质量那一段）。

**第一节就是纯白噪声的误报率** —— 项目铁律:此前已有多个「噪声/质量」判据在这
一条上被实测推翻。这里三条新判据(饱和 / 谱 SNR / 正反扫迟滞)各跑 ≥200 个种子
× 三个点数档,实测值写在断言旁边。

## 三把**从判据外面**拿来的尺子

判据可以稳定、可重复、看起来合理,同时量的根本不是你要的那个东西。所以本文件
的验证一律不依赖被测判据自己的输出:

1. **纯白噪声对照** —— 判据不该在没有信号的地方说有;
2. **两族曲线对照**(金属式 vs 带隙式)—— §1.3 那张表就是靠这把尺子发现
   ``assess_iv`` 被用错了地方;
3. **注入已知扰动** —— 在带隙曲线上叠一次已知幅度的 tip switch,看判据认不认得
   出来。这一把直接量出「``n_spikes`` 差值为 0」这个盲区。

## 这里的数字是**可执行的证据**,不是装饰

``test_two_family_table`` 逐个钉住 S4 STS 设计 §1.3 那张实测表。有人把
``n_spikes`` 重新接成带隙衬底的闸门时,它当场红 —— 这就是它存在的全部理由。
"""

from __future__ import annotations

import numpy as np
import pytest

from mast.vision import spectroscopy as S

N_SEEDS = 200
POINT_COUNTS = (40, 200, 400)


# ── 合成曲线:S4 STS 设计 §1.3 那张表用的就是这两条 ─────────────────────


def metallic_iv(n: int = 400):
    """金属式:``sinh(2V)``,−1..1 V。反对称、无隙、处处光滑。"""
    V = np.linspace(-1.0, 1.0, n)
    return V, np.sinh(2.0 * V)


def gapped_iv(n: int = 400):
    """带隙式:隙 0.6 V,−2..2 V,两支不等权(导带/价带态密度本来就不对称)。

    参数取自设计 §1.3 —— 这条曲线上 ``assess_iv`` 的四个数与设计文档逐位吻合
    (0.966 / 0.777 / 53 / 2.05),所以它就是那张表的可执行版本。
    """
    V = np.linspace(-2.0, 2.0, n)
    e = np.clip(np.abs(V) - 0.3, 0.0, None)
    I = np.sign(V) * (np.exp(3.0 * e) - 1.0)
    return V, np.where(V < 0, I * 0.4, I)


def with_tip_switch(I, *, gain: float, at: float = 0.5):
    """一次**真实形态**的 tip switch:从某一点起电流整体乘一个增益。

    注意它落在扫描中点(偏压过零处)—— 带隙曲线在那里电流本来就是 0,所以单方向
    的「突跳」统计量看不到任何东西。这不是构造出来为难判据的例子,这正是 tip
    switch 在带隙材料上最常见、也最难发现的样子。
    """
    out = np.asarray(I, dtype=float).copy()
    k = int(len(out) * at)
    out[k:] *= gain
    return out


def noisy(I, *, frac: float, seed: int):
    """满量程比例的高斯噪声。"""
    amp = frac * float(np.max(np.abs(I)))
    return I + amp * np.random.default_rng(seed).normal(size=len(I))


# ══════════════════════════════════════════════════════════════════════
# 1. 纯白噪声误报（合入前置条件，不是可选项）
# ══════════════════════════════════════════════════════════════════════


def test_white_noise_saturation_never_fires():
    """饱和判据在纯白噪声上 **0/600** 误报（实测）。

    这正是「重复的极值」而不是「接近最大值」的价值:白噪声里总有几个点接近极值,
    但它们不会连着好几点取同一个值。
    """
    fp = 0
    for npts in POINT_COUNTS:
        for seed in range(N_SEEDS):
            x = np.random.default_rng(90000 + 1000 * npts + seed).normal(size=npts)
            if S.saturation_frac(x):
                fp += 1
    assert fp == 0, f"饱和判据在纯白噪声上误报 {fp}/{3 * N_SEEDS}"


def test_white_noise_snr_stays_at_the_noise_floor():
    """SNR 在纯白噪声上是 3.0-5.3（实测；理论 span/σ ≈ 3.29）。

    这条不是「误报率为 0」而是**写出实测值**:任何标定出来的 ``min_snr`` 必须明显
    高于这个数,否则一堆噪声会被判成一条好谱。
    """
    worst = 0.0
    for npts in POINT_COUNTS:
        vals = [S.spectrum_snr(
            np.random.default_rng(90000 + 1000 * npts + s).normal(size=npts))
            for s in range(N_SEEDS)]
        assert all(v is not None for v in vals)
        worst = max(worst, max(vals))
        assert 2.0 < float(np.median(vals)) < 4.5
    assert worst <= 5.3, f"纯白噪声上的 SNR 上界实测 5.249，这次 {worst:.3f}"


def test_white_noise_hysteresis_reports_no_signal_and_almost_no_outliers():
    """两条**独立**白噪声:超阈点数 4/600 例非零、最大占比 0.025（实测）。

    同时钉住第二个数:``median_frac`` 的中位数是 0.297 —— 两条独立噪声的中位差
    就是噪声本身的量级。**只看超阈点数会把一堆噪声判成「正反扫一致」**,这就是
    为什么三个数必须全报。
    """
    fp = 0
    worst_frac = 0.0
    meds = []
    for npts in POINT_COUNTS:
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(90000 + 1000 * npts + seed)
            h = S.assess_hysteresis(rng.normal(size=npts), rng.normal(size=npts))
            assert h.available
            meds.append(h.median_frac)
            if h.outlier_points:
                fp += 1
            worst_frac = max(worst_frac, h.outlier_frac)
    assert fp <= 4, f"实测 4/{3 * N_SEEDS}，这次 {fp}"
    assert worst_frac <= 0.025, f"实测上界 0.025，这次 {worst_frac}"
    assert 0.25 < float(np.median(meds)) < 0.35, "纯噪声的 median_frac 实测 0.297"


def test_white_noise_never_produces_a_bad_verdict_end_to_end():
    """整条链路:纯白噪声进去,出来只能是 unrated —— 阈值全未标定时**没有**
    任何一条路径能产出 discard。"""
    for npts in POINT_COUNTS:
        for seed in range(0, N_SEEDS, 20):
            rng = np.random.default_rng(555000 + npts + seed)
            r = S.assess_spectrum_quality(
                kind="iv", bias_v=np.linspace(-1, 1, npts),
                current=rng.normal(size=npts),
                current_bwd=rng.normal(size=npts))
            assert r.verdict == "unrated"
            assert r.reasons == ("all_criteria_uncalibrated",)


# ══════════════════════════════════════════════════════════════════════
# 2. 两族对照表（S4 STS 设计 §1.3，逐数钉死）
# ══════════════════════════════════════════════════════════════════════


def test_two_family_table_metallic():
    """金属式干净曲线:``assess_iv`` 在它被设计针对的问题上工作得很好。"""
    V, I = metallic_iv()
    q = S.assess_iv(V, I)
    assert q.n_spikes == 0
    assert q.is_stable is True
    assert q.smoothness == pytest.approx(0.978, abs=0.001)
    assert q.symmetry == pytest.approx(1.000, abs=0.001)
    assert q.gap_ev == pytest.approx(0.105, abs=0.001)     # 无隙曲线上的**假隙**


def test_two_family_table_metallic_step_is_detected():
    """金属式 + 阶跃:分辨力很好 —— 一个 5% 满量程的阶跃就翻。

    (设计 §1.3 那一行记的 smoothness 是 0.566,对应约 1.4% 满量程的阶跃;
    「5%」的口径设计里没写死。结论方向一致:这一族**抓得到**。)
    """
    V, I = metallic_iv()
    q = S.assess_iv(V, with_step(I, frac=0.05))
    assert q.n_spikes == 1
    assert q.is_stable is False
    assert q.smoothness < 0.6


def with_step(I, *, frac: float, at: float = 0.5):
    out = np.asarray(I, dtype=float).copy()
    k = int(len(out) * at)
    out[k:] += frac * float(np.max(np.abs(out)))
    return out


def test_two_family_table_gapped_calls_a_perfect_spectrum_bad():
    """带隙式、**无噪声、教科书式**的好谱 ⇒ 53 个突跳、is_stable=False。

    根因:spike 判据拿全曲线的 MAD(d1) 当噪声尺度,而隙内那一段平坦点把 MAD 压
    到很低,于是整条指数上升段每一点都「超过 8 MAD」。**这不是坏谱,是错判据。**
    """
    V, I = gapped_iv()
    q = S.assess_iv(V, I)
    assert q.n_spikes == 53
    assert q.is_stable is False
    assert q.smoothness == pytest.approx(0.966, abs=0.001)
    assert q.symmetry == pytest.approx(0.777, abs=0.001)


def test_two_family_table_gapped_is_blind_to_a_real_tip_switch():
    """同一条带隙曲线叠一次**真** 15% tip switch ⇒ ``n_spikes`` 差值为 **0**。

    这是全套里最重要的一条:不是保守,是**没有分辨力**。一个把 ``n_spikes`` 当
    带隙衬底闸门的流程,会同时把好谱全判掉、又把真正的针尖变化全放过去。
    """
    V, I = gapped_iv()
    before = S.assess_iv(V, I).n_spikes
    after = S.assess_iv(V, with_tip_switch(I, gain=1.15)).n_spikes
    assert before == 53 and after == 53
    assert after - before == 0


def test_gap_ev_is_wrong_in_both_directions():
    """``gap_ev`` 无隙曲线报出 0.105 V 的隙、0.6 V 真隙报 2.05 V(大 3.4 倍)。

    它是一个粗糙代理。这条测试是「别把它当带隙值上报」的唯一护栏。
    """
    V, I = metallic_iv()
    assert S.assess_iv(V, I).gap_ev == pytest.approx(0.105, abs=0.001)
    V, I = gapped_iv()
    assert S.assess_iv(V, I).gap_ev == pytest.approx(2.05, abs=0.01)


def test_gap_ev_never_gates_and_always_carries_the_caveat():
    """``gap_ev`` 不在任何族的 ``gated_criteria`` 里,而且每次上报都带告诫。"""
    V, I = gapped_iv()
    for family in S.SPECTRAL_FAMILIES:
        r = S.assess_spectrum_quality(
            kind="iv", bias_v=V, current=I, current_bwd=I,
            spectral_family=family, max_saturation_frac=0.5)
        assert r.iv_gap_ev is not None
        assert "gap_ev" not in r.gated_criteria
        assert "gap_ev_is_not_a_gap_measurement" in r.warnings


def test_smoothness_drifts_with_point_count():
    """同一条解析曲线,点数越多 smoothness 越高 —— 单调,且跨度极大。

    任何写死的 ``min_smoothness`` 只对某一个 ``num_points`` 成立,而点数是流程的
    自由参数。归一化形式待真机数据定,在此之前这个阈值必须保持未标定。
    """
    vals = []
    for n in (40, 100, 200, 400, 1000):
        V, I = metallic_iv(n)
        vals.append(S.assess_iv(V, I).smoothness)
    assert vals == sorted(vals)
    assert vals[0] == pytest.approx(0.788, abs=0.002)
    assert vals[-1] == pytest.approx(0.991, abs=0.002)


def test_gating_metallic_on_is_stable_would_fail_a_clean_spectrum():
    """为什么金属族当闸的是 ``n_spikes`` 而不是 ``is_stable``。

    一条**干净的** 400 点金属曲线加上 0.1% 满量程噪声:``n_spikes`` 还是 0
    (稳),但 ``smoothness`` 掉到 0.13 ⇒ ``is_stable`` 是 False。拿 is_stable 当闸
    等于把那个未归一化的点数依赖阈值从后门放进来。
    """
    V, I = metallic_iv()
    q = S.assess_iv(V, noisy(I, frac=0.001, seed=7))
    assert q.n_spikes == 0
    assert q.smoothness < 0.5
    assert q.is_stable is False
    r = S.assess_spectrum_quality(kind="iv", bias_v=V,
                                  current=noisy(I, frac=0.001, seed=7),
                                  spectral_family="metallic")
    assert "iv_stability" in r.gated_criteria
    assert r.verdict != "discard"


# ══════════════════════════════════════════════════════════════════════
# 3. tip switch 分辨力：旧判据的盲区与新判据的能力，同一条测
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("gain,expect_outliers", [(1.05, 40), (1.15, 60), (1.3, 80)])
def test_hysteresis_sees_the_switch_that_spikes_cannot(gain, expect_outliers):
    """带隙曲线 + 0.1% 满量程噪声,注入已知 tip switch:

    * ``assess_iv.n_spikes`` 在有 switch 与没 switch 之间**差值为 0**;
    * 正反扫超阈点数从 0 跳到几十 —— 这是唯一抓得到它的一条。
    """
    V, I = gapped_iv()
    clean_f = noisy(I, frac=0.001, seed=7)
    clean_b = noisy(I, frac=0.001, seed=11)
    switched = noisy(with_tip_switch(I, gain=gain), frac=0.001, seed=7)

    assert S.assess_iv(V, switched).n_spikes == S.assess_iv(V, clean_f).n_spikes

    assert S.assess_hysteresis(clean_f, clean_b).outlier_points == 0
    assert S.assess_hysteresis(switched, clean_b).outlier_points >= expect_outliers


def test_hysteresis_needs_a_backward_column_and_says_so():
    """缺反扫列 ⇒ 三个数**全 None** + ``no_backward_column``,**不是** passed。"""
    V, I = gapped_iv()
    r = S.assess_spectrum_quality(kind="iv", bias_v=V, current=I,
                                  spectral_family="gapped",
                                  max_saturation_frac=0.5)
    assert r.hysteresis_median_frac is None
    assert r.hysteresis_max_frac is None
    assert r.hysteresis_outlier_points is None
    assert "no_backward_column" in r.reasons
    assert "hysteresis" in r.ungated_criteria
    # require_backward=False 时它仍然可以是 keep —— 缺一列不等于数据不好。
    assert r.verdict in ("keep", "keep_flagged")
    strict = S.assess_spectrum_quality(kind="iv", bias_v=V, current=I,
                                       spectral_family="gapped",
                                       max_saturation_frac=0.5,
                                       require_backward=True)
    assert strict.verdict == "unrated"
    assert "no_backward_column" in strict.reasons


def test_saturation_catches_a_railed_preamp_but_not_an_exponential_tail():
    """削顶 ⇒ 抓到;干净的指数曲线尾巴 ⇒ 0。"""
    V, I = metallic_iv()
    assert S.saturation_frac(I) == 0.0
    _, Ig = gapped_iv()
    assert S.saturation_frac(Ig) == 0.0
    rail = 0.5 * float(np.max(np.abs(I)))
    assert S.saturation_frac(np.clip(I, -rail, rail)) == pytest.approx(0.325,
                                                                      abs=0.01)


def test_snr_tracks_the_injected_noise_level():
    """外部尺子:注入已知噪声,SNR 必须跟着走(0.1% → ~3300,5% → ~33)。"""
    V, I = metallic_iv()
    got = [S.spectrum_snr(noisy(I, frac=f, seed=1)) for f in (0.0005, 0.005, 0.05)]
    assert got[0] > got[1] > got[2]
    assert got[0] == pytest.approx(3300, rel=0.25)
    assert got[2] == pytest.approx(33, rel=0.25)


# ══════════════════════════════════════════════════════════════════════
# 4. 族分闸 + 未标定绝不产 bad（D3/D4 的结构强制）
# ══════════════════════════════════════════════════════════════════════


def test_unknown_family_gates_only_the_family_independent_three():
    """``unknown`` **不是** metallic 的同义词。"""
    V, I = gapped_iv()
    r = S.assess_spectrum_quality(
        kind="iv", bias_v=V, current=I, current_bwd=I,
        spectral_family="unknown",
        min_snr=10.0, max_saturation_frac=0.1,
        max_hysteresis_outlier_frac=0.05)
    assert set(r.gated_criteria) == {"saturation", "snr", "hysteresis"}
    assert "iv_stability" not in r.gated_criteria
    assert "symmetry" not in r.gated_criteria


def test_empty_or_bogus_family_falls_back_to_unknown_not_metallic():
    V, I = gapped_iv()
    for family in ("", "  ", "Au-ish", None):
        r = S.assess_spectrum_quality(kind="iv", bias_v=V, current=I,
                                      current_bwd=I, spectral_family=family,
                                      max_saturation_frac=0.5)
        assert r.spectral_family == "unknown"
        assert "iv_stability" not in r.gated_criteria


def test_gapped_family_never_gates_on_spikes_or_symmetry():
    """§5.6 钉子①:被否掉的方案 —— 带隙衬底上接 ``is_stable`` 当闸。"""
    V, I = gapped_iv()
    r = S.assess_spectrum_quality(
        kind="iv", bias_v=V, current=I, current_bwd=I, spectral_family="gapped",
        min_snr=1.0, max_saturation_frac=0.5, max_hysteresis_outlier_frac=0.5,
        min_symmetry=0.95)
    assert "iv_stability" in r.ungated_criteria
    assert "symmetry" in r.ungated_criteria
    assert r.iv_n_spikes == 53          # 数照报
    assert r.iv_is_stable is False      # 数照报
    assert r.verdict == "keep"          # 但它一个 bad 都产不出来
    assert "unstable_iv" not in r.reasons


def test_mutation_gapped_gated_like_metallic_would_discard_a_good_spectrum(
        monkeypatch):
    """变异验证 —— 先证变异**真的落到了地上**,再证结论翻转。

    把带隙族的可闸集合改成金属族的那一套(这正是被否掉的方案),同一条好谱当场
    从 ``keep`` 变成 ``discard``。这一条红了,说明 D3 的分族不是装饰。
    """
    V, I = gapped_iv()
    kw = dict(kind="iv", bias_v=V, current=I, current_bwd=I,
              spectral_family="gapped", max_saturation_frac=0.5,
              max_hysteresis_outlier_frac=0.5)
    assert S.assess_spectrum_quality(**kw).verdict == "keep"

    mutated = dict(S._FAMILY_GATEABLE)
    mutated["gapped"] = S._FAMILY_GATEABLE["metallic"]
    monkeypatch.setattr(S, "_FAMILY_GATEABLE", mutated)
    # 变异已应用
    assert "iv_stability" in S._FAMILY_GATEABLE["gapped"]

    r = S.assess_spectrum_quality(**kw)
    assert r.verdict == "discard"
    assert "unstable_iv" in r.reasons


def test_all_thresholds_none_is_unrated_not_keep():
    """所有可当闸项都未标定 ⇒ ``unrated`` + ``all_criteria_uncalibrated``。

    ``unrated`` 在闸门那头既不计产量也不计不合格 —— 这与 ``keep`` 是两句完全不同
    的话,而「阈值还没标定」时能说的只有前者。
    """
    V, I = gapped_iv()
    r = S.assess_spectrum_quality(kind="iv", bias_v=V, current=I, current_bwd=I,
                                  spectral_family="gapped")
    assert r.verdict == "unrated"
    assert r.reasons == ("all_criteria_uncalibrated",)
    assert r.gated_criteria == ()
    # 数一个都没少报 —— 「判不了」不等于「没测」。
    assert r.spectrum_snr is not None
    assert r.saturation_frac is not None
    assert r.hysteresis_outlier_points is not None


def test_uncalibrated_threshold_can_never_produce_a_bad_verdict():
    """结构强制:阈值 None 时,即使数据很差也只能是 unrated。"""
    V, I = metallic_iv()
    rail = 0.2 * float(np.max(np.abs(I)))
    bad = np.clip(noisy(I, frac=0.3, seed=3), -rail, rail)   # 又饱和又全是噪声
    r = S.assess_spectrum_quality(kind="iv", bias_v=V, current=bad,
                                  current_bwd=noisy(bad, frac=0.3, seed=4),
                                  spectral_family="unknown")
    assert r.saturation_frac > 0.3
    assert r.verdict == "unrated"
    assert "saturated" not in r.reasons
    assert "low_snr" not in r.reasons


def test_mutation_treating_none_as_zero_turns_unrated_into_discard():
    """变异验证 —— 「``None`` 当 0 处理」是一个**会产出假不合格**的改动。

    用一条**边缘**的谱(1% 的增益漂移,6/400 点超阈)——正是没有标定就说不出话的
    那种。阈值 None ⇒ ``unrated``;把 None 当成 0 ⇒ ``discard``。两者的差别就是
    「判不了」与「不合格」的差别,而这正是 D4 要用结构挡住的那一步。
    """
    V, I = gapped_iv()
    f = noisy(with_tip_switch(I, gain=1.01), frac=0.001, seed=7)
    b = noisy(I, frac=0.001, seed=11)
    kw = dict(kind="iv", bias_v=V, current=f, current_bwd=b,
              spectral_family="gapped")

    base = S.assess_spectrum_quality(**kw)
    assert base.verdict == "unrated"
    assert base.hysteresis_outlier_points == 6      # 数照报,只是没人拿它当闸

    # 变异:未标定 ⇒ 0。先证变异真的落到了地上(阈值从 None 变成了 0.0)。
    zeroed = dict(kw, min_snr=0.0, max_saturation_frac=0.0,
                  max_hysteresis_outlier_frac=0.0)
    assert kw.get("max_hysteresis_outlier_frac") is None
    assert zeroed["max_hysteresis_outlier_frac"] == 0.0
    r = S.assess_spectrum_quality(**zeroed)
    assert r.gated_criteria != ()
    assert r.verdict == "discard"
    assert "hysteresis_exceeded" in r.reasons


def test_iz_gates_on_physics_not_on_a_sample_calibration():
    """I(z) 全套能当闸:隧穿衰减是真空势垒的事,不随材料带隙变。"""
    z = np.linspace(0.0, 0.5, 60)
    I = np.exp(-21.7 * z)
    r = S.assess_spectrum_quality(kind="iz", z_nm=z, current=I,
                                  spectral_family="unknown")
    assert "iz_exponential" in r.gated_criteria
    assert r.iz_barrier_ev == pytest.approx(4.5, abs=0.6)
    assert r.verdict in ("keep", "keep_flagged")

    broken = I.copy()
    broken[30:] *= 0.2
    bad = S.assess_spectrum_quality(kind="iz", z_nm=z, current=broken,
                                    spectral_family="unknown")
    assert bad.verdict == "discard"
    assert {"poor_iz_fit", "barrier_out_of_range"} & set(bad.reasons)


# ══════════════════════════════════════════════════════════════════════
# 5. 出口闭集 / kind 内容判据 / dI/dV 极性
# ══════════════════════════════════════════════════════════════════════


def test_verdict_reasons_and_warnings_stay_inside_the_closed_vocabularies():
    """闭集就是闭集 —— 下游按词表分支,多一个词等于多一条没人写的分支。"""
    V, I = gapped_iv()
    cases = [
        dict(kind="iv", bias_v=V, current=I),
        dict(kind="iv", bias_v=V, current=I, current_bwd=I,
             min_snr=1e9, max_saturation_frac=0.0),
        dict(kind="iv", bias_v=V, current=I[:3]),
        dict(kind="zzz", current=I),
        dict(kind="iz", current=I),
        dict(kind="iz", z_nm=np.linspace(0, 0.5, 60), current=np.exp(
            -21.7 * np.linspace(0, 0.5, 60))),
    ]
    for kw in cases:
        for family in S.SPECTRAL_FAMILIES:
            r = S.assess_spectrum_quality(spectral_family=family, **kw)
            assert r.verdict in S.SPECTRUM_VERDICTS
            assert set(r.reasons) <= set(S.SPECTRUM_REASONS), r.reasons
            assert set(r.warnings) <= set(S.SPECTRUM_WARNINGS), r.warnings
            assert set(r.gated_criteria) <= set(S.SPECTRUM_CRITERIA)
            assert set(r.ungated_criteria) <= set(S.SPECTRUM_CRITERIA)
            assert not (set(r.gated_criteria) & set(r.ungated_criteria))


def test_structural_failures_are_unrated_never_discard():
    V, I = gapped_iv()
    assert S.assess_spectrum_quality(kind="", current=I).verdict == "unrated"
    assert S.assess_spectrum_quality(kind="", current=I).reasons == (
        "kind_undetermined",)
    short = S.assess_spectrum_quality(kind="iv", bias_v=V[:4], current=I[:4])
    assert short.verdict == "unrated" and short.reasons == ("insufficient_points",)
    nobias = S.assess_spectrum_quality(kind="iv", current=I)
    assert nobias.verdict == "unrated" and nobias.reasons == ("no_bias_column",)
    noz = S.assess_spectrum_quality(kind="iz", current=I)
    assert noz.verdict == "unrated" and noz.reasons == ("no_z_column",)


def test_a_truncated_last_row_does_not_blow_up_the_judgement():
    """真机场景:``read_dat`` 把**截断的末行**用 NaN 补齐。

    ``assess_iv`` 不过滤 NaN,一个 NaN 会让 smoothness 变成 NaN,而结果模型的
    ``0 ≤ x ≤ 1`` 约束当场抛异常。纯函数承诺不抛,所以这里先把非有限行丢掉 ——
    而且**丢了多少要说出来**,不能只剩一个悄悄变小的 n_points。
    """
    V, I = metallic_iv(100)
    holed = I.copy()
    holed[-1] = np.nan
    with pytest.raises(Exception):
        S.assess_iv(V, holed)           # 上游确实会炸(这里钉住「为什么要过滤」)

    r = S.assess_spectrum_quality(kind="iv", bias_v=V, current=holed,
                                  current_bwd=holed, spectral_family="metallic")
    assert r.n_points == 99
    assert r.n_points_dropped == 1
    assert r.verdict in S.SPECTRUM_VERDICTS

    allnan = S.assess_spectrum_quality(kind="iv", bias_v=V,
                                       current=np.full(100, np.nan))
    assert allnan.verdict == "unrated"
    assert allnan.reasons == ("insufficient_points",)
    assert allnan.n_points_dropped == 100


def test_mismatched_axis_length_is_unrated_not_a_crash():
    V, I = metallic_iv(100)
    r = S.assess_spectrum_quality(kind="iv", bias_v=V[:50], current=I)
    assert r.verdict == "unrated" and r.reasons == ("insufficient_points",)


def test_resolve_kind_reads_the_data_not_the_label():
    V, _ = metallic_iv()
    kind, ev = S.resolve_spectrum_kind(bias_v=V)
    assert kind == "iv" and "偏压" in ev

    z = np.linspace(0.0, 5e-10, 60)
    kind, ev = S.resolve_spectrum_kind(bias_v=np.full(60, 0.5), z_m=z)
    assert kind == "iz" and "Z" in ev

    # 两条都在扫 / 都不在扫 ⇒ 判不了,而不是挑一个。
    assert S.resolve_spectrum_kind(bias_v=V, z_m=np.linspace(0, 5e-10, 400))[0] == ""
    assert S.resolve_spectrum_kind(bias_v=np.full(60, 0.5),
                                   z_m=np.full(60, 1e-9))[0] == ""
    assert S.resolve_spectrum_kind()[0] == ""


def test_inverted_didv_is_flagged_and_never_flipped():
    """§5.6 钉子③:倒置的 dI/dV **报警,绝不自动翻转**。

    自动取反会把「lock-in 相位设错了」悄悄变成一条看起来正常的谱 —— 接线错误从
    此永久消失,而后面所有基于它的结论都是错的。
    """
    V, I = metallic_iv()
    didv = np.gradient(I, V)
    r = S.assess_spectrum_quality(kind="iv", bias_v=V, current=I,
                                  didv=-didv, spectral_family="metallic")
    assert "didv_polarity_suspect" in r.warnings
    ok = S.assess_spectrum_quality(kind="iv", bias_v=V, current=I,
                                   didv=didv, spectral_family="metallic")
    assert "didv_polarity_suspect" not in ok.warnings
    # 判据不持有 dI/dV 的任何取反版本:结果里根本没有它的数值字段。
    assert not any("didv" in f for f in vars(r))


def test_sts_is_not_a_damage_kind():
    """§5.6 钉子②:``sts`` **不在** ``DAMAGE_KINDS`` 里,别加。

    加了之后一条几十点的线谱会各自生成避让圈,一次性废掉整个中心区并逼出一次
    不必要的粗动。谱学不改表面 —— 「测过」不等于「弄坏了」。
    """
    from mast.io.map_analysis import DAMAGE_KINDS
    assert "sts" not in DAMAGE_KINDS
    assert set(DAMAGE_KINDS) == {"tip_shape", "pulse", "crash", "approach"}
